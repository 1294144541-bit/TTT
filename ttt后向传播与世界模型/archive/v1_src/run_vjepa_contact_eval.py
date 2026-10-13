# Pretrain & Evaluate V-JEPA Physics World Model with Velocity Projector & TTT
# Evaluates on 224x224 Rigid-Body Collision & Impulse Datasets

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
DATA_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
CKPT_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints"

def extract_features(model, videos, batch_size=16):
    # videos: (N, T, 3, 224, 224)
    N, T, C, H, W = videos.shape
    flat = videos.reshape(-1, C, H, W)
    feats = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], batch_size):
            b = flat[i:i+batch_size]
            z = model.encoder(b)
            feats.append(z.cpu())
    return torch.cat(feats, dim=0).reshape(N, T, -1)

def main():
    print("======================================================================")
    print("Executing V-JEPA 2.1 Physical Scale Evaluation with Dynamics Projector")
    print("======================================================================")

    model = VJEPAPhysicsTTTWorldModel(
        vjepa_weights_path=WEIGHTS_PATH,
        dyn_dim=128,
        context_k_dim=64,
        lr_ttt=0.12,
        beta_proximal=0.01,
        core_rank=16
    )

    # 1. Feature extraction on nominal collision videos
    train_data = torch.load(os.path.join(DATA_DIR, "contact_train_nominal.pt"))
    print("Extracting V-JEPA ViT-B/16 representations for 224x224 collision videos...")
    z_train = extract_features(model, train_data["videos"]) # (N=30, T=20, 768)
    N, T, _ = z_train.shape

    # Construct velocity-aware physical states: s_t = Projector(z_t, z_{t-1})
    print("Constructing velocity-aware momentum states and pretraining core...")
    optimizer = torch.optim.Adam(
        list(model.dyn_neck.parameters()) + list(model.invariant_core.parameters()),
        lr=2e-3
    )

    # Pairs for pretraining: (s_t -> s_{t+1}) for t in 1 .. T-2
    # s_t requires (z_t, z_{t-1}), s_{t+1} requires (z_{t+1}, z_t)
    z_prev = z_train[:, :-2].reshape(-1, 768)
    z_curr = z_train[:, 1:-1].reshape(-1, 768)
    z_next = z_train[:, 2:].reshape(-1, 768)
    total_pairs = z_curr.shape[0]

    model.dyn_neck.train()
    model.invariant_core.train()
    for epoch in range(16):
        perm = torch.randperm(total_pairs)
        epoch_loss = 0.0
        for i in range(0, total_pairs, 64):
            idx = perm[i:i+64]
            s_t = model.dyn_neck(z_curr[idx], z_prev[idx])
            s_next_target = model.dyn_neck(z_next[idx], z_curr[idx]).detach()
            s_pred = model.invariant_core(s_t)
            loss = F.mse_loss(s_pred, s_next_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 4 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/16 | Physical Dynamics MSE: {epoch_loss / total_pairs:.6f}")

    # 2. Compute Null-space projection
    model.eval()
    with torch.no_grad():
        s_sample = model.dyn_neck(z_curr[:200], z_prev[:200])
        ctx_sample = model.context_mapping(s_sample)
        model.compute_nullspace_projection(ctx_sample)
        model.W_0.zero_()
        model.reset_plastic_weights()

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "vjepa_physics_neck.pt"))
    print("Pretrained dynamics neck and invariant core saved successfully.")

    # 3. Benchmark evaluation
    benchmarks = [
        ("contact_test_nominal.pt", "In-Distribution Collision"),
        ("contact_test_heavy_mass.pt", "OOD Heavy Mass Collision (Impulse Shift)"),
        ("contact_test_high_friction.pt", "OOD Extreme Friction Braking")
    ]

    for fname, bname in benchmarks:
        print(f"\n--- Benchmark: {bname} ---")
        tdata = torch.load(os.path.join(DATA_DIR, fname))
        z_test = extract_features(model, tdata["videos"])
        params_test = tdata["params"].numpy()
        N_t, T_t, _ = z_test.shape

        static_errs, ttt_errs = [], []
        all_W, all_p = [], []

        for ep in range(N_t):
            model.reset_plastic_weights()
            for t in range(1, T_t - 1):
                with torch.no_grad():
                    s_t = model.dyn_neck(z_test[ep, t:t+1], z_test[ep, t-1:t])
                    s_next_true = model.dyn_neck(z_test[ep, t+1:t+2], z_test[ep, t:t+1])

                    # Static prediction
                    pred_static = model.forward_state_prediction(s_t, W=model.W_0)
                    static_errs.append(F.mse_loss(pred_static, s_next_true).item())

                    # TTT prediction
                    pred_ttt = model.forward_state_prediction(s_t, W=model.W_t)
                    ttt_errs.append(F.mse_loss(pred_ttt, s_next_true).item())

                # Online TTT adaptation step
                model.ttt_step(s_t, s_next_true)
                all_W.append(model.W_t.detach().cpu().numpy().flatten())
                all_p.append(params_test[ep, t])

        m_static = np.mean(static_errs)
        m_ttt = np.mean(ttt_errs)
        drop = (m_static - m_ttt) / m_static * 100.0

        # Linear Probe for Physical Decodability (Mass Ratio m1/m2 and Friction mu)
        W_mat = np.array(all_W)
        P_mat = np.array(all_p)

        W_norm = (W_mat - np.mean(W_mat, axis=0, keepdims=True)) / (np.std(W_mat, axis=0, keepdims=True) + 1e-6)
        U, S, _ = np.linalg.svd(W_norm, full_matrices=False)
        k_comp = min(30, S.shape[0])
        W_proj = U[:, :k_comp] @ np.diag(S[:k_comp])
        W_proj_b = np.concatenate([W_proj, np.ones((W_proj.shape[0], 1))], axis=1)

        lambda_reg = 0.5
        A = W_proj_b.T @ W_proj_b + lambda_reg * np.eye(W_proj_b.shape[1])
        b = W_proj_b.T @ P_mat[:, :2]
        coeffs = np.linalg.solve(A, b)
        pred_p = W_proj_b @ coeffs

        ss_tot = np.sum((P_mat[:, :2] - np.mean(P_mat[:, :2], axis=0)) ** 2, axis=0)
        ss_res = np.sum((P_mat[:, :2] - pred_p) ** 2, axis=0)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        print(f"Static Physical State MSE : {m_static:.6f}")
        print(f"TTT Physical State MSE    : {m_ttt:.6f}")
        print(f"Relative Prediction Gain  : {drop:.2f}%")
        print(f"Linear Probe R^2 (Mass Ratio) : {r2[0]:.4f}")
        print(f"Linear Probe R^2 (Friction)   : {r2[1]:.4f}")

if __name__ == "__main__":
    main()
