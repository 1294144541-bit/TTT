# Pretrain & Evaluate Official Meta V-JEPA 2.1 World Model with TTT Adaptation
# Pipeline:
# 1. Extract 768-dim spatiotemporal embeddings using official V-JEPA 2.1 ViT-B/16 checkpoint (419MB)
# 2. Train Invariant Dynamics Core f_\theta on nominal video features
# 3. Compute Null-space Projection P_\perp
# 4. Run Test-Time Training (TTT) online streaming adaptation on OOD non-stationary physical test videos
# 5. Measure Latent Prediction MSE and Linear Probe R^2 for Physical Scale (Mass & Friction)

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_official_loader import VJEPAWorldModelTTT

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
DATA_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
CKPT_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints"

def extract_vjepa_features(model: VJEPAWorldModelTTT, video_tensor: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    # video_tensor: (N, T, 3, H, W)
    N, T, C, H, W = video_tensor.shape
    flat_frames = video_tensor.reshape(-1, C, H, W)
    features = []
    with torch.no_grad():
        for i in range(0, flat_frames.shape[0], batch_size):
            batch = flat_frames[i:i+batch_size]
            z = model.encoder(batch)
            features.append(z.cpu())
    all_z = torch.cat(features, dim=0)
    return all_z.reshape(N, T, -1)

def run_vjepa_ttt_pipeline():
    print("================================================================")
    print("Initializing Meta V-JEPA 2.1 + In-Weights TTT Full Pipeline")
    print("================================================================")

    # 1. Instantiate Model
    model = VJEPAWorldModelTTT(
        vjepa_weights_path=WEIGHTS_PATH,
        context_k_dim=64,
        lr_ttt=0.08,
        beta_proximal=0.01,
        core_rank=16
    )

    # 2. Extract Official V-JEPA Representations for Nominal Train Data
    train_data = torch.load(os.path.join(DATA_DIR, "vision_train_nominal.pt"))
    print("Extracting 768-dim V-JEPA representations for nominal training videos...")
    z_train = extract_vjepa_features(model, train_data["videos"]) # (N=40, T=25, 768)
    N, T, d_z = z_train.shape

    # 3. Train Invariant Dynamics Core f_\theta
    print("Training Invariant Dynamics Core on 768-dim V-JEPA latent manifolds...")
    optimizer = torch.optim.Adam(model.invariant_core.parameters(), lr=1e-3)
    z_curr_flat = z_train[:, :-1].reshape(-1, d_z)
    z_next_flat = z_train[:, 1:].reshape(-1, d_z)
    num_samples = z_curr_flat.shape[0]

    model.invariant_core.train()
    for epoch in range(15):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(0, num_samples, 64):
            idx = perm[i:i+64]
            pred = model.invariant_core(z_curr_flat[idx])
            loss = F.mse_loss(pred, z_next_flat[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/15 | V-JEPA Dynamics MSE: {epoch_loss / num_samples:.6f}")

    # 4. Compute Null-space Projection Operator
    model.eval()
    with torch.no_grad():
        ctx_train = model.context_mapping(z_curr_flat[:200])
        model.compute_nullspace_projection(ctx_train)
        model.W_0.zero_()
        model.reset_plastic_weights()

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "vjepa_nominal_core.pt"))
    print("Saved trained Invariant Dynamics Core & Projection matrix.")

    # 5. Evaluate on Physical Benchmarks (Nominal vs OOD Shock vs OOD Low Gravity)
    benchmarks = [
        ("vision_test_nominal.pt", "In-Distribution Nominal"),
        ("vision_test_heavy_mu.pt", "OOD Heavy Mass & Friction Shock"),
        ("vision_test_low_g.pt", "OOD Low Gravity & High Bounce")
    ]

    results_summary = []

    for file_name, bench_name in benchmarks:
        print(f"\n--- Evaluating Benchmark: {bench_name} ---")
        test_data = torch.load(os.path.join(DATA_DIR, file_name))
        z_test = extract_vjepa_features(model, test_data["videos"])
        params_test = test_data["params"].numpy()
        N_test, T_test, _ = z_test.shape

        static_errs = []
        ttt_errs = []
        all_W = []
        all_p = []

        for ep in range(N_test):
            model.reset_plastic_weights()
            for t in range(T_test - 1):
                z_t = z_test[ep, t:t+1]
                z_next_true = z_test[ep, t+1:t+2]
                p_t = params_test[ep, t]

                with torch.no_grad():
                    # Static prediction
                    pred_static = model.forward_latent_prediction(z_t, W=model.W_0)
                    static_errs.append(F.mse_loss(pred_static, z_next_true).item())

                    # TTT prediction
                    pred_ttt = model.forward_latent_prediction(z_t, W=model.W_t)
                    ttt_errs.append(F.mse_loss(pred_ttt, z_next_true).item())

                # Online TTT adaptation step
                model.ttt_step(z_t, z_next_true)
                all_W.append(model.W_t.detach().cpu().numpy().flatten())
                all_p.append(p_t)

        mean_static = np.mean(static_errs)
        mean_ttt = np.mean(ttt_errs)
        error_drop = (mean_static - mean_ttt) / mean_static * 100.0

        # Linear Probe for Physical Decodability (Mass & Friction)
        W_mat = np.array(all_W) # (N*(T-1), 64*768 = 49152)
        P_mat = np.array(all_p)

        # SVD projection to top-25 components of weight manifold
        W_centered = W_mat - np.mean(W_mat, axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(W_centered, full_matrices=False)
        k_comp = min(25, S.shape[0])
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

        print(f"Static V-JEPA World Model MSE : {mean_static:.6f}")
        print(f"TTT-Augmented V-JEPA MSE      : {mean_ttt:.6f}")
        print(f"Relative Latent Error Drop     : {error_drop:.2f}%")
        print(f"Linear Probe R^2 (Mass m)      : {r2[0]:.4f}")
        print(f"Linear Probe R^2 (Friction mu) : {r2[1]:.4f}")

        results_summary.append({
            "name": bench_name,
            "static_mse": mean_static,
            "ttt_mse": mean_ttt,
            "drop": error_drop,
            "r2_m": r2[0],
            "r2_mu": r2[1]
        })

    return results_summary

if __name__ == "__main__":
    run_vjepa_ttt_pipeline()
