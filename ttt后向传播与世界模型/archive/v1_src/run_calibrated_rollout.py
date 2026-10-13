# Dual-Loop Long-Horizon Rollout with Soft-Constraint Projected Gradient
# Implements:
# 1. Soft Null-space constraint: G_adapted = P_perp G_raw + lambda_leak (I - P_perp) G_raw
# 2. Multi-step open-loop rollout over H=10 frames
# 3. Measures cumulative compounding error and trajectory divergence between Static and TTT-Adapted models

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
DATA_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
CKPT_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa_physics_neck.pt"

def extract_features(model, videos, batch_size=16):
    N, T, C, H, W = videos.shape
    flat = videos.reshape(-1, C, H, W)
    feats = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], batch_size):
            b = flat[i:i+batch_size]
            z = model.encoder(b)
            feats.append(z.cpu())
    return torch.cat(feats, dim=0).reshape(N, T, -1)

def run_calibrated_long_horizon_rollout(
    test_file: str,
    benchmark_name: str,
    t_adapt: int = 7,
    rollout_horizon: int = 10,
    lr_ttt: float = 0.35,
    leak_ratio: float = 0.25 # Allows controlled adaptation on physics-drift subspace
):
    model = VJEPAPhysicsTTTWorldModel(
        vjepa_weights_path=WEIGHTS_PATH,
        dyn_dim=128,
        context_k_dim=64,
        lr_ttt=lr_ttt,
        beta_proximal=0.002,
        core_rank=16
    )
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    data = torch.load(os.path.join(DATA_DIR, test_file))
    videos = data["videos"]
    N, T, _, _, _ = videos.shape

    z_test = extract_features(model, videos)
    print(f"\n======================================================================")
    print(f"Benchmark: {benchmark_name}")
    print(f"Adaptation: {t_adapt} frames -> Open-Loop Rollout: {rollout_horizon} steps")
    print(f"======================================================================")

    static_errors = np.zeros((N, rollout_horizon))
    ttt_errors = np.zeros((N, rollout_horizon))

    for ep in range(N):
        model.reset_plastic_weights()

        states_seq = []
        with torch.no_grad():
            for t in range(1, T):
                st = model.dyn_neck(z_test[ep, t:t+1], z_test[ep, t-1:t])
                states_seq.append(st)
        states_seq = torch.cat(states_seq, dim=0) # (T-1, 128)

        # 1. Warm-up Adaptation with Soft-Constraint Gradient Projection
        for t in range(t_adapt):
            s_curr = states_seq[t:t+1]
            s_next = states_seq[t+1:t+2]

            x_t = model.context_mapping(s_curr)
            f_nom = model.invariant_core(s_curr)
            y_target = s_next - f_nom
            err = torch.matmul(x_t, model.W_t) - y_target

            g_fwd = torch.matmul(x_t.T, err)
            g_prox = model.beta * (model.W_t - model.W_0)
            G_raw = g_fwd + g_prox

            # Soft Projection: allow energy-leakage into variation subspace
            # G_eff = P_perp G_raw + leak_ratio * (I - P_perp) G_raw
            G_proj = torch.matmul(model.P_perp, G_raw)
            G_core = G_raw - G_proj
            G_eff = G_proj + leak_ratio * G_core

            model.W_t.sub_(lr_ttt * G_eff)

        W_adapted = model.W_t.clone()

        # 2. Open-loop Autoregressive Imagination
        s_init = states_seq[t_adapt:t_adapt+1]
        curr_static = s_init.clone()
        curr_ttt = s_init.clone()

        for h in range(rollout_horizon):
            target_idx = t_adapt + 1 + h
            if target_idx >= len(states_seq):
                break
            s_target = states_seq[target_idx:target_idx+1]

            with torch.no_grad():
                # Static autoregressive step
                p_static = model.forward_state_prediction(curr_static, W=model.W_0)
                static_errors[ep, h] = F.mse_loss(p_static, s_target).item()
                curr_static = p_static

                # TTT-Adapted autoregressive step
                p_ttt = model.forward_state_prediction(curr_ttt, W=W_adapted)
                ttt_errors[ep, h] = F.mse_loss(p_ttt, s_target).item()
                curr_ttt = p_ttt

    mean_static = np.mean(static_errors, axis=0)
    mean_ttt = np.mean(ttt_errors, axis=0)

    print("Step-by-step Autoregressive Compounding Error (MSE):")
    print(f" Horizon | Static World Model | TTT-Adapted Model | Error Reduction")
    print(f"------------------------------------------------------------------")
    for h in range(rollout_horizon):
        st_e = mean_static[h]
        tt_e = mean_ttt[h]
        red = (st_e - tt_e) / (st_e + 1e-8) * 100.0
        print(f" Step +{h+1:2d} |    {st_e:.8f}    |    {tt_e:.8f}   |   {red:+6.2f}%")

    cum_static = np.sum(mean_static)
    cum_ttt = np.sum(mean_ttt)
    cum_drop = (cum_static - cum_ttt) / (cum_static + 1e-8) * 100.0

    print(f"------------------------------------------------------------------")
    print(f"Cumulative {rollout_horizon}-Step Rollout MSE : Static = {cum_static:.8f} | TTT = {cum_ttt:.8f}")
    print(f"Overall Long-Horizon Error Reduction: {cum_drop:+.2f}%\n")

    return cum_drop

if __name__ == "__main__":
    drop_fric = run_calibrated_long_horizon_rollout(
        test_file="contact_test_high_friction.pt",
        benchmark_name="OOD Extreme Friction Braking (Long-Horizon Slide)",
        t_adapt=7,
        rollout_horizon=10,
        lr_ttt=0.45,
        leak_ratio=0.30
    )

    drop_mass = run_calibrated_long_horizon_rollout(
        test_file="contact_test_heavy_mass.pt",
        benchmark_name="OOD Heavy Mass Collision & Post-Impact Rollout",
        t_adapt=7,
        rollout_horizon=10,
        lr_ttt=0.45,
        leak_ratio=0.30
    )
