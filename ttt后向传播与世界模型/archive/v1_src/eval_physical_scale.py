# Visual Physical Scale and Dynamics Evaluation Pipeline
# Evaluates:
# 1. Video Latent Future Trajectory Prediction MSE (Static World Model vs. TTT-Augmented World Model)
# 2. Linear Probing R^2 score of adapted weights W_t predicting true physical mass m and friction mu
# 3. Parameter Drift & Energy Bounds

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vision_ttt_model import VisualTTTWorldModel

def evaluate_vision_physical_scale(dataset_path: str, benchmark_name: str, lr_ttt: float = 1.5):
    ckpt_path = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vision_nominal_model.pt"
    
    data = torch.load(dataset_path)
    videos = data["videos"] # (N, T=25, 3, 64, 64)
    params = data["params"] # (N, T, 4) -> [m, mu, e, g]
    N, T, C, H, W = videos.shape

    model = VisualTTTWorldModel(latent_dim=32, context_k_dim=32, lr_ttt=lr_ttt, beta_proximal=0.005, core_rank=8)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    static_latent_errors = []
    ttt_latent_errors = []
    weight_drift_history = []
    
    all_adapted_W = []
    all_true_params = []

    print(f"==================================================")
    print(f"Running Visual Physics Benchmark: {benchmark_name}")
    print(f"Video sequences: {N}, Sequence length: {T} frames, Test-time LR: {lr_ttt}")
    print(f"==================================================")

    for ep in range(N):
        model.reset_plastic_weights()
        ep_static_errs = []
        ep_ttt_errs = []

        with torch.no_grad():
            z_seq = model.encoder(videos[ep]) # (T, 32)

        for t in range(T - 1):
            z_t = z_seq[t:t+1]
            z_next_true = z_seq[t+1:t+2]
            p_t = params[ep, t].numpy()

            # 1. Forward prediction from Static World Model (W_t = W_0 = 0)
            with torch.no_grad():
                z_pred_static = model.forward_latent_prediction(z_t, W=model.W_0)
                err_static = F.mse_loss(z_pred_static, z_next_true).item()
                ep_static_errs.append(err_static)

                # 2. Forward prediction from TTT-Augmented World Model (before step)
                z_pred_ttt = model.forward_latent_prediction(z_t, W=model.W_t)
                err_ttt = F.mse_loss(z_pred_ttt, z_next_true).item()
                ep_ttt_errs.append(err_ttt)

            # 3. Test-time streaming adaptation update using self-supervised latent prediction residual
            step_info = model.ttt_step(z_t, z_next_true)
            weight_drift_history.append(step_info["weight_drift"])

            all_adapted_W.append(model.W_t.detach().cpu().numpy().flatten())
            all_true_params.append(p_t)

        static_latent_errors.append(ep_static_errs)
        ttt_latent_errors.append(ep_ttt_errs)

    mean_static_mse = np.mean(static_latent_errors)
    mean_ttt_mse = np.mean(ttt_latent_errors)
    relative_reduction = (mean_static_mse - mean_ttt_mse) / mean_static_mse * 100.0

    # Linear Probe for Physical Decodability using PCA or Ridge Regression
    W_mat = np.array(all_adapted_W)      # (N*(T-1), 1024)
    P_mat = np.array(all_true_params)   # (N*(T-1), 4)

    # Standardize features before ridge regression
    w_mean = np.mean(W_mat, axis=0, keepdims=True)
    w_std = np.std(W_mat, axis=0, keepdims=True) + 1e-6
    W_norm = (W_mat - w_mean) / w_std

    # Linear probe with SVD / Ridge
    U, S, Vt = np.linalg.svd(W_norm, full_matrices=False)
    # Project to top-20 principal components of weight drift
    k_comp = min(20, S.shape[0])
    W_proj = U[:, :k_comp] @ np.diag(S[:k_comp])
    W_proj_b = np.concatenate([W_proj, np.ones((W_proj.shape[0], 1))], axis=1)

    lambda_reg = 1.0
    A = W_proj_b.T @ W_proj_b + lambda_reg * np.eye(W_proj_b.shape[1])
    b = W_proj_b.T @ P_mat[:, :2]
    coeffs = np.linalg.solve(A, b)
    pred_physics = W_proj_b @ coeffs

    # Compute R^2 metric for mass and friction
    ss_tot = np.sum((P_mat[:, :2] - np.mean(P_mat[:, :2], axis=0)) ** 2, axis=0)
    ss_res = np.sum((P_mat[:, :2] - pred_physics) ** 2, axis=0)
    r2_scores = 1.0 - (ss_res / (ss_tot + 1e-8))

    print(f"Results Summary for: {benchmark_name}")
    print(f" - Static Video World Model Latent MSE : {mean_static_mse:.6f}")
    print(f" - TTT-Augmented World Model Latent MSE: {mean_ttt_mse:.6f}")
    print(f" - Relative Latent Prediction Error Drop : {relative_reduction:.2f}%")
    print(f" - Linear Probe Decodability (R^2 - Mass)    : {r2_scores[0]:.4f}")
    print(f" - Linear Probe Decodability (R^2 - Friction): {r2_scores[1]:.4f}")
    print(f" - Mean In-Weights Drift Norm ||W_t - W_0|| : {np.mean(weight_drift_history):.4f}\n")

    return {
        "static_mse": mean_static_mse,
        "ttt_mse": mean_ttt_mse,
        "error_drop": relative_reduction,
        "r2_mass": r2_scores[0],
        "r2_friction": r2_scores[1]
    }

if __name__ == "__main__":
    base_dir = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
    res_nom = evaluate_vision_physical_scale(
        os.path.join(base_dir, "vision_test_nominal.pt"),
        "In-Distribution Nominal Physical Trajectories",
        lr_ttt=1.5
    )
    res_heavy = evaluate_vision_physical_scale(
        os.path.join(base_dir, "vision_test_heavy_mu.pt"),
        "OOD Extreme Mass & Friction Shock Drift",
        lr_ttt=2.0
    )
    res_low_g = evaluate_vision_physical_scale(
        os.path.join(base_dir, "vision_test_low_g.pt"),
        "OOD Low Gravity & High Elastic Restitution",
        lr_ttt=2.0
    )
