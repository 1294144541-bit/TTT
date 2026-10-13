# Benchmark: Standard SGD-TTT vs. Calibrated Bayesian In-Weights TTT (Kalman Natural Gradient)
# Evaluates on 200-frame real-world video trajectory:
# 1. Online Streaming Adaptation (150 frames)
# 2. Open-Loop Future Autoregressive Rollout (45 frames)

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel
from src.calibrated_bayesian_in_weights import CalibratedBayesianInWeightsTTT

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
CKPT_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa_physics_neck.pt"
LONG_VIDEO_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/real_long_video_200frames.pt"

def run_calibrated_in_weights_benchmark():
    print("==========================================================================================")
    print("Executing In-Weights Bayesian TTT (Kalman Natural Gradient on Parameter Manifold)")
    print("==========================================================================================")

    frames = torch.load(LONG_VIDEO_PATH)
    total_frames = frames.shape[0]

    base_model = VJEPAPhysicsTTTWorldModel(
        vjepa_weights_path=WEIGHTS_PATH,
        dyn_dim=128,
        context_k_dim=64,
        lr_ttt=0.35,
        beta_proximal=0.002,
        core_rank=16
    )
    base_model.load_state_dict(torch.load(CKPT_PATH))
    base_model.eval()

    batch_size = 20
    feats = []
    with torch.no_grad():
        for i in range(0, total_frames, batch_size):
            b = frames[i:i+batch_size]
            z = base_model.encoder(b)
            feats.append(z.cpu())
    z_all = torch.cat(feats, dim=0)

    states = []
    with torch.no_grad():
        for t in range(1, total_frames):
            st = base_model.dyn_neck(z_all[t:t+1], z_all[t-1:t])
            states.append(st)
    states = torch.cat(states, dim=0) # (199, 128)
    T_states = states.shape[0]

    t_warmup = 150
    rollout_h = 45

    # 1. Setup Models
    # Model 1: Standard SGD-TTT (Heuristic scalar learning rate)
    model_sgd = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64, lr_ttt=0.35, beta_proximal=0.002, core_rank=16)
    model_sgd.load_state_dict(torch.load(CKPT_PATH))
    model_sgd.eval()
    model_sgd.reset_plastic_weights()

    # Model 2: Calibrated Bayesian In-Weights TTT (Kalman Preconditioned Parameter Covariance)
    bayes_in_weights = CalibratedBayesianInWeightsTTT(
        d_k=64,
        d_v=128,
        prior_var=1.0,
        obs_noise=1e-2,
        forgetting_factor=0.99,
        lr_bayes=0.20,
        beta_ou=0.002,
        max_step_norm=0.35
    )
    bayes_in_weights.reset()

    P_perp = model_sgd.P_perp

    # -------------------------------------------------------------
    # Phase 1: 150-Frame Online Streaming Adaptation
    # -------------------------------------------------------------
    print(f"\n--- Phase 1: 150-Frame Streaming Adaptation ---")
    err_sgd = []
    err_bayes = []

    for t in range(t_warmup):
        s_curr = states[t:t+1]
        s_next_true = states[t+1:t+2]

        with torch.no_grad():
            f_nom = model_sgd.invariant_core(s_curr)
            x_t = model_sgd.context_mapping(s_curr)

            # Model 1: SGD prediction before step
            p_s = f_nom + torch.matmul(x_t, model_sgd.W_t)
            err_sgd.append(F.mse_loss(p_s, s_next_true).item())

            # Model 2: Bayesian in-weights prediction before step
            p_b = f_nom + torch.matmul(x_t, bayes_in_weights.W_t)
            err_bayes.append(F.mse_loss(p_b, s_next_true).item())

        y_target = s_next_true - f_nom

        # Update Model 1: SGD-TTT step
        err_vec = torch.matmul(x_t, model_sgd.W_t) - y_target
        g_fwd = torch.matmul(x_t.T, err_vec)
        g_prox = model_sgd.beta * (model_sgd.W_t - model_sgd.W_0)
        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(P_perp, G_raw)
        G_eff = G_proj + 0.30 * (G_raw - G_proj)
        model_sgd.W_t.sub_(model_sgd.lr_ttt * G_eff)

        # Update Model 2: Bayesian Kalman In-Weights Update
        bayes_in_weights.update_step(x_t, y_target, P_perp)

    mean_sgd_stream = np.mean(err_sgd)
    mean_bayes_stream = np.mean(err_bayes)

    print("Phase 1 Online Streaming MSE (Lower is Better):")
    print(f"  * Standard SGD-TTT (Scalar LR)        : {mean_sgd_stream:.8f}")
    print(f"  * Bayesian In-Weights TTT (Kalman P_t) : {mean_bayes_stream:.8f}")
    print(f"  * Streaming Adaptation Gain           : {((mean_sgd_stream - mean_bayes_stream)/mean_sgd_stream)*100:+.2f}%")
    print(f"  * Final SGD Weight Norm ||W_150||     : {torch.norm(model_sgd.W_t).item():.4f}")
    print(f"  * Final Bayesian Weight Norm ||W_150||: {torch.norm(bayes_in_weights.W_t).item():.4f}")

    # -------------------------------------------------------------
    # Phase 2: 45-Step Ultra-Long Autoregressive Rollout
    # -------------------------------------------------------------
    print(f"\n--- Phase 2: 45-Step Ultra-Long Autoregressive Rollout (Frames 151 to 195) ---")
    s_init = states[t_warmup:t_warmup+1]

    curr_sgd = s_init.clone()
    curr_bayes = s_init.clone()

    roll_sgd = []
    roll_bayes = []

    W_sgd_150 = model_sgd.W_t.clone()
    W_bayes_150 = bayes_in_weights.W_t.clone()

    for h in range(rollout_h):
        target_idx = t_warmup + 1 + h
        if target_idx >= T_states:
            break
        s_target = states[target_idx:target_idx+1]

        with torch.no_grad():
            # [1] Standard SGD-TTT Step
            p_s = model_sgd.forward_state_prediction(curr_sgd, W=W_sgd_150)
            roll_sgd.append(F.mse_loss(p_s, s_target).item())
            curr_sgd = p_s

            # [2] Bayesian In-Weights TTT Step
            p_b = model_sgd.forward_state_prediction(curr_bayes, W=W_bayes_150)
            roll_bayes.append(F.mse_loss(p_b, s_target).item())
            curr_bayes = p_b

    cum_sgd = np.sum(roll_sgd)
    cum_bayes = np.sum(roll_bayes)
    roll_gain = (cum_sgd - cum_bayes) / cum_sgd * 100.0

    print("Milestone Autoregressive MSE across 45-Step Horizon:")
    print(f" Horizon   |   Standard SGD-TTT    | Bayesian In-Weights TTT | Gain ")
    print(f"----------------------------------------------------------------------")
    for step_idx in [0, 4, 9, 19, 29, 39, 44]:
        if step_idx < len(roll_sgd):
            s_val = roll_sgd[step_idx]
            b_val = roll_bayes[step_idx]
            g_step = (s_val - b_val) / (s_val + 1e-8) * 100.0
            print(f" Step +{step_idx+1:2d}  |      {s_val:.8f}     |       {b_val:.8f}       | {g_step:+6.2f}%")

    print(f"----------------------------------------------------------------------")
    print(f"Cumulative 45-Step Rollout MSE:")
    print(f"  * Standard SGD-TTT (Scalar LR)        : {cum_sgd:.8f}")
    print(f"  * Bayesian In-Weights TTT (Kalman P_t) : {cum_bayes:.8f}")
    print(f"  * Open-Loop Imagination Gain          : {roll_gain:+.2f}%\n")

    return {
        "cum_sgd": cum_sgd,
        "cum_bayes": cum_bayes,
        "gain": roll_gain
    }

if __name__ == "__main__":
    run_calibrated_in_weights_benchmark()
