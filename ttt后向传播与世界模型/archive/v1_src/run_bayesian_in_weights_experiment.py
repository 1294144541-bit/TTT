# Benchmark: Standard TTT (SGD) vs. Bayesian In-Weights TTT (Kalman Natural Gradient)
# Evaluates on the 200-frame real-world video trajectory:
# 1. Online Streaming Adaptation (150 frames)
# 2. Open-Loop Future Autoregressive Rollout (45 frames)

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel
from src.bayesian_in_weights_engine import MatrixBayesianTTTEngine

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
CKPT_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa_physics_neck.pt"
LONG_VIDEO_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/real_long_video_200frames.pt"

def run_bayesian_in_weights_experiment():
    print("==========================================================================================")
    print("Executing In-Weights Bayesian TTT (Kalman Natural Gradient on Parameter Manifold)")
    print("==========================================================================================")

    # 1. Load Data & Extract Features
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

    # -------------------------------------------------------------
    # Setup Models:
    # -------------------------------------------------------------
    # Model 1: Standard In-Weights TTT (SGD with constant learning rate \eta)
    model_sgd_ttt = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64, lr_ttt=0.35, beta_proximal=0.002, core_rank=16)
    model_sgd_ttt.load_state_dict(torch.load(CKPT_PATH))
    model_sgd_ttt.eval()
    model_sgd_ttt.reset_plastic_weights()

    # Model 2: Bayesian In-Weights TTT (Kalman Natural Gradient on Parameter Space)
    bayes_engine = MatrixBayesianTTTEngine(
        d_k=64,
        d_v=128,
        obs_noise_var=5e-5,
        prior_precision=2.0,
        forgetting_factor=0.99
    )
    bayes_engine.reset_belief()

    # -------------------------------------------------------------
    # Phase 1: 150-Frame Online Streaming Adaptation
    # -------------------------------------------------------------
    print(f"\n--- Phase 1: 150-Frame Streaming Adaptation ---")
    err_sgd_list = []
    err_bayes_weight_list = []
    param_uncertainty_list = []

    P_perp = model_sgd_ttt.P_perp

    for t in range(t_warmup):
        s_curr = states[t:t+1]
        s_next_true = states[t+1:t+2]

        with torch.no_grad():
            f_nom = model_sgd_ttt.invariant_core(s_curr)
            x_t = model_sgd_ttt.context_mapping(s_curr)

            # 1. Standard TTT Prediction (before step)
            p_sgd = f_nom + torch.matmul(x_t, model_sgd_ttt.W_t)
            err_sgd_list.append(F.mse_loss(p_sgd, s_next_true).item())

            # 2. Bayesian In-Weights TTT Prediction (before step)
            p_bayes_w = f_nom + torch.matmul(x_t, bayes_engine.W_t)
            err_bayes_weight_list.append(F.mse_loss(p_bayes_w, s_next_true).item())

        y_target = s_next_true - f_nom

        # Update Model 1: Standard SGD step
        err_vec = torch.matmul(x_t, model_sgd_ttt.W_t) - y_target
        g_fwd = torch.matmul(x_t.T, err_vec)
        g_prox = model_sgd_ttt.beta * (model_sgd_ttt.W_t - model_sgd_ttt.W_0)
        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(P_perp, G_raw)
        G_eff = G_proj + 0.30 * (G_raw - G_proj)
        model_sgd_ttt.W_t.sub_(model_sgd_ttt.lr_ttt * G_eff)

        # Update Model 2: Matrix Bayesian Kalman Update on W_t
        step_bayes = bayes_engine.bayesian_weight_update_step(x_t, y_target, P_perp)
        param_uncertainty_list.append(step_bayes["param_trace_cov"])

    print("Phase 1 Online Streaming MSE (Lower is Better):")
    print(f"  * Standard SGD-TTT (Heuristic LR)     : {np.mean(err_sgd_list):.8f}")
    print(f"  * Bayesian In-Weights TTT (Kalman LR) : {np.mean(err_bayes_weight_list):.8f}")
    print(f"  * Relative Streaming Gain             : {((np.mean(err_sgd_list) - np.mean(err_bayes_weight_list)) / np.mean(err_sgd_list)) * 100:.2f}%")
    print(f"  * Parameter Covariance Trace Tr(P_t) : {param_uncertainty_list[0]:.2f} -> {param_uncertainty_list[-1]:.4f} (Posterior Convergence)")

    # -------------------------------------------------------------
    # Phase 2: 45-Step Open-Loop Autoregressive Rollout
    # -------------------------------------------------------------
    print(f"\n--- Phase 2: 45-Step Ultra-Long Autoregressive Rollout (Frames 151 to 195) ---")
    s_init = states[t_warmup:t_warmup+1]

    curr_sgd = s_init.clone()
    curr_bayes_w = s_init.clone()

    roll_sgd = []
    roll_bayes_w = []

    W_sgd_150 = model_sgd_ttt.W_t.clone()
    W_bayes_150 = bayes_engine.W_t.clone()

    for h in range(rollout_h):
        target_idx = t_warmup + 1 + h
        if target_idx >= T_states:
            break
        s_target = states[target_idx:target_idx+1]

        with torch.no_grad():
            # [1] Standard SGD-TTT Autoregressive Step
            p_sgd = model_sgd_ttt.forward_state_prediction(curr_sgd, W=W_sgd_150)
            roll_sgd.append(F.mse_loss(p_sgd, s_target).item())
            curr_sgd = p_sgd

            # [2] Bayesian In-Weights TTT Autoregressive Step
            # Uses parameter posterior mean W_bayes_150
            p_bw = model_sgd_ttt.forward_state_prediction(curr_bayes_w, W=W_bayes_150)
            roll_bayes_w.append(F.mse_loss(p_bw, s_target).item())
            curr_bayes_w = p_bw

    cum_sgd = np.sum(roll_sgd)
    cum_bw = np.sum(roll_bayes_w)
    gain_roll = (cum_sgd - cum_bw) / cum_sgd * 100.0

    print("Milestone Autoregressive MSE across 45-Step Horizon:")
    print(f" Horizon   |   Standard SGD-TTT    | Bayesian In-Weights TTT | Gain ")
    print(f"----------------------------------------------------------------------")
    for step_idx in [0, 4, 9, 19, 29, 39, 44]:
        if step_idx < len(roll_sgd):
            s_val = roll_sgd[step_idx]
            b_val = roll_bayes_w[step_idx]
            g_step = (s_val - b_val) / (s_val + 1e-8) * 100.0
            print(f" Step +{step_idx+1:2d}  |      {s_val:.8f}     |       {b_val:.8f}       | {g_step:+6.2f}%")

    print(f"----------------------------------------------------------------------")
    print(f"Cumulative 45-Step Rollout MSE:")
    print(f"  * Standard SGD-TTT (Heuristic LR)     : {cum_sgd:.8f}")
    print(f"  * Bayesian In-Weights TTT (Kalman LR) : {cum_bw:.8f}")
    print(f"  * Open-Loop Imagination Gain          : {gain_roll:+.2f}%\n")

    return {
        "cum_sgd": cum_sgd,
        "cum_bayes_w": cum_bw,
        "gain": gain_roll
    }

if __name__ == "__main__":
    run_bayesian_in_weights_experiment()
