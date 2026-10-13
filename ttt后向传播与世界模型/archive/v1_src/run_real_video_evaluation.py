# Real-World Video Evaluation Pipeline for Bayesian-TTT World Model
# Evaluates:
# 1. Official Meta V-JEPA 2.1 85.65M feature extraction on real camera video frames (60 frames @ 224x224)
# 2. Online streaming adaptation: Static World Model vs. Pure-TTT vs. Bayesian-TTT
# 3. Compares Latent Predictive Drift, Noise Resilience, and Future Horizon Extrapolation

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel
from src.bayesian_ttt_manager import BayesianTTTDynamicsManager

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
CKPT_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa_physics_neck.pt"
REAL_VIDEO_TENSOR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/real_video_frames.pt"

def evaluate_real_world_video():
    print("==========================================================================================")
    print("Executing Real-World Video Evaluation Pipeline on Meta V-JEPA 2.1 + Bayesian-TTT")
    print("==========================================================================================")

    # 1. Load Model
    model = VJEPAPhysicsTTTWorldModel(
        vjepa_weights_path=WEIGHTS_PATH,
        dyn_dim=128,
        context_k_dim=64,
        lr_ttt=0.35,
        beta_proximal=0.002,
        core_rank=16
    )
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    bayes_manager = BayesianTTTDynamicsManager(
        dyn_dim=128,
        obs_noise_var=1e-4, # Calibrated for real-world sensor noise
        process_noise_var=2e-5,
        prior_var=1e-3
    )

    # 2. Load Real Video Tensor (60 frames, 3, 224, 224)
    frames = torch.load(REAL_VIDEO_TENSOR)
    total_frames = frames.shape[0]
    print(f"Loaded real-world video tensor: {frames.shape} (Total {total_frames} frames)")

    # 3. Extract 768-dim V-JEPA features in batches
    batch_size = 16
    feats = []
    with torch.no_grad():
        for i in range(0, total_frames, batch_size):
            b = frames[i:i+batch_size]
            z = model.encoder(b)
            feats.append(z.cpu())
    z_all = torch.cat(feats, dim=0) # (60, 768)
    print(f"Extracted V-JEPA 768-dim spatiotemporal embeddings: {z_all.shape}")

    # 4. Construct momentum states: s_t = Projector(z_t, z_{t-1})
    states = []
    with torch.no_grad():
        for t in range(1, total_frames):
            st = model.dyn_neck(z_all[t:t+1], z_all[t-1:t])
            states.append(st)
    states = torch.cat(states, dim=0) # (59, 128)
    T_states = states.shape[0]
    print(f"Constructed physical momentum states: {states.shape}")

    # 5. Continuous Online Streaming Adaptation Protocol (Frames 1 to 40)
    # Stream frames sequentially and measure forward prediction error before adaptation
    t_warmup = 35
    rollout_h = 12

    static_stream_errs = []
    pure_ttt_stream_errs = []
    bayes_ttt_stream_errs = []
    kalman_gain_history = []

    model.reset_plastic_weights()
    bayes_manager.reset_belief()

    print(f"\n--- Phase 1: Real-Time Streaming Adaptation (Frames 1 to {t_warmup}) ---")
    for t in range(t_warmup):
        s_curr = states[t:t+1]
        s_next_true = states[t+1:t+2]

        with torch.no_grad():
            f_nom = model.invariant_core(s_curr)
            # Static prediction
            pred_static = model.forward_state_prediction(s_curr, W=model.W_0)
            err_static = F.mse_loss(pred_static, s_next_true).item()
            static_stream_errs.append(err_static)

            # Pure-TTT prediction (before step)
            pred_pure = model.forward_state_prediction(s_curr, W=model.W_t)
            err_pure = F.mse_loss(pred_pure, s_next_true).item()
            pure_ttt_stream_errs.append(err_pure)

            # Bayesian-TTT prediction (before step)
            x_b = model.context_mapping(s_curr)
            pred_bayes = f_nom + bayes_manager.kalman_gain * torch.matmul(x_b, model.W_t)
            err_bayes = F.mse_loss(pred_bayes, s_next_true).item()
            bayes_ttt_stream_errs.append(err_bayes)

        # Measurement update for Bayesian filter
        bayes_info = bayes_manager.update_with_feedback(s_observed=s_next_true, s_pred_ttt=pred_pure, s_pred_nom=f_nom)
        kalman_gain_history.append(bayes_info["kalman_gain"])

        # TTT Gradient Step on W_t
        y_target = s_next_true - f_nom
        err_vec = torch.matmul(x_b, model.W_t) - y_target
        g_fwd = torch.matmul(x_b.T, err_vec)
        g_prox = model.beta * (model.W_t - model.W_0)
        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(model.P_perp, G_raw)
        G_core = G_raw - G_proj
        G_eff = G_proj + 0.30 * G_core
        model.W_t.sub_(model.lr_ttt * G_eff)

    mean_static_stream = np.mean(static_stream_errs)
    mean_pure_stream = np.mean(pure_ttt_stream_errs)
    mean_bayes_stream = np.mean(bayes_ttt_stream_errs)

    print(f"Phase 1 Streaming Online Step MSE:")
    print(f"  * Static World Model Baseline : {mean_static_stream:.8f}")
    print(f"  * Pure-TTT Streaming Model    : {mean_pure_stream:.8f}")
    print(f"  * Bayesian-TTT Filtered Model : {mean_bayes_stream:.8f}")
    print(f"  * Mean Bayesian Kalman Gain   : {np.mean(kalman_gain_history):.4f}")

    # 6. Phase 2: Open-Loop Future Multi-Step Rollout on Real Camera Footage (Horizon = 12 frames)
    print(f"\n--- Phase 2: Open-Loop Future Autoregressive Rollout ({rollout_h} Steps into the Future) ---")
    W_adapted = model.W_t.clone()
    s_init = states[t_warmup:t_warmup+1]

    curr_static = s_init.clone()
    curr_pure = s_init.clone()
    curr_bayes = s_init.clone()
    prev_bayes_st = states[t_warmup-1:t_warmup]

    rollout_static_errs = []
    rollout_pure_errs = []
    rollout_bayes_errs = []

    for h in range(rollout_h):
        target_idx = t_warmup + 1 + h
        if target_idx >= T_states:
            break
        s_target = states[target_idx:target_idx+1]

        with torch.no_grad():
            # [1] Static Model (W = 0)
            p_static = model.forward_state_prediction(curr_static, W=model.W_0)
            rollout_static_errs.append(F.mse_loss(p_static, s_target).item())
            curr_static = p_static

            # [2] Pure-TTT Model
            p_pure = model.forward_state_prediction(curr_pure, W=W_adapted)
            rollout_pure_errs.append(F.mse_loss(p_pure, s_target).item())
            curr_pure = p_pure

            # [3] Bayesian-TTT Model
            vel_norm = torch.norm(curr_bayes - prev_bayes_st).item()
            shrinkage = bayes_manager.compute_rollout_shrinkage(horizon_step=h, velocity_norm=vel_norm)
            f_nom = model.invariant_core(curr_bayes)
            x_b = model.context_mapping(curr_bayes)
            p_bayes = f_nom + shrinkage * torch.matmul(x_b, W_adapted)
            rollout_bayes_errs.append(F.mse_loss(p_bayes, s_target).item())
            prev_bayes_st = curr_bayes
            curr_bayes = p_bayes

    cum_static_roll = np.sum(rollout_static_errs)
    cum_pure_roll = np.sum(rollout_pure_errs)
    cum_bayes_roll = np.sum(rollout_bayes_errs)

    gain_bayes_vs_static = (cum_static_roll - cum_bayes_roll) / cum_static_roll * 100.0
    gain_bayes_vs_pure = (cum_pure_roll - cum_bayes_roll) / cum_pure_roll * 100.0

    print("Step-by-step Real Video Autoregressive MSE:")
    print(f" Horizon | Static Baseline |    Pure-TTT     |   Bayesian-TTT  | Bayes vs Pure Gain")
    print(f"--------------------------------------------------------------------------------------")
    for h in range(len(rollout_static_errs)):
        st = rollout_static_errs[h]
        pu = rollout_pure_errs[h]
        ba = rollout_bayes_errs[h]
        g = (pu - ba) / (pu + 1e-8) * 100.0
        print(f" Step +{h+1:2d} |   {st:.8f}  |   {pu:.8f}  |   {ba:.8f}  |       {g:+6.2f}%")

    print(f"--------------------------------------------------------------------------------------")
    print(f"Cumulative {len(rollout_static_errs)}-Step Real Video Rollout MSE:")
    print(f"  * Static Baseline World Model : {cum_static_roll:.8f}")
    print(f"  * Pure-TTT (Over-Correction)  : {cum_pure_roll:.8f}")
    print(f"  * Bayesian-TTT (Filtered)     : {cum_bayes_roll:.8f}")
    print(f"--------------------------------------------------------------------------------------")
    print(f"Real Video Extrapolation: Bayesian-TTT vs. Pure-TTT Gain: {gain_bayes_vs_pure:+.2f}%\n")

    return {
        "static_roll": cum_static_roll,
        "pure_roll": cum_pure_roll,
        "bayes_roll": cum_bayes_roll,
        "gain": gain_bayes_vs_pure
    }

if __name__ == "__main__":
    evaluate_real_world_video()
