# 200-Frame Ultra-Long Real-World Video Evaluation Pipeline
# Compares:
# 1. Frozen Static World Model (Baseline)
# 2. Open-Weights Online Fine-Tuning (Unconstrained SGD on Core)
# 3. In-Weights TTT (Projected Plastic Layer W_t)
# 4. Bayesian-TTT (Kalman Innovation Filter)
# Protocol:
# - Streaming Adaptation across 150 frames (Measuring online tracking & forgetting)
# - Long-Horizon Autoregressive Rollout across 45 frames (Measuring deep open-loop imagination compounding error)

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
LONG_VIDEO_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/real_long_video_200frames.pt"

def run_200_frames_evaluation():
    print("==========================================================================================")
    print("Executing 200-Frame Ultra-Long Real-World Video Evaluation on Meta V-JEPA 2.1")
    print("==========================================================================================")

    # 1. Load Real Video (200 frames @ 224x224)
    frames = torch.load(LONG_VIDEO_PATH)
    total_frames = frames.shape[0]
    print(f"Loaded 200-frame video tensor: {frames.shape}")

    # 2. Extract Official V-JEPA 2.1 768-dim Representations in Batches
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

    print("Extracting V-JEPA ViT-B/16 768-dim embeddings across 200 frames...")
    batch_size = 20
    feats = []
    with torch.no_grad():
        for i in range(0, total_frames, batch_size):
            b = frames[i:i+batch_size]
            z = base_model.encoder(b)
            feats.append(z.cpu())
    z_all = torch.cat(feats, dim=0) # (200, 768)

    # 3. Construct Momentum States (199 states, 128-dim)
    states = []
    with torch.no_grad():
        for t in range(1, total_frames):
            st = base_model.dyn_neck(z_all[t:t+1], z_all[t-1:t])
            states.append(st)
    states = torch.cat(states, dim=0) # (199, 128)
    T_states = states.shape[0]
    print(f"Constructed physical momentum trajectory: {states.shape} (Length: {T_states} transitions)")

    # 4. Experimental Splits: 150 Frames Streaming Adaptation -> 45 Frames Open-Loop Rollout
    t_warmup = 150
    rollout_h = 45

    # Instantiate Models
    # Model 1: Frozen Static
    model_static = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64)
    model_static.load_state_dict(torch.load(CKPT_PATH))
    model_static.eval()

    # Model 2: Open-Weights Fine-Tuning
    model_ft = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64)
    model_ft.load_state_dict(torch.load(CKPT_PATH))
    for p in model_ft.invariant_core.parameters():
        p.requires_grad = True
    opt_ft = torch.optim.SGD(model_ft.invariant_core.parameters(), lr=0.06)

    # Model 3: In-Weights TTT
    model_ttt = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64, lr_ttt=0.35, beta_proximal=0.002, core_rank=16)
    model_ttt.load_state_dict(torch.load(CKPT_PATH))
    model_ttt.eval()
    model_ttt.reset_plastic_weights()

    # Model 4: Bayesian-TTT
    bayes_manager = BayesianTTTDynamicsManager(
        dyn_dim=128,
        obs_noise_var=2e-5,
        process_noise_var=5e-5,
        prior_var=1e-2
    )
    bayes_manager.reset_belief()

    # ------------------------------------------------------------------------------------------
    # Phase 1: 150-Frame Long-Term Continuous Streaming Adaptation
    # ------------------------------------------------------------------------------------------
    print(f"\n--- Phase 1: 150-Frame Long-Term Streaming Adaptation ---")
    err_static_seq = []
    err_ft_seq = []
    err_ttt_seq = []
    err_bayes_seq = []
    drift_norm_seq = []

    for t in range(t_warmup):
        s_curr = states[t:t+1]
        s_next_true = states[t+1:t+2]

        with torch.no_grad():
            # Static prediction
            p_s = model_static.forward_state_prediction(s_curr, W=model_static.W_0)
            err_static_seq.append(F.mse_loss(p_s, s_next_true).item())

            # Open-weights FT prediction
            p_ft = model_ft.invariant_core(s_curr)
            err_ft_seq.append(F.mse_loss(p_ft, s_next_true).item())

            # TTT prediction
            p_t = model_ttt.forward_state_prediction(s_curr, W=model_ttt.W_t)
            err_ttt_seq.append(F.mse_loss(p_t, s_next_true).item())

            # Bayesian prediction
            f_nom = model_ttt.invariant_core(s_curr)
            x_b = model_ttt.context_mapping(s_curr)
            p_b = f_nom + bayes_manager.kalman_gain * torch.matmul(x_b, model_ttt.W_t)
            err_bayes_seq.append(F.mse_loss(p_b, s_next_true).item())

        # Update Model 2: Open-Weights SGD
        opt_ft.zero_grad()
        loss_ft = F.mse_loss(model_ft.invariant_core(s_curr), s_next_true)
        loss_ft.backward()
        opt_ft.step()

        # Update Model 3 & 4: TTT Step
        bayes_manager.update_with_feedback(s_observed=s_next_true, s_pred_ttt=p_t, s_pred_nom=f_nom)
        y_target = s_next_true - f_nom
        err_vec = torch.matmul(x_b, model_ttt.W_t) - y_target
        g_fwd = torch.matmul(x_b.T, err_vec)
        g_prox = model_ttt.beta * (model_ttt.W_t - model_ttt.W_0)
        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(model_ttt.P_perp, G_raw)
        G_eff = G_proj + 0.30 * (G_raw - G_proj)
        model_ttt.W_t.sub_(model_ttt.lr_ttt * G_eff)
        drift_norm_seq.append(torch.norm(model_ttt.W_t).item())

    # Segment metrics: early (1-50), mid (51-100), late (101-150)
    print("Phase 1 Long-Horizon Streaming Tracking MSE across 150 Frames:")
    print(f"  * [Frames 001-050] : Static = {np.mean(err_static_seq[:50]):.6f} | FT = {np.mean(err_ft_seq[:50]):.6f} | TTT = {np.mean(err_ttt_seq[:50]):.6f}")
    print(f"  * [Frames 051-100] : Static = {np.mean(err_static_seq[50:100]):.6f} | FT = {np.mean(err_ft_seq[50:100]):.6f} | TTT = {np.mean(err_ttt_seq[50:100]):.6f}")
    print(f"  * [Frames 101-150] : Static = {np.mean(err_static_seq[100:150]):.6f} | FT = {np.mean(err_ft_seq[100:150]):.6f} | TTT = {np.mean(err_ttt_seq[100:150]):.6f}")
    print(f"  * [Overall 1-150]  : Static = {np.mean(err_static_seq):.6f} | FT = {np.mean(err_ft_seq):.6f} | TTT = {np.mean(err_ttt_seq):.6f}")
    print(f"  * Final Adapted In-Weights Norm ||W_150||: {drift_norm_seq[-1]:.4f} (Bounded Stable)")

    # ------------------------------------------------------------------------------------------
    # Phase 2: 45-Step Ultra-Long Open-Loop Autoregressive Rollout
    # ------------------------------------------------------------------------------------------
    print(f"\n--- Phase 2: 45-Step Ultra-Long Autoregressive Rollout (Frames 151 to 195) ---")
    s_init = states[t_warmup:t_warmup+1]

    curr_s = s_init.clone()
    curr_ft = s_init.clone()
    curr_ttt = s_init.clone()
    curr_b = s_init.clone()
    prev_b = states[t_warmup-1:t_warmup]

    roll_s = []
    roll_ft = []
    roll_ttt = []
    roll_b = []

    W_150 = model_ttt.W_t.clone()

    for h in range(rollout_h):
        target_idx = t_warmup + 1 + h
        if target_idx >= T_states:
            break
        s_target = states[target_idx:target_idx+1]

        with torch.no_grad():
            # [1] Frozen Static
            p_s = model_static.forward_state_prediction(curr_s, W=model_static.W_0)
            roll_s.append(F.mse_loss(p_s, s_target).item())
            curr_s = p_s

            # [2] Open-Weights FT
            p_ft = model_ft.invariant_core(curr_ft)
            roll_ft.append(F.mse_loss(p_ft, s_target).item())
            curr_ft = p_ft

            # [3] In-Weights TTT
            p_t = model_ttt.forward_state_prediction(curr_ttt, W=W_150)
            roll_ttt.append(F.mse_loss(p_t, s_target).item())
            curr_ttt = p_t

            # [4] Bayesian-TTT
            vel_norm = torch.norm(curr_b - prev_b).item()
            shrink = bayes_manager.compute_rollout_shrinkage(horizon_step=h, velocity_norm=vel_norm)
            f_nom = model_ttt.invariant_core(curr_b)
            x_b = model_ttt.context_mapping(curr_b)
            p_b = f_nom + shrink * torch.matmul(x_b, W_150)
            roll_b.append(F.mse_loss(p_b, s_target).item())
            prev_b = curr_b
            curr_b = p_b

    cum_s = np.sum(roll_s)
    cum_ft = np.sum(roll_ft)
    cum_ttt = np.sum(roll_ttt)
    cum_b = np.sum(roll_b)

    print("Milestone Autoregressive MSE across 45-Step Horizon:")
    print(f" Horizon   | Frozen Static |  Open-Weights FT  |   In-Weights TTT  |    Bayesian-TTT   ")
    print(f"---------------------------------------------------------------------------------------")
    for step_idx in [0, 4, 9, 19, 29, 39, 44]:
        if step_idx < len(roll_s):
            print(f" Step +{step_idx+1:2d}  |  {roll_s[step_idx]:.6f}     |    {roll_ft[step_idx]:.6f}       |    {roll_ttt[step_idx]:.6f}       |    {roll_b[step_idx]:.6f}   ")

    print(f"---------------------------------------------------------------------------------------")
    print(f"Cumulative 45-Step Ultra-Long Rollout Compounding MSE:")
    print(f"  * [1] Frozen Static Model            : {cum_s:.6f} (100.0%)")
    print(f"  * [2] Open-Weights Fine-Tuning (SGD) : {cum_ft:.6f} ({(cum_ft/cum_s)*100:.2f}% vs Static)")
    print(f"  * [3] In-Weights TTT (Projected W_t) : {cum_ttt:.6f} ({(cum_ttt/cum_s)*100:.2f}% vs Static - Drop {((cum_s-cum_ttt)/cum_s)*100:.2f}%)")
    print(f"  * [4] Bayesian-TTT Filtered Model    : {cum_b:.6f} ({(cum_b/cum_s)*100:.2f}% vs Static - Drop {((cum_s-cum_b)/cum_s)*100:.2f}%)\n")

    return {
        "cum_static": cum_s,
        "cum_ft": cum_ft,
        "cum_ttt": cum_ttt,
        "cum_b": cum_b
    }

if __name__ == "__main__":
    run_200_frames_evaluation()
