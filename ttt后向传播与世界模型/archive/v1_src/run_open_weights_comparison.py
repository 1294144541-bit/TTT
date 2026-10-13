# Comparative Experiment: Online Fine-Tuning Baseline vs. TTT vs. Bayesian-TTT
# "Static World Model with Open Weights" -> Online Full / Core Fine-Tuning (Naive TTA Baseline)
# Protocol:
# 1. Model A: Frozen Backbone (Static Baseline)
# 2. Model B: Online Fine-Tuning Core f_\theta (Open weights on all dynamics parameters, unconstrained SGD)
# 3. Model C: In-Weights TTT (Zero-space projected local plastic layer W_t)
# 4. Model D: Bayesian-TTT (TTT + Kalman Innovation Filter)

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

def run_open_weights_experiment():
    print("==========================================================================================")
    print("Executing Comparative Experiment: Open-Weights Online Fine-Tuning vs. TTT vs. Bayesian-TTT")
    print("==========================================================================================")

    # 1. Load Real Video States (59 physical momentum states, 128-dim)
    frames = torch.load(REAL_VIDEO_TENSOR)
    total_frames = frames.shape[0]

    # Instantiate feature extractor
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

    # Extract 768-dim features and construct momentum states
    batch_size = 16
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
    states = torch.cat(states, dim=0) # (59, 128)
    T_states = states.shape[0]

    t_warmup = 35
    rollout_h = 12

    # -------------------------------------------------------------
    # Setup Models:
    # -------------------------------------------------------------
    # Model 1: Frozen Static Baseline
    model_static = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64)
    model_static.load_state_dict(torch.load(CKPT_PATH))
    model_static.eval()

    # Model 2: Open-Weights Online Fine-Tuning Baseline (Unconstrained SGD on core f_\theta)
    model_finetune = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64)
    model_finetune.load_state_dict(torch.load(CKPT_PATH))
    # Open weights on invariant core f_\theta
    for p in model_finetune.invariant_core.parameters():
        p.requires_grad = True
    optimizer_finetune = torch.optim.SGD(model_finetune.invariant_core.parameters(), lr=0.08)

    # Model 3: Pure-TTT (In-Weights Plastic W_t, Frozen f_\theta, Null-space Projection)
    model_ttt = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64, lr_ttt=0.35, beta_proximal=0.002, core_rank=16)
    model_ttt.load_state_dict(torch.load(CKPT_PATH))
    model_ttt.eval()
    model_ttt.reset_plastic_weights()

    # Model 4: Bayesian-TTT (High Innovation Trust for real domain shift)
    bayes_manager = BayesianTTTDynamicsManager(
        dyn_dim=128,
        obs_noise_var=1e-5, # High trust in real video domain shift
        process_noise_var=5e-5,
        prior_var=1e-2
    )
    bayes_manager.reset_belief()

    # -------------------------------------------------------------
    # Phase 1: Online Streaming Adaptation (Frames 0 to t_warmup-1)
    # -------------------------------------------------------------
    print(f"\n--- Phase 1: Online Streaming Adaptation (Frames 1 to {t_warmup}) ---")
    err_static_list = []
    err_finetune_list = []
    err_ttt_list = []
    err_bayes_list = []

    for t in range(t_warmup):
        s_curr = states[t:t+1]
        s_next_true = states[t+1:t+2]

        with torch.no_grad():
            # 1. Static Error (before step)
            p_stat = model_static.forward_state_prediction(s_curr, W=model_static.W_0)
            err_stat = F.mse_loss(p_stat, s_next_true).item()
            err_static_list.append(err_stat)

            # 2. Open-Weights Fine-Tuning Prediction (before step)
            p_ft = model_finetune.invariant_core(s_curr)
            err_ft = F.mse_loss(p_ft, s_next_true).item()
            err_finetune_list.append(err_ft)

            # 3. Pure-TTT Prediction (before step)
            p_ttt = model_ttt.forward_state_prediction(s_curr, W=model_ttt.W_t)
            err_ttt = F.mse_loss(p_ttt, s_next_true).item()
            err_ttt_list.append(err_ttt)

            # 4. Bayesian-TTT Prediction (before step)
            f_nom = model_ttt.invariant_core(s_curr)
            x_b = model_ttt.context_mapping(s_curr)
            p_bayes = f_nom + bayes_manager.kalman_gain * torch.matmul(x_b, model_ttt.W_t)
            err_bayes = F.mse_loss(p_bayes, s_next_true).item()
            err_bayes_list.append(err_bayes)

        # Update Model 2: Open-Weights Online Fine-Tuning (Unconstrained Backprop)
        optimizer_finetune.zero_grad()
        loss_ft = F.mse_loss(model_finetune.invariant_core(s_curr), s_next_true)
        loss_ft.backward()
        optimizer_finetune.step()

        # Update Model 3 & 4: TTT Step
        bayes_manager.update_with_feedback(s_observed=s_next_true, s_pred_ttt=p_ttt, s_pred_nom=f_nom)
        y_target = s_next_true - f_nom
        err_vec = torch.matmul(x_b, model_ttt.W_t) - y_target
        g_fwd = torch.matmul(x_b.T, err_vec)
        g_prox = model_ttt.beta * (model_ttt.W_t - model_ttt.W_0)
        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(model_ttt.P_perp, G_raw)
        G_eff = G_proj + 0.30 * (G_raw - G_proj)
        model_ttt.W_t.sub_(model_ttt.lr_ttt * G_eff)

    print("Phase 1 Online Streaming MSE (Lower is Better):")
    print(f"  * [1] Frozen Static Model            : {np.mean(err_static_list):.8f}")
    print(f"  * [2] Open-Weights Fine-Tuning (SGD) : {np.mean(err_finetune_list):.8f}")
    print(f"  * [3] In-Weights TTT (Projected W_t) : {np.mean(err_ttt_list):.8f}")
    print(f"  * [4] Bayesian-TTT Filtered Model    : {np.mean(err_bayes_list):.8f}")

    # -------------------------------------------------------------
    # Phase 2: Open-Loop Multi-Step Autoregressive Rollout (12 steps)
    # -------------------------------------------------------------
    print(f"\n--- Phase 2: Open-Loop Future Rollout ({rollout_h} Steps into the Future) ---")
    s_init = states[t_warmup:t_warmup+1]

    curr_static = s_init.clone()
    curr_ft = s_init.clone()
    curr_ttt = s_init.clone()
    curr_bayes = s_init.clone()
    prev_bayes_st = states[t_warmup-1:t_warmup]

    rollout_static = []
    rollout_ft = []
    rollout_ttt = []
    rollout_bayes = []

    W_adapted = model_ttt.W_t.clone()

    for h in range(rollout_h):
        target_idx = t_warmup + 1 + h
        if target_idx >= T_states:
            break
        s_target = states[target_idx:target_idx+1]

        with torch.no_grad():
            # [1] Frozen Static Model
            p_s = model_static.forward_state_prediction(curr_static, W=model_static.W_0)
            rollout_static.append(F.mse_loss(p_s, s_target).item())
            curr_static = p_s

            # [2] Open-Weights Fine-Tuned Model (Core f_\theta weights were altered online)
            p_ft = model_finetune.invariant_core(curr_ft)
            rollout_ft.append(F.mse_loss(p_ft, s_target).item())
            curr_ft = p_ft

            # [3] In-Weights TTT Model
            p_t = model_ttt.forward_state_prediction(curr_ttt, W=W_adapted)
            rollout_ttt.append(F.mse_loss(p_t, s_target).item())
            curr_ttt = p_t

            # [4] Bayesian-TTT Model
            vel_norm = torch.norm(curr_bayes - prev_bayes_st).item()
            shrinkage = bayes_manager.compute_rollout_shrinkage(horizon_step=h, velocity_norm=vel_norm)
            f_nom = model_ttt.invariant_core(curr_bayes)
            x_b = model_ttt.context_mapping(curr_bayes)
            p_b = f_nom + shrinkage * torch.matmul(x_b, W_adapted)
            rollout_bayes.append(F.mse_loss(p_b, s_target).item())
            prev_bayes_st = curr_bayes
            curr_bayes = p_b

    cum_s = np.sum(rollout_static)
    cum_ft = np.sum(rollout_ft)
    cum_t = np.sum(rollout_ttt)
    cum_b = np.sum(rollout_bayes)

    print("Step-by-step Autoregressive Compounding Error (MSE):")
    print(f" Horizon | Frozen Static |  Open-Weights FT  |   In-Weights TTT  |    Bayesian-TTT   ")
    print(f"-------------------------------------------------------------------------------------")
    for h in range(len(rollout_static)):
        print(f" Step +{h+1:2d} |  {rollout_static[h]:.8f} |    {rollout_ft[h]:.8f}   |    {rollout_ttt[h]:.8f}   |    {rollout_bayes[h]:.8f}   ")

    print(f"-------------------------------------------------------------------------------------")
    print(f"Cumulative {len(rollout_static)}-Step Rollout MSE:")
    print(f"  * [1] Frozen Static Model            : {cum_s:.8f} (100.0%)")
    print(f"  * [2] Open-Weights Fine-Tuning (SGD) : {cum_ft:.8f} ({(cum_ft/cum_s)*100.2:.1f}% vs Static)")
    print(f"  * [3] In-Weights TTT (Projected W_t) : {cum_t:.8f} ({(cum_t/cum_s)*100:.2f}% vs Static - Drop {((cum_s-cum_t)/cum_s)*100:.2f}%)")
    print(f"  * [4] Bayesian-TTT Filtered Model    : {cum_b:.8f} ({(cum_b/cum_s)*100:.2f}% vs Static - Drop {((cum_s-cum_b)/cum_s)*100:.2f}%)\n")

    return {
        "cum_static": cum_s,
        "cum_finetune": cum_ft,
        "cum_ttt": cum_t,
        "cum_bayes": cum_b
    }

if __name__ == "__main__":
    run_open_weights_experiment()
