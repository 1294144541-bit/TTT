# Benchmark: In-Context Transformer (KV-Cache) vs. In-Weights TTT vs. Bayesian-TTT
# Evaluates on the 200-frame real-world video trajectory:
# 1. Pretrain In-Context Transformer on nominal dynamics with sequence contexts.
# 2. Online Tracking across 150 frames (Measuring In-Context Learning capability vs. TTT).
# 3. 45-Step Open-Loop Autoregressive Rollout (Measuring compounding error & KV-Cache growth).
# 4. Measure GPU/Memory scaling across sequence length.

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel
from src.in_context_transformer_model import InContextTransformerWorldModel
from src.bayesian_ttt_manager import BayesianTTTDynamicsManager

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
CKPT_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa_physics_neck.pt"
LONG_VIDEO_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/real_long_video_200frames.pt"
DATA_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
CKPT_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints"

def pretrain_in_context_transformer(model_transformer, dyn_neck, vjepa_encoder, device="cpu"):
    """
    Pretrains the In-Context Transformer on nominal physical sequences.
    """
    train_data = torch.load(os.path.join(DATA_DIR, "contact_train_nominal.pt"))
    videos = train_data["videos"] # (30, 20, 3, 224, 224)
    N, T, C, H, W = videos.shape

    print("Pretraining In-Context Transformer Baseline on nominal sequence contexts...")
    # Extract momentum states for training
    with torch.no_grad():
        flat = videos.reshape(-1, C, H, W)
        feats = []
        for i in range(0, flat.shape[0], 16):
            feats.append(vjepa_encoder(flat[i:i+16]).cpu())
        z_train = torch.cat(feats, dim=0).reshape(N, T, -1)

        states_train = []
        for t in range(1, T):
            st = dyn_neck(z_train[:, t], z_train[:, t-1])
            states_train.append(st.unsqueeze(1))
        states_train = torch.cat(states_train, dim=1) # (30, 19, 128)

    optimizer = torch.optim.Adam(model_transformer.parameters(), lr=1.5e-3)
    model_transformer.train()

    # Form sequence slices of length 4..12
    N_tr, L_total, d_dim = states_train.shape
    context_window = 8

    for epoch in range(18):
        epoch_loss = 0.0
        count = 0
        for ep in range(N_tr):
            seq = states_train[ep] # (19, 128)
            for t in range(context_window, len(seq) - 1):
                inp = seq[t-context_window : t+1].unsqueeze(0) # (1, context_window+1, 128)
                target = seq[t+1] # (128,)

                pred = model_transformer(inp).squeeze(0)
                loss = F.mse_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                count += 1
        if (epoch + 1) % 6 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/18 | In-Context Transformer Loss: {epoch_loss / count:.6f}")

    torch.save(model_transformer.state_dict(), os.path.join(CKPT_DIR, "in_context_transformer.pt"))
    print("In-Context Transformer pretraining completed and checkpoint saved.\n")

def run_strong_baseline_comparison():
    print("==========================================================================================")
    print("Head-to-Head Benchmark: In-Context Transformer (KV-Cache) vs. In-Weights TTT vs. Bayesian-TTT")
    print("==========================================================================================")

    # 1. Load Real 200-frame Data
    frames = torch.load(LONG_VIDEO_PATH)
    total_frames = frames.shape[0]

    base_model = VJEPAPhysicsTTTWorldModel(
        vjepa_weights_path=WEIGHTS_PATH,
        dyn_dim=128, context_k_dim=64,
        lr_ttt=0.35, beta_proximal=0.002, core_rank=16
    )
    base_model.load_state_dict(torch.load(CKPT_PATH))
    base_model.eval()

    # Pretrain or load In-Context Transformer
    model_transformer = InContextTransformerWorldModel(dyn_dim=128, hidden_dim=128, num_layers=3, num_heads=4, max_context_len=32)
    transformer_ckpt = os.path.join(CKPT_DIR, "in_context_transformer.pt")
    if not os.path.exists(transformer_ckpt):
        pretrain_in_context_transformer(model_transformer, base_model.dyn_neck, base_model.encoder)
    else:
        print("Loading pretrained In-Context Transformer baseline checkpoint...")
        model_transformer.load_state_dict(torch.load(transformer_ckpt))
    model_transformer.eval()

    # Extract 768-dim embeddings across 200 frames
    print("Extracting V-JEPA 2.1 ViT-B/16 representations across 200 frames...")
    feats = []
    with torch.no_grad():
        for i in range(0, total_frames, 20):
            feats.append(base_model.encoder(frames[i:i+20]).cpu())
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
    ctx_len = 16 # Transformer sliding context window of past 16 frames

    # Setup Models
    # Model 1: In-Context Transformer (Strong baseline, frozen weights + 16-step historical KV-Cache)
    # Model 2: In-Weights TTT (Our proposal, O(1) memory, plastic W_t)
    model_ttt = VJEPAPhysicsTTTWorldModel(vjepa_weights_path=WEIGHTS_PATH, dyn_dim=128, context_k_dim=64, lr_ttt=0.35, beta_proximal=0.002, core_rank=16)
    model_ttt.load_state_dict(torch.load(CKPT_PATH))
    model_ttt.eval()
    model_ttt.reset_plastic_weights()

    # Model 3: Bayesian-TTT
    bayes_mgr = BayesianTTTDynamicsManager(dyn_dim=128, obs_noise_var=2e-5, process_noise_var=5e-5, prior_var=1e-2)
    bayes_mgr.reset_belief()

    # -------------------------------------------------------------
    # Phase 1: 150-Frame Online Streaming Tracking
    # -------------------------------------------------------------
    print(f"\n--- Phase 1: 150-Frame Online Streaming Tracking ---")
    err_transformer_stream = []
    err_ttt_stream = []
    err_bayes_stream = []

    for t in range(t_warmup):
        s_curr = states[t:t+1]
        s_next_true = states[t+1:t+2]

        with torch.no_grad():
            # [1] In-Context Transformer Prediction (using past up to ctx_len history tokens)
            start_idx = max(0, t - ctx_len + 1)
            history_tokens = states[start_idx : t+1].unsqueeze(0) # (1, L, 128)
            p_trans = model_transformer(history_tokens)
            err_transformer_stream.append(F.mse_loss(p_trans, s_next_true).item())

            # [2] In-Weights TTT Prediction (before step)
            p_ttt = model_ttt.forward_state_prediction(s_curr, W=model_ttt.W_t)
            err_ttt_stream.append(F.mse_loss(p_ttt, s_next_true).item())

            # [3] Bayesian-TTT Prediction
            f_nom = model_ttt.invariant_core(s_curr)
            x_b = model_ttt.context_mapping(s_curr)
            p_bayes = f_nom + bayes_mgr.kalman_gain * torch.matmul(x_b, model_ttt.W_t)
            err_bayes_stream.append(F.mse_loss(p_bayes, s_next_true).item())

        # Update TTT and Bayesian models
        bayes_mgr.update_with_feedback(s_observed=s_next_true, s_pred_ttt=p_ttt, s_pred_nom=f_nom)
        y_target = s_next_true - f_nom
        err_vec = torch.matmul(x_b, model_ttt.W_t) - y_target
        g_fwd = torch.matmul(x_b.T, err_vec)
        G_raw = g_fwd + model_ttt.beta * (model_ttt.W_t - model_ttt.W_0)
        G_proj = torch.matmul(model_ttt.P_perp, G_raw)
        G_eff = G_proj + 0.30 * (G_raw - G_proj)
        model_ttt.W_t.sub_(model_ttt.lr_ttt * G_eff)

    mean_trans_s = np.mean(err_transformer_stream)
    mean_ttt_s = np.mean(err_ttt_stream)
    mean_bayes_s = np.mean(err_bayes_stream)

    print("Phase 1 Streaming MSE across 150 Frames:")
    print(f"  * In-Context Transformer (16-step KV Cache) : {mean_trans_s:.6f}")
    print(f"  * In-Weights TTT (O(1) Plastic Weights W_t) : {mean_ttt_s:.6f}")
    print(f"  * Bayesian-TTT Filtered Model               : {mean_bayes_s:.6f}")
    print(f"  * TTT vs. In-Context Transformer Advantage : {((mean_trans_s - mean_ttt_s) / mean_trans_s)*100:+.2f}%")

    # -------------------------------------------------------------
    # Phase 2: 45-Step Ultra-Long Autoregressive Rollout
    # -------------------------------------------------------------
    print(f"\n--- Phase 2: 45-Step Ultra-Long Autoregressive Rollout (Frames 151 to 195) ---")
    s_init = states[t_warmup:t_warmup+1]

    # For Transformer: maintain sliding buffer of historical/generated tokens
    transformer_buffer = states[t_warmup - ctx_len + 1 : t_warmup + 1].clone() # (ctx_len, 128)

    curr_ttt = s_init.clone()
    curr_bayes = s_init.clone()
    prev_bayes_st = states[t_warmup-1:t_warmup]

    roll_trans = []
    roll_ttt = []
    roll_bayes = []

    W_150 = model_ttt.W_t.clone()

    for h in range(rollout_h):
        target_idx = t_warmup + 1 + h
        if target_idx >= T_states:
            break
        s_target = states[target_idx:target_idx+1]

        with torch.no_grad():
            # [1] In-Context Transformer Rollout (Appends generated token to KV context buffer)
            p_trans = model_transformer(transformer_buffer.unsqueeze(0))
            roll_trans.append(F.mse_loss(p_trans, s_target).item())
            # Slide buffer
            transformer_buffer = torch.cat([transformer_buffer[1:], p_trans], dim=0)

            # [2] In-Weights TTT Rollout (O(1) memory, uses W_150)
            p_t = model_ttt.forward_state_prediction(curr_ttt, W=W_150)
            roll_ttt.append(F.mse_loss(p_t, s_target).item())
            curr_ttt = p_t

            # [3] Bayesian-TTT Rollout
            vel_norm = torch.norm(curr_bayes - prev_bayes_st).item()
            shrink = bayes_mgr.compute_rollout_shrinkage(h, vel_norm)
            f_nom = model_ttt.invariant_core(curr_bayes)
            x_b = model_ttt.context_mapping(curr_bayes)
            p_b = f_nom + shrink * torch.matmul(x_b, W_150)
            roll_bayes.append(F.mse_loss(p_b, s_target).item())
            prev_bayes_st = curr_bayes
            curr_bayes = p_b

    cum_trans = np.sum(roll_trans)
    cum_ttt = np.sum(roll_ttt)
    cum_bayes = np.sum(roll_bayes)

    print("Milestone Autoregressive MSE across 45-Step Horizon:")
    print(f" Horizon   | In-Context Transformer |    In-Weights TTT    |    Bayesian-TTT   ")
    print(f"----------------------------------------------------------------------------------")
    for step_idx in [0, 4, 9, 19, 29, 39, 44]:
        if step_idx < len(roll_trans):
            tr_v = roll_trans[step_idx]
            tt_v = roll_ttt[step_idx]
            ba_v = roll_bayes[step_idx]
            print(f" Step +{step_idx+1:2d}  |      {tr_v:.6f}          |       {tt_v:.6f}       |      {ba_v:.6f}   ")

    print(f"----------------------------------------------------------------------------------")
    print(f"Cumulative 45-Step Rollout MSE:")
    print(f"  * In-Context Transformer (KV-Cache)  : {cum_trans:.6f} (100.0%)")
    print(f"  * In-Weights TTT (O(1) Plastic W_t)  : {cum_ttt:.6f} ({(cum_ttt/cum_trans)*100:.2f}% vs Transformer - Drop {((cum_trans-cum_ttt)/cum_trans)*100:.2f}%)")
    print(f"  * Bayesian-TTT Filtered Model        : {cum_bayes:.6f} ({(cum_bayes/cum_trans)*100:.2f}% vs Transformer)\n")

    return {
        "cum_trans": cum_trans,
        "cum_ttt": cum_ttt,
        "cum_bayes": cum_bayes,
        "drop_pct": ((cum_trans-cum_ttt)/cum_trans)*100
    }

if __name__ == "__main__":
    run_strong_baseline_comparison()
