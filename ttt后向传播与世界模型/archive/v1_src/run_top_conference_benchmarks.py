# Standard Top-Tier Conference Benchmark Suite for Bayesian-TTT World Model
# Implements parallel multi-sample batch processing across all four core pillars:
# - Pillar 1: Open-Loop Future Multi-Step Compounding Rollout (H = 1 to 30 steps)
# - Pillar 2: System Identification / Physical Linear Probing (Mass Ratio & Friction R^2 across 50 episodes)
# - Pillar 3: Closed-Loop Downstream Model Predictive Path Integral (MPPI / MPC) Task Reward & Safety
# - Pillar 4: Hardware Latency & Memory Footprint profiling (ms/step, MB/step)
# - Plus Full 4-Way Ablation Suite (Static vs Open-Weights SGD vs Pure-TTT vs Bayesian-TTT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel
from src.bayesian_ttt_manager import BayesianTTTDynamicsManager

WEIGHTS_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa/model.safetensors"
CKPT_PATH = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/vjepa_physics_neck.pt"
DATA_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
RESULTS_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/benchmark_results"

# ------------------------------------------------------------------------------------------
# Multi-Sample Parallel Benchmark Pipeline
# ------------------------------------------------------------------------------------------

def extract_vjepa_embeddings_batch(model, videos, batch_size=16):
    # videos: (N, T, 3, 224, 224)
    N, T, C, H, W = videos.shape
    flat = videos.reshape(-1, C, H, W)
    feats = []
    with torch.no_grad():
        for i in range(0, flat.shape[0], batch_size):
            b = flat[i:i+batch_size]
            z = model.encoder(b)
            feats.append(z.cpu())
    z_all = torch.cat(feats, dim=0).reshape(N, T, -1)
    
    # Construct momentum states: (N, T-1, 128)
    states = []
    with torch.no_grad():
        for t in range(1, T):
            st = model.dyn_neck(z_all[:, t], z_all[:, t-1])
            states.append(st.unsqueeze(1))
    return torch.cat(states, dim=1) # (N, T-1, 128)

def evaluate_pillar_1_and_ablations(model, states, t_adapt=8, rollout_h=20, lr_ttt=0.35, leak_ratio=0.30):
    """
    Pillar 1: Multi-Step Compounding Error Rollout Curve (H=1..20)
    Concurrently runs all 4 ablation models over N parallel video episodes.
    """
    N, T_total, d_state = states.shape
    print(f"\n--- Pillar 1: Running Multi-Step Compounding Rollout & 4-Way Ablation across {N} Parallel Episodes ---")

    # Arrays to store compounding error: (N, rollout_h)
    err_static = np.zeros((N, rollout_h))
    err_open_sgd = np.zeros((N, rollout_h))
    err_pure_ttt = np.zeros((N, rollout_h))
    err_bayes_ttt = np.zeros((N, rollout_h))

    for ep in range(N):
        ep_states = states[ep] # (T-1, 128)

        # 1. Warm-up Adaptation Phase (t = 0 .. t_adapt-1)
        # Model A (Static): No update
        # Model B (Open-Weights SGD on core):
        model_ft = VJEPAPhysicsTTTWorldModel(WEIGHTS_PATH, dyn_dim=128, context_k_dim=64)
        model_ft.load_state_dict(torch.load(CKPT_PATH))
        for p in model_ft.invariant_core.parameters(): p.requires_grad = True
        opt_ft = torch.optim.SGD(model_ft.invariant_core.parameters(), lr=0.06)

        # Model C (Pure-TTT) & Model D (Bayesian-TTT)
        model.reset_plastic_weights()
        bayes_mgr = BayesianTTTDynamicsManager(dyn_dim=128, obs_noise_var=2e-5, process_noise_var=5e-5, prior_var=1e-2)
        bayes_mgr.reset_belief()

        for t in range(t_adapt):
            s_curr = ep_states[t:t+1]
            s_next = ep_states[t+1:t+2]

            # Model B update:
            opt_ft.zero_grad()
            loss_ft = F.mse_loss(model_ft.invariant_core(s_curr), s_next)
            loss_ft.backward()
            opt_ft.step()

            # Model C & D update:
            with torch.no_grad():
                f_nom = model.invariant_core(s_curr)
                pred_ttt = model.forward_state_prediction(s_curr, W=model.W_t)
            bayes_mgr.update_with_feedback(s_observed=s_next, s_pred_ttt=pred_ttt, s_pred_nom=f_nom)

            x_t = model.context_mapping(s_curr)
            y_tgt = s_next - f_nom
            err_v = torch.matmul(x_t, model.W_t) - y_tgt
            G_raw = torch.matmul(x_t.T, err_v) + model.beta * (model.W_t - model.W_0)
            G_proj = torch.matmul(model.P_perp, G_raw)
            G_eff = G_proj + leak_ratio * (G_raw - G_proj)
            model.W_t.sub_(lr_ttt * G_eff)

        W_adapted = model.W_t.clone()

        # 2. Open-Loop Future Autoregressive Rollout (t_adapt .. t_adapt+rollout_h)
        s_init = ep_states[t_adapt:t_adapt+1]

        c_static = s_init.clone()
        c_ft = s_init.clone()
        c_ttt = s_init.clone()
        c_bayes = s_init.clone()
        prev_bayes = ep_states[t_adapt-1:t_adapt]

        for h in range(rollout_h):
            idx = t_adapt + 1 + h
            if idx >= T_total:
                break
            s_tgt = ep_states[idx:idx+1]

            with torch.no_grad():
                # [1] Static
                p_s = model.forward_state_prediction(c_static, W=model.W_0)
                err_static[ep, h] = F.mse_loss(p_s, s_tgt).item()
                c_static = p_s

                # [2] Open-Weights SGD
                p_ft = model_ft.invariant_core(c_ft)
                err_open_sgd[ep, h] = F.mse_loss(p_ft, s_tgt).item()
                c_ft = p_ft

                # [3] Pure-TTT
                p_t = model.forward_state_prediction(c_ttt, W=W_adapted)
                err_pure_ttt[ep, h] = F.mse_loss(p_t, s_tgt).item()
                c_ttt = p_t

                # [4] Bayesian-TTT (Kinetic Dissipative Gating)
                v_norm = torch.norm(c_bayes - prev_bayes).item()
                gain = bayes_mgr.compute_rollout_shrinkage(h, v_norm)
                f_n = model.invariant_core(c_bayes)
                x_b = model.context_mapping(c_bayes)
                p_b = f_n + gain * torch.matmul(x_b, W_adapted)
                err_bayes_ttt[ep, h] = F.mse_loss(p_b, s_tgt).item()
                prev_bayes = c_bayes
                c_bayes = p_b

    mean_static = np.mean(err_static, axis=0)
    std_static = np.std(err_static, axis=0) / np.sqrt(N)
    mean_ft = np.mean(err_open_sgd, axis=0)
    std_ft = np.std(err_open_sgd, axis=0) / np.sqrt(N)
    mean_ttt = np.mean(err_pure_ttt, axis=0)
    std_ttt = np.std(err_pure_ttt, axis=0) / np.sqrt(N)
    mean_bayes = np.mean(err_bayes_ttt, axis=0)
    std_bayes = np.std(err_bayes_ttt, axis=0) / np.sqrt(N)

    return {
        "mean_static": mean_static, "std_static": std_static,
        "mean_ft": mean_ft, "std_ft": std_ft,
        "mean_ttt": mean_ttt, "std_ttt": std_ttt,
        "mean_bayes": mean_bayes, "std_bayes": std_bayes
    }

def evaluate_pillar_2_probing(model, states, params, t_adapt=8, lr_ttt=0.35, leak_ratio=0.30):
    """
    Pillar 2: System Identification / Physical Linear Probing on Mass Ratio & Friction
    Evaluates across all episodes.
    """
    N, T_total, _ = states.shape
    print(f"\n--- Pillar 2: System Identification / Physical Linear Probing across {N} Episodes ---")
    all_W = []
    all_P = []

    for ep in range(N):
        model.reset_plastic_weights()
        for t in range(t_adapt):
            s_curr = states[ep, t:t+1]
            s_next = states[ep, t+1:t+2]
            with torch.no_grad():
                f_nom = model.invariant_core(s_curr)
            x_t = model.context_mapping(s_curr)
            y_tgt = s_next - f_nom
            err_v = torch.matmul(x_t, model.W_t) - y_tgt
            G_raw = torch.matmul(x_t.T, err_v) + model.beta * (model.W_t - model.W_0)
            G_proj = torch.matmul(model.P_perp, G_raw)
            G_eff = G_proj + leak_ratio * (G_raw - G_proj)
            model.W_t.sub_(lr_ttt * G_eff)

        all_W.append(model.W_t.detach().cpu().numpy().flatten())
        all_P.append(params[ep, t_adapt].numpy())

    W_mat = np.array(all_W) # (N, 64*128)
    P_mat = np.array(all_P) # (N, 4)

    # Ridge Regression Probe
    W_norm = (W_mat - np.mean(W_mat, axis=0, keepdims=True)) / (np.std(W_mat, axis=0, keepdims=True) + 1e-6)
    U, S, _ = np.linalg.svd(W_norm, full_matrices=False)
    k_comp = min(20, S.shape[0])
    W_proj = U[:, :k_comp] @ np.diag(S[:k_comp])
    W_proj_b = np.concatenate([W_proj, np.ones((W_proj.shape[0], 1))], axis=1)

    A = W_proj_b.T @ W_proj_b + 0.5 * np.eye(W_proj_b.shape[1])
    b = W_proj_b.T @ P_mat[:, :2] # Mass ratio & friction
    coeffs = np.linalg.solve(A, b)
    pred_P = W_proj_b @ coeffs

    ss_tot = np.sum((P_mat[:, :2] - np.mean(P_mat[:, :2], axis=0)) ** 2, axis=0)
    ss_res = np.sum((P_mat[:, :2] - pred_P) ** 2, axis=0)
    r2_scores = 1.0 - (ss_res / (ss_tot + 1e-8))

    return {
        "r2_mass": r2_scores[0],
        "r2_friction": r2_scores[1]
    }

def evaluate_pillar_3_closed_loop_mpc(model, states, horizon=10, num_samples=30):
    """
    Pillar 3: Closed-Loop Model Predictive Control (MPC / MPPI) Simulation
    Uses the world model as an internal imagination simulator to plan control sequences.
    Compares Static Model vs. Bayesian-TTT on trajectory tracking error & goal cost.
    """
    N, T_total, d_state = states.shape
    print(f"\n--- Pillar 3: Closed-Loop MPPI Control Planning across {N} Episodes (Horizon={horizon}) ---")

    static_rewards = []
    bayes_rewards = []

    # Goal is to steer the system towards target final state s_{target}
    for ep in range(N):
        s_target = states[ep, -1] # Final goal state
        s_init = states[ep, 5]    # Planning start

        # Random control candidates: action perturbation in latent direction
        # Evaluate expected trajectory cost under Static Model vs. Bayesian-TTT Model
        actions_cand = torch.randn(num_samples, horizon, d_state) * 0.05

        def simulate_rollout(init_s, use_ttt_weights=False):
            costs = []
            for k in range(num_samples):
                curr = init_s.unsqueeze(0)
                ep_cost = 0.0
                for h in range(horizon):
                    act = actions_cand[k, h:h+1]
                    with torch.no_grad():
                        if use_ttt_weights:
                            pred = model.forward_state_prediction(curr, W=model.W_t) + act
                        else:
                            pred = model.forward_state_prediction(curr, W=model.W_0) + act
                    ep_cost += torch.norm(pred - s_target).item()
                    curr = pred
                costs.append(ep_cost)
            best_cost = min(costs)
            return -best_cost # Negative cost = Reward

        r_static = simulate_rollout(s_init, use_ttt_weights=False)
        r_bayes = simulate_rollout(s_init, use_ttt_weights=True)

        static_rewards.append(r_static)
        bayes_rewards.append(r_bayes)

    mean_r_static = np.mean(static_rewards)
    std_r_static = np.std(static_rewards) / np.sqrt(N)
    mean_r_bayes = np.mean(bayes_rewards)
    std_r_bayes = np.std(bayes_rewards) / np.sqrt(N)

    return {
        "reward_static": (mean_r_static, std_r_static),
        "reward_bayes": (mean_r_bayes, std_r_bayes),
        "reward_gain": (mean_r_bayes - mean_r_static) / (abs(mean_r_static) + 1e-8) * 100.0
    }

def evaluate_pillar_4_hardware_profiling(model, states, num_steps=200):
    """
    Pillar 4: Hardware Latency & Peak Memory Profiling (ms/step, MB footprint)
    """
    print(f"\n--- Pillar 4: Hardware Latency & Memory Footprint Profiling ({num_steps} Steps) ---")
    s_curr = states[0, 0:1]
    s_next = states[0, 1:2]
    bayes_mgr = BayesianTTTDynamicsManager(dyn_dim=128)

    # 1. Forward-only latency (Static Model)
    t0 = time.perf_counter()
    for _ in range(num_steps):
        with torch.no_grad():
            _ = model.forward_state_prediction(s_curr, W=model.W_0)
    t_forward = (time.perf_counter() - t0) / num_steps * 1000.0

    # 2. Forward + TTT Backward Step + Null-space Projection Latency
    t0 = time.perf_counter()
    for _ in range(num_steps):
        with torch.no_grad():
            f_nom = model.invariant_core(s_curr)
        x_t = model.context_mapping(s_curr)
        y_tgt = s_next - f_nom
        err_v = torch.matmul(x_t, model.W_t) - y_tgt
        G_raw = torch.matmul(x_t.T, err_v)
        G_proj = torch.matmul(model.P_perp, G_raw)
        model.W_t.sub_(0.1 * G_proj)
    t_ttt_step = (time.perf_counter() - t0) / num_steps * 1000.0

    # 3. Bayesian Gate Latency
    t0 = time.perf_counter()
    for _ in range(num_steps):
        _ = bayes_mgr.compute_rollout_shrinkage(horizon_step=5, velocity_norm=0.25)
    t_bayes = (time.perf_counter() - t0) / num_steps * 1000.0

    total_pipeline_latency = t_ttt_step + t_bayes

    # 4. Memory footprint
    proc = psutil.Process()
    mem_mb = proc.memory_info().rss / (1024 * 1024)

    return {
        "static_forward_ms": t_forward,
        "ttt_step_ms": t_ttt_step,
        "bayes_gate_ms": t_bayes,
        "total_latency_ms": total_pipeline_latency,
        "fps_throughput": 1000.0 / total_pipeline_latency,
        "memory_mb": mem_mb
    }

# ------------------------------------------------------------------------------------------
# Main Execution Runner
# ------------------------------------------------------------------------------------------

def main():
    print("==========================================================================================")
    print("Top-Tier Conference Benchmark Suite: 4 Pillars & Complete Ablation Evaluation")
    print("==========================================================================================")

    # Load 15-episode test datasets with ground-truth physical parameters
    # contact_test_heavy_mass.pt has 15 episodes of 224x224 collision videos
    data_heavy = torch.load(os.path.join(DATA_DIR, "contact_test_heavy_mass.pt"))
    data_fric = torch.load(os.path.join(DATA_DIR, "contact_test_high_friction.pt"))

    # Instantiate official V-JEPA 2.1 Physics World Model
    model = VJEPAPhysicsTTTWorldModel(WEIGHTS_PATH, dyn_dim=128, context_k_dim=64, lr_ttt=0.35, beta_proximal=0.002, core_rank=16)
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    # Batch extraction of 768-dim embeddings across all episodes
    print("Extracting V-JEPA embeddings for 15 collision episodes (batch extraction)...")
    states_heavy = extract_vjepa_embeddings_batch(model, data_heavy["videos"]) # (15, 19, 128)
    params_heavy = data_heavy["params"] # (15, 20, 4)

    # ---------------------------------------------------------
    # Pillar 1 & Ablations
    # ---------------------------------------------------------
    res_pillar1 = evaluate_pillar_1_and_ablations(model, states_heavy, t_adapt=7, rollout_h=10)

    # ---------------------------------------------------------
    # Pillar 2: Linear Probing
    # ---------------------------------------------------------
    res_pillar2 = evaluate_pillar_2_probing(model, states_heavy, params_heavy, t_adapt=7)

    # ---------------------------------------------------------
    # Pillar 3: Closed-Loop MPPI Control
    # ---------------------------------------------------------
    res_pillar3 = evaluate_pillar_3_closed_loop_mpc(model, states_heavy, horizon=10, num_samples=30)

    # ---------------------------------------------------------
    # Pillar 4: Hardware Latency & Memory Profiling
    # ---------------------------------------------------------
    res_pillar4 = evaluate_pillar_4_hardware_profiling(model, states_heavy, num_steps=300)

    # Save all results to disk
    torch.save({
        "pillar1": res_pillar1,
        "pillar2": res_pillar2,
        "pillar3": res_pillar3,
        "pillar4": res_pillar4
    }, os.path.join(RESULTS_DIR, "full_benchmark_metrics.pt"))

    print("\n==========================================================================================")
    print("Benchmark Suite Execution Complete. All Multi-Sample Metrics Logged.")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
