# Full Benchmark: Static vs. Pure-TTT vs. Bayesian-TTT Long-Horizon Rollout
# Tests whether Bayesian Filtering resolves the Over-Correction trap in open-loop extrapolation.

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel
from src.bayesian_ttt_manager import BayesianTTTDynamicsManager

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

def run_bayesian_ttt_benchmark(
    test_file: str,
    benchmark_name: str,
    t_adapt: int = 7,
    rollout_horizon: int = 10,
    lr_ttt: float = 0.35,
    leak_ratio: float = 0.30
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

    bayes_manager = BayesianTTTDynamicsManager(
        dyn_dim=128,
        obs_noise_var=5e-5,
        process_noise_var=1e-5,
        prior_var=1e-3
    )

    data = torch.load(os.path.join(DATA_DIR, test_file))
    videos = data["videos"]
    N, T, _, _, _ = videos.shape

    z_test = extract_features(model, videos)
    print(f"\n==========================================================================================")
    print(f"Benchmark: {benchmark_name}")
    print(f"Protocol: Observe {t_adapt} frames (Bayesian-TTT Adaptation) -> Autoregressive Rollout {rollout_horizon} steps")
    print(f"Tri-Model Comparison: [1] Static Baseline  |  [2] Pure-TTT  |  [3] Bayesian-TTT")
    print(f"==========================================================================================")

    static_errors = np.zeros((N, rollout_horizon))
    pure_ttt_errors = np.zeros((N, rollout_horizon))
    bayes_ttt_errors = np.zeros((N, rollout_horizon))

    for ep in range(N):
        model.reset_plastic_weights()
        bayes_manager.reset_belief()

        states_seq = []
        with torch.no_grad():
            for t in range(1, T):
                st = model.dyn_neck(z_test[ep, t:t+1], z_test[ep, t-1:t])
                states_seq.append(st)
        states_seq = torch.cat(states_seq, dim=0) # (T-1, 128)

        # 1. Warm-up Phase: stream frames with TTT update + Bayesian Feedback Filter
        for t in range(t_adapt):
            s_curr = states_seq[t:t+1]
            s_next = states_seq[t+1:t+2]

            with torch.no_grad():
                f_nom = model.invariant_core(s_curr)
                pred_ttt_before = model.forward_state_prediction(s_curr, W=model.W_t)

            # Update Bayesian Filter with real-time feedback discrepancy
            bayes_manager.update_with_feedback(s_observed=s_next, s_pred_ttt=pred_ttt_before, s_pred_nom=f_nom)

            # TTT gradient step
            x_t = model.context_mapping(s_curr)
            y_target = s_next - f_nom
            err = torch.matmul(x_t, model.W_t) - y_target
            g_fwd = torch.matmul(x_t.T, err)
            g_prox = model.beta * (model.W_t - model.W_0)
            G_raw = g_fwd + g_prox
            G_proj = torch.matmul(model.P_perp, G_raw)
            G_core = G_raw - G_proj
            G_eff = G_proj + leak_ratio * G_core
            model.W_t.sub_(lr_ttt * G_eff)

        W_adapted = model.W_t.clone()

        # 2. Open-loop Autoregressive Imagination
        s_init = states_seq[t_adapt:t_adapt+1]
        curr_static = s_init.clone()
        curr_pure_ttt = s_init.clone()
        curr_bayes_ttt = s_init.clone()

        prev_bayes_state = states_seq[t_adapt-1:t_adapt] if t_adapt > 0 else s_init

        for h in range(rollout_horizon):
            target_idx = t_adapt + 1 + h
            if target_idx >= len(states_seq):
                break
            s_target = states_seq[target_idx:target_idx+1]

            with torch.no_grad():
                # [1] Static Model (W = 0)
                p_static = model.forward_state_prediction(curr_static, W=model.W_0)
                static_errors[ep, h] = F.mse_loss(p_static, s_target).item()
                curr_static = p_static

                # [2] Pure-TTT Model (Constant W_adapted)
                p_pure_ttt = model.forward_state_prediction(curr_pure_ttt, W=W_adapted)
                pure_ttt_errors[ep, h] = F.mse_loss(p_pure_ttt, s_target).item()
                curr_pure_ttt = p_pure_ttt

                # [3] Bayesian-TTT Model (Dynamic shrinkage + Velocity dissipation)
                # Compute estimated velocity norm
                vel_norm = torch.norm(curr_bayes_ttt - prev_bayes_state).item()
                shrinkage_gain = bayes_manager.compute_rollout_shrinkage(horizon_step=h, velocity_norm=vel_norm)

                # Bayesian modulated prediction: f_nom + shrinkage * (W^\top x)
                nom_part = model.invariant_core(curr_bayes_ttt)
                x_b = model.context_mapping(curr_bayes_ttt)
                res_part = torch.matmul(x_b, W_adapted)
                p_bayes = nom_part + shrinkage_gain * res_part

                bayes_ttt_errors[ep, h] = F.mse_loss(p_bayes, s_target).item()
                prev_bayes_state = curr_bayes_ttt
                curr_bayes_ttt = p_bayes

    m_static = np.mean(static_errors, axis=0)
    m_pure = np.mean(pure_ttt_errors, axis=0)
    m_bayes = np.mean(bayes_ttt_errors, axis=0)

    print("Step-by-step Autoregressive Compounding Error (MSE):")
    print(f" Horizon | Static Baseline |    Pure-TTT     |   Bayesian-TTT  | Bayes vs Static Gain")
    print(f"--------------------------------------------------------------------------------------")
    for h in range(rollout_horizon):
        st = m_static[h]
        pu = m_pure[h]
        ba = m_bayes[h]
        gain = (st - ba) / (st + 1e-8) * 100.0
        print(f" Step +{h+1:2d} |   {st:.8f}  |   {pu:.8f}  |   {ba:.8f}  |       {gain:+6.2f}%")

    cum_static = np.sum(m_static)
    cum_pure = np.sum(m_pure)
    cum_bayes = np.sum(m_bayes)

    drop_bayes_vs_static = (cum_static - cum_bayes) / cum_static * 100.0
    drop_bayes_vs_pure = (cum_pure - cum_bayes) / cum_pure * 100.0

    print(f"--------------------------------------------------------------------------------------")
    print(f"Cumulative {rollout_horizon}-Step Rollout MSE:")
    print(f"  * Static Baseline World Model : {cum_static:.8f}")
    print(f"  * Pure-TTT (Over-Correction)  : {cum_pure:.8f}")
    print(f"  * Bayesian-TTT (Filtered)     : {cum_bayes:.8f}")
    print(f"--------------------------------------------------------------------------------------")
    print(f"Bayesian-TTT vs. Static Baseline Gain : {drop_bayes_vs_static:+.2f}%")
    print(f"Bayesian-TTT vs. Pure-TTT Gain        : {drop_bayes_vs_pure:+.2f}%\n")

    return {
        "static": cum_static,
        "pure": cum_pure,
        "bayes": cum_bayes,
        "gain": drop_bayes_vs_static
    }

if __name__ == "__main__":
    res_fric = run_bayesian_ttt_benchmark(
        test_file="contact_test_high_friction.pt",
        benchmark_name="OOD Extreme Friction Braking (Long-Horizon Slide)",
        t_adapt=7,
        rollout_horizon=10,
        lr_ttt=0.45,
        leak_ratio=0.30
    )

    res_mass = run_bayesian_ttt_benchmark(
        test_file="contact_test_heavy_mass.pt",
        benchmark_name="OOD Heavy Mass Collision & Post-Impact Rollout",
        t_adapt=7,
        rollout_horizon=10,
        lr_ttt=0.45,
        leak_ratio=0.30
    )
