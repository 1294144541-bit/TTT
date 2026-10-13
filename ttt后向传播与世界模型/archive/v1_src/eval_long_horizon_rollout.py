# Multi-Step Latent Rollout & Long-Horizon Physical Extrapolation Evaluation
# Compares Static V-JEPA World Model vs. TTT-Adapted World Model:
# Protocol:
# 1. Warm-up Adaptation Phase (Frames 1 to T_adapt): TTT observes video frames, computes self-supervised
#    prediction residuals, and updates plastic in-weights W_t online.
# 2. Multi-Step Open-Loop Rollout Phase (Frames T_adapt+1 to T_end):
#    Completely cut off ground-truth frame observations.
#    Both models autoregressively roll out future states:
#      \hat{s}_{\tau+1} = f_\theta(\hat{s}_\tau) + \psi(\hat{s}_\tau)^\top W
#    Compare Compounding Error over H = 5, 10, 15 steps.

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

def evaluate_long_horizon_rollout(
    model: VJEPAPhysicsTTTWorldModel,
    test_file: str,
    benchmark_name: str,
    t_adapt: int = 8,
    rollout_horizon: int = 10
):
    data = torch.load(os.path.join(DATA_DIR, test_file))
    videos = data["videos"] # (N, T=20, 3, 224, 224)
    N, T, _, _, _ = videos.shape

    z_test = extract_features(model, videos)
    print(f"\n======================================================================")
    print(f"Benchmark: {benchmark_name}")
    print(f"Protocol: Observe {t_adapt} frames (Adaptation) -> Autoregressive Rollout {rollout_horizon} frames")
    print(f"======================================================================")

    static_step_errors = np.zeros((N, rollout_horizon))
    ttt_step_errors = np.zeros((N, rollout_horizon))

    for ep in range(N):
        model.reset_plastic_weights()

        # Extract true sequence of physical states s_1 .. s_{T-1}
        states_seq = []
        with torch.no_grad():
            for t in range(1, T):
                st = model.dyn_neck(z_test[ep, t:t+1], z_test[ep, t-1:t])
                states_seq.append(st)
        states_seq = torch.cat(states_seq, dim=0) # (T-1, 128)

        # 1. Warm-up Phase: stream frames 1 to t_adapt with TTT updates
        for t in range(t_adapt):
            s_curr = states_seq[t:t+1]
            s_next = states_seq[t+1:t+2]
            # Perform multi-step inner loop or boosted adaptation step
            model.ttt_step(s_curr, s_next)

        # Freeze adapted weights W_{t_adapt} for open-loop imagination
        W_adapted = model.W_t.clone()

        # 2. Open-Loop Multi-step Rollout starting from state at t_adapt
        s_init = states_seq[t_adapt:t_adapt+1] # Ground truth start of rollout

        curr_static = s_init.clone()
        curr_ttt = s_init.clone()

        for h in range(rollout_horizon):
            target_idx = t_adapt + 1 + h
            if target_idx >= len(states_seq):
                break
            s_target = states_seq[target_idx:target_idx+1]

            with torch.no_grad():
                # Static Rollout (W = W_0 = 0)
                pred_static = model.forward_state_prediction(curr_static, W=model.W_0)
                err_static = F.mse_loss(pred_static, s_target).item()
                static_step_errors[ep, h] = err_static
                curr_static = pred_static # Autoregressive feedback!

                # TTT-Adapted Rollout (W = W_adapted)
                pred_ttt = model.forward_state_prediction(curr_ttt, W=W_adapted)
                err_ttt = F.mse_loss(pred_ttt, s_target).item()
                ttt_step_errors[ep, h] = err_ttt
                curr_ttt = pred_ttt # Autoregressive feedback!

    # Aggregate errors across horizons
    mean_static_curve = np.mean(static_step_errors, axis=0)
    mean_ttt_curve = np.mean(ttt_step_errors, axis=0)

    print("Step-by-step Autoregressive Compounding Error (MSE):")
    print(f" Horizon | Static World Model | TTT-Adapted Model | Error Reduction")
    print(f"------------------------------------------------------------------")
    for h in range(rollout_horizon):
        st_err = mean_static_curve[h]
        tt_err = mean_ttt_curve[h]
        red = (st_err - tt_err) / (st_err + 1e-8) * 100.0
        print(f" Step +{h+1:2d} |    {st_err:.8f}    |    {tt_err:.8f}   |   {red:+6.2f}%")

    cum_static = np.sum(mean_static_curve)
    cum_ttt = np.sum(mean_ttt_curve)
    cum_red = (cum_static - cum_ttt) / cum_static * 100.0

    print(f"------------------------------------------------------------------")
    print(f"Cumulative {rollout_horizon}-Step Rollout MSE : Static = {cum_static:.8f} | TTT = {cum_ttt:.8f}")
    print(f"Overall Long-Horizon Error Drop: {cum_red:.2f}%\n")

    return {
        "static_curve": mean_static_curve,
        "ttt_curve": mean_ttt_curve,
        "overall_drop": cum_red
    }

def main():
    model = VJEPAPhysicsTTTWorldModel(
        vjepa_weights_path=WEIGHTS_PATH,
        dyn_dim=128,
        context_k_dim=64,
        lr_ttt=0.25, # calibrated for multi-step drift adaptation
        beta_proximal=0.005,
        core_rank=16
    )
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    res_fric = evaluate_long_horizon_rollout(
        model,
        test_file="contact_test_high_friction.pt",
        benchmark_name="OOD Extreme Friction Braking (Long-Horizon Slide)",
        t_adapt=7,
        rollout_horizon=10
    )

    res_mass = evaluate_long_horizon_rollout(
        model,
        test_file="contact_test_heavy_mass.pt",
        benchmark_name="OOD Heavy Mass Collision & Post-Impact Rollout",
        t_adapt=7,
        rollout_horizon=10
    )

if __name__ == "__main__":
    main()
