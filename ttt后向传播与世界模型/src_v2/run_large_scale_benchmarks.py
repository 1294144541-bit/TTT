# -*- coding: utf-8 -*-
"""
Large-Scale Multi-Threaded Benchmark Suite v2:
Bayesian-TTT vs. JEPA-TTT (arXiv:2610.00722) vs. Static Baseline
Across N=100 Independent Episodes & 3 Dynamics Shift Regimes.
Uses concurrent.futures.ThreadPoolExecutor for reliable parallelism on macOS.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src_v2.actuated_env import NonStationaryActuatedPhysicsEnv
from src_v2.bayesian_ttt_action_model import ActionConditionedBayesianTTTWorldModel
from src_v2.jepa_ttt_reproduction import JEPATTTWorldModelBaseline
from src_v2.robust_mppi_planner import RobustMPPIPlanner

OUT_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/results_v2"

def run_single_episode(task_args):
    seed, shift_type = task_args
    env = NonStationaryActuatedPhysicsEnv(dt=0.05)
    
    # 1. Models (isolated instances per thread)
    model_b = ActionConditionedBayesianTTTWorldModel(state_dim=4, action_dim=1, context_k_dim=16)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/bayesian_action_model.pt"))
    model_b.eval()
    model_b.reset_plastic_weights()

    model_j = JEPATTTWorldModelBaseline(state_dim=4, action_dim=1, hidden_dim=64)
    model_j.load_state_dict(torch.load(f"{OUT_DIR}/jepa_action_model.pt"))
    model_j.eval()

    # 2. Planners
    planner_static = RobustMPPIPlanner(horizon=10, num_samples=30, uncertainty_lambda=0.0)
    planner_jepa = RobustMPPIPlanner(horizon=10, num_samples=30, uncertainty_lambda=0.0)
    planner_bayes = RobustMPPIPlanner(horizon=10, num_samples=30, uncertainty_lambda=0.15)

    s_s = env.reset(seed=seed)
    s_j = s_s.copy()
    s_b = s_s.copy()

    rew_s, rew_j, rew_b = 0.0, 0.0, 0.0
    ep_len = 50

    for t in range(ep_len):
        # Shift schedules
        if shift_type == "mass":
            mp_val = 0.2 * (3.5 if t >= ep_len // 2 else 1.0)
            mu_val = 0.15
        elif shift_type == "ice":
            mp_val = 0.2
            mu_val = 0.15 * (0.2 if t >= ep_len // 2 else 1.0)
        else: # Coupled non-stationary
            mp_val = 0.2 * (1.0 + 1.2 * np.sin(2.0 * np.pi * t / 25.0))
            mu_val = 0.15 * (0.3 if t >= ep_len // 2 else 1.0)
        params = np.array([mp_val, mu_val, 9.81], dtype=np.float32)

        # 1. Static
        t_s = torch.tensor(s_s, dtype=torch.float32).unsqueeze(0)
        a_s = planner_static.plan(t_s, transition_fn=lambda st, at: model_b.forward_transition(st, at, W=model_b.W_0))
        s_s, r_s, _, _ = env.step(a_s, params=params)
        rew_s += r_s

        # 2. JEPA-TTT
        t_j = torch.tensor(s_j, dtype=torch.float32).unsqueeze(0)
        a_j = planner_jepa.plan(t_j, transition_fn=lambda st, at: model_j.forward(st, at))
        next_s_j, r_j, _, _ = env.step(a_j, params=params)
        rew_j += r_j
        model_j.update_from_transition(t_j, torch.tensor([[a_j]], dtype=torch.float32), torch.tensor(next_s_j, dtype=torch.float32).unsqueeze(0), replay_steps=2)
        s_j = next_s_j

        # 3. Bayesian-TTT
        t_b = torch.tensor(s_b, dtype=torch.float32).unsqueeze(0)
        a_b = planner_bayes.plan(
            t_b,
            transition_fn=lambda st, at: model_b.forward_transition(st, at, W=model_b.W_t),
            uncertainty_fn=lambda st, at: model_b.compute_epistemic_uncertainty(st, at)
        )
        next_s_b, r_b, _, _ = env.step(a_b, params=params)
        rew_b += r_b
        model_b.ttt_bayesian_update_step(t_b, torch.tensor([[a_b]], dtype=torch.float32), torch.tensor(next_s_b, dtype=torch.float32).unsqueeze(0))
        s_b = next_s_b

    return (rew_s, rew_j, rew_b, len(model_j.replay_buffer))

def main():
    total_episodes = 100
    shifts = ["mass", "ice", "coupled"]
    print(f"Executing {total_episodes} threaded evaluations across 3 dynamics shift environments...")
    
    seeds_and_shifts = [(3000 + i * 37, shifts[i % len(shifts)]) for i in range(total_episodes)]
    
    # Run with ThreadPoolExecutor (8 concurrent worker threads)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_single_episode, seeds_and_shifts))

    r_s = [r[0] for r in results]
    r_j = [r[1] for r in results]
    r_b = [r[2] for r in results]
    jepa_buf_max = max(r[3] for r in results)

    # Statistical significance testing
    from scipy import stats
    diff_bj = np.array(r_b) - np.array(r_j)
    t_stat_bj, p_val_bj = stats.ttest_1samp(diff_bj, 0.0)

    diff_bs = np.array(r_b) - np.array(r_s)
    t_stat_bs, p_val_bs = stats.ttest_1samp(diff_bs, 0.0)

    print("\n==========================================================================================")
    print(f"LARGE-SCALE BENCHMARK RESULTS (Across N={total_episodes} Independent Episodes)")
    print("==========================================================================================")
    print(f"Static Model Reward  : {np.mean(r_s):.2f} +- {np.std(r_s)/np.sqrt(total_episodes):.2f}")
    print(f"JEPA-TTT Reward      : {np.mean(r_j):.2f} +- {np.std(r_j)/np.sqrt(total_episodes):.2f}")
    print(f"Bayesian-TTT Reward  : {np.mean(r_b):.2f} +- {np.std(r_b)/np.sqrt(total_episodes):.2f}")
    print(f"Paired t-test (Bayes vs. JEPA-TTT): Delta = {np.mean(diff_bj):+.2f} (t={t_stat_bj:+.3f}, p={p_val_bj:.4e})")
    print(f"Paired t-test (Bayes vs. Static)  : Delta = {np.mean(diff_bs):+.2f} (t={t_stat_bs:+.3f}, p={p_val_bs:.4e})")
    print(f"Memory Footprint Status:")
    print(f"  * JEPA-TTT Max Replay Buffer   : {jepa_buf_max} transitions/episode (linear growth)")
    print(f"  * Bayesian-TTT Parameter Memory: 320 elements (Strict Constant O(1))")

    torch.save({
        "r_s": r_s, "r_j": r_j, "r_b": r_b,
        "t_stat_bj": t_stat_bj, "p_val_bj": p_val_bj,
        "t_stat_bs": t_stat_bs, "p_val_bs": p_val_bs
    }, f"{OUT_DIR}/large_scale_100ep_results.pt")
    print(f"Results saved to {OUT_DIR}/large_scale_100ep_results.pt")

if __name__ == "__main__":
    main()
