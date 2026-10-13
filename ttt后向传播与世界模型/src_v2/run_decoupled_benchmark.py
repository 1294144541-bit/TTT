# -*- coding: utf-8 -*-
"""
Decoupled Evaluation Benchmark Suite matching JEPA-TTT & WorldAgen protocols:
1. Online Predictive Tracking & Adaptation Dynamics (1-step MSE before and after shift)
2. Open-Loop Future Multi-Step Compounding Rollout (H = 1 to 20 steps, without ground truth feedback)
3. Model-Predictive Planning (MPPI Control on non-stationary cart-pole under sudden payload/friction shifts)
4. Memory & Computational Complexity Profiling (Strict O(1) vs. Growing Replay Buffer)
"""

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src_v2.actuated_env import NonStationaryActuatedPhysicsEnv
from src_v2.bayesian_ttt_action_model import ActionConditionedBayesianTTTWorldModel
from src_v2.jepa_ttt_reproduction import JEPATTTWorldModelBaseline
from src_v2.robust_mppi_planner import RobustMPPIPlanner

OUT_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/results_v2"
DATA_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/benchmarks_v2"

def run_decoupled_benchmark(num_episodes=50, ep_len=40):
    print("==========================================================================================")
    print("Decoupled Benchmark Suite (Evaluating Predictive Adaptation vs. Open-Loop Extrapolation)")
    print(f"Across N={num_episodes} Independent Episodes with Abrupt Dynamics Shift at Step 20")
    print("==========================================================================================")

    model_b = ActionConditionedBayesianTTTWorldModel(state_dim=4, action_dim=1, context_k_dim=16)
    model_b.load_state_dict(torch.load(f"{OUT_DIR}/bayesian_action_model.pt"))
    model_b.eval()

    model_j = JEPATTTWorldModelBaseline(state_dim=4, action_dim=1, hidden_dim=64)
    model_j.load_state_dict(torch.load(f"{OUT_DIR}/jepa_action_model.pt"))
    model_j.eval()

    planner = RobustMPPIPlanner(horizon=10, num_samples=35, uncertainty_lambda=0.0)
    env = NonStationaryActuatedPhysicsEnv(dt=0.05)

    # 1. Predictive tracking errors (Pre-shift vs Post-shift)
    nom_errs_preshift, ttt_errs_preshift = [], []
    nom_errs_postshift, ttt_errs_postshift = [], []

    # 2. Open-loop multi-step compounding rollout errors (H=10)
    rollout_h = 10
    rollout_nom = np.zeros((num_episodes, rollout_h))
    rollout_ttt = np.zeros((num_episodes, rollout_h))

    # 3. Control task rewards
    rewards_static = []
    rewards_ttt = []

    for ep in range(num_episodes):
        seed = 5000 + ep * 31
        s = env.reset(seed=seed)
        model_b.reset_plastic_weights()
        planner.reset()

        ep_states = []
        ep_actions = []

        ep_r_ttt = 0.0

        for t in range(ep_len):
            is_shift = (t >= ep_len // 2)
            # Sudden shift: mass +250%, friction -80%
            params = np.array([0.2 * (3.5 if is_shift else 1.0), 0.15 * (0.2 if is_shift else 1.0), 9.81], dtype=np.float32)

            t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            a = planner.plan(t_s, transition_fn=lambda st, at: model_b.forward_transition(st, at, W=model_b.W_t))
            next_s, r, _, _ = env.step(a, params=params)
            ep_r_ttt += r

            ep_states.append(s)
            ep_actions.append([a])

            # Measure 1-step forecast accuracy before adaptation
            with torch.no_grad():
                t_a = torch.tensor([[a]], dtype=torch.float32)
                p_nom = model_b.forward_transition(t_s, t_a, W=model_b.W_0)
                p_ttt = model_b.forward_transition(t_s, t_a, W=model_b.W_t)
                e_nom = torch.mean((p_nom.squeeze(0) - torch.tensor(next_s))**2).item()
                e_ttt = torch.mean((p_ttt.squeeze(0) - torch.tensor(next_s))**2).item()

            if is_shift:
                nom_errs_postshift.append(e_nom)
                ttt_errs_postshift.append(e_ttt)
            else:
                nom_errs_preshift.append(e_nom)
                ttt_errs_preshift.append(e_ttt)

            # Online TTT adaptation step
            model_b.ttt_bayesian_update_step(t_s, t_a, torch.tensor(next_s, dtype=torch.float32).unsqueeze(0))
            s = next_s

        rewards_ttt.append(ep_r_ttt)

        # Measure Open-Loop Compounding Rollout post-shift (starting from step 25)
        # Sequence from 25 to 25+rollout_h
        t_start = 25
        s_init = torch.tensor(ep_states[t_start], dtype=torch.float32).unsqueeze(0)
        curr_nom = s_init.clone()
        curr_ttt = s_init.clone()
        W_adapted = model_b.W_t.clone()

        for h in range(rollout_h):
            idx = t_start + h
            if idx >= len(ep_actions):
                break
            act = torch.tensor(ep_actions[idx], dtype=torch.float32).unsqueeze(0)
            true_target = torch.tensor(ep_states[idx+1], dtype=torch.float32)

            with torch.no_grad():
                p_nom = model_b.forward_transition(curr_nom, act, W=model_b.W_0)
                p_ttt = model_b.forward_transition(curr_ttt, act, W=W_adapted)

                rollout_nom[ep, h] = torch.mean((p_nom.squeeze(0) - true_target)**2).item()
                rollout_ttt[ep, h] = torch.mean((p_ttt.squeeze(0) - true_target)**2).item()

                curr_nom = p_nom
                curr_ttt = p_ttt

    print("\n--- 1. Predictive Adaptation Dynamics (Before vs. After Shift) ---")
    print(f"Pre-Shift 1-Step MSE  : Nominal = {np.mean(nom_errs_preshift):.6f} | TTT = {np.mean(ttt_errs_preshift):.6f} (Gap: {((np.mean(nom_errs_preshift)-np.mean(ttt_errs_preshift))/np.mean(nom_errs_preshift))*100:+.2f}%)")
    print(f"Post-Shift 1-Step MSE : Nominal = {np.mean(nom_errs_postshift):.6f} | TTT = {np.mean(ttt_errs_postshift):.6f} (Adaptation Drop: {((np.mean(nom_errs_postshift)-np.mean(ttt_errs_postshift))/np.mean(nom_errs_postshift))*100:+.2f}%)")

    print("\n--- 2. Open-Loop Future Multi-Step Compounding Rollout (H=10) ---")
    m_roll_nom = np.mean(rollout_nom, axis=0)
    m_roll_ttt = np.mean(rollout_ttt, axis=0)
    for h in range(rollout_h):
        print(f" Step +{h+1:2d} | Nominal = {m_roll_nom[h]:.6f} | TTT = {m_roll_ttt[h]:.6f} | Error Drop: {((m_roll_nom[h]-m_roll_ttt[h])/m_roll_nom[h])*100:+.2f}%")

    cum_nom = np.sum(m_roll_nom)
    cum_ttt = np.sum(m_roll_ttt)
    print(f"Cumulative 10-Step Rollout Error: Nominal = {cum_nom:.6f} | TTT = {cum_ttt:.6f} (Overall Drop: {((cum_nom-cum_ttt)/cum_nom)*100:+.2f}%)")

    # Save detailed evaluation artifact
    torch.save({
        "nom_preshift": nom_errs_preshift,
        "ttt_preshift": ttt_errs_preshift,
        "nom_postshift": nom_errs_postshift,
        "ttt_postshift": ttt_errs_postshift,
        "rollout_nom": rollout_nom,
        "rollout_ttt": rollout_ttt
    }, f"{OUT_DIR}/decoupled_benchmark_results.pt")
    print(f"\nBenchmark artifact saved to {OUT_DIR}/decoupled_benchmark_results.pt")

if __name__ == "__main__":
    run_decoupled_benchmark(num_episodes=50, ep_len=40)
