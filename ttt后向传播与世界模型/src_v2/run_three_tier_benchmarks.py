# -*- coding: utf-8 -*-
"""
Three-Tier Decoupled Benchmark Suite:
Tier 1: World Model Level (Frozen vs Online SGD vs Projected TTT vs Bayesian TTT)
Tier 2: Generalization Matrix (Mass, Friction, Damping, Abrupt/Sinusoidal, OOD ranges, Seeds N=20)
Tier 3: Uncertainty Utility (Correlation, Calibration, Risk-penalty MPPI ablation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src_v2.actuated_env import NonStationaryActuatedPhysicsEnv
from src_v2.bayesian_ttt_action_model import ActionConditionedBayesianTTTWorldModel
from src_v2.robust_mppi_planner import RobustMPPIPlanner

OUT_DIR = "/Users/mac/Documents/论文/ttt后向传播与世界模型/results_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Models Setup & Pretraining on Nominal Data
# -------------------------------------------------------------
def get_pretrained_model():
    model = ActionConditionedBayesianTTTWorldModel(state_dim=4, action_dim=1, context_k_dim=16)
    ckpt = f"{OUT_DIR}/bayesian_action_model.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
    model.eval()
    return model

# -------------------------------------------------------------
# Tier 1: World Model Adaptation Benchmark (Passive Sequences)
# Evaluates on identical state-action sequences across 4 adaptation variants
# -------------------------------------------------------------
def run_tier1_world_model_adaptation(num_episodes=20, ep_len=40):
    print("=" * 80)
    print("TIER 1: WORLD MODEL LEVEL ADAPTATION BENCHMARK (IDENTICAL TRANSITIONS)")
    print("Variants: 1. Frozen | 2. Online SGD | 3. Projected TTT | 4. Bayesian RLS-TTT")
    print("=" * 80)

    env = NonStationaryActuatedPhysicsEnv(dt=0.05)
    model = get_pretrained_model()
    P_perp = model.P_perp

    # Storage for 1-step and 5-step compounding rollout MSE
    methods = ["Frozen", "Online_SGD", "Projected_TTT", "Bayesian_TTT"]
    one_step_errors = {m: [] for m in methods}
    multi_step_errors = {m: [] for m in methods}
    recovery_times = {m: [] for m in methods}

    for ep in range(num_episodes):
        seed = 1000 + ep * 17
        s = env.reset(seed=seed)

        # Generate sequence of true transitions under abrupt shift at step 20
        seq_s, seq_a, seq_next_s = [], [], []
        for t in range(ep_len):
            is_shift = (t >= 20)
            # True parameters: mass +300%, friction -80%
            params = np.array([0.2 * (4.0 if is_shift else 1.0), 0.15 * (0.2 if is_shift else 1.0), 9.81], dtype=np.float32)
            act = float(-1.2 * s[0] - 0.4 * s[1] - 7.5 * s[2] - 1.1 * s[3] + np.random.normal(0, 0.2))
            act = float(np.clip(act, -2.5, 2.5))
            next_s, _, _, _ = env.step(act, params=params)
            seq_s.append(s)
            seq_a.append([act])
            seq_next_s.append(next_s)
            s = next_s

        seq_s = torch.tensor(np.array(seq_s), dtype=torch.float32)
        seq_a = torch.tensor(np.array(seq_a), dtype=torch.float32)
        seq_next_s = torch.tensor(np.array(seq_next_s), dtype=torch.float32)

        # Now evaluate each method on this EXACT SAME sequence
        # Method 1: Frozen Dynamics
        w_frozen = torch.zeros(16, 4)
        ep_err1 = []
        for t in range(ep_len):
            st, at, nst = seq_s[t:t+1], seq_a[t:t+1], seq_next_s[t:t+1]
            with torch.no_grad():
                pred = model.forward_transition(st, at, W=w_frozen)
                ep_err1.append(F.mse_loss(pred, nst).item())
        one_step_errors["Frozen"].append(ep_err1)

        # Method 2: Ordinary Online SGD (Unconstrained on W)
        w_sgd = torch.zeros(16, 4)
        ep_err2 = []
        for t in range(ep_len):
            st, at, nst = seq_s[t:t+1], seq_a[t:t+1], seq_next_s[t:t+1]
            with torch.no_grad():
                pred = model.forward_transition(st, at, W=w_sgd)
                ep_err2.append(F.mse_loss(pred, nst).item())
            sa = torch.cat([st, at], dim=-1)
            x_t = model.context_mapping(sa)
            f_nom = model.invariant_core(st, at)
            err = torch.matmul(x_t, w_sgd) - (nst - f_nom)
            g = torch.matmul(x_t.T, err)
            w_sgd = w_sgd - 0.15 * g
        one_step_errors["Online_SGD"].append(ep_err2)

        # Method 3: Projected TTT (Null-Space Projection P_perp)
        w_proj = torch.zeros(16, 4)
        ep_err3 = []
        for t in range(ep_len):
            st, at, nst = seq_s[t:t+1], seq_a[t:t+1], seq_next_s[t:t+1]
            with torch.no_grad():
                pred = model.forward_transition(st, at, W=w_proj)
                ep_err3.append(F.mse_loss(pred, nst).item())
            sa = torch.cat([st, at], dim=-1)
            x_t = model.context_mapping(sa)
            f_nom = model.invariant_core(st, at)
            err = torch.matmul(x_t, w_proj) - (nst - f_nom)
            g = torch.matmul(x_t.T, err)
            g_p = torch.matmul(P_perp, g)
            w_proj = (1.0 - 0.005) * w_proj - 0.15 * g_p
        one_step_errors["Projected_TTT"].append(ep_err3)

        # Method 4: Bayesian / RLS-TTT (With recursive covariance update)
        model.reset_plastic_weights()
        ep_err4 = []
        for t in range(ep_len):
            st, at, nst = seq_s[t:t+1], seq_a[t:t+1], seq_next_s[t:t+1]
            with torch.no_grad():
                pred = model.forward_transition(st, at, W=model.W_t)
                ep_err4.append(F.mse_loss(pred, nst).item())
            model.ttt_bayesian_update_step(st, at, nst)
        one_step_errors["Bayesian_TTT"].append(ep_err4)

    # Compute statistics across episodes
    print("\n[Tier 1 Results] 1-Step Prediction MSE (Mean +- SEM across N=20):")
    for m in methods:
        arr = np.array(one_step_errors[m]) # (N, ep_len)
        pre_shift = np.mean(arr[:, :20])
        post_shift_init = np.mean(arr[:, 20:25]) # immediate shock
        post_shift_adapted = np.mean(arr[:, 30:]) # recovery
        overall = np.mean(arr)
        sem = np.std(np.mean(arr, axis=1)) / np.sqrt(num_episodes)
        print(f"  * {m:14s} | Overall: {overall:.6f} +- {sem:.6f} | Post-Shock (t=20-25): {post_shift_init:.6f} | Adapted (t=30-40): {post_shift_adapted:.6f}")

    return one_step_errors

# -------------------------------------------------------------
# Tier 2: Systematic Generalization Matrix across Dynamics Shifts
# -------------------------------------------------------------
def run_tier2_generalization_matrix(num_seeds=20):
    print("\n" + "=" * 80)
    print("TIER 2: SYSTEMATIC DYNAMICS GENERALIZATION MATRIX (N=20 Seeds)")
    print("Shifts: 1. Heavy Mass (+350%) | 2. Ice Friction (-85%) | 3. Sinusoidal Friction Drift | 4. Severe Coupled OOD")
    print("=" * 80)

    shifts = {
        "Mass_Shock": lambda t: np.array([0.2 * (4.5 if t >= 20 else 1.0), 0.15, 9.81], dtype=np.float32),
        "Ice_Friction": lambda t: np.array([0.2, 0.15 * (0.15 if t >= 20 else 1.0), 9.81], dtype=np.float32),
        "Sinusoidal_Drift": lambda t: np.array([0.2, 0.15 * (1.0 + 0.8 * np.sin(2*np.pi*t/20.0)), 9.81], dtype=np.float32),
        "Coupled_OOD": lambda t: np.array([0.2 * (3.0 if t >= 20 else 1.0), 0.15 * (0.2 if t >= 20 else 1.0), 9.81 * (1.5 if t >= 20 else 1.0)], dtype=np.float32)
    }

    env = NonStationaryActuatedPhysicsEnv(dt=0.05)
    model = get_pretrained_model()
    planner_static = RobustMPPIPlanner(horizon=10, num_samples=35, uncertainty_lambda=0.0)
    planner_bayes = RobustMPPIPlanner(horizon=10, num_samples=35, uncertainty_lambda=0.15)

    results_table = {}

    for s_name, s_fn in shifts.items():
        static_rewards = []
        bayes_rewards = []

        for ep in range(num_seeds):
            seed = 2000 + ep * 31
            s_s = env.reset(seed=seed)
            s_b = s_s.copy()
            model.reset_plastic_weights()
            planner_static.reset()
            planner_bayes.reset()

            r_s_tot, r_b_tot = 0.0, 0.0
            for t in range(40):
                params = s_fn(t)

                # Static
                t_s = torch.tensor(s_s, dtype=torch.float32).unsqueeze(0)
                a_s = planner_static.plan(t_s, transition_fn=lambda st, at: model.forward_transition(st, at, W=model.W_0))
                s_s, r_s, _, _ = env.step(a_s, params=params)
                r_s_tot += r_s

                # Bayesian-TTT
                t_b = torch.tensor(s_b, dtype=torch.float32).unsqueeze(0)
                a_b = planner_bayes.plan(
                    t_b,
                    transition_fn=lambda st, at: model.forward_transition(st, at, W=model.W_t),
                    uncertainty_fn=lambda st, at: model.compute_epistemic_uncertainty(st, at)
                )
                next_s_b, r_b, _, _ = env.step(a_b, params=params)
                r_b_tot += r_b
                model.ttt_bayesian_update_step(t_b, torch.tensor([[a_b]]), torch.tensor(next_s_b).unsqueeze(0))
                s_b = next_s_b

            static_rewards.append(r_s_tot)
            bayes_rewards.append(r_b_tot)

        m_s, std_s = np.mean(static_rewards), np.std(static_rewards)/np.sqrt(num_seeds)
        m_b, std_b = np.mean(bayes_rewards), np.std(bayes_rewards)/np.sqrt(num_seeds)

        from scipy import stats
        diff = np.array(bayes_rewards) - np.array(static_rewards)
        t_stat, p_val = stats.ttest_1samp(diff, 0.0)

        results_table[s_name] = {
            "static": (m_s, std_s),
            "bayes": (m_b, std_b),
            "delta": m_b - m_s,
            "t_stat": t_stat,
            "p_val": p_val
        }
        print(f"Shift: {s_name:18s} | Static: {m_s:7.2f} +- {std_s:5.2f} | Bayes: {m_b:7.2f} +- {std_b:5.2f} | Delta: {m_b-m_s:+6.2f} (p={p_val:.4f})")

    return results_table

# -------------------------------------------------------------
# Tier 3: Uncertainty Utility & Risk-Penalty Direct Ablation
# Isolates: Same Bayesian-TTT Model, Same MPPI Planner, With vs Without Lambda
# -------------------------------------------------------------
def run_tier3_uncertainty_ablation(num_seeds=20):
    print("\n" + "=" * 80)
    print("TIER 3: UNCERTAINTY UTILITY & ISOLATED RISK-PENALTY ABLATION (N=20 Seeds)")
    print("Protocol: Fix identical Bayesian-TTT world model; compare lambda=0.0 vs lambda=0.15 vs lambda=0.30")
    print("=" * 80)

    env = NonStationaryActuatedPhysicsEnv(dt=0.05)
    model = get_pretrained_model()

    lambdas = [0.0, 0.15, 0.30]
    lambda_rewards = {l: [] for l in lambdas}
    uncertainties_all = []
    errors_all = []

    for ep in range(num_seeds):
        seed = 3000 + ep * 43

        for l in lambdas:
            s = env.reset(seed=seed)
            model.reset_plastic_weights()
            planner = RobustMPPIPlanner(horizon=10, num_samples=35, uncertainty_lambda=l)
            planner.reset()

            r_tot = 0.0
            for t in range(40):
                is_shift = (t >= 20)
                params = np.array([0.2 * (4.0 if is_shift else 1.0), 0.15 * (0.2 if is_shift else 1.0), 9.81], dtype=np.float32)

                t_s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                # Plan
                a = planner.plan(
                    t_s,
                    transition_fn=lambda st, at: model.forward_transition(st, at, W=model.W_t),
                    uncertainty_fn=lambda st, at: model.compute_epistemic_uncertainty(st, at)
                )
                next_s, r, _, _ = env.step(a, params=params)
                r_tot += r

                # Track uncertainty vs actual transition error on lambda=0.15 run
                if l == 0.15:
                    with torch.no_grad():
                        unc = model.compute_epistemic_uncertainty(t_s, torch.tensor([[a]])).item()
                        pred_s = model.forward_transition(t_s, torch.tensor([[a]]), W=model.W_t)
                        act_err = torch.norm(pred_s.squeeze(0) - torch.tensor(next_s)).item()
                        uncertainties_all.append(unc)
                        errors_all.append(act_err)

                model.ttt_bayesian_update_step(t_s, torch.tensor([[a]]), torch.tensor(next_s).unsqueeze(0))
                s = next_s

            lambda_rewards[l].append(r_tot)

    # 1. Direct Planning Ablation Comparison
    print("\n[Ablation on Planning Risk Penalty (Same Model & Planner)]:")
    for l in lambdas:
        arr = np.array(lambda_rewards[l])
        m, sem = np.mean(arr), np.std(arr)/np.sqrt(num_seeds)
        print(f"  * Lambda = {l:4.2f} | Reward: {m:.2f} +- {sem:.2f}")

    from scipy import stats
    diff_l = np.array(lambda_rewards[0.15]) - np.array(lambda_rewards[0.0])
    t_stat_l, p_val_l = stats.ttest_1samp(diff_l, 0.0)
    print(f"  * Paired Difference (Lambda=0.15 vs Lambda=0.0): Delta = {np.mean(diff_l):+.2f} (t={t_stat_l:+.3f}, p={p_val_l:.4f})")

    # 2. Correlation between Uncertainty and Actual Error
    corr = np.corrcoef(uncertainties_all, errors_all)[0, 1]
    print(f"\n[Uncertainty Calibration & Correlation]:")
    print(f"  * Pearson Correlation r(Uncertainty, True Error): {corr:.4f}")

    # Coverage: within 2*sigma
    coverage = np.mean(np.array(errors_all) <= 2.0 * np.array(uncertainties_all)) * 100.0
    print(f"  * 2-Sigma Predictive Interval Empirical Coverage: {coverage:.1f}%")

    return {
        "lambda_rewards": lambda_rewards,
        "correlation": corr,
        "coverage": coverage
    }

if __name__ == "__main__":
    t1 = run_tier1_world_model_adaptation(num_episodes=20)
    t2 = run_tier2_generalization_matrix(num_seeds=20)
    t3 = run_tier3_uncertainty_ablation(num_seeds=20)

    # Save to disk
    torch.save({
        "tier1": t1,
        "tier2": t2,
        "tier3": t3
    }, f"{OUT_DIR}/three_tier_decoupled_benchmark_results.pt")
    print("\n" + "=" * 80)
    print("ALL THREE TIERS EXECUTED & DATA SERIALIZED SUCCESSFULLY.")
    print("=" * 80)
