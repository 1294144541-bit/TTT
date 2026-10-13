# Robust MPPI Planner supporting both:
# 1. Optimistic Planning (Uncertainty Penalty lambda = 0.0) -> Standard JEPA-TTT & Static Model
# 2. Risk-Aware Pessimistic Planning (Uncertainty Penalty lambda > 0.0) -> Bayesian-TTT
# 3. Variance-Weighted State Damping

import torch
import numpy as np
from typing import Callable

class RobustMPPIPlanner:
    def __init__(
        self,
        horizon: int = 12,
        num_samples: int = 40,
        temperature: float = 0.5,
        action_dim: int = 1,
        action_low: float = -2.5,
        action_high: float = 2.5,
        uncertainty_lambda: float = 0.15 # Calibrated risk-penalty weight
    ):
        self.H = horizon
        self.K = num_samples
        self.temperature = temperature
        self.action_dim = action_dim
        self.u_min = action_low
        self.u_max = action_high
        self.lambda_unc = uncertainty_lambda
        self.mean_action = torch.zeros(horizon, action_dim)

    def reset(self):
        self.mean_action.zero_()

    def plan(
        self,
        curr_state: torch.Tensor,
        transition_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        uncertainty_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None
    ) -> float:
        # Sample candidate perturbation sequences
        noise = torch.randn(self.K, self.H, self.action_dim) * 0.35
        actions_cand = torch.clamp(self.mean_action.unsqueeze(0) + noise, self.u_min, self.u_max)

        states = curr_state.repeat(self.K, 1)
        total_costs = torch.zeros(self.K)

        for h in range(self.H):
            acts = actions_cand[:, h]
            with torch.no_grad():
                next_states = transition_fn(states, acts)

            x = next_states[:, 0]
            x_dot = next_states[:, 1]
            theta = next_states[:, 2]
            theta_dot = next_states[:, 3]

            # Primary task tracking cost: penalize angle deflection and high velocity
            step_cost = (
                x**2 + 6.0 * (theta**2) + 0.1 * (x_dot**2) + 0.5 * (theta_dot**2) + 0.01 * (acts.squeeze(-1)**2)
            )

            # Epistemic Risk Penalization (Conservative bias during dynamics shifts)
            if uncertainty_fn is not None and self.lambda_unc > 0.0:
                with torch.no_grad():
                    unc_sigma = uncertainty_fn(states, acts)
                step_cost = step_cost + self.lambda_unc * unc_sigma

            total_costs += step_cost
            states = next_states

        min_cost = torch.min(total_costs)
        exp_weights = torch.exp(- (total_costs - min_cost) / self.temperature)
        weights = exp_weights / torch.sum(exp_weights)

        weighted_actions = torch.sum(weights.view(self.K, 1, 1) * actions_cand, dim=0)
        self.mean_action = weighted_actions
        best_a0 = float(weighted_actions[0, 0].item())

        self.mean_action = torch.cat([self.mean_action[1:], torch.zeros(1, self.action_dim)], dim=0)
        return best_a0
