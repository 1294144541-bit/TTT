# Matrix-Valued Bayesian In-Weights TTT (Calibrated Natural-Gradient Engine)
# Solves the parameter explosion problem by incorporating:
# 1. Natural Gradient Damping & Step-Size Normalization (Trust-Region update)
# 2. Ornstein-Uhlenbeck Mean-Reversion Prior on W_t:
#      W_{t+1} = (1 - beta) W_t + beta * W_0 - eta * Delta_W_Kalman
# 3. Information Matrix Symmetrization and Condition Number Clamping

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CalibratedBayesianInWeightsTTT:
    """
    Stabilized Online Bayesian Filter for the In-Weights Parameter Manifold W_t \in R^{d_k \times d_v}.
    Combines Recursive Parameter Covariance Tracking (P_t) with Natural Gradient Normalization
    and Ornstein-Uhlenbeck Pull-back Regularization.
    """
    def __init__(
        self,
        d_k: int = 64,
        d_v: int = 128,
        prior_var: float = 1.0,
        obs_noise: float = 1e-2,
        forgetting_factor: float = 0.99,
        lr_bayes: float = 0.15,
        beta_ou: float = 0.005,
        max_step_norm: float = 0.20
    ):
        self.d_k = d_k
        self.d_v = d_v
        self.R = obs_noise
        self.gamma = forgetting_factor
        self.eta = lr_bayes
        self.beta = beta_ou
        self.max_norm = max_step_norm

        self.W_t = torch.zeros(d_k, d_v)
        self.W_0 = torch.zeros(d_k, d_v)
        self.P_t = (1.0 / prior_var) * torch.eye(d_k)

    def reset(self):
        self.W_t.zero_()
        self.P_t = torch.eye(self.d_k)

    def update_step(
        self,
        x_t: torch.Tensor,       # (1, d_k)
        y_target: torch.Tensor,  # (1, d_v)
        P_perp: torch.Tensor     # (d_k, d_k)
    ) -> dict:
        x = x_t.squeeze(0) # (d_k,)
        y = y_target.squeeze(0) # (d_v,)

        # 1. Innovation Error
        y_pred = torch.matmul(x, self.W_t)
        innov = y - y_pred # (d_v,)

        # 2. Natural Gradient Kalman Preconditioning
        # P_pred with forgetting
        P_pred = self.P_t / self.gamma
        Px = torch.matmul(P_pred, x)
        denom = self.R + torch.dot(x, Px)
        k_t = Px / (denom + 1e-6) # (d_k,)

        # 3. Outer product parameter gradient
        Delta_W_raw = torch.outer(k_t, innov) # (d_k, d_v)

        # 4. Step-size clipping / Trust-region normalization
        raw_norm = torch.norm(Delta_W_raw).item()
        if raw_norm > self.max_norm:
            Delta_W_raw = Delta_W_raw * (self.max_norm / (raw_norm + 1e-8))

        # 5. Null-space projection onto variant manifold
        Delta_W_proj = torch.matmul(P_perp, Delta_W_raw)

        # 6. Bayesian Parameter Update with OU mean-reversion pull-back
        # W_{t+1} = (1 - beta) W_t + beta * W_0 + eta * Delta_W_proj
        self.W_t = (1.0 - self.beta) * self.W_t + self.beta * self.W_0 + self.eta * Delta_W_proj

        # 7. Riccati update on Covariance
        self.P_t = P_pred - torch.outer(k_t, Px)
        # Symmetrize and regularize condition number
        self.P_t = 0.5 * (self.P_t + self.P_t.T)
        self.P_t = torch.clamp(self.P_t, -5.0, 5.0)

        return {
            "innov_mse": torch.mean(innov ** 2).item(),
            "W_norm": torch.norm(self.W_t).item(),
            "step_norm": torch.norm(Delta_W_proj).item(),
            "cov_trace": torch.trace(self.P_t).item()
        }
