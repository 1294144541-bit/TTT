# Matrix-Valued Bayesian In-Weights TTT (Bayesian-Weight-TTT)
# Directly maintains an online Gaussian belief over the plastic weight matrix W_t:
#   p(vec(W_t) | D_t) = N(mu_t, Sigma_t)
# Computes Natural Gradient Kalman Gain K_t on parameter space, updating both
# posterior mean W_t and parameter precision Sigma_t^{-1} in O(1) memory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel

class MatrixBayesianTTTEngine:
    """
    Online Recursive Bayesian Estimation (Extended Kalman / Recursive Least Squares Filter)
    directly applied to the in-weights parameter manifold W_t \in R^{d_k \times d_v}.
    
    Observation Model:
      y_t = W_t^\top x_t + v_t,   v_t ~ N(0, R * I_{d_v})
    Parameter Dynamics (Random Walk with Drift):
      W_{t+1} = gamma * W_t + (1 - gamma) * W_0 + w_t,   w_t ~ N(0, Q * I)
      
    Sufficient Statistics:
      - Posterior Mean: W_t \in R^{d_k \times d_v}
      - Input Precision Matrix: P_t \in R^{d_k \times d_k} (Inverse Covariance)
    """
    def __init__(
        self,
        d_k: int = 64,
        d_v: int = 128,
        obs_noise_var: float = 1e-4,     # R
        prior_precision: float = 1.0,     # lambda_prior * I (Ridge/Bayesian Prior)
        forgetting_factor: float = 0.98   # Exponential memory decay (Kalman drift)
    ):
        self.d_k = d_k
        self.d_v = d_v
        self.R = obs_noise_var
        self.gamma = forgetting_factor
        self.prior_lambda = prior_precision

        # Precision matrix inverse P_t \in R^{d_k \times d_k} initialized to lambda^{-1} * I
        self.P_t = (1.0 / prior_precision) * torch.eye(d_k)
        self.W_t = torch.zeros(d_k, d_v)
        self.W_0 = torch.zeros(d_k, d_v)

    def reset_belief(self):
        self.P_t = (1.0 / self.prior_lambda) * torch.eye(self.d_k)
        self.W_t.zero_()

    def bayesian_weight_update_step(
        self,
        x_t: torch.Tensor,       # (1, d_k)
        y_target: torch.Tensor,  # (1, d_v)
        P_perp: torch.Tensor     # (d_k, d_k) Null-space projection
    ) -> dict:
        """
        Matrix-level Bayesian Measurement Update:
        1. Innovation Vector: e_t = y_target - x_t W_t \in R^{1 x d_v}
        2. Parameter Covariance Time Update: P_t|t-1 = (1 / gamma) * P_{t-1}
        3. Parameter Kalman Gain Vector: k_t = P_pred * x_t^\top / (R + x_t * P_pred * x_t^\top) \in R^{d_k x 1}
        4. Parameter Posterior Update: W_t = W_{t-1} + P_perp (k_t @ e_t)
        5. Riccati Covariance Measurement Update: P_t = (I - k_t x_t) P_pred
        """
        x = x_t.squeeze(0) # (d_k,)
        y = y_target.squeeze(0) # (d_v,)
        
        # 1. Prediction error
        y_pred = torch.matmul(x, self.W_t) # (d_v,)
        innov = y - y_pred # (d_v,)

        # 2. Time-update on parameter covariance
        P_pred = self.P_t / self.gamma

        # 3. Kalman Gain vector k_t \in R^{d_k}
        Px = torch.matmul(P_pred, x) # (d_k,)
        scalar_denom = self.R + torch.dot(x, Px)
        k_t = Px / scalar_denom # (d_k,)

        # 4. In-Weights Update with Null-Space Projection
        # Outer product: k_t @ innov \in R^{d_k x d_v}
        Delta_W_raw = torch.outer(k_t, innov) # (d_k, d_v)
        # Project onto variant parameter subspace
        Delta_W_proj = torch.matmul(P_perp, Delta_W_raw)

        # Update posterior mean parameter matrix
        self.W_t.add_(Delta_W_proj)

        # 5. Riccati Measurement Update on P_t
        # P_new = P_pred - k_t @ (x^\top P_pred) = P_pred - outer(k_t, Px)
        self.P_t = P_pred - torch.outer(k_t, Px)

        # Numerical symmetrization
        self.P_t = 0.5 * (self.P_t + self.P_t.T)

        return {
            "innov_mse": torch.mean(innov ** 2).item(),
            "kalman_gain_norm": torch.norm(k_t).item(),
            "param_trace_cov": torch.trace(self.P_t).item(),
            "W_norm": torch.norm(self.W_t).item()
        }

    def compute_predictive_variance(self, x_query: torch.Tensor) -> float:
        """
        Computes epistemic predictive uncertainty for a state during open-loop rollout:
        Var(\hat{s}) = x^\top P_t x + R
        """
        x = x_query.squeeze(0)
        epistemic_var = torch.dot(x, torch.matmul(self.P_t, x)).item()
        return epistemic_var + self.R
