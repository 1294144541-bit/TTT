# Bayesian Filtered TTT World Model (Bayesian-TTT)
# Solves the "Over-Correction Trap" identified in long-horizon rollouts:
# 1. Maintains an Online Gaussian Belief over the Plastic Weight Manifold:
#      p(W_t) = N(mu_t, Sigma_t)
# 2. Predictive Rollout Uncertainty Calibration:
#      Uncertainty grows with autoregressive rollout steps H -> Gated Kalman Weighting
# 3. Dynamic Residual Shrinkage / Dissipative Gating:
#      g(s_t, W_t) = K_t * (W_t^\top x_t), where Kalman Gain K_t balances
#      prior nominal physics confidence against empirical TTT drift evidence.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_physics_ttt import VJEPAPhysicsTTTWorldModel

class BayesianTTTDynamicsManager:
    """
    Bayesian Filtering Layer operating on top of TTT Plastic Weights.
    Tracks state innovation covariance and updates Bayesian Kalman gain K_t
    to dynamically balance Nominal Physics Prior vs. TTT Empirical Adaptation.
    """
    def __init__(
        self,
        dyn_dim: int = 128,
        obs_noise_var: float = 1e-4,   # R: Observation noise covariance
        process_noise_var: float = 1e-5, # Q: Dynamics drift covariance
        prior_var: float = 1e-3        # P_0: Prior variance of adaptation parameter
    ):
        self.dyn_dim = dyn_dim
        self.R = obs_noise_var
        self.Q = process_noise_var
        self.P_t = prior_var # Scalar isotropic variance representation for efficiency
        self.kalman_gain = 0.5 # Initial balanced trust

    def reset_belief(self, prior_var: float = 1e-3):
        self.P_t = prior_var
        self.kalman_gain = 0.5

    def update_with_feedback(self, s_observed: torch.Tensor, s_pred_ttt: torch.Tensor, s_pred_nom: torch.Tensor):
        """
        Bayesian measurement update step when true feedback s_{t+1} is observed.
        Computes Innovation Residual:
          y_innov = s_{observed} - s_{pred_ttt}
        Computes Innovation Variance:
          S_t = P_t + R
        Computes Optimal Kalman Gain:
          K_t = P_t / (P_t + R)
        """
        # Innovation magnitude
        innov_err = torch.mean((s_observed - s_pred_ttt) ** 2).item()
        nom_err = torch.mean((s_observed - s_pred_nom) ** 2).item()

        # Prior time-update: P_t|t-1 = P_{t-1} + Q
        P_pred = self.P_t + self.Q

        # Measurement update: K_t = P_pred / (P_pred + R)
        S_t = P_pred + self.R
        K_t = P_pred / S_t
        self.kalman_gain = float(np.clip(K_t, 0.05, 0.95))

        # Posterior covariance: P_t = (1 - K_t) * P_pred
        self.P_t = (1.0 - self.kalman_gain) * P_pred

        # Likelihood ratio: If TTT error is significantly smaller than Nominal error, increase trust in TTT
        if innov_err < nom_err:
            confidence_boost = min(1.5, nom_err / (innov_err + 1e-8))
            self.kalman_gain = float(np.clip(self.kalman_gain * confidence_boost, 0.1, 0.95))
        else:
            confidence_decay = max(0.5, nom_err / (innov_err + 1e-8))
            self.kalman_gain = float(np.clip(self.kalman_gain * confidence_decay, 0.05, 0.8))

        return {
            "kalman_gain": self.kalman_gain,
            "innov_err": innov_err,
            "nom_err": nom_err,
            "posterior_var": self.P_t
        }

    def compute_rollout_shrinkage(self, horizon_step: int, velocity_norm: float) -> float:
        """
        During open-loop autoregressive rollout:
        Uncertainty P grows with horizon h: P_h = P_0 + h * Q
        Kalman Gain decays as empirical evidence goes stale without real-time feedback:
          alpha_h = 1.0 / (1.0 + gamma * h)
        Physical Velocity Gating:
          v_gate = tanh(velocity_norm * 10.0) -> shrinks to 0 as object stops
        """
        uncertainty_decay = 1.0 / (1.0 + 0.15 * horizon_step)
        velocity_gate = float(np.tanh(velocity_norm * 5.0))
        total_weight = self.kalman_gain * uncertainty_decay * velocity_gate
        return float(np.clip(total_weight, 0.0, 1.0))
