# Epistemic-Uncertainty-Aware Bayesian-TTT World Model with Explicit Control Interface
# Models action-conditioned transition:
#   \hat{s}_{t+1} = f_\theta(s_t, a_t) + W_t^\top \psi(s_t, a_t)
# Features:
# 1. Explicit Action Conditioning (a_t \in R^{d_a})
# 2. Recursive Epistemic Covariance Tracking (P_t on feature space \psi(s_t, a_t))
# 3. Closed-Form Predictive Epistemic Variance for Planning:
#      \sigma^2(s, a) = \psi(s, a)^\top P_t \psi(s, a) + R
# 4. Strict Constant O(1) Memory Footprint (No Replay Buffer needed)
# 5. Null-Space Projection P_\perp onto variant parameter manifold

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActionConditionedInvariantCore(nn.Module):
    """
    Nominal Backbone f_\theta(s_t, a_t): (s_t, a_t) -> \hat{s}_{t+1}
    Pretrained on nominal dynamics and frozen during test-time.
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # s: (B, state_dim), a: (B, action_dim)
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)

class ActionConditionedBayesianTTTWorldModel(nn.Module):
    """
    Action-Conditioned Bayesian In-Weights World Model.
    Maintains:
    - Weight posterior mean W_t \in R^{d_k \times state_dim}
    - Precision / Inverse Covariance matrix P_t \in R^{d_k \times d_k}
    """
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 1,
        context_k_dim: int = 16,
        lr_ttt: float = 0.15,
        beta_proximal: float = 0.005,
        obs_noise_var: float = 1e-4,
        core_rank: int = 4
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_k = context_k_dim
        self.d_v = state_dim
        self.lr_ttt = lr_ttt
        self.beta = beta_proximal
        self.R = obs_noise_var
        self.core_rank = core_rank

        # 1. Invariant Nominal Core (Frozen at test-time)
        self.invariant_core = ActionConditionedInvariantCore(state_dim, action_dim)

        # 2. Context feature mapping \psi(s_t, a_t) \in R^{d_k}
        self.context_mapping = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.d_k)
        )

        # 3. In-Weights buffers
        self.register_buffer("W_0", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("W_t", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("P_perp", torch.eye(self.d_k))
        self.register_buffer("U_core", torch.zeros(self.d_k, core_rank))

        # 4. Online Covariance buffer for epistemic uncertainty P_t
        self.register_buffer("P_t", torch.eye(self.d_k))

    def reset_plastic_weights(self):
        self.W_t.copy_(self.W_0)
        self.P_t.copy_(torch.eye(self.d_k))

    def compute_nullspace_projection(self, context_data: torch.Tensor):
        # context_data: (N, d_k)
        cov = torch.matmul(context_data.T, context_data) / context_data.shape[0]
        U, S, V = torch.linalg.svd(cov)
        U_core = U[:, :self.core_rank]
        P_perp = torch.eye(self.d_k, device=cov.device) - torch.matmul(U_core, U_core.T)
        self.U_core.copy_(U_core)
        self.P_perp.copy_(P_perp)

    def forward_transition(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        W: torch.Tensor = None
    ) -> torch.Tensor:
        if W is None:
            W = self.W_t
        sa = torch.cat([s_t, a_t], dim=-1)
        s_nom = self.invariant_core(s_t, a_t)
        x_ctx = self.context_mapping(sa)
        s_res = torch.matmul(x_ctx, W)
        return s_nom + s_res

    def compute_epistemic_uncertainty(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Computes closed-form epistemic predictive variance for MPPI planning:
        \sigma^2(s, a) = x^\top P_t x + R
        Returns: uncertainty standard deviation \sigma(s, a) \in R^{B}
        """
        sa = torch.cat([s, a], dim=-1)
        x = self.context_mapping(sa) # (B, d_k)
        # Batch quadratic form: diag(x P_t x^\top)
        xP = torch.matmul(x, self.P_t) # (B, d_k)
        var = torch.sum(xP * x, dim=-1) + self.R # (B,)
        return torch.sqrt(torch.clamp(var, min=1e-6))

    def ttt_bayesian_update_step(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next_true: torch.Tensor
    ) -> dict:
        """
        Online test-time training step with recursive covariance tracking:
        1. Forward residual error e_t
        2. Projection onto variant parameter manifold \tilde{G}_t = P_\perp G_t
        3. Parameter update W_{t+1} with proximal pull-back
        4. Recursive covariance update P_{t+1} (Riccati rank-1 update)
        """
        s_t = s_t.detach()
        a_t = a_t.detach()
        s_next_obs = s_next_true.detach()

        sa = torch.cat([s_t, a_t], dim=-1)
        x_t = self.context_mapping(sa) # (1, d_k)
        f_nom = self.invariant_core(s_t, a_t) # (1, d_v)
        y_target = s_next_obs - f_nom # (1, d_v)

        # 1. Forward residual error
        err = torch.matmul(x_t, self.W_t) - y_target # (1, d_v)
        g_fwd = torch.matmul(x_t.T, err) # (d_k, d_v)
        g_prox = self.beta * (self.W_t - self.W_0) # (d_k, d_v)

        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(self.P_perp, G_raw)

        # 2. Update plastic parameters W_t
        self.W_t.sub_(self.lr_ttt * G_proj)

        # 3. Update recursive epistemic covariance P_t (RLS / Kalman formulation)
        # P_{t+1} = (1 / gamma) * P_t - K_x * x * P_t
        x_sq = x_t.squeeze(0)
        Px = torch.matmul(self.P_t, x_sq)
        denom = 1.0 + torch.dot(x_sq, Px)
        k_gain = Px / denom
        self.P_t.sub_(torch.outer(k_gain, Px))
        # Condition number stabilization
        self.P_t = torch.clamp(0.5 * (self.P_t + self.P_t.T), -2.0, 2.0)

        uncertainty_step = self.compute_epistemic_uncertainty(s_t, a_t).item()

        return {
            "loss_forward": 0.5 * torch.sum(err ** 2).item(),
            "weight_norm": torch.norm(self.W_t - self.W_0).item(),
            "epistemic_uncertainty": uncertainty_step
        }
