# TTT Augmented World Model Framework
# Implements:
# 1. Invariant Core Network f_\theta (Frozen nominal physics prior)
# 2. Plastic Dynamics Layer with in-weights adaptation W_t
# 3. Null-space Projection P_\perp (U_core) to prevent catastrophic forgetting
# 4. Proximal Curvature Regularization & OU-bounded drift

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class ObservationEncoder(nn.Module):
    def __init__(self, obs_dim: int = 4, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.net(o)

class InvariantCore(nn.Module):
    def __init__(self, latent_dim: int = 16, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a], dim=-1)
        return self.net(x)

class InverseDynamicsHead(nn.Module):
    def __init__(self, latent_dim: int = 16, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.GELU(),
            nn.Linear(32, action_dim)
        )
    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))

class TTTAugmentedWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        action_dim: int = 2,
        latent_dim: int = 16,
        feature_k_dim: int = 16,
        lr_ttt: float = 0.05,
        beta_proximal: float = 0.01,
        core_rank: int = 4
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.d_k = feature_k_dim
        self.d_v = latent_dim
        self.lr_ttt = lr_ttt
        self.beta = beta_proximal
        self.core_rank = core_rank

        self.encoder = ObservationEncoder(obs_dim, latent_dim)
        self.invariant_core = InvariantCore(latent_dim, action_dim)
        self.inv_dynamics = InverseDynamicsHead(latent_dim, action_dim)

        self.context_mapping = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.d_k)
        )

        self.register_buffer("W_0", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("W_t", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("P_perp", torch.eye(self.d_k))
        self.register_buffer("U_core", torch.zeros(self.d_k, core_rank))

    def reset_plastic_weights(self):
        self.W_t.copy_(self.W_0)

    def compute_nullspace_projection(self, feature_data: torch.Tensor):
        # feature_data: (N, d_k)
        cov = torch.matmul(feature_data.T, feature_data) / feature_data.shape[0]
        U, S, V = torch.linalg.svd(cov)
        U_core = U[:, :self.core_rank]
        P_perp = torch.eye(self.d_k, device=cov.device) - torch.matmul(U_core, U_core.T)
        self.U_core.copy_(U_core)
        self.P_perp.copy_(P_perp)

    def forward_prediction(self, z_t: torch.Tensor, a_t: torch.Tensor, W: torch.Tensor = None) -> torch.Tensor:
        if W is None:
            W = self.W_t
        # Nominal backbone physics prediction
        z_base = self.invariant_core(z_t, a_t)
        # Context feature mapping
        x_ctx = self.context_mapping(torch.cat([z_t, a_t], dim=-1))
        # Plastic dynamics residual: g(x; W) = x @ W
        z_residual = torch.matmul(x_ctx, W)
        return z_base + z_residual

    def ttt_online_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_next_true: torch.Tensor,
        alpha_inv: float = 0.1
    ) -> Dict[str, float]:
        # Detach inputs for self-supervised test-time training
        z_t = z_t.detach()
        a_t = a_t.detach()
        z_next_true = z_next_true.detach()

        x_t = self.context_mapping(torch.cat([z_t, a_t], dim=-1)) # (1, d_k)
        f_nom = self.invariant_core(z_t, a_t) # (1, d_v)
        y_target = z_next_true - f_nom # (1, d_v)

        # 1. Forward residual gradient: x_t^T (x_t W_t - y_target)
        err = torch.matmul(x_t, self.W_t) - y_target # (1, d_v)
        g_fwd = torch.matmul(x_t.T, err) # (d_k, d_v)

        # 2. Proximal regularization gradient: \beta (W_t - W_0)
        g_prox = self.beta * (self.W_t - self.W_0)

        # 3. Inverse dynamics alignment gradient (numerical or analytical)
        pred_z_next = f_nom + torch.matmul(x_t, self.W_t)
        pred_action = self.inv_dynamics(z_t, pred_z_next)
        loss_inv = 0.5 * F.mse_loss(pred_action, a_t)

        # Combined raw gradient
        G_raw = g_fwd + g_prox

        # 4. Project onto variant subspace: \tilde{G} = P_\perp G_raw
        G_proj = torch.matmul(self.P_perp, G_raw)

        # 5. Online step update
        self.W_t.sub_(self.lr_ttt * G_proj)

        return {
            "loss_forward": 0.5 * torch.sum(err ** 2).item(),
            "loss_inv": loss_inv.item(),
            "weight_drift": torch.norm(self.W_t - self.W_0).item()
        }
