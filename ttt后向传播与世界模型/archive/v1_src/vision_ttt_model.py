# Visual World Model with Test-Time Training (TTT) Adaptation
# Components:
# 1. ConvEncoder E_\phi: R^{3x64x64} -> R^{d_z} (Latent space representation)
# 2. Invariant Physical Dynamics Core f_\theta: R^{d_z} -> R^{d_z}
# 3. Plastic Dynamics Layer W_t \in R^{d_k \times d_v} with TTT-Linear adaptation
# 4. Null-Space Projection P_\perp and OU-bounded proximal force

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ConvObservationEncoder(nn.Module):
    """
    Encodes 64x64 RGB frames into compact latent embedding z_t \in R^{d_z}.
    Architecture: 3 Conv blocks with GroupNorm & GELU -> AdaptiveAvgPool -> Linear
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1), # 32x32
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),          # 16x16
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),          # 8x8
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Linear(64 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 64, 64)
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        z = self.fc(h)
        return z

class InvariantDynamicsCore(nn.Module):
    """
    Frozen spatiotemporal backbone representing nominal physical laws.
    Maps z_t -> \hat{z}_{t+1} (Nominal estimate)
    """
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)

class VisualTTTWorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        context_k_dim: int = 32,
        lr_ttt: float = 0.05,
        beta_proximal: float = 0.02,
        core_rank: int = 8
    ):
        super().__init__()
        self.d_z = latent_dim
        self.d_k = context_k_dim
        self.d_v = latent_dim
        self.lr_ttt = lr_ttt
        self.beta = beta_proximal
        self.core_rank = core_rank

        # 1. Feature extractor (gradient truncated during test-time training)
        self.encoder = ConvObservationEncoder(in_channels=3, latent_dim=latent_dim)

        # 2. Invariant physics core
        self.invariant_core = InvariantDynamicsCore(latent_dim=latent_dim)

        # 3. Context feature mapping \psi(z_t)
        self.context_mapping = nn.Sequential(
            nn.Linear(latent_dim, 48),
            nn.GELU(),
            nn.Linear(48, self.d_k)
        )

        # 4. In-weights plastic parameters
        self.register_buffer("W_0", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("W_t", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("P_perp", torch.eye(self.d_k))
        self.register_buffer("U_core", torch.zeros(self.d_k, core_rank))

    def reset_plastic_weights(self):
        self.W_t.copy_(self.W_0)

    def compute_nullspace_projection(self, context_features: torch.Tensor):
        # context_features: (N, d_k)
        cov = torch.matmul(context_features.T, context_features) / context_features.shape[0]
        U, S, V = torch.linalg.svd(cov)
        U_core = U[:, :self.core_rank]
        P_perp = torch.eye(self.d_k, device=cov.device) - torch.matmul(U_core, U_core.T)
        self.U_core.copy_(U_core)
        self.P_perp.copy_(P_perp)

    def forward_latent_prediction(self, z_t: torch.Tensor, W: torch.Tensor = None) -> torch.Tensor:
        if W is None:
            W = self.W_t
        # Nominal backbone prediction
        z_nom = self.invariant_core(z_t)
        # Dynamic context feature
        x_ctx = self.context_mapping(z_t)
        # Plastic dynamics adaptation residual
        z_res = torch.matmul(x_ctx, W)
        return z_nom + z_res

    def ttt_step(self, z_t: torch.Tensor, z_next_observed: torch.Tensor) -> Dict[str, float]:
        """
        Streaming Test-Time Training online update step:
        1. Forward residual error e_t
        2. Proximal regularization pull-back -\beta(W_t - W_0)
        3. Null-space projection onto variant dynamics subspace: \tilde{G}_t = P_\perp G_t
        """
        # Detach targets (no backpropagation into encoder during test time)
        z_t = z_t.detach()
        z_next_obs = z_next_observed.detach()

        x_t = self.context_mapping(z_t) # (1, d_k)
        f_nom = self.invariant_core(z_t) # (1, d_v)
        y_target = z_next_obs - f_nom    # (1, d_v)

        # Forward prediction error
        err = torch.matmul(x_t, self.W_t) - y_target # (1, d_v)
        g_fwd = torch.matmul(x_t.T, err)             # (d_k, d_v)

        # Proximal pull-back force (Ornstein-Uhlenbeck drift bound)
        g_prox = self.beta * (self.W_t - self.W_0)   # (d_k, d_v)

        # Raw gradient
        G_raw = g_fwd + g_prox

        # Subspace null-space projection (prevents catastrophic degradation of core physical laws)
        G_proj = torch.matmul(self.P_perp, G_raw)

        # Online in-weights update
        self.W_t.sub_(self.lr_ttt * G_proj)

        loss_fwd = 0.5 * torch.sum(err ** 2).item()
        weight_norm = torch.norm(self.W_t - self.W_0).item()

        return {
            "loss_forward": loss_fwd,
            "weight_drift": weight_norm
        }
