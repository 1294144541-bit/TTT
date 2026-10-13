# Meta V-JEPA 2.1 World Model with Dynamics Velocity Projector & In-Weights TTT
# Resolves semantic invariance collapse by operating on velocity delta manifolds:
#   \Delta z_t = z_t - z_{t-1}
# Incorporates:
# - Official 85.65M ViT-B/16 V-JEPA 2.1 feature extractor
# - Dynamics velocity neck (projecting Delta z_t into physical momentum state)
# - In-weights plastic layer W_t \in R^{d_k \times d_v}
# - Null-space projection P_\perp and OU-bounded drift

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vjepa_official_loader import VJEPAOfficialEncoder

class VJEPADynamicsProjector(nn.Module):
    """
    Physical dynamics projection neck.
    Combines instantaneous state z_t with velocity delta \Delta z_t = z_t - z_{t-1}
    to preserve acceleration and collision impulse dynamics.
    """
    def __init__(self, in_dim: int = 768, dyn_dim: int = 128):
        super().__init__()
        # State + Velocity -> Physical State
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, dyn_dim),
            nn.LayerNorm(dyn_dim)
        )

    def forward(self, z_curr: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        delta_v = z_curr - z_prev
        combined = torch.cat([z_curr, delta_v], dim=-1)
        return self.proj(combined)

class VJEPAPhysicsTTTWorldModel(nn.Module):
    def __init__(
        self,
        vjepa_weights_path: str,
        dyn_dim: int = 128,
        context_k_dim: int = 64,
        lr_ttt: float = 0.05,
        beta_proximal: float = 0.01,
        core_rank: int = 16
    ):
        super().__init__()
        self.dyn_dim = dyn_dim
        self.d_k = context_k_dim
        self.d_v = dyn_dim
        self.lr_ttt = lr_ttt
        self.beta = beta_proximal
        self.core_rank = core_rank

        # 1. Official V-JEPA 2.1 ViT-B/16 (Frozen)
        self.encoder = VJEPAOfficialEncoder(vjepa_weights_path)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2. Dynamics Velocity Neck
        self.dyn_neck = VJEPADynamicsProjector(in_dim=768, dyn_dim=dyn_dim)

        # 3. Invariant Physical Dynamics Core f_\theta: s_t -> s_{t+1}
        self.invariant_core = nn.Sequential(
            nn.Linear(dyn_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, dyn_dim)
        )

        # 4. Context feature mapping \psi(s_t)
        self.context_mapping = nn.Sequential(
            nn.Linear(dyn_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.d_k)
        )

        # 5. In-weights plastic parameters
        self.register_buffer("W_0", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("W_t", torch.zeros(self.d_k, self.d_v))
        self.register_buffer("P_perp", torch.eye(self.d_k))
        self.register_buffer("U_core", torch.zeros(self.d_k, core_rank))

    def reset_plastic_weights(self):
        self.W_t.copy_(self.W_0)

    def compute_nullspace_projection(self, context_features: torch.Tensor):
        cov = torch.matmul(context_features.T, context_features) / context_features.shape[0]
        U, S, V = torch.linalg.svd(cov)
        U_core = U[:, :self.core_rank]
        P_perp = torch.eye(self.d_k, device=cov.device) - torch.matmul(U_core, U_core.T)
        self.U_core.copy_(U_core)
        self.P_perp.copy_(P_perp)

    def forward_state_prediction(self, s_t: torch.Tensor, W: torch.Tensor = None) -> torch.Tensor:
        if W is None:
            W = self.W_t
        s_nom = self.invariant_core(s_t)
        x_ctx = self.context_mapping(s_t)
        s_res = torch.matmul(x_ctx, W)
        return s_nom + s_res

    def ttt_step(self, s_t: torch.Tensor, s_next_observed: torch.Tensor) -> dict:
        s_t = s_t.detach()
        s_next_obs = s_next_observed.detach()

        x_t = self.context_mapping(s_t)
        f_nom = self.invariant_core(s_t)
        y_target = s_next_obs - f_nom

        err = torch.matmul(x_t, self.W_t) - y_target
        g_fwd = torch.matmul(x_t.T, err)
        g_prox = self.beta * (self.W_t - self.W_0)

        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(self.P_perp, G_raw)

        self.W_t.sub_(self.lr_ttt * G_proj)

        return {
            "loss_forward": 0.5 * torch.sum(err ** 2).item(),
            "weight_drift": torch.norm(self.W_t - self.W_0).item()
        }
