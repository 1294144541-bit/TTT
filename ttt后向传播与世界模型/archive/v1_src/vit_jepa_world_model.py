# ViT-JEPA World Model with Test-Time Training (TTT) Adaptation
# Components:
# 1. ViT-Base Encoder E_\phi: R^{3x224x224} -> R^{768} (Standard JEPA ViT-B/16 architecture)
# 2. Invariant Physical Dynamics Core f_\theta: R^{768} -> R^{768}
# 3. Plastic Dynamics Layer W_t \in R^{d_k \times 768} with TTT-Linear adaptation
# 4. Null-Space Projection P_\perp and OU-bounded proximal regularization

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict

class ViTJEPAEncoder(nn.Module):
    """
    Standard ViT-Base/16 architecture matching V-JEPA encoder topology.
    Encodes 224x224 RGB video frames into 768-dimensional token representation.
    """
    def __init__(self, latent_dim: int = 768):
        super().__init__()
        # 12-layer Vision Transformer backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.embed_dim = self.vit.embed_dim # 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) -> (B, 768)
        return self.vit(x)

class ViTJEPAWorldModel(nn.Module):
    """
    Full JEPA World Model with In-Weights Test-Time Training.
    State representation z_t \in R^{768}
    Plastic adaptation layer W_t \in R^{d_k \times 768}
    """
    def __init__(
        self,
        latent_dim: int = 768,
        context_k_dim: int = 64,
        lr_ttt: float = 0.05,
        beta_proximal: float = 0.01,
        core_rank: int = 16
    ):
        super().__init__()
        self.d_z = latent_dim
        self.d_k = context_k_dim
        self.d_v = latent_dim
        self.lr_ttt = lr_ttt
        self.beta = beta_proximal
        self.core_rank = core_rank

        # 1. ViT-JEPA Observation Encoder (Gradient truncated during test time)
        self.encoder = ViTJEPAEncoder(latent_dim=latent_dim)

        # 2. Invariant Spatiotemporal Dynamics Backbone f_\theta
        self.invariant_core = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )

        # 3. Dynamic Context Feature Mapping \psi(z_t)
        self.context_mapping = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.d_k)
        )

        # 4. In-Weights Plastic Parameter Buffers
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
        # Nominal backbone physics prediction
        z_nom = self.invariant_core(z_t)
        # Context feature mapping
        x_ctx = self.context_mapping(z_t)
        # Plastic dynamics adaptation residual: g(x; W) = x @ W
        z_res = torch.matmul(x_ctx, W)
        return z_nom + z_res

    def ttt_step(self, z_t: torch.Tensor, z_next_observed: torch.Tensor) -> Dict[str, float]:
        """
        Streaming Test-Time Training online update step on ViT representations:
        1. Forward residual error e_t \in R^{1 x 768}
        2. Proximal regularization pull-back -\beta(W_t - W_0)
        3. Null-space projection onto variant dynamics subspace: \tilde{G}_t = P_\perp G_t
        """
        # Detach targets (no backpropagation into ViT encoder during test time)
        z_t = z_t.detach()
        z_next_obs = z_next_observed.detach()

        x_t = self.context_mapping(z_t) # (1, d_k)
        f_nom = self.invariant_core(z_t) # (1, 768)
        y_target = z_next_obs - f_nom    # (1, 768)

        # Forward prediction error
        err = torch.matmul(x_t, self.W_t) - y_target # (1, 768)
        g_fwd = torch.matmul(x_t.T, err)             # (d_k, 768)

        # Proximal pull-back force (Ornstein-Uhlenbeck drift bound)
        g_prox = self.beta * (self.W_t - self.W_0)   # (d_k, 768)

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
