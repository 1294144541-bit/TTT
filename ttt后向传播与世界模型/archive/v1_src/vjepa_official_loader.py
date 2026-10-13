# Meta V-JEPA 2.1 Official Weights Loader & TTT-Augmented World Model
# Loads pretrained ViT-Base weights (model.safetensors, 419MB)
# Connects official 768-dim spatiotemporal representations with In-Weights Plastic Layer W_t

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
import json
import os
from typing import Dict, Tuple

class VJEPAOfficialEncoder(nn.Module):
    """
    Standard Vision Transformer (ViT-B/16) initialized directly from
    Meta V-JEPA 2.1 official pretrained weights (419MB safetensors).
    Extracts 768-dimensional latent tokens from video frames.
    """
    def __init__(self, weights_path: str):
        super().__init__()
        self.weights_path = weights_path
        self.embed_dim = 768
        self.patch_size = 16

        # Standard ViT Components matching V-JEPA encoder
        # Patch embedding for single-frame (3, 16, 16) -> 768
        self.patch_proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)

        # 12 Transformer Encoder Blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=12,
                dim_feedforward=768 * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            for _ in range(12)
        ])
        self.norm = nn.LayerNorm(768)

        # Load weights from safetensors
        self.load_official_weights()

    def load_official_weights(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"V-JEPA weights not found at {self.weights_path}")

        print(f"Loading official V-JEPA 2.1 weights from: {self.weights_path} ...")
        with safetensors.torch.safe_open(self.weights_path, framework="pt") as f:
            # Map 2D image patch embedding weights
            if "encoder.embeddings.patch_embeddings_img.proj.weight" in f.keys():
                w = f.get_tensor("encoder.embeddings.patch_embeddings_img.proj.weight") # (768, 3, 1, 16, 16)
                b = f.get_tensor("encoder.embeddings.patch_embeddings_img.proj.bias")
                self.patch_proj.weight.data.copy_(w.squeeze(2))
                self.patch_proj.bias.data.copy_(b)

            # Map attention and MLP weights layer by layer
            for i in range(12):
                prefix = f"encoder.layer.{i}."
                # Attention projections
                q_w = f.get_tensor(f"{prefix}attention.query.weight")
                k_w = f.get_tensor(f"{prefix}attention.key.weight")
                v_w = f.get_tensor(f"{prefix}attention.value.weight")
                q_b = f.get_tensor(f"{prefix}attention.query.bias")
                k_b = f.get_tensor(f"{prefix}attention.key.bias")
                v_b = f.get_tensor(f"{prefix}attention.value.bias")

                # Pack in_proj for nn.MultiheadAttention
                in_proj_w = torch.cat([q_w, k_w, v_w], dim=0)
                in_proj_b = torch.cat([q_b, k_b, v_b], dim=0)
                self.layers[i].self_attn.in_proj_weight.data.copy_(in_proj_w)
                self.layers[i].self_attn.in_proj_bias.data.copy_(in_proj_b)

                # Attention Output projection
                out_w = f.get_tensor(f"{prefix}attention.proj.weight")
                out_b = f.get_tensor(f"{prefix}attention.proj.bias")
                self.layers[i].self_attn.out_proj.weight.data.copy_(out_w)
                self.layers[i].self_attn.out_proj.bias.data.copy_(out_b)

                # Feed forward MLP: fc1 and fc2
                fc1_w = f.get_tensor(f"{prefix}mlp.fc1.weight")
                fc1_b = f.get_tensor(f"{prefix}mlp.fc1.bias")
                fc2_w = f.get_tensor(f"{prefix}mlp.fc2.weight")
                fc2_b = f.get_tensor(f"{prefix}mlp.fc2.bias")
                self.layers[i].linear1.weight.data.copy_(fc1_w)
                self.layers[i].linear1.bias.data.copy_(fc1_b)
                self.layers[i].linear2.weight.data.copy_(fc2_w)
                self.layers[i].linear2.bias.data.copy_(fc2_b)

                # LayerNorms
                ln1_w = f.get_tensor(f"{prefix}norm1.weight")
                ln1_b = f.get_tensor(f"{prefix}norm1.bias")
                ln2_w = f.get_tensor(f"{prefix}norm2.weight")
                ln2_b = f.get_tensor(f"{prefix}norm2.bias")
                self.layers[i].norm1.weight.data.copy_(ln1_w)
                self.layers[i].norm1.bias.data.copy_(ln1_b)
                self.layers[i].norm2.weight.data.copy_(ln2_w)
                self.layers[i].norm2.bias.data.copy_(ln2_b)

            # Final layer norm if present
            if "encoder.norm.weight" in f.keys():
                self.norm.weight.data.copy_(f.get_tensor("encoder.norm.weight"))
                self.norm.bias.data.copy_(f.get_tensor("encoder.norm.bias"))

        print("Meta V-JEPA 2.1 ViT-Base 12-layer backbone loaded successfully!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        # Patch projection -> (B, 768, H/16, W/16) -> (B, N_patches, 768)
        B = x.shape[0]
        h = self.patch_proj(x)
        h = h.flatten(2).transpose(1, 2)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Global spatial pooling -> token embedding z \in R^{768}
        z = h.mean(dim=1)
        return z

class VJEPAWorldModelTTT(nn.Module):
    """
    World Model integrating Meta V-JEPA 2.1 with In-Weights Test-Time Training.
    Latent space: z_t \in R^{768}
    Plastic dynamics: W_t \in R^{d_k \times 768}
    """
    def __init__(
        self,
        vjepa_weights_path: str,
        context_k_dim: int = 64,
        lr_ttt: float = 0.05,
        beta_proximal: float = 0.01,
        core_rank: int = 16
    ):
        super().__init__()
        self.d_z = 768
        self.d_k = context_k_dim
        self.d_v = 768
        self.lr_ttt = lr_ttt
        self.beta = beta_proximal
        self.core_rank = core_rank

        # 1. Official V-JEPA 2.1 Feature Encoder (Frozen during test-time)
        self.encoder = VJEPAOfficialEncoder(vjepa_weights_path)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2. Invariant Spatiotemporal Dynamics Backbone f_\theta (768 -> 768)
        self.invariant_core = nn.Sequential(
            nn.Linear(self.d_z, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.d_z)
        )

        # 3. Dynamic Context Feature Mapping \psi(z_t)
        self.context_mapping = nn.Sequential(
            nn.Linear(self.d_z, 128),
            nn.GELU(),
            nn.Linear(128, self.d_k)
        )

        # 4. In-Weights Plastic Buffers
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

    def forward_latent_prediction(self, z_t: torch.Tensor, W: torch.Tensor = None) -> torch.Tensor:
        if W is None:
            W = self.W_t
        z_nom = self.invariant_core(z_t)
        x_ctx = self.context_mapping(z_t)
        z_res = torch.matmul(x_ctx, W)
        return z_nom + z_res

    def ttt_step(self, z_t: torch.Tensor, z_next_observed: torch.Tensor) -> Dict[str, float]:
        z_t = z_t.detach()
        z_next_obs = z_next_observed.detach()

        x_t = self.context_mapping(z_t)
        f_nom = self.invariant_core(z_t)
        y_target = z_next_obs - f_nom

        err = torch.matmul(x_t, self.W_t) - y_target
        g_fwd = torch.matmul(x_t.T, err)
        g_prox = self.beta * (self.W_t - self.W_0)

        G_raw = g_fwd + g_prox
        G_proj = torch.matmul(self.P_perp, G_raw)

        self.W_t.sub_(self.lr_ttt * G_proj)

        loss_fwd = 0.5 * torch.sum(err ** 2).item()
        weight_norm = torch.norm(self.W_t - self.W_0).item()

        return {
            "loss_forward": loss_fwd,
            "weight_drift": weight_norm
        }
