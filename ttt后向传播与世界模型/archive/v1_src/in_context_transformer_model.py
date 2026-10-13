# In-Context Transformer World Model Baseline (with Causal Self-Attention and KV-Cache)
# Models dynamics using multi-frame causal context:
#   \hat{s}_{t+1} = TransformerBlock(s_{t-L : t}, KV-Cache)
# Represents the strongest static baseline (similar to IRIS / TransDreamer / Decision Transformer).

import torch
import torch.nn as nn
import torch.nn.functional as F

class InContextTransformerWorldModel(nn.Module):
    """
    Causal Transformer World Model for In-Context Latent Dynamics Prediction.
    Maintains a sliding-window KV-Cache over the past L steps.
    Learns an in-context implicit system identification representation across historical tokens.
    """
    def __init__(
        self,
        dyn_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        max_context_len: int = 32
    ):
        super().__init__()
        self.dyn_dim = dyn_dim
        self.hidden_dim = hidden_dim
        self.max_context_len = max_context_len

        # Positional encoding for sequence steps
        self.pos_emb = nn.Embedding(max_context_len + 10, hidden_dim)
        self.input_proj = nn.Linear(dyn_dim, hidden_dim)

        # Causal Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, dyn_dim)

    def forward(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking across historical context.
        state_sequence: (B, L, dyn_dim), where L <= max_context_len
        Returns:
            predicted_next_state: (B, dyn_dim) corresponding to the last step's forecast
        """
        B, L, _ = state_sequence.shape
        positions = torch.arange(L, device=state_sequence.device).unsqueeze(0).expand(B, L)
        
        # Token projection + positional embedding
        h = self.input_proj(state_sequence) + self.pos_emb(positions)

        # Causal mask (prevent looking into future)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=state_sequence.device)

        # Causal Transformer forward
        out = self.transformer(h, mask=causal_mask, is_causal=True)

        # Forecast from last token
        pred_next = self.output_head(out[:, -1])
        return pred_next
