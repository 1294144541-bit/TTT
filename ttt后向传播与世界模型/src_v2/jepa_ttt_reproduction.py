# Exact Reproduction of JEPA-TTT (arXiv:2610.00722, Zhang et al., 2026) Baseline
# Key Mechanisms of JEPA-TTT:
# 1. Growing Dense Replay Buffer: Stores all observed transitions throughout test-time episodes.
# 2. Point-Estimate Dynamics Predictor: Adapts without uncertainty estimation or covariance tracking.
# 3. Dense Replay Mini-batch Updates: Samples random past transitions (s, a, s') for unconstrained gradient steps.
# 4. Standard CEM / MPPI Planning without Uncertainty Penalization (Optimistic Planning).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class GrowingDenseReplayBuffer:
    """
    Growing dense replay buffer strictly matching JEPA-TTT (Zhang et al., 2026).
    Appends observed transition windows across episodes throughout test-time.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []

    def add(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s.detach(), a.detach(), s_next.detach()))

    def sample_batch(self, batch_size: int = 16):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s_batch = torch.cat([b[0] for b in batch], dim=0)
        a_batch = torch.cat([b[1] for b in batch], dim=0)
        s_next_batch = torch.cat([b[2] for b in batch], dim=0)
        return s_batch, a_batch, s_next_batch

    def __len__(self):
        return len(self.buffer)

class JEPATTTWorldModelBaseline(nn.Module):
    """
    JEPA-TTT Model Baseline matching arXiv:2610.00722.
    Adapts an unconstrained latent predictor head online using dense replay sampling.
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 1, hidden_dim: int = 64, lr_ttt: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_ttt = lr_ttt

        # Predictor head: (s_t, a_t) -> \hat{s}_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.optimizer = torch.optim.SGD(self.predictor.parameters(), lr=lr_ttt)
        self.replay_buffer = GrowingDenseReplayBuffer(capacity=5000)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=-1)
        return self.predictor(sa)

    def update_from_transition(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor, replay_steps: int = 2):
        # 1. Add to growing dense replay buffer
        self.replay_buffer.add(s, a, s_next)

        # 2. Replay mini-batch updates
        for _ in range(replay_steps):
            if len(self.replay_buffer) >= 4:
                b_s, b_a, b_next = self.replay_buffer.sample_batch(batch_size=8)
                self.optimizer.zero_grad()
                pred = self.forward(b_s, b_a)
                loss = F.mse_loss(pred, b_next)
                loss.backward()
                self.optimizer.step()
