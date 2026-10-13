# Pretraining script for Vision World Model
# Trains ConvEncoder and Invariant Dynamics Core on nominal physics videos.

import torch
import torch.nn.functional as F
import os
import sys

# Ensure local module visibility
sys.path.append("/Users/mac/Documents/论文/ttt后向传播与世界模型")
from src.vision_ttt_model import VisualTTTWorldModel

def pretrain_vision_world_model():
    data_path = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/vision_train_nominal.pt"
    ckpt_dir = "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    data = torch.load(data_path)
    videos = data["videos"] # (N=40, T=25, 3, 64, 64)
    N, T, C, H, W = videos.shape

    # Pair consecutive frames: (I_t, I_{t+1})
    frames_curr = videos[:, :-1].reshape(-1, C, H, W) # (N*(T-1), 3, 64, 64)
    frames_next = videos[:, 1:].reshape(-1, C, H, W)  # (N*(T-1), 3, 64, 64)

    total_samples = frames_curr.shape[0]
    print(f"Loaded visual dataset: {total_samples} frame transition pairs.")

    model = VisualTTTWorldModel(latent_dim=32, context_k_dim=32, lr_ttt=0.06, beta_proximal=0.015, core_rank=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)

    batch_size = 64
    num_epochs = 12

    print("Beginning visual invariant world model pretraining...")
    model.train()
    for epoch in range(num_epochs):
        perm = torch.randperm(total_samples)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, total_samples, batch_size):
            idx = perm[i:i + batch_size]
            b_curr = frames_curr[idx]
            b_next = frames_next[idx]

            optimizer.zero_grad()
            z_curr = model.encoder(b_curr)
            z_next = model.encoder(b_next)

            # Predict next latent state with invariant dynamics core
            z_pred = model.invariant_core(z_curr)
            loss_latent = F.mse_loss(z_pred, z_next)

            # Smooth representation regularization
            loss_reg = 1e-4 * torch.mean(z_curr ** 2)

            loss = loss_latent + loss_reg
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | Latent Dynamics MSE: {avg_loss:.5f}")

    # Compute Null-Space Projection operator from training set context features
    model.eval()
    with torch.no_grad():
        sample_curr = frames_curr[:200]
        z_sample = model.encoder(sample_curr)
        ctx_sample = model.context_mapping(z_sample)
        model.compute_nullspace_projection(ctx_sample)
        model.W_0.zero_()
        model.reset_plastic_weights()

    ckpt_path = os.path.join(ckpt_dir, "vision_nominal_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Vision world model checkpoint and null-space projection saved to: {ckpt_path}\n")

if __name__ == "__main__":
    pretrain_vision_world_model()
