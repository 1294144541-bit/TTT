# Train nominal backbone and run physical evaluation benchmark
# Compares Static Baseline World Model vs. TTT-Augmented In-Weights Adaptation

import torch
import torch.nn.functional as F
import numpy as np
from src.ttt_world_model import TTTAugmentedWorldModel

def pretrain_nominal_world_model():
    data = torch.load("/Users/mac/Documents/论文/ttt后向传播与世界模型/data/train_nominal.pt")
    states = data["states"] # (50, 150, 4)
    actions = data["actions"] # (50, 150, 2)
    next_states = data["next_states"] # (50, 150, 4)

    model = TTTAugmentedWorldModel(obs_dim=4, action_dim=2, latent_dim=16, feature_k_dim=16, lr_ttt=0.08, beta_proximal=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Flatten trajectory pairs
    N, T, _ = states.shape
    s_flat = states.reshape(-1, 4)
    a_flat = actions.reshape(-1, 2)
    s_next_flat = next_states.reshape(-1, 4)

    print("Pretraining nominal invariant backbone on baseline physics...")
    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        z_t = model.encoder(s_flat)
        z_next_true = model.encoder(s_next_flat)
        
        # Predict with nominal core
        z_next_pred = model.invariant_core(z_t, a_flat)
        loss_dyn = F.mse_loss(z_next_pred, z_next_true)
        
        # Train inverse dynamics head
        pred_a = model.inv_dynamics(z_t, z_next_true)
        loss_inv = F.mse_loss(pred_a, a_flat)
        
        loss = loss_dyn + 0.1 * loss_inv
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Dynamics Loss: {loss_dyn.item():.5f} | Inv Loss: {loss_inv.item():.5f}")

    # Compute nullspace projection matrix from nominal context features
    model.eval()
    with torch.no_grad():
        z_all = model.encoder(s_flat)
        x_ctx_all = model.context_mapping(torch.cat([z_all, a_flat], dim=-1))
        model.compute_nullspace_projection(x_ctx_all)
        # Initialize W_0 as zero nominal offset
        model.W_0.zero_()
        model.reset_plastic_weights()

    torch.save(model.state_dict(), "/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/nominal_model.pt")
    print("Nominal backbone pretraining & Nullspace projection computation completed.\n")
    return model

def evaluate_adaptation_on_benchmark(test_data_path: str, scenario_name: str):
    data = torch.load(test_data_path)
    states = data["states"] # (20, 200, 4)
    actions = data["actions"]
    next_states = data["next_states"]
    params = data["params"] # (20, 200, 4) -> [m, mu, e, g]

    model = TTTAugmentedWorldModel(obs_dim=4, action_dim=2, latent_dim=16, feature_k_dim=16, lr_ttt=0.08, beta_proximal=0.02)
    model.load_state_dict(torch.load("/Users/mac/Documents/论文/ttt后向传播与世界模型/checkpoints/nominal_model.pt"))
    model.eval()

    num_episodes, T, _ = states.shape

    static_errors = []
    ttt_errors = []
    drift_magnitudes = []
    W_records = []
    param_records = []

    print(f"--- Running Benchmark: {scenario_name} ---")
    for ep in range(min(5, num_episodes)):
        model.reset_plastic_weights()
        ep_static_err = []
        ep_ttt_err = []

        for t in range(T):
            s_t = states[ep, t:t+1]
            a_t = actions[ep, t:t+1]
            s_next = next_states[ep, t:t+1]
            p_t = params[ep, t].numpy()

            with torch.no_grad():
                z_t = model.encoder(s_t)
                z_next_true = model.encoder(s_next)

                # 1. Static World Model Prediction (W_t = W_0)
                z_pred_static = model.forward_prediction(z_t, a_t, W=model.W_0)
                err_static = F.mse_loss(z_pred_static, z_next_true).item()
                ep_static_err.append(err_static)

                # 2. TTT Model Prediction (before step)
                z_pred_ttt = model.forward_prediction(z_t, a_t, W=model.W_t)
                err_ttt = F.mse_loss(z_pred_ttt, z_next_true).item()
                ep_ttt_err.append(err_ttt)

            # 3. Stream step test-time update
            step_info = model.ttt_online_step(z_t, a_t, z_next_true)
            drift_magnitudes.append(step_info["weight_drift"])

            # Store adapted weights W_t and ground truth physical parameters for linear probe decoding
            W_records.append(model.W_t.detach().cpu().numpy().flatten())
            param_records.append(p_t)

        static_errors.append(ep_static_err)
        ttt_errors.append(ep_ttt_err)

    mean_static = np.mean(static_errors)
    mean_ttt = np.mean(ttt_errors)
    improvement = (mean_static - mean_ttt) / mean_static * 100.0

    print(f"Static World Model Latent MSE: {mean_static:.6f}")
    print(f"TTT-Augmented World Model MSE: {mean_ttt:.6f}")
    print(f"Dynamics Error Reduction Rate: {improvement:.2f}%")

    # Evaluate Physical Decodability (R^2 Score of W_t vs true parameter xi_t)
    W_mat = np.array(W_records) # (N*T, d_k * d_v)
    P_mat = np.array(param_records) # (N*T, 4)
    # Fit linear probe from W_t to mass and friction
    from numpy.linalg import lstsq
    W_bias = np.concatenate([W_mat, np.ones((W_mat.shape[0], 1))], axis=1)
    coeffs, _, _, _ = lstsq(W_bias, P_mat[:, :2], rcond=None)
    pred_params = W_bias @ coeffs

    # Compute R2 for mass and friction
    ss_tot = np.sum((P_mat[:, :2] - np.mean(P_mat[:, :2], axis=0)) ** 2, axis=0)
    ss_res = np.sum((P_mat[:, :2] - pred_params) ** 2, axis=0)
    r2_scores = 1.0 - ss_res / ss_tot
    print(f"Physical Decodability (Linear Probe R^2) -> Mass: {r2_scores[0]:.4f}, Friction: {r2_scores[1]:.4f}")
    print(f"Mean In-Weights Weight Drift norm: {np.mean(drift_magnitudes):.4f}\n")

    return {
        "mean_static_mse": mean_static,
        "mean_ttt_mse": mean_ttt,
        "improvement_pct": improvement,
        "r2_mass": r2_scores[0],
        "r2_friction": r2_scores[1]
    }

if __name__ == "__main__":
    pretrain_nominal_world_model()
    res_sin = evaluate_adaptation_on_benchmark("/Users/mac/Documents/论文/ttt后向传播与世界模型/data/test_sinusoidal.pt", "Continuous Sinusoidal Parameter Drift")
    res_shock = evaluate_adaptation_on_benchmark("/Users/mac/Documents/论文/ttt后向传播与世界模型/data/test_shock.pt", "Abrupt Mass/Friction Shock Drift")
