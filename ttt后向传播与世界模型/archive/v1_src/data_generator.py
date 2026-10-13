# Non-Stationary Dynamics Data Generator
# Generates continuous physical interaction trajectories with non-stationary parameter drift

import numpy as np
import torch
from typing import Dict

class NonStationaryPhysicsSimulator:
    def __init__(self, dt: float = 0.05, noise_std: float = 0.01):
        self.dt = dt
        self.noise_std = noise_std
        
    def step(self, state: np.ndarray, action: np.ndarray, params: np.ndarray) -> np.ndarray:
        pos = state[:2]
        vel = state[2:]
        
        m, mu, e, g = params[0], params[1], params[2], params[3]
        m = max(m, 0.1)
        
        speed = np.linalg.norm(vel)
        if speed > 1e-4:
            f_drag = -mu * vel
        else:
            f_drag = np.zeros_like(vel)
            
        acc_x = (action[0] + f_drag[0]) / m
        acc_y = (action[1] + f_drag[1]) / m - g
        
        new_pos = pos + vel * self.dt
        new_vel = vel + np.array([acc_x, acc_y]) * self.dt
        
        if new_pos[1] <= 0.0:
            new_pos[1] = 0.0
            new_vel[1] = -e * new_vel[1]
            
        noise = np.random.normal(0, self.noise_std, size=4)
        next_state = np.concatenate([new_pos, new_vel]) + noise
        return next_state


def generate_drift_trajectories(
    num_episodes: int = 100,
    episode_length: int = 200,
    drift_type: str = "sinusoidal",
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    np.random.seed(seed)
    sim = NonStationaryPhysicsSimulator(dt=0.05, noise_std=0.005)
    
    all_states = []
    all_actions = []
    all_next_states = []
    all_params = []
    
    for ep in range(num_episodes):
        ep_states = []
        ep_actions = []
        ep_next_states = []
        ep_params = []
        
        curr_state = np.array([0.0, np.random.uniform(1.0, 5.0), np.random.uniform(-1.0, 1.0), 0.0])
        
        base_m = np.random.uniform(1.0, 2.0)
        base_mu = np.random.uniform(0.2, 0.8)
        base_e = 0.7
        base_g = 9.81
        
        for t in range(episode_length):
            if drift_type == "sinusoidal":
                m_t = base_m * (1.0 + 0.6 * np.sin(2 * np.pi * t / 80.0))
                mu_t = base_mu * (1.0 + 0.5 * np.cos(2 * np.pi * t / 60.0))
            elif drift_type == "abrupt_shock":
                scale = 2.5 if t > (episode_length // 2) else 1.0
                m_t = base_m * scale
                mu_t = base_mu * (0.2 if t > (episode_length // 2) else 1.0)
            else:
                m_t = base_m
                mu_t = base_mu
                
            param_t = np.array([m_t, mu_t, base_e, base_g])
            
            action_t = np.array([
                np.sin(t * 0.1) * 3.0 + np.random.normal(0, 0.5),
                np.cos(t * 0.08) * 2.0 + np.random.normal(0, 0.5)
            ])
            
            next_state = sim.step(curr_state, action_t, param_t)
            
            ep_states.append(curr_state)
            ep_actions.append(action_t)
            ep_next_states.append(next_state)
            ep_params.append(param_t)
            
            curr_state = next_state
            
        all_states.append(ep_states)
        all_actions.append(ep_actions)
        all_next_states.append(ep_next_states)
        all_params.append(ep_params)
        
    return {
        "states": torch.tensor(np.array(all_states), dtype=torch.float32),
        "actions": torch.tensor(np.array(all_actions), dtype=torch.float32),
        "next_states": torch.tensor(np.array(all_next_states), dtype=torch.float32),
        "params": torch.tensor(np.array(all_params), dtype=torch.float32)
    }

if __name__ == "__main__":
    data_train = generate_drift_trajectories(num_episodes=50, episode_length=150, drift_type="stationary", seed=101)
    data_test = generate_drift_trajectories(num_episodes=20, episode_length=200, drift_type="sinusoidal", seed=202)
    data_shock = generate_drift_trajectories(num_episodes=20, episode_length=200, drift_type="abrupt_shock", seed=303)
    
    torch.save(data_train, "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/train_nominal.pt")
    torch.save(data_test, "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/test_sinusoidal.pt")
    torch.save(data_shock, "/Users/mac/Documents/论文/ttt后向传播与世界模型/data/test_shock.pt")
    print(f"Data generation complete. Train: {data_train['states'].shape}, Test: {data_test['states'].shape}")
