# Actuated Continuous Control Physical Environment with Explicit Actions
# Simulates actuated cart-pendulum mechanics with time-varying physical parameters.
# State: s_t = [x, x_dot, theta, theta_dot]
# Action: a_t in [-3.0, 3.0]
# Dynamics Parameters: [pole_mass, friction_mu, gravity]

import numpy as np
import torch
from typing import Dict, Tuple

class NonStationaryActuatedPhysicsEnv:
    def __init__(
        self,
        dt: float = 0.05,
        gravity: float = 9.81,
        cart_mass: float = 1.0,
        pole_mass_nominal: float = 0.2,
        pole_length: float = 0.5,
        friction_nominal: float = 0.15
    ):
        self.dt = dt
        self.g = gravity
        self.m_c = cart_mass
        self.m_p_nominal = pole_mass_nominal
        self.l = pole_length
        self.mu_nominal = friction_nominal
        self.state = np.zeros(4, dtype=np.float32)

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state = np.array([
            0.0,
            0.0,
            np.random.uniform(-0.08, 0.08),
            np.random.uniform(-0.04, 0.04)
        ], dtype=np.float32)
        return self.state.copy()

    def step(self, action: float, params: np.ndarray = None) -> Tuple[np.ndarray, float, bool, Dict]:
        if params is not None:
            m_p, mu, g = params[0], params[1], params[2]
        else:
            m_p, mu, g = self.m_p_nominal, self.mu_nominal, self.g

        x, x_dot, theta, theta_dot = self.state
        force = float(np.clip(action, -3.0, 3.0))

        # Friction drag opposing cart linear motion
        f_drag = -mu * x_dot if abs(x_dot) > 1e-4 else 0.0

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        total_m = self.m_c + m_p
        pole_mass_length = m_p * self.l

        temp = (force + f_drag + pole_mass_length * (theta_dot**2) * sintheta) / total_m
        theta_acc = (g * sintheta - costheta * temp) / (self.l * (4.0/3.0 - m_p * (costheta**2) / total_m))
        x_acc = temp - pole_mass_length * theta_acc * costheta / total_m

        # Symplectic update
        x_dot = x_dot + x_acc * self.dt
        x = x + x_dot * self.dt
        theta_dot = theta_dot + theta_acc * self.dt
        theta = theta + theta_dot * self.dt

        # Angle normalization
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # Standard quadratic tracking cost
        reward = -(x**2 + 6.0 * (theta**2) + 0.1 * (x_dot**2) + 0.5 * (theta_dot**2) + 0.01 * (force**2))
        done = bool(abs(x) > 4.0 or abs(theta) > np.pi / 2.0)

        return self.state.copy(), reward, done, {"m_p": m_p, "mu": mu}

def generate_actuated_dataset(
    num_episodes: int = 50,
    episode_length: int = 50,
    dynamics_shift: str = "nominal",
    seed: int = 42
) -> Dict[str, torch.Tensor]:
    np.random.seed(seed)
    env = NonStationaryActuatedPhysicsEnv(dt=0.05)

    all_s, all_a, all_next_s, all_p = [], [], [], []

    for ep in range(num_episodes):
        s = env.reset(seed=seed + ep * 13)
        ep_s, ep_a, ep_next_s, ep_p = [], [], [], []

        base_mp = 0.2
        base_mu = 0.15
        base_g = 9.81

        for t in range(episode_length):
            if dynamics_shift == "nominal":
                m_p_t = base_mp * np.random.uniform(0.95, 1.05)
                mu_t = base_mu * np.random.uniform(0.95, 1.05)
            elif dynamics_shift == "heavy_payload_shift":
                m_p_t = base_mp * (3.5 if t >= episode_length // 2 else 1.0)
                mu_t = base_mu
            elif dynamics_shift == "ice_friction_shift":
                m_p_t = base_mp
                mu_t = base_mu * (0.2 if t >= episode_length // 2 else 1.0)

            params = np.array([m_p_t, mu_t, base_g], dtype=np.float32)

            # Stabilizing exploratory policy
            act = float(-1.2 * s[0] - 0.4 * s[1] - 7.5 * s[2] - 1.1 * s[3] + np.random.normal(0, 0.35))
            act = float(np.clip(act, -2.5, 2.5))

            next_s, _, _, _ = env.step(act, params=params)

            ep_s.append(s)
            ep_a.append([act])
            ep_next_s.append(next_s)
            ep_p.append(params)
            s = next_s

        all_s.append(ep_s)
        all_a.append(ep_a)
        all_next_s.append(ep_next_s)
        all_p.append(ep_p)

    return {
        "states": torch.tensor(np.array(all_s), dtype=torch.float32),
        "actions": torch.tensor(np.array(all_a), dtype=torch.float32),
        "next_states": torch.tensor(np.array(all_next_s), dtype=torch.float32),
        "params": torch.tensor(np.array(all_p), dtype=torch.float32)
    }

if __name__ == "__main__":
    out_dir = "/Users/mac/Documents/论文/ttt后向传播与世界模型/benchmarks_v2"
    print("Generating actuated continuous control datasets with explicit action sequences...")
    d_train = generate_actuated_dataset(num_episodes=50, episode_length=50, dynamics_shift="nominal", seed=42)
    d_test_nom = generate_actuated_dataset(num_episodes=20, episode_length=50, dynamics_shift="nominal", seed=101)
    d_test_mass = generate_actuated_dataset(num_episodes=20, episode_length=50, dynamics_shift="heavy_payload_shift", seed=202)
    d_test_fric = generate_actuated_dataset(num_episodes=20, episode_length=50, dynamics_shift="ice_friction_shift", seed=303)

    torch.save(d_train, f"{out_dir}/actuated_train_nominal.pt")
    torch.save(d_test_nom, f"{out_dir}/actuated_test_nominal.pt")
    torch.save(d_test_mass, f"{out_dir}/actuated_test_heavy_mass.pt")
    torch.save(d_test_fric, f"{out_dir}/actuated_test_ice_friction.pt")
    print("Done. Actuated datasets saved.")
