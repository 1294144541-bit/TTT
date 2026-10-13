# Visual Physical Dynamics Data Generator
# Generates continuous RGB collision and sliding video clips (N, T, 3, 64, 64)
# using PIL and pure Python/NumPy, with physical parameter annotations.

import numpy as np
import torch
from PIL import Image, ImageDraw
import os

class VisualPhysicsSimulator:
    def __init__(self, width: int = 64, height: int = 64, dt: float = 0.05):
        self.width = width
        self.height = height
        self.dt = dt
        self.radius = 4.0

    def step_physics(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        # state: [x, y, vx, vy]
        # params: [mass m, friction mu, restitution e, gravity g]
        x, y, vx, vy = state
        m, mu, e, g = params
        m = max(m, 0.2)

        speed = np.sqrt(vx**2 + vy**2)
        if speed > 1e-3:
            f_drag_x = -mu * vx
            f_drag_y = -mu * vy
        else:
            f_drag_x, f_drag_y = 0.0, 0.0

        ax = f_drag_x / m
        ay = (f_drag_y / m) + g # downward gravity in screen coords

        new_x = x + vx * self.dt
        new_y = y + vy * self.dt
        new_vx = vx + ax * self.dt
        new_vy = vy + ay * self.dt

        # Boundary collisions (floor and walls)
        floor_y = self.height - self.radius - 2.0
        if new_y >= floor_y:
            new_y = floor_y
            new_vy = -e * new_vy
            # kinetic friction on bounce contact
            new_vx = new_vx * 0.9

        if new_x <= self.radius:
            new_x = self.radius
            new_vx = -e * new_vx
        elif new_x >= self.width - self.radius:
            new_x = self.width - self.radius
            new_vx = -e * new_vx

        return np.array([new_x, new_y, new_vx, new_vy], dtype=np.float32)

    def render_frame(self, state: np.ndarray, target_pos: np.ndarray = None) -> np.ndarray:
        # Returns (3, 64, 64) normalized float32 tensor
        img = Image.new("RGB", (self.width, self.height), color=(20, 24, 30))
        draw = ImageDraw.Draw(img)

        # Draw ground line
        floor_y = self.height - 2.0
        draw.line([(0, floor_y), (self.width, floor_y)], fill=(80, 85, 95), width=2)

        # Draw moving rigid sphere (reddish-cyan gradient tone)
        x, y = state[0], state[1]
        r = self.radius
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(230, 80, 70), outline=(255, 200, 190))

        # If secondary object / obstacle present
        if target_pos is not None:
            tx, ty = target_pos[0], target_pos[1]
            draw.rectangle([tx - 3, ty - 3, tx + 3, ty + 3], fill=(60, 160, 220))

        arr = np.array(img, dtype=np.float32) / 255.0 # (64, 64, 3)
        arr = np.transpose(arr, (2, 0, 1)) # (3, 64, 64)
        return arr

def generate_video_dataset(
    num_episodes: int = 40,
    seq_len: int = 25,
    drift_mode: str = "nominal",
    seed: int = 42
):
    np.random.seed(seed)
    sim = VisualPhysicsSimulator(width=64, height=64, dt=0.06)

    all_videos = []
    all_states = []
    all_params = []

    for ep in range(num_episodes):
        ep_frames = []
        ep_states = []
        ep_params = []

        # Initial random position & velocity
        init_x = np.random.uniform(8.0, 25.0)
        init_y = np.random.uniform(10.0, 30.0)
        init_vx = np.random.uniform(15.0, 45.0)
        init_vy = np.random.uniform(-10.0, 10.0)
        state = np.array([init_x, init_y, init_vx, init_vy], dtype=np.float32)

        # Base physical parameters
        if drift_mode == "nominal":
            base_m = np.random.uniform(1.0, 1.5)
            base_mu = np.random.uniform(0.3, 0.6)
            base_e = 0.75
            base_g = 18.0
        elif drift_mode == "heavy_friction_ood": # OOD extreme friction & heavy mass
            base_m = np.random.uniform(2.5, 4.0)
            base_mu = np.random.uniform(1.2, 2.0)
            base_e = 0.4
            base_g = 18.0
        elif drift_mode == "low_gravity_elastic_ood": # OOD low gravity & bouncy
            base_m = np.random.uniform(0.5, 0.8)
            base_mu = np.random.uniform(0.05, 0.15)
            base_e = 0.92
            base_g = 6.0

        for t in range(seq_len):
            # Dynamic time-varying drift within episode
            if drift_mode == "nominal":
                m_t = base_m
                mu_t = base_mu
            else:
                m_t = base_m * (1.0 + 0.3 * np.sin(2 * np.pi * t / 20.0))
                mu_t = base_mu * (1.0 + 0.25 * np.cos(2 * np.pi * t / 15.0))

            params = np.array([m_t, mu_t, base_e, base_g], dtype=np.float32)

            frame = sim.render_frame(state)
            ep_frames.append(frame)
            ep_states.append(state.copy())
            ep_params.append(params)

            state = sim.step_physics(state, params)

        all_videos.append(ep_frames)
        all_states.append(ep_states)
        all_params.append(ep_params)

    return {
        "videos": torch.tensor(np.array(all_videos), dtype=torch.float32), # (N, T, 3, 64, 64)
        "states": torch.tensor(np.array(all_states), dtype=torch.float32), # (N, T, 4)
        "params": torch.tensor(np.array(all_params), dtype=torch.float32)  # (N, T, 4)
    }

if __name__ == "__main__":
    out_dir = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
    os.makedirs(out_dir, exist_ok=True)
    print("Generating pure-vision physics video datasets...")
    train_data = generate_video_dataset(num_episodes=40, seq_len=25, drift_mode="nominal", seed=10)
    test_nominal = generate_video_dataset(num_episodes=15, seq_len=25, drift_mode="nominal", seed=20)
    test_heavy_mu = generate_video_dataset(num_episodes=15, seq_len=25, drift_mode="heavy_friction_ood", seed=30)
    test_low_g = generate_video_dataset(num_episodes=15, seq_len=25, drift_mode="low_gravity_elastic_ood", seed=40)

    torch.save(train_data, os.path.join(out_dir, "vision_train_nominal.pt"))
    torch.save(test_nominal, os.path.join(out_dir, "vision_test_nominal.pt"))
    torch.save(test_heavy_mu, os.path.join(out_dir, "vision_test_heavy_mu.pt"))
    torch.save(test_low_g, os.path.join(out_dir, "vision_test_low_g.pt"))
    print(f"Generated datasets. Video tensor shape: {train_data['videos'].shape}")
