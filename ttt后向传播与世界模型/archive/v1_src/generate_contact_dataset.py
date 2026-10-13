# High-Fidelity 224x224 Physical Contact & Collision Video Generator
# Generates (N, T, 3, 224, 224) RGB video clips for V-JEPA evaluation.
# Implements:
# - Elastic/Inelastic rigid-body collisions with impulses (Delta v)
# - Continuous and OOD friction/mass parameter variations
# - Two-Phase protocol: Initial contact (Warmup) -> Post-collision slide (Rollout)

import numpy as np
import torch
from PIL import Image, ImageDraw
import os

class ContactPhysicsRenderer:
    def __init__(self, width: int = 224, height: int = 224, dt: float = 0.05):
        self.width = width
        self.height = height
        self.dt = dt
        self.r1 = 14.0 # Active moving sphere radius
        self.r2 = 14.0 # Target obstacle sphere radius

    def simulate_episode(self, m1: float, m2: float, mu: float, e: float, steps: int = 24):
        # Active sphere starts from left, moving towards target sphere in center
        # pos = [x, y], vel = [vx, vy]
        p1 = np.array([25.0, 112.0], dtype=np.float32)
        v1 = np.array([np.random.uniform(55.0, 75.0), 0.0], dtype=np.float32)

        # Target sphere initially stationary near center
        p2 = np.array([105.0, 112.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0], dtype=np.float32)

        trajectory_p1 = []
        trajectory_p2 = []
        collision_occurred = False

        for t in range(steps):
            trajectory_p1.append(p1.copy())
            trajectory_p2.append(p2.copy())

            # Friction deceleration
            s1 = np.linalg.norm(v1)
            if s1 > 1e-3:
                v1 -= (mu * 9.81 * (v1 / s1)) * self.dt
            else:
                v1 = np.zeros(2, dtype=np.float32)

            s2 = np.linalg.norm(v2)
            if s2 > 1e-3:
                v2 -= (mu * 9.81 * (v2 / s2)) * self.dt
            else:
                v2 = np.zeros(2, dtype=np.float32)

            # Check collision between sphere 1 and sphere 2
            dist = np.linalg.norm(p2 - p1)
            min_dist = self.r1 + self.r2
            if dist < min_dist and not collision_occurred:
                collision_occurred = True
                # Normal impulse collision dynamics:
                # v1' = (m1 - e*m2)/(m1 + m2) * v1 + (1 + e)*m2/(m1 + m2) * v2
                # v2' = (1 + e)*m1/(m1 + m2) * v1 + (m2 - e*m1)/(m1 + m2) * v2
                v1_new = ((m1 - e * m2) / (m1 + m2)) * v1 + (((1.0 + e) * m2) / (m1 + m2)) * v2
                v2_new = (((1.0 + e) * m1) / (m1 + m2)) * v1 + ((m2 - e * m1) / (m1 + m2)) * v2
                v1 = v1_new
                v2 = v2_new

            # Position updates
            p1 += v1 * self.dt
            p2 += v2 * self.dt

        return trajectory_p1, trajectory_p2

    def render_frame(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        # High contrast scene for ViT patches (3, 224, 224)
        img = Image.new("RGB", (self.width, self.height), color=(28, 32, 42))
        draw = ImageDraw.Draw(img)

        # Draw physical ground arena boundary lines
        draw.line([(0, 60), (self.width, 60)], fill=(65, 75, 95), width=2)
        draw.line([(0, 164), (self.width, 164)], fill=(65, 75, 95), width=2)

        # Draw Sphere 1 (Active collider: Coral Red)
        x1, y1 = p1[0], p1[1]
        draw.ellipse([x1 - self.r1, y1 - self.r1, x1 + self.r1, y1 + self.r1], fill=(235, 70, 60), outline=(255, 180, 170))

        # Draw Sphere 2 (Target object: Cyan Blue)
        x2, y2 = p2[0], p2[1]
        draw.ellipse([x2 - self.r2, y2 - self.r2, x2 + self.r2, y2 + self.r2], fill=(45, 170, 230), outline=(180, 235, 255))

        arr = np.array(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))

def generate_contact_dataset(num_episodes: int = 25, steps: int = 20, mode: str = "nominal", seed: int = 42):
    np.random.seed(seed)
    renderer = ContactPhysicsRenderer(width=224, height=224, dt=0.06)

    all_videos = []
    all_params = []

    for ep in range(num_episodes):
        if mode == "nominal":
            m1 = np.random.uniform(1.0, 1.8)
            m2 = 1.0 # reference target mass
            mu = np.random.uniform(0.3, 0.7)
            e = 0.8
        elif mode == "ood_heavy_mass":
            m1 = np.random.uniform(3.0, 5.0) # 3x-5x heavier collider
            m2 = 1.0
            mu = np.random.uniform(0.3, 0.7)
            e = 0.8
        elif mode == "ood_high_friction":
            m1 = np.random.uniform(1.0, 1.8)
            m2 = 1.0
            mu = np.random.uniform(1.2, 2.2) # extreme rough surface
            e = 0.4

        traj1, traj2 = renderer.simulate_episode(m1=m1, m2=m2, mu=mu, e=e, steps=steps)
        frames = []
        params_seq = []
        for t in range(steps):
            frame = renderer.render_frame(traj1[t], traj2[t])
            frames.append(frame)
            # Annotate mass ratio (m1/m2) and friction coefficient mu
            params_seq.append([m1 / m2, mu, e, 0.0])

        all_videos.append(frames)
        all_params.append(params_seq)

    return {
        "videos": torch.tensor(np.array(all_videos), dtype=torch.float32), # (N, T, 3, 224, 224)
        "params": torch.tensor(np.array(all_params), dtype=torch.float32)  # (N, T, 4)
    }

if __name__ == "__main__":
    out_dir = "/Users/mac/Documents/论文/ttt后向传播与世界模型/data"
    os.makedirs(out_dir, exist_ok=True)
    print("Generating 224x224 Contact & Collision datasets...")
    train_data = generate_contact_dataset(num_episodes=30, steps=20, mode="nominal", seed=101)
    test_nom = generate_contact_dataset(num_episodes=15, steps=20, mode="nominal", seed=202)
    test_mass = generate_contact_dataset(num_episodes=15, steps=20, mode="ood_heavy_mass", seed=303)
    test_fric = generate_contact_dataset(num_episodes=15, steps=20, mode="ood_high_friction", seed=404)

    torch.save(train_data, os.path.join(out_dir, "contact_train_nominal.pt"))
    torch.save(test_nom, os.path.join(out_dir, "contact_test_nominal.pt"))
    torch.save(test_mass, os.path.join(out_dir, "contact_test_heavy_mass.pt"))
    torch.save(test_fric, os.path.join(out_dir, "contact_test_high_friction.pt"))
    print(f"Generated 224x224 contact datasets. Tensor shape: {train_data['videos'].shape}")
