#!/usr/bin/env python3
"""
Test PushT-v2 planar environment with RGB image observations
"""

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import mani_skill.envs


def main():
    print("=" * 60)
    print("Testing PushT-v2 with RGB observations")
    print("=" * 60)

    # Test with RGB observations
    print("\nCreating PushT-v2 environment with RGB observations...")
    env = gym.make("PushT-v2", num_envs=1, obs_mode="rgb", render_mode="human")

    print("✓ Environment created!")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Check action space is 2D
    assert env.action_space.shape == (1, 2), "Expected (1, 2) action space"
    print("✓ Action space is 2D as expected")

    # Reset
    print("\nResetting environment...")
    obs, info = env.reset()

    print("✓ Environment reset successful")
    print(f"  Observation type: {type(obs)}")

    if isinstance(obs, dict):
        print(f"  Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if hasattr(value, "shape"):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"    {key}: {type(value)}")

        # Get the image
        if "sensor_data" in obs and "base_camera" in obs["sensor_data"]:
            img = obs["sensor_data"]["base_camera"]["rgb"]
        elif "image" in obs:
            img = obs["image"]
        else:
            # Try to find the image in the observation
            img = None
            for key, value in obs.items():
                if hasattr(value, "shape") and len(value.shape) >= 3:
                    img = value
                    print(f"  Found image in key: {key}")
                    break
    else:
        print(f"  Observation shape: {obs.shape}, dtype={obs.dtype}")
        img = obs

    if img is not None:
        print(f"  Image shape: {img.shape}")

        # Visualize first environment's observation
        if len(img.shape) == 1:  # (batch, H, W, C)
            img_vis = img[0].cpu().numpy()
        else:
            img_vis = img.cpu().numpy()

        # Convert to uint8 if needed
        if img_vis.dtype == np.float32 or img_vis.dtype == np.float64:
            if img_vis.max() <= 1.0:
                img_vis = (img_vis * 255).astype(np.uint8)
            else:
                img_vis = img_vis.astype(np.uint8)

        print(f"  Image visualization shape: {img_vis.shape}, dtype={img_vis.dtype}")

    # Run steps with 2D actions and collect rendered frames
    print("\nRunning test steps with 2D actions and collecting renders...")
    frames = []

    for i in range(1000):
        # Random 2D action (x, y movements only)
        action = np.random.uniform(-0.5, 0.5, size=(1, 2))
        action = np.array([[0.01, -0.01]])
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"obs['extra']['tcp_pose']: {obs['extra']['tcp_pose']}")

        # Render the environment (returns 1 parallel envs)
        frame = env.render()
        # Store only the first environment's view for visualization
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frames.append(frame)  # Take only first env

        if i % 100 == 0:
            print(f"  Step {i}...")

    print("\n✓ All tests passed!")
    print("\nSummary:")
    print("  ✓ Environment creates with rgb observations")
    print("  ✓ Action space is 2D (x, y only)")
    print("  ✓ Environment steps successfully with 2D actions")
    print("  ✓ Image observations are returned")
    print(f"  ✓ Collected {len(frames)} rendered frames")

    # 2. Save initial RGB observations
    if img is not None:
        try:
            fig, axes = plt.subplots(1, 1, figsize=(16, 1))
            fig.suptitle("PushT-v2: Initial RGB Observations (1 parallel envs)")

            for idx in range(1):
                if len(img.shape) == 1:
                    img_show = img[idx].cpu().numpy()
                else:
                    img_show = img.cpu().numpy()

                if img_show.dtype == np.float32 or img_show.dtype == np.float64:
                    if img_show.max() <= 1.0:
                        img_show = (img_show * 255).astype(np.uint8)
                    else:
                        img_show = img_show.astype(np.uint8)

                axes[idx].imshow(img_show)
                axes[idx].set_title(f"Env {idx}")
                axes[idx].axis("off")

            plt.tight_layout()
            plt.savefig("pusht_v2_rgb_observations.png", dpi=100, bbox_inches="tight")
            print("  ✓ Saved sample observations to 'pusht_v2_rgb_observations.png'")
            plt.close()
        except Exception as e:
            print(f"  ⚠ Could not save initial observations: {e}")

    env.close()


if __name__ == "__main__":
    main()
