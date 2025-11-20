"""
DAGGER Step 1: Run policy inference and record failed episodes
Records trajectories with full environment states for later review
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import dill
import gymnasium as gym
import hydra
import numpy as np
import torch

# Import ManiSkill to register environments
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

# Add your diffusion policy repo to Python path
DIFFUSION_POLICY_PATH = Path("~/diffusion-policy").expanduser()
sys.path.insert(0, str(DIFFUSION_POLICY_PATH))
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Configuration
ENV_ID = "PushT-v1"
NUM_FAILURES_TO_STOP = 2
CHECKPOINT = "~/diffusion-policy/data/outputs/maniskill/2_obs/checkpoints/latest.ckpt"
OUTPUT_DIR = "~/diffusion-policy/dagger_data/maniskill/failures"  # Where to save failed episode recordings

# Must match training configuration
STATE_MODE = "qpos_qvel"  # "qpos", "qpos_qvel", "tcp_pose"


def parse_args():
    parser = argparse.ArgumentParser(description="Record failed policy episodes for DAGGER")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save trajectory recordings (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        help="Path to policy checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=ENV_ID,
        help="Environment ID (default: %(default)s)",
    )
    parser.add_argument(
        "--num-failures",
        type=int,
        default=NUM_FAILURES_TO_STOP,
        help="Number of failed episodes to collect (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--state-mode",
        type=str,
        default=STATE_MODE,
        choices=["qpos", "qpos_qvel", "tcp_pose"],
        help="Robot state mode, must match training config (default: %(default)s)",
    )
    args = parser.parse_args()
    return args


def extract_state(obs, state_mode):
    """Extract robot state from observation based on mode."""
    if state_mode == "qpos_qvel":
        qpos = obs["agent"]["qpos"]
        qvel = obs["agent"]["qvel"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().numpy()
        if isinstance(qvel, torch.Tensor):
            qvel = qvel.cpu().numpy()
        return np.concatenate([qpos, qvel], axis=-1)
    elif state_mode == "qpos":
        state = obs["agent"]["qpos"]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        return state
    elif state_mode == "tcp_pose":
        state = obs["extra"]["tcp_pose"]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        return state
    else:
        raise ValueError(f"Unknown state mode: {state_mode}")


def extract_and_process_image(obs, camera_key):
    """Extract and process camera image from observation."""
    rgb = obs["sensor_data"][camera_key]["rgb"]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if rgb.shape[0] == 1:
        rgb = rgb.squeeze(0)
    if len(rgb.shape) == 3 and rgb.shape[-1] in [3, 4]:
        rgb = np.transpose(rgb, (2, 0, 1))
    if rgb.shape[0] == 4:
        rgb = rgb[:3]
    rgb = rgb.astype(np.float32) / 255.0
    return rgb


def load_policy(checkpoint_path, device="cuda"):
    """Load a trained diffusion policy from checkpoint"""
    checkpoint_path = Path(checkpoint_path).expanduser()
    payload = torch.load(checkpoint_path, pickle_module=dill, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    normalizer_path = checkpoint_path.parent.parent / "normalizer.pt"
    normalizer = torch.load(normalizer_path, map_location=device, weights_only=False)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    return policy, cfg


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load policy
    print("Loading policy...")
    policy, cfg = load_policy(args.checkpoint, device=DEVICE)
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    print(f"Policy config: obs_steps={n_obs_steps}, action_steps={n_action_steps}")

    # Create base environment
    print(f"Creating environment: {args.env_id}")
    base_env = gym.make(
        args.env_id,
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )

    # Wrap with RecordEpisode to save failed trajectories
    output_dir = Path(args.output_dir).expanduser()
    env = RecordEpisode(
        base_env,
        output_dir=str(output_dir),
        save_trajectory=True,
        trajectory_name="trajectory",  # Fixed name instead of timestamp
        save_video=True,
        save_on_reset=False,  # We'll manually save only failed episodes
        record_reward=True,
        record_env_state=True,  # Essential for DAGGER - saves full env state
        source_type="diffusion_policy",
        source_desc="Failed episodes from diffusion policy for DAGGER dataset collection",
    )

    # Find camera keys from config
    camera_keys = []
    camera_shapes = {}
    for key, val in cfg.shape_meta.obs.items():
        if val.get("type") == "rgb":
            camera_keys.append(key)
            camera_shapes[key] = val["shape"]
    print(f"Using cameras: {camera_keys}")

    # Run episodes until we collect enough failures
    failed_episodes = []
    successful_episodes = []
    episode = 0

    print(f"\nCollecting {args.num_failures} failed episodes for DAGGER...")
    print(f"Trajectories will be saved to: {output_dir}")

    while len(failed_episodes) < args.num_failures:
        episode_seed = args.seed + episode
        print(f"\n=== Episode {episode + 1} (seed={episode_seed}) ===")
        obs, _ = env.reset(seed=episode_seed)

        # Initialize observation buffers
        state_buffer = deque(maxlen=n_obs_steps)
        image_buffers = {key: deque(maxlen=n_obs_steps) for key in camera_keys}

        state = extract_state(obs, args.state_mode)
        images = {key: extract_and_process_image(obs, key) for key in camera_keys}

        for _ in range(n_obs_steps):
            state_buffer.append(state)
            for camera_key in camera_keys:
                image_buffers[camera_key].append(images[camera_key])

        action_queue = deque()
        episode_reward = 0
        step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Predict new actions when queue is empty
            if len(action_queue) == 0:
                obs_dict = {
                    "obs": {
                        "agent_pos": torch.from_numpy(np.stack(list(state_buffer), axis=0)).unsqueeze(0).to(DEVICE),
                    }
                }
                for camera_key in camera_keys:
                    stacked_images = np.stack(list(image_buffers[camera_key]), axis=0)
                    obs_dict["obs"][camera_key] = torch.from_numpy(stacked_images).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    result = policy.predict_action(obs_dict, use_DDIM=True)
                    actions = result["action"][0].cpu().numpy()

                for action in actions:
                    action_queue.append(action)

            # Execute next action
            action = action_queue.popleft()
            obs, reward, terminated, truncated, info = env.step(action)

            # Update observation buffers
            state = extract_state(obs, args.state_mode)
            state_buffer.append(state)
            for camera_key in camera_keys:
                rgb = extract_and_process_image(obs, camera_key)
                image_buffers[camera_key].append(rgb)

            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            episode_reward += reward
            step += 1

            if info.get("success", False):
                break

        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()

        # Save trajectory only if failed
        if not success:
            env.flush_trajectory(verbose=True)
            env.flush_video(verbose=True)
            failed_episodes.append((episode_seed, episode_reward, step))
            print(f"❌ FAILED - Saved trajectory (seed={episode_seed})")
        else:
            successful_episodes.append((episode_seed, episode_reward, step))
            print(f"✅ Success (seed={episode_seed})")

        print(f"Episode stats: reward={episode_reward:.3f}, steps={step}, success={success}")
        print(f"Progress: {len(failed_episodes)}/{NUM_FAILURES_TO_STOP} failures collected")

        episode += 1

    env.close()

    print("\n" + "=" * 80)
    print("DAGGER Data Collection Complete!")
    print("=" * 80)
    print(f"\nCollected {len(failed_episodes)} failed episodes")
    print(f"Also ran {len(successful_episodes)} successful episodes")
    print("\nFailed episodes (seed, reward, steps):")
    for seed, reward, steps in failed_episodes:
        print(f"  - Seed {seed}: reward={reward:.3f}, steps={steps}")

    print(f"\nTrajectories saved to: {output_dir}")


if __name__ == "__main__":
    main()
