"""
Minimal example showing the core concepts of loading and running
a trained diffusion policy model in ManiSkill environments.
"""

import sys
from collections import deque
from pathlib import Path

import dill
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import ManiSkill to register environments
import mani_skill.envs

# Add your diffusion policy repo to Python path
DIFFUSION_POLICY_PATH = Path("~/diffusion-policy").expanduser()
sys.path.insert(0, str(DIFFUSION_POLICY_PATH))
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Configuration
ENV_ID = "PushT-v1"
NUM_FAILURES_TO_STOP = 10
CHECKPOINT = "/home/michzeng/diffusion-policy/data/outputs/maniskill/2_obs/checkpoints/latest.ckpt"

# Must match training configuration
STATE_MODE = "qpos_qvel"  # "qpos", "qpos_qvel", "tcp_pose"
N_ACTION_STEPS = 8  # Action horizon: number of actions to execute before getting new prediction


def extract_state(obs, state_mode):
    """
    Extract robot state from observation based on mode.

    Args:
        obs: ManiSkill observation dict
        state_mode: One of "qpos", "qpos_qvel", "tcp_pose"

    Returns:
        state: numpy array of robot state
    """
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
    """
    Extract and process camera image from observation.

    Args:
        obs: ManiSkill observation dict
        camera_key: Name of camera to extract

    Returns:
        rgb: Processed image in C x H x W format, normalized to [0, 1]
    """
    # Extract camera image from ManiSkill observation
    rgb = obs["sensor_data"][camera_key]["rgb"]

    # Convert to numpy if needed
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    # Squeeze batch dimension if present: [1, H, W, C] -> [H, W, C]
    if rgb.shape[0] == 1:
        rgb = rgb.squeeze(0)

    # Transpose from H x W x C to C x H x W
    if len(rgb.shape) == 3 and rgb.shape[-1] in [3, 4]:
        rgb = np.transpose(rgb, (2, 0, 1))

    # Take only RGB channels (in case of RGBA with C=4)
    if rgb.shape[0] == 4:
        rgb = rgb[:3]

    # Normalize to [0, 1] range
    rgb = rgb.astype(np.float32) / 255.0

    return rgb


def load_policy(checkpoint_path, device="cuda"):
    """Load a trained diffusion policy from checkpoint"""
    checkpoint_path = Path(checkpoint_path).expanduser()

    # Load checkpoint
    payload = torch.load(checkpoint_path, pickle_module=dill, map_location=device, weights_only=False)
    cfg = payload["cfg"]

    # Create workspace and load model
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Load normalizer
    normalizer_path = checkpoint_path.parent.parent / "normalizer.pt"
    normalizer = torch.load(normalizer_path, map_location=device, weights_only=False)

    # Get policy (use EMA model if available)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()

    return policy, cfg


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Base seed for reproducibility (None to disable)
    # Episode N uses seed = SEED + N
    # To reproduce episode 2, set SEED=42 and run episodes 0,1,2 or set SEED=44 and run 1 episode
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Load policy
    print("Loading policy...")
    policy, cfg = load_policy(CHECKPOINT, device=DEVICE)
    n_obs_steps = cfg.n_obs_steps  # Number of observation history steps
    n_action_steps = cfg.n_action_steps  # Number of action steps to predict
    print(f"Policy config: obs_steps={n_obs_steps}, action_steps={n_action_steps}")
    print(f"Action horizon (executing first N actions): {N_ACTION_STEPS}")

    # Action indexing: skip first (n_obs_steps - 1) actions for temporal alignment
    action_start_idx = n_obs_steps - 1
    action_end_idx = action_start_idx + N_ACTION_STEPS
    print(f"Action indexing: taking actions[{action_start_idx}:{action_end_idx}] from predictions")

    # Enable interactive plotting mode
    plt.ion()

    # Create environment
    print(f"Creating environment: {ENV_ID}")
    env = gym.make(
        ENV_ID,
        obs_mode="rgbd",  # Get both state and camera images
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        max_episode_steps=300,
    )

    # Find camera keys and expected image sizes from config
    camera_keys = []
    camera_shapes = {}  # Store expected (C, H, W) for each camera
    for key, val in cfg.shape_meta.obs.items():
        if val.get("type") == "rgb":
            camera_keys.append(key)
            camera_shapes[key] = val["shape"]  # [C, H, W]
    print(f"Using cameras: {camera_keys}")

    # Run episodes
    failed_episodes = []
    episode = 0
    debug_fig = None  # Reusable figure for image debugging
    while len(failed_episodes) < NUM_FAILURES_TO_STOP:
        # Seed each episode deterministically (episode 0 uses SEED, episode 1 uses SEED+1, etc.)
        episode_seed = SEED + episode
        print(f"\n=== Episode {episode + 1} (seed={episode_seed}) ===")
        obs, _ = env.reset(seed=episode_seed)

        # Create observation history buffers
        state_buffer = deque(maxlen=n_obs_steps)
        image_buffers = {key: deque(maxlen=n_obs_steps) for key in camera_keys}

        # Initialize buffers with first observation
        # Extract once, then fill buffer with n_obs_steps copies
        state = extract_state(obs, STATE_MODE)
        images = {key: extract_and_process_image(obs, key) for key in camera_keys}

        print(f"State shape: {state.shape}, mode: {STATE_MODE}")

        # Fill buffers with initial observation
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
                # Prepare observation dict for policy
                obs_dict = {
                    "obs": {
                        "agent_pos": torch.from_numpy(np.stack(list(state_buffer), axis=0)).unsqueeze(0).to(DEVICE),
                    }
                }
                print(f"obs_dict.agent_pos: {obs_dict['obs']['agent_pos']}")

                # Add camera images
                for camera_key in camera_keys:
                    stacked_images = np.stack(list(image_buffers[camera_key]), axis=0)
                    obs_dict["obs"][camera_key] = torch.from_numpy(stacked_images).unsqueeze(0).to(DEVICE)

                ########################################################################################################
                ### Debug: Display images from observation
                ########################################################################################################
                if True:  # Set to True to enable image visualization
                    if debug_fig is None:
                        # Create figure on first use
                        debug_fig, axes = plt.subplots(
                            n_obs_steps, len(camera_keys), figsize=(5 * len(camera_keys), 5 * n_obs_steps)
                        )
                        if len(camera_keys) == 1:
                            axes = axes.reshape(-1, 1)
                        debug_fig.canvas.manager.set_window_title("Policy Observations")
                    else:
                        # Clear existing figure
                        axes = debug_fig.axes
                        for ax in axes:
                            ax.clear()
                        # Reshape axes back to grid
                        axes = np.array(axes).reshape(n_obs_steps, len(camera_keys))

                    for t in range(n_obs_steps):
                        for cam_idx, camera_key in enumerate(camera_keys):
                            img = list(image_buffers[camera_key])[t]
                            # img is in C x H x W format, normalized to [0, 1]
                            img_display = np.transpose(img, (1, 2, 0))  # Convert to H x W x C for display
                            axes[t, cam_idx].imshow(img_display)
                            axes[t, cam_idx].set_title(f"{camera_key} (t={t}, step={step})")
                            axes[t, cam_idx].axis("off")
                    debug_fig.tight_layout()
                    debug_fig.canvas.draw()
                    debug_fig.canvas.flush_events()
                    plt.pause(0.001)  # Small pause to allow GUI to update
                ########################################################################################################

                # Run policy inference
                with torch.no_grad():
                    result = policy.predict_action(obs_dict, use_DDIM=True)
                    actions = result["action"][0].cpu().numpy()

                # Add actions to queue using proper indexing (skip first n_obs_steps-1 for temporal alignment)
                for action in actions[action_start_idx:action_end_idx]:
                    action_queue.append(action)

            # Execute next action
            action = action_queue.popleft()
            obs, reward, terminated, truncated, info = env.step(action)

            # Update observation buffers
            state = extract_state(obs, STATE_MODE)
            state_buffer.append(state)
            for camera_key in camera_keys:
                rgb = extract_and_process_image(obs, camera_key)
                image_buffers[camera_key].append(rgb)

            # Convert reward to float if it's a tensor
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            episode_reward += reward

            step += 1

            env.render()

            if info.get("success", False):
                break

        success = info.get("success", False)
        # Convert success to bool if it's a tensor
        if isinstance(success, torch.Tensor):
            success = success.item()
        if not success:
            failed_episodes.append(episode_seed)
        print(f"Episode finished: reward={episode_reward:.3f}, steps={step}, success={success}, seed={episode_seed}")

        episode += 1

    env.close()
    print("\nDone!")
    print(f"Failed episodes: {failed_episodes}")


if __name__ == "__main__":
    main()
