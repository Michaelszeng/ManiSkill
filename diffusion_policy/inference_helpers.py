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
from mani_skill.utils.wrappers.record import RecordEpisode


class DiffusionPolicy:
    def __init__(self, env, checkpoint_path, control_mode, obs_mode, state_mode, n_action_steps, device="cuda"):
        self.env = env
        self.control_mode = control_mode
        self.obs_mode = obs_mode
        self.state_mode = state_mode
        self.n_action_steps = n_action_steps
        self.device = device

        self.load_policy(checkpoint_path)

        # State observation slicing (for flat state observations)
        self.qpos_slice = (0, 7)  # obs[:, 0:7]
        self.qvel_slice = (7, 14)  # obs[:, 7:14]
        self.tee_pose_slice = (14, 21)  # obs[:, 14:21] - Tee position + quaternion
        self.tcp_pose_slice = None  # Only needed if STATE_MODE = "tcp_pose"

        # Create observation history buffers
        self.state_buffer = deque(maxlen=self.cfg.n_obs_steps)
        self.image_buffers = {key: deque(maxlen=self.cfg.n_obs_steps) for key in self.camera_keys}

        # For state obs mode, maintain buffer for tee_pose too
        if self.obs_mode == "state":
            self.tee_pose_buffer = deque(maxlen=self.cfg.n_obs_steps)

        # Create action buffer
        self.action_queue = deque()

    def load_policy(self, checkpoint_path, device="cuda"):
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

        self.policy = policy
        self.cfg = cfg

        # Find camera keys and expected image sizes from config
        camera_keys = []
        camera_shapes = {}  # Store expected (C, H, W) for each camera
        if self.obs_mode != "state":
            for key, val in cfg.shape_meta.obs.items():
                if val.get("type") == "rgb":
                    camera_keys.append(key)
                    camera_shapes[key] = val["shape"]  # [C, H, W]
        print(f"Using cameras: {camera_keys}")
        self.camera_keys = camera_keys

        # Action indexing: skip first (cfg.n_obs_steps - 1) actions for temporal alignment
        self.action_start_idx = self.cfg.n_obs_steps - 1
        self.action_end_idx = self.action_start_idx + self.n_action_steps

    def extract_state(self, obs, state_mode, obs_mode):
        """
        Extract robot state from observation based on mode. Used in state mode only.

        Args:
            obs: ManiSkill observation (dict for state_dict/rgbd modes, or flat tensor for state mode)
            state_mode: One of "qpos", "qpos_qvel", "tcp_pose"
            obs_mode: The observation mode used ("state", "state_dict", "rgbd", etc.)

        Returns:
            state: numpy array of robot state
        """
        # When obs_mode="state", ManiSkill returns a flattened tensor directly
        # Extract specific components based on state_mode
        if obs_mode == "state":
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = np.array(obs)

            # Squeeze batch dimension if present (e.g., [1, state_dim] -> [state_dim])
            if obs_np.ndim == 2 and obs_np.shape[0] == 1:
                obs_np = obs_np.squeeze(0)

            # Extract relevant slice based on state_mode
            if state_mode == "qpos_qvel":
                qpos = obs_np[..., self.qpos_slice[0] : self.qpos_slice[1]]
                qvel = obs_np[..., self.qvel_slice[0] : self.qvel_slice[1]]
                return np.concatenate([qpos, qvel], axis=-1)
            elif state_mode == "qpos":
                return obs_np[..., self.qpos_slice[0] : self.qpos_slice[1]]
            elif state_mode == "tcp_pose":
                if self.tcp_pose_slice is None:
                    raise ValueError("TCP_POSE_SLICE must be specified for tcp_pose mode")
                return obs_np[..., self.tcp_pose_slice[0] : self.tcp_pose_slice[1]]
            else:
                raise ValueError(f"Unknown state mode: {state_mode}")

        # For state_dict or visual modes, extract from nested dictionary
        else:
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

    def extract_tee_pose(self, obs, obs_mode):
        """
        Extract Tee pose from observation. Used in state mode only.

        Args:
            obs: ManiSkill observation (dict for state_dict/rgbd modes, or flat tensor for state mode)
            obs_mode: The observation mode used ("state", "state_dict", "rgbd", etc.)

        Returns:
            tee_pose: numpy array of Tee pose [position(3) + quaternion(4)] = 7 dims
        """
        # When obs_mode="state", extract from flat observation
        if obs_mode == "state":
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = np.array(obs)

            # Squeeze batch dimension if present (e.g., [1, state_dim] -> [state_dim])
            if obs_np.ndim == 2 and obs_np.shape[0] == 1:
                obs_np = obs_np.squeeze(0)

            return obs_np[..., self.tee_pose_slice[0] : self.tee_pose_slice[1]]

        # For state_dict or visual modes, try to extract from nested structure
        # This depends on how the observation is structured in non-state modes
        if "extra" in obs and "tee_pose" in obs["extra"]:
            tee_pose = obs["extra"]["tee_pose"]
            if isinstance(tee_pose, torch.Tensor):
                return tee_pose.cpu().numpy()
            return np.array(tee_pose)
        else:
            raise ValueError(
                "Cannot extract tee_pose from observation. "
                "Tee pose should be in obs['extra']['tee_pose'] for non-state modes."
            )

    def extract_and_process_image(self, obs, camera_key):
        """
        Extract and process camera image from observation. Used in rgbd mode only.

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

    def run_one_episode(self, seed):
        obs, _ = self.env.reset(seed=seed)

        state = self.extract_state(obs, self.state_mode, self.obs_mode)
        images = {key: self.extract_and_process_image(obs, key) for key in self.camera_keys}
        if self.obs_mode == "state":
            tee_pose = self.extract_tee_pose(obs, self.obs_mode)
        print(f"State shape: {state.shape}, mode: {self.state_mode}")

        # Fill buffers with initial observation
        for _ in range(self.cfg.n_obs_steps):
            self.state_buffer.append(state)
            if self.obs_mode == "state":
                self.tee_pose_buffer.append(tee_pose)
            for camera_key in self.camera_keys:
                self.image_buffers[camera_key].append(images[camera_key])

        episode_reward = 0
        step = 0

        terminated = False
        truncated = False
        while not (terminated or truncated):
            # Predict new actions when queue is empty
            if len(self.action_queue) == 0:
                # Prepare observation dict for policy
                obs_dict = {
                    "obs": {
                        "agent_pos": torch.from_numpy(np.stack(list(self.state_buffer), axis=0))
                        .unsqueeze(0)
                        .to(self.device),
                    }
                }

                # Add tee_pose to observation dict
                if self.obs_mode == "state":
                    obs_dict["obs"]["tee_pose"] = (
                        torch.from_numpy(np.stack(list(self.tee_pose_buffer), axis=0)).unsqueeze(0).to(self.device)
                    )

                # Add camera images to observation dict
                for camera_key in self.camera_keys:
                    stacked_images = np.stack(list(self.image_buffers[camera_key]), axis=0)
                    obs_dict["obs"][camera_key] = torch.from_numpy(stacked_images).unsqueeze(0).to(self.policy.device)

                ########################################################################################################
                ### Debug: Display images from observation
                ########################################################################################################
                if False and self.obs_mode == "rgbd":  # Set to True to enable image visualization
                    if self.debug_fig is None:
                        # Create figure on first use
                        self.debug_fig, axes = plt.subplots(
                            self.cfg.n_obs_steps,
                            len(self.camera_keys),
                            figsize=(5 * len(self.camera_keys), 5 * self.cfg.n_obs_steps),
                        )
                        if len(self.camera_keys) == 1:
                            axes = axes.reshape(-1, 1)
                        self.debug_fig.canvas.manager.set_window_title("Policy Observations")
                    else:
                        # Clear existing figure
                        axes = self.debug_fig.axes
                        for ax in axes:
                            ax.clear()
                        # Reshape axes back to grid
                        axes = np.array(axes).reshape(self.cfg.n_obs_steps, len(self.camera_keys))

                    for t in range(self.cfg.n_obs_steps):
                        for cam_idx, camera_key in enumerate(self.camera_keys):
                            img = list(self.image_buffers[camera_key])[t]
                            # img is in C x H x W format, normalized to [0, 1]
                            img_display = np.transpose(img, (1, 2, 0))  # Convert to H x W x C for display
                            axes[t, cam_idx].imshow(img_display)
                            axes[t, cam_idx].set_title(f"{camera_key} (t={t}, step={step})")
                            axes[t, cam_idx].axis("off")
                    self.debug_fig.tight_layout()
                    self.debug_fig.canvas.draw()
                    self.debug_fig.canvas.flush_events()
                    plt.pause(0.001)  # Small pause to allow GUI to update
                ########################################################################################################

                # Run policy inference
                with torch.no_grad():
                    result = self.policy.predict_action(obs_dict, use_DDIM=True)
                    actions = result["action_pred"][0].cpu().numpy()

                # Add actions to action queue using proper indexing (skip first cfg.n_obs_steps-1 for temporal alignment)
                for action in actions[self.action_start_idx : self.action_end_idx]:
                    self.action_queue.append(action)

            # Execute next action from action queue
            action = self.action_queue.popleft()
            # print(f"action: {action}")
            # Add batch dimension for vectorized environment (num_envs=1 expects shape [1, action_dim])
            if action.ndim == 1:
                action = action[np.newaxis, :]  # [action_dim] -> [1, action_dim]

            obs, reward, terminated, truncated, info = self.env.step(action)

            # Update observation buffers
            state = self.extract_state(obs, self.state_mode, self.obs_mode)
            self.state_buffer.append(state)
            if self.obs_mode == "state":
                tee_pose = self.extract_tee_pose(obs, self.obs_mode)
                self.tee_pose_buffer.append(tee_pose)
            for camera_key in self.camera_keys:
                rgb = self.extract_and_process_image(obs, camera_key)
                self.image_buffers[camera_key].append(rgb)

            # Accumulate reward
            episode_reward += reward if isinstance(reward, float) else reward.item()

            step += 1
            self.env.render()

            # Print detailed success metrics
            # SUCCESS CRITERIA: The T block must cover ≥90% of the goal T's 2D area
            # This is measured by projecting both T shapes onto a 64x64 grid and
            # calculating: intersection_area / goal_area >= 0.90
            # The T must be correctly positioned, rotated, and aligned within ~10% tolerance
            success = (
                info.get("success", False)
                if isinstance(info.get("success", False), torch.Tensor)
                else info.get("success", False).item()
            )

            # Calculate and display intersection percentage
            # Access the base environment to get intersection data
            base_env = self.env.unwrapped
            intersection_ratio = base_env.pseudo_render_intersection()
            if isinstance(intersection_ratio, torch.Tensor):
                intersection_ratio = intersection_ratio.item()

            if step % 25 == 0:
                print(f"Step {step}: intersection={intersection_ratio:.4f}, success={success}")

            if success:
                print("✓ SUCCESS!")
                break
        return episode_reward, success, intersection_ratio, step
