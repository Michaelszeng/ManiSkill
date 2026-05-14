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

        # Determine number of parallel environments
        self.num_envs = getattr(env, "num_envs", 1)
        print(f"Initializing policy for {self.num_envs} parallel environments")

        self.load_policy(checkpoint_path)

        # State observation slicing (for flat state observations)
        self.qpos_slice = (0, 7)  # obs[:, 0:7]
        self.qvel_slice = (7, 14)  # obs[:, 7:14]
        self.tee_pose_slice = (14, 21)  # obs[:, 14:21] - Tee position + quaternion
        self.tcp_pose_slice = None  # Only needed if STATE_MODE = "tcp_pose"

        # Create observation history buffers (one per environment)
        self.state_buffers = [deque(maxlen=self.cfg.n_obs_steps) for _ in range(self.num_envs)]
        self.image_buffers = [
            {key: deque(maxlen=self.cfg.n_obs_steps) for key in self.camera_keys} for _ in range(self.num_envs)
        ]

        # For state obs mode, maintain buffer for tee_pose too
        if self.obs_mode == "state":
            self.tee_pose_buffers = [deque(maxlen=self.cfg.n_obs_steps) for _ in range(self.num_envs)]

        # Create action buffers (one per environment)
        self.action_queues = [deque() for _ in range(self.num_envs)]

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

    def extract_state(self, obs, state_mode, obs_mode, env_idx=None):
        """
        Extract robot state from observation based on mode. Used in state mode only.

        Args:
            obs: ManiSkill observation (dict for state_dict/rgbd modes, or flat tensor for state mode)
            state_mode: One of "qpos", "qpos_qvel", "tcp_pose"
            obs_mode: The observation mode used ("state", "state_dict", "rgbd", etc.)
            env_idx: Index of environment to extract from (for vectorized envs). If None, returns all envs.

        Returns:
            state: numpy array of robot state (num_envs, state_dim) or (state_dim,) if env_idx is specified
        """
        # When obs_mode="state", ManiSkill returns a flattened tensor directly
        # Extract specific components based on state_mode
        if obs_mode == "state":
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = np.array(obs)

            # Handle batch dimension: keep it for vectorized envs
            # obs_np shape: (num_envs, state_dim) for num_envs > 1, or (state_dim,) for num_envs = 1
            if obs_np.ndim == 1:
                # Single env, no batch dimension - reshape to (1, state_dim)
                obs_np = obs_np[np.newaxis, :]

            # Extract specific environment if requested
            if env_idx is not None:
                obs_np = obs_np[env_idx]

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

                # Handle batch dimension
                if qpos.ndim == 1:
                    qpos = qpos[np.newaxis, :]
                    qvel = qvel[np.newaxis, :]
                if env_idx is not None:
                    qpos = qpos[env_idx]
                    qvel = qvel[env_idx]

                return np.concatenate([qpos, qvel], axis=-1)
            elif state_mode == "qpos":
                state = obs["agent"]["qpos"]
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()

                if state.ndim == 1:
                    state = state[np.newaxis, :]
                if env_idx is not None:
                    state = state[env_idx]
                return state
            elif state_mode == "tcp_pose":
                state = obs["extra"]["tcp_pose"]
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()

                if state.ndim == 1:
                    state = state[np.newaxis, :]
                if env_idx is not None:
                    state = state[env_idx]
                return state
            else:
                raise ValueError(f"Unknown state mode: {state_mode}")

    def extract_tee_pose(self, obs, obs_mode, env_idx=None):
        """
        Extract Tee pose from observation. Used in state mode only.

        Args:
            obs: ManiSkill observation (dict for state_dict/rgbd modes, or flat tensor for state mode)
            obs_mode: The observation mode used ("state", "state_dict", "rgbd", etc.)
            env_idx: Index of environment to extract from (for vectorized envs). If None, returns all envs.

        Returns:
            tee_pose: numpy array of Tee pose [position(3) + quaternion(4)] = 7 dims
        """
        # When obs_mode="state", extract from flat observation
        if obs_mode == "state":
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = np.array(obs)

            # Handle batch dimension
            if obs_np.ndim == 1:
                obs_np = obs_np[np.newaxis, :]

            if env_idx is not None:
                obs_np = obs_np[env_idx]

            return obs_np[..., self.tee_pose_slice[0] : self.tee_pose_slice[1]]

        # For state_dict or visual modes, try to extract from nested structure
        # This depends on how the observation is structured in non-state modes
        if "extra" in obs and "tee_pose" in obs["extra"]:
            tee_pose = obs["extra"]["tee_pose"]
            if isinstance(tee_pose, torch.Tensor):
                tee_pose = tee_pose.cpu().numpy()
            else:
                tee_pose = np.array(tee_pose)

            if tee_pose.ndim == 1:
                tee_pose = tee_pose[np.newaxis, :]
            if env_idx is not None:
                tee_pose = tee_pose[env_idx]
            return tee_pose
        else:
            raise ValueError(
                "Cannot extract tee_pose from observation. "
                "Tee pose should be in obs['extra']['tee_pose'] for non-state modes."
            )

    def extract_and_process_image(self, obs, camera_key, env_idx=None):
        """
        Extract and process camera image from observation. Used in rgbd mode only.

        Args:
            obs: ManiSkill observation dict
            camera_key: Name of camera to extract
            env_idx: Index of environment to extract from (for vectorized envs). If None, returns all envs.

        Returns:
            rgb: Processed image in C x H x W format (or num_envs x C x H x W), normalized to [0, 1]
        """
        # Extract camera image from ManiSkill observation
        rgb = obs["sensor_data"][camera_key]["rgb"]

        # Convert to numpy if needed
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()

        # Handle batch dimension
        # rgb shape: (num_envs, H, W, C) for num_envs > 1, or (H, W, C) for num_envs = 1
        is_batched = len(rgb.shape) == 4

        if not is_batched:
            # Single env: (H, W, C) -> add batch dim -> (1, H, W, C)
            rgb = rgb[np.newaxis, ...]

        # Extract specific environment if requested
        if env_idx is not None:
            rgb = rgb[env_idx]  # (H, W, C)
            # Transpose from H x W x C to C x H x W
            rgb = np.transpose(rgb, (2, 0, 1))
        else:
            # Keep batch: (num_envs, H, W, C) -> (num_envs, C, H, W)
            rgb = np.transpose(rgb, (0, 3, 1, 2))

        # # Take only RGB channels (in case of RGBA with C=4)
        # if env_idx is not None:
        #     if rgb.shape[0] == 4:
        #         rgb = rgb[:3]
        # else:
        #     if rgb.shape[1] == 4:
        #         rgb = rgb[:, :3]

        # Normalize to [0, 1] range
        rgb = rgb.astype(np.float32) / 255.0

        return rgb

    def run_one_episode(self, seed, record_video: bool = False):
        """Run one episode (or num_envs episodes in parallel for vectorized envs).

        If ``record_video=True``, the env must be in render_mode="rgb_array";
        per-step frames will be accumulated per env and made available as
        ``self.last_frames`` (list of length num_envs, each entry a list of
        (H, W, 3) uint8 frames). Reset to None when ``record_video=False``.
        """
        self.last_frames = None
        if record_video:
            if self.env.render_mode != "rgb_array":
                raise RuntimeError(
                    f"record_video=True requires render_mode='rgb_array', got {self.env.render_mode!r}"
                )
            frame_buffers = [[] for _ in range(self.num_envs)]
        # For vectorized envs, pass list of seeds
        if self.num_envs > 1:
            seeds = [seed + i for i in range(self.num_envs)]
            obs, _ = self.env.reset(seed=seeds)
        else:
            obs, _ = self.env.reset(seed=seed)

        # Extract observations for all environments at once (more efficient)
        states_all = self.extract_state(obs, self.state_mode, self.obs_mode, env_idx=None)
        images_all = {key: self.extract_and_process_image(obs, key, env_idx=None) for key in self.camera_keys}
        if self.obs_mode == "state":
            tee_poses_all = self.extract_tee_pose(obs, self.obs_mode, env_idx=None)

        # Fill buffers with initial observation for this environment
        for env_idx in range(self.num_envs):
            state = states_all[env_idx] if self.num_envs > 1 else states_all
            images = {
                key: images_all[key][env_idx] if self.num_envs > 1 else images_all[key] for key in self.camera_keys
            }
            if self.obs_mode == "state":
                tee_pose = tee_poses_all[env_idx] if self.num_envs > 1 else tee_poses_all

            for _ in range(self.cfg.n_obs_steps):
                self.state_buffers[env_idx].append(state)
                if self.obs_mode == "state":
                    self.tee_pose_buffers[env_idx].append(tee_pose)
                for camera_key in self.camera_keys:
                    self.image_buffers[env_idx][camera_key].append(images[camera_key])

        # Track per-environment episode state
        episode_rewards = [0.0] * self.num_envs
        steps = [0] * self.num_envs
        active_envs = [True] * self.num_envs  # Track which environments are still running/not terminated
        final_intersection_ratios = [0.0] * self.num_envs
        successes = [False] * self.num_envs

        step = 0
        while any(active_envs):
            # Check which environments need new actions due to queue emptying
            envs_need_actions = [i for i in range(self.num_envs) if active_envs[i] and len(self.action_queues[i]) == 0]

            if len(envs_need_actions) > 0:
                # Prepare batched observation dict for policy (only for active envs that need actions)
                # Stack state observations from each environment's buffer
                state_batch = []
                for env_idx in envs_need_actions:
                    state_batch.append(np.stack(list(self.state_buffers[env_idx]), axis=0))
                state_batch = np.stack(state_batch, axis=0)  # (batch_size, n_obs_steps, state_dim)

                obs_dict = {
                    "obs": {
                        "agent_pos": torch.from_numpy(state_batch).to(self.device),
                    }
                }

                # Add tee_pose to observation dict
                if self.obs_mode == "state":
                    tee_pose_batch = []
                    for env_idx in envs_need_actions:
                        tee_pose_batch.append(np.stack(list(self.tee_pose_buffers[env_idx]), axis=0))
                    tee_pose_batch = np.stack(tee_pose_batch, axis=0)
                    obs_dict["obs"]["tee_pose"] = torch.from_numpy(tee_pose_batch).to(self.device)

                # Add camera images to observation dict
                for camera_key in self.camera_keys:
                    image_batch = []
                    for env_idx in envs_need_actions:
                        image_batch.append(np.stack(list(self.image_buffers[env_idx][camera_key]), axis=0))
                    image_batch = np.stack(image_batch, axis=0)  # (batch_size, n_obs_steps, C, H, W)
                    obs_dict["obs"][camera_key] = torch.from_numpy(image_batch).to(self.policy.device)

                # Run policy inference
                with torch.no_grad():
                    result = self.policy.predict_action(obs_dict, use_DDIM=True)
                    actions_batch = result["action_pred"].cpu().numpy()  # (batch_size, pred_horizon, action_dim)

                # Add actions to action queues using proper indexing
                for i, env_idx in enumerate(envs_need_actions):
                    actions = actions_batch[i]  # (pred_horizon, action_dim)
                    for action in actions[self.action_start_idx : self.action_end_idx]:
                        self.action_queues[env_idx].append(action)

            # Execute next action from action queues for all active environments
            actions_to_execute = []
            for env_idx in range(self.num_envs):
                if active_envs[env_idx] and len(self.action_queues[env_idx]) > 0:
                    action = self.action_queues[env_idx].popleft()
                    actions_to_execute.append(action)
                else:
                    # Pad with zeros for inactive environments (they won't be used but need correct shape)
                    actions_to_execute.append(
                        np.zeros_like(actions_to_execute[0]) if actions_to_execute else np.zeros(2)
                    )

            # Stack actions for vectorized step
            if self.num_envs == 1:
                actions_array = actions_to_execute[0][np.newaxis, :]  # (1, action_dim)
            else:
                actions_array = np.stack(actions_to_execute, axis=0)  # (num_envs, action_dim)

            obs, reward, terminated, truncated, info = self.env.step(actions_array)

            # Extract observations for all environments at once
            states_all = self.extract_state(obs, self.state_mode, self.obs_mode, env_idx=None)
            if self.obs_mode == "state":
                tee_poses_all = self.extract_tee_pose(obs, self.obs_mode, env_idx=None)
            images_all = {key: self.extract_and_process_image(obs, key, env_idx=None) for key in self.camera_keys}

            # Update observation buffers and track results for each environment
            for env_idx in range(self.num_envs):
                if not active_envs[env_idx]:
                    continue

                # Extract this environment's data from the batch
                state = states_all[env_idx] if self.num_envs > 1 else states_all
                self.state_buffers[env_idx].append(state)
                if self.obs_mode == "state":
                    tee_pose = tee_poses_all[env_idx] if self.num_envs > 1 else tee_poses_all
                    self.tee_pose_buffers[env_idx].append(tee_pose)
                for camera_key in self.camera_keys:
                    rgb = images_all[camera_key][env_idx] if self.num_envs > 1 else images_all[camera_key]
                    self.image_buffers[env_idx][camera_key].append(rgb)

                # Accumulate reward
                reward_val = reward[env_idx] if self.num_envs > 1 else reward
                if isinstance(reward_val, torch.Tensor):
                    reward_val = reward_val.item()
                episode_rewards[env_idx] += reward_val

                steps[env_idx] += 1

                # Check termination for this environment
                terminated_val = terminated[env_idx] if self.num_envs > 1 else terminated
                truncated_val = truncated[env_idx] if self.num_envs > 1 else truncated
                if isinstance(terminated_val, torch.Tensor):
                    terminated_val = terminated_val.item()
                if isinstance(truncated_val, torch.Tensor):
                    truncated_val = truncated_val.item()

                # Get success status
                success_val = info.get("success", False)
                if self.num_envs > 1:
                    success_val = success_val[env_idx]
                if isinstance(success_val, torch.Tensor):
                    success_val = success_val.item()

                # Calculate intersection ratio
                base_env = self.env.unwrapped
                intersection_ratio = base_env.pseudo_render_intersection()
                if isinstance(intersection_ratio, torch.Tensor):
                    if self.num_envs > 1:
                        intersection_ratio = intersection_ratio[env_idx].item()
                    else:
                        intersection_ratio = intersection_ratio.item()

                final_intersection_ratios[env_idx] = intersection_ratio
                successes[env_idx] = bool(success_val)

                # Mark environment as done if terminated or truncated
                if terminated_val or truncated_val:
                    active_envs[env_idx] = False
                    if success_val:
                        print(f"✓ Env {env_idx} SUCCESS at step {steps[env_idx]}!")

            # Print compact status for all environments
            if step % 25 == 0:
                status = [
                    f"{'✓' if successes[i] else '✗'}{final_intersection_ratios[i]:.2f}" if active_envs[i] else "-"
                    for i in range(self.num_envs)
                ]
                print(f"Step {step}: [{' '.join(status)}]")

            step += 1

            # Render: either for the human viewer, or to capture per-step frames.
            if record_video:
                frame = self.env.render()  # (num_envs, H, W, 3) uint8 tensor
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                frame = frame.astype(np.uint8, copy=False)
                if frame.ndim == 3:  # single env: (H, W, 3) → (1, H, W, 3)
                    frame = frame[np.newaxis, ...]
                for env_idx in range(self.num_envs):
                    # Only append while the env is still running; this keeps each
                    # trial's video length aligned with its actual trial_time.
                    if active_envs[env_idx] or steps[env_idx] == step:
                        frame_buffers[env_idx].append(frame[env_idx])
            elif self.env.render_mode is not None:
                self.env.render()

        if record_video:
            self.last_frames = frame_buffers

        # Return results for all environments
        if self.num_envs == 1:
            return episode_rewards[0], successes[0], final_intersection_ratios[0], steps[0]
        else:
            return episode_rewards, successes, final_intersection_ratios, steps
