# eval_checkpoint.py - Evaluate a trained PPO checkpoint with human rendering
import math
import time

import gymnasium as gym
import numpy as np
import torch
from ppo_fast import Agent  # Now we can import directly since we're in the same directory


def extract_state(obs, state_mode):
    """
    Extract robot state from flattened observation.

    For PushT-v1 with obs_mode="state", the flattened observation structure is:
    - qpos: indices 0:7 (robot joint positions)
    - qvel: indices 7:14 (robot joint velocities)
    - tcp_pose: indices 14:21 (end-effector pose: position + quaternion)
    - goal_pos: indices 21:24 (goal position)
    - obj_pose: indices 24:31 (object pose: position + quaternion)

    Args:
        obs: Flattened observation tensor of shape [num_envs, 31]
        state_mode: One of "qpos", "qpos_qvel", "tcp_pose"

    Returns:
        state: numpy array of robot state
    """
    if state_mode == "qpos_qvel":
        # Extract qpos (0:7) and qvel (7:14)
        qpos = obs[:, 0:7]
        qvel = obs[:, 7:14]
        # print(f"qpos: {qpos}")
        # print(f"qvel: {qvel}")
        if isinstance(qpos, torch.Tensor):
            return torch.cat([qpos, qvel], dim=-1)
        else:
            return np.concatenate([qpos, qvel], axis=-1)
    elif state_mode == "qpos":
        # Extract only qpos (0:7)
        state = obs[:, 0:7]
        # print(f"qpos: {state}")
        if isinstance(state, torch.Tensor):
            return state
        else:
            return state
    elif state_mode == "tcp_pose":
        # Extract tcp_pose (14:21)
        state = obs[:, 14:21]
        # print(f"tcp_pose: {state}")
        if isinstance(state, torch.Tensor):
            return state
        else:
            return state
    else:
        raise ValueError(f"Unknown state mode: {state_mode}")


# Configuration
# control_mode = "pd_joint_delta_pos"
control_mode = "pd_ee_delta_pos"
checkpoint_path = f"/home/michzeng/.maniskill/demos/PushT-v1/rl/ppo_{control_mode}_ckpt.pt"
env_id = "PushT-v1"
num_episodes = 5
seed = 1  # Set seed for reproducibility

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create a single environment with human rendering
env = gym.make(
    env_id,
    num_envs=1,
    obs_mode="state",
    render_mode="human",  # This opens a visualization window
    control_mode=control_mode,
    sim_backend="physx_cuda",
    max_episode_steps=500,
)

# Load the trained agent
n_obs = math.prod(env.single_observation_space.shape)
n_act = math.prod(env.single_action_space.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Observation space: {n_obs}, Action space: {n_act}")
print(f"Loading checkpoint from: {checkpoint_path}")

agent = Agent(n_obs, n_act, device=device)
checkpoint = torch.load(checkpoint_path, map_location=device)
agent.load_state_dict(checkpoint)
agent.eval()

print("Checkpoint loaded successfully!")

# Run episodes
for episode in range(num_episodes):
    # Reset with seed for reproducibility (seed increments for each episode)
    obs, _ = env.reset(seed=seed + episode)
    episode_reward = 0
    done = False
    step = 0

    # Render the initial state
    env.render()
    time.sleep(0.5)  # Give time for rendering to initialize

    print(f"\nStarting Episode {episode + 1}...")

    while not done:
        with torch.no_grad():
            # Use actor_mean for deterministic actions (no exploration)
            action = agent.actor_mean(obs)
            # Get value function output
            value = agent.get_value(obs)

        print(f"action (shape: {action.shape}): {action}")
        print(f"value function: {value[0].item():.4f}")
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f"info: {info}")
        state = extract_state(obs, "qpos_qvel")
        print(f"qpos_qvel (shape: {state.shape}): {state}")
        env.render()  # Explicitly render each step
        time.sleep(0.01)  # Small delay to see the motion

        done = terminated[0] or truncated[0]
        episode_reward += reward[0].item()
        step += 1

    print(
        f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}, "
        f"Success: {info.get('success', [False])[0]}"
    )

env.close()
