"""
Evaluate a trained PPO checkpoint

Runs a fixed number of episodes and prints success statistics.

Use FAST_MODE=True for fast evaluation (no rendering/video), False for human visualization.
"""

import math
import time

import gymnasium as gym
import numpy as np
import torch
from ppo_fast import Agent  # Now we can import directly since we're in the same directory

import mani_skill.envs  # Required to register ManiSkill environments


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
FAST_MODE = True  # True: 16 parallel envs, no rendering | False: 1 env with human rendering
control_mode = "pd_ee_delta_pose"
checkpoint_path = "/home/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1766536544/checkpoints/final_ckpt.pt"
ENV_ID = "Planar-PushT-v1"
num_episodes = 100
# Base seed for reproducibility - Episode N uses seed = SEED + N
SEED = 1

# Set random seeds for reproducibility before environment creation
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create environment based on FAST_MODE
num_envs = 16 if FAST_MODE else 1
render_mode = None if FAST_MODE else "human"

print(f"Creating environment with ENV_ID_: {ENV_ID}")
print(f"ENV_ID_ repr: {repr(ENV_ID)}")
print(f"FAST_MODE: {FAST_MODE} (num_envs={num_envs}, render_mode={render_mode})")
from gymnasium.envs.registration import registry

print(f"Registered envs with 'Push': {[k for k in registry.keys() if 'Push' in k]}")
env = gym.make(
    ENV_ID,
    num_envs=num_envs,
    obs_mode="state",
    render_mode=render_mode,
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
success_count = 0
episode_rewards = []
episode_steps = []
for episode in range(num_episodes):
    # Reset with seed for reproducibility (seed increments for each episode)
    # Episode N uses seed = SEED + N (same as simple_inference.py)
    episode_seed = SEED + episode
    obs, _ = env.reset(seed=episode_seed)
    episode_reward = 0
    done = False
    step = 0

    # Render the initial state (only in non-FAST_MODE)
    if not FAST_MODE:
        env.render()
        time.sleep(0.1)  # Give time for rendering to initialize

    while not done:
        with torch.no_grad():
            # Use actor_mean for deterministic actions (no exploration)
            action = agent.actor_mean(obs)
            # Clip action to [-1, 1]
            # action = torch.clamp(action, -1.0, 1.0)

            # Get value function output
            value = agent.get_value(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if not FAST_MODE:
            print(f"action (shape: {action.shape}): {action}")
            print(f"value function: {value[0].item():.4f}")
            print(f"obs (shape: {obs.shape}): {obs}")
            # print(f"info: {info}")
            state = extract_state(obs, "qpos_qvel")
            print(f"qpos_qvel (shape: {state.shape}): {state}")
            env.render()  # Explicitly render each step
            time.sleep(0.01)  # Small delay to see the motion

        done = terminated[0] or truncated[0]
        episode_reward += reward[0].item()
        step += 1

    # Track episode metrics
    is_success = info.get("success", [False])[0]
    episode_rewards.append(episode_reward)
    episode_steps.append(step)

    if is_success:
        success_count += 1

    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}, Success: {is_success}")

# Calculate aggregate metrics
total_reward = sum(episode_rewards)
total_steps = sum(episode_steps)
avg_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
min_reward = np.min(episode_rewards)
max_reward = np.max(episode_rewards)
avg_steps = np.mean(episode_steps)
std_steps = np.std(episode_steps)
min_steps = np.min(episode_steps)
max_steps = np.max(episode_steps)

# Print comprehensive evaluation metrics
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"Checkpoint: {checkpoint_path}")
print(f"Control Mode: {control_mode}")
print(f"Number of Episodes: {num_episodes}")
print("-" * 60)
print(f"Success Rate: {success_count}/{num_episodes} ({100 * success_count / num_episodes:.1f}%)")
print("-" * 60)
print("REWARD STATISTICS:")
print(f"  Total Reward: {total_reward:.2f}")
print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
print(f"  Min Reward: {min_reward:.2f}")
print(f"  Max Reward: {max_reward:.2f}")
print("-" * 60)
print("EPISODE LENGTH STATISTICS:")
print(f"  Total Steps: {total_steps}")
print(f"  Average Steps: {avg_steps:.2f} ± {std_steps:.2f}")
print(f"  Min Steps: {min_steps}")
print(f"  Max Steps: {max_steps}")
print("=" * 60)

env.close()
