# eval_checkpoint_random_init.py - Evaluate with extreme random initializations
import math
import time

import gymnasium as gym
import numpy as np
import torch
from ppo_fast import Agent

import mani_skill.envs
from mani_skill.envs.tasks.tabletop.planar_push_t import PlanarPushTEnv
from mani_skill.utils.registration import register_env


# Custom environment with extreme randomization
@register_env("Planar-PushT-ExtremeRandom-v1", max_episode_steps=200)
class PlanarPushTExtremeRandomEnv(PlanarPushTEnv):
    """
    Modified PushT environment with extreme random initialization for stress testing.

    Configurable randomization ranges for both robot EE and T object positions.
    """

    def __init__(
        self,
        *args,
        extreme_spawn_box=True,
        # Robot end effector spawn range
        ee_x_min=0,
        ee_x_max=0.1,
        ee_y_min=-0.2,
        ee_y_max=0.1,
        # T object spawn range
        tee_x_min=-0.256,
        tee_x_max=-0.056,
        tee_y_min=-0.2,
        tee_y_max=0.1,
        **kwargs,
    ):
        self.extreme_spawn_box = extreme_spawn_box

        # Robot EE randomization bounds
        self.ee_x_min = ee_x_min
        self.ee_x_max = ee_x_max
        self.ee_y_min = ee_y_min
        self.ee_y_max = ee_y_max

        # T object randomization bounds
        self.tee_x_min = tee_x_min
        self.tee_x_max = tee_x_max
        self.tee_y_min = tee_y_min
        self.tee_y_max = tee_y_max

        super().__init__(*args, robot_init_qpos_noise=0, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Override initialization to allow more extreme randomization"""
        with torch.device(self.device):
            b = len(env_idx)

            # First, let table scene initialize (this will reset robot to default pose)
            self.table_scene.initialize(env_idx)

            # NOW do IK to set custom robot EE position (after table scene reset)
            # Sample random position within configured bounds
            random_ee_x = torch.rand(b, device=self.device) * (self.ee_x_max - self.ee_x_min) + self.ee_x_min
            random_ee_y = torch.rand(b, device=self.device) * (self.ee_y_max - self.ee_y_min) + self.ee_y_min
            random_ee_z = torch.full((b,), self._target_z, device=self.device)  # Keep fixed z height

            desired_pos = torch.zeros((b, 3), device=self.device)
            desired_pos[:, 0] = random_ee_x
            desired_pos[:, 1] = random_ee_y
            desired_pos[:, 2] = random_ee_z

            # Keep vertical orientation
            desired_quat = self._target_quat.to(device=self.device).unsqueeze(0).expand(b, -1)

            # Use IK to compute joint positions that achieve this pose
            from mani_skill.utils.structs import Pose

            arm_controller = self.agent.controller.controllers["arm"]
            to_base = arm_controller.root_link.pose.inv()
            target_pose_world = Pose.create_from_pq(desired_pos, desired_quat)
            target_pose_base = to_base * target_pose_world
            q0 = self.agent.robot.get_qpos()

            # Use more IK iterations for better orientation accuracy
            qpos_target = arm_controller.kinematics.compute_ik(
                pose=target_pose_base,
                q0=q0,  # current joint positions as initial guess
                is_delta_pose=False,
                current_pose=None,
                solver_config=dict(
                    type="levenberg_marquardt",
                    solver_iterations=100,  # Increase to 100 for better convergence
                    alpha=1.0,
                ),
            )

            if qpos_target is not None:
                # Set robot joint positions and velocities
                self.agent.robot.set_qpos(qpos_target)
                self.agent.robot.set_qvel(torch.zeros_like(qpos_target))

                # CRITICAL: Set PD controller drive targets to match the IK solution
                # This ensures the robot's internal controller will hold this position
                # We set the targets BEFORE any physics stepping happens
                arm_controller = self.agent.controller.controllers["arm"]
                arm_controller.set_drive_targets(qpos_target)

                # Log the target IK configuration
                print(f"Robot IK target: ({random_ee_x[0]:.3f}, {random_ee_y[0]:.3f}, {random_ee_z[0]:.3f})")
            else:
                raise ValueError("IK failed to find a solution")

            # Set goal tee position (same as parent)
            target_region_xyz = torch.zeros((b, 3))
            target_region_xyz[:, 0] += self.goal_offset[0]
            target_region_xyz[:, 1] += self.goal_offset[1]
            target_region_xyz[..., 2] = 1e-3
            from transforms3d.euler import euler2quat

            self.goal_tee.set_pose(
                self.goal_tee.pose.__class__.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, 0, self.goal_z_rot),
                )
            )

            if self.extreme_spawn_box:
                # EXTREME RANDOMIZATION: Use configured spawn box bounds
                target_region_xyz[..., 0] = torch.rand(b) * (self.tee_x_max - self.tee_x_min) + self.tee_x_min
                target_region_xyz[..., 1] = torch.rand(b) * (self.tee_y_max - self.tee_y_min) + self.tee_y_min
            else:
                # Original randomization
                target_region_xyz[..., 0] += torch.rand(b) * self.tee_spawnbox_xlength + self.tee_spawnbox_xoffset
                target_region_xyz[..., 1] += torch.rand(b) * self.tee_spawnbox_ylength + self.tee_spawnbox_yoffset

            target_region_xyz[..., 2] = 0.04 / 2 + 1e-3

            # Random Z rotation
            q_euler_angle = torch.rand(b) * (2 * torch.pi)
            q = torch.zeros((b, 4))
            q[:, 0] = (q_euler_angle / 2).cos()
            q[:, -1] = (q_euler_angle / 2).sin()

            obj_pose = self.tee.pose.__class__.create_from_pq(p=target_region_xyz, q=q)
            self.tee.set_pose(obj_pose)

            # EE starting position marker
            xyz = torch.zeros((b, 3))
            xyz[:] = self.ee_starting_pos2D
            self.ee_goal_pos.set_pose(
                self.ee_goal_pos.pose.__class__.create_from_pq(
                    p=xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )


# Configuration
control_mode = "pd_ee_delta_pose"
checkpoint_path = "/home/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1765441128/checkpoints/ckpt_9051.pt"
ENV_ID = "Planar-PushT-ExtremeRandom-v1"
num_episodes = 10
SEED = 1

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create environment with extreme randomization
print("Creating environment with extreme random initialization")
env = gym.make(
    ENV_ID,
    num_envs=1,
    obs_mode="state",
    render_mode="human",
    control_mode=control_mode,
    sim_backend="physx_cuda",
    max_episode_steps=100,
    extreme_spawn_box=False,  # Use extreme spawn box
    intersection_thresh=0.75,
)

# Load trained agent
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
print("=" * 60)
print("EXTREME RANDOMIZATION MODE")
print("=" * 60)

# Run episodes
success_count = 0
for episode in range(num_episodes):
    episode_seed = SEED + episode
    obs, info = env.reset(seed=episode_seed)
    episode_reward = 0
    done = False
    step = 0

    env.render()
    time.sleep(0.5)

    print(f"\nEpisode {episode + 1}/{num_episodes}")
    print(f"Initial T position (XY): {env.unwrapped.tee.pose.p[0, :2].cpu().numpy()}")

    # Verify actual robot TCP position and orientation after initialization
    tcp_pos = env.unwrapped.agent.tcp.pose.p[0].cpu().numpy()
    tcp_quat = env.unwrapped.agent.tcp.pose.q[0].cpu()
    target_quat = torch.tensor([0, 1.0, 0, 0], dtype=torch.float32)
    quat_dot = torch.abs(torch.sum(tcp_quat * target_quat))
    orientation_error_deg = 2 * torch.acos(torch.clamp(quat_dot, -1, 1)) * 180 / 3.14159

    target_z = 0.02
    z_error_mm = abs(tcp_pos[2] - target_z) * 1000

    print(f"Initial TCP position (XYZ): ({tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f})")
    print(f"Initial TCP z-height error: {z_error_mm:.1f}mm (target: {target_z:.3f}m)")
    print(f"Initial TCP orientation error: {orientation_error_deg:.2f}°")

    if z_error_mm > 2.0 or orientation_error_deg > 10.0:
        print("⚠️  WARNING: TCP not at correct height/orientation!")

    while not done:
        with torch.no_grad():
            action = agent.actor_mean(obs)
            value = agent.get_value(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01)

        done = terminated[0] or truncated[0] or info.get("success", [False])[0]
        episode_reward += reward[0].item()
        step += 1

    success = info.get("success", [False])[0]
    if success:
        success_count += 1

    print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={step}, Success={success}")

print("\n" + "=" * 60)
print(f"Overall Success Rate: {success_count}/{num_episodes} ({100 * success_count / num_episodes:.1f}%)")
print("=" * 60)

env.close()
