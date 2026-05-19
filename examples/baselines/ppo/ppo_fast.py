import os

from planar_return_planner import compute_return_action

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.normal import Normal

import wandb


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking we reconfigure eval env each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment. If None, uses the environment's default control mode."""
    render_mode: Optional[str] = None
    """render mode for training environments. None = no rendering (fastest), 'sensors' = minimal, 'rgb_array' = full rendering (slow!)"""
    eval_render_mode: str = "rgb_array"
    """render mode for evaluation environments. Use 'rgb_array' if you want to save videos."""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = False
    eval_temperature: float = 1.0
    """temperature for action sampling during evaluation (>1 increases diversity, <1 decreases it, 0 uses mean)"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Torch optimizations
    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""
    debug: bool = False
    """whether to print detailed timing information for debugging."""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std, validate_args=False)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)


class Logger:
    def __init__(self, log_wandb=False) -> None:
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if not self.log_wandb:
            return
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.item()
        wandb.log({tag: scalar_value}, step=step)

    def close(self):
        if self.log_wandb:
            wandb.finish()


def gae(next_obs, next_done, container, final_values):
    gae_start = time.perf_counter()
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        # real_next_values = nextvalues * nextnonterminal
        real_next_values = nextnonterminal * nextvalues + final_values[t]  # t instead of t+1
        delta = rewards[t] + args.gamma * real_next_values - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done):
    ts = []
    final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    step_times = []
    policy_times = []
    env_step_times = []
    logging_times = []

    for step in range(args.num_steps):
        step_start = time.perf_counter()

        # ALGO LOGIC: action logic
        policy_start = time.perf_counter()
        action, logprob, _, value = policy(obs=obs)
        policy_times.append(time.perf_counter() - policy_start)

        # TRY NOT TO MODIFY: execute the game and log data.
        env_start = time.perf_counter()
        next_obs, reward, next_done, infos = step_func(action)
        env_step_times.append(time.perf_counter() - env_start)

        logging_start = time.perf_counter()
        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            with torch.no_grad():
                final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(
                    infos["final_observation"][done_mask]
                ).view(-1)
        logging_times.append(time.perf_counter() - logging_start)

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )
        # NOTE (stao): change here for gpu env
        obs = next_obs = next_obs
        done = next_done
        step_times.append(time.perf_counter() - step_start)

    # NOTE (stao): need to do .to(device) i think? otherwise container.device is None, not sure if this affects anything
    if args.debug:
        total_policy_time = sum(policy_times)
        total_env_time = sum(env_step_times)
        total_logging_time = sum(logging_times)
        total_step_time = sum(step_times)

        print(f"  Rollout breakdown ({args.num_steps} steps):")
        print(f"    Policy inference:  {total_policy_time:.4f}s ({total_policy_time / total_step_time * 100:.1f}%)")
        print(f"    Env stepping:      {total_env_time:.4f}s ({total_env_time / total_step_time * 100:.1f}%)")
        print(f"    Logging/metrics:   {total_logging_time:.4f}s ({total_logging_time / total_step_time * 100:.1f}%)")
        print(f"    Total:             {total_step_time:.4f}s")

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container, final_values


def update(obs, actions, logprobs, advantages, returns, vals):
    update_start = time.perf_counter()
    optimizer.zero_grad()
    zero_grad_time = time.perf_counter() - update_start

    forward_start = time.perf_counter()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    forward_pass_time = time.perf_counter() - forward_start

    loss_calc_start = time.perf_counter()
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    loss_calculation_time = time.perf_counter() - loss_calc_start

    backward_start = time.perf_counter()
    loss.backward()
    backward_pass_time = time.perf_counter() - backward_start

    grad_clip_start = time.perf_counter()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    gradient_clipping_time = time.perf_counter() - grad_clip_start

    opt_step_start = time.perf_counter()
    optimizer.step()
    optimizer_step_time = time.perf_counter() - opt_step_start

    if args.debug:
        print(f"    Optimizer zero_grad time: {zero_grad_time:.4f}s")
        print(f"    Forward pass time: {forward_pass_time:.4f}s")
        print(f"    Loss calculation time: {loss_calculation_time:.4f}s")
        print(f"    Backward pass time: {backward_pass_time:.4f}s")
        print(f"    Gradient clipping time: {gradient_clipping_time:.4f}s")
        print(f"    Optimizer step time: {optimizer_step_time:.4f}s")

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

if __name__ == "__main__":
    args = tyro.cli(Args)
    # if not args.evaluate: exit()

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Fix for CUDA cusolver error with orthogonal initialization
    # Try using magma or default backend instead of cusolver
    if device.type == "cuda":
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except (RuntimeError, AttributeError):
            # If magma is not available, use default
            torch.backends.cuda.preferred_linalg_library("default")

    ####### Environment setup #######
    # Training environments
    train_env_kwargs = dict(
        obs_mode="state",
        sim_backend="physx_cuda",
        sensor_configs={},  # Disable all sensors for state-based training
    )
    if args.render_mode is not None:
        train_env_kwargs["render_mode"] = args.render_mode
    if args.control_mode is not None and args.control_mode != "":
        train_env_kwargs["control_mode"] = args.control_mode

    if args.debug:
        print("\n=== Environment Configuration ===")
        print(f"Training env kwargs: {train_env_kwargs}")
        print(f"Num envs: {args.num_envs}")
        print(f"Num steps: {args.num_steps}")
        print("=================================\n")

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **train_env_kwargs,
    )

    # Evaluation environments
    eval_env_kwargs = dict(obs_mode="state", sim_backend="physx_cuda")
    if args.eval_render_mode is not None:
        eval_env_kwargs["render_mode"] = args.eval_render_mode
    if args.control_mode is not None and args.control_mode != "":
        eval_env_kwargs["control_mode"] = args.control_mode
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **eval_env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=envs.unwrapped.control_freq,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=eval_envs.unwrapped.control_freq,
        )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            config = vars(args)
            config["env_cfg"] = dict(
                **train_env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **eval_env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=False,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient", f"GPU:{torch.cuda.get_device_name()}"],
            )
        else:
            print("WARNING: --track is False, no metrics will be logged. Pass --track to enable wandb.")
        logger = Logger(log_wandb=args.track)
        # Create checkpoints directory
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    else:
        print("Running evaluation")
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE (stao): change here for gpu env
        next_obs, reward, terminations, truncations, info = envs.step(action)
        next_done = torch.logical_or(terminations, truncations)
        return next_obs, reward, next_done, info

    ####### Agent #######
    agent = Agent(n_obs, n_act, device=device)
    # Reset linalg library to default to avoid slowdowns in some environments
    if device.type == "cuda":
        torch.backends.cuda.preferred_linalg_library("default")
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    # Make a version of agent with detached params
    agent_inference = Agent(n_obs, n_act, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    global_step = 0
    start_time = time.time()
    container_local = None
    next_obs = envs.reset()[0]
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    cumulative_times = defaultdict(float)

    for iteration in pbar:
        agent.eval()
        if iteration % args.eval_freq == 1:
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            # Once tee_place_success fires for an env, hand control to the return-to-start
            # planner for the rest of that episode. Only used in --evaluate mode.
            in_return_phase = np.zeros(args.num_eval_envs, dtype=bool)
            print(
                f"Running evaluation for {args.num_eval_steps} steps with {args.num_eval_envs} parallel environments..."
            )
            for step_idx in range(args.num_eval_steps):
                with torch.no_grad():
                    # Use temperature to control action diversity
                    # Use unified path to avoid CUDA graph recompilation from conditionals
                    action_mean = agent.actor_mean(eval_obs)
                    action_logstd = agent.actor_logstd.expand_as(action_mean)
                    action_std = torch.exp(action_logstd) * args.eval_temperature
                    # When temperature=0, this becomes deterministic (std=0)
                    action = action_mean + action_std * torch.randn_like(action_mean)
                    if args.evaluate and in_return_phase.any():
                        base_env = eval_envs.unwrapped
                        for env_idx in np.where(in_return_phase)[0]:
                            planner_act = compute_return_action(base_env, env_idx=int(env_idx))
                            action[env_idx] = torch.as_tensor(planner_act, device=action.device, dtype=action.dtype)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(action)
                    if args.evaluate and "tee_place_success" in eval_infos:
                        place = eval_infos["tee_place_success"]
                        if isinstance(place, torch.Tensor):
                            place = place.cpu().numpy()
                        in_return_phase |= np.asarray(place, dtype=bool)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
                        if args.evaluate:
                            done_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
                            in_return_phase[done_np] = False
                # Print progress every 100 steps or at the end
                if (step_idx + 1) % 200 == 0 or step_idx == args.num_eval_steps - 1:
                    print(
                        f"  Eval progress: {step_idx + 1}/{args.num_eval_steps} steps, "
                        f"{num_episodes} episodes collected"
                    )
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            print(f"Evaluation complete: {num_episodes} episodes collected")
            if eval_metrics_mean:
                print(
                    f"  Success rate: {eval_metrics_mean.get('success_once', 0):.2%}, "
                    f"Avg return: {eval_metrics_mean.get('return', 0):.2f}"
                )
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                print(f"\nEvaluation complete! Trajectories saved to: {eval_output_dir}")
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/checkpoints/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        iteration_start = time.perf_counter()
        if args.debug:
            print(f"\n=== Iteration {iteration} ===")

        torch.compiler.cudagraph_mark_step_begin()
        rollout_time = time.perf_counter()
        next_obs, next_done, container, final_values = rollout(next_obs, next_done)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        global_step += container.numel()

        update_time = time.perf_counter()
        container = gae(next_obs, next_done, container, final_values)
        gae_time = time.perf_counter() - update_time

        if args.debug:
            print(f"Total rollout time: {rollout_time:.4f}s")
            print(f"Total GAE time: {gae_time:.4f}s")

        container_flat = container.view(-1)

        update_time = time.perf_counter()

        # Optimizing the policy and value network
        opt_start = time.perf_counter()
        clipfracs = []
        for epoch in range(args.update_epochs):
            epoch_start = time.perf_counter()
            if args.debug:
                print(f"  Epoch {epoch + 1}/{args.update_epochs}")
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for batch_idx, b in enumerate(b_inds):
                batch_start = time.perf_counter()
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                if args.debug:
                    print(
                        f"    Total minibatch {batch_idx + 1}/{len(b_inds)} time: {time.perf_counter() - batch_start:.4f}s"
                    )
                clipfracs.append(out["clipfrac"])
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    if args.debug:
                        print(f"  Early stopping at epoch {epoch + 1} due to target KL")
                    break
            if args.debug:
                print(f"  Epoch time: {time.perf_counter() - epoch_start:.4f}s")
            if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                break
        optimization_time = time.perf_counter() - opt_start
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        iteration_time = time.perf_counter() - iteration_start
        if args.debug:
            print(f"=== Total iteration time: {iteration_time:.4f}s ===\n")

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", out["v_loss"].item(), global_step)
        logger.add_scalar("losses/policy_loss", out["pg_loss"].item(), global_step)
        logger.add_scalar("losses/entropy", out["entropy_loss"].item(), global_step)
        logger.add_scalar("losses/old_approx_kl", out["old_approx_kl"].item(), global_step)
        logger.add_scalar("losses/approx_kl", out["approx_kl"].item(), global_step)
        logger.add_scalar("losses/clipfrac", torch.stack(clipfracs).mean().cpu().item(), global_step)
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar(
            "time/total_rollout+update_time",
            cumulative_times["rollout_time"] + cumulative_times["update_time"],
            global_step,
        )
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/checkpoints/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
