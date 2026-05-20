# ManiSkill 3 (Beta)

# Michael's Notes

## Install

For display issues, run:
```bash
QT_QPA_PLATFORM=xcb
```

For policy evaluation, clone my [diffusion policy repo](https://github.com/Michaelszeng/diffusion-policy-experiments) and install (along with required dependencies):
```bash
pip install -e /home/michzeng/diffusion-policy --no-deps
pip install dill==0.3.5.1
pip install accelerate==0.13.2
pip install numba
```


### Installation on CSAIL SLURM Cluster:

```bash
# 1. Create + activate env
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda create -p /data/locomotion/michzeng/conda_envs/Maniskill \
  -c conda-forge --override-channels --strict-channel-priority \
  python=3.11 pip libvulkan-loader vulkan-tools
conda activate /data/locomotion/michzeng/conda_envs/Maniskill

# 2. Install ManiSkill (from the cloned repo root)
pip install -e .

# 3. Install torch
pip install torch

# 4. Install diffusion-policy
pip install -e /data/locomotion/michzeng/diffusion-policy-experiments --no-deps
pip install dill==0.3.5.1
pip install accelerate==0.13.2
pip install numba
```


## Data Collection

### Teleop Data Collection

```bash
python diffusion_policy/gamepad_teleop.py --output maniskill_planar_push_t.zarr
```

### RL-Expert Data Generation:

Train PPO policy:
```bash
python examples/baselines/ppo/ppo_fast.py \
  --env_id="Planar-PushT-v1" \
  --num_envs=4096 \
  --num-steps=16 \
  --update_epochs=8 \
  --num_minibatches=32 \
  --gamma=0.99 \
  --total_timesteps=2_000_000_000 \
  --num_eval_steps=400 \
  --num_eval_envs=16 \
  --control_mode=pd_ee_delta_pose \
  --ent-coef=0.005 \
  --cudagraphs
```

Here, we used increased entropy to try to maintain a more stochastic/diverse policy

Optionally (to continue training), add:
```bash
--checkpoint=/home/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1765228643/final_ckpt.pt
```

Generate data from PPO checkpoint (note: you must convert to rgb dataset using the command below after running this):
```bash
python examples/baselines/ppo/ppo_fast.py \
  --env_id="Planar-PushT-v1" \
  --control-mode="pd_ee_delta_pose" \
  --evaluate \
  --checkpoint=/home/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1766536544/checkpoints/final_ckpt.pt \
  --num_eval_envs=1 \
  --num-eval-steps=100000 \
  --save-trajectory \
  --no-capture-video \
  --eval-partial-reset \
  --eval-temperature=0.0
```
Note: `num-eval-steps` is the number of times we call `step()` on the vectorized environment. So the total number of steps taken is `num-eval-steps * num_eval_envs`.
Note: with `eval-partial-reset`, the policy will terminate and reset immediately upon success, but this only works with `num_eval_envs=1`.
Note: this script does record failures, but they'll be filtered out in the dataset regeneration step.

We also set `eval-temperature=1.5` to try to introduce more diversity into the data. Set to `0` for deterministic, mean evaluation.

This will save `trajectory.h5` and `trajectory.json` files to the `checkpoints/test_videos` folder. You will need to rename these to `trajectory.none.pd_ee_delta_pose.physx_cuda.*` before regenerating the dataset with RGB observations (the filename informs the `replay_trajectory` script of how the dataset was generated).

Regenerate Dataset with RGB observations (add `--count <n>` to limit the number of episodes replayed):
```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path /home/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1765298439/test_videos/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
    -o rgb \
    -b physx_cuda \
    -n 16 \
    --save-traj \
    --use-env-states \
    --count 500
```

Convert `.h5` dataset to `.zarr` format for training with the [diffusion policy repo](https://github.com/Michaelszeng/diffusion-policy-experiments)

```bash
python diffusion_policy/h5_to_zarr.py \
    --h5 /home/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1766536544/checkpoints/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    --output /home/michzeng/diffusion-policy/data/diffusion_experiments/maniskill/maniskill_planar_push_t.zarr
```

Now, you are ready to train!




## Diffusion Policy Evaluation

```bash
python diffusion_policy/evaluation.py \
        --checkpoints-dir /path/to/checkpoints \
        --action-horizons 1 2 3 4 5 6 8 10 12 15 \
        --n-video-trials 20 \
        --output-dir outputs/maniskill/2_obs/eval
```




## Dataset Utilities

### For State Observations (obs_mode='state')

To visualize trajectories collected with GPU simulation, use CPU rendering backend..

To visualize dataset without RGB images:

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path <path-to-trajectory.h5> \
    -b physx_cuda \
    -r cpu \
    --vis \
    --use-env-states \
    --count 5
```

To visualize dataset with RGB images:

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path /home/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1765441128/checkpoints/test_videos/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
    -b physx_cuda \
    --save-video \
    --use-env-states \
    --count 5
```

Key points:
- `-b physx_cuda`: Use GPU simulation (matches how trajectory was collected)
- `-r cpu`: Use CPU rendering backend (required for interactive viewer with `--vis`)
- `--vis`: Enable interactive visualization window
- `--use-env-states`: Replay using environment states for exact reproduction

### For RGB Observations (obs_mode='rgb')

Interactive visualization (`--vis`) is not supported for RGB trajectories with GPU simulation because camera rendering requires GPU backend. Instead, save videos:

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path <path-to-trajectory.rgb.h5> \
    -b physx_cuda \
    -n 16 \
    --save-video \
    --use-env-states \
    --count 20
```

IMPORTANT: `-n` must be set to the same value as used during conversion.

Videos will be saved to the same directory as the trajectory file.




![teaser](figures/teaser.jpg)
<p style="text-align: center; font-size: 0.8rem; color: #999;margin-top: -1rem;">Sample of environments/robots rendered with ray-tracing. Scene datasets sourced from AI2THOR and ReplicaCAD</p>

[![Downloads](https://static.pepy.tech/badge/mani_skill)](https://pepy.tech/project/mani_skill)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb)
[![PyPI version](https://badge.fury.io/py/mani-skill.svg)](https://badge.fury.io/py/mani-skill)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://maniskill.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/discord/996566046414753822?logo=discord)](https://discord.gg/x8yUZe5AdN)

ManiSkill is a powerful unified framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/), with a strong focus on manipulation skills. The entire tech stack is as open-source as possible and ManiSkill v3 is in beta release now. Among its features include:
- GPU parallelized visual data collection system. On the high end you can collect RGBD + Segmentation data at 30,000+ FPS with a 4090 GPU!
- GPU parallelized simulation, enabling high throughput state-based synthetic data collection in simulation
- GPU parallelized heterogeneous simulation, where every parallel environment has a completely different scene/set of objects
- Example tasks cover a wide range of different robot embodiments (humanoids, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, drawing/cleaning, dextrous manipulation)
- Flexible and simple task building API that abstracts away much of the complex GPU memory management code via an object oriented design
- Real2sim environments for scalably evaluating real-world policies 100x faster via GPU simulation.
- Sim2real support for deploying policies trained in simulation to the real world
- Many tuned robot learning baselines in Reinforcement Learning (e.g. PPO, SAC, [TD-MPC2](https://github.com/nicklashansen/tdmpc2)), Imitation Learning (e.g. Behavior Cloning, [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)), and large Vision Language Action (VLA) models (e.g. [Octo](https://github.com/octo-models/octo), [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer), [RT-x](https://robotics-transformer-x.github.io/))

For more details we encourage you to take a look at our [paper](https://arxiv.org/abs/2410.00425), published at [RSS 2025](https://roboticsconference.org/).

Please refer to our [documentation](https://maniskill.readthedocs.io/en/latest/user_guide) to learn more information from tutorials on building tasks to sim2real to running baselines.

**NOTE:**
This project currently is in a **beta release**, so not all features have been added in yet and there may be some bugs. If you find any bugs or have any feature requests please post them to our [GitHub issues](https://github.com/haosulab/ManiSkill/issues/) or discuss about them on [GitHub discussions](https://github.com/haosulab/ManiSkill/discussions/). We also have a [Discord Server](https://discord.gg/x8yUZe5AdN) through which we make announcements and discuss about ManiSkill.

Users looking for the original ManiSkill2 can find the commit for that codebase at the [v0.5.3 tag](https://github.com/haosulab/ManiSkill/tree/v0.5.3)


## Installation
Installation of ManiSkill is extremely simple, you only need to run a few pip installs and setup Vulkan for rendering.

```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch
```

Finally you also need to set up Vulkan with [instructions here](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

For more details about installation (e.g. from source, or doing troubleshooting) see [the documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
)

## Getting Started

To get started, check out the quick start documentation: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html

We also have a quick start [colab notebook](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb) that lets you try out GPU parallelized simulation without needing your own hardware. Everything is runnable on Colab free tier.

For a full list of example scripts you can run, see [the docs](https://maniskill.readthedocs.io/en/latest/user_guide/demos/index.html).

## System Support

We currently best support Linux based systems. There is limited support for windows and MacOS at the moment. We are working on trying to support more features on other systems but this may take some time. Most constraints stem from what the [SAPIEN](https://github.com/haosulab/SAPIEN/) package is capable of supporting.

| System / GPU         | CPU Sim | GPU Sim | Rendering |
| -------------------- | ------- | ------- | --------- |
| Linux / NVIDIA GPU   | ✅      | ✅      | ✅        |
| Windows / NVIDIA GPU | ✅      | ❌      | ✅        |
| Windows / AMD GPU    | ✅      | ❌      | ✅        |
| WSL / Anything       | ✅      | ❌      | ❌        |
| MacOS / Anything     | ✅      | ❌      | ✅        |

## Citation


If you use ManiSkill3 (versions `mani_skill>=3.0.0`) in your work please cite our [ManiSkill3 paper](https://arxiv.org/abs/2410.00425) as so:

```
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
} 
```

If you use ManiSkill2 (version `mani_skill==0.5.3` or lower) in your work please cite the ManiSkill2 paper as so:
```
@inproceedings{gu2023maniskill2,
  title={ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills},
  author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiang and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and Yuan, Xiaodi and Xie, Pengwei and Huang, Zhiao and Chen, Rui and Su, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

Note that some other assets, algorithms, etc. in ManiSkill are from other sources/research. We try our best to include the correct citation bibtex where possible when introducing the different components provided by ManiSkill.

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
