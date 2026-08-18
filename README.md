# ManiSkill 3

This repo is a fork of [Maniskill](https://github.com/mani-skill/ManiSkill), and is a part of the project: [Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies]().

This fork contains the following additions to Maniskill:
 - "Planar-PushT" environment that mimics the official "PushT" environment but constrains the robot end effector into the XY plane (with fixed Z coordinate).
 - A USB game-pad (Logitech G F310) teleop interface to collect teleoperation data on the "Planar-PushT" task.
 - Working training pipeline to train an RL-based Markovian expert to solve "Planar-PushT" with very high reliability.
 - An Evaluation pipeline in order to evaluate diffusion policies trained on "Planar-PushT" demonstration data.
 - Scripts for usage on a SLURM cluster


## Installation

Create a virtual environment:
```bash
python -m venv env
```

Source the virtual environment to activate it:
```bash
source env/bin/activate
```

Install python packages:
```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch
pip install zarr
```

Set up Vulkan with [instructions here](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan)

For more details about installation (e.g. from source, or doing troubleshooting) see [the documentation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
)



## Teleop Data Collection for Planar-PushT

Obtain a Logitech G F310 (or else adapt the script for your own teleop interface). Run:

```bash
python diffusion_policy/gamepad_teleop.py --output maniskill_planar_push_t.zarr
```

Notes:
 - **Controls**: Hold `Right Trigger` for half-speed (fine control) or `Left Trigger` for triple-speed (fast repositioning). The `Left Joystick` drives the pusher in the XY plane.
 - **Episode saving**: per-timestep pusher pose, T pose, 2D action, and base + wrist camera RGB frames are recorded. Pressing `A` again or the environment terminating on success writes the episode to a `.npz` file in `--traj-dir` (default `planar_pusht_trajs`). On quit (`X`), all `.npz` files are concatenated into the output zarr.
 - **Zarr output is append-only across runs**: re-running with the same `--output` path *appends* new trajectories to the existing zarr.
 - **Debugging the gamepad**: if button/axis mappings look wrong for your controller, run with `--debug-axes` to get a live readout of every axis and button, then adjust `_BUTTON_MAP` in the script accordingly.




## Training a RL Policy for Planar-PushT

Train PPO policy:
```bash
./examples/baselines/ppo/train_ppo.sh
```
You can optionally pass an `env_id` and a .pt path to resume training from:
```bash
./examples/baselines/ppo/train_ppo.sh Planar-PushT-v1 runs/*/*.pt
```
Note: passing a checkpoint only loads the policy/value network weights (a warm start). The optimizer state, global step counter, and W&B run are **not** restored.





## Generating a Dataset from the Trained RL Policy

The easiest way to turn a trained PPO checkpoint into a `.zarr` dataset is the `gen_data.sh` script, which runs the full pipeline end-to-end: evaluate the checkpoint to collect trajectories, rename them, replay them with RGB observations, and convert the result to `.zarr`.

Edit the constants at the top of `examples/baselines/ppo/gen_data.sh` to point at your run, then run it:
```bash
./examples/baselines/ppo/gen_data.sh
```


Key constants to set:
- `CHECKPOINT`: path to the `.pt` checkpoint to generate data from.
- `ENV_ID` / `CONTROL_MODE`: must match how the policy was trained.
- `NUM_EVAL_STEPS`: number of `step()` calls; total steps taken is `NUM_EVAL_STEPS * NUM_EVAL_ENVS`. The script does not allow specifying a desired number of episodes, only a desired number of total env steps.
- `EVAL_TEMPERATURE`: `0.0` for deterministic (mean) actions; higher (e.g. `1.5`) for more diverse data.
- `REPLAY_COUNT`: max number of episodes to replay (pick a value comfortably above the number you want in the dataset to account for failures).
- `ZARR_OUTPUT`: output path for the final `.zarr` dataset.




## Usage on a SLURM Cluster

Follow these instructions for installation onto the SLURM cluster:

```bash
# 0. Make sure pip installs land in the env and that imports never silently fall
#    back to user-level site-packages in your home directory. Without this, an
#    env can appear to work on the login node while failing on compute nodes
#    with errors like "ModuleNotFoundError: No module named 'sapien'" (the
#    sbatch scripts override HOME, which moves the user site-packages path).
export PYTHONNOUSERSITE=1

# 1. Create + activate env
conda create -p PATH-TO/conda_envs/Maniskill \
  -c conda-forge --override-channels --strict-channel-priority \
  python=3.11 pip libvulkan-loader vulkan-tools
conda activate PATH-TO/conda_envs/Maniskill

# 2. Install ManiSkill (from the cloned repo root)
pip install -e .

# 3. Install PyTorch (modify so the CUDA build matches your cluster drivers)
pip install --index-url https://download.pytorch.org/whl/cu129 "torch==2.11.0" "torchvision==0.26.0"

# 4. Install torchrl + tensordict pinned to versions compatible with torch 2.11.
#    --no-deps prevents pip from trying to upgrade torch back to the latest.
pip install --no-deps "torchrl==0.11.1" "tensordict==0.11.0"

# 5. Pinned so the environment stays reproducible over time. zarr must stay on
#    the 2.x series: the diffusion policy code uses the zarr 2 API.
pip install "wandb==0.28.1" "zarr==2.12.0"
```

We provide SLURM scripts for PPO policy training and dataset generation. Set the cluster parameters at the top of the sbatch scripts, and set the path to your Maniskill Conda environment, then:

Train a PPO policy:
```bash
sbatch examples/baselines/ppo/submit_ppo_train.sbatch
```

Generating a Dataset from the Trained RL Policy:
```bash
sbatch examples/baselines/ppo/submit_ppo_gen_data.sbatch
```



## Evaluating a Diffusion Policy

This repo supports evaluating a diffusion policy trained using my [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments) repo.

Presuming you have collected a `.zarr` dataset using the Teleop or RL-expert data generation pipelines described above, you can train a policy following the instructions in [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments) then follow these instrutions to run/evaluate it:

Firstly, install the dependencies of the diffusion policy training repo into your Maniskill environment (these instructions apply both locally and for installation onto a SLURM cluster).

```bash
pip install -e /data/locomotion/michzeng/diffusion-policy-experiments --no-deps

pip install \
    "numpy==2.4.6" \
    "diffusers==0.11.1" \
    "huggingface-hub==0.25.2" \
    "zarr==2.12.0" \
    "numcodecs==0.12.1" \
    "accelerate==0.13.2" \
    "wandb==0.28.1" \
    "numba==0.66.0" \
    "hydra-core==1.3.4" \
    "termcolor==3.3.0" \
    "pymunk==7.3.0" \
    "shapely==2.1.2" \
    "einops==0.8.2" \
    "scikit-image==0.26.0" \
    "scikit-video==1.1.11" \
    "threadpoolctl==3.6.0" \
    "boto3==1.43.62" \
    "datasets==5.0.1" \
    "cffi==2.1.0" \
    "cython==3.2.9" \
    "imageio==2.37.4" \
    "imageio-ffmpeg==0.6.0" \
    "robomimic==0.3.0"

pip install --no-deps "dill==0.3.5.1"
```

To verify the environment before submitting a job, load one checkpoint end to end:

```bash
python -c "
import dill, torch, hydra
payload = torch.load('/path/to/checkpoint.ckpt', pickle_module=dill, map_location='cpu', weights_only=False)
cfg = payload['cfg']
w = hydra.utils.get_class(cfg._target_)(cfg)
w.load_payload(payload, exclude_keys=None, include_keys=None)
print('OK')
"
```

Then, run evaluation using this script:


```bash
python diffusion_policy/evaluation.py \
        --checkpoints-dir /path/to/checkpoint(s) \
        --env-id Planar-PushT-v1 \
        --action-horizons 1 2 3 4 5 6 8 10 12 15 \
        --num-trials-per-horizon 500 \
        --num-envs 16 \
        --n-video-trials 20 \
        --output-dir outputs/maniskill/2_obs/eval
```

Notes:
 - **`--checkpoints-dir`**: accepts either a directory of `.ckpt` files (every checkpoint is evaluated) or a single `.ckpt` file.
 - **`--n-video-trials`**: save MP4s for the first N trials per (checkpoint, horizon). `0` disables recording (fastest), `-1` records all trials.
 - **`--output-dir`**: where results are written.
 - **`--resume`**: continue partially-finished evaluations from the last saved round in the given output-dir.
 - **`--num-envs`**: number of parallel environments to use
 - **`--action-horizons`**: space-separated list of action horizons \(T_a\) to sweep; each checkpoint is run against every horizon and results are grouped under `T_a_<horizon>/` subfolders.
 - **`--num-trials-per-horizon`**: number of trials to run for each action horizon.


To run eval on a SLURM cluster, set `CHECKPOINT_PATH` at the top of `diffusion_policy/submit_evaluation.sbatch`, then:

```bash
sbatch diffusion_policy/submit_evaluation.sbatch 1 2 3 4 5 6 8 10 12 15
```



## Helpful Utility Scripts

`mani_skill.trajectory.replay_trajectory` re-runs saved `.h5` trajectories so you can sanity-check collected data, either in an interactive viewer or as rendered videos.

### For State Observations (obs_mode='state')

Trajectories collected with GPU simulation must be replayed with the CPU rendering backend to use the interactive viewer.

The command below opens an interactive window to inspect the first few trajectories without rendering RGB camera images (fastest, geometry only):

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path <path-to-trajectory.h5> \
    -b physx_cuda \
    -r cpu \
    --vis \
    --use-env-states \
    --count 5
```

The command below instead renders the RGB camera views and saves them as videos, useful for checking what the policy's cameras actually see:

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path <path-to-trajectory.h5> \
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

Interactive visualization (`--vis`) is not supported for RGB trajectories with GPU simulation because camera rendering requires GPU backend. The command below replays an RGB trajectory across parallel envs and saves the camera views as videos instead:

```bash
python -m mani_skill.trajectory.replay_trajectory \
    --traj-path <path-to-trajectory.rgb.h5> \
    -b physx_cuda \
    -n 16 \
    --save-video \
    --use-env-states \
    --count 20
```

IMPORTANT: `-n` must be set to the same value as used when converting the state-only trajectory to contain RGB data also.

Videos will be saved to the same directory as the trajectory file.



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

If you use this repo in your work, please cite:


1. The [ManiSkill3 paper](https://arxiv.org/abs/2410.00425):

```
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
} 
```

2. [Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies](https://arxiv.org/abs/2608.15938)

```
@misc{zeng2026revisitingopenloopexecutionrobotics,
      title={Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies}, 
      author={Michael Zeng and Abhinav Agarwal and Ajay Bati and Brian Lee and Siddharth Ancha and Russ Tedrake},
      year={2026},
      eprint={2608.15938},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.15938}, 
}
```

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
