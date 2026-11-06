# RARM: policy training on LIBERO

Train a randomly initialized DrQ-v2 policy with a pretrained Reference-Anchored
Reward Model (RARM). This branch contains the LIBERO-10 simulation pipeline;
reward-model training is on the `main` branch.

The policy code is extracted from `yiduo/drq_v2_libero` at
`ba3e9c3286aa8c289905bcee63d9b1d5606de28b`. It retains the original
reward alignment, dual-camera observations, demo initialization, and DrQ-v2
hyperparameters. The public default is **`checkpoint_epoch_65.pt`**. Historical
simulation runs used a separate LIBERO checkpoint that stored epoch 20, so
results with this default must
be evaluated again; this release does not claim to reproduce those scores.
No checkpoints, demonstrations, videos, or simulator assets are bundled.

## Install

Use Linux, Python 3.10, and a CUDA GPU with working EGL rendering for training.
Run these commands from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
export PYTHONPATH="$PWD/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PWD/.cache/libero"
python scripts/configure_libero.py
export MUJOCO_GL=egl
```

Keep the three environment variables above in each new shell. The explicit
`PYTHONPATH` makes the pinned LIBERO namespace package available; no changes to
its source are needed. The configuration helper writes paths under `.cache/`
and avoids LIBERO's interactive first-import setup. Use the pinned requirements
here instead of LIBERO's historical training requirements. A CUDA-specific
PyTorch wheel index may be needed for your system; keep PyTorch/torchvision at
the matching versions in `requirements.txt`.

DINOv3 source is pinned to `54694f7627fd815f62a5dcc82944ffa6153bbb76` and
loaded through `torch.hub`. For offline use, clone that revision separately and
set `RARM_DINOV3_REPO=/absolute/path/to/dinov3`.

## Prepare checkpoints and demonstrations

Place these separately obtained files at the following paths:

```text
checkpoints/
  checkpoint_epoch_65.pt
  dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
data/
  LIBERO_datasets/
    libero_10/
      KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
      ...
```

`LIBERO_datasets` is case-sensitive. Obtain LIBERO-10 demonstrations through
the [official LIBERO dataset instructions](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets).
The HDF5 files must include demonstration `states`, `actions`, and
`obs/agentview_rgb`. The first run renders reference states at 384×384 and caches
JPEGs under `data/`; it does not train the reward model. `--seed` also selects the
reference `demo_<seed>`, so that demonstration must exist for the selected task.

Obtain the DINOv3 ViT-B/16 LVD-1689M weights under the
[upstream access terms](https://github.com/facebookresearch/dinov3#pretrained-models).
The RARM checkpoint includes its adapted encoder, but the current constructor
still loads the original DINOv3 checkpoint before applying the full RARM state.
Neither file is downloaded by this repository.

Expected SHA-256 for `checkpoint_epoch_65.pt`:

```text
1d8791d4d5741892873f3fad607ee326349a797603f3cd07484ba80323d42e81
```

Both training and verbose evaluation accept `--vit_checkpoint_path` and
`--comparator_checkpoint_path` to use absolute paths outside the repository.
All RARM entry points default to epoch 65. The datasets and checkpoints are
ignored by Git.

## Train

The following uses the historical LIBERO initialization settings with the
released epoch-65 reward model:

```bash
python -m reward_model_baseline.Libero.task_training \
  --algorithm drqv2 --task_id 2 --reward_type 1 \
  --use_demo_init --use_demo_decay \
  --seed 0 --total_timesteps 1000000
```

`--algorithm drqv2` and `--reward_type 1` are the defaults. Policies consume three
stacked 84×84 RGB frames from both agentview and wrist cameras. RARM uses five
384×384 agentview frames sampled every four environment steps.

The policy starts from random weights; the reward model is pretrained and fixed.
Training still uses reference demonstrations: `--use_demo_init` enables random
reference-state initialization (RSI), which may replay up to 70 noisy demo
actions with probability 0.5. `--use_demo_decay` decreases the RSI probability
linearly from 1 to 0. Without RSI, reset uses the selected demo's initial state
plus small arm/object perturbations. Evaluation disables demo action replay;
RSI during evaluation is enabled only by `--use_eval_rsi` during training or
`--use_demo_init` in the evaluation command.

The RARM component gives +1 only when a confident reference match advances the
maximum reached progress. The runner aligns clip rewards to environment steps
and trains with `RARM_reward + 5 × success - 0.5`. By default RARM stays at full
strength. `--rm_decay` optionally fades it to success-only rewards over the
fraction specified by `--reward_ratio` (default 0.3). `--reward_type 0` and `2`
select environment and success-only reward controls, respectively.

Task IDs use the pinned `libero_10` ordering:

| ID | Task |
|---|---|
| 0 | Alphabet soup and tomato sauce into basket |
| 1 | Cream cheese and butter into basket |
| 2 | Turn on stove and place moka pot on it |
| 3 | Black bowl into bottom drawer, then close it |
| 4 | White mug on left plate; yellow/white mug on right plate |
| 5 | Book into back compartment of caddy |
| 6 | White mug on plate; chocolate pudding to its right |
| 7 | Alphabet soup and cream cheese into basket |
| 8 | Both moka pots on stove |
| 9 | Yellow/white mug into microwave, then close it |

## Evaluate and resume

Runs are written to `output/Libero/<task_id>_<task_name>_<version>/<run-folder>/`.
The `models/` directory contains `drqv2_best.pt`, periodic policy checkpoints,
and `drqv2_final.pt`; `logs/` contains TensorBoard metrics.

```bash
python -m reward_model_baseline.Libero.task_eval \
  --task_id 2 --seed 0 --experiment_run 0 \
  --training_run <run-folder> --num_episodes 10
```

Evaluation prefers the best saved policy. Use `--model_name drqv2_final.pt` to
select the final policy, and `--verbose` to also compute RARM scores and save
rendered frames. Plain evaluation does not load the reward model.

For resumable jobs, add `--steps_this_link 500000` **on the first training call**.
This selects a deterministic run folder and saves the replay buffer plus
`checkpoint.pt`. Repeat the same command with `--resume` for subsequent calls.
A link stops at the next episode boundary. Keep the same task, seed, version,
reward settings, and replay directory when resuming.

## Provenance and validation

The extraction removes other reward baselines and PPO/SAC entry points, moves
the shared DrQ-v2 agent into the LIBERO package, exposes checkpoint paths, and
changes the default reward selection to RARM. It preserves the simulation
training code's gated-progress reward implementation; the standalone similarity
inference on `main` serves a different entry point.

Validation includes strict epoch-65 model loading and a five-frame CPU forward
pass, a dual-camera DrQ-v2 CPU action/update, and a comparison of the cleaned
runner with the source using a controlled simulator double. An EGL CPU
LIBERO reset and one rendered action were also checked with NumPy 1.26.4 and
OpenCV 4.10. These checks do
not substitute for a complete CUDA training run. The dependency set is a
release environment, not an archived environment from the historical runs.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream code licenses.
