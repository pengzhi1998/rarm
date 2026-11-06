# RARM: policy training on LIBERO

Train a DrQ-v2 policy from scratch with a pretrained Reference-Anchored Reward
Model (RARM) on LIBERO-10.

[Project page](https://rarm-robotics.github.io/) · [Reward-model training](https://github.com/pengzhi1998/rarm/tree/main)

## Install

Requires Linux, Python 3.10, and a CUDA GPU with working EGL rendering.

```bash
git clone --branch policy_training_RARM https://github.com/pengzhi1998/rarm.git
cd rarm
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

Run commands from the repository root. Activate the environment and repeat the
three `export` commands in each new shell. Use this repository's requirements;
LIBERO's historical requirements are not needed.

## Prepare files

Checkpoints and demonstrations are not bundled. Obtain the RARM checkpoint
separately, the [DINOv3 ViT-B/16 weights](https://github.com/facebookresearch/dinov3#pretrained-models),
and the [official LIBERO-10 demonstrations](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets).
Place them as follows:

```text
checkpoints/
  checkpoint_epoch_65.pt
  dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
data/LIBERO_datasets/libero_10/
  <task_name>_demo.hdf5
```

Both checkpoint files are required. `LIBERO_datasets` is case-sensitive.
`--seed` also selects the reference `demo_<seed>` in the task's HDF5 file.
Use `--vit_checkpoint_path` and `--comparator_checkpoint_path` for other locations.

## Train

```bash
python -m reward_model_baseline.Libero.task_training \
  --algorithm drqv2 --task_id 2 --reward_type 1 \
  --use_demo_init --use_demo_decay \
  --seed 0 --total_timesteps 1000000
```

Task 2 turns on the stove and places the moka pot on it. Other IDs are listed in
[the task mapping](reward_model_baseline/Libero/constants/libero_task_mapping.py).

The policy starts from random weights; RARM stays fixed. Demonstrations provide
reference images and initialization states. With `--use_demo_init`, training
can also replay noisy demo actions; `--use_demo_decay` gradually reduces this
initialization. The default reward is `RARM_reward + 5 × success - 0.5`, where
RARM rewards confident forward progress. Evaluation does not replay demo actions.

This release defaults to **epoch 65**. Historical LIBERO runs used an epoch-20
checkpoint, so their results are not claimed reproduced here.

## Evaluate

Training prints the run directory under `output/Libero/`. Replace
`YOUR_RUN_FOLDER` below with its final folder name:

```bash
python -m reward_model_baseline.Libero.task_eval \
  --task_id 2 --seed 0 --experiment_run 0 \
  --training_run "YOUR_RUN_FOLDER" --num_episodes 10
```

Evaluation loads the best saved policy. Add `--model_name drqv2_final.pt` to
select the final policy or `--verbose` to compute RARM scores and save frames.
Use either command's `--help` for additional options.

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream licenses.
