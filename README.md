# RARM

[Project page](https://rarm-robotics.github.io/) · [LIBERO policy training](https://github.com/pengzhi1998/rarm/tree/policy_training_RARM)

This branch contains reward-model training and clip-similarity inference. For DrQ-v2 policy training with RARM, use `policy_training_RARM`.

## Install

Use Python 3.10 and a CUDA GPU for training. Run commands from the repository root.

```bash
git clone --branch main https://github.com/pengzhi1998/rarm.git
cd rarm
conda create -n rarm python=3.10 -y
conda activate rarm
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Prepare model files

Download the [official DINOv3 ViT-B/16 weights](https://github.com/facebookresearch/dinov3#pretrained-models). Inference also requires the RARM checkpoint:

```text
checkpoints/
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
└── checkpoint_epoch_65.pt
```

Data and weights are not included. The current inference loader requires both files, even though the RARM checkpoint also contains the adapted DINOv3 encoder.

## Train

Organize training frames as:

```text
<data_dir>/JPEGImages/<video_id>/000000.jpg
```

Both `.jpg` and `.png` are supported. Use zero-padded filenames; videos shorter than 30 frames are skipped.

```bash
python Reference-Anchored_RM/scripts/train.py \
  --data_dir datasets/dataset_1500/ReferenceAnchoredRewardModelDataset \
  --batch_size 64 --device cuda
```

Defaults are 200 epochs and five frames per clip. Checkpoints are saved every five epochs under `checkpoints/dinov3_comparator/`. Adjust `--batch_size` and `--num_workers` for your machine; use `--help` for all options. Weights & Biases is disabled by default.

## Run inference

Prepare separate demo and rollout frame folders, each with at least seven JPG/PNG frames:

```bash
python Reference-Anchored_RM/scripts/reward_model.py \
  --demo_dir path/to/demo_frames \
  --rollout_dir path/to/rollout_frames \
  --comparator_checkpoint_path checkpoints/checkpoint_epoch_65.pt \
  --num_frames 5 --frame_stride 1 --clip_stride 5 \
  --device cuda --output_heatmap outputs/similarity.png
```

This saves a clip-similarity heatmap. The policy branch adds the progress-gating logic used for RL. Override `--vit_checkpoint_path` or `--comparator_checkpoint_path` to load weights from elsewhere.

See [PROVENANCE.md](PROVENANCE.md) for source versions and attribution.
