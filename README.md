# RARM: Reference-Anchored Reward Model

[Project page](https://rarm-robotics.github.io/) · [Reward model (`main`)](https://github.com/pengzhi1998/rarm/tree/main) · [LIBERO policy training (`policy_training_RARM`)](https://github.com/pengzhi1998/rarm/tree/policy_training_RARM)

RARM compares a rollout video clip with reference demonstration clips using DINOv3 features, LoRA, temporal attention, and cross-attention. This branch contains reward-model training and similarity inference. The `policy_training_RARM` branch contains from-scratch DrQ-v2 policy training in LIBERO using RARM rewards.

## Installation

Use Python 3.10 and a CUDA-capable GPU for training. The model uses bfloat16. The release uses PyTorch 2.10, torchvision 0.25, PEFT 0.15.2, and Transformers 4.57.1. Checkpoint loading and inference were checked on CPU; full GPU training has not been rerun for this release.

```bash
git clone --branch main https://github.com/pengzhi1998/rarm.git
cd rarm
conda create -n rarm python=3.10 -y
conda activate rarm
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Run all commands below from the repository root. DINOv3 source is loaded through Torch Hub at commit `54694f7627fd815f62a5dcc82944ffa6153bbb76`. For an offline checkout at that revision, set `RARM_DINOV3_REPO=/path/to/dinov3` before running the scripts.

## Model files

Supply these files locally; this repository contains **no model weights, training data, or demonstration assets**:

```text
checkpoints/
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
└── checkpoint_epoch_65.pt
```

Obtain the DINOv3 ViT-B/16 pretrained weights through the [official DINOv3 repository](https://github.com/facebookresearch/dinov3). Supply your RARM checkpoint as `checkpoints/checkpoint_epoch_65.pt`, or override the path. No RARM model download is bundled with this code release.

The inspected epoch-65 checkpoint contains the DINOv3 backbone, LoRA adapters, temporal processor, cross-attention, comparison head, and optimizer state. The existing loader first initializes DINOv3 from its separate pretrained file, then loads the complete RARM state strictly. Both paths are therefore required by this release's loader. See [provenance](PROVENANCE.md) for the source-version boundary.

## Train the reward model

Prepare frame folders with sortable, zero-padded names:

```text
datasets/dataset_1500/ReferenceAnchoredRewardModelDataset/
└── JPEGImages/
    ├── video_0001/
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   └── ...
    └── video_0002/
        ├── 000000.png
        └── ...
```

The loader accepts `.jpg` and `.png`, sorts paths lexicographically, and skips videos with fewer than 30 frames. It uses only `JPEGImages`; foreground images and masks are not needed.

```bash
python Reference-Anchored_RM/scripts/train.py \
  --data_dir datasets/dataset_1500/ReferenceAnchoredRewardModelDataset \
  --vit_checkpoint_path checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output_dir checkpoints/dinov3_comparator \
  --num_epochs 200 --batch_size 64 --learning_rate 1e-4 \
  --num_frames 5 --temperature 0.1 --num_workers 12 --device cuda
```

The five-frame sampler and contrastive objective follow the January 23 training version. Batch size 64 and the `dataset_1500` path reflect the supplied training edits. Adjust batch size and workers to fit your machine; changing batch size also changes the cross-video negatives. Checkpoints, `config.json`, and `losses.json` are written under a timestamped output directory, with a checkpoint every five epochs. Weights & Biases is disabled by default; use `--wandb_mode offline` or `--wandb_mode online` to enable logging.

## Run similarity inference

Place demo and rollout frames in separate directories, then run:

```bash
python Reference-Anchored_RM/scripts/reward_model.py \
  --demo_dir test/IMG_2025_frames/ \
  --rollout_dir test/IMG_2028_frames/ \
  --vit_checkpoint_path checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --comparator_checkpoint_path checkpoints/checkpoint_epoch_65.pt \
  --num_frames 5 --frame_stride 1 --clip_stride 5 \
  --device cuda --output_heatmap outputs/clip_similarity_heatmap.png
```

Both frame directories accept `.jpg` and `.png`. The script prints the maximum rollout reward and saves a rollout-versus-demo similarity heatmap. Each online reward is the maximum similarity to the demonstration clips; it returns `-3.0` until a complete rollout clip is available.

The original inference sampling is preserved: `clip_span = (num_frames - 1) * frame_stride + 3`. With the command above, five frames are sampled from a seven-frame span, so both input sequences need at least seven frames. Similarity is an unbounded score, not a calibrated probability or a success label. This branch does not add the later progress-gating analysis.

## Source and attribution

See [PROVENANCE.md](PROVENANCE.md) for original commits, release changes, and attribution. A project license has not yet been specified; third-party dependencies retain their upstream terms.
