# Source provenance

This release is a curated code snapshot from [Reward-Model-MVP](https://github.com/daidai189963/Reward-Model-MVP), with no inherited Git history or bundled data and weights.

## Reward-model training

Base commit: `b0cb34ed58821b92ee6f50ad4e59a4c104b9cae0`, January 23, 2026, “making sure there are no skipped clips”. The core model architecture, loss, and sampling algorithm are retained from this version.

The supplied screenshot identified two edited files:

- `scripts/train.py`: add `dataset_1500` to the dataset path and change the main-call batch size from 21 to 64. Applying only these edits to the original file produces Git blob `1a4df446152b4b56c1b29ea633b69e67c34d3865`, matching the screenshot.
- `utils/dataset.py`: scan both `.jpg` and `.png`. The released dataset file is exactly Git blob `917f3310e94a5c62a71a8c0c3ad36a4181a3b24c`, matching the screenshot.

Release packaging changes to training replace machine-specific paths with command-line options, expose workers and logging mode, validate arguments, and use consistent function/CLI defaults (200 epochs, batch size 64, temperature 0.1). Dataset paths are relative to the invocation directory. The training sampler remains fixed to five frames. The Jan25 static/jittery negative-sampling changes are not included.

## Inference

The Jan23 commit has no `reward_model.py`. The inference script comes from `7a0c607f54a6baaf2a395644efea0ae24414c222`, January 25, 2026. Its model architecture is identical to the Jan23 model. Release changes add PNG input, portable default checkpoint paths, argument/path checks, explicit CPU/weights-only checkpoint deserialization, inference-mode heatmap evaluation, and output-directory creation. The similarity definition and original clip-index selection are retained. Later progress-gating and visualization variants are not included.

The documented checkpoint filename is `checkpoint_epoch_65.pt`. Inspection of an existing file confirmed a complete model state with backbone and LoRA parameters, but the original training run/configuration has not been independently established. The checkpoint filename alone does not prove it was trained with the Jan23 commit and screenshot changes.

## Dependencies and authorship

- DINOv3 source is pinned to official commit `54694f7627fd815f62a5dcc82944ffa6153bbb76`, matching the clean local DINO checkout used for release verification. Setting `RARM_DINOV3_REPO` opts into a user-managed local checkout.
- The stable core (PyTorch 2.10.0, torchvision 0.25.0, PEFT 0.15.2, Transformers 4.57.1, timm 1.0.24) was validated in the existing Python 3.10.19 evaluation environment: all 293 model-state keys loaded strictly, a five-frame forward pass produced a finite score, and the full CLI rendered a heatmap from synthetic JPG/PNG sequences. NumPy 1.26.4 and OpenCV 4.10.0.84 are selected for compatibility with the policy branch; wandb 0.24.0 comes from the RM development environment. The dependency list is not a reconstruction of the original checkpoint's training environment or a complete lockfile.
- Original source authorship remains with the Reward-Model-MVP contributors. The selected original tree contained no root license; this snapshot does not invent one. Third-party dependencies remain subject to their respective upstream terms.
