"""
DINOv3 Vision Encoder for video/image encoding.
Loads pretrained DINOv3 ViT-B/16 and extracts patch embeddings.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn

DINO_REPOSITORY = "facebookresearch/dinov3:54694f7627fd815f62a5dcc82944ffa6153bbb76"


class vit_dinov3(nn.Module):
    """DINOv3 ViT-B/16 encoder for video/image encoding.
    """
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint_path}")
        print(f"Loading DINOv3 ViT-B/16 from: {checkpoint_path}")

        local_repo = os.environ.get("RARM_DINOV3_REPO")
        if local_repo and not (Path(local_repo) / "hubconf.py").is_file():
            raise FileNotFoundError(f"RARM_DINOV3_REPO has no hubconf.py: {local_repo}")
        self.dinov3 = torch.hub.load(
            repo_or_dir=local_repo or DINO_REPOSITORY,
            model='dinov3_vitb16',
            weights=checkpoint_path,
            source='local' if local_repo else 'github',
            trust_repo=True,
            skip_validation=True,  # The full official commit SHA is pinned above.
        )

        self.dinov3 = self.dinov3.to(device).to(dtype)
        self.dinov3.eval()

        # Store dtype for later use (PEFT models don't have .dtype attribute)
        self.dtype = dtype

        # Store as buffer (not parameters) so they move with .to(device)
        self.register_buffer('normalize_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device).to(dtype))
        self.register_buffer('normalize_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device).to(dtype))

        self.hidden_size = 768
        self.patch_size = 16

        print(f"✓ DINOv3 loaded: hidden_size={self.hidden_size}, patch_size={self.patch_size}")

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode frames to patch embeddings.

        Args:
            frames: [T, C, H, W] or [T, H, W, C] frames tensor
                    Values in [0, 1] range from dataset

        Returns:
            embeddings: [T * N_patches, hidden_size] where N_patches = 576 at 384×384
        """
        if frames.dim() == 4 and frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)

        # Move to model's device and dtype (works now that we're nn.Module!)
        frames = frames.to(next(self.parameters()).device).to(self.dtype)
        frames = (frames - self.normalize_mean) / self.normalize_std
       
        # Use gradient checkpointing to trade compute for memory
        from torch.utils.checkpoint import checkpoint

        def forward_chunk(chunk):
            features = self.dinov3.forward_features(chunk)
            patch_embeddings = features['x_norm_patchtokens']
            T, N_patches, D = patch_embeddings.shape
            return patch_embeddings.reshape(T * N_patches, D)

        # Process in chunks (optionally with checkpointing)
        chunk_size = 24
        use_checkpointing = True  # Set to False if you have enough GPU memory (32GB+) for 2× backward speedup
        all_patch_embeddings = []

        for i in range(0, frames.size(0), chunk_size):
            chunk = frames[i:i+chunk_size]

            if torch.is_grad_enabled() and use_checkpointing:
                # RNG state is preserved by use_reentrant=False (safe with dropout if enabled)
                emb = checkpoint(forward_chunk, chunk, use_reentrant=False)
            else:
                emb = forward_chunk(chunk)

            all_patch_embeddings.append(emb)

        return torch.cat(all_patch_embeddings, dim=0)

    @property
    def vision_encoder(self):
        return self.dinov3
