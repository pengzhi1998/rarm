import sys
from pathlib import Path

# Add parent directory to path so we can import from models, utils, etc.
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import torch
import torch.nn as nn
from models.vit_lora import ViTWithLoRA
from models.temporal_processor import TemporalProcessor
from models.cross_attn import CrossAttentionComparator
from models.comp_head import ComparisonHead

class VideoComparator(nn.Module):
    """Full pipeline: Two videos → similarity score."""

    def __init__(
        self,
        vit_checkpoint_path,
        lora_r=16,
        embed_dim=768,
        hidden_dim=512,
        num_self_attn_layers=2,
        num_heads=8,
        device="cuda",
        use_temporal_processor: bool = True,
    ):
        super().__init__()

        # Enable Flash SDPA backend explicitly for PyTorch SDPA
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        self.use_temporal_processor = use_temporal_processor

        # ViT encoder (shared for all videos)
        self.vit = ViTWithLoRA(
            checkpoint_path=vit_checkpoint_path,
            lora_r=lora_r,
            device=device
        )

        # Temporal processor (spatial pooling + temporal RoPE + cross-frame self-attention).
        # Optional — when use_temporal_processor=False the embeddings flow straight from the
        # ViT into cross_attn (no spatial pooling, no temporal mixing).
        if use_temporal_processor:
            self.temporal_processor = TemporalProcessor(
                embed_dim=embed_dim,
                num_heads=num_heads,
                device=device
            ).to(device).to(torch.bfloat16)

        # Cross-attention comparator
        self.cross_attn = CrossAttentionComparator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim
        ).to(device).to(torch.bfloat16)  # ← Flash Attention 2 requires fp16/bf16!

        # Comparison head
        self.comparison_head = ComparisonHead(
            hidden_dim=hidden_dim,
            num_layers=num_self_attn_layers,
            num_heads=num_heads
        ).to(device).to(torch.bfloat16)  # ← Flash Attention 2 requires fp16/bf16!

        self.device = device
        self.dtype = torch.bfloat16  # Track dtype for consistency

    def forward(self, frames_A, frames_B):
        """
        Args:
            frames_A: [T, H, W, C] frame tensor for video A
            frames_B: [T, H, W, C] frame tensor for video B

        Returns:
            similarity: scalar similarity value
        """
        # Encode frames through ViT+LoRA (gradients enabled by default)
        # Output: [T*576, 768] where T=num_frames, 576=24×24 patches
        emb_A = self.vit.encode(frames_A)
        emb_B = self.vit.encode(frames_B)

        # Temporal processing: spatial pooling + temporal RoPE + cross-frame self-attention.
        # Output: [T*64, 768] where 64=8×8 patches. Skipped when use_temporal_processor=False;
        # cross_attn then sees the raw ViT embeddings [T*576, 768].
        if self.use_temporal_processor:
            emb_A = self.temporal_processor(emb_A)
            emb_B = self.temporal_processor(emb_B)

        # Cross-attention comparison
        emb_A_vs_B = self.cross_attn(emb_A, emb_B)

        # Similarity score
        similarity = self.comparison_head(emb_A_vs_B)

        return similarity

    def get_trainable_parameters(self):
        """Get all trainable parameters for optimizer."""
        params = []

        # LoRA parameters from ViT
        params.extend(self.vit.get_trainable_parameters())

        # Temporal processor parameters (skipped when use_temporal_processor=False)
        if self.use_temporal_processor:
            params.extend(self.temporal_processor.parameters())

        # Cross-attention parameters
        params.extend(self.cross_attn.parameters())

        # Comparison head parameters
        params.extend(self.comparison_head.parameters())

        return params