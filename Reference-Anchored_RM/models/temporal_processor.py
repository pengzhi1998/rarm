"""
Temporal Processing Module for Video Embeddings.

Applies:
1. Spatial pooling (24×24 → 8×8 patches per frame)
2. Temporal RoPE (time-based positional encoding)
3. Cross-frame self-attention (attend across all frames)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalProcessor(nn.Module):
    """
    Process temporal video embeddings with spatial pooling, temporal RoPE, and cross-frame attention.

    Pipeline:
        Input: [T*576, D] where T=num_frames, 576=24×24 patches, D=768
        1. Reshape to frames: [T, 576, D]
        2. Spatial pooling: [T, 576, D] → [T, 64, D] (24×24 → 8×8)
        3. Temporal RoPE: Add time-based positional encoding
        4. Cross-frame self-attention: Attend across all T*64 tokens
        5. Reshape back: [T, 64, D] → [T*64, D]
        Output: [T*64, D]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        rope_base: float = 10000.0,
        device: str = "cuda"
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rope_base = rope_base
        self.device = device

        # Original spatial resolution from DINOv3
        self.orig_spatial_size = 24  # 24×24 patches
        self.orig_num_patches = self.orig_spatial_size * self.orig_spatial_size  # 576

        # Target spatial resolution after pooling
        self.pool_spatial_size = 8  # 8×8 patches
        self.pool_num_patches = self.pool_spatial_size * self.pool_spatial_size  # 64
        self.pool_factor = self.orig_spatial_size // self.pool_spatial_size  # 3×3 pooling

        # Cross-frame self-attention (single layer as requested)
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

        # Precompute RoPE rotation frequencies
        self._init_rope_freqs()

    def _init_rope_freqs(self):
        """
        Precompute RoPE rotation frequencies for each channel pair.

        For d-dimensional embeddings, we have d/2 pairs.
        ω_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        """
        # Channel pair indices: 0, 1, 2, ..., d/2-1
        pair_indices = torch.arange(0, self.embed_dim // 2, dtype=torch.float32)

        # Frequencies: ω_i = base^(-2i/d)
        freqs = self.rope_base ** (-2.0 * pair_indices / self.embed_dim)

        # Register as buffer (moves with .to(device) but not trained)
        self.register_buffer('rope_freqs', freqs)

    def apply_rope(self, x: torch.Tensor, time_indices: torch.Tensor) -> torch.Tensor:
        """
        Apply Rotary Position Embedding (RoPE) based on time indices.

        Args:
            x: [B, N, D] embeddings where B=batch, N=num_tokens, D=embed_dim
            time_indices: [N] time index for each token (same time for all patches in a frame)

        Returns:
            x_rotated: [B, N, D] embeddings with temporal RoPE applied
        """
        B, N, D = x.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        # Compute rotation angles: θ_{t,i} = t * ω_i
        # time_indices: [N], rope_freqs: [D/2]
        # angles: [N, D/2]
        angles = time_indices.unsqueeze(-1) * self.rope_freqs.unsqueeze(0)  # [N, D/2]

        # Compute cos and sin for rotation, ensuring they match x's dtype
        cos = torch.cos(angles).to(x.dtype)  # [N, D/2]
        sin = torch.sin(angles).to(x.dtype)  # [N, D/2]

        # Reshape x to separate channel pairs: [B, N, D] → [B, N, D/2, 2]
        x_pairs = x.reshape(B, N, D // 2, 2)

        # Extract even and odd channels
        x_even = x_pairs[..., 0]  # [B, N, D/2]
        x_odd = x_pairs[..., 1]   # [B, N, D/2]

        # Apply rotation:
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd = x_even * sin + x_odd * cos
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        # Recombine pairs: [B, N, D/2, 2] → [B, N, D]
        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        x_rotated = x_rotated.reshape(B, N, D)

        return x_rotated

    def spatial_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply fixed average pooling to reduce spatial resolution.

        Args:
            x: [B, T, orig_num_patches, D] where orig_num_patches = 24×24 = 576

        Returns:
            x_pooled: [B, T, pool_num_patches, D] where pool_num_patches = 8×8 = 64
        """
        B, T, N, D = x.shape
        assert N == self.orig_num_patches, f"Expected {self.orig_num_patches} patches, got {N}"

        # Reshape to spatial grid: [B, T, 576, D] → [B*T, 24, 24, D]
        x_spatial = x.reshape(B * T, self.orig_spatial_size, self.orig_spatial_size, D)

        # Permute to [B*T, D, 24, 24] for pooling
        x_spatial = x_spatial.permute(0, 3, 1, 2)  # [B*T, D, 24, 24]

        # Average pool with 3×3 kernel, stride 3: [B*T, D, 24, 24] → [B*T, D, 8, 8]
        x_pooled = F.avg_pool2d(
            x_spatial,
            kernel_size=self.pool_factor,
            stride=self.pool_factor
        )  # [B*T, D, 8, 8]

        # Permute back and flatten: [B*T, D, 8, 8] → [B*T, 8, 8, D] → [B, T, 64, D]
        x_pooled = x_pooled.permute(0, 2, 3, 1)  # [B*T, 8, 8, D]
        x_pooled = x_pooled.reshape(B, T, self.pool_num_patches, D)  # [B, T, 64, D]

        return x_pooled

    def cross_frame_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention across all frames and spatial patches.

        Args:
            x: [B, T, num_patches, D] where B=batch, T=num_frames, num_patches=64, D=768

        Returns:
            x_attn: [B, T, num_patches, D] after cross-frame self-attention
        """
        B, T, num_patches, D = x.shape

        # Reshape to [B, T*num_patches, D] for self-attention across frames
        # Each batch element attends only to its own frames (no cross-batch attention)
        x_flat = x.reshape(B, T * num_patches, D)

        # Project to Q, K, V
        Q = self.q_proj(x_flat)  # [B, T*num_patches, D]
        K = self.k_proj(x_flat)  # [B, T*num_patches, D]
        V = self.v_proj(x_flat)  # [B, T*num_patches, D]

        # Reshape for multi-head attention: [B, num_heads, T*num_patches, head_dim]
        seq_len = T * num_patches

        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention (uses Flash Attention if available)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=0.0,
            is_causal=False
        )  # [B, num_heads, T*num_patches, head_dim]

        # Reshape back: [B, num_heads, seq_len, head_dim] → [B, seq_len, D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, seq_len, D)

        # Output projection
        attn_output = self.out_proj(attn_output)  # [B, T*num_patches, D]

        # Residual connection
        x_attn = x_flat + attn_output

        # Layer normalization
        x_attn = self.norm(x_attn)  # [B, T*num_patches, D]

        # Reshape back to [B, T, num_patches, D]
        x_attn = x_attn.reshape(B, T, num_patches, D)

        return x_attn

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process video embeddings with spatial pooling, temporal RoPE, and cross-frame attention.

        Args:
            embeddings: [T*orig_num_patches, D] or [B, T*orig_num_patches, D]
                       where T=num_frames, orig_num_patches=576, D=768, B=batch_size

        Returns:
            embeddings_processed: [T*pool_num_patches, D] or [B, T*pool_num_patches, D]
                                 where pool_num_patches=64, D=768
        """
        # Ensure input matches module dtype (handles float32 -> bfloat16 conversion)
        module_dtype = next(self.parameters()).dtype
        embeddings = embeddings.to(dtype=module_dtype)

        # Handle both batched [B, T*576, D] and unbatched [T*576, D] inputs
        is_batched = (embeddings.dim() == 3)

        if not is_batched:
            # Add batch dimension for single clip
            embeddings = embeddings.unsqueeze(0)  # [1, T*576, D]

        B, total_patches, D = embeddings.shape

        # Infer number of frames from input shape
        assert total_patches % self.orig_num_patches == 0, \
            f"Total patches {total_patches} must be divisible by orig_num_patches {self.orig_num_patches}"
        num_frames = total_patches // self.orig_num_patches

        # Step 1: Reshape to frames [B, T*576, D] → [B, T, 576, D]
        x = embeddings.reshape(B, num_frames, self.orig_num_patches, D)

        # Step 2: Spatial pooling [B, T, 576, D] → [B, T, 64, D]
        x = self.spatial_pooling(x)

        # Step 3: Temporal RoPE
        # Create time indices: all patches from frame t get time index t
        # time_indices: [T*64] where first 64 values are 0, next 64 are 1, ..., last 64 are T-1
        time_indices = torch.arange(num_frames, device=x.device, dtype=torch.float32)
        time_indices = time_indices.repeat_interleave(self.pool_num_patches)  # [T*64]

        # Flatten for RoPE: [B, T, 64, D] → [B, T*64, D]
        x_flat = x.reshape(B, num_frames * self.pool_num_patches, D)

        # Apply RoPE (same time indices for all batch elements)
        x_rope = self.apply_rope(x_flat, time_indices)  # [B, T*64, D]

        # Reshape back: [B, T*64, D] → [B, T, 64, D]
        x = x_rope.reshape(B, num_frames, self.pool_num_patches, D)

        # Step 4: Cross-frame self-attention [B, T, 64, D] → [B, T, 64, D]
        x = self.cross_frame_self_attention(x)

        # Step 5: Reshape back to flat format [B, T, 64, D] → [B, T*64, D]
        embeddings_processed = x.reshape(B, num_frames * self.pool_num_patches, D)

        # Remove batch dimension if input was unbatched
        if not is_batched:
            embeddings_processed = embeddings_processed.squeeze(0)  # [T*64, D]

        return embeddings_processed
