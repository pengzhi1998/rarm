import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionComparator(nn.Module):
    """Cross-attention to compare two video embeddings."""

    def __init__(self, embed_dim=1024, num_heads=8, hidden_dim=512):
        super().__init__()

        # Use manual Q, K, V projections for Flash Attention support
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Project to hidden_dim for downstream processing
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, emb_A, emb_B):
        """
        Args:
            emb_A: [N_patches_A, embed_dim] OR [B, N_patches_A, embed_dim] - Query (anchor)
            emb_B: [N_patches_B, embed_dim] OR [B, N_patches_B, embed_dim] - Key/Value (comparison target)

        Returns:
            emb_A_vs_B: [N_patches_A, hidden_dim] OR [B, N_patches_A, hidden_dim]
        """
        # Handle both batched and unbatched inputs
        input_is_batched = (emb_A.dim() == 3)

        if not input_is_batched:
            # Add batch dimension for single pair
            emb_A = emb_A.unsqueeze(0)  # [1, N_patches_A, embed_dim]
            emb_B = emb_B.unsqueeze(0)  # [1, N_patches_B, embed_dim]

        # Project to Q, K, V
        Q = self.q_proj(emb_A)  # [1, N_patches_A, embed_dim]
        K = self.k_proj(emb_B)  # [1, N_patches_B, embed_dim]
        V = self.v_proj(emb_B)  # [1, N_patches_B, embed_dim]

        # Reshape for multi-head attention: [batch, num_heads, seq_len, head_dim]
        batch_size = Q.size(0)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention (PyTorch 2.0+ SDPA)
        # PyTorch automatically selects the best backend (Flash Attention 2/3 if available)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=0.0,
            is_causal=False
        )  # [batch, num_heads, N_patches_A, head_dim]

        # Reshape back to [batch, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)  # [1, N_patches_A, embed_dim]

        # Project to hidden_dim and normalize
        output = self.proj(attn_output)  # [B, N_patches_A, hidden_dim]
        output = self.norm(output)

        # Return in same format as input (batched or unbatched)
        if not input_is_batched:
            return output.squeeze(0)  # [N_patches_A, hidden_dim]
        else:
            return output  # [B, N_patches_A, hidden_dim]