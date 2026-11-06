import torch
import torch.nn as nn

class ComparisonHead(nn.Module):
    """Self-attention + MLP to produce similarity score."""

    def __init__(self, hidden_dim=512, num_layers=2, num_heads=8, mlp_hidden=256):
        super().__init__()

        # Self-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=mlp_hidden * 2,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        # MLP to scalar
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden // 2, 1)  # Unbounded scalar output
        )

    def forward(self, emb_A_vs_B):
        """
        Args:
            emb_A_vs_B: [N_patches, hidden_dim] OR [B, N_patches, hidden_dim]

        Returns:
            sim_scalar: scalar similarity value OR [B] tensor of similarities
        """
        # Handle both batched and unbatched inputs
        input_is_batched = (emb_A_vs_B.dim() == 3)

        if not input_is_batched:
            # Add batch dimension for single input
            x = emb_A_vs_B.unsqueeze(0)  # [1, N_patches, hidden_dim]
        else:
            x = emb_A_vs_B  # [B, N_patches, hidden_dim]

        # Self-attention layers (batch_first=True handles batched input)
        for layer in self.self_attn_layers:
            x = layer(x)  # [B, N_patches, hidden_dim]

        # Global average pooling
        pooled = x.mean(dim=1)  # [B, hidden_dim]

        # MLP to scalar(s)
        sim_output = self.mlp(pooled)  # [B, 1]

        # Return in same format as input
        if not input_is_batched:
            return sim_output.squeeze()  # scalar
        else:
            return sim_output.squeeze(-1)  # [B]