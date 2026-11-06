import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SoftNearestNeighborsLoss(nn.Module):
    """
    Soft-Nearest Neighbors Loss for contrastive learning (Formulation B).

    For each anchor, computes:
    L = -log( Σ_p exp(sim(a,p)/τ) / (Σ_p exp(sim(a,p)/τ) + Σ_n exp(sim(a,n)/τ)) )

    where:
    - p: positive clips (from same anchor region)
    - n: negative clips (first-source + second-source + third-source)

    Args:
        temperature: Temperature parameter τ for scaling (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        pos_similarities: List[torch.Tensor],
        first_source_neg_similarities: List[torch.Tensor],
        second_source_neg_similarities: List[torch.Tensor],
        third_source_neg_similarities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Soft-Nearest Neighbors Loss.

        Args:
            pos_similarities: List of [num_positives_i] tensors for each anchor
            first_source_neg_similarities: List of [num_first_negs_i] tensors
            second_source_neg_similarities: List of [num_second_negs_i] tensors
            third_source_neg_similarities: [B, B-1] tensor (anchor vs other anchors)

        Returns:
            loss: SNNL averaged over the batch
        """
        batch_size = len(pos_similarities)
        device = pos_similarities[0].device

        losses = []
        for i in range(batch_size):
            # Positive logits
            pos_logits = pos_similarities[i] / self.temperature  # [num_pos]

            # Negative logits from all three sources
            neg_logits_list = []

            if len(first_source_neg_similarities[i]) > 0:
                neg_logits_list.append(first_source_neg_similarities[i] / self.temperature)

            if len(second_source_neg_similarities[i]) > 0:
                neg_logits_list.append(second_source_neg_similarities[i] / self.temperature)

            # Third-source: similarities with other anchors
            neg_logits_list.append(third_source_neg_similarities[i] / self.temperature)

            neg_logits = torch.cat(neg_logits_list)  # [total_negs]

            # Compute SNNL (Formulation B)
            # L = -log( Σ exp(pos) / (Σ exp(pos) + Σ exp(neg)) )
            pos_exp_sum = torch.logsumexp(pos_logits, dim=0)  # log(Σ exp(pos))
            all_logits = torch.cat([pos_logits, neg_logits])
            all_exp_sum = torch.logsumexp(all_logits, dim=0)  # log(Σ exp(all))

            loss_i = all_exp_sum - pos_exp_sum  # -log(pos_sum / all_sum)
            losses.append(loss_i)

        return torch.stack(losses).mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning (kept for backward compatibility).

    Loss = -log(exp(sim(anchor, positive)/τ) / Σ exp(sim(anchor, all_samples)/τ))
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        similarity_matrix: torch.Tensor,
        explicit_neg_similarities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss from pre-computed similarity scores.

        Args:
            similarity_matrix: [B, B] where diagonal = positive pairs
            explicit_neg_similarities: Optional [B] explicit negatives

        Returns:
            loss: InfoNCE loss averaged over the batch
        """
        batch_size = similarity_matrix.shape[0]
        device = similarity_matrix.device

        similarity_matrix = similarity_matrix / self.temperature

        pos_similarity = similarity_matrix.diagonal().unsqueeze(1)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_similarity = similarity_matrix[mask].view(batch_size, -1)

        if explicit_neg_similarities is not None:
            explicit_neg = (explicit_neg_similarities / self.temperature).unsqueeze(1)
            neg_similarity = torch.cat([neg_similarity, explicit_neg], dim=1)

        logits = torch.cat([pos_similarity, neg_similarity], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        return F.cross_entropy(logits, labels)
