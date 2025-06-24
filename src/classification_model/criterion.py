import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from common.constants import ColumnNames

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks.
    Focuses learning on hard-to-classify examples by down-weighting easy ones.

    Args:
        alpha (float): Weighting factor for the loss (balances positive vs. negative examples).
        gamma (float): Focusing parameter that reduces the loss contribution from easy examples.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha   # weight for the focal term
        self.gamma = gamma   # focusing parameter
        # BCEWithLogitsLoss combines a sigmoid layer and the BCELoss in one class.
        # reduction='none' so we can apply the focal term per element before averaging.
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, ground_truth_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between `logits` and `ground_truth_labels`.

        Args:
            logits: Raw model outputs (logits), shape (N, *).
            ground_truth_labels: Ground-truth labels (0 or 1), same shape as logits.

        Returns:
            Scalar focal loss averaged over the batch.
        """

        # Compute the un-reduced element-wise binary cross-entropy loss
        binary_cross_entropy_loss = self.bce_with_logits(logits, ground_truth_labels)

        # Convert logits to predicted probabilities
        predicted_probabilities = torch.sigmoid(logits)
        # Select p_t: probability of the true class for each example
        probability_true_class = (
            predicted_probabilities * ground_truth_labels
            + (1 - predicted_probabilities) * (1 - ground_truth_labels)
        )

        # Compute the focal scaling factor (1 - p_t)^gamma
        focal_scaling_factor = (1 - probability_true_class).pow(self.gamma)

        # Scale the BCE loss by the focal term and alpha
        focal_loss = self.alpha * focal_scaling_factor * binary_cross_entropy_loss

        # Return the mean focal loss over all elements
        return focal_loss.mean()
    
class ClassBalancedFocalLoss(nn.Module):
    """
    Implements Class-Balanced Focal Loss with per-(skin_tone, class) weighting.

    References:
      Cui et al. “Class-Balanced Loss Based on Effective Number of Samples”
      https://arxiv.org/abs/1901.05555

    Args:
        metadata_df: pd.DataFrame with columns [ColumnNames.SKIN_TONE, ColumnNames.TARGET]
        beta: float in [0,1). Closer to 1.0 → more aggressive re-weighting of rare classes.
        gamma: focal-loss focusing parameter γ ≥ 0
    """
    def __init__(self, metadata_df: pd.DataFrame, beta: float = 0.9999, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

        # 1) compute counts per (skin_tone, class)
        counts = metadata_df.groupby(
            [ColumnNames.SKIN_TONE, ColumnNames.TARGET]
        ).size()

        # 2) compute α = (1 - β) / (1 - β^n) and build a tensor lookup
        #    assume skin_tone labels are 0..(max_tone)
        max_tone = int(metadata_df[ColumnNames.SKIN_TONE].max())
        num_classes = int(metadata_df[ColumnNames.TARGET].max()) + 1

        alphas = torch.zeros(max_tone + 1, num_classes)
        for (tone, cls), n_samples in counts.items():
            effective_num = 1.0 - beta**n_samples
            weight = (1.0 - beta) / effective_num
            alphas[tone, int(cls)] = weight

        # register as buffer so it moves with .to(device) and saved in checkpoints
        self.register_buffer('alphas', alphas)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skin_tones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            logits: Tensor of shape [B] (raw model outputs)
            targets: Tensor of shape [B], values 0 or 1
            skin_tones: Tensor of shape [B], integer skin-tone group IDs

        Returns:
            scalar loss tensor
        """
        # convert logits → probabilities
        probs = torch.sigmoid(logits)                # [B]
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)  # [B]

        # focal term: (1 - p_t)^γ
        focal_weight = (1.0 - pt).pow(self.gamma)     # [B]

        # lookup per-sample α from precomputed buffer
        sample_alphas = self.alphas[skin_tones.long(), targets.long()]  # [B]

        # compute binary cross-entropy per sample
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )  # [B]

        # final weighted focal loss
        loss = sample_alphas * focal_weight * bce_loss  # [B]
        return loss.mean()
