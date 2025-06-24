import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    """
    Tversky Loss for imbalanced segmentation tasks.
    A generalization of Dice Loss that adds separate weights for false positives and false negatives.

    Args:
        alpha (float): Weighting factor for false positives (FP).
        beta (float): Weighting factor for false negatives (FN).
        smooth (float): Small constant added to numerator and denominator for numerical stability.
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha   # weight for FP in denominator
        self.beta = beta     # weight for FN in denominator
        self.smooth = smooth # smoothing constant to avoid division by zero

    def forward(self, logits: torch.Tensor, ground_truth_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the Tversky loss between model logits and the ground-truth mask.

        Args:
            logits: Raw model outputs (logits), shape (N, *).
            ground_truth_mask: Ground-truth binary mask, same shape as logits.

        Returns:
            Scalar Tversky loss averaged over the batch.
        """

        # Convert logits to probabilities via sigmoid
        probabilities = torch.sigmoid(logits)

        # Flatten tensors to 1D for computing TP, FP, FN
        probabilities_flattened = probabilities.view(-1)
        ground_truth_flattened = ground_truth_mask.view(-1)

        # Calculate true positive, false positive, and false negative counts
        true_positive_count = (probabilities_flattened * ground_truth_flattened).sum()
        false_positive_count = (probabilities_flattened * (1 - ground_truth_flattened)).sum()
        false_negative_count = ((1 - probabilities_flattened) * ground_truth_flattened).sum()

        # Compute the Tversky index
        tversky_index = (
            true_positive_count + self.smooth
        ) / (
            true_positive_count
            + self.alpha * false_positive_count
            + self.beta * false_negative_count
            + self.smooth
        )

        # Tversky loss is 1 minus the index
        tversky_loss = 1 - tversky_index
        return tversky_loss
