import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from fairlearn.metrics import MetricFrame, selection_rate
from functools import partial
from sklearn.metrics import classification_report

# Define scoring functions that won't raise errors when encountering zero division
precision_z0 = partial(precision_score, zero_division=0)
recall_z0    = partial(recall_score, zero_division=0)
f1_z0        = partial(f1_score, zero_division=0)

def evaluate_model(
    all_targets,
    all_probabilities,
    all_skin_tones,
    threshold: float,
    val_loss: float = None
) -> None:
    """
    Evaluate a classification model on performance and fairness metrics.

    Args:
        all_lbls: Ground truth labels (0 or 1).
        all_probs: Predicted probabilities for the positive class.
        all_skin: Sensitive attribute values (e.g., skin tone group).
        threshold (float): Probability threshold to convert probabilities to class predictions.
        val_loss (float, optional): Validation loss to report alongside metrics.
    """

    # Convert inputs to NumPy arrays for consistency
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    all_skin_tones = np.array(all_skin_tones)

    # Convert probabilities to binary predictions using threshold
    predictions = (all_probabilities > threshold).astype(int)

    # Compute overall accuracy in percentage
    acc = 100. * (predictions == all_targets).mean()

    print("\n========== MODEL EVALUATION ==========")
    if val_loss is not None:
        print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {acc:.2f}%")
    print(f"Optimal Threshold: {threshold:.4f}")

    # Display standard classification metrics
    print("\nClassification Report:")
    print(classification_report(all_targets, predictions, target_names=["Benign", "Malignant"], zero_division=0))

    # Compute fairness-aware metrics grouped by sensitive attribute (e.g., skin tone)
    fair_metrics = MetricFrame(
        metrics={
            "accuracy":       accuracy_score,
            "recall":         recall_z0,
            "precision":      precision_z0,
            "f1":             f1_z0,
            "selection_rate": selection_rate,  # % of samples predicted positive
        },
        y_true=all_targets,
        y_pred=predictions,
        sensitive_features=all_skin_tones,
    )

    # Display group-wise performance
    print("\nFairness Metrics by Skin Tone:")
    print(fair_metrics.by_group)

    # Show disparities between groups (e.g., max difference in accuracy)
    print("\nMax-Min Disparities:")
    print(fair_metrics.difference(method='between_groups'))
    print("======================================\n")
