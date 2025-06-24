import numpy as np
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, recall_score, precision_score
from collections           import defaultdict

def find_best_threshold_balanced(y_true, y_prob, groups, β=1.0, num_thresholds=100):
    """
    Find the best threshold balancing global F1 and worst-group recall.

    Args:
        y_true: ground-truth labels
        y_prob: predicted probabilities
        groups: sensitive attribute (skin tone)
        β: beta for Fβ score (currently unused, you could adjust Fβ if wanted)
        num_thresholds: number of thresholds to evaluate
    Returns:
        best threshold
    """

    # Subsample thresholds uniformly between min and max probability
    T_sampled = np.linspace(min(y_prob), max(y_prob), num_thresholds)

    best_score = -np.inf
    best_thr = 0.5

    for thr in T_sampled:
        preds = (y_prob >= thr).astype(int)

        # Global precision, recall, F1
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = (1 + β**2) * p * r / (β**2 * p + r + 1e-8)

        # Per-group recall
        rec_by_group = defaultdict(list)
        for y, g, yp in zip(y_true, groups, preds):
            rec_by_group[g].append((y, yp))

        worst_rec = min(
            recall_score([y for y, _ in recs], [yp for _, yp in recs])
            for recs in rec_by_group.values()
        )

        # Composite objective: global F1 + worst recall
        score = f1 + worst_rec

        if score > best_score:
            best_score, best_thr = score, thr

    return best_thr
