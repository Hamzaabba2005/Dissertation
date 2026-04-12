import numpy as np
from sklearn.metrics import f1_score


def tune_thresholds_per_class(probs, targets, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    probs = np.asarray(probs)
    targets = np.asarray(targets).astype(int)

    n_classes = probs.shape[1]
    best_thresholds = np.zeros(n_classes, dtype=np.float32)
    best_f1s = np.zeros(n_classes, dtype=np.float32)

    for c in range(n_classes):
        y_true = targets[:, c]
        y_prob = probs[:, c]

        best_t = 0.5
        best_f1 = -1.0

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[c] = best_t
        best_f1s[c] = best_f1

    return best_thresholds, best_f1s


def predict_with_thresholds(probs, thresholds):
    probs = np.asarray(probs)
    thresholds = np.asarray(thresholds).reshape(1, -1)
    return (probs >= thresholds).astype(int)


def evaluate_with_thresholds(probs, targets, thresholds):
    targets = np.asarray(targets).astype(int)
    preds = predict_with_thresholds(probs, thresholds)

    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
    exact_match = (preds == targets).all(axis=1).mean()
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "exact_match": exact_match,
        "per_class_f1": per_class_f1,
        "preds": preds,
    }


def threshold_report(label_cols, base_per_class_f1, tuned_per_class_f1, best_thresholds):
    rows = []
    for i, label in enumerate(label_cols):
        rows.append({
            "class": label,
            "threshold": float(best_thresholds[i]),
            "f1_default": float(base_per_class_f1[i]),
            "f1_tuned": float(tuned_per_class_f1[i]),
            "gain": float(tuned_per_class_f1[i] - base_per_class_f1[i]),
        })
    return rows