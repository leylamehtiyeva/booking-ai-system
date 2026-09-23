from __future__ import annotations

RELATIONS = ["SUPPORT", "CONTRADICT", "RELEVANT_NEUTRAL"]


def safe_divide(num: float, denom: float) -> float | None:
    if denom == 0:
        return None
    return num / denom


def compute_confusion_matrix(rows: list[dict]) -> dict:
    matrix = {gold: {pred: 0 for pred in RELATIONS} for gold in RELATIONS}
    for row in rows:
        matrix[row["gold_relation"]][row["predicted_relation"]] += 1
    return matrix


def compute_per_class(rows: list[dict]) -> dict:
    """
    Per-class precision/recall/F1 with sklearn-style zero_division=0
    convention: a class with gold examples (n_gold > 0) that was never
    predicted has precision treated as 0 (not undefined), so its F1 is
    a real 0.0 - not None. Only a class with n_gold == 0 (no ground
    truth instances at all, e.g. CONTRADICT before real examples
    exist) is genuinely insufficient_data and gets f1=None, excluded
    from macro averages.

    Earlier version conflated these two cases: "never predicted" was
    silently treated the same as "no ground truth", which caused
    RELEVANT_NEUTRAL (n_gold=5, but 0 TP since the model never
    predicted it) to be dropped from macro_f1_observed_classes even
    though it should count as a real F1=0 data point.
    """
    per_class = {}
    for label in RELATIONS:
        tp = sum(r["gold_relation"] == label and r["predicted_relation"] == label for r in rows)
        fp = sum(r["gold_relation"] != label and r["predicted_relation"] == label for r in rows)
        fn = sum(r["gold_relation"] == label and r["predicted_relation"] != label for r in rows)
        n_gold = sum(r["gold_relation"] == label for r in rows)
        insufficient_data = n_gold == 0

        raw_precision = safe_divide(tp, tp + fp)
        raw_recall = safe_divide(tp, tp + fn)

        if insufficient_data:
            precision = raw_precision
            recall = raw_recall
            f1 = None
        else:
            # zero_division=0 convention: undefined precision (class
            # never predicted, tp+fp==0) counts as precision=0 for F1
            # purposes, since there ARE gold instances to have missed.
            eff_precision = raw_precision if raw_precision is not None else 0.0
            eff_recall = raw_recall if raw_recall is not None else 0.0
            precision = raw_precision
            recall = raw_recall
            f1 = 0.0 if (eff_precision + eff_recall) == 0 else 2 * eff_precision * eff_recall / (eff_precision + eff_recall)

        per_class[label] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_gold": n_gold,
            "insufficient_data": insufficient_data,
        }
    return per_class


def compute_macro_f1(per_class: dict, labels: list[str]) -> float | None:
    """
    Averages F1 only over classes that are NOT insufficient_data
    (n_gold > 0). A class with n_gold > 0 but zero predictions still
    contributes its F1=0.0 - it is not silently skipped.
    """
    f1_values = [
        per_class[label]["f1"]
        for label in labels
        if not per_class[label]["insufficient_data"]
    ]
    if not f1_values:
        return None
    return round(sum(f1_values) / len(f1_values), 4)


def compute_accuracy(rows: list[dict]) -> float | None:
    if not rows:
        return None
    correct = sum(r["gold_relation"] == r["predicted_relation"] for r in rows)
    return round(correct / len(rows), 4)


def compute_transition_rate(rows: list[dict], gold_label: str, predicted_label: str) -> dict:
    subset = [r for r in rows if r["gold_relation"] == gold_label]
    n = len(subset)
    count = sum(r["predicted_relation"] == predicted_label for r in subset)
    return {
        "rate": round(count / n, 4) if n else None,
        "count": count,
        "n": n,
        "insufficient_data": n == 0,
    }


def compute_metrics_block(rows: list[dict]) -> dict:
    per_class = compute_per_class(rows)
    return {
        "n": len(rows),
        "accuracy": compute_accuracy(rows),
        "confusion_matrix": compute_confusion_matrix(rows),
        "per_class": per_class,
        "macro_f1_observed_classes": compute_macro_f1(per_class, ["SUPPORT", "RELEVANT_NEUTRAL"]),
        "macro_f1_3class": {
            "value": None,
            "insufficient_data": True,
            "reason": "0 CONTRADICT examples in the current dataset - a 3-class macro F1 "
            "would be undefined/misleading (recall for CONTRADICT is 0/0). "
            "Recompute once real CONTRADICT examples exist.",
        }
        if per_class["CONTRADICT"]["insufficient_data"]
        else {"value": compute_macro_f1(per_class, RELATIONS), "insufficient_data": False},
    }
