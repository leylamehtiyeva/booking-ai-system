from __future__ import annotations

RELATIONS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_EVIDENCE"]


def safe_divide(num: float, denom: float) -> float | None:
    if denom == 0:
        return None
    return num / denom


def compute_confusion_matrix(rows: list[dict], pred_key: str = "predicted_relation") -> dict:
    matrix = {gold: {pred: 0 for pred in RELATIONS} for gold in RELATIONS}
    for row in rows:
        gold = row["gold_relation"]
        pred = row.get(pred_key)
        if pred not in RELATIONS:
            continue  # parse failures excluded from the confusion matrix, counted separately
        matrix[gold][pred] += 1
    return matrix


def compute_per_class(rows: list[dict], pred_key: str = "predicted_relation") -> dict:
    """
    sklearn-style zero_division=0: a class with gold examples (n_gold>0)
    but zero predictions gets F1=0.0, not None/dropped. Only a class
    with n_gold==0 is genuinely insufficient_data.
    """
    scored_rows = [r for r in rows if r.get(pred_key) in RELATIONS]
    per_class = {}
    for label in RELATIONS:
        tp = sum(r["gold_relation"] == label and r[pred_key] == label for r in scored_rows)
        fp = sum(r["gold_relation"] != label and r[pred_key] == label for r in scored_rows)
        fn = sum(r["gold_relation"] == label and r[pred_key] != label for r in scored_rows)
        n_gold = sum(r["gold_relation"] == label for r in rows)
        insufficient_data = n_gold == 0

        raw_precision = safe_divide(tp, tp + fp)
        raw_recall = safe_divide(tp, tp + fn)

        if insufficient_data:
            f1 = None
        else:
            eff_p = raw_precision if raw_precision is not None else 0.0
            eff_r = raw_recall if raw_recall is not None else 0.0
            f1 = 0.0 if (eff_p + eff_r) == 0 else 2 * eff_p * eff_r / (eff_p + eff_r)

        per_class[label] = {
            "precision": round(raw_precision, 4) if raw_precision is not None else None,
            "recall": round(raw_recall, 4) if raw_recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "tp": tp, "fp": fp, "fn": fn, "n_gold": n_gold,
            "insufficient_data": insufficient_data,
        }
    return per_class


def compute_macro_f1(per_class: dict) -> float | None:
    values = [per_class[label]["f1"] for label in RELATIONS if not per_class[label]["insufficient_data"]]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def compute_accuracy(rows: list[dict], pred_key: str = "predicted_relation") -> float | None:
    scored_rows = [r for r in rows if r.get(pred_key) in RELATIONS]
    if not scored_rows:
        return None
    correct = sum(r["gold_relation"] == r[pred_key] for r in scored_rows)
    return round(correct / len(scored_rows), 4)


def compute_transition_rate(rows: list[dict], gold_label: str, predicted_label: str, pred_key: str = "predicted_relation") -> dict:
    subset = [r for r in rows if r["gold_relation"] == gold_label]
    n = len(subset)
    count = sum(r.get(pred_key) == predicted_label for r in subset)
    return {"rate": round(count / n, 4) if n else None, "count": count, "n": n, "insufficient_data": n == 0}


def compute_false_relation_rate(rows: list[dict], relation: str, pred_key: str = "predicted_relation") -> dict:
    """False X rate = predicted X when gold != X, among all gold != X."""
    subset = [r for r in rows if r["gold_relation"] != relation]
    n = len(subset)
    count = sum(r.get(pred_key) == relation for r in subset)
    return {"rate": round(count / n, 4) if n else None, "count": count, "n": n, "insufficient_data": n == 0}


def compute_metrics_block(rows: list[dict], pred_key: str = "predicted_relation") -> dict:
    per_class = compute_per_class(rows, pred_key)
    n_parse_failures = sum(1 for r in rows if r.get(pred_key) not in RELATIONS)
    return {
        "n": len(rows),
        "n_parse_failures": n_parse_failures,
        "accuracy": compute_accuracy(rows, pred_key),
        "confusion_matrix": compute_confusion_matrix(rows, pred_key),
        "per_class": per_class,
        "macro_f1": compute_macro_f1(per_class),
        "reliability_critical": {
            "NEE_to_SUPPORT_rate": compute_transition_rate(rows, "NOT_ENOUGH_EVIDENCE", "SUPPORT", pred_key),
            "CONTRADICT_to_SUPPORT_rate": compute_transition_rate(rows, "CONTRADICT", "SUPPORT", pred_key),
            "false_SUPPORT_rate": compute_false_relation_rate(rows, "SUPPORT", pred_key),
            "SUPPORT_to_NEE_rate": compute_transition_rate(rows, "SUPPORT", "NOT_ENOUGH_EVIDENCE", pred_key),
            "SUPPORT_to_CONTRADICT_rate": compute_transition_rate(rows, "SUPPORT", "CONTRADICT", pred_key),
            "false_CONTRADICT_rate": compute_false_relation_rate(rows, "CONTRADICT", pred_key),
        },
    }


def deduplicate_by_template(rows: list[dict]) -> list[dict]:
    """
    One representative row per template_group_id (first occurrence),
    plus all singleton rows (template_group_id is None) - prevents a
    repeated Booking template from being counted as multiple
    independent semantic successes/failures.
    """
    seen_groups: set[str] = set()
    out = []
    for r in rows:
        gid = r.get("template_group_id")
        if gid is None:
            out.append(r)
            continue
        if gid in seen_groups:
            continue
        seen_groups.add(gid)
        out.append(r)
    return out


def template_consistency(rows: list[dict], pred_key: str = "predicted_relation") -> dict:
    """For each template group, did the model predict the SAME relation across all its (property-varying) instances?"""
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        gid = r.get("template_group_id")
        if gid is not None:
            by_group.setdefault(gid, []).append(r)

    results = {}
    for gid, group in by_group.items():
        preds = {r.get(pred_key) for r in group}
        results[gid] = {
            "n_instances": len(group),
            "predictions": [r.get(pred_key) for r in group],
            "consistent": len(preds) == 1,
        }
    n_consistent = sum(1 for v in results.values() if v["consistent"])
    return {"n_groups": len(results), "n_consistent": n_consistent, "groups": results}
