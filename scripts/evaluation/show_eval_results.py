import sys
import json
from collections import defaultdict
import pandas as pd


DEFAULT_REPORT_PATH = "evaluation/outputs/constraint_extraction_report.json"


def load_report(report_path: str):
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_metrics(report):
    print("\n=== OVERALL METRICS ===\n")
    metrics = report["metrics"]

    for k, v in metrics.items():
        if k == "counts":
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def show_group_breakdown(report):
    print("\n=== GROUP BREAKDOWN ===\n")

    cases = report["cases"]

    group_stats = defaultdict(lambda: {
        "cases": 0,
        "gold": 0,
        "pred": 0,
        "correct": 0,
        "priority_total": 0,
        "priority_correct": 0,
    })

    for c in cases:
        group = c["group"]
        group_stats[group]["cases"] += 1

        ce = c["constraint_extraction"]
        group_stats[group]["gold"] += ce["gold_count"]
        group_stats[group]["pred"] += ce["pred_count"]
        group_stats[group]["correct"] += ce["correct_count"]

        for p in c.get("priority_rows", []):
            group_stats[group]["priority_total"] += 1
            if p["priority_correct"]:
                group_stats[group]["priority_correct"] += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    rows = []
    for group, st in group_stats.items():
        precision = safe_div(st["correct"], st["pred"])
        recall = safe_div(st["correct"], st["gold"])
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append({
            "group": group,
            "cases": st["cases"],
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "priority_acc": round(safe_div(st["priority_correct"], st["priority_total"]), 3),
        })

    df = pd.DataFrame(rows).sort_values("group")
    print(df.to_string(index=False))


def show_bad_cases(report, n=5):
    print(f"\n=== WORST CASES (top {n}) ===\n")

    bad_cases = []

    for c in report["cases"]:
        if "error" in c:
            bad_cases.append({
                "case_id": c["case_id"],
                "group": c["group"],
                "user_message": c["user_message"],
                "problem": c["error"],
            })
            continue

        ce = c["constraint_extraction"]
        wrong_priorities = [p for p in c.get("priority_rows", []) if not p["priority_correct"]]

        if ce["missed_constraints"] or ce["extra_constraints"] or wrong_priorities:
            bad_cases.append({
                "case_id": c["case_id"],
                "group": c["group"],
                "user_message": c["user_message"],
                "missed": ce["missed_constraints"],
                "extra": ce["extra_constraints"],
                "wrong_priorities": wrong_priorities[:2],
            })

    for case in bad_cases[:n]:
        print("\n---")
        print("ID:", case["case_id"])
        print("GROUP:", case["group"])
        print("TEXT:", case["user_message"])

        if "problem" in case:
            print("ERROR:", case["problem"])
        else:
            print("MISSED:", case["missed"])
            print("EXTRA:", case["extra"])
            print("WRONG PRIORITY:", case["wrong_priorities"])


def main():
    report_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPORT_PATH
    report = load_report(report_path)

    print(f"Using report: {report_path}")
    print_metrics(report)
    show_group_breakdown(report)
    show_bad_cases(report, n=10)


if __name__ == "__main__":
    main()