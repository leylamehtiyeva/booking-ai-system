import json
from collections import Counter

REPORT_PATH = "evaluation/outputs/constraint_extraction_report.json"


def load_report():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_policy_name(name: str) -> bool:
    return name in {
        "pet_friendly",
        "smoking_allowed",
        "non_smoking",
        "parties_allowed",
        "children_allowed",
        "parking",
        "free_cancellation",
    }


def classify_case_risk(case: dict) -> list[str]:
    ce = case["constraint_extraction"]
    risks = []

    # 1. Any missed constraints
    for missed in ce["missed_constraints"]:
        risks.append(f"missed_constraint::{missed}")

    # 2. Exact-set mismatch in multi-constraint case
    if ce["gold_count"] >= 2 and not ce["exact_constraint_set_match"]:
        risks.append("multi_constraint_set_mismatch")

    # 3. Check matched rows for dangerous priority drift
    for row in case["matched_rows"]:
        gold_p = row["gold_priority"]
        pred_p = row["pred_priority"]
        name = row["normalized_text"]

        # must -> nice
        if gold_p == "must" and pred_p == "nice":
            risks.append(f"must_downgraded_to_nice::{name}")

        # forbidden -> nice or must
        if gold_p == "forbidden" and pred_p in {"nice", "must"}:
            risks.append(f"forbidden_weakened::{name}")

        # priority mismatch on policy constraints
        if gold_p != pred_p and is_policy_name(name):
            risks.append(f"policy_priority_mismatch::{name}")

        # mapped_fields mismatch on policy constraints
        if (not row["mapped_fields_exact_match"]) and is_policy_name(name):
            risks.append(f"policy_mapping_mismatch::{name}")

    return sorted(set(risks))


def is_class_a(case: dict) -> bool:
    risks = classify_case_risk(case)

    for r in risks:
        if r.startswith("missed_constraint::"):
            return True
        if r.startswith("must_downgraded_to_nice::"):
            return True
        if r.startswith("forbidden_weakened::"):
            return True
        if r.startswith("policy_priority_mismatch::"):
            return True
        if r.startswith("policy_mapping_mismatch::"):
            return True
        if r == "multi_constraint_set_mismatch":
            return True

    return False


def print_case(case: dict, risks: list[str]):
    ce = case["constraint_extraction"]

    print("\n" + "=" * 100)
    print("ID:", case["case_id"])
    print("GROUP:", case["group"])
    print("TEXT:", case["user_message"])
    print("RISKS:", risks)
    print()
    print("gold_count:", ce["gold_count"])
    print("pred_count:", ce["pred_count"])
    print("correct_count:", ce["correct_count"])
    print("missed:", ce["missed_constraints"])
    print("extra:", ce["extra_constraints"])
    print("exact_constraint_set_match:", ce["exact_constraint_set_match"])
    print("exact_full_case_match:", ce["exact_full_case_match"])

    print("\nMATCHED ROWS:")
    if not case["matched_rows"]:
        print("  <none>")
    else:
        for row in case["matched_rows"]:
            print("  -", row["normalized_text"])
            print(
                "    priority:",
                f'{row["gold_priority"]} -> {row["pred_priority"]}',
                "| category:",
                f'{row["gold_category"]} -> {row["pred_category"]}',
            )
            print(
                "    mapping_status_correct:",
                row["mapping_status_correct"],
                "| mapped_fields_exact_match:",
                row["mapped_fields_exact_match"],
                "| evidence_strategy_correct:",
                row["evidence_strategy_correct"],
            )


def main():
    report = load_report()
    cases = report["cases"]

    risky_cases = []
    risk_counter = Counter()

    for case in cases:
        risks = classify_case_risk(case)
        if is_class_a(case):
            risky_cases.append((case, risks))
            risk_counter.update(risks)

    print("CLASS A PRODUCT-RISK CASES:", len(risky_cases))
    print("\nTOP RISK TAGS:")
    for k, v in risk_counter.most_common(20):
        print(f"{k}: {v}")

    print("\nFIRST 15 CLASS A CASES:")
    for case, risks in risky_cases[:15]:
        print_case(case, risks)


if __name__ == "__main__":
    main()