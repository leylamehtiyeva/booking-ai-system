import json

REPORT_PATH = "evaluation/outputs/constraint_extraction_report.json"


def load_report():
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def print_case(case: dict):
    ce = case["constraint_extraction"]

    print("\n" + "=" * 100)
    print("ID:", case["case_id"])
    print("GROUP:", case["group"])
    print("TEXT:", case["user_message"])
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
            print("  - normalized_text:", row["normalized_text"])
            print("    gold_priority:", row["gold_priority"], "| pred_priority:", row["pred_priority"])
            print("    gold_category:", row["gold_category"], "| pred_category:", row["pred_category"])
            print("    mapping_status_correct:", row["mapping_status_correct"])
            print("    mapped_fields_exact_match:", row["mapped_fields_exact_match"])
            print("    evidence_strategy_correct:", row["evidence_strategy_correct"])
            print("    fully_correct_on_matched_constraint:", row["fully_correct_on_matched_constraint"])
            print()


def main():
    report = load_report()
    cases = report["cases"]

    # Только multi-constraint кейсы с ошибками
    bad_multi_cases = []
    for c in cases:
        ce = c["constraint_extraction"]
        if ce["gold_count"] >= 2 and not ce["exact_full_case_match"]:
            bad_multi_cases.append(c)

    print("Problematic multi-constraint cases:", len(bad_multi_cases))

    for case in bad_multi_cases[:15]:
        print_case(case)


if __name__ == "__main__":
    main()
