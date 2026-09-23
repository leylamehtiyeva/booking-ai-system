"""
Canonical NLI sanity test - textbook entailment/neutral/contradiction
pairs with unambiguous correct answers, run through the SAME model
wrapper as the Booking oracle experiment.

Purpose: rule out "the model/tokenizer/wrapper integration is broken"
as an explanation for the poor Booking-domain results (e.g. the
"quiet" hypothesis scoring 10% accuracy, or the model never predicting
"neutral" at all on the Booking set). If the model handles these
textbook cases well, the Booking-domain failures are a genuine
domain/phrasing behavior of the model, not an integration bug.

Deliberately NOT mixed into the Booking checkpoint report - separate
dataset, separate output file.

Run with the experiment's isolated environment:
    evaluation/experiments/nli_oracle/.venv/bin/python -m evaluation.experiments.nli_oracle.sanity_check
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.core.io import save_json
from evaluation.experiments.nli_oracle.metrics import compute_metrics_block
from evaluation.experiments.nli_oracle.model import MODEL_NAME, NliOracleModel

OUTPUT_PATH = PROJECT_ROOT / "evaluation/outputs/nli_oracle_sanity_check_report.json"

CASES = [
    # entailment (-> SUPPORT)
    {
        "id": "ent_1",
        "premise": "A man is playing guitar on stage.",
        "hypothesis": "A person is performing music.",
        "gold_relation": "SUPPORT",
    },
    {
        "id": "ent_2",
        "premise": "The dog is sleeping on the sofa.",
        "hypothesis": "An animal is resting.",
        "gold_relation": "SUPPORT",
    },
    {
        "id": "ent_3",
        "premise": "She bought three apples and two oranges.",
        "hypothesis": "She purchased fruit.",
        "gold_relation": "SUPPORT",
    },
    # neutral (-> RELEVANT_NEUTRAL)
    {
        "id": "neu_1",
        "premise": "The chef prepared a three-course meal.",
        "hypothesis": "The meal was delicious.",
        "gold_relation": "RELEVANT_NEUTRAL",
    },
    {
        "id": "neu_2",
        "premise": "John walked to the store.",
        "hypothesis": "John walked to the store to buy milk.",
        "gold_relation": "RELEVANT_NEUTRAL",
    },
    {
        "id": "neu_3",
        "premise": "The hotel has a swimming pool.",
        "hypothesis": "The swimming pool is heated.",
        "gold_relation": "RELEVANT_NEUTRAL",
    },
    # contradiction (-> CONTRADICT)
    {
        "id": "con_1",
        "premise": "The store is open 24 hours a day.",
        "hypothesis": "The store closes at 9pm every night.",
        "gold_relation": "CONTRADICT",
    },
    {
        "id": "con_2",
        "premise": "All rooms have private bathrooms.",
        "hypothesis": "Some rooms share a bathroom with other guests.",
        "gold_relation": "CONTRADICT",
    },
    {
        "id": "con_3",
        "premise": "The car is red.",
        "hypothesis": "The car is blue.",
        "gold_relation": "CONTRADICT",
    },
]


def run() -> dict:
    print(f"Loading model {MODEL_NAME} for canonical sanity check...")
    model = NliOracleModel()
    print(f"Running {len(CASES)} canonical NLI cases...")

    examples = []
    for case in CASES:
        result = model.predict(case["premise"], case["hypothesis"])
        examples.append(
            {
                **case,
                "raw_label": result.raw_label,
                "predicted_relation": result.predicted_relation,
                "probs": result.probs,
                "correct": result.predicted_relation == case["gold_relation"],
            }
        )

    metrics = compute_metrics_block(examples)

    report = {
        "run_metadata": {
            "model": MODEL_NAME,
            "purpose": "Canonical NLI sanity check - NOT the Booking domain benchmark. "
            "Confirms the model/wrapper behaves correctly on unambiguous textbook cases.",
            "n_cases": len(CASES),
        },
        "metrics": metrics,
        "examples": examples,
    }

    save_json(OUTPUT_PATH, report)

    print()
    print("=== CANONICAL NLI SANITY CHECK ===")
    print(f"accuracy: {metrics['accuracy']} ({sum(e['correct'] for e in examples)}/{len(examples)})")
    print("confusion matrix:", metrics["confusion_matrix"])
    for ex in examples:
        mark = "OK" if ex["correct"] else "FAIL"
        print(f"  [{mark}] {ex['id']}: gold={ex['gold_relation']} pred={ex['predicted_relation']} probs={ex['probs']}")
    print()
    print(f"Saved to: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()
