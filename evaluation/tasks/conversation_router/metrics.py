from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from collections import Counter, defaultdict
from app.schemas.conversation_route import ConversationAction
from evaluation.tasks.conversation_router.runner import (
    RouterEvalPrediction,
)


ALL_ACTIONS = list(ConversationAction)


def _calculate_case_level_metrics(
    predictions: list[RouterEvalPrediction],
) -> dict[str, int | float]:
    grouped: dict[
        str,
        list[RouterEvalPrediction],
    ] = defaultdict(list)

    for row in predictions:
        grouped[row.case_id].append(row)

    total_cases = len(grouped)

    majority_correct_cases = 0
    consistent_cases = 0
    cases_without_majority = 0

    for case_id, rows in grouped.items():
        expected_actions = {
            row.expected_action
            for row in rows
        }

        if len(expected_actions) != 1:
            raise ValueError(
                f"Inconsistent expected_action "
                f"for case {case_id}"
            )

        expected_action = next(
            iter(expected_actions)
        )

        all_successful = all(
            row.success
            and row.predicted_action is not None
            for row in rows
        )

        predicted_actions = [
            row.predicted_action
            for row in rows
            if (
                row.success
                and row.predicted_action is not None
            )
        ]

        if (
            all_successful
            and len(set(predicted_actions)) == 1
        ):
            consistent_cases += 1

        action_counts = Counter(
            predicted_actions
        )

        if not action_counts:
            cases_without_majority += 1
            continue

        majority_action, majority_count = (
            action_counts.most_common(1)[0]
        )

        has_strict_majority = (
            majority_count
            > len(rows) / 2
        )

        if not has_strict_majority:
            cases_without_majority += 1
            continue

        if majority_action == expected_action:
            majority_correct_cases += 1

    return {
        "unique_cases": total_cases,
        "majority_correct_cases": (
            majority_correct_cases
        ),
        "majority_vote_accuracy": (
            majority_correct_cases
            / total_cases
        ),
        "consistent_cases": consistent_cases,
        "consistency_rate": (
            consistent_cases
            / total_cases
        ),
        "cases_without_majority": (
            cases_without_majority
        ),
    }

def _percentile(
    values: list[float],
    percentile: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * fraction
    )


def _calculate_macro_f1(
    predictions: list[RouterEvalPrediction],
) -> float:
    f1_scores: list[float] = []

    for action in ALL_ACTIONS:
        true_positive = sum(
            1
            for row in predictions
            if (
                row.expected_action == action
                and row.predicted_action == action
            )
        )

        false_positive = sum(
            1
            for row in predictions
            if (
                row.expected_action != action
                and row.predicted_action == action
            )
        )

        false_negative = sum(
            1
            for row in predictions
            if (
                row.expected_action == action
                and row.predicted_action != action
            )
        )

        precision_denominator = (
            true_positive + false_positive
        )
        recall_denominator = (
            true_positive + false_negative
        )

        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2
                * precision
                * recall
                / (precision + recall)
            )

        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def _build_confusion_matrix(
    predictions: list[RouterEvalPrediction],
) -> dict[str, dict[str, int]]:
    matrix = {
        expected.value: {
            predicted.value: 0
            for predicted in ALL_ACTIONS
        }
        for expected in ALL_ACTIONS
    }

    for row in predictions:
        if row.predicted_action is None:
            continue

        matrix[row.expected_action.value][
            row.predicted_action.value
        ] += 1

    return matrix


def _build_per_action_accuracy(
    predictions: list[RouterEvalPrediction],
) -> dict[str, dict[str, float | int | None]]:
    result = {}

    for action in ALL_ACTIONS:
        action_rows = [
            row
            for row in predictions
            if row.expected_action == action
        ]

        correct = sum(
            row.correct
            for row in action_rows
        )

        result[action.value] = {
            "cases": len(action_rows),
            "correct": correct,
            "accuracy": (
                correct / len(action_rows)
                if action_rows
                else None
            ),
        }

    return result


def _build_category_accuracy(
    predictions: list[RouterEvalPrediction],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[
        str,
        list[RouterEvalPrediction],
    ] = defaultdict(list)

    for row in predictions:
        grouped[row.category].append(row)

    result = {}

    for category, rows in sorted(grouped.items()):
        correct = sum(row.correct for row in rows)

        result[category] = {
            "cases": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
        }

    return result


def _build_error_counts(
    predictions: list[RouterEvalPrediction],
) -> dict[str, int]:
    errors = Counter(
        row.error
        for row in predictions
        if row.error is not None
    )

    return dict(sorted(errors.items()))


def calculate_profile_metrics(
    predictions: list[RouterEvalPrediction],
) -> dict:
    if not predictions:
        raise ValueError(
            "Cannot calculate metrics for empty predictions"
        )

    profiles = {
        row.profile
        for row in predictions
    }

    if len(profiles) != 1:
        raise ValueError(
            "calculate_profile_metrics expects one profile"
        )

    successful = [
        row
        for row in predictions
        if row.success
    ]

    latencies = [
        row.latency_ms
        for row in predictions
    ]

    known_costs = [
        row.estimated_cost_usd
        for row in predictions
        if row.estimated_cost_usd is not None
    ]

    correct = sum(
        row.correct
        for row in predictions
    )

    total = len(predictions)

    total_cost = (
        sum(known_costs)
        if len(known_costs) == total
        else None
    )

    average_cost = (
        total_cost / total
        if total_cost is not None
        else None
    )

    total_llm_attempts = sum(
        row.llm_attempts
        for row in predictions
    )

    case_level = _calculate_case_level_metrics(
        predictions
    )

    return {
        "profile": predictions[0].profile,
        "model": predictions[0].model,

        "runs": total,
        "successful_runs": len(successful),
        "failed_runs": total - len(successful),
        "success_rate": (
            len(successful) / total
        ),

        "correct_runs": correct,
        "accuracy": correct / total,

        "macro_f1": _calculate_macro_f1(
            predictions
        ),

        "case_level": case_level,

        "per_action": _build_per_action_accuracy(
            predictions
        ),

        "per_category": _build_category_accuracy(
            predictions
        ),

        "confusion_matrix": _build_confusion_matrix(
            predictions
        ),

        "errors": _build_error_counts(
            predictions
        ),

        "latency_ms": {
            "mean": (
                sum(latencies)
                / len(latencies)
            ),
            "median": median(
                latencies
            ),
            "p90": _percentile(
                latencies,
                0.90,
            ),
            "p95": _percentile(
                latencies,
                0.95,
            ),
            "min": min(
                latencies
            ),
            "max": max(
                latencies
            ),
        },

        "cost_usd": {
            "total": total_cost,

            "mean_per_call": (
                average_cost
            ),

            "estimated_per_1000_calls": (
                average_cost * 1000
                if average_cost is not None
                else None
            ),
        },

        "llm_attempts": {
            "total": total_llm_attempts,

            "mean_per_run": (
                total_llm_attempts
                / total
            ),
        },
    }