from __future__ import annotations

import importlib.util
from pathlib import Path

# dashboards/ is not a package under app/ - import the module directly by
# path, matching how this dashboard is run standalone (streamlit run).
_MODULE_PATH = Path(__file__).resolve().parents[1] / "dashboards" / "telemetry_dashboard.py"
_spec = importlib.util.spec_from_file_location("telemetry_dashboard", _MODULE_PATH)
telemetry_dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(telemetry_dashboard)

flatten_soft_evidence_records = telemetry_dashboard.flatten_soft_evidence_records


def _record(soft_evidence_summary, shadow_detail=None, trace_id="t1", timestamp="2026-09-09T10:00:00"):
    return {
        "timestamp": timestamp,
        "telemetry": {
            "trace_id": trace_id,
            "soft_evidence_summary": soft_evidence_summary,
            "soft_evidence_shadow_detail": shadow_detail,
        },
    }


def _summary(**overrides):
    base = {
        "total_hotels": 5,
        "hotels_requiring_gemini": 2,
        "hotels_requiring_gemini_pct": 40.0,
        "deterministic_evidence_item_count": 3,
        "gemini_evidence_item_count": 4,
        "claim_relation_counts": {"SUPPORT": 2, "CONTRADICT": 1, "NOT_ENOUGH_EVIDENCE": 3, "MIXED": 1},
        "evidence_resolution_status_counts": {"resolved": 6, "verification_failed": 1, "skipped_call_limit": 0},
        "retrieval_status_counts": {"success": 5, "partial": 0, "failed": 0},
        "verifier": {
            "total_calls": 4, "total_latency_ms": 400.0, "total_input_tokens": 100,
            "total_output_tokens": 20, "total_tokens": 120, "total_cost_usd": 0.001,
            "total_parse_failures": 0,
        },
    }
    base.update(overrides)
    return base


def test_records_without_soft_evidence_summary_are_excluded():
    records = [_record(None), {"timestamp": "x", "telemetry": {}}]
    df = flatten_soft_evidence_records(records)
    assert df.empty


def test_populated_summary_is_flattened_correctly():
    records = [_record(_summary())]
    df = flatten_soft_evidence_records(records)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["total_hotels"] == 5
    assert row["hotels_requiring_gemini_pct"] == 40.0
    assert row["relation_support"] == 2
    assert row["relation_mixed"] == 1
    assert row["resolved"] == 6
    assert row["verification_failed"] == 1
    assert row["retrieval_success"] == 5
    assert row["verifier_calls"] == 4
    assert row["verifier_cost_usd"] == 0.001


def test_multiple_requests_produce_multiple_rows():
    records = [
        _record(_summary(total_hotels=3), trace_id="t1"),
        _record(_summary(total_hotels=5), trace_id="t2"),
    ]
    df = flatten_soft_evidence_records(records)
    assert len(df) == 2
    assert set(df["trace_id"]) == {"t1", "t2"}


def test_shadow_detail_preserved_for_expander():
    detail = [{"listing_id": "h1", "evidence_pool_size": 42}]
    records = [_record(_summary(), shadow_detail=detail)]
    df = flatten_soft_evidence_records(records)
    assert df.iloc[0]["shadow_detail"] == detail
