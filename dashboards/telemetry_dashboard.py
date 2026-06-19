from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
TELEMETRY_DIR = PROJECT_ROOT / "logs" / "telemetry"


def read_jsonl_files(telemetry_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(telemetry_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["_file"] = path.name
                    record["_line"] = line_number
                    records.append(record)
                except json.JSONDecodeError:
                    st.warning(f"Skipped invalid JSON: {path.name}:{line_number}")
    return records


def percentile_95(series: pd.Series) -> float:
    return series.quantile(0.95)


def percentile_99(series: pd.Series) -> float:
    return series.quantile(0.99)


def ms_to_s(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if value < 1000:
        return f"{value:.1f} ms"
    return f"{value / 1000:.2f} s"


def flatten_records(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    request_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []

    for r in records:
        telemetry = r.get("telemetry", {}) or {}
        latency = telemetry.get("latency_ms", {}) or {}
        cost = telemetry.get("cost", {}) or {}
        scenario = telemetry.get("scenario", {}) or {}
        result_summary = r.get("result_summary", {}) or {}
        llm = telemetry.get("llm", {}) or {}
        external = telemetry.get("external", {}) or {}

        trace_id = telemetry.get("trace_id")
        timestamp = r.get("timestamp")

        request_rows.append(
            {
                "timestamp": timestamp,
                "date": r.get("date"),
                "time": r.get("time"),
                "file": r.get("_file"),
                "line": r.get("_line"),
                "attempt_number": r.get("attempt_number"),
                "trace_id": trace_id,
                "source": r.get("source"),
                "top_n": r.get("top_n"),
                "max_items": r.get("max_items"),
                "user_message": r.get("user_message"),
                "need_clarification": result_summary.get("need_clarification"),
                "results_count": result_summary.get("results_count"),
                "total_latency_ms": latency.get("total_observed"),
                "estimated_total_usd": cost.get("estimated_total_usd"),
                "estimated_llm_usd": cost.get("estimated_llm_usd"),
                "estimated_external_usd": cost.get("estimated_external_usd"),
                "used_apify": scenario.get("used_apify"),
                "used_fallback": scenario.get("used_fallback"),
                "used_intent_extraction": scenario.get("used_intent_extraction"),
                "used_conversation_router": scenario.get("used_conversation_router"),
                "used_intent_update": scenario.get("used_intent_update"),
                "used_intent_repair": scenario.get("used_intent_repair"),
                "llm_calls_count": scenario.get("llm_calls_count", llm.get("calls_count")),
                "external_calls_count": scenario.get("external_calls_count", external.get("calls_count")),
            }
        )

        for step in latency.get("steps", []) or []:
            step_rows.append(
                {
                    "timestamp": timestamp,
                    "date": r.get("date"),
                    "trace_id": trace_id,
                    "source": r.get("source"),
                    "user_message": r.get("user_message"),
                    "step": step.get("name"),
                    "step_latency_ms": step.get("latency_ms"),
                    "metadata": step.get("metadata", {}),
                }
            )

        for call in llm.get("calls", []) or []:
            llm_rows.append(
                {
                    "timestamp": timestamp,
                    "date": r.get("date"),
                    "trace_id": trace_id,
                    "step": call.get("step"),
                    "model": call.get("model"),
                    "prompt_tokens": call.get("prompt_tokens"),
                    "completion_tokens": call.get("completion_tokens"),
                    "total_tokens": call.get("total_tokens"),
                    "estimated_cost_usd": call.get("estimated_cost_usd"),
                    "success": call.get("success"),
                    "error": call.get("error"),
                }
            )

        for call in external.get("calls", []) or []:
            external_rows.append(
                {
                    "timestamp": timestamp,
                    "date": r.get("date"),
                    "trace_id": trace_id,
                    "name": call.get("name") or call.get("step") or call.get("provider"),
                    "latency_ms": call.get("latency_ms"),
                    "success": call.get("success"),
                    "error": call.get("error"),
                    "raw": call,
                }
            )

    requests_df = pd.DataFrame(request_rows)
    steps_df = pd.DataFrame(step_rows)
    llm_df = pd.DataFrame(llm_rows)
    external_df = pd.DataFrame(external_rows)

    for df in (requests_df, steps_df, llm_df, external_df):
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_columns = {
        "requests": ["total_latency_ms", "estimated_total_usd", "estimated_llm_usd", "estimated_external_usd", "llm_calls_count", "external_calls_count"],
        "steps": ["step_latency_ms"],
        "llm": ["prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"],
        "external": ["latency_ms"],
    }
    for df, cols in [(requests_df, numeric_columns["requests"]), (steps_df, numeric_columns["steps"]), (llm_df, numeric_columns["llm"]), (external_df, numeric_columns["external"])]:
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return requests_df, steps_df, llm_df, external_df


def apply_filters(requests_df: pd.DataFrame, steps_df: pd.DataFrame, llm_df: pd.DataFrame, external_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Filters")

    sources = sorted(requests_df["source"].dropna().unique()) if "source" in requests_df else []
    selected_sources = st.sidebar.multiselect("Source", sources, default=sources)

    if not requests_df["timestamp"].dropna().empty:
        min_date = requests_df["timestamp"].dt.date.min()
        max_date = requests_df["timestamp"].dt.date.max()
        date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None

    flags = [
        "used_apify",
        "used_fallback",
        "used_conversation_router",
        "used_intent_update",
        "used_intent_repair",
        "need_clarification",
    ]
    selected_flag = st.sidebar.selectbox("Show only scenario", ["all"] + flags)

    filtered = requests_df.copy()
    if selected_sources:
        filtered = filtered[filtered["source"].isin(selected_sources)]

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["timestamp"].dt.date >= start_date)
            & (filtered["timestamp"].dt.date <= end_date)
        ]

    if selected_flag != "all" and selected_flag in filtered.columns:
        filtered = filtered[filtered[selected_flag] == True]

    trace_ids = set(filtered["trace_id"].dropna())
    steps_filtered = steps_df[steps_df["trace_id"].isin(trace_ids)] if not steps_df.empty else steps_df
    llm_filtered = llm_df[llm_df["trace_id"].isin(trace_ids)] if not llm_df.empty else llm_df
    external_filtered = external_df[external_df["trace_id"].isin(trace_ids)] if not external_df.empty else external_df

    return filtered, steps_filtered, llm_filtered, external_filtered


def show_overview(requests_df: pd.DataFrame, llm_df: pd.DataFrame) -> None:
    st.subheader("Overview")

    total_requests = len(requests_df)
    median_latency = requests_df["total_latency_ms"].median()
    p95_latency = requests_df["total_latency_ms"].quantile(0.95)
    p99_latency = requests_df["total_latency_ms"].quantile(0.99)
    total_cost = requests_df["estimated_total_usd"].fillna(0).sum()
    avg_cost = requests_df["estimated_total_usd"].fillna(0).mean() if total_requests else 0
    fallback_rate = requests_df["used_fallback"].fillna(False).mean() * 100 if "used_fallback" in requests_df else 0
    apify_rate = requests_df["used_apify"].fillna(False).mean() * 100 if "used_apify" in requests_df else 0
    llm_calls = int(requests_df["llm_calls_count"].fillna(0).sum()) if "llm_calls_count" in requests_df else len(llm_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Requests", f"{total_requests}")
    c2.metric("Median latency", ms_to_s(median_latency))
    c3.metric("P95 latency", ms_to_s(p95_latency))
    c4.metric("P99 latency", ms_to_s(p99_latency))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total cost", f"${total_cost:.5f}")
    c6.metric("Avg cost/request", f"${avg_cost:.5f}")
    c7.metric("Fallback rate", f"{fallback_rate:.1f}%")
    c8.metric("Apify rate", f"{apify_rate:.1f}%")

    c9, _, _, _ = st.columns(4)
    c9.metric("LLM calls", f"{llm_calls}")


def show_latency_distribution(requests_df: pd.DataFrame) -> None:
    st.subheader("Request latency distribution")
    latency = requests_df["total_latency_ms"].dropna()
    if latency.empty:
        st.info("No request latency data.")
        return

    latency_seconds = latency / 1000
    st.bar_chart(latency_seconds.value_counts(bins=20).sort_index())

    summary = pd.DataFrame(
        {
            "metric": ["p50 / median", "p95", "p99", "max"],
            "latency_ms": [latency.quantile(0.50), latency.quantile(0.95), latency.quantile(0.99), latency.max()],
        }
    )
    summary["latency"] = summary["latency_ms"].apply(ms_to_s)
    st.dataframe(summary[["metric", "latency", "latency_ms"]], use_container_width=True, hide_index=True)


def show_pipeline_latency(steps_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Pipeline latency by step")
    if steps_df.empty:
        st.info("No pipeline step data.")
        return pd.DataFrame()

    step_summary = (
        steps_df.groupby("step", dropna=False)["step_latency_ms"]
        .agg(count="count", mean="mean", median="median", p95=percentile_95, p99=percentile_99, max="max")
        .sort_values("p95", ascending=False)
        .reset_index()
    )

    view_mode = st.radio("Chart metric", ["median", "p95", "max"], horizontal=True, index=1)
    st.bar_chart(step_summary.set_index("step")[[view_mode]])

    pretty = step_summary.copy()
    for col in ["mean", "median", "p95", "p99", "max"]:
        pretty[col + "_readable"] = pretty[col].apply(ms_to_s)

    st.dataframe(
        pretty[["step", "count", "mean_readable", "median_readable", "p95_readable", "p99_readable", "max_readable", "mean", "median", "p95", "p99", "max"]],
        use_container_width=True,
        hide_index=True,
    )
    return step_summary


def show_outliers(requests_df: pd.DataFrame, steps_df: pd.DataFrame) -> None:
    st.subheader("Outliers: slowest requests")
    if requests_df.empty:
        st.info("No requests.")
        return

    slowest = requests_df.sort_values("total_latency_ms", ascending=False).head(20).copy()
    slowest["total_latency"] = slowest["total_latency_ms"].apply(ms_to_s)

    cols = [
        "timestamp",
        "source",
        "total_latency",
        "estimated_total_usd",
        "llm_calls_count",
        "external_calls_count",
        "used_apify",
        "used_fallback",
        "need_clarification",
        "user_message",
        "trace_id",
    ]
    st.dataframe(slowest[[c for c in cols if c in slowest.columns]], use_container_width=True, hide_index=True)

    trace_options = slowest["trace_id"].dropna().tolist()
    if not trace_options:
        return

    selected_trace = st.selectbox("Open trace", trace_options)
    request_row = requests_df[requests_df["trace_id"] == selected_trace].head(1)
    trace_steps = steps_df[steps_df["trace_id"] == selected_trace].copy()

    if not request_row.empty:
        row = request_row.iloc[0]
        st.markdown(f"**User message:** {row.get('user_message', '')}")
        st.markdown(f"**Total latency:** {ms_to_s(row.get('total_latency_ms'))} | **Cost:** ${row.get('estimated_total_usd', 0) or 0:.5f}")

    if not trace_steps.empty:
        trace_steps["latency"] = trace_steps["step_latency_ms"].apply(ms_to_s)
        st.bar_chart(trace_steps.set_index("step")[["step_latency_ms"]])
        st.dataframe(
            trace_steps[["step", "latency", "step_latency_ms", "metadata"]],
            use_container_width=True,
            hide_index=True,
        )


def show_cost(requests_df: pd.DataFrame, llm_df: pd.DataFrame) -> None:
    st.subheader("Cost")
    if requests_df.empty:
        st.info("No cost data.")
        return

    cost_df = requests_df.dropna(subset=["timestamp"]).copy()
    if not cost_df.empty:
        cost_df = cost_df.set_index("timestamp").sort_index()
        st.line_chart(cost_df[["estimated_total_usd", "estimated_llm_usd", "estimated_external_usd"]].fillna(0))

    if not llm_df.empty:
        model_summary = (
            llm_df.groupby("model", dropna=False)
            .agg(
                calls=("model", "count"),
                total_tokens=("total_tokens", "sum"),
                avg_tokens=("total_tokens", "mean"),
                total_cost=("estimated_cost_usd", "sum"),
                success_rate=("success", "mean"),
            )
            .reset_index()
        )
        model_summary["success_rate"] = model_summary["success_rate"] * 100
        st.dataframe(model_summary, use_container_width=True, hide_index=True)


def show_branch_usage(requests_df: pd.DataFrame) -> None:
    st.subheader("Branch usage")
    flags = [
        "used_apify",
        "used_fallback",
        "used_intent_extraction",
        "used_conversation_router",
        "used_intent_update",
        "used_intent_repair",
        "need_clarification",
    ]
    available = [f for f in flags if f in requests_df.columns]
    if not available:
        st.info("No scenario flags.")
        return

    usage = []
    total = len(requests_df)
    for flag in available:
        count = int(requests_df[flag].fillna(False).sum())
        usage.append({"branch": flag, "count": count, "rate_percent": count / total * 100 if total else 0})

    usage_df = pd.DataFrame(usage).sort_values("rate_percent", ascending=False)
    st.bar_chart(usage_df.set_index("branch")[["rate_percent"]])
    st.dataframe(usage_df, use_container_width=True, hide_index=True)


def show_errors(llm_df: pd.DataFrame, external_df: pd.DataFrame) -> None:
    st.subheader("Errors")
    errors = []

    if not llm_df.empty:
        llm_errors = llm_df[(llm_df["success"] == False) | (llm_df["error"].notna())].copy()
        if not llm_errors.empty:
            llm_errors["type"] = "llm"
            errors.append(llm_errors[["type", "timestamp", "trace_id", "step", "model", "error"]])

    if not external_df.empty:
        external_errors = external_df[(external_df["success"] == False) | (external_df["error"].notna())].copy()
        if not external_errors.empty:
            external_errors["type"] = "external"
            external_errors["step"] = external_errors["name"]
            external_errors["model"] = None
            errors.append(external_errors[["type", "timestamp", "trace_id", "step", "model", "error"]])

    if not errors:
        st.success("No errors found in selected telemetry.")
        return

    st.dataframe(pd.concat(errors, ignore_index=True), use_container_width=True, hide_index=True)


def show_raw_tables(requests_df: pd.DataFrame, steps_df: pd.DataFrame, llm_df: pd.DataFrame, external_df: pd.DataFrame) -> None:
    st.subheader("Raw data")
    with st.expander("Requests"):
        st.dataframe(requests_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    with st.expander("Pipeline steps"):
        st.dataframe(steps_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    with st.expander("LLM calls"):
        st.dataframe(llm_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    with st.expander("External calls"):
        st.dataframe(external_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Booking AI Agent Telemetry", layout="wide")
    st.title("Booking AI Agent Telemetry Dashboard")

    telemetry_dir = st.sidebar.text_input("Telemetry folder", str(TELEMETRY_DIR))
    telemetry_path = Path(telemetry_dir).expanduser()

    if not telemetry_path.exists():
        st.error(f"Folder not found: {telemetry_path}")
        return

    records = read_jsonl_files(telemetry_path)
    if not records:
        st.warning("No telemetry records found. Expected .jsonl files in the telemetry folder.")
        return

    requests_df, steps_df, llm_df, external_df = flatten_records(records)
    requests_df, steps_df, llm_df, external_df = apply_filters(requests_df, steps_df, llm_df, external_df)

    if requests_df.empty:
        st.warning("No records match selected filters.")
        return

    tabs = st.tabs(["Overview", "Latency", "Outliers", "Cost & LLM", "Branches", "Errors", "Raw"])

    with tabs[0]:
        show_overview(requests_df, llm_df)
        show_branch_usage(requests_df)

    with tabs[1]:
        show_latency_distribution(requests_df)
        show_pipeline_latency(steps_df)

    with tabs[2]:
        show_outliers(requests_df, steps_df)

    with tabs[3]:
        show_cost(requests_df, llm_df)

    with tabs[4]:
        show_branch_usage(requests_df)

    with tabs[5]:
        show_errors(llm_df, external_df)

    with tabs[6]:
        show_raw_tables(requests_df, steps_df, llm_df, external_df)


if __name__ == "__main__":
    main()
