import html
import json

import streamlit as st

from ui.state import get_messages


def render_messages() -> None:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for idx, message in enumerate(get_messages()):
        role = message["role"]
        raw_content = message["content"]
        safe_content = html.escape(raw_content).replace("\n", "<br>")

        if role == "user":
            st.markdown(
                f"""
                <div class="message-row user-row">
                    <div class="chat-bubble user-bubble">{safe_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="message-row assistant-row">
                    <div class="chat-bubble assistant-bubble">{safe_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            debug_data = message.get("debug_data")
            if debug_data:
                with st.expander("Debug: parsed intent / search / selection", expanded=False):
                    parsed_intent = debug_data.get("parsed_intent")
                    search_request = debug_data.get("search_request")
                    state_after = debug_data.get("state_after")
                    answer_payload = debug_data.get("answer_payload")

                    if parsed_intent is not None:
                        st.markdown("**Parsed intent**")
                        st.code(
                            json.dumps(parsed_intent, ensure_ascii=False, indent=2, default=str),
                            language="json",
                        )

                    if search_request is not None:
                        st.markdown("**Search request**")
                        st.code(
                            json.dumps(search_request, ensure_ascii=False, indent=2, default=str),
                            language="json",
                        )

                    if state_after is not None:
                        st.markdown("**Current state after turn**")
                        st.code(
                            json.dumps(state_after, ensure_ascii=False, indent=2, default=str),
                            language="json",
                        )
                        
                    if answer_payload is not None:
                        st.markdown("**Answer payload**")
                        st.code(
                            json.dumps(answer_payload, ensure_ascii=False, indent=2, default=str),
                            language="json",
                        )

                    telemetry = debug_data.get("telemetry")
                    telemetry_log_info = debug_data.get("telemetry_log_info")

                    if telemetry_log_info is not None:
                        st.markdown("**Telemetry saved**")
                        st.code(
                            json.dumps(telemetry_log_info, ensure_ascii=False, indent=2, default=str),
                            language="json",
                        )

                    if telemetry is not None:
                        st.markdown("**Telemetry**")
                        st.code(
                            json.dumps(telemetry, ensure_ascii=False, indent=2, default=str),
                            language="json",
                        )

                        top_results = answer_payload.get("top_results") or []
                        if top_results:
                            st.markdown("**Selection summary by result**")
                            for idx, result in enumerate(top_results, start=1):
                                debug_selection = result.get("debug_selection") or {}
                                title = result.get("title") or f"Result {idx}"

                                with st.container():
                                    st.markdown(f"**{idx}. {title}**")
                                    st.markdown(
                                        f"""
- score: `{debug_selection.get('score')}`
- eligibility_status: `{debug_selection.get('eligibility_status')}`
- match_tier: `{debug_selection.get('match_tier')}`
- matched_must: `{debug_selection.get('matched_must_count')}/{debug_selection.get('matched_must_total')}`
- selection_reasons: `{debug_selection.get('selection_reasons')}`
- blocking_reasons: `{debug_selection.get('blocking_reasons')}`
"""
                                    )

                                    with st.expander(f"Raw debug_selection: {title}", expanded=False):
                                        st.code(
                                            json.dumps(debug_selection, ensure_ascii=False, indent=2, default=str),
                                            language="json",
                                        )

    st.markdown("</div>", unsafe_allow_html=True)