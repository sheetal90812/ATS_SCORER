import requests
import streamlit as st

from frontend.services import api_client


def _show_backend_error(exc: Exception) -> None:
    if isinstance(exc, requests.ConnectionError):
        st.error("Could not reach the backend. Is it running on port 8000?")
    elif isinstance(exc, requests.HTTPError) and exc.response is not None:
        st.error(f"Backend returned {exc.response.status_code}: {exc.response.text}")
    else:
        st.error(f"Unexpected error: {exc}")


def render() -> None:
    st.title("📊 Analysis History")
    st.markdown("Review your previous resume analyses.")

    access_token = st.session_state.get("access_token")
    if not access_token:
        st.warning("⚠️ Sign in from the sidebar to view your history.")
        return

    try:
        history = api_client.get_history(access_token)
    except requests.RequestException as exc:
        _show_backend_error(exc)
        return

    if not history:
        st.info(
            "No analyses yet for this account. "
            "Run a scoring on the ATS Scorer page first."
        )
        if st.button("🎯 Go to ATS Scorer"):
            st.session_state.current_view = "scorer"
            st.rerun()
        return

    st.markdown(f"**{len(history)} analysis{'es' if len(history) != 1 else ''} saved**")

    c1, c2, c3 = st.columns([2.5, 1, 1])

    with c1:
        search = st.text_input(
            "🔎 Search",
            placeholder="Search by resume filename...",
            label_visibility="collapsed",
        ).strip().lower()

    with c2:
        filter_type = st.selectbox(
            "Filter",
            ["All", "With JD", "Resume only"],
            label_visibility="collapsed",
        )

    with c3:
        sort_by = st.selectbox(
            "Sort",
            ["Newest", "Oldest", "Highest score", "Lowest score"],
            label_visibility="collapsed",
        )

    filtered_history = []

    for entry in history:
        filename = entry.get("filename", "resume")
        analysis = entry.get("analysis_result", {}) or {}
        jd_comparison = (
            analysis.get("jd_comparison")
            or analysis.get("jd_match_analysis")
        )

        if search and search not in filename.lower():
            continue

        if filter_type == "With JD" and not jd_comparison:
            continue

        if filter_type == "Resume only" and jd_comparison:
            continue

        filtered_history.append(entry)

    if sort_by == "Newest":
        filtered_history.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )
    elif sort_by == "Oldest":
        filtered_history.sort(
            key=lambda x: x.get("created_at", ""),
        )
    elif sort_by == "Highest score":
        filtered_history.sort(
            key=lambda x: float(x.get("ats_score", 0)),
            reverse=True,
        )
    elif sort_by == "Lowest score":
        filtered_history.sort(
            key=lambda x: float(x.get("ats_score", 0)),
        )

    st.markdown("---")

    if not filtered_history:
        st.info("No analyses match your search or filter.")
        return



    for idx, entry in enumerate(filtered_history):
        filename = entry.get("filename", "resume")
        ats_score = float(entry.get("ats_score", 0))
        created_at = entry.get("created_at", "")
        analysis = entry.get("analysis_result", {}) or {}

        component_scores = analysis.get("component_scores", {}) or {}
        jd_comparison = (
            analysis.get("jd_comparison")
            or analysis.get("jd_match_analysis")
        )

        formatting = float(component_scores.get("formatting", 0))
        keywords = float(component_scores.get("keywords", 0))
        content = float(component_scores.get("content", 0))
        skill_validation = float(component_scores.get("skill_validation", 0))
        ats_compatibility = float(component_scores.get("ats_compatibility", 0))

        with st.container(border=True):
            header_col, score_col = st.columns([4, 1])

            with header_col:
                st.markdown(f"### 📄 {filename}")
                if created_at:
                    st.caption(f"Analyzed: {created_at}")

            with score_col:
                st.metric("ATS Score", f"{ats_score:.0f}/100")

            st.divider()

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                st.metric("Formatting", f"{formatting:.0f}/20")

            with c2:
                st.metric("Keywords", f"{keywords:.0f}/25")

            with c3:
                st.metric("Content", f"{content:.0f}/25")

            with c4:
                st.metric("Skills", f"{skill_validation:.0f}/15")

            with c5:
                st.metric("ATS", f"{ats_compatibility:.0f}/15")

            if jd_comparison:
                st.markdown(
                    f"🎯 **JD Match:** "
                    f"{float(jd_comparison.get('match_percentage', 0)):.0f}%"
                )

            entry_id = entry.get("id")

            if entry_id:
                b1, b2 = st.columns(3)

                with b1:
                    if st.button(
                        "👁️ View Details",
                        key=f"view_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state[f"history_details_{entry_id}"] = not st.session_state.get(
                            f"history_details_{entry_id}",
                            False,
                        )

                with b2:
                    if st.button(
                        "📄 PDF",
                        key=f"pdf_{idx}",
                        use_container_width=True,
                    ):
                        try:
                            pdf_bytes = api_client.get_history_pdf(
                                str(entry_id),
                                access_token,
                            )

                            st.download_button(
                                "⬇️ Download PDF",
                                data=pdf_bytes,
                                file_name=f"ats_report_{entry_id}.pdf",
                                mime="application/pdf",
                                key=f"download_pdf_{idx}",
                                use_container_width=True,
                            )
                        except requests.RequestException as exc:
                            _show_backend_error(exc)

                with b3:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state[f"confirm_delete_{entry_id}"] = True
                        st.rerun()

                if st.session_state.get(f"confirm_delete_{entry_id}", False):
                    st.warning(f"Delete **{filename}** from your history?")

                    d1, d2 = st.columns(2)

                    with d1:
                        if st.button(
                            "Cancel",
                            key=f"cancel_delete_{idx}",
                            use_container_width=True,
                        ):
                            st.session_state[f"confirm_delete_{entry_id}"] = False
                            st.rerun()

                    with d2:
                        if st.button(
                            "Yes, Delete",
                            key=f"confirm_delete_yes_{idx}",
                            use_container_width=True,
                        ):
                            try:
                                api_client.delete_history_entry(
                                    str(entry_id),
                                    access_token,
                                )
                                st.session_state[f"confirm_delete_{entry_id}"] = False
                                st.success("Analysis deleted.")
                                st.rerun()
                            except requests.RequestException as exc:
                                _show_backend_error(exc)

                if st.session_state.get(f"history_details_{entry_id}", False):
                    st.markdown("#### 📋 Analysis Details")

                    issues = analysis.get("issues_summary", [])
                    feedback = analysis.get("detailed_feedback", [])
                    skills = analysis.get("skills", [])

                    if issues:
                        st.markdown("**Issues / Summary**")
                        for item in issues:
                            st.write(f"• {item}")

                    if skills:
                        st.markdown("**Skills Detected**")
                        st.write(", ".join(skills[:20]))

                    if feedback:
                        st.markdown("**Detailed Feedback**")
                        for item in feedback:
                            if isinstance(item, dict):
                                title = item.get("title") or item.get("category") or "Feedback"
                                message = item.get("message") or item.get("description") or str(item)
                                st.write(f"**{title}:** {message}")
                            else:
                                st.write(f"• {item}")
                                
                