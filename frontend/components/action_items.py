from typing import Any, Dict, List, Tuple
import re

import streamlit as st


SEVERITY_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def _collect_action_items(
    analysis: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    """Return list of (severity, source_title, action_text)."""

    items: List[Tuple[str, str, str]] = []

    for issue in analysis.get("detailed_feedback") or []:
        level = (
            issue.get("severity_level") or "low"
        ).lower()

        title = issue.get(
            "issue_title",
            ""
        )

        for action in issue.get("action_items") or []:
            items.append(
                (level, title, action)
            )

    if not items:
        for suggestion in analysis.get("suggestions") or []:
            items.append(
                ("medium", "General", suggestion)
            )

    items.sort(
        key=lambda row: SEVERITY_RANK.get(
            row[0],
            99
        )
    )

    return items


def display_action_items(
    analysis: Dict[str, Any]
) -> None:

    items = _collect_action_items(analysis)

    if not items:
        return

    st.markdown("### ⚡ Action Items")
    st.caption(
        "Concrete steps to improve your score, "
        "sorted by urgency."
    )

    icons = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }

    #    # ---------------------------------------------------------
    # GROUP ACTION ITEMS
    # ---------------------------------------------------------

    grouped: Dict[
        Tuple[str, str],
        Dict[str, Any]
    ] = {}

    for level, source, action in items:

        source_display = " ".join(
            str(source or "General").split()
        ).strip()

        source_key = re.sub(
            r"[^a-z0-9]+",
            " ",
            source_display.lower()
        ).strip()

        # Group all skill-evidence recommendations together.
        if "skills lack supporting evidence" in source_key:
            source_key = "skills_lack_supporting_evidence"
            source_display = "Most Skills Lack Supporting Evidence"

        key = (
            level,
            source_key
        )

        if key not in grouped:
            grouped[key] = {
                "source": source_display,
                "actions": [],
            }

        if action not in grouped[key]["actions"]:
            grouped[key]["actions"].append(action)

    # ---------------------------------------------------------
    # DISPLAY GROUPED CARDS
    # ---------------------------------------------------------

    for (level, _source_key), group in grouped.items():

        source = group["source"]
        actions = group["actions"]

        icon = icons.get(
            level,
            "🟢"
        )

        with st.container(border=True):

            st.markdown(
                f"**{icon} {level.title()} priority**"
            )

            if source and source != "General":
                st.caption(source)

            if len(actions) == 1:
                st.markdown(actions[0])

            else:
                st.markdown(
                    f"**{len(actions)} related "
                    f"recommendations**"
                )

                for action in actions:
                    st.markdown(
                        f"- {action}"
                    )