"""Screen 3: Log (Inside the twin, restricted)."""
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from loaders import (
    _rebuild_twin_and_season,
    humanize_log_table,
    load_saved_log,
    load_saved_reviews_public,
    restricted_header,
)


def _extract_zones_from_log(log_df: pd.DataFrame, reviews_df: pd.DataFrame) -> tuple[list[str], set[str], set[str]]:
    """Find zones that appear in the log and identify surveyed and declared-free zones."""
    zone_pattern = re.compile(r"\b([A-Z]{3}-Z\d{2})\b")
    found_zones: set[str] = set()
    for col in ("subject", "decision", "reason"):
        if col in log_df.columns:
            for val in log_df[col].astype(str):
                found_zones.update(zone_pattern.findall(val))
    if not reviews_df.empty and "zone_id" in reviews_df.columns:
        found_zones.update(reviews_df["zone_id"].dropna().astype(str).unique())

    free_zones = set(
        log_df.loc[
            (log_df["entry"] == "decision")
            & (log_df["kind"] == "zone")
            & (log_df["decision"] == "declare free"),
            "subject",
        ].astype(str)
    )
    surveyed_zones = set()
    for val in log_df.loc[log_df["kind"] == "team report", "reason"].astype(str):
        if "surveyed:" in val:
            surveyed_zones.update(zone_pattern.findall(val))

    # Order: declared-free zones first (by week declared free), then surveyed zones, then the rest
    free_order = (
        log_df.loc[
            (log_df["entry"] == "decision")
            & (log_df["kind"] == "zone")
            & (log_df["decision"] == "declare free"),
            "subject",
        ]
        .astype(str)
        .tolist()
    )
    ordered = []
    for z in free_order:
        if z not in ordered:
            ordered.append(z)
    for z in sorted(surveyed_zones):
        if z not in ordered:
            ordered.append(z)
    for z in sorted(found_zones):
        if z not in ordered:
            ordered.append(z)
    return ordered, surveyed_zones, free_zones


def render_log(root: Path) -> None:
    restricted_header(
        "Operational Log",
        "The single operational log of the season: every field report received and every decision signed off, "
        "in chronological order, plus the complete story of any selected operational zone.",
    )

    log_df = load_saved_log(str(root))
    reviews_df = load_saved_reviews_public(str(root))

    if log_df.empty:
        st.error("The operational log file was not found.")
        return

    tab_story, tab_full = st.tabs(
        [
            "📖 Story of One Operational Zone",
            "🔎 Search & Filter the Full Log",
        ]
    )

    with tab_story:
        st.subheader("The story of one zone")
        st.markdown(
            "Select an operational zone to trace every entry about it — from its initial consultation or "
            "zone-survey task order through its area visits, dossier sign-offs and clearances to its declaration as free."
        )

        zones_list, surveyed_set, free_set = _extract_zones_from_log(log_df, reviews_df)
        if not zones_list:
            st.info("No operational zones found in the log.")
        else:
            def _zone_label(z: str) -> str:
                if z in free_set:
                    return f"{z} — Declared free"
                if z in surveyed_set:
                    return f"{z} — Surveyed"
                return z

            chosen_zone = st.selectbox("Select an operational zone:", options=zones_list, index=0, format_func=_zone_label)

            story_areas: set[str] = set()
            if not reviews_df.empty and "zone_id" in reviews_df.columns:
                story_areas.update(
                    reviews_df.loc[reviews_df["zone_id"] == chosen_zone, "area_id"].astype(str).tolist()
                )
            if st.session_state.get("_twin_bundle_ready", False):
                bundle, _ = _rebuild_twin_and_season(str(root))
                story_areas.update(bundle.get("zone_areas", {}).get(chosen_zone, ()))
                tasks_df = bundle.get("tasks_df", pd.DataFrame())
                if not tasks_df.empty and "zone" in tasks_df.columns:
                    story_tasks = set(tasks_df.loc[tasks_df["zone"] == chosen_zone, "task_id"].astype(str).tolist())
                else:
                    story_tasks = set()
            else:
                story_tasks = set()

            task_orders_df = log_df[(log_df["entry"] == "decision") & (log_df["kind"] == "task order")]
            for _, trow in task_orders_df.iterrows():
                dec_text = str(trow["decision"])
                if chosen_zone in dec_text or any(a in dec_text for a in story_areas):
                    story_tasks.add(str(trow["subject"]))

            mask = (
                log_df["subject"].astype(str).isin(story_areas | story_tasks | {chosen_zone})
                | log_df["decision"].astype(str).str.contains(chosen_zone, regex=False)
                | log_df["reason"].astype(str).str.contains(chosen_zone, regex=False)
            )
            if story_areas:
                area_regex = "|".join(re.escape(a) for a in sorted(story_areas))
                mask = mask | log_df["reason"].astype(str).str.contains(area_regex, regex=True)

            story_df = log_df.loc[mask].copy()
            s1, s2, s3 = st.columns(3)
            s1.metric("Zone", chosen_zone, "Declared free" if chosen_zone in free_set else ("Surveyed" if chosen_zone in surveyed_set else "Not surveyed"))
            s2.metric("Task Orders Issued", len(story_tasks))
            s3.metric("Log Entries Recorded", len(story_df))

            st.dataframe(
                humanize_log_table(
                    story_df[
                        [
                            "n",
                            "week",
                            "entry",
                            "kind",
                            "subject",
                            "decision",
                            "decided_by",
                            "recommended_by",
                            "rule_check",
                            "reason",
                        ]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    with tab_full:
        st.subheader("Filter the operational log")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            week_range = st.slider("Week range", min_value=1, max_value=52, value=(1, 52))
        with c2:
            entry_options = sorted(log_df["entry"].unique().tolist())
            entry_sel = st.multiselect("Record type", options=entry_options, default=entry_options)
        with c3:
            kind_options = sorted(log_df["kind"].unique().tolist())
            kind_sel = st.multiselect("Category", options=kind_options, default=kind_options)
        with c4:
            subject_query = st.text_input("Search subject (e.g. ECV-Z16, ECV-0060, S1)", value="").strip()

        filtered = log_df[
            (log_df["week"] >= week_range[0]) & (log_df["week"] <= week_range[1])
        ].copy()
        if entry_sel:
            filtered = filtered[filtered["entry"].isin(entry_sel)]
        if kind_sel:
            filtered = filtered[filtered["kind"].isin(kind_sel)]
        if subject_query:
            filtered = filtered[
                filtered["subject"].astype(str).str.contains(subject_query, case=False, regex=False)
            ]

        st.caption(f"Showing {len(filtered)} of {len(log_df)} log entries.")
        st.dataframe(humanize_log_table(filtered), width="stretch", hide_index=True)

        with st.expander("Summary of log entries by record type and category", expanded=False):
            counts = (
                log_df.groupby(["entry", "kind"])
                .size()
                .rename("Entries")
                .reset_index()
                .rename(columns={"entry": "Record Type", "kind": "Category"})
            )
            st.dataframe(counts, width="stretch", hide_index=True)
