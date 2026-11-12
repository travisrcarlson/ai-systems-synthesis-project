"""Screen 3: Log (Inside the twin, restricted)."""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from twin.season import ROLES
from loaders import (
    humanize_log_table,
    load_saved_log,
    load_saved_reviews_public,
    nav_to_dossier_review,
    nav_to_operations_week,
    nav_to_pipeline,
    peek_twin_bundle,
    restricted_header,
)


ROLE_COLOURS = {
    ROLES["ops"]: "#1f4e78",
    ROLES["qc"]: "#6d28d9",
    ROLES["plan"]: "#b45309",
    ROLES["na"]: "#0d6e52",
}

ROLE_SHORT_NAMES = {
    ROLES["plan"]: "Departmental office (`ROLES['plan']`)",
    ROLES["ops"]: "Operations officer (`ROLES['ops']`)",
    ROLES["qc"]: "QC reviewer (`ROLES['qc']`)",
    ROLES["na"]: "National authority (`ROLES['na']`)",
}


def _plot_roles_and_writeback_ledger(log_df: pd.DataFrame) -> plt.Figure:
    """Render a 2-panel diagnostic figure of the season log (`runs/season_log.csv`):
    - Left panel: decisions signed off by human role (`twin.season.ROLES`) and category (`kind`).
    - Right panel: weekly field reports received (`entry == 'report'`) vs. human decisions signed off (`entry == 'decision'`).
    """
    fig, (ax_roles, ax_wk) = plt.subplots(1, 2, figsize=(15.6, 4.8), facecolor="#faf9f5")
    for ax in (ax_roles, ax_wk):
        ax.set_facecolor("#faf9f5")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    decisions_df = log_df[log_df["entry"] == "decision"].copy()
    reports_df = log_df[log_df["entry"] == "report"].copy()

    # Left panel: horizontal bar chart of (role, kind) decision counts
    grp = (
        decisions_df.groupby(["decided_by", "kind"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    if not grp.empty:
        y_pos = np.arange(len(grp))
        labels = []
        cols = []
        for _, r in grp.iterrows():
            role_full = str(r["decided_by"])
            role_head = role_full.split("(")[0].strip().capitalize()
            labels.append(f"{role_head} · {r['kind']}")
            cols.append(ROLE_COLOURS.get(role_full, "#64748b"))
        vals = grp["count"].to_numpy(dtype=int)
        ax_roles.barh(y_pos, vals, color=cols, height=0.62, alpha=0.88, edgecolor="#181816", linewidth=0.5)
        for yi, v in enumerate(vals):
            pct = 100.0 * v / max(1, len(decisions_df))
            ax_roles.text(v + max(vals) * 0.018, yi, f"{v} ({pct:.0f}%)", va="center", fontsize=8.0, fontweight="bold")
        ax_roles.set_yticks(y_pos)
        ax_roles.set_yticklabels(labels, fontsize=8.2)
        ax_roles.set_xlim(0, max(vals, default=10) * 1.24)
        ax_roles.set_xlabel(f"Decisions signed off in the season (out of {len(decisions_df):,})", fontsize=8.4)
        ax_roles.set_title(
            f"1. Human-Signed Decisions by Role (`twin.season.ROLES`) & Category ({len(decisions_df):,} Decisions)",
            fontsize=9.6,
            fontweight="bold",
        )
        ax_roles.grid(axis="x", linestyle=":", alpha=0.45)

    # Right panel: weekly reports vs. human decisions across weeks 1..52
    weeks = np.arange(1, 53)
    rep_wk = reports_df.groupby("week").size().reindex(weeks, fill_value=0).to_numpy(dtype=int)
    dec_wk = decisions_df.groupby("week").size().reindex(weeks, fill_value=0).to_numpy(dtype=int)

    ax_wk.bar(weeks, rep_wk, color="#8cb8e8", width=0.78, label=f"Field & feed reports received ({len(reports_df):,})")
    ax_wk.bar(
        weeks,
        dec_wk,
        bottom=rep_wk,
        color="#1f4e78",
        width=0.78,
        label=f"Human decisions signed off ({len(decisions_df):,})",
    )
    ax_wk.set_xlim(0.4, 52.6)
    ax_wk.set_xlabel("Week of the season (1 to 52)", fontsize=8.4)
    ax_wk.set_ylabel("Log entries per week", fontsize=8.4)
    ax_wk.set_title(
        f"2. Weekly Rhythm of Reports Into the Twin vs. Approved Write-Back Decisions ({len(log_df):,} Total Entries)",
        fontsize=9.6,
        fontweight="bold",
    )
    ax_wk.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="#ffffff")
    ax_wk.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


def _render_role_and_writeback_ledger(log_df: pd.DataFrame, reviews_df: pd.DataFrame) -> None:
    """Render the Human-in-the-Loop Role & Approved Write-Back Ledger tab in Screen 3."""
    st.subheader("Human-in-the-loop roles and approved write-back (`twin.season.ROLES` & `runs/season_log.csv`)")
    st.markdown(
        "The twin is by design a **digital shadow with approved write-back** (Kritzinger et al., 2018; notebook "
        "sections 2.1 and 5): field and feed reports (`entry == 'report'`) flow into the twin as they arrive, while "
        "every change to a plan, task, dossier, area, zone or municipality (`entry == 'decision'`) is recorded with "
        "**who decided (`decided_by`)** under the season's stated sign-off policy (`twin/season.py`, `ROLES`), "
        "**what recommended it (`recommended_by` / `overlays`)**, and **any coded rule check (`rule_check`)**."
    )

    decisions_df = log_df[log_df["entry"] == "decision"].copy()
    reports_df = log_df[log_df["entry"] == "report"].copy()
    with_rule = decisions_df[decisions_df["rule_check"].astype(str).str.strip() != ""]
    with_overlay = decisions_df[decisions_df["overlays"].astype(str).str.strip() != ""]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Total Log Entries (52 Weeks)",
        f"{len(log_df):,}",
        f"{len(reports_df):,} reports · {len(decisions_df):,} decisions",
    )
    k2.metric(
        "Human-Signed Decisions",
        f"{len(decisions_df):,}",
        f"Across the {len(ROLES)} roles in `twin.season.ROLES`",
    )
    k3.metric(
        "Decisions With a Rule Check",
        f"{len(with_rule):,}",
        f"{int((with_rule['rule_check'] == 'pass').sum())} pass · {int((with_rule['rule_check'] != 'pass').sum())} vetoed",
    )
    k4.metric(
        "Decisions Citing an Overlay Tool",
        f"{len(with_overlay):,}",
        "Risk, planner, or dossier reviewer",
    )

    fig_roles = _plot_roles_and_writeback_ledger(log_df)
    st.pyplot(fig_roles)
    plt.close(fig_roles)

    # Summary table of the 4 human roles in twin.season.ROLES
    role_rows = []
    for r_key, r_policy in ROLES.items():
        sub_r = decisions_df[decisions_df["decided_by"] == r_policy]
        kinds_counts = ", ".join(f"{k} ({n})" for k, n in sub_r["kind"].value_counts().items()) or "—"
        rec_counts = ", ".join(
            f"{k} ({n})" for k, n in sub_r["recommended_by"].replace("", "—").value_counts().head(4).items()
        )
        role_rows.append(
            {
                "Role Key (`ROLES`)": f"`{r_key}`",
                "Role & Stated Sign-Off Policy (`twin/season.py`)": r_policy,
                "Decisions Signed Off": int(len(sub_r)),
                "Categories Decided (`kind`)": kinds_counts,
                "Top Recommenders / Checks (`recommended_by`)": rec_counts,
            }
        )
    st.markdown("#### 📋 The four human roles in `twin.season.ROLES` and their logged sign-offs")
    st.dataframe(pd.DataFrame(role_rows), width="stretch", hide_index=True)

    # Interactive filter for decisions by role, category, recommender, and rule check
    st.markdown("#### 🔎 Filter logged decisions by human role, category, tool consulted and rule check")
    f1, f2, f3 = st.columns(3)
    with f1:
        role_opts = list(ROLES.values())
        sel_roles = st.multiselect(
            "Human role (`decided_by`):",
            options=role_opts,
            default=role_opts,
            format_func=lambda r: ROLE_SHORT_NAMES.get(r, r),
            key="log_role_filter_roles",
        )
    with f2:
        kind_opts = sorted(decisions_df["kind"].unique().tolist())
        sel_kinds = st.multiselect(
            "Decision category (`kind`):",
            options=kind_opts,
            default=kind_opts,
            key="log_role_filter_kinds",
        )
    with f3:
        rc_opts = ["All decisions", "Only decisions with a rule check (`pass` or `fail`)", "Only decisions citing an overlay tool"]
        sel_rc_mode = st.selectbox(
            "Rule check / overlay filter:",
            options=rc_opts,
            index=0,
            key="log_role_filter_rc",
        )

    sub_dec = decisions_df.copy()
    if sel_roles:
        sub_dec = sub_dec[sub_dec["decided_by"].isin(sel_roles)]
    if sel_kinds:
        sub_dec = sub_dec[sub_dec["kind"].isin(sel_kinds)]
    if sel_rc_mode.startswith("Only decisions with a rule check"):
        sub_dec = sub_dec[sub_dec["rule_check"].astype(str).str.strip() != ""]
    elif sel_rc_mode.startswith("Only decisions citing"):
        sub_dec = sub_dec[sub_dec["overlays"].astype(str).str.strip() != ""]

    st.caption(f"Showing **{len(sub_dec):,}** of **{len(decisions_df):,}** human-signed decisions.")
    st.dataframe(humanize_log_table(sub_dec), width="stretch", hide_index=True)



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

    tab_story, tab_roles, tab_full = st.tabs(
        [
            "📖 Story of One Operational Zone",
            "👥 Decisions by Human Role & Tool (`ROLES`)",
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

            if st.session_state.get("log_story_zone_select") not in zones_list:
                st.session_state["log_story_zone_select"] = zones_list[0]

            chosen_zone = st.selectbox(
                "Select an operational zone:",
                options=zones_list,
                format_func=_zone_label,
                key="log_story_zone_select",
            )

            story_areas: set[str] = set()
            if not reviews_df.empty and "zone_id" in reviews_df.columns:
                story_areas.update(
                    reviews_df.loc[reviews_df["zone_id"] == chosen_zone, "area_id"].astype(str).tolist()
                )
            bundle = peek_twin_bundle(root)
            visit_areas: list[str] = []
            if bundle is not None:
                story_areas.update(bundle.get("zone_areas", {}).get(chosen_zone, ()))
                tasks_df = bundle.get("tasks_df", pd.DataFrame())
                if not tasks_df.empty and "zone" in tasks_df.columns:
                    z_tasks = tasks_df[tasks_df["zone"] == chosen_zone]
                    story_tasks = set(z_tasks["task_id"].astype(str).tolist())
                    if "kind" in z_tasks.columns and "area" in z_tasks.columns:
                        visit_areas = (
                            z_tasks.loc[z_tasks["kind"] == "VISIT", "area"]
                            .dropna()
                            .astype(str)
                            .loc[lambda s: s.str.strip() != ""]
                            .tolist()
                        )
                else:
                    story_tasks = set()
            else:
                story_tasks = set()

            task_orders_df = log_df[(log_df["entry"] == "decision") & (log_df["kind"] == "task order")]
            area_pat = re.compile(r"\b([A-Z]{3}-\d{4})\b")
            for _, trow in task_orders_df.iterrows():
                dec_text = str(trow["decision"])
                if chosen_zone in dec_text or any(a in dec_text for a in story_areas):
                    story_tasks.add(str(trow["subject"]))
                    if "VISIT" in dec_text:
                        for m_aid in area_pat.findall(dec_text):
                            if m_aid not in visit_areas:
                                visit_areas.append(m_aid)

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

            z_revs = (
                reviews_df[reviews_df["zone_id"].astype(str) == chosen_zone].sort_values(["week", "visit"])
                if not reviews_df.empty and "zone_id" in reviews_df.columns
                else pd.DataFrame()
            )
            first_z_area = (
                str(z_revs.iloc[0]["area_id"])
                if not z_revs.empty
                else (visit_areas[0] if visit_areas else (sorted(story_areas)[0] if story_areas else None))
            )
            last_z_wk = int(story_df["week"].max()) if not story_df.empty else 52

            lj1, lj2, lj3 = st.columns(3)
            with lj1:
                st.button(
                    f"🔄 Open {first_z_area} in Land Release Pipeline" if first_z_area else "🔄 Open Land Release Pipeline",
                    key=f"log_jump_pipe_{chosen_zone}",
                    width="stretch",
                    on_click=nav_to_pipeline,
                    args=(first_z_area, 52, 4),
                )
            with lj2:
                st.button(
                    f"📋 Open Week {last_z_wk} in Weekly Operations",
                    key=f"log_jump_ops_{chosen_zone}_{last_z_wk}",
                    width="stretch",
                    on_click=nav_to_operations_week,
                    args=(last_z_wk,),
                )
            with lj3:
                if not z_revs.empty:
                    lr = z_revs.iloc[-1]
                    st.button(
                        f"📄 Open {lr['area_id']} Sign-Off Sheet (Week {int(lr['week'])})",
                        key=f"log_jump_rev_{chosen_zone}_{lr['area_id']}_{int(lr['week'])}",
                        width="stretch",
                        on_click=nav_to_dossier_review,
                        args=(int(lr["week"]), str(lr["area_id"]), int(lr["visit"])),
                    )
                else:
                    st.button(
                        f"📄 No Dossier Review in {chosen_zone}",
                        key=f"log_jump_rev_none_{chosen_zone}",
                        width="stretch",
                        disabled=True,
                    )

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

    with tab_roles:
        _render_role_and_writeback_ledger(log_df, reviews_df)

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
