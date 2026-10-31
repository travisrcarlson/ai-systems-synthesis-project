"""Screen 4: Dossier Sign-Off Reviews (Inside the twin, restricted)."""
from pathlib import Path

import pandas as pd
import streamlit as st

import grounding_panels
import operational_analytics
from loaders import (
    HUMAN_OUTCOME,
    humanize_reviews_table,
    load_review_detail,
    load_run_info,
    load_saved_reviews_public,
    nav_to_log_zone,
    nav_to_operations_week,
    nav_to_pipeline,
    restricted_header,
)


def render_dossier_reviews(root: Path) -> None:
    reviews_df = load_saved_reviews_public(str(root))
    run_info = load_run_info(str(root))
    restricted_header(
        "Dossier Sign-Off Reviews",
        f"Inspect the {len(reviews_df)} area dossier reviews completed after non-technical survey visits, "
        "and read the sign-off sheets prepared for the QC reviewer who signs off.",
    )

    if reviews_df.empty:
        st.error("The dossier reviews file was not found.")
        return

    models_used = sorted(m for m in reviews_df["model"].unique().tolist() if m)
    model_str = ", ".join(models_used) if models_used else str(run_info.get("model", "saved model"))
    efforts_used = sorted(e for e in reviews_df["reasoning_effort"].unique().tolist() if e)
    effort_str = f" ({', '.join(efforts_used)} reasoning)" if efforts_used else ""

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dossier Reviews Completed", int(len(reviews_df)))
    c2.metric("Reviewer Model", f"{model_str}{effort_str}")
    c3.metric("Insufficient Evidence (Fallbacks)", int(reviews_df["fallback"].sum()))
    c4.metric("Vetoed by the Rule Check", int(reviews_df["vetoed at sign-off"].sum()))

    with st.expander("ℹ️ How the Dossier Reviewer Works (Information Used, Uncertainty & Safeguards)", expanded=False):
        st.markdown(
            f"- **Reviewer model:** `{model_str}`{effort_str}\n"
            "- **Status:** `ok` when the submitted recommendation passes the coded rule check (R1 to R6, "
            "`twin/rules.py`); `insufficient evidence` (another survey visit) when no recommendation passes within "
            "the step limit or a call fails.\n"
            "- **Information relied on:** Past landmine accidents in the desk study, land use, the area's survey dossier "
            "and evidence log, and the national land-release criteria.\n"
            "- **Uncertainty:** The reviewer states its confidence (`low`, `medium` or `high`) and lists any open questions for the person signing off.\n"
            "- **Key limitation:** The reviewer recommends only and never changes an area's status. The coded rule check "
            "vetoes a recommendation that breaks a rule, but it tests the register, not the world: if an item is classed "
            "wrongly, it checks the wrong class. A person signs off every decision; in this simulated season the "
            "season sign-off policy stands in for that person (\"QC reviewer: signs off what passes the rule check\")."
        )

    # Filter bar shared by both the sign-off reader and the summary table
    f1, f2, f3 = st.columns(3)
    with f1:
        outcomes = sorted(reviews_df["outcome"].unique().tolist())
        out_sel = st.multiselect(
            "Filter by recommendation",
            options=outcomes,
            default=outcomes,
            format_func=lambda k: HUMAN_OUTCOME.get(str(k), str(k)),
        )
    with f2:
        confs = sorted(reviews_df["confidence"].unique().tolist())
        conf_sel = st.multiselect("Filter by confidence", options=confs, default=confs)
    with f3:
        munis = sorted(reviews_df["municipality"].unique().tolist())
        muni_sel = st.multiselect("Filter by municipality (leave empty for all)", options=munis, default=[])

    filtered = reviews_df.copy()
    if out_sel:
        filtered = filtered[filtered["outcome"].isin(out_sel)]
    if conf_sel:
        filtered = filtered[filtered["confidence"].isin(conf_sel)]
    if muni_sel:
        filtered = filtered[filtered["municipality"].isin(muni_sel)]

    tab_sheet, tab_audit, tab_table = st.tabs(
        [
            "📄 Read Sign-Off Sheet",
            "🛡️ Rule Check (R1–R6) & the Agent's Safeguards",
            f"📋 All Dossier Reviews ({len(filtered)} of {len(reviews_df)})",
        ]
    )

    with tab_sheet:
        options_df = filtered if not filtered.empty else reviews_df
        option_indices = options_df.index.tolist()

        target_av = st.session_state.pop("reviews_target_area_visit", None)
        if target_av is not None:
            t_week, t_area, t_visit = target_av
            m_rows = reviews_df[
                (reviews_df["week"] == int(t_week))
                & (reviews_df["area_id"].astype(str) == str(t_area))
                & (reviews_df["visit"] == int(t_visit))
            ]
            if m_rows.empty:
                m_rows = reviews_df[reviews_df["area_id"].astype(str) == str(t_area)]
            if not m_rows.empty:
                match_idx = int(m_rows.index[-1])
                if match_idx not in option_indices:
                    option_indices = reviews_df.index.tolist()
                st.session_state["reviews_sheet_select_idx"] = match_idx

        if option_indices and st.session_state.get("reviews_sheet_select_idx") not in option_indices:
            st.session_state["reviews_sheet_select_idx"] = option_indices[0]

        def _format_review_option(idx: int) -> str:
            row = reviews_df.loc[idx]
            out_lbl = HUMAN_OUTCOME.get(str(row["outcome"]), str(row["outcome"]))
            dec_lbl = HUMAN_OUTCOME.get(str(row["decision"]), str(row["decision"]))
            return (
                f"Week {int(row['week']):02d} · Area {row['area_id']} ({row['municipality']}) · "
                f"Visit {int(row['visit'])} · Recommended: {out_lbl} ({row['confidence']} confidence) -> "
                f"Signed off: {dec_lbl}"
            )

        chosen_idx = st.selectbox(
            "Select a dossier review to read its sign-off sheet:",
            options=option_indices,
            format_func=_format_review_option,
            key="reviews_sheet_select_idx",
        )
        chosen_row = reviews_df.loc[chosen_idx]
        sel_w = int(chosen_row["week"])
        sel_a = str(chosen_row["area_id"])
        sel_z = str(chosen_row["zone_id"])
        sel_v = int(chosen_row["visit"])
        detail = load_review_detail(
            str(root),
            week=sel_w,
            area_id=sel_a,
            visit=sel_v,
        )

        st.markdown(
            '<div class="signoff-banner">Recommendation only: a person decides</div>',
            unsafe_allow_html=True,
        )

        rec = detail.get("record", {})
        if rec:
            rc_status = rec.get("rule_check", {}).get("result", "")
            st.caption(
                f"Area **{sel_a}** ({chosen_row['municipality']}, Zone `{sel_z}`) · "
                f"Week {sel_w}, Visit {sel_v} · "
                f"Model: **{rec.get('model', '')}** · "
                f"Rule check: **{rc_status}**"
            )
            if rec.get("fallback"):
                st.warning(
                    "**Insufficient evidence (fallback):** No recommendation passed the rule check, "
                    "so the area was sent back for another survey visit (`FURTHER_NTS`)."
                )

        j1, j2, j3 = st.columns(3)
        with j1:
            st.button(
                f"🔄 Open {sel_a} in Land Release Pipeline (Week {sel_w}, Step 4)",
                key=f"rev_jump_pipe_{sel_w}_{sel_a}_{sel_v}",
                width="stretch",
                on_click=nav_to_pipeline,
                args=(sel_a, sel_w, 4),
            )
        with j2:
            st.button(
                f"📖 Open Zone {sel_z} Story in Operational Log",
                key=f"rev_jump_log_{sel_w}_{sel_z}_{sel_v}",
                width="stretch",
                on_click=nav_to_log_zone,
                args=(sel_z,),
            )
        with j3:
            st.button(
                f"📋 Open Week {sel_w} in Weekly Operations",
                key=f"rev_jump_ops_{sel_w}_{sel_a}_{sel_v}",
                width="stretch",
                on_click=nav_to_operations_week,
                args=(sel_w,),
            )

        with st.container(border=True):
            if detail.get("signoff_md"):
                st.markdown(detail["signoff_md"])
            else:
                st.warning("Sign-off sheet not found.")

        if rec:
            with st.expander(
                "⚖️ Inspect the evidence register (IMAS 08.10 reliability classes) and the rule check (`twin.rules.rule_check`)",
                expanded=True,
            ):
                rc = rec.get("rule_check", {})
                stats = rec.get("stats", {}) or {}
                usage = rec.get("usage", {}) or {}
                hb = rec.get("handling_blocked", []) or []
                st.markdown(
                    f"- **Submitted recommendation:** `{rec.get('outcome')}` (confidence: `{rec.get('confidence')}`)\n"
                    f"- **Signed-off decision (season sign-off policy):** `{chosen_row['decision']}`\n"
                    f"- **Dossier sections read:** `{', '.join(rec.get('sections_read', []))}`\n"
                    f"- **Rules file sections read (`rules/land_release_rules.md`):** `{', '.join(rec.get('rules_read', []))}`\n"
                    f"- **Rule check vetoes / sentences removed by the handling-advice filter:** `{len(rec.get('vetoes', []))}` vetoes · `{len(hb)}` sentence(s) removed\n"
                    f"- **Agent telemetry:** `{stats.get('model_calls', 0)}` model calls · `{stats.get('tool_calls', 0)}` tool calls · `{float(stats.get('seconds', 0.0)):.1f}s` · `{usage.get('input_tokens', 0):,}` input tokens (`{usage.get('cached_tokens', 0):,}` cached) · `${float(rec.get('cost_usd') or 0.0):.4f}`"
                )
                if hb:
                    st.info(f"**Sentence(s) removed by the handling-advice filter:** {' | '.join(str(x) for x in hb)}")
                grounding_panels.render_imas_and_rule_check_inspector(rec)

    with tab_audit:
        operational_analytics.render_safety_rule_audit_studio(root)

    with tab_table:
        display_cols = [
            "week",
            "area_id",
            "zone_id",
            "municipality",
            "visit",
            "recorded",
            "outcome",
            "confidence",
            "decision",
            "vetoed at sign-off",
            "fallback",
            "model",
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.caption(f"Showing {len(filtered)} of {len(reviews_df)} dossier reviews.")
        st.dataframe(
            humanize_reviews_table(filtered[display_cols].fillna("")),
            width="stretch",
            hide_index=True,
        )
