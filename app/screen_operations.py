"""Screen 1: Weekly Operations (Inside the twin, restricted)."""
from pathlib import Path

import pandas as pd
import streamlit as st

from twin import config
import grid_map
import operational_analytics
from loaders import (
    get_twin_bundle,
    humanize_log_table,
    humanize_zone_table,
    load_area_ordnance_findings,
    load_gate_table,
    load_run_info,
    load_saved_log,
    load_saved_sitreps,
    nav_to_log_zone,
    nav_to_pipeline,
    nav_to_sitrep_week,
    render_recommendation_box,
    restricted_header,
)


def _month_name_for_week(week: int, month_weeks: tuple = (5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4)) -> str:
    w = 1
    for m_idx, count in enumerate(month_weeks):
        if w <= week < w + count:
            return config.MONTHS[m_idx]
        w += count
    return config.MONTHS[-1]


def render_operations(root: Path) -> None:
    restricted_header(
        "Weekly Operations",
        "Inspect any week of the 52-week season: the plan in force, task orders, team reports, "
        "area and zone sign-offs, any insufficient-evidence notices, and the season-end operational picture.",
    )

    log_df = load_saved_log(str(root))
    run_info = load_run_info(str(root))
    gate_df = load_gate_table(str(root))
    saved_sitreps = load_saved_sitreps(str(root))

    if log_df.empty:
        st.error("The operational log file was not found.")
        return

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        week = st.slider(
            "Select week of the season (1 to 52)",
            min_value=1,
            max_value=52,
            value=3,
            step=1,
            key="ops_week_slider",
        )
    month_label = _month_name_for_week(week)
    with col_info:
        st.markdown(f"### Week {week} — {month_label}")
        agent_model = run_info.get("model", "saved model")
        gate_summary = (
            ", ".join(f"{k} in {v}" for k, v in gate_df["chosen_label"].value_counts().items())
            if not gate_df.empty and "chosen_label" in gate_df.columns
            else "fitted risk models"
        )
        st.caption(
            f"Reviewer & report model: **{agent_model}** · "
            f"Risk prioritisation by municipality: {gate_summary}"
        )

    bundle = get_twin_bundle(root)
    week_log = log_df[log_df["week"] == week].copy()

    # Quick summary metrics for the chosen week
    plans_up_to_week = log_df[
        (log_df["entry"] == "decision") & (log_df["kind"] == "plan") & (log_df["week"] <= week)
    ]
    active_plan_id = str(plans_up_to_week.iloc[-1]["subject"]) if not plans_up_to_week.empty else "—"
    task_orders = week_log[
        (week_log["entry"] == "decision") & (week_log["kind"].isin(["task order", "pause", "consultation"]))
    ]
    reports_logged = week_log[week_log["entry"] == "report"]
    team_reports = reports_logged[reports_logged["kind"] == "team report"]
    land_decisions = week_log[
        (week_log["entry"] == "decision") & (week_log["kind"].isin(["area", "zone", "municipality"]))
    ]

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Plan in Force", active_plan_id)
    w2.metric("Task Orders / Actions This Week", len(task_orders))
    w3.metric("Field Team Reports Received", len(team_reports))
    w4.metric("Area & Zone Decisions Signed Off", len(land_decisions))

    area_dec_w = land_decisions[land_decisions["kind"] == "area"]
    first_area_w = str(area_dec_w.iloc[0]["subject"]) if not area_dec_w.empty else None
    zone_dec_w = land_decisions[land_decisions["kind"] == "zone"]
    first_zone_w = str(zone_dec_w.iloc[0]["subject"]) if not zone_dec_w.empty else ""
    if not first_zone_w and first_area_w and first_area_w in bundle["areas_meta"].index:
        first_zone_w = str(bundle["areas_meta"].loc[first_area_w, "zone_id"])

    oj1, oj2, oj3 = st.columns(3)
    with oj1:
        st.button(
            f"📰 Open Week {week} Weekly Situation Report",
            key=f"ops_jump_sitrep_{week}",
            width="stretch",
            on_click=nav_to_sitrep_week,
            args=(week,),
        )
    with oj2:
        pipe_lbl = (
            f"🔄 Open {first_area_w} in Land Release Pipeline (Week {week})"
            if first_area_w
            else f"🔄 Open Land Release Pipeline at Week {week}"
        )
        st.button(
            pipe_lbl,
            key=f"ops_jump_pipe_{week}",
            width="stretch",
            on_click=nav_to_pipeline,
            args=(first_area_w, week, 4),
        )
    with oj3:
        if first_zone_w:
            st.button(
                f"📖 Open Zone {first_zone_w} Story in Operational Log",
                key=f"ops_jump_log_{week}_{first_zone_w}",
                width="stretch",
                on_click=nav_to_log_zone,
                args=(first_zone_w,),
            )
        else:
            st.button(
                "📖 Open Operational Log",
                key=f"ops_jump_log_{week}",
                width="stretch",
                on_click=nav_to_log_zone,
                args=("",),
            )

    tab_week, tab_teams, tab_end, tab_map = st.tabs(
        [
            f"📋 Week {week} Activity & Sign-Offs",
            f"🚜 Field Teams (S1–S4, C1–C4), Weather/Access Attrition & Clearance Timeline",
            f"🏁 Operations Picture at Season's End (All {len(bundle['zones_meta'])} Zones)",
            f"🗺️ Regional Map in Week {week}",
        ]
    )

    with tab_week:
        # 1. The plan in force in the chosen week
        st.subheader(f"Plan in force in week {week}")
        if plans_up_to_week.empty:
            st.write("No plan approved on or before this week.")
        else:
            latest_plan_row = plans_up_to_week.iloc[-1]
            approved_this_week = int(latest_plan_row["week"]) == week
            badge = "approved this week" if approved_this_week else f"approved in week {int(latest_plan_row['week'])}"
            with st.container(border=True):
                c_p1, c_p2 = st.columns([2, 3])
                with c_p1:
                    st.markdown(f"**Active Plan:** `{active_plan_id}` ({badge})")
                    st.markdown(f"**Approved by:** {latest_plan_row['decided_by']}")
                    st.markdown(f"**Recommended by:** {latest_plan_row['recommended_by']}")
                with c_p2:
                    st.markdown(f"**Why planned:** {latest_plan_row['reason']}")
                    st.caption(f"Tools consulted: {latest_plan_row['overlays']}")

            if active_plan_id != "—" and bundle.get("plans"):
                matching_plans = [p for p in bundle["plans"] if p.get("plan_id") == active_plan_id]
                if matching_plans:
                    p_info = matching_plans[0]
                    with st.expander(f"View scheduled zones and planner note for {active_plan_id}"):
                        expected = p_info.get("expected", {})
                        st.markdown(
                            f"- **Planning method:** {p_info.get('planner', 'integer programme')}\n"
                            f"- **Planned survey zones ({len(p_info.get('survey', []))}):** "
                            f"{', '.join(p_info.get('survey', [])) or '—'}\n"
                            f"- **Planned clearance zones ({len(p_info.get('clearance', []))}):** "
                            f"{', '.join(p_info.get('clearance', [])) or '—'}\n"
                            f"- **Prior consultations requested ({len(p_info.get('consultations', []))}):** "
                            f"{', '.join(p_info.get('consultations', [])) or 'none'}\n"
                            f"- **Expected yield (from planning factors):** "
                            f"{expected.get('mine areas cleared', 0.0):.1f} mine areas cleared, "
                            f"{expected.get('zones declared free', 0)} zones declared free"
                        )
                        plan_week = p_info.get("approved_week")
                        rec_matches = [r for r in bundle.get("plan_recs", []) if r.week == plan_week]
                        if rec_matches:
                            render_recommendation_box(
                                rec_matches[-1],
                                title=f"Season planner note for {active_plan_id} (week {plan_week})",
                                model_name=p_info.get("planner", "integer programme"),
                            )

        # 2. Task orders and team reports side-by-side
        col_to, col_tr = st.columns(2)
        with col_to:
            st.subheader(f"Task orders, pauses and consultation requests in week {week}")
            if task_orders.empty:
                st.write(f"No new task orders, pauses or consultation requests in week {week}.")
            else:
                st.dataframe(
                    humanize_log_table(
                        task_orders[["week", "kind", "subject", "decision", "decided_by", "reason"]]
                    ),
                    width="stretch",
                    hide_index=True,
                )

        with col_tr:
            st.subheader(f"Teams' reports in week {week}")
            if team_reports.empty:
                st.write(f"No field team reports received in week {week}.")
            else:
                st.dataframe(
                    humanize_log_table(team_reports[["week", "subject", "source", "reason"]]),
                    width="stretch",
                    hide_index=True,
                )
            other_reports = reports_logged[reports_logged["kind"] != "team report"]
            checker_decisions = week_log[
                (week_log["entry"] == "decision") & (week_log["kind"].isin(["report", "dossier"]))
            ]
            if not other_reports.empty or not checker_decisions.empty:
                with st.expander(f"Access feed, consultations held, and report and dossier checks in week {week}"):
                    if not other_reports.empty:
                        st.dataframe(
                            humanize_log_table(other_reports[["week", "kind", "subject", "source", "reason"]]),
                            width="stretch",
                            hide_index=True,
                        )
                    if not checker_decisions.empty:
                        st.dataframe(
                            humanize_log_table(
                                checker_decisions[["week", "kind", "subject", "decision", "decided_by", "reason"]]
                            ),
                            width="stretch",
                            hide_index=True,
                        )

        # 4. Decisions on areas and zones (with who decided and the rule check)
        st.subheader(f"Decisions on areas and zones in week {week}")
        if land_decisions.empty:
            st.write(f"No area, zone or municipality status decisions in week {week}.")
        else:
            st.dataframe(
                humanize_log_table(
                    land_decisions[
                        [
                            "week",
                            "kind",
                            "subject",
                            "decision",
                            "before",
                            "after",
                            "decided_by",
                            "recommended_by",
                            "rule_check",
                            "evidence",
                            "reason",
                        ]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        # 5. Any "insufficient evidence" answers that week, with their reasons
        st.subheader(f'"Insufficient evidence" notices in week {week}')
        insuff_week = [
            {"Week": w, "Decision Tool": ov, "Reason": rsn}
            for (w, ov, rsn) in bundle.get("insufficient", [])
            if int(w) == week
        ]
        sitrep_entry = saved_sitreps.get(week)
        if sitrep_entry and not sitrep_entry["recommendation"].ok:
            insuff_week.append(
                {
                    "Week": week,
                    "Decision Tool": sitrep_entry["recommendation"].overlay,
                    "Reason": sitrep_entry["recommendation"].reason,
                }
            )
        rev_pub = bundle.get("reviews_public_df", pd.DataFrame())
        if not rev_pub.empty:
            fb_week = rev_pub[(rev_pub["week"] == week) & (rev_pub["fallback"] == True)]
            for _, fb_row in fb_week.iterrows():
                insuff_week.append(
                    {
                        "Week": week,
                        "Decision Tool": f"Dossier reviewer ({fb_row['area_id']}, visit {int(fb_row['visit'])})",
                        "Reason": "Insufficient evidence — sent back for another survey visit.",
                    }
                )

        # The risk overlay's record in force in the chosen week (its chances are updated at each monthly plan)
        _chance, risk_rec = grid_map.chance_in_force(bundle, week)
        flagged_zones = pd.DataFrame()
        if risk_rec is not None and risk_rec.ok:
            rz = risk_rec.answer["zones"]
            flagged_zones = rz[rz["insufficient"]]

        if insuff_week:
            st.warning(f'{len(insuff_week)} "insufficient evidence" notice(s) in week {week}:')
            st.dataframe(pd.DataFrame(insuff_week), width="stretch", hide_index=True)
        else:
            st.success(
                f'No decision tool answered "insufficient evidence" in week {week}'
                + (
                    f"; the risk overlay's standing note below flags {len(flagged_zones)} zone(s)."
                    if len(flagged_zones)
                    else "."
                )
            )

        if risk_rec is not None and risk_rec.ok:
            if len(flagged_zones):
                with st.expander(
                    f'Standing calibration note: {len(flagged_zones)} zone(s) flagged "insufficient evidence" '
                    "(scores mostly outside calibrated range)"
                ):
                    show_fz = flagged_zones.copy()
                    show_fz["Municipality"] = show_fz.index.map(bundle["zones_meta"]["municipality"])
                    ema_col = f"Expected Mine Areas (in force in week {week})"
                    show_fz = show_fz.rename(
                        columns={
                            "areas": "Total Areas",
                            "expected mine areas": ema_col,
                            "sd": "Uncertainty (SD)",
                            "outside": "Share Outside Calibration Range",
                            "insufficient": "Flagged Insufficient",
                        }
                    )
                    st.dataframe(
                        show_fz[
                            [
                                "Municipality",
                                "Total Areas",
                                ema_col,
                                "Uncertainty (SD)",
                                "Share Outside Calibration Range",
                                "Flagged Insufficient",
                            ]
                        ].round(3),
                        width="stretch",
                    )

    with tab_teams:
        ord_findings = load_area_ordnance_findings(str(root))
        operational_analytics.render_team_deployment_and_attrition_studio(bundle, ord_findings, week)

    with tab_end:
        st.subheader("Operations picture at the season's end")
        zones_df = bundle.get("zones")
        if zones_df is None or zones_df.empty:
            st.info("The season-end operations picture is unavailable.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Operational Zones", int(len(zones_df)))
            c2.metric("Zones Surveyed", int((zones_df["surveyed_week"] > 0).sum()))
            c3.metric("Zones Declared Free", int((zones_df["free_week"] > 0).sum()))
            c4.metric(
                "Areas Released (Cancelled, Reduced or Cleared)",
                int(zones_df[["cancelled", "reduced", "cleared"]].sum().sum()),
            )

            f_col1, f_col2 = st.columns(2)
            with f_col1:
                state_filter = st.multiselect(
                    "Filter by zone state at season's end",
                    options=["free", "surveyed", "in survey", "not surveyed"],
                    default=["free", "surveyed", "in survey"],
                )
            with f_col2:
                muni_options = sorted(zones_df["municipality"].unique().tolist())
                muni_filter = st.multiselect(
                    "Filter by municipality (leave empty for all)", options=muni_options, default=[]
                )

            filtered_zones = zones_df.copy()
            if state_filter:
                filtered_zones = filtered_zones[filtered_zones["state"].isin(state_filter)]
            if muni_filter:
                filtered_zones = filtered_zones[filtered_zones["municipality"].isin(muni_filter)]

            filtered_zones = filtered_zones.sort_values(["state", "free_week", "surveyed_week"])
            st.caption(f"Showing {len(filtered_zones)} of {len(zones_df)} operational zones.")
            st.dataframe(humanize_zone_table(filtered_zones), width="stretch")

    with tab_map:
        grid_map.render_municipality_map_explorer(root, bundle, log_df, gate_df, saved_sitreps, week)
