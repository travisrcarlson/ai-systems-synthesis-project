"""Screen 2: Plan & Completion Forecast (Inside the twin, restricted)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from twin import config
import grid_map
import operational_analytics
import screen_dashboard
from loaders import (
    get_twin_bundle,
    load_gate_table,
    load_saved_log,
    nav_to_log_zone,
    nav_to_operations_week,
    render_recommendation_box,
    restricted_header,
)


def _build_plan_table(
    plan_dict: dict,
    zones_meta: pd.DataFrame,
    victims_by_muni: pd.Series,
    gate_choice_by_muni: dict,
    limit: int,
    kind: str = "survey",
) -> pd.DataFrame:
    workloads = plan_dict["workloads"]
    months_map = plan_dict.get("months", {})
    rows = []
    for rank, zn in enumerate(plan_dict.get(kind, [])[:limit], start=1):
        w = workloads.loc[zn]
        muni = str(zones_meta.loc[zn, "municipality"])
        m_planned = months_map.get(zn, {})
        s_months = ", ".join(config.MONTHS[m - 1][:3] for m in m_planned.get("survey", []))
        c_months = ", ".join(config.MONTHS[m - 1][:3] for m in m_planned.get("clearance", []))
        row = {
            "Order": rank,
            "Zone": zn,
            "Municipality": muni,
            "Selected Risk Method": gate_choice_by_muni.get(muni, ""),
            "Survey Months": s_months or "—",
            "Clearance Months": c_months or "—",
            "Expected Mine Areas": round(float(w["expected_mine_areas"]), 2),
            "Survey Days": int(round(float(w["survey_days"]))),
            "Clearance Days": int(round(float(w["clearance_known"] + w["clearance_future"]))),
            "Mine Areas Cleared per 100 Days": round(100.0 * float(w["value_per_day"]), 2),
            "Recorded Victims in Municipality (Not Weighed in Plan)": int(victims_by_muni.get(muni, 0)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def render_plan_and_forecast(root: Path) -> None:
    restricted_header(
        "Plan & Completion Forecast",
        "The first plan as notebook section 6.3 computes it (week 1, before week 1's access feed; the approved P01 "
        "differs): the integer programme beside the rule of thumb, municipal victim counts shown as context "
        "(not weighed in the plan), and the completion forecast that assumes the plan holds; then the plans approved "
        "through the season.",
    )

    gate_df = load_gate_table(str(root))
    bundle = get_twin_bundle(root)

    risk_rec = bundle["risk"]
    milp_rec = bundle["plan"]
    rule_rec = bundle["rule"]
    fc_rec = bundle["fc"]
    zones_meta = bundle["zones_meta"]
    victims_by_muni = bundle["victims_by_municipality"]

    gate_choice_by_muni = {}
    if not gate_df.empty and "municipality" in gate_df.columns and "chosen_label" in gate_df.columns:
        gate_choice_by_muni = dict(zip(gate_df["municipality"], gate_df["chosen_label"]))

    season_plans = bundle.get("plans", [])
    evol_span = (
        f" ({season_plans[0].get('plan_id', 'P01')} to {season_plans[-1].get('plan_id', '')})" if season_plans else ""
    )
    tab_dash, tab_comp, tab_evol, tab_bayes, tab_fc, tab_map = st.tabs(
        [
            "📊 Visual Planning Dashboard",
            "⚖️ Optimised Plan vs. Rule of Thumb",
            f"🔄 Plan Evolution{evol_span} & Plan-to-Plan Comparison",
            "📈 Bayesian Risk Learning (W1 vs. W52) & Spatial Gate Explorer",
            "⏳ Completion Forecast (P10 / P50 / P90)",
            "🗺️ Regional Planning Map",
        ]
    )

    with tab_dash:
        screen_dashboard.render_planning_dashboard_section(root, bundle, gate_df)

    with tab_comp:
        st.subheader("The first plan (week 1, as notebook section 6.3 computes it): integer programme beside the rule of thumb")
        p01_exp = season_plans[0].get("expected", {}).get("mine areas cleared") if season_plans else None
        this_exp = milp_rec.answer.get("expected", {}).get("mine areas cleared") if milp_rec.ok else None
        p01_txt = (
            f" (expected mine areas cleared: {p01_exp:.2f} for P01, {this_exp:.2f} for the plan here)"
            if p01_exp is not None and this_exp is not None
            else ""
        )
        st.caption(
            "**Which plan this is:** both plans as notebook section 6.3 computes them, on the twin before week 1's "
            "access feed. The plan the season approved in week 1 (P01, on the Plan Evolution tab) came after that "
            f"feed, so it differs{p01_txt}.  \n"
            "**Recorded victims beside the plan (not weighed in the plan):** "
            "For each zone, the recorded victims of its municipality (counts only, all years) are shown beside "
            "the plan as the impact the departmental office weighs. Neither the integer programme nor the rule "
            "of thumb weights victim counts in its objective, so the plan's expected yield stays in mine areas."
        )

        col_m, col_r = st.columns(2)
        with col_m:
            with st.container(border=True):
                st.markdown("#### Optimised Plan (Integer Programme)")
                if milp_rec.ok:
                    exp = milp_rec.answer.get("expected", {})
                    st.markdown(
                        f"- **Expected mine areas cleared (from planning factors):** "
                        f"**{exp.get('mine areas cleared', 0.0):.1f}**\n"
                        f"- **Expected zones declared free (from planning factors):** "
                        f"**{exp.get('zones declared free', 0)}**\n"
                        f"- **Consultations this plan would request in week 1:** "
                        f"`{', '.join(milp_rec.answer.get('consultations', [])) or 'none'}`"
                    )
                solver = milp_rec.answer.get("solver", {}) if milp_rec.ok else {}
                solver_desc = (
                    f"Integer programme ({solver.get('variables', 0)} variables, "
                    f"{solver.get('constraints', 0)} constraints, gap {solver.get('gap', 0.0):.3%})"
                    if solver
                    else "Integer programme"
                )
                render_recommendation_box(
                    milp_rec,
                    title="Integer programme planner note (week 1)",
                    model_name=solver_desc,
                )

        with col_r:
            with st.container(border=True):
                st.markdown("#### Rule of Thumb Plan")
                if rule_rec.ok:
                    st.markdown(
                        f"- **Ordering rule:** Zones in progress first, then zones with the most expected mine areas\n"
                        f"- **Zones in its survey list (every zone with survey work):** "
                        f"**{len(rule_rec.answer.get('survey', []))}**\n"
                        f"- **Consultations this plan would request in week 1:** "
                        f"`{', '.join(rule_rec.answer.get('consultations', [])) or 'none'}`"
                    )
                render_recommendation_box(
                    rule_rec,
                    title="Rule of thumb planner note (week 1)",
                    model_name="Rule of thumb (highest expected mine areas first)",
                )

        if milp_rec.ok and rule_rec.ok:
            max_show = max(len(milp_rec.answer.get("survey", [])), 12)
            n_zones = st.slider(
                "Number of planned survey zones to compare side by side",
                min_value=5,
                max_value=min(60, max_show),
                value=12,
                step=1,
            )

            milp_tbl = _build_plan_table(
                milp_rec.answer, zones_meta, victims_by_muni, gate_choice_by_muni, limit=n_zones, kind="survey"
            )
            rule_tbl = _build_plan_table(
                rule_rec.answer, zones_meta, victims_by_muni, gate_choice_by_muni, limit=n_zones, kind="survey"
            )

            st.markdown(f"#### Side-by-side zone order (first {n_zones} survey zones)")
            side_rows = []
            for idx in range(n_zones):
                m_row = milp_tbl.iloc[idx] if idx < len(milp_tbl) else None
                r_row = rule_tbl.iloc[idx] if idx < len(rule_tbl) else None
                side_rows.append(
                    {
                        "Order": idx + 1,
                        "Integer Programme: Zone": m_row["Zone"] if m_row is not None else "—",
                        "Integer Programme: Municipality": m_row["Municipality"] if m_row is not None else "—",
                        "Integer Programme: Survey Months": m_row["Survey Months"] if m_row is not None else "—",
                        "Integer Programme: Expected Mine Areas": (
                            m_row["Expected Mine Areas"] if m_row is not None else None
                        ),
                        "Integer Programme: Recorded Victims in Municipality (Not Weighed)": (
                            int(m_row["Recorded Victims in Municipality (Not Weighed in Plan)"])
                            if m_row is not None
                            else None
                        ),
                        "Rule of Thumb: Zone": r_row["Zone"] if r_row is not None else "—",
                        "Rule of Thumb: Municipality": r_row["Municipality"] if r_row is not None else "—",
                        "Rule of Thumb: Expected Mine Areas": (
                            r_row["Expected Mine Areas"] if r_row is not None else None
                        ),
                        "Rule of Thumb: Recorded Victims in Municipality (Not Weighed)": (
                            int(r_row["Recorded Victims in Municipality (Not Weighed in Plan)"])
                            if r_row is not None
                            else None
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(side_rows), width="stretch", hide_index=True)

            with st.expander("View the zone workload tables (the first zones of each plan, as set above)"):
                t_m, t_r = st.tabs(
                    [
                        f"Integer programme ({len(milp_tbl)} zones shown)",
                        f"Rule of thumb ({len(rule_tbl)} zones shown)",
                    ]
                )
                with t_m:
                    st.dataframe(milp_tbl, width="stretch", hide_index=True)
                with t_r:
                    st.dataframe(rule_tbl, width="stretch", hide_index=True)

        with st.expander("Area risk prioritisation details by municipality"):
            gate_counts = (
                ", ".join(f"{k} in {v}" for k, v in gate_df["chosen_label"].value_counts().items())
                if not gate_df.empty and "chosen_label" in gate_df.columns
                else "baseline-first gate"
            )
            render_recommendation_box(
                risk_rec,
                title="Risk prioritisation note (week 1)",
                model_name=f"Baseline-first gate: {gate_counts}",
            )
            if not gate_df.empty:
                n_others = len(gate_df) - 1
                show_gate = pd.DataFrame(
                    {
                        "Municipality": gate_df["municipality"],
                        "Selected by Gate (Ranking + Calibration)": gate_df["chosen_label"],
                        "Selected by Ranking Alone": gate_df["ranking_alone_label"],
                        f"Logistic Regression: Mean PR-AUC on the Other {n_others} Municipalities": gate_df[
                            "lr: model mean PR-AUC (others)"
                        ].round(3),
                        f"Trees: Mean PR-AUC on the Other {n_others} Municipalities": gate_df[
                            "hgb: model mean PR-AUC (others)"
                        ].round(3),
                        f"Baseline: Mean PR-AUC on the Other {n_others} Municipalities": gate_df[
                            "lr: baseline mean PR-AUC (others)"
                        ].round(3),
                    }
                )
                st.dataframe(show_gate, width="stretch", hide_index=True)

    with tab_evol:
        _render_plan_evolution_studio(bundle, zones_meta, victims_by_muni, gate_choice_by_muni)

    with tab_bayes:
        log_df_b = load_saved_log(str(root))
        operational_analytics.render_bayesian_learning_and_gate_studio(bundle, log_df_b, gate_df)

    with tab_fc:
        st.subheader("Completion forecast (week 1) — forecast that assumes its plan holds")
        st.caption(
            f"**Forecast assumption:** This forecast simulates {fc_rec.answer.get('draws', 200) if fc_rec.ok else 200} "
            "draws of the rest of the season "
            "**assuming that the week-1 plan holds unchanged** for all 52 weeks "
            "(in operations, the programme replans each month as new survey results and feeds arrive)."
        )

        if fc_rec.ok:
            totals = fc_rec.answer["totals"]
            zf_tot = totals.get("zones declared free", {})
            mc_tot = totals.get("mine areas cleared", {})

            f1, f2, f3, f4 = st.columns(4)
            f1.metric(
                "Zones Declared Free (P50 Median)",
                int(zf_tot.get("P50", 0)),
                f"P10: {int(zf_tot.get('P10', 0))} · P90: {int(zf_tot.get('P90', 0))}",
            )
            f2.metric(
                "Mine Areas Cleared (P50 Median)",
                int(mc_tot.get("P50", 0)),
                f"P10: {int(mc_tot.get('P10', 0))} · P90: {int(mc_tot.get('P90', 0))}",
            )
            zf = fc_rec.answer["zones"].copy()
            n_p50 = int(zf["P50 week"].notna().sum())
            n_p90 = int(zf["P90 week"].notna().sum())
            f3.metric("Zones With a P50 Week in Season", f"{n_p50} of {len(zf)}")
            f4.metric("Zones With a P90 Week in Season", f"{n_p90} of {len(zf)}")

            totals_df = (
                pd.DataFrame(totals)
                .T[["P10", "P50", "P90"]]
                .astype(int)
                .rename_axis("Season Total (Assuming Week-1 Plan Holds)")
            )
            st.markdown("#### Season totals: P10, P50 and P90 (assuming the week-1 plan holds)")
            st.dataframe(totals_df, width="stretch")

            st.markdown(
                f"#### P50 and P90 completion weeks of the {len(zf)} zones the forecast covers "
                "(assuming the week-1 plan holds)"
            )
            show_zf = pd.DataFrame(
                {
                    "Zone": zf.index,
                    "Municipality": zf["municipality"].to_numpy(),
                    "P50 Week (Assumes Plan Holds)": [
                        "After this season" if pd.isna(w) else f"Week {int(w)}" for w in zf["P50 week"]
                    ],
                    "P90 Week (Assumes Plan Holds)": [
                        "After this season" if pd.isna(w) else f"Week {int(w)}" for w in zf["P90 week"]
                    ],
                    "Chance Declared Free This Season": [f"{float(c):.0%}" for c in zf["chance this season"]],
                    "Recorded Victims in Municipality (Not Weighed in Plan)": [
                        int(victims_by_muni.get(m, 0)) for m in zf["municipality"]
                    ],
                }
            )
            only_with_chance = st.checkbox(
                "Show only zones with a non-zero chance of completion this season", value=True
            )
            if only_with_chance:
                show_zf = show_zf[zf["chance this season"].to_numpy() > 0]
            st.dataframe(show_zf, width="stretch", hide_index=True)

            with st.expander("View forecast summary by municipality (assuming the week-1 plan holds)"):
                muni_fc = fc_rec.answer["municipalities"].copy().round(2)
                st.dataframe(muni_fc, width="stretch")

        render_recommendation_box(
            fc_rec,
            title="Completion forecaster note (week 1, assuming the week-1 plan holds)",
            model_name=f"Monte Carlo simulation ({fc_rec.answer.get('draws', 200) if fc_rec.ok else 200} draws)",
        )

    with tab_map:
        st.subheader("Regional Planning Map Across All 15 Municipalities")
        plan_map_labels = {
            "Initial Optimised Plan (Integer Programme)": "Week-1 plan: integer programme (MILP)",
            "Initial Rule of Thumb Plan": "Week-1 plan: rule of thumb",
            "Forecast P50 Completion Week": "Completion forecast: P50 week",
            "Chance of Holding Mines (Week 1)": "Area risk chance (week 1 gated model)",
        }
        chosen_lbl = st.radio(
            "Select planning layer to display:",
            options=list(plan_map_labels.keys()),
            horizontal=True,
        )
        log_df = load_saved_log(str(root))
        fig_plan_map = grid_map.plot_all_municipalities_grid(bundle, plan_map_labels[chosen_lbl], 1, log_df)
        st.pyplot(fig_plan_map)
        plt.close(fig_plan_map)


def _is_monthly_plan(reason: str) -> bool:
    """Whether a plan's reason in the season log names it as the season's first plan or a monthly plan
    (`twin/season.py`, `plan_step`); any other plan came between the monthly plans, after changes in the feeds."""
    r = str(reason).strip().lower()
    return r.startswith("the season's first plan") or r.startswith("the monthly plan")


def _plot_plan_evolution_and_forecasts(
    plans_list: list[dict],
    reason_by_id: dict,
    forecasts_list: list,
    plan_a: dict,
    plan_b: dict,
) -> plt.Figure:
    """Plot a 2-panel figure showing:
    - Left panel: the Monte Carlo completion forecasts made at the start of each month with that month's plan
      (P10/P50/P90 of the season's mine areas cleared and zones declared free; each assumes its plan holds), with
      dotted lines at the plans whose reason in the season log also lists a change in the feeds.
    - Right panel: Zone Survey Queue Rank Shift between `plan_a` and `plan_b` (top 15 zones).
    """
    fig, (ax_fc, ax_shift) = plt.subplots(1, 2, figsize=(15.6, 5.0), facecolor="#ffffff")
    for ax in (ax_fc, ax_shift):
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Left Panel: Monthly Forecast P10-P50-P90 Fan Trajectory across the season
    fc_weeks = []
    mc_p10, mc_p50, mc_p90 = [], [], []
    zf_p10, zf_p50, zf_p90 = [], [], []
    for fc_rec in forecasts_list:
        if not getattr(fc_rec, "ok", False):
            continue
        tots = fc_rec.answer.get("totals", {})
        mc = tots.get("mine areas cleared", {})
        zf = tots.get("zones declared free", {})
        fc_weeks.append(int(fc_rec.week))
        mc_p10.append(float(mc.get("P10", 0)))
        mc_p50.append(float(mc.get("P50", 0)))
        mc_p90.append(float(mc.get("P90", 0)))
        zf_p10.append(float(zf.get("P10", 0)))
        zf_p50.append(float(zf.get("P50", 0)))
        zf_p90.append(float(zf.get("P90", 0)))

    if fc_weeks:
        ax_fc.plot(fc_weeks, mc_p50, marker="o", linewidth=2.2, color="#1f4e78", label="Forecast Mine Areas Cleared (P50)")
        ax_fc.fill_between(fc_weeks, mc_p10, mc_p90, color="#1f4e78", alpha=0.15, label="Mine Areas Cleared (P10–P90 Band)")
        ax_fc.plot(fc_weeks, zf_p50, marker="s", linewidth=2.2, color="#0d6e52", label="Forecast Zones Declared Free (P50)")
        ax_fc.fill_between(fc_weeks, zf_p10, zf_p90, color="#0d6e52", alpha=0.15, label="Zones Declared Free (P10–P90 Band)")

    # Mark the plans whose reason in the season log also lists a change in the feeds (a zone of the plan closed or
    # reopened, a team stood down or back, an incident)
    n_feed = 0
    for p in plans_list:
        reason = str(reason_by_id.get(p.get("plan_id"), ""))
        if "changes in the feeds" in reason.lower():
            ax_fc.axvline(int(p.get("week", 1)), color="#dc2626", linestyle=":", linewidth=1.1, alpha=0.75)
            n_feed += 1
    n_between = sum(1 for p in plans_list if not _is_monthly_plan(reason_by_id.get(p.get("plan_id"), "")))

    ax_fc.set_title(
        "Forecast Made with Each Month's Plan (P10 / P50 / P90; Each Assumes Its Plan Holds)\n"
        f"(Red dotted lines: the {n_feed} plans whose reason also lists a change in the feeds;\n"
        f"plans between the monthly plans: {n_between})",
        fontsize=9.4,
        fontweight="bold",
    )
    ax_fc.set_xlabel("Week of Season (Week the Forecast Was Made)", fontsize=8.2)
    ax_fc.set_ylabel("Forecast Season Total", fontsize=8.2)
    ax_fc.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax_fc.grid(axis="y", linestyle=":", alpha=0.45)

    # Right Panel: Survey Queue Rank Shift between Plan A and Plan B
    id_a = plan_a.get("plan_id", "Plan A")
    id_b = plan_b.get("plan_id", "Plan B")
    surv_a = list(plan_a.get("survey", []))[:15]
    surv_b = list(plan_b.get("survey", []))
    rank_b_map = {z: idx + 1 for idx, z in enumerate(surv_b)}

    y_idx = np.arange(len(surv_a))
    deltas = []
    labels_z = []
    for idx_a, z in enumerate(surv_a):
        r_a = idx_a + 1
        r_b = rank_b_map.get(z)
        if r_b is None:
            deltas.append(0)
            labels_z.append(f"{z} (#{r_a} in {id_a} → not in {id_b}'s survey list)")
        else:
            # Positive delta = moved UP in priority (e.g. from #10 to #3 -> +7)
            deltas.append(r_a - r_b)
            labels_z.append(f"{z} (#{r_a} → #{r_b})")

    b_cols = ["#0d6e52" if d > 0 else ("#94a3b8" if d == 0 else "#d97706") for d in deltas]
    ax_shift.barh(y_idx, deltas, color=b_cols, height=0.60, alpha=0.88)
    ax_shift.axvline(0, color="#0f172a", linewidth=0.9)
    ax_shift.set_yticks(y_idx)
    ax_shift.set_yticklabels(labels_z, fontsize=7.7)
    ax_shift.invert_yaxis()
    ax_shift.set_xlabel(f"Priority Rank Shift ({id_a} Rank minus {id_b} Rank; + = Promoted)", fontsize=8.1)
    ax_shift.set_title(
        f"Survey Queue Priority Shift: `{id_a}` (Wk {plan_a.get('week', 1)}) → `{id_b}` (Wk {plan_b.get('week', 1)})\n"
        "(Green = Promoted in Queue, Amber = Deferred,\n"
        f"Grey = Unchanged or Not in {id_b}'s Survey List)",
        fontsize=9.3,
        fontweight="bold",
    )
    ax_shift.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


def _render_plan_evolution_studio(
    bundle: dict,
    zones_meta: pd.DataFrame,
    victims_by_muni: pd.Series,
    gate_choice_by_muni: dict,
) -> None:
    """Render the plans approved through the season (`twin.plans`), each with its reason from its approval entry in
    the season log, the forecasts made with them, and a plan-to-plan comparison of their zone lists."""
    plans_list = bundle.get("plans", [])
    replans_list = bundle.get("replans", [])
    forecasts_list = bundle.get("forecasts", [])
    log_df = bundle.get("log_df", pd.DataFrame())

    if not plans_list:
        st.info("No saved season plans found in the twin bundle.")
        return

    plan_ids = [p.get("plan_id", f"P{i+1:02d}") for i, p in enumerate(plans_list)]
    first_id, last_id = plan_ids[0], plan_ids[-1]

    # Each plan's reason: its approval entry in the season log (decision of kind "plan", subject = the plan's ID);
    # the season's own list of plans and reasons (`replans`, by week) only where the log has no entry
    reason_by_id = {}
    if not log_df.empty and {"entry", "kind", "subject", "reason"}.issubset(log_df.columns):
        approvals = log_df[(log_df["entry"] == "decision") & (log_df["kind"] == "plan")]
        reason_by_id = dict(zip(approvals["subject"].astype(str), approvals["reason"].fillna("").astype(str)))
    replan_reason_by_week = {int(w): str(r) for w, r in replans_list}
    for pid, p in zip(plan_ids, plans_list):
        if not reason_by_id.get(pid):
            reason_by_id[pid] = replan_reason_by_week.get(int(p.get("week", 0)), "—")

    n_total_plans = len(plans_list)
    n_between = sum(1 for pid in plan_ids if not _is_monthly_plan(reason_by_id.get(pid, "")))
    n_limited = sum(1 for p in plans_list if p.get("solver", {}).get("status") == 1)
    total_consults_req = sum(len(p.get("consultations", [])) for p in plans_list)
    lead_weeks = int(bundle["setup"].twin.season.consultation_weeks)
    plan_weeks = ", ".join(str(int(p.get("week", 0))) for p in plans_list)
    when_txt = (
        f"in this season all {n_total_plans} plans were the monthly ones (weeks {plan_weeks})"
        if n_between == 0
        else f"in this season {n_between} of the {n_total_plans} plans came between the monthly plans"
    )

    st.subheader(f"🔄 MILP Plan Evolution (`{first_id}` to `{last_id}`), Plan Reasons & Forecast Trajectory Studio")
    st.caption(
        "Through the season, the integer programme (`twin/planner.py`; SciPy's `milp`, solved by HiGHS) plans at the "
        "start of each month, and in any week when a feed changes the teams or the zones of the plan; "
        f"{when_txt}. Inspect every approved plan, its solver figures, its reason in the season log, and how its "
        "zone lists change from plan to plan."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Approved Plans", f"{n_total_plans} Plans", f"{first_id} to {last_id}")
    k2.metric("Plans Between the Monthly Plans", f"{n_between} Plans", "after changes in the feeds")
    k3.metric("Plans Stopped by the Solver's Time Limit", f"{n_limited} Plans", "solved to a 1% optimality gap")
    k4.metric("Consultations Requested by Plans", f"{total_consults_req} Zones", f"{lead_weeks}-week lead time")

    # Build Ledger of All Approved Plans
    ledger_rows = []
    for idx, p in enumerate(plans_list):
        wk = int(p.get("week", 1))
        pid = plan_ids[idx]
        exp = p.get("expected", {})
        slv = p.get("solver", {})
        gap = slv.get("gap")
        surv_q = p.get("survey", [])
        clr_q = p.get("clearance", [])
        cons_q = p.get("consultations", [])
        ledger_rows.append(
            {
                "Plan ID": pid,
                "Week": wk,
                "Month": bundle["calendar_labels"].get(wk, ""),
                "Reason (Season Log)": reason_by_id.get(pid, "—"),
                "Top 4 Survey Queue": ", ".join(surv_q[:4]) or "—",
                "Top 4 Clearance Queue": ", ".join(clr_q[:4]) or "—",
                "Consultations Requested": ", ".join(cons_q) or "None",
                "Expected Mine Areas Cleared (Remaining)": round(float(exp.get("mine areas cleared", 0.0)), 2),
                "Expected Zones Free (Remaining)": int(exp.get("zones declared free", 0)),
                "MILP Variables / Constraints": f"{slv.get('variables', '—')} / {slv.get('constraints', '—')}",
                "Optimality Gap": f"{float(gap):.2%}" if gap is not None else "—",
            }
        )
    st.markdown("#### 📋 All Approved Plans (`twin.plans`; Reasons from the Season Log)")
    st.dataframe(pd.DataFrame(ledger_rows), width="stretch", hide_index=True)

    # Interactive Plan-to-Plan Diff Selector
    st.markdown("#### 🔍 Interactive Plan-to-Plan Queue Diff & the Forecast Made with Each Month's Plan")
    plan_by_id = dict(zip(plan_ids, plans_list))

    c_pa, c_pb = st.columns(2)
    with c_pa:
        sel_id_a = st.selectbox(
            "Select Baseline Plan (Plan A):",
            options=plan_ids,
            index=0,
            format_func=lambda pid: f"{pid} (Week {plan_by_id[pid].get('week', 1)} — {reason_by_id[pid]})",
            key="plan_evol_sel_a",
        )
    with c_pb:
        default_b = min(len(plan_ids) - 1, 5)
        sel_id_b = st.selectbox(
            "Select Comparison Plan (Plan B):",
            options=plan_ids,
            index=default_b,
            format_func=lambda pid: f"{pid} (Week {plan_by_id[pid].get('week', 1)} — {reason_by_id[pid]})",
            key="plan_evol_sel_b",
        )

    pa = plan_by_id[sel_id_a]
    pb = plan_by_id[sel_id_b]
    wk_a = int(pa.get("week", 1))
    wk_b = int(pb.get("week", 1))
    top_z_a = pa.get("survey", [""])[0] if pa.get("survey") else ""

    pj1, pj2, pj3 = st.columns(3)
    with pj1:
        st.button(
            f"📋 Open {sel_id_a} Approval (Week {wk_a}) in Weekly Operations",
            key=f"plan_jump_ops_a_{sel_id_a}_{wk_a}",
            width="stretch",
            on_click=nav_to_operations_week,
            args=(wk_a,),
        )
    with pj2:
        st.button(
            f"📋 Open {sel_id_b} Approval (Week {wk_b}) in Weekly Operations",
            key=f"plan_jump_ops_b_{sel_id_b}_{wk_b}",
            width="stretch",
            on_click=nav_to_operations_week,
            args=(wk_b,),
        )
    with pj3:
        if top_z_a:
            st.button(
                f"📖 Open {top_z_a} (#1 in {sel_id_a}) Story in Operational Log",
                key=f"plan_jump_log_{sel_id_a}_{top_z_a}",
                width="stretch",
                on_click=nav_to_log_zone,
                args=(str(top_z_a),),
            )

    fig_evol = _plot_plan_evolution_and_forecasts(plans_list, reason_by_id, forecasts_list, pa, pb)
    st.pyplot(fig_evol)
    plt.close(fig_evol)

    # Side-by-side table of top 15 zones in Plan A vs Plan B
    diff_rows = []
    sa = list(pa.get("survey", []))
    sb = list(pb.get("survey", []))
    ca = list(pa.get("clearance", []))
    cb = list(pb.get("clearance", []))
    for r_i in range(15):
        za = sa[r_i] if r_i < len(sa) else "—"
        zb = sb[r_i] if r_i < len(sb) else "—"
        cza = ca[r_i] if r_i < len(ca) else "—"
        czb = cb[r_i] if r_i < len(cb) else "—"
        diff_rows.append(
            {
                "Queue Rank": r_i + 1,
                f"{sel_id_a} (Wk {pa.get('week', 1)}): Survey Zone": za,
                f"{sel_id_a}: Survey Municipality": str(zones_meta.loc[za, "municipality"]) if za in zones_meta.index else "—",
                f"{sel_id_b} (Wk {pb.get('week', 1)}): Survey Zone": zb,
                f"{sel_id_b}: Survey Municipality": str(zones_meta.loc[zb, "municipality"]) if zb in zones_meta.index else "—",
                f"{sel_id_a}: Clearance Zone": cza,
                f"{sel_id_b}: Clearance Zone": czb,
            }
        )
    st.markdown(f"**Top 15 Survey & Clearance Queue Comparison: `{sel_id_a}` vs. `{sel_id_b}`**")
    st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)
