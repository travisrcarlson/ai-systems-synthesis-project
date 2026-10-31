"""Command & Planning Overview (Inside the twin, restricted).

Brings together the layers of the programme: the season's figures at a chosen week, the interactive map of the
twin's cell grid, the risk model's chance of mines beside the land release status, and five tabs:
1) Contamination categories (the national register's MAP, MUSE and AEI, and the dossier evidence),
2) Real-world grounding,
3) Municipality explorer,
4) Planning & forecast (month-by-month schedule, integer programme vs rule of thumb, and completion forecast),
5) Season progress & team capacity (52-week curves, rain, municipal progress, and sign-off outcomes).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from twin import config
import clearance_pipeline
import grid_map
import grounding_panels
import interactive_map
import operational_analytics
import ordnance_library
from loaders import (
    REVIEW_OUTCOMES,
    HUMAN_OUTCOME,
    get_twin_bundle,
    load_gate_table,
    load_run_info,
    load_saved_log,
    load_saved_reviews_public,
    load_saved_sitreps,
    restricted_header,
)

SURFACE = "#faf9f5"
INK = "#181816"
INK_2 = "#52514e"
MUTED = "#898781"
BLUE = "#235789"
ORANGE = "#d97236"
GREEN = "#2d7d53"
RED = "#c44536"
GOLD = "#d9a73e"


def plot_season_overview_panels(
    bundle: dict,
    log_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    selected_week: int,
) -> plt.Figure:
    """Render a 4-panel executive summary of the 52-week season using only programme-visible records."""
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.2), facecolor=SURFACE)
    ax_prog, ax_days, ax_muni, ax_rev = axes.ravel()
    for ax in (ax_prog, ax_days, ax_muni, ax_rev):
        ax.set_facecolor(SURFACE)

    weeks = np.arange(1, 53)
    zone_state_df = bundle.get("zone_state_df", pd.DataFrame())
    zones_meta = bundle["zones_meta"]
    tasks_df = bundle.get("tasks_df", pd.DataFrame())

    # Panel 1: Cumulative programme progress across weeks 1..52
    if not zone_state_df.empty:
        free_curve = [
            int(((zone_state_df["free_week"] > 0) & (zone_state_df["free_week"] <= w)).sum())
            for w in weeks
        ]
        surv_curve = [
            int(((zone_state_df["surveyed_week"] > 0) & (zone_state_df["surveyed_week"] <= w)).sum())
            for w in weeks
        ]
    else:
        free_curve = [0] * 52
        surv_curve = [0] * 52

    released_curve = []
    for w in weeks:
        st_w = grid_map.area_status_at_week(bundle, int(w))
        released_curve.append(int(st_w.isin(["cancelled", "reduced", "cleared"]).sum()))

    ax_prog.plot(weeks, surv_curve, color=BLUE, linewidth=2.0, label="Zones surveyed (cumulative)")
    ax_prog.plot(weeks, free_curve, color=GREEN, linewidth=2.3, label="Zones declared free (cumulative)")
    ax_prog.plot(
        weeks,
        released_curve,
        color=ORANGE,
        linewidth=1.8,
        linestyle="--",
        label="Areas released (cancelled, reduced or cleared)",
    )
    for rep_w, _ in bundle.get("replans", []):
        ax_prog.axvline(rep_w, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.7)
    ax_prog.axvline(selected_week, color=RED, linestyle="-", linewidth=1.4, label=f"Selected week ({selected_week})")
    ax_prog.set_title("1. Cumulative Season Progress & Monthly Replans (Weeks 1–52)", fontsize=10, fontweight="bold")
    ax_prog.set_xlabel("Week of season", fontsize=8.5)
    ax_prog.set_ylabel("Count", fontsize=8.5)
    ax_prog.set_xlim(1, 52)
    ax_prog.legend(loc="upper left", fontsize=7.8, frameon=False)
    ax_prog.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 2: Weekly team-days worked, weather lost, and idle
    week_days = bundle.get("week_days", {})
    idle_dict = bundle.get("idle", {})
    worked = np.array([week_days.get(int(w), {}).get("worked", 0) for w in weeks], dtype=float)
    weather = np.array([week_days.get(int(w), {}).get("weather", 0) for w in weeks], dtype=float)
    idle = np.array([idle_dict.get(int(w), 0) for w in weeks], dtype=float)

    ax_days.bar(weeks, worked, color=BLUE, width=0.8, label="Team-days worked")
    ax_days.bar(weeks, weather, bottom=worked, color=GOLD, width=0.8, label="Days lost to rain / weather")
    ax_days.bar(weeks, idle, bottom=worked + weather, color="#c8c6be", width=0.8, label="Team-days with no task")
    ax_days.axvline(selected_week, color=RED, linestyle="-", linewidth=1.4)
    team_counts = bundle.get("team_counts", {})
    n_teams = int(sum(team_counts.values()))
    teams_txt = ", ".join(f"{n} {k}" for k, n in sorted(team_counts.items())) or "teams"
    ax_days.set_title(f"2. Weekly Field Team Capacity & Rainy Seasons ({n_teams} Teams: {teams_txt})",
                      fontsize=10, fontweight="bold")
    ax_days.set_xlabel("Week of season (March–April & October–November are rainy months)", fontsize=8.5)
    ax_days.set_ylabel("Team-days per week", fontsize=8.5)
    ax_days.set_xlim(0.5, 52.5)
    ax_days.legend(loc="upper right", fontsize=7.8, frameon=False)
    ax_days.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 3: 15-municipality zone state breakdown at selected_week
    z_state_w = grid_map.zone_state_at_week(bundle, selected_week)
    muni_names = list(config.NAME_OF.values())
    free_counts, surv_counts, in_surv_counts, uns_counts = [], [], [], []
    for code, name in config.NAME_OF.items():
        m_idx = zones_meta.index[zones_meta["municipality_code"] == code]
        st_m = z_state_w.reindex(m_idx)
        free_counts.append(int((st_m == "free").sum()))
        surv_counts.append(int((st_m == "surveyed").sum()))
        in_surv_counts.append(int((st_m == "in survey").sum()))
        uns_counts.append(int((st_m == "not surveyed").sum()))

    y_pos = np.arange(len(muni_names))
    f_arr = np.array(free_counts)
    s_arr = np.array(surv_counts)
    i_arr = np.array(in_surv_counts)
    u_arr = np.array(uns_counts)

    ax_muni.barh(y_pos, f_arr, color=GREEN, label="Declared free")
    ax_muni.barh(y_pos, s_arr, left=f_arr, color="#8cb8e8", label="Surveyed (open)")
    ax_muni.barh(y_pos, i_arr, left=f_arr + s_arr, color=GOLD, label="In survey")
    ax_muni.barh(y_pos, u_arr, left=f_arr + s_arr + i_arr, color="#e6e4dd", label="Not surveyed")
    ax_muni.set_yticks(y_pos)
    ax_muni.set_yticklabels(muni_names, fontsize=8.0)
    ax_muni.invert_yaxis()
    ax_muni.set_xlabel("Operational zones", fontsize=8.5)
    ax_muni.set_title(f"3. Zone Progress by Municipality at Week {selected_week}", fontsize=10, fontweight="bold")
    ax_muni.legend(loc="lower right", fontsize=7.8, frameon=False)
    ax_muni.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel 4: Dossier reviews up to selected_week
    rev_w = reviews_df[reviews_df["week"] <= selected_week] if not reviews_df.empty else pd.DataFrame()
    outcomes = list(REVIEW_OUTCOMES)
    outcome_labels = ["Cancel\n(release)", "Technical\nsurvey", "Clearance", "Further\nsurvey"]
    dec_counts = [int((rev_w["decision"] == k).sum()) if not rev_w.empty else 0 for k in outcomes]
    out_counts = [int((rev_w["outcome"] == k).sum()) if not rev_w.empty else 0 for k in outcomes]
    x_pos = np.arange(len(outcomes))
    w_bar = 0.36
    ax_rev.bar(x_pos - w_bar / 2, out_counts, width=w_bar, color="#8cb8e8", label="Reviewer recommendation")
    ax_rev.bar(x_pos + w_bar / 2, dec_counts, width=w_bar, color=BLUE, label="Final sign-off decision")
    for xi, (oc, dc) in enumerate(zip(out_counts, dec_counts)):
        ax_rev.text(xi - w_bar / 2, oc + 0.5, str(oc), ha="center", va="bottom", fontsize=8.0)
        ax_rev.text(xi + w_bar / 2, dc + 0.5, str(dc), ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ax_rev.set_xticks(x_pos)
    ax_rev.set_xticklabels(outcome_labels, fontsize=8.2)
    ax_rev.set_ylabel("Dossier reviews", fontsize=8.5)
    ax_rev.set_title(
        f"4. Area Dossier Sign-Offs Through Week {selected_week} ({len(rev_w)} of {len(reviews_df)})",
        fontsize=10,
        fontweight="bold",
    )
    ax_rev.legend(loc="upper right", fontsize=7.8, frameon=False)
    ax_rev.grid(axis="y", linestyle=":", alpha=0.5)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.34, wspace=0.22)
    return fig


def plot_planning_schedule_and_allocation(
    plan_dict: dict,
    rule_dict: dict,
    fc_dict: dict,
    zones_meta: pd.DataFrame,
    victims_by_muni: pd.Series,
) -> plt.Figure:
    """Render a 3-panel visual Planning Dashboard:
    1) Month-by-month (Jan–Dec) zone schedule matrix for the selected plan, highlighting rainy months,
    2) Municipality zone allocation (total vs. integer programme vs. rule of thumb's first zones vs. forecast zones free),
    3) Zone-level forecast P50 and P90 completion weeks (the plan's zones that have a P50 week, in the plan's order,
       up to 24; a zone with no P90 week is shown as "P90: after the season").
    """
    fig = plt.figure(figsize=(15.4, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.24)
    ax_sched = fig.add_subplot(gs[0, :])
    ax_muni = fig.add_subplot(gs[1, 0])
    ax_fc = fig.add_subplot(gs[1, 1])
    for ax in (ax_sched, ax_muni, ax_fc):
        ax.set_facecolor(SURFACE)

    # Panel 1: Month-by-Month Schedule Matrix (Months 1..12 across planned zones)
    months_map = plan_dict.get("months", {})
    planned_zones = []
    for z in plan_dict.get("survey", []) + plan_dict.get("clearance", []):
        if z not in planned_zones and z in months_map:
            planned_zones.append(z)
    planned_zones = planned_zones[:35]

    if planned_zones:
        mat = np.zeros((12, len(planned_zones)), dtype=float)
        for z_idx, z in enumerate(planned_zones):
            m_info = months_map.get(z, {})
            s_set = set(m_info.get("survey", []))
            c_set = set(m_info.get("clearance", []))
            for m in range(1, 13):
                if m in s_set and m in c_set:
                    mat[m - 1, z_idx] = 3.0
                elif m in c_set:
                    mat[m - 1, z_idx] = 2.0
                elif m in s_set:
                    mat[m - 1, z_idx] = 1.0
        sched_cmap = ListedColormap(["#f2f1ed", "#8cb8e8", "#d97236", "#235789"])
        ax_sched.imshow(mat, aspect="auto", cmap=sched_cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax_sched.set_xticks(np.arange(len(planned_zones)))
        ax_sched.set_xticklabels(planned_zones, rotation=60, ha="right", fontsize=7.5)
        ax_sched.set_yticks(np.arange(12))
        month_labels = [
            f"{m_name[:3]} (rainy)" if (idx + 1) in (3, 4, 10, 11) else m_name[:3]
            for idx, m_name in enumerate(config.MONTHS)
        ]
        ax_sched.set_yticklabels(month_labels, fontsize=8.0)
        ax_sched.legend(
            handles=[
                Patch(color="#f2f1ed", label="No task scheduled"),
                Patch(color="#8cb8e8", label="Non-technical survey"),
                Patch(color="#d97236", label="Clearance only"),
                Patch(color="#235789", label="Survey + clearance in same month"),
            ],
            loc="upper right",
            ncol=4,
            fontsize=8.0,
            frameon=True,
            facecolor=SURFACE,
        )
    else:
        ax_sched.text(0.5, 0.5, "No month-by-month schedule in this plan", ha="center", va="center")

    plan_id = plan_dict.get("plan_id", f"Week {plan_dict.get('week', 1)} Optimised Plan")
    ax_sched.set_title(
        f"1. Monthly Schedule Across Operational Zones ({plan_id}) — Rainy seasons in Mar–Apr & Oct–Nov",
        fontsize=10,
        fontweight="bold",
    )

    # Panel 2: Municipality allocation (Total zones vs Optimised Plan vs Rule of Thumb vs Forecast P50 free)
    muni_names = list(config.NAME_OF.values())
    milp_surv = set(plan_dict.get("survey", [])) | set(plan_dict.get("clearance", []))
    rule_surv = (
        (set(rule_dict.get("survey", [])) | set(rule_dict.get("clearance", [])))
        if rule_dict
        else set()
    )
    fc_muni = fc_dict.get("municipalities", pd.DataFrame()) if fc_dict else pd.DataFrame()

    tot_z, milp_z, rule_z, fc_p50, fc_p10, fc_p90 = [], [], [], [], [], []
    for code, name in config.NAME_OF.items():
        m_z = zones_meta.index[zones_meta["municipality_code"] == code]
        tot_z.append(len(m_z))
        milp_z.append(sum(1 for z in m_z if z in milp_surv))
        rule_z.append(sum(1 for z in m_z if z in rule_surv))
        if not fc_muni.empty and name in fc_muni.index:
            # the forecaster counts the further zones its draws declare free; zones already free are a separate column
            already = float(fc_muni.loc[name, "already free"]) if "already free" in fc_muni.columns else 0.0
            already = 0.0 if pd.isna(already) else already
            fc_p50.append(already + float(fc_muni.loc[name, "zones free P50"]))
            fc_p10.append(already + float(fc_muni.loc[name, "zones free P10"]))
            fc_p90.append(already + float(fc_muni.loc[name, "zones free P90"]))
        else:
            fc_p50.append(0.0)
            fc_p10.append(0.0)
            fc_p90.append(0.0)

    y_m = np.arange(len(muni_names))
    h = 0.26
    ax_muni.barh(y_m - h, tot_z, height=h, color="#d8d6ce", label="Total zones in municipality")
    ax_muni.barh(y_m, milp_z, height=h, color=BLUE, label="Planned zones (integer programme)")
    if rule_dict:
        ax_muni.barh(y_m + h, rule_z, height=h, color=ORANGE,
                     label="Rule of thumb's first zones (cut to the integer programme's count)")
    ax_muni.errorbar(
        fc_p50,
        y_m,
        xerr=[
            np.maximum(0, np.array(fc_p50) - np.array(fc_p10)),
            np.maximum(0, np.array(fc_p90) - np.array(fc_p50)),
        ],
        fmt="o",
        color=GREEN,
        markersize=4.5,
        linewidth=1.4,
        label="Forecast zones free by the season's end, P50 (P10–P90), incl. zones already free",
    )
    ax_muni.set_yticks(y_m)
    ax_muni.set_yticklabels(
        [f"{m} ({int(victims_by_muni.get(m, 0))} victims)" for m in muni_names],
        fontsize=7.8,
    )
    ax_muni.invert_yaxis()
    ax_muni.set_xlabel("Zones", fontsize=8.5)
    ax_muni.set_title(
        "2. Planned Zones & Forecast Completion by Municipality (victims shown as context)",
        fontsize=9.8,
        fontweight="bold",
    )
    ax_muni.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax_muni.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel 3: Zone-level Forecast P50 to P90 completion weeks (assumes plan holds)
    fc_zones = fc_dict.get("zones", pd.DataFrame()) if fc_dict else pd.DataFrame()
    if not fc_zones.empty and "P50 week" in fc_zones.columns:
        valid_fc = fc_zones[fc_zones["P50 week"].notna()].head(24)
        if not valid_fc.empty:
            y_z = np.arange(len(valid_fc))
            p50_vals = valid_fc["P50 week"].to_numpy(dtype=float)
            p90_raw = valid_fc["P90 week"].to_numpy(dtype=float)
            shown_after = False
            shown_p90 = False
            for yi, (z_id, p50_v, p90_v) in enumerate(zip(valid_fc.index, p50_vals, p90_raw)):
                if np.isfinite(p90_v):
                    ax_fc.plot([p50_v, p90_v], [yi, yi], color=BLUE, linewidth=2.4, alpha=0.75)
                    ax_fc.scatter([p90_v], [yi], color=ORANGE, s=28, zorder=3,
                                  label="" if shown_p90 else "P90 week")
                    shown_p90 = True
                else:
                    # the forecaster gives no P90 week: done in fewer than 90% of draws ("after the season")
                    ax_fc.plot([p50_v, 52.6], [yi, yi], color=BLUE, linewidth=2.4, alpha=0.35, linestyle="--",
                               label="" if shown_after else "P90: after the season (no P90 week)")
                    shown_after = True
                ax_fc.scatter([p50_v], [yi], color=GREEN, s=28, zorder=3, label="P50 week" if yi == 0 else "")
            ax_fc.set_yticks(y_z)
            ax_fc.set_yticklabels(list(valid_fc.index), fontsize=7.5)
            ax_fc.invert_yaxis()
            ax_fc.set_xlim(1, 53)
            ax_fc.set_xlabel("Week of season (1–52)", fontsize=8.5)
            ax_fc.legend(loc="lower right", fontsize=8.0, frameon=False)
            ax_fc.grid(axis="x", linestyle=":", alpha=0.5)
        else:
            ax_fc.text(0.5, 0.5, "No zone in this forecast has a P50 week (done in at least half the draws)",
                       ha="center")
    else:
        ax_fc.text(0.5, 0.5, "Forecast table unavailable", ha="center")
    ax_fc.set_title(
        "3. Zone Completion Forecast Intervals: P50 to P90 Week (assumes plan holds)",
        fontsize=9.8,
        fontweight="bold",
    )

    return fig


def render_planning_dashboard_section(root: Path, bundle: dict, gate_df: pd.DataFrame) -> None:
    """Reusable interactive Planning & Forecast Dashboard section."""
    st.subheader("Season Planning & Completion Forecast Dashboard")
    st.caption(
        "Compare the week-1 plan from the integer programme against the rule of thumb (the zones with the most "
        "expected mine areas first), or step through any of the monthly plans approved during the 52-week season "
        "with its completion forecast. The forecast assumes its plan holds. Recorded victims by municipality are "
        "shown alongside as programme context and are not weighed in the plan."
    )

    plans_list = bundle.get("plans", [])
    forecasts_list = bundle.get("forecasts", [])
    milp_rec = bundle["plan"]
    rule_rec = bundle["rule"]
    fc_rec = bundle["fc"]
    zones_meta = bundle["zones_meta"]
    victims_by_muni = bundle["victims_by_municipality"]

    # Map forecast by week
    fc_by_week = {int(f.week): f for f in forecasts_list if hasattr(f, "week")}
    if 1 not in fc_by_week and fc_rec is not None:
        fc_by_week[1] = fc_rec

    # A plan carries no reason; the reason is in the plan's approval entry in the log (runs/season_log.csv)
    log_df = load_saved_log(str(root))
    plan_reason = {}
    if not log_df.empty and {"entry", "kind", "subject", "reason"} <= set(log_df.columns):
        approvals = log_df[(log_df["entry"] == "decision") & (log_df["kind"] == "plan")]
        plan_reason = dict(zip(approvals["subject"].astype(str), approvals["reason"].astype(str)))

    plan_choices = ["Week 1: Integer Programme Plan vs. Rule of Thumb"]
    for p in plans_list:
        p_id = p.get("plan_id", f"plan-w{int(p.get('approved_week', p.get('week', 1))):02d}")
        w_app = int(p.get("approved_week", p.get("week", 1)))
        rsn = plan_reason.get(str(p_id), "")
        plan_choices.append(f"{p_id} — Approved in Week {w_app}" + (f" ({rsn})" if rsn else ""))

    selected_plan_label = st.selectbox(
        "Select approved plan & forecast to inspect:",
        options=plan_choices,
        index=0,
    )

    if selected_plan_label.startswith("Week 1:"):
        active_plan = milp_rec.answer if (milp_rec and milp_rec.ok) else {}
        active_fc_rec = fc_rec
        n_s = len(active_plan.get("survey", []))
        n_c = len(active_plan.get("clearance", []))
        raw_rule = rule_rec.answer if (rule_rec and rule_rec.ok) else {}
        compare_rule = {
            "survey": raw_rule.get("survey", [])[:n_s],
            "clearance": raw_rule.get("clearance", [])[:n_c],
        }
    else:
        p_idx = plan_choices.index(selected_plan_label) - 1
        active_plan = plans_list[p_idx]
        w_app = int(active_plan.get("approved_week", active_plan.get("week", 1)))
        active_fc_rec = fc_by_week.get(w_app, fc_rec)
        compare_rule = {}

    active_fc_dict = active_fc_rec.answer if (active_fc_rec and active_fc_rec.ok) else {}
    expected = active_plan.get("expected", {})
    totals_fc = active_fc_dict.get("totals", {})
    free_fc = totals_fc.get("zones declared free", {})
    clr_fc = totals_fc.get("mine areas cleared", {})

    # Planning KPI metrics
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric(
        "Planned Survey / Clearance Zones",
        f"{len(active_plan.get('survey', []))} / {len(active_plan.get('clearance', []))}",
    )
    p2.metric(
        "Consultations the Plan Requests",
        len(active_plan.get("consultations", [])),
    )
    p3.metric(
        "Expected Yield (Planning Factors)",
        f"{expected.get('zones declared free', 0)} zones free · {expected.get('mine areas cleared', 0.0):.1f} mine areas cleared",
    )
    p4.metric(
        "Forecast Zones Free (P10 / P50 / P90)",
        (
            f"{int(free_fc.get('P10', 0))} / {int(free_fc.get('P50', 0))} / {int(free_fc.get('P90', 0))}"
            if free_fc
            else "—"
        ),
    )
    p5.metric(
        "Forecast Mine Areas Cleared (P10 / P50 / P90)",
        (
            f"{int(clr_fc.get('P10', 0))} / {int(clr_fc.get('P50', 0))} / {int(clr_fc.get('P90', 0))}"
            if clr_fc
            else "—"
        ),
    )

    # Render the 3-panel visual planning figure
    fig_plan = plot_planning_schedule_and_allocation(
        active_plan,
        compare_rule,
        active_fc_dict,
        zones_meta,
        victims_by_muni,
    )
    st.pyplot(fig_plan)
    plt.close(fig_plan)

    # Summary of all 12 monthly plans / replans across the season
    if plans_list:
        with st.expander(
            f"Compare all {len(plans_list)} monthly plans approved across the 52-week season, with each plan's "
            "completion forecast",
            expanded=False,
        ):
            replan_rows = []
            for p in plans_list:
                w_app = int(p.get("approved_week", p.get("week", 1)))
                exp_p = p.get("expected", {})
                f_rec = fc_by_week.get(w_app)
                f_tot = f_rec.answer.get("totals", {}) if (f_rec and f_rec.ok) else {}
                zf = f_tot.get("zones declared free", {})
                mc = f_tot.get("mine areas cleared", {})
                replan_rows.append(
                    {
                        "Plan": p.get("plan_id", ""),
                        "Week Approved": w_app,
                        "Planning Method": p.get("planner", ""),
                        "Reason for Plan (log)": plan_reason.get(str(p.get("plan_id", "")), ""),
                        "Survey Zones": len(p.get("survey", [])),
                        "Clearance Zones": len(p.get("clearance", [])),
                        "Consultations Requested": len(p.get("consultations", [])),
                        "Expected Zones Declared Free": int(exp_p.get("zones declared free", 0)),
                        "Expected Mine Areas Cleared": round(
                            float(exp_p.get("mine areas cleared", 0.0)), 1
                        ),
                        "Forecast Zones Declared Free (P10 / P50 / P90)": (
                            f"{int(zf.get('P10', 0))} / {int(zf.get('P50', 0))} / {int(zf.get('P90', 0))}"
                            if zf
                            else "—"
                        ),
                        "Forecast Mine Areas Cleared (P10 / P50 / P90)": (
                            f"{int(mc.get('P10', 0))} / {int(mc.get('P50', 0))} / {int(mc.get('P90', 0))}"
                            if mc
                            else "—"
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(replan_rows), width="stretch", hide_index=True)


def render_total_dashboard(root: Path) -> None:
    """Render the unified Command & Planning Overview (Inside the twin, restricted)."""
    restricted_header(
        "Command & Planning Overview",
        "One operational picture for the 15 pilot municipalities of Antioquia: regional grid map, "
        "municipality drill-down, season planning & completion forecasts, and weekly progress.",
    )

    log_df = load_saved_log(str(root))
    reviews_df = load_saved_reviews_public(str(root))
    run_info = load_run_info(str(root))
    gate_df = load_gate_table(str(root))
    saved_sitreps = load_saved_sitreps(str(root))
    bundle = get_twin_bundle(root)

    # Top Week Slider & Model Provenance Bar
    c_sl, c_meta = st.columns([2, 3])
    with c_sl:
        week = st.slider(
            "Select week of the season (1 to 52)",
            min_value=1,
            max_value=52,
            value=52,
            step=1,
            key="total_dash_week",
        )
    with c_meta:
        agent_model = run_info.get("model", "saved model")
        gate_summary = (
            ", ".join(f"{k} in {v}" for k, v in gate_df["chosen_label"].value_counts().items())
            if not gate_df.empty and "chosen_label" in gate_df.columns
            else "fitted risk models"
        )
        st.markdown(f"### Programme Snapshot — Week {week} of 52")
        st.caption(
            f"Dossier reviews & situation reports model: **{agent_model}** · "
            f"Area risk prioritisation by municipality: {gate_summary}"
        )

    # Executive KPI Strip at selected week
    zones_meta = bundle["zones_meta"]
    areas_meta = bundle["areas_meta"]
    tasks_df = bundle.get("tasks_df", pd.DataFrame())
    z_state_w = grid_map.zone_state_at_week(bundle, week)
    a_status_w = grid_map.area_status_at_week(bundle, week)
    rev_w = reviews_df[reviews_df["week"] <= week] if not reviews_df.empty else pd.DataFrame()
    vetoes_w = (
        int(rev_w["vetoed at sign-off"].sum())
        if (not rev_w.empty and "vetoed at sign-off" in rev_w.columns)
        else 0
    )
    plans_w = [p for p in bundle.get("plans", []) if int(p.get("approved_week", 1)) <= week]
    total_victims = int(bundle["victims_by_municipality"].sum())

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric(
        "Zones Declared Free",
        f"{int((z_state_w == 'free').sum())} of {len(zones_meta)}",
        f"{int((z_state_w == 'surveyed').sum())} surveyed open",
    )
    m2.metric(
        "Areas Released",
        int(a_status_w.isin(["cancelled", "reduced", "cleared"]).sum()),
        f"{int((a_status_w == 'cleared').sum())} cleared",
    )
    m3.metric(
        "Suspected / Confirmed Areas",
        f"{int((a_status_w == 'suspected').sum())} / {int((a_status_w == 'confirmed').sum())}",
    )
    m4.metric(
        "Sign-Off Reviews Completed",
        f"{len(rev_w)} of {len(reviews_df)}",
        f"{vetoes_w} vetoed by the rule check",
    )
    m5.metric(
        "Monthly Plans Approved",
        len(plans_w),
        f"Active: {plans_w[-1].get('plan_id', '')}" if plans_w else "—",
    )
    m6.metric(
        "Recorded Victims (All Years)",
        f"{total_victims:,}",
        "15 municipalities (context)",
    )

    # ---- Interactive map of the twin's cell grid + the risk model's chance of mines --------------------------------
    st.markdown("---")
    st.subheader("Interactive Map and the Risk Model's Chance of Mines")
    st.caption(
        "The 15 pilot municipalities of Antioquia on the twin's schematic grid, with no geographic coordinates, "
        "axes or scale. **Scroll to zoom, drag to pan, hover over any area or zone for details, click to pin it in "
        "the inspector, or press ▶ Play Season** to step through weeks 1 to 52. The heat map shows the risk "
        "model's chance that an area holds mines: where to look first. A chance is not evidence about an area, and "
        "no area is released on it; released areas are drawn on top."
    )

    interactive_map.render_interactive_country_explorer(
        bundle=bundle,
        log_df=log_df,
        reviews_df=reviews_df,
        gate_df=gate_df,
        initial_week=week,
    )

    with st.expander(
        "Side-by-side regional maps: the risk model's chance of mines, and land release status in the selected week",
        expanded=True,
    ):
        hm_c1, hm_c2, hm_c3 = st.columns([2.4, 1.4, 1.2])
        risk_source_map = {
            "Chance in force in the selected week (updated at each monthly plan)": "updated",
            "Start of season (week 1)": "initial",
            "Nearness to past accidents (the baseline)": "baseline",
        }
        with hm_c1:
            chosen_heat_label = st.selectbox(
                "Risk model's chance of mines:",
                options=list(risk_source_map.keys()),
                index=0,
                key="hero_heat_source",
            )
        with hm_c2:
            show_cleaned_contours = st.checkbox(
                "Outline the areas released by the selected week",
                value=True,
                key="hero_show_cleaned",
            )
        with hm_c3:
            show_muni_labels = st.checkbox(
                "Show municipality names",
                value=True,
                key="hero_show_labels",
            )

        fig_hero = grid_map.plot_regional_hero_dashboard_map(
            bundle,
            week=week,
            log_df=log_df,
            risk_source=risk_source_map[chosen_heat_label],
            show_cleaned_contours=show_cleaned_contours,
            show_muni_labels=show_muni_labels,
        )
        st.pyplot(fig_hero)
        plt.close(fig_hero)

    st.markdown("---")
    st.markdown("### Contamination Categories, Real-World Grounding & Programme Breakdowns")
    tab_reg, tab_ground, tab_map, tab_plan, tab_prog = st.tabs(
        [
            "💣 Contamination Categories (Ordnance Library)",
            "🌍 Real-World Grounding: Terrain, Past Output & Rules",
            "🗺️ Municipality Explorer & All-15 Grid Layers",
            "📅 Planning & Forecast Dashboard",
            "📈 Season Progress & Team Capacity",
        ]
    )

    with tab_reg:
        ordnance_library.render_catalogue(root, bundle)

    with tab_ground:
        grounding_panels.render_grounding_and_calibration_section(root, bundle, log_df, reviews_df, week)

    with tab_map:
        grid_map.render_municipality_map_explorer(root, bundle, log_df, gate_df, saved_sitreps, week)

    with tab_plan:
        render_planning_dashboard_section(root, bundle, gate_df)

    with tab_prog:
        st.subheader("Season Progress, Field Team Capacity & Sign-Off Summary")
        st.caption(
            "Tracks cumulative zones surveyed and declared free across the 52 weeks (vertical dotted lines "
            "mark each monthly replan), weekly team-days worked vs. lost to rain during the March–April and "
            "October–November rainy seasons, zone progress across the 15 municipalities, and sign-off decisions."
        )
        fig_exec = plot_season_overview_panels(bundle, log_df, reviews_df, week)
        st.pyplot(fig_exec)
        plt.close(fig_exec)
