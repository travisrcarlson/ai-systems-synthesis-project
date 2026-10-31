"""The land release steps for one area, as the twin records them (restricted).

Follows an area (`area_id`, in its zone `zone_id`) through the twin's own steps (`twin/season.py`, `twin/core.py`):
  Step 1: Desk study: past accidents in the national database, and the risk model's chance of holding mines
  Step 2: Zone survey: the plan, community consultation where required, the access feed, and the zone survey
  Step 3: Visit: the non-technical survey visit, and the area's evidence as the reviewer's register classes it
  Step 4: Review and sign-off: the reviewer's recommendation, the rule check (R1 to R6) and the QC reviewer's sign-off
  Step 5: Clearance: the clearance team's task and its result, as the log records it
  Step 6: Zone declared free: the area released (cancelled, reduced or cleared) and its zone declared free

Each step shows the twin's records at the end of the selected week (the tasks table, the saved reviews and
`runs/season_log.csv`); step 4 also shows the saved sign-off sheet. Beside them, a console lets the user practise the
QC reviewer's check. It changes nothing in the twin or the saved season, whose sign-offs follow the season's stated
sign-off policy; the practice choices stay in `st.session_state["hitl_pipeline_validations"]`. Read-only on disk.
"""
from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import streamlit as st

from twin.core import RELEASED, event_band
from twin.season import ROLES
import grid_map
import ordnance_library
from loaders import (
    get_twin_bundle,
    load_area_ordnance_findings,
    load_gate_table,
    load_review_detail,
    load_saved_log,
    load_saved_reviews_public,
    nav_to_dossier_review,
    nav_to_log_zone,
    nav_to_operations_week,
    restricted_header,
)

SURFACE = "#faf9f5"
INK = "#181816"
INK_2 = "#52514e"
MUTED = "#898781"

# The twin's steps for an area (twin/season.py, twin/core.py). IMAS 07.11 (§4.2) sets no fixed sequence for land
# release; these are the steps this twin follows.
PIPELINE_STAGES = [
    {
        "stage_num": 1,
        "short_title": "1. Desk study",
        "full_title": "Step 1 — Desk study: past accidents in the national database, and the risk model's chance",
        "icon": "📋",
        "colour": "#475569",
        "basis": "IMAS 08.10, §7 (desk studies), as the rules file's <code>desk_study</code> section paraphrases it; "
                 "the desk study's record in <code>twin/core.py</code>",
        "purpose": "Before the season, the twin holds each area's record in the national database. An area with a past "
                   "landmine accident recorded inside starts the season suspected: the desk study's record is "
                   "indirect evidence (reliability class 6 in the rules file). The risk model's chance of holding "
                   "mines sets where to survey first; a chance is not evidence about an area, and no area is "
                   "released on it.",
    },
    {
        "stage_num": 2,
        "short_title": "2. Zone survey",
        "full_title": "Step 2 — Plan, community consultation and zone survey",
        "icon": "🗺️",
        "colour": "#2563eb",
        "basis": "The twin's tasking check (<code>twin/core.py</code>): a zone is tasked only when the access feed "
                 "shows it open and, where required, its communities have been consulted",
        "purpose": "The monthly plan, approved by the departmental office, lists the zones to survey. Where the "
                   "consultation layer requires it, the communities are consulted first. A survey team (S1 to S4) "
                   "surveys the zone with its communities and flags the areas to visit: every area with a recorded "
                   "accident, and the areas people point to.",
    },
    {
        "stage_num": 3,
        "short_title": "3. Visit",
        "full_title": "Step 3 — Non-technical survey visit and evidence log",
        "icon": "🥾",
        "colour": "#0d9488",
        "basis": "IMAS 08.10, §6.1 (direct and indirect evidence) and §8.2 (reliability classes 1 to 7), as the "
                 "rules file paraphrases them",
        "purpose": "A survey team visits each flagged area; what it sees and what people tell it go into the area's "
                   "evidence log. The reviewer agent's register classes each item: direct, indirect, evidence of "
                   "absence or not evidence, and the source's reliability class (1 or 2 for what the survey team "
                   "sees, 4 for first-hand sources, 5 or 7 for people told by others, 3 or 6 for records).",
    },
    {
        "stage_num": 4,
        "short_title": "4. Review and sign-off",
        "full_title": "Step 4 — Dossier review, rule check (R1 to R6) and the QC reviewer's sign-off",
        "icon": "⚖️",
        "colour": "#6d28d9",
        "basis": "<code>rules/land_release_rules.md</code> (IMAS paraphrase and project rules); the rule check R1 to "
                 "R6 (<code>twin/rules.py</code>); the QC reviewer signs off",
        "purpose": "The dossier reviewer recommends CANCEL (release by non-technical survey), CLEARANCE (send to "
                   "clearance; the area is confirmed only on direct evidence) or FURTHER_NTS (another visit); "
                   "TECHNICAL_SURVEY is not open, since the pilot's national criteria provide no technical survey. "
                   "The rule check (R1 to R6) tests the recommendation on the reviewer's register, and the QC "
                   "reviewer signs off (season sign-off policy: signs off what passes the rule check).",
    },
    {
        "stage_num": 5,
        "short_title": "5. Clearance",
        "full_title": "Step 5 — Clearance",
        "icon": "🚜",
        "colour": "#b91c1c",
        "basis": "IMAS 07.11, §3.3 and §4.4, as the rules file's <code>clearance</code> section paraphrases them",
        "purpose": "A clearance team (C1 to C4) clears an area sent to clearance. Its result is signed off as the log "
                   "records it (\"N mine cell(s) found and cleared\", or \"clearance found no mines\"), and the area "
                   "is then cleared, which releases it.",
    },
    {
        "stage_num": 6,
        "short_title": "6. Zone declared free",
        "full_title": "Step 6 — Release and zone declared free",
        "icon": "🏁",
        "colour": "#15803d",
        "basis": "IMAS 07.11, §3.2 (cancelled land) and §3.3 (cleared land), as the rules file paraphrases them; the "
                 "twin's check before a zone is declared free (<code>twin/core.py</code>)",
        "purpose": "An area is released when it is cancelled, reduced or cleared. The national authority declares "
                   "the zone free once it is surveyed, every flagged area is released and no task is open.",
    },
]

# Listed first in the area selector: eight areas whose records reach different steps (visits sent back, clearance with
# and without mines found, a cancellation, a clearance not finished at week 52). Their labels, like every area's, are
# computed from the log at the selected week.
LISTED_FIRST = ("ECV-0050", "NAR-0283", "COC-0004", "ECV-0018", "COC-0090", "NAR-0282", "ABE-0294", "SAB-0214")

STAGE_DROPDOWN_KEYS = ("pipeline_stage_dropdown_main", "pipeline_stage_dropdown_emb")

GREEN, GREEN_BG = "#15803d", "#dcfce7"
AMBER, AMBER_BG = "#b45309", "#fef3c7"
GREY, GREY_BG = "#64748b", "#f1f5f9"


def _task_status_at_week(tasks_df: pd.DataFrame, log_df: pd.DataFrame, week: int) -> pd.Series:
    """Each task's status at the end of `week`: ordered, in progress, paused or done, from the tasks table's weeks
    and the log's pauses and resumed orders ("not yet ordered" for a task ordered later)."""
    if tasks_df is None or tasks_df.empty:
        return pd.Series(dtype=object)
    t = tasks_df.set_index("task_id")
    out = pd.Series("ordered", index=t.index, dtype=object)
    out[(t["started_week"] > 0) & (t["started_week"] <= week)] = "in progress"
    if log_df is not None and not log_df.empty:
        ev = log_df[
            (log_df["entry"] == "decision")
            & (log_df["kind"].isin(["pause", "task order"]))
            & (log_df["week"] <= week)
            & (log_df["subject"].astype(str).isin(t.index))
        ].sort_values("n")
        last_kind = ev.groupby("subject")["kind"].last()
        out.loc[last_kind.index[last_kind == "pause"]] = "paused"
    out[(t["finished_week"] > 0) & (t["finished_week"] <= week)] = "done"
    out[t["ordered_week"] > week] = "not yet ordered"
    return out


def _access_feed_at_week(log_df: pd.DataFrame, week: int) -> dict[str, str]:
    """Each zone's status in the access feed at `week`, replayed from the log's access reports up to that week
    ("access feed: ABE-Z14 closed, ..." or "access feed: no change"). The twin starts every zone open."""
    state: dict[str, str] = {}
    if log_df is None or log_df.empty:
        return state
    acc = log_df[(log_df["entry"] == "report") & (log_df["kind"] == "access") & (log_df["week"] <= week)]
    for reason in acc.sort_values("n")["reason"].astype(str):
        for zone, status in re.findall(r"([A-Z]{3}-Z\d{2}) (open|closed)", reason.split("access feed:", 1)[-1]):
            state[zone] = status
    return state


def _task_table(tasks_df: pd.DataFrame, t_status: pd.Series, mask: pd.Series, week: int) -> pd.DataFrame:
    """The tasks selected by `mask` that were ordered by the end of `week`, with their status at that week."""
    t = tasks_df[mask & (tasks_df["ordered_week"] <= week)].copy()
    t["status"] = t["task_id"].map(t_status)
    for col in ("started_week", "finished_week"):
        t[col] = t[col].where(t[col] <= week, 0)
    return t


def _select_stage(stage: int) -> None:
    """Show step `stage` (the step buttons' callback), keeping the step dropdowns in step with it."""
    st.session_state["pipeline_selected_stage"] = int(stage)
    for key in STAGE_DROPDOWN_KEYS:
        if key in st.session_state:
            st.session_state[key] = int(stage)


def _stage_from_dropdown(key: str) -> None:
    st.session_state["pipeline_selected_stage"] = int(st.session_state[key])


@st.cache_data(show_spinner=False)
def build_area_pipeline_profiles(
    root_str: str,
    week: int = 52,
) -> pd.DataFrame:
    """One row per area: the furthest of the twin's steps it has reached at the end of `week`, and its counts for that
    week (visits, reviews, clearance), from the twin's tasks table, the saved reviews and the log."""
    root = Path(root_str)
    bundle = get_twin_bundle(root)
    log_df = load_saved_log(root_str)
    rev_df = load_saved_reviews_public(root_str)
    gate_df = load_gate_table(root_str)
    ord_findings = load_area_ordnance_findings(root_str)
    classified_df = ordnance_library.classify_area_evidence(bundle, ord_findings, gate_df)

    areas_meta = bundle["areas_meta"]
    tasks_df = bundle.get("tasks_df", pd.DataFrame())

    status_w = grid_map.area_status_at_week(bundle, week)
    z_state_w = grid_map.zone_state_at_week(bundle, week)
    t_status = _task_status_at_week(tasks_df, log_df, week)

    # Tasks ordered by the end of `week`, by area and by zone
    visits_by_area: dict[str, list[str]] = {}
    clr_by_area: dict[str, list[str]] = {}
    zsurv_zones: set[str] = set()
    if not tasks_df.empty:
        ordered = tasks_df[tasks_df["ordered_week"] <= week].sort_values("task_id")
        for kind, tid, area, zone in ordered[["kind", "task_id", "area", "zone"]].itertuples(index=False):
            if kind == "VISIT" and isinstance(area, str):
                visits_by_area.setdefault(area, []).append(str(tid))
            elif kind == "CLEARANCE" and isinstance(area, str):
                clr_by_area.setdefault(area, []).append(str(tid))
            elif kind == "ZONE_SURVEY":
                zsurv_zones.add(str(zone))

    # Reviews signed off by the end of `week`, by area
    revs_by_area: dict[str, list[dict]] = {}
    if not rev_df.empty:
        for rrow in rev_df[rev_df["week"] <= week].sort_values(["week", "visit"]).to_dict("records"):
            revs_by_area.setdefault(str(rrow["area_id"]), []).append(rrow)

    rows = []
    for aid, arow in areas_meta.iterrows():
        aid_s = str(aid)
        zid_s = str(arow["zone_id"])
        a_st = str(status_w.get(aid_s, "unknown"))
        z_st = str(z_state_w.get(zid_s, "not surveyed"))

        v_ids = visits_by_area.get(aid_s, [])
        c_ids = clr_by_area.get(aid_s, [])
        r_list = revs_by_area.get(aid_s, [])
        v_done = sum(1 for t in v_ids if t_status.get(t) == "done")
        latest = str(r_list[-1]["decision"]) if r_list else "—"
        n_further = sum(1 for r in r_list if str(r.get("decision")) == "FURTHER_NTS")

        # the clearance result, only once the area's CLEARED sign-off is in the log by this week
        f_info = ord_findings.get(aid_s, {})
        cw = int(f_info.get("cleared_week", 0) or 0)
        cleared_by_w = 0 < cw <= week
        c_row = classified_df.loc[aid_s] if aid_s in classified_df.index else None

        # The furthest step reached at `week`
        if a_st in RELEASED:
            stage_num = 6
            stage_label = "6. Released (zone declared free)" if z_st == "free" else "6. Released (zone not yet free)"
        elif c_ids or latest == "CLEARANCE" or a_st == "confirmed":
            stage_num, stage_label = 5, "5. Clearance"
        elif r_list:
            stage_num, stage_label = 4, "4. Review and sign-off"
        elif v_ids:
            stage_num, stage_label = 3, "3. Visit"
        elif zid_s in zsurv_zones or z_st in ("in survey", "surveyed", "free"):
            stage_num, stage_label = 2, "2. Zone survey"
        else:
            stage_num, stage_label = 1, "1. Desk study"

        rows.append(
            {
                "area_id": aid_s,
                "zone_id": zid_s,
                "municipality_code": str(arow["municipality_code"]),
                "municipality": str(arow["municipality"]),
                "land_use": str(arow["main_land_use"]),
                "event_inside": bool(arow["event_inside"]),
                "status_w": a_st,
                "zone_state_w": z_st,
                "stage_num": stage_num,
                "stage_label": stage_label,
                "zone_survey_ordered": zid_s in zsurv_zones,
                "visits_ordered": len(v_ids),
                "visits_done": v_done,
                "reviews_count": len(r_list),
                "further_nts_count": n_further,
                "has_resurvey_loop": n_further > 0,
                "latest_review_outcome": latest,
                "clearance_tasks_count": len(c_ids),
                "clearance_status_w": str(t_status.get(c_ids[-1], "")) if c_ids else "",
                "cleared_by_w": cleared_by_w,
                "cleared_cells": int(f_info.get("cleared_cells", 0) or 0) if cleared_by_w else 0,
                "cleared_week": cw if cleared_by_w else 0,
                "clearance_note": str(f_info.get("clearance_note", "")) if cleared_by_w else "",
                "subtypes": str(c_row["tag_str"]) if c_row is not None else "—",
                "has_dossier_or_task": bool(v_ids or r_list or c_ids),
            }
        )
    return pd.DataFrame(rows).set_index("area_id", drop=False)


def _area_step_states(arow: pd.Series) -> list[tuple[str, str, str, str]]:
    """The selected area's state at each step at the selected week: (badge, text, edge colour, fill colour)."""
    a_st = str(arow["status_w"])
    z_st = str(arow["zone_state_w"])
    zone = str(arow["zone_id"])
    states = [("DONE", "Before the season", GREEN, GREEN_BG)]
    # Step 2: zone survey
    if z_st in ("surveyed", "free"):
        states.append(("DONE", f"Zone {zone}\nsurveyed", GREEN, GREEN_BG))
    elif z_st == "in survey" or bool(arow["zone_survey_ordered"]):
        states.append(("IN PROGRESS", f"Zone {zone}\nin survey", AMBER, AMBER_BG))
    else:
        states.append(("NOT YET", f"Zone {zone}\nnot surveyed", GREY, GREY_BG))
    # Step 3: visits
    v_done, v_ord = int(arow["visits_done"]), int(arow["visits_ordered"])
    if v_done:
        txt = f"{v_done} visit(s) done" + (f"\n{v_ord - v_done} not finished" if v_ord > v_done else "")
        states.append(("DONE" if v_ord == v_done else "IN PROGRESS", txt, "#0d9488", "#ccfbf1"))
    elif v_ord:
        states.append(("IN PROGRESS", "Visit ordered,\nnot finished", AMBER, AMBER_BG))
    else:
        states.append(("NOT YET", "No visit yet", GREY, GREY_BG))
    # Step 4: review and sign-off
    r_cnt = int(arow["reviews_count"])
    latest = str(arow["latest_review_outcome"])
    if r_cnt:
        if latest == "FURTHER_NTS":
            states.append(("SENT BACK", f"{r_cnt} review(s), latest:\n{latest}", AMBER, AMBER_BG))
        else:
            states.append(("SIGNED OFF", f"{r_cnt} review(s), latest:\n{latest}", "#6d28d9", "#ede9fe"))
    else:
        states.append(("NOT YET", "No review yet", GREY, GREY_BG))
    # Step 5: clearance (its result only after the area's CLEARED sign-off)
    if bool(arow["cleared_by_w"]):
        found = int(arow["cleared_cells"]) > 0
        states.append(("CLEARED", textwrap.fill(str(arow["clearance_note"]), 20),
                       "#b91c1c" if found else "#1d4ed8", "#fee2e2" if found else "#dbeafe"))
    elif int(arow["clearance_tasks_count"]):
        c_st = str(arow["clearance_status_w"])
        states.append((c_st.upper(), f"Clearance {c_st}", AMBER, AMBER_BG))
    elif a_st == "cancelled":
        states.append(("NOT NEEDED", "Cancelled by\nnon-technical survey", "#0f766e", "#e0f2fe"))
    elif latest == "CLEARANCE" or a_st == "confirmed":
        states.append(("WAITING", "Sent to clearance,\nno team yet", AMBER, AMBER_BG))
    else:
        states.append(("NOT ORDERED", "No clearance ordered", GREY, GREY_BG))
    # Step 6: release and zone declared free
    if z_st == "free":
        states.append(("ZONE FREE", f"Area {a_st};\nzone declared free", GREEN, GREEN_BG))
    elif a_st in RELEASED:
        states.append(("RELEASED", f"Area {a_st};\nzone not yet free", GREEN, GREEN_BG))
    else:
        states.append(("NOT YET", f"Area {a_st}", AMBER, AMBER_BG))
    return states


def plot_pipeline_stage_funnel_and_Area_stepper(
    pipe_df: pd.DataFrame,
    selected_area_id: str,
    selected_stage: int,
    week: int,
) -> plt.Figure:
    """Draw two rows:
    - top: all areas, counted at each of the twin's six steps by the end of `week` (zone surveys finished, visits
      done, reviews signed off, areas cleared, zones declared free), with the areas sent back for another visit;
    - bottom: the selected area's state at each step at that week, with the step shown below outlined.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(15.8, 4.9), facecolor=SURFACE, gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.28}
    )
    for ax in (ax_top, ax_bot):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ---- Top row: all areas, by the end of the week ----
    ax_top.text(
        1.0,
        9.1,
        f"All areas: the twin's six steps, by the end of week {week}",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        va="center",
    )

    xs = [2.0, 18.2, 34.4, 50.6, 66.8, 83.0]
    w_box = 14.2
    h_box = 6.2
    y_box = 1.6

    zones = pipe_df.drop_duplicates("zone_id")
    n_total = len(pipe_df)
    n_event = int(pipe_df["event_inside"].sum())
    n_zones = len(zones)
    n_zs = int(zones["zone_state_w"].isin(["surveyed", "free"]).sum())
    n_free = int((zones["zone_state_w"] == "free").sum())
    n_areas_sz = int(pipe_df["zone_state_w"].isin(["surveyed", "free"]).sum())
    n_visits = int(pipe_df["visits_done"].sum())
    n_visited = int((pipe_df["visits_done"] > 0).sum())
    n_rev = int(pipe_df["reviews_count"].sum())
    n_reviewed = int((pipe_df["reviews_count"] > 0).sum())
    n_resurv = int(pipe_df["has_resurvey_loop"].sum())
    n_clr = int(pipe_df["cleared_by_w"].sum())
    n_clr_cells = int(pipe_df["cleared_cells"].sum())
    st_counts = pipe_df["status_w"].value_counts()
    n_canc = int(st_counts.get("cancelled", 0))
    n_red = int(st_counts.get("reduced", 0))
    n_clr_st = int(st_counts.get("cleared", 0))
    n_rel = n_canc + n_red + n_clr_st
    rel_parts = [f"{n_canc} cancelled"] + ([f"{n_red} reduced"] if n_red else []) + [f"{n_clr_st} cleared"]

    stage_throughput = [
        (f"{n_total:,} areas", f"{n_event:,} with an accident\nrecorded inside"),
        (f"{n_zs} of {n_zones} zones surveyed", f"{n_areas_sz:,} areas in\nsurveyed zones"),
        (f"{n_visits} visits done", f"{n_visited} areas visited"),
        (f"{n_rev} reviews signed off", f"{n_reviewed} areas reviewed,\n{n_resurv} sent back for a visit"),
        (f"{n_clr} areas cleared", f"{n_clr_cells} mine cells\nfound and cleared"),
        (f"{n_free} zones declared free", f"{n_rel} areas released\n({', '.join(rel_parts)})"),
    ]

    for idx, st_info in enumerate(PIPELINE_STAGES):
        x0 = xs[idx]
        col = st_info["colour"]
        is_sel = st_info["stage_num"] == selected_stage
        ax_top.add_patch(
            FancyBboxPatch(
                (x0, y_box),
                w_box,
                h_box,
                boxstyle="round,pad=0.18,rounding_size=0.45",
                linewidth=2.4 if is_sel else 1.2,
                edgecolor=col if is_sel else "#b8b4a9",
                facecolor="#ffffff" if not is_sel else "#f3efe4",
            )
        )
        # Top colour header strip
        ax_top.add_patch(
            FancyBboxPatch(
                (x0 + 0.15, y_box + h_box - 1.55),
                w_box - 0.30,
                1.40,
                boxstyle="round,pad=0.05,rounding_size=0.3",
                linewidth=0,
                facecolor=col,
            )
        )
        ax_top.text(
            x0 + w_box / 2,
            y_box + h_box - 0.85,
            f"{st_info['short_title']}",
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color="#ffffff",
        )
        main_stat, sub_stat = stage_throughput[idx]
        ax_top.text(
            x0 + w_box / 2,
            y_box + 3.25,
            main_stat,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=INK,
        )
        ax_top.text(
            x0 + w_box / 2,
            y_box + 1.45,
            sub_stat,
            ha="center",
            va="center",
            fontsize=7.3,
            color=INK_2,
        )

        if idx < 5:
            ax_top.annotate(
                "",
                xy=(xs[idx + 1] - 0.15, y_box + h_box / 2),
                xytext=(x0 + w_box + 0.15, y_box + h_box / 2),
                arrowprops={"arrowstyle": "->,head_width=0.35,head_length=0.4", "color": "#52514e", "lw": 1.6},
            )

    # The two ways out of review that skip a step
    ax_top.text(
        49.5,
        0.45,
        f"FURTHER_NTS sent {n_resurv} area(s) back from review (step 4) for another visit (step 3)   ·   "
        "CANCEL releases an area by non-technical survey, without clearance (step 4 to step 6)",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#6d28d9",
    )

    # ---- Bottom row: the selected area ----
    if selected_area_id is None or selected_area_id not in pipe_df.index:
        ax_bot.text(
            50.0,
            5.0,
            f"No area has a visit ordered by the end of week {week}.",
            ha="center",
            va="center",
            fontsize=10.0,
            color=INK_2,
        )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.03)
        return fig

    arow = pipe_df.loc[selected_area_id]
    ax_bot.text(
        1.0,
        9.0,
        f"Area {selected_area_id} ({arow['municipality']}, zone {arow['zone_id']}): {arow['status_w']} at the end "
        f"of week {week} · showing step {selected_stage} ({PIPELINE_STAGES[selected_stage - 1]['short_title'][3:]})",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
        va="center",
    )

    area_stage_states = _area_step_states(arow)
    for idx, st_info in enumerate(PIPELINE_STAGES):
        x0 = xs[idx]
        is_sel = st_info["stage_num"] == selected_stage
        badge_txt, desc_txt, edge_c, fill_c = area_stage_states[idx]

        ax_bot.add_patch(
            FancyBboxPatch(
                (x0, 1.3),
                w_box,
                6.3,
                boxstyle="round,pad=0.18,rounding_size=0.45",
                linewidth=2.8 if is_sel else 1.3,
                edgecolor="#181816" if is_sel else edge_c,
                facecolor=fill_c,
            )
        )
        ax_bot.text(
            x0 + w_box / 2,
            6.55,
            st_info["short_title"],
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=INK,
        )
        ax_bot.text(
            x0 + w_box / 2,
            4.85,
            badge_txt,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=edge_c,
        )
        ax_bot.text(
            x0 + w_box / 2,
            3.15,
            desc_txt,
            ha="center",
            va="center",
            fontsize=7.4,
            color=INK,
        )
        if is_sel:
            ax_bot.text(
                x0 + w_box / 2,
                1.80,
                "SHOWN BELOW",
                ha="center",
                va="center",
                fontsize=6.9,
                fontweight="bold",
                color="#181816",
            )
        if idx < 5:
            ax_bot.annotate(
                "",
                xy=(xs[idx + 1] - 0.15, 4.45),
                xytext=(x0 + w_box + 0.15, 4.45),
                arrowprops={"arrowstyle": "->,head_width=0.32,head_length=0.38", "color": edge_c, "lw": 1.5},
            )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.03)
    return fig


def _render_stage_paperwork_and_hitl_panel(
    root: Path,
    bundle: dict,
    log_df: pd.DataFrame,
    rev_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    ord_findings: dict[str, dict],
    classified_df: pd.DataFrame,
    pipe_df: pd.DataFrame,
    area_id: str,
    stage_num: int,
    week: int,
) -> None:
    """Show the twin's records for `area_id` at step `stage_num`, at the end of `week` (left), beside the console where
    the user practises the QC reviewer's check (right), which changes nothing in the twin or the saved season.
    """
    if "hitl_pipeline_validations" not in st.session_state:
        st.session_state["hitl_pipeline_validations"] = {}

    st_info = PIPELINE_STAGES[stage_num - 1]
    arow = pipe_df.loc[area_id]
    crow = classified_df.loc[area_id]
    zid = str(arow["zone_id"])
    mcode = str(arow["municipality_code"])
    mname = str(arow["municipality"])
    zones_meta = bundle["zones_meta"]
    zrow = zones_meta.loc[zid]
    tasks_df = bundle.get("tasks_df", pd.DataFrame())
    t_status = _task_status_at_week(tasks_df, log_df, week)
    log_w = log_df[log_df["week"] <= week] if not log_df.empty else log_df
    a_revs = (
        rev_df[(rev_df["area_id"] == area_id) & (rev_df["week"] <= week)].sort_values(["week", "visit"])
        if not rev_df.empty
        else pd.DataFrame()
    )
    chance_w1, _ = grid_map.chance_in_force(bundle, 1)
    chance_w, _ = grid_map.chance_in_force(bundle, week)
    p1 = float(chance_w1.get(area_id, float("nan")))
    pw = float(chance_w.get(area_id, float("nan")))
    task_cols = [
        "task_id",
        "kind",
        "team",
        "zone",
        "area",
        "plan_id",
        "ordered_week",
        "started_week",
        "finished_week",
        "days_planned",
        "days_done",
        "days_lost_weather",
        "days_lost_access",
        "status",
    ]
    task_caption = (
        f"Status at the end of week {week}, from the tasks table and the log; days worked and lost are each task's "
        "totals at the season's end (week 52)."
    )
    entry_cols = ["week", "entry", "kind", "subject", "decision", "before", "after", "decided_by", "rule_check", "reason"]

    st.markdown(
        f"""
        <div style="border:1px solid #d5d0c3;border-left:6px solid {st_info['colour']};border-radius:8px;padding:12px 16px;background:#ffffff;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <div>
              <span style="font-size:1.08rem;font-weight:800;color:#181816;">{st_info['icon']} {st_info['full_title']}</span>
            </div>
            <span style="font-size:0.80rem;color:#52514e;"><strong>Basis:</strong> {st_info['basis']}</span>
          </div>
          <div style="font-size:0.83rem;color:#2c2a26;margin-top:4px;">{st_info['purpose']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_doc, col_hitl = st.columns([1.45, 1.0])

    # =========================================================================
    # LEFT COLUMN: THE TWIN'S RECORDS FOR THIS STEP, AT THE END OF THE WEEK
    # =========================================================================
    with col_doc:
        with st.container(border=True):
            if stage_num == 1:
                st.markdown(f"#### 📋 Desk study — Area `{area_id}`")
                setup_obj = bundle.get("setup")
                twin_areas = (
                    setup_obj.twin.areas
                    if (setup_obj is not None and hasattr(setup_obj, "twin") and hasattr(setup_obj.twin, "areas"))
                    else pd.DataFrame()
                )
                a_full = twin_areas.loc[area_id] if area_id in twin_areas.index else None
                areas_meta = bundle["areas_meta"]
                act_df = bundle.get("activity_df", pd.DataFrame())
                m_act = act_df[act_df["municipality_code"] == mcode] if not act_df.empty else pd.DataFrame()
                reg_counts = {
                    c["code"]: (int(m_act[c["column"]].sum()) if not m_act.empty else 0)
                    for c in ordnance_library.CATEGORIES
                }
                reg_txt = ", ".join(f"{n:,} {code}" for code, n in reg_counts.items())
                reg_years = (
                    f"{int(m_act['year'].min())}–{int(m_act['year'].max())}" if not m_act.empty else "no records"
                )
                vic_m = int(bundle["victims_by_municipality"].get(mname, 0))
                recorded = bool(crow["event_inside"])

                ds_rows = [
                    {"Field": "Area / zone / municipality", "Record": f"{area_id} (zone {zid}, {mname})"},
                    {
                        "Field": "Size and main land use",
                        "Record": f"{float(areas_meta.loc[area_id, 'km2']):.2f} km² · {areas_meta.loc[area_id, 'main_land_use']}",
                    },
                ]
                if a_full is not None:
                    ds_rows += [
                        {
                            "Field": "Land cover (RELand's base layer)",
                            "Record": f"{float(a_full['forest_share']):.0%} forest, "
                                      f"{float(a_full['used_share']):.0%} farmed, grazed or built on",
                        },
                        {
                            "Field": "Past landmine accidents in the national database",
                            "Record": event_band(float(a_full["event_km"])),
                        },
                    ]
                ds_rows += [
                    {
                        "Field": "The desk study's record of an accident inside (event_inside)",
                        "Record": (
                            "Yes: indirect evidence (reliability class 6 in the rules file), so the area starts the "
                            "season suspected"
                            if recorded
                            else "None: the area starts the season with status unknown"
                        ),
                    },
                    {
                        "Field": "Ranking chosen by the baseline-first gate (models/gate.csv)",
                        "Record": f"{crow['risk_model_family']}",
                    },
                    {"Field": "Week-1 calibrated chance of holding mines", "Record": f"{p1:.2%}"},
                    {
                        "Field": f"For context: items destroyed in {mname}, {reg_years} (datos.gov.co jbqm-s5tn)",
                        "Record": reg_txt,
                    },
                    {
                        "Field": f"For context: recorded victims in {mname} (datos.gov.co yhxn-eqqw)",
                        "Record": f"{vic_m:,} (all years)",
                    },
                ]
                st.dataframe(pd.DataFrame(ds_rows), width="stretch", hide_index=True)
                st.caption(
                    "The two context rows are municipal counts from open data; the twin's desk study does not use "
                    "them (the demining layer is a check of the simulated capacity, and victims are shown beside the "
                    "plan). A chance is not evidence about an area, and no area is released on it."
                )
                checklist_items = [
                    ("ds_1", f"Checked the desk study's record for `{area_id}` (accident recorded inside: {'yes' if recorded else 'no'})."),
                    ("ds_2", f"Read the context figures for {mname} ({reg_txt} items destroyed; {vic_m:,} recorded victims)."),
                    ("ds_3", f"Noted that the week-1 chance of holding mines ({p1:.2%}, ranking: {crow['risk_model_family']}) is not evidence about the area."),
                ]

            elif stage_num == 2:
                st.markdown(f"#### 🗺️ Plan, consultation and zone survey — Zone `{zid}`")
                access_w = _access_feed_at_week(log_df, week).get(zid, "open")
                consult_req = bool(zrow["consultation"])
                n_req = int(zones_meta["consultation"].astype(bool).sum())
                c_log = log_w[(log_w["kind"] == "consultation") & (log_w["subject"] == zid)]
                req_w = c_log.loc[c_log["entry"] == "decision", "week"]
                held_w = c_log.loc[c_log["entry"] == "report", "week"]
                if consult_req:
                    consult_txt = "Yes" + (
                        f"; requested in week {int(req_w.min())}" if len(req_w) else f"; not requested by week {week}"
                    ) + (f", held in week {int(held_w.min())}" if len(held_w) else "")
                else:
                    consult_txt = "No"
                st.markdown(
                    f"- **Zone:** `{zid}` ({mname}) · **Areas in the zone:** `{int(zrow['areas'])}` · "
                    f"**Access by road:** {zrow['access']}\n"
                    f"- **Access feed in week {week}:** `{access_w}` (security coordination feed, simulated)\n"
                    f"- **Community consultation required before tasking:** {consult_txt} "
                    f"(simulated layer; {n_req} of {len(zones_meta)} zones)\n"
                    f"- **Zone state at the end of week {week}:** `{arow['zone_state_w']}`"
                )
                z_tasks = (
                    _task_table(tasks_df, t_status, (tasks_df["zone"] == zid) & (tasks_df["kind"] == "ZONE_SURVEY"), week)
                    if not tasks_df.empty
                    else pd.DataFrame()
                )
                if z_tasks.empty:
                    st.info(f"No zone survey of `{zid}` was ordered by the end of week {week}.")
                else:
                    st.markdown("**Zone survey task order (`tasks_df`):**")
                    st.dataframe(z_tasks[task_cols], width="stretch", hide_index=True)
                    st.caption(task_caption)

                z_task_ids = set(z_tasks["task_id"].astype(str)) if not z_tasks.empty else set()
                z_logs = log_w[
                    log_w["subject"].astype(str).isin(z_task_ids | {zid})
                    | (
                        log_w["kind"].isin(["access", "team report"])
                        & log_w["reason"].astype(str).str.contains(zid, regex=False)
                    )
                ]
                if not z_logs.empty:
                    st.markdown(f"**Log entries for zone `{zid}` up to week {week} (`runs/season_log.csv`):**")
                    st.dataframe(
                        z_logs[["week", "entry", "kind", "subject", "decision", "decided_by", "reason"]],
                        width="stretch",
                        hide_index=True,
                    )

                checklist_items = [
                    ("to_1", f"Checked the access feed for zone `{zid}` in week {week} (`{access_w}`) and its road access ({zrow['access']})."),
                    ("to_2", f"Checked whether community consultation is required in zone `{zid}` ({consult_txt})."),
                    ("to_3", f"Checked the zone survey task order(s) for `{zid}` ({len(z_tasks)} by week {week})."),
                ]

            elif stage_num == 3:
                st.markdown(f"#### 🥾 Visits and evidence register — Area `{area_id}`")
                st.markdown(
                    f"- **What the twin and the saved runs show over the whole season (no device type is assigned):** {crow['tag_str']}\n"
                    f"- **Visits done by the end of week {week}:** `{int(arow['visits_done'])}` "
                    f"(of `{int(arow['visits_ordered'])}` ordered)"
                )
                v_tasks = (
                    _task_table(tasks_df, t_status, (tasks_df["area"] == area_id) & (tasks_df["kind"] == "VISIT"), week)
                    if not tasks_df.empty
                    else pd.DataFrame()
                )
                if not v_tasks.empty:
                    st.markdown("**Visit task orders (`tasks_df`):**")
                    st.dataframe(v_tasks[task_cols], width="stretch", hide_index=True)
                    st.caption(task_caption)

                if not a_revs.empty:
                    for _, r_v in a_revs.iterrows():
                        v_num = int(r_v["visit"])
                        v_wk = int(r_v["week"])
                        det = load_review_detail(str(root), v_wk, area_id, v_num)
                        rec = det.get("record", {}) or {}
                        reg = rec.get("register") or {}
                        reg_rows = []
                        if isinstance(reg, dict):
                            for eid_k, edata in reg.items():
                                if isinstance(edata, dict):
                                    reg_rows.append(
                                        {
                                            "Item": eid_k,
                                            "Evidence Class": edata.get("evidence_class", ""),
                                            "Reliability Class": edata.get("reliability_class", ""),
                                            "Explains": ", ".join(str(x) for x in (edata.get("explains") or [])) or "—",
                                            "Quote": edata.get("quote", ""),
                                            "Reviewer's Note": edata.get("note", ""),
                                        }
                                    )
                        st.markdown(
                            f"##### 📄 Visit {v_num}: the reviewer's evidence register (week {v_wk}, {len(reg_rows)} items)"
                        )
                        if reg_rows:
                            st.dataframe(pd.DataFrame(reg_rows), width="stretch", hide_index=True)
                else:
                    st.info(
                        f"No visit dossier of `{area_id}` was reviewed by the end of week {week}. "
                        + (
                            "The desk study records a past landmine accident inside this area (`event_inside`)."
                            if bool(crow["event_inside"])
                            else "The desk study records no past landmine accident inside this area."
                        )
                    )

                checklist_items = [
                    ("nts_1", "Checked that each item of the register quotes the area's evidence log (items E1, E2, …)."),
                    ("nts_2", "Checked each item's evidence class (direct, indirect, absence, not evidence) and its source's reliability class (1 to 7, IMAS 08.10, §8.2)."),
                    ("nts_3", "Confirmed the reviewer's texts give no advice on handling items (the handling-advice filter; rules file: `safety`)."),
                ]

            elif stage_num == 4:
                st.markdown(f"#### ⚖️ Sign-off sheet and rule check (R1 to R6) — Area `{area_id}`")
                if a_revs.empty:
                    st.info(f"No review of `{area_id}` was signed off by the end of week {week}.")
                    checklist_items = [
                        ("qc_1", f"Noted that no review of `{area_id}` is signed off by week {week}."),
                    ]
                else:
                    visit_opts = a_revs["visit"].astype(int).tolist()
                    sel_v = (
                        st.radio(
                            "Sign-off sheet to show:",
                            options=visit_opts,
                            index=len(visit_opts) - 1,
                            format_func=lambda v: f"Visit {v} (week {int(a_revs[a_revs['visit'] == v].iloc[0]['week'])}) -> {a_revs[a_revs['visit'] == v].iloc[0]['decision']}",
                            horizontal=True,
                            key=f"pipe_qc_visit_{area_id}_{week}",
                        )
                        if len(visit_opts) > 1
                        else visit_opts[0]
                    )
                    v_row = a_revs[a_revs["visit"] == sel_v].iloc[0]
                    det = load_review_detail(str(root), int(v_row["week"]), area_id, int(sel_v))
                    rec = det.get("record", {}) or {}
                    rc = rec.get("rule_check", {}) or {}
                    rc_result = str(rc.get("result", "—")).upper()
                    so_rows = log_df[
                        (log_df["entry"] == "decision")
                        & (log_df["kind"] == "area")
                        & (log_df["subject"] == area_id)
                        & (log_df["week"] == int(v_row["week"]))
                        & (log_df["decision"] == v_row["decision"])
                    ]
                    signed_by = str(so_rows.iloc[0]["decided_by"]) if not so_rows.empty else ROLES["qc"]

                    st.markdown(
                        '<div class="signoff-banner">Recommendation only: a person decides</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"- **Week `{int(v_row['week'])}` · Visit `{sel_v}`** · **Reviewer's recommendation:** `{rec.get('outcome')}` "
                        f"(`{rec.get('confidence')}` confidence) -> **Signed off:** `{v_row['decision']}`, by the {signed_by}\n"
                        f"- **Rule check (R1 to R6, `twin/rules.py`):** `{rc_result}`"
                    )
                    checks_list = rc.get("checks", []) or []
                    if checks_list:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Rule": c.get("rule", ""),
                                        "Passed": "✅ PASS" if c.get("passed") else "❌ VETO",
                                        "Detail": c.get("detail", ""),
                                    }
                                    for c in checks_list
                                ]
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                    if det.get("signoff_md"):
                        with st.expander("📄 Saved sign-off sheet (`runs/reviews/…_signoff.md`)", expanded=True):
                            st.markdown(det["signoff_md"])

                    checklist_items = [
                        ("qc_1", f"Checked the rule check's result (`{rc_result}`) for the recommendation (R1 to R6)."),
                        ("qc_2", f"Read the reviewer's open flags ({len(rec.get('flags', []))} flag(s)) for the QC reviewer who signs off."),
                        ("qc_3", f"Checked the sign-off (`{v_row['decision']}`) for area `{area_id}` (visit {sel_v})."),
                    ]

            elif stage_num == 5:
                st.markdown(f"#### 🚜 Clearance — Area `{area_id}`")
                c_tasks = (
                    _task_table(tasks_df, t_status, (tasks_df["area"] == area_id) & (tasks_df["kind"] == "CLEARANCE"), week)
                    if not tasks_df.empty
                    else pd.DataFrame()
                )
                c_task_ids = set(c_tasks["task_id"].astype(str)) if not c_tasks.empty else set()
                c_logs = log_w[
                    (
                        (log_w["entry"] == "decision")
                        & (log_w["kind"] == "area")
                        & (log_w["subject"] == area_id)
                        & (log_w["decision"].isin(["CLEARANCE", "CLEARED"]))
                    )
                    | log_w["subject"].astype(str).isin(c_task_ids)
                ]
                cancel_revs = a_revs[a_revs["decision"] == "CANCEL"] if not a_revs.empty else pd.DataFrame()

                if c_tasks.empty:
                    if str(arow["status_w"]) == "cancelled" and not cancel_revs.empty:
                        c_rev = cancel_revs.iloc[-1]
                        c_det = load_review_detail(str(root), int(c_rev["week"]), area_id, int(c_rev["visit"]))
                        c_checks = ((c_det.get("record", {}) or {}).get("rule_check", {}) or {}).get("checks", []) or []
                        passed = [str(c.get("rule")) for c in c_checks if c.get("passed")]
                        st.success(
                            f"**Clearance not needed:** `{area_id}` was cancelled by non-technical survey in week "
                            f"{int(c_rev['week'])} (CANCEL; the rule check passed {', '.join(passed) or '—'}). "
                            "The saved rule check:\n\n"
                            + "\n".join(f"- **{c.get('rule')}:** {c.get('detail')}" for c in c_checks)
                        )
                    elif str(arow["latest_review_outcome"]) == "CLEARANCE" or str(arow["status_w"]) == "confirmed":
                        st.info(
                            f"`{area_id}` was sent to clearance (status `{arow['status_w']}`), but no clearance team was "
                            f"ordered to it by the end of week {week}."
                        )
                    else:
                        st.info(f"No clearance was ordered for `{area_id}` by the end of week {week}.")
                else:
                    st.markdown("**Clearance task order (`tasks_df`):**")
                    st.dataframe(c_tasks[task_cols], width="stretch", hide_index=True)
                    st.caption(task_caption)
                    if bool(arow["cleared_by_w"]):
                        st.markdown(
                            f"- **Clearance result, signed off in week {int(arow['cleared_week'])} "
                            f"(`runs/season_log.csv`):** **{arow['clearance_note']}.** The log records no device type."
                        )
                    else:
                        st.markdown(
                            f"- **No clearance result by the end of week {week}:** the clearance is "
                            f"`{arow['clearance_status_w']}`, so no result is signed off yet."
                        )
                if not c_logs.empty:
                    st.markdown(f"**Clearance entries in the log up to week {week} (`runs/season_log.csv`):**")
                    st.dataframe(c_logs[entry_cols], width="stretch", hide_index=True)
                clr_2 = (
                    f"Checked the clearance result in the log ({arow['clearance_note']})."
                    if bool(arow["cleared_by_w"])
                    else f"Noted that no clearance result is signed off for `{area_id}` by week {week}."
                )
                checklist_items = [
                    ("clr_1", f"Checked the clearance task for `{area_id}`: team (C1 to C4), status in week {week}, days worked and lost."),
                    ("clr_2", clr_2),
                ]

            else:
                st.markdown(f"#### 🏁 Release and zone declaration — Area `{area_id}`")
                a_status = str(arow["status_w"])
                rel_logs = log_w[
                    (log_w["subject"].isin([area_id, zid]))
                    & (log_w["after"].isin(["cancelled", "reduced", "cleared", "free"]))
                ]
                area_rel = rel_logs[rel_logs["subject"] == area_id]
                rel_week = int(area_rel.iloc[-1]["week"]) if not area_rel.empty else 0
                zone_free = rel_logs[(rel_logs["subject"] == zid) & (rel_logs["after"] == "free")]
                if a_status == "cancelled":
                    release_txt = f"Cancelled by non-technical survey (CANCEL signed off in week {rel_week})"
                elif a_status == "cleared":
                    release_txt = f"Cleared (CLEARED signed off in week {rel_week})"
                elif a_status == "reduced":
                    release_txt = f"Reduced by technical survey (week {rel_week})"
                else:
                    release_txt = f"Not released by the end of week {week}"
                zone_txt = str(arow["zone_state_w"])
                if str(arow["zone_state_w"]) == "free" and not zone_free.empty:
                    zone_txt += f" (declared free in week {int(zone_free.iloc[-1]['week'])} by the national authority)"

                rec_df = pd.DataFrame(
                    [
                        {"Field": "Area / zone / municipality", "Record": f"{area_id} / {zid} / {mname}"},
                        {"Field": f"Area status (end of week {week})", "Record": a_status},
                        {"Field": "Release", "Record": release_txt},
                        {
                            "Field": "Clearance result (log)",
                            "Record": str(arow["clearance_note"]) if bool(arow["cleared_by_w"]) else "—",
                        },
                        {
                            "Field": "Chance of holding mines (risk model)",
                            "Record": f"{p1:.2%} in week 1 -> {pw:.2%} in force in week {week}",
                        },
                        {"Field": f"Zone {zid} (end of week {week})", "Record": zone_txt},
                    ]
                )
                st.dataframe(rec_df, width="stretch", hide_index=True)
                st.caption(
                    "A chance is not evidence about an area: no area is released on it, and release does not set it "
                    "to 0. The national authority declares a zone free once it is surveyed, every flagged area is "
                    "released and no task is open."
                )
                if not rel_logs.empty:
                    st.markdown(
                        f"**Release sign-off and zone declaration entries up to week {week} (`runs/season_log.csv`):**"
                    )
                    st.dataframe(rel_logs[entry_cols], width="stretch", hide_index=True)
                checklist_items = [
                    ("lr_1", f"Checked the release sign-off in `runs/season_log.csv` (decided by the QC reviewer): `{a_status}`."),
                    ("lr_2", f"Checked the zone's declaration: surveyed, every flagged area released, no task open (zone `{zid}`: {arow['zone_state_w']})."),
                    ("lr_3", "Confirmed that public outputs show only `twin.municipal_summary()` (no zone, area or team named)."),
                ]

    # =========================================================================
    # RIGHT COLUMN: PRACTISE THE QC REVIEWER'S CHECK (CHANGES NOTHING)
    # =========================================================================
    with col_hitl:
        with st.container(border=True):
            st.markdown(f"#### 🧑‍⚖️ QC reviewer's check (practice) — step {stage_num}")
            st.caption(
                "Practise the QC reviewer's check on the records on the left. Nothing here changes the twin or the "
                f"saved season: there, sign-offs are by the {ROLES['qc']}. Your choices stay in this browser session."
            )

            val_key = f"{area_id}::stage_{stage_num}"
            existing_val = st.session_state["hitl_pipeline_validations"].get(val_key, {})
            existing_checked = existing_val.get("checked", {}) or {}

            st.markdown("**1. Checklist (practice; tick what you checked):**")
            checked = {}
            for chk_id, chk_label in checklist_items:
                checked[chk_id] = st.checkbox(
                    chk_label,
                    value=bool(existing_checked.get(chk_id, False)),
                    key=f"hitl_chk_{area_id}_{stage_num}_{chk_id}",
                )
            all_checked = all(checked.values())

            st.markdown("**2. Your choice (practice):**")
            action_options = [
                "AGREE",
                "FURTHER_NTS",
                "CLEARANCE",
                "OVERRIDE",
            ]
            action_labels = {
                "AGREE": "Agree with the recorded decision (practice)",
                "FURTHER_NTS": "Send back for another visit (FURTHER_NTS)",
                "CLEARANCE": "Send to clearance (CLEARANCE)",
                "OVERRIDE": "Override the recommendation (in the twin, an override needs a written reason, and the twin "
                            "refuses any release its rule check vetoes)",
            }
            prev_action = existing_val.get("action", "AGREE")
            chosen_action = st.selectbox(
                "QC reviewer's choice (practice):",
                options=action_options,
                index=action_options.index(prev_action) if prev_action in action_options else 0,
                format_func=lambda k: action_labels[k],
                key=f"hitl_act_{area_id}_{stage_num}",
            )

            role_name = st.text_input(
                "Role:",
                value=existing_val.get("role", "QC reviewer"),
                key=f"hitl_off_{area_id}_{stage_num}",
            )
            practice_note = st.text_area(
                "Your note (practice):",
                value=existing_val.get("note", ""),
                height=78,
                key=f"hitl_note_{area_id}_{stage_num}",
            )

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button(
                    "✍️ Save my practice choice",
                    key=f"btn_hitl_advance_{area_id}_{stage_num}",
                    width="stretch",
                ):
                    st.session_state["hitl_pipeline_validations"][val_key] = {
                        "area_id": area_id,
                        "zone_id": zid,
                        "municipality": mname,
                        "week": week,
                        "stage_num": stage_num,
                        "step": st_info["short_title"],
                        "action": chosen_action,
                        "action_label": action_labels[chosen_action],
                        "role": role_name,
                        "note": practice_note,
                        "checked": checked,
                        "all_checked": all_checked,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

            with b_col2:
                if stage_num > 1:
                    st.button(
                        f"◀ Back to step {stage_num - 1}",
                        key=f"btn_hitl_prev_{area_id}_{stage_num}",
                        width="stretch",
                        on_click=_select_stage,
                        args=(stage_num - 1,),
                    )

            # This session's practice choices for the area
            area_vals = [
                v
                for k, v in st.session_state["hitl_pipeline_validations"].items()
                if v.get("area_id") == area_id
            ]
            if area_vals:
                st.markdown("---")
                st.markdown(f"**Your practice choices for `{area_id}` ({len(area_vals)} saved in this browser session):**")
                for v in sorted(area_vals, key=lambda x: x["stage_num"]):
                    st.caption(
                        f"• **Step {v['stage_num']} ({v.get('step', '')}), week {v.get('week', '')}:** {v['action_label']} — "
                        f"*{v.get('role', '')}* ({v['timestamp']})"
                    )
                export = {
                    "note": "Practice choices made in the app's QC reviewer console. They are not sign-offs, and they "
                            "change nothing in the twin or the saved season.",
                    "choices": area_vals,
                }
                st.download_button(
                    label=f"⬇ Export my practice choices for {area_id} (.json; not a sign-off)",
                    data=json.dumps(export, indent=2).encode("utf-8"),
                    file_name=f"practice_choices_{area_id.lower()}.json",
                    mime="application/json",
                    key=f"dl_hitl_json_{area_id}_{stage_num}",
                )


def render_clearance_pipeline_studio(
    root: Path,
    week: int = 52,
    embedded: bool = False,
) -> None:
    """Render the land release steps for an area: the twin's records at each step, and the practice console."""
    if not embedded:
        restricted_header(
            "Land Release Steps for an Area",
            "Follow an area through the twin's six steps (desk study → zone survey → visit → review and sign-off → "
            "clearance → zone declared free). Click a step to see the twin's records for it, and practise the QC "
            "reviewer's check; nothing here changes the twin or the saved season.",
        )
        c_w1, c_w2 = st.columns([1.8, 2.2])
        with c_w1:
            week = st.slider(
                "Select season week to show (1 to 52):",
                min_value=1,
                max_value=52,
                value=52,
                step=1,
                key="standalone_pipeline_week_slider",
            )
        with c_w2:
            st.info(
                "**How to use:** select an area below, then **click one of the six step buttons** to open the twin's "
                "records for that step at the end of the selected week. The console beside them is practice: it "
                "changes nothing in the twin or the saved season."
            )
    else:
        st.markdown("### 🔄 Land release steps for an area (practice view)")
        st.caption(
            "How an area moves through the **twin's six steps**. **Click a step button (`1` to `6`)** to open that "
            "step's records, and practise the QC reviewer's check (nothing here changes the twin or the saved season)."
        )

    bundle = get_twin_bundle(root)
    log_df = load_saved_log(str(root))
    rev_df = load_saved_reviews_public(str(root))
    gate_df = load_gate_table(str(root))
    ord_findings = load_area_ordnance_findings(str(root))
    classified_df = ordnance_library.classify_area_evidence(bundle, ord_findings, gate_df)
    pipe_df = build_area_pipeline_profiles(str(root), week=week)

    # Each area's path at the selected week, from the saved reviews and the log
    rev_w = rev_df[rev_df["week"] <= week].sort_values(["week", "visit"]) if not rev_df.empty else pd.DataFrame()
    decisions_by_area: dict[str, list[str]] = {}
    for rrow in rev_w.to_dict("records"):
        decisions_by_area.setdefault(str(rrow["area_id"]), []).append(f"visit {int(rrow['visit'])}: {rrow['decision']}")

    def _area_label(aid: str) -> str:
        prow = pipe_df.loc[aid]
        steps = list(decisions_by_area.get(aid, []))
        if bool(prow["cleared_by_w"]):
            steps.append(f"clearance: {prow['clearance_note']} (week {int(prow['cleared_week'])})")
        elif int(prow["clearance_tasks_count"]):
            steps.append(f"clearance {prow['clearance_status_w']}")
        elif not steps and int(prow["visits_ordered"]):
            steps.append("visit ordered")
        return (
            f"{aid} ({prow['municipality']}, zone {prow['zone_id']}) — {' -> '.join(steps)} · status: {prow['status_w']}"
        )

    # Filter bar to choose which area to follow
    p_col1, p_col2, p_col3 = st.columns([2.2, 1.4, 1.4])
    active_areas = pipe_df[pipe_df["has_dossier_or_task"]].index.astype(str).tolist()
    with p_col1:
        ordered_active = [a for a in LISTED_FIRST if a in active_areas] + [
            a for a in active_areas if a not in LISTED_FIRST
        ]
        area_key = f"pipeline_area_sel_{'emb' if embedded else 'main'}"
        if ordered_active and st.session_state.get(area_key) not in ordered_active:
            st.session_state[area_key] = ordered_active[0]   # the week changed and the area chosen is not listed in it
        selected_area_id = st.selectbox(
            f"1. Select an area ({len(ordered_active)} with a visit ordered by the end of week {week}):",
            options=ordered_active,
            index=0,
            format_func=_area_label,
            key=area_key,
        )

    if "pipeline_selected_stage" not in st.session_state:
        st.session_state["pipeline_selected_stage"] = 4  # step 4 (review and sign-off) first
    dd_key = f"pipeline_stage_dropdown_{'emb' if embedded else 'main'}"
    if st.session_state.get(dd_key) != int(st.session_state["pipeline_selected_stage"]):
        st.session_state[dd_key] = int(st.session_state["pipeline_selected_stage"])

    with p_col2:
        st.selectbox(
            "2. Step to show:",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda s: f"{PIPELINE_STAGES[s - 1]['icon']} {PIPELINE_STAGES[s - 1]['short_title']}",
            key=dd_key,
            on_change=_stage_from_dropdown,
            args=(dd_key,),
        )

    queue_labels = {
        0: f"All {len(active_areas)} areas with a visit ordered by the end of week {week}",
        3: "Areas with a visit done",
        4: "Areas with a review signed off",
        5: "Areas with a clearance ordered",
        6: "Areas released (cancelled, reduced or cleared)",
    }
    with p_col3:
        stage_queue_filter = st.selectbox(
            "3. Filter the table below:",
            options=list(queue_labels),
            index=0,
            format_func=lambda s: queue_labels[s],
            key=f"pipeline_queue_filter_{'emb' if embedded else 'main'}",
        )

    # Step buttons: clicking one opens that step's records
    st.markdown("**👇 Click a step to see its records and practise the QC reviewer's check:**")
    btn_cols = st.columns(6)
    for idx, st_info in enumerate(PIPELINE_STAGES):
        s_num = st_info["stage_num"]
        is_active = int(st.session_state["pipeline_selected_stage"]) == s_num
        with btn_cols[idx]:
            btn_lbl = f"{'🟢 ' if is_active else ''}{st_info['icon']} {st_info['short_title']}"
            st.button(
                btn_lbl,
                key=f"stage_click_btn_{s_num}_{'emb' if embedded else 'main'}",
                width="stretch",
                on_click=_select_stage,
                args=(s_num,),
            )

    current_stage = int(st.session_state["pipeline_selected_stage"])

    # The two rows: all areas at each step, and the selected area's state at each step
    fig_pipe = plot_pipeline_stage_funnel_and_Area_stepper(
        pipe_df=pipe_df,
        selected_area_id=selected_area_id,
        selected_stage=current_stage,
        week=week,
    )
    st.pyplot(fig_pipe)
    plt.close(fig_pipe)

    if selected_area_id is None:
        st.info(f"No area has a visit ordered by the end of week {week}: choose a later week.")
        return

    sel_z_id = str(pipe_df.loc[selected_area_id, "zone_id"])
    area_revs_w = (
        rev_df[(rev_df["area_id"].astype(str) == str(selected_area_id)) & (rev_df["week"] <= week)].sort_values(["week", "visit"])
        if not rev_df.empty
        else pd.DataFrame()
    )
    if area_revs_w.empty and not rev_df.empty:
        area_revs_w = rev_df[rev_df["area_id"].astype(str) == str(selected_area_id)].sort_values(["week", "visit"])

    sfx = "emb" if embedded else "main"
    nj1, nj2, nj3 = st.columns(3)
    with nj1:
        st.button(
            f"📖 Open Zone {sel_z_id} Story in Operational Log",
            key=f"pipe_jump_log_{selected_area_id}_{week}_{sfx}",
            width="stretch",
            on_click=nav_to_log_zone,
            args=(sel_z_id,),
        )
    with nj2:
        st.button(
            f"📋 Open Week {week} in Weekly Operations",
            key=f"pipe_jump_ops_{selected_area_id}_{week}_{sfx}",
            width="stretch",
            on_click=nav_to_operations_week,
            args=(week,),
        )
    with nj3:
        if not area_revs_w.empty:
            last_r = area_revs_w.iloc[-1]
            rw = int(last_r["week"])
            rv = int(last_r["visit"])
            st.button(
                f"📄 Open Sign-Off Sheet (Week {rw}, Visit {rv}) in Dossier Reviews",
                key=f"pipe_jump_rev_{selected_area_id}_{rw}_{rv}_{sfx}",
                width="stretch",
                on_click=nav_to_dossier_review,
                args=(rw, str(selected_area_id), rv),
            )
        else:
            st.button(
                f"📄 No Dossier Review Yet for {selected_area_id}",
                key=f"pipe_jump_rev_none_{selected_area_id}_{week}_{sfx}",
                width="stretch",
                disabled=True,
            )

    # The selected step's records (left) and the practice console (right)
    _render_stage_paperwork_and_hitl_panel(
        root=root,
        bundle=bundle,
        log_df=log_df,
        rev_df=rev_df,
        gate_df=gate_df,
        ord_findings=ord_findings,
        classified_df=classified_df,
        pipe_df=pipe_df,
        area_id=selected_area_id,
        stage_num=current_stage,
        week=week,
    )

    # The table of all areas with a visit ordered by the week
    with st.expander(
        f"📋 Table: the {len(active_areas)} areas with a visit ordered by the end of week {week}", expanded=False
    ):
        q_df = pipe_df[pipe_df["has_dossier_or_task"]].copy()
        if stage_queue_filter == 3:
            q_df = q_df[q_df["visits_done"] > 0]
        elif stage_queue_filter == 4:
            q_df = q_df[q_df["reviews_count"] > 0]
        elif stage_queue_filter == 5:
            q_df = q_df[q_df["clearance_tasks_count"] > 0]
        elif stage_queue_filter == 6:
            q_df = q_df[q_df["stage_num"] == 6]

        st.dataframe(
            q_df[
                [
                    "area_id",
                    "zone_id",
                    "municipality",
                    "land_use",
                    "subtypes",
                    "stage_label",
                    "visits_done",
                    "reviews_count",
                    "has_resurvey_loop",
                    "latest_review_outcome",
                    "clearance_tasks_count",
                    "cleared_cells",
                    "status_w",
                ]
            ].rename(
                columns={
                    "area_id": "Area ID",
                    "zone_id": "Zone ID",
                    "municipality": "Municipality",
                    "land_use": "Land Use",
                    "subtypes": "What the Twin and the Saved Runs Show (Whole Season)",
                    "stage_label": f"Furthest Step (W{week})",
                    "visits_done": "Visits Done",
                    "reviews_count": "Reviews Signed Off",
                    "has_resurvey_loop": "Sent Back for Another Visit?",
                    "latest_review_outcome": "Latest Sign-Off",
                    "clearance_tasks_count": "Clearance Tasks Ordered",
                    "cleared_cells": "Mine Cells Found and Cleared",
                    "status_w": f"Status (W{week})",
                }
            ),
            width="stretch",
            hide_index=True,
        )
