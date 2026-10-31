"""Cached loaders and shared UI helpers for the read-only application."""
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from twin import config, evaluate, field, forecast, layers, overlays, planner, prioritise, review, season, sitrep

# The review columns a person in the programme would see (no truth columns); zone and area IDs are restricted layers
PROGRAMME_REVIEW_COLS = [
    "week",
    "area_id",
    "zone_id",
    "municipality",
    "visit",
    "recorded",
    "outcome",
    "decision",
    "fallback",
    "confidence",
    "vetoed at sign-off",
    "vetoes",
    "held since",
]


# The dossier reviewer's four outcomes (twin/rules.py OUTCOMES; twin/core.py REVIEW_OUTCOMES)
HUMAN_OUTCOME = {
    "CANCEL": "Cancel (release without clearance)",
    "TECHNICAL_SURVEY": "Technical survey",
    "CLEARANCE": "Clearance",
    "FURTHER_NTS": "Further survey visit",
}
REVIEW_OUTCOMES = list(HUMAN_OUTCOME)

# The twin's four task kinds (twin/core.py TASK_KINDS)
HUMAN_TASK_KIND = {
    "ZONE_SURVEY": "Zone survey",
    "VISIT": "Area visit (non-technical survey)",
    "TS": "Technical survey",
    "CLEARANCE": "Clearance",
}

# Area statuses as the twin records them (twin.status), in display order
HUMAN_AREA_STATUS = {
    "unknown": "unknown",
    "not_suspected": "not suspected (zone surveyed, area not flagged)",
    "suspected": "suspected",
    "confirmed": "confirmed",
    "cancelled": "cancelled",
    "reduced": "reduced",
    "cleared": "cleared",
}

# Sidebar screen labels and cross-screen navigation callbacks (`on_click`)
SCREEN_OVERVIEW = "Command & Planning Overview"
SCREEN_PIPELINE = "🔄 Land Release & Clearance Pipeline"
SCREEN_LIBRARY = "📚 Literature, Standards & Governance Library"
SCREEN_OPS = "1. Weekly Operations"
SCREEN_PLAN = "2. Plan & Completion Forecast"
SCREEN_LOG = "3. Operational Log"
SCREEN_REVIEWS = "4. Dossier Sign-Off Reviews"
SCREEN_SITREPS = "5. Weekly Situation Reports"
SCREEN_PUBLIC = "6. Public Summary"
SCREEN_EVAL = "7. Evaluation (Hidden Truth)"
SCREEN_ABOUT = "8. About & Data Rules"


def nav_to_pipeline(area_id: str | None = None, week: int = 52, stage: int = 4) -> None:
    """Switch the sidebar view to the Land Release & Clearance Pipeline at `week` and `stage` for `area_id`."""
    st.session_state["app_nav_screen"] = SCREEN_PIPELINE
    st.session_state["standalone_pipeline_week_slider"] = int(min(max(int(week), 1), 52))
    st.session_state["pipeline_selected_stage"] = int(min(max(int(stage), 1), 6))
    st.session_state["pipeline_stage_dropdown_main"] = int(min(max(int(stage), 1), 6))
    if area_id:
        st.session_state["pipeline_area_sel_main"] = str(area_id)


def nav_to_log_zone(zone_id: str) -> None:
    """Switch the sidebar view to the Operational Log with `zone_id` selected in the zone story reader."""
    st.session_state["app_nav_screen"] = SCREEN_LOG
    if zone_id:
        st.session_state["log_story_zone_select"] = str(zone_id)


def nav_to_operations_week(week: int) -> None:
    """Switch the sidebar view to Weekly Operations at `week`."""
    st.session_state["app_nav_screen"] = SCREEN_OPS
    st.session_state["ops_week_slider"] = int(min(max(int(week), 1), 52))


def nav_to_dossier_review(week: int, area_id: str, visit: int = 1) -> None:
    """Switch the sidebar view to Dossier Sign-Off Reviews with `(week, area_id, visit)` selected."""
    st.session_state["app_nav_screen"] = SCREEN_REVIEWS
    st.session_state["reviews_target_area_visit"] = (int(week), str(area_id), int(visit))


def nav_to_sitrep_week(week: int) -> None:
    """Switch the sidebar view to Weekly Situation Reports at `week`."""
    st.session_state["app_nav_screen"] = SCREEN_SITREPS
    st.session_state["sitreps_week_slider"] = int(min(max(int(week), 1), 52))



def apply_editorial_style() -> None:
    """Apply clean local CSS so the viewer reads like a warm editorial programme dashboard."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e2dfd7;
            border-radius: 10px;
            padding: 12px 16px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            color: #52514e !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem !important;
            color: #181816 !important;
            font-weight: 700 !important;
        }
        .programme-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #f4efe6;
            border-left: 4px solid #b5651d;
            border-radius: 6px;
            padding: 8px 14px;
            margin-bottom: 0.8rem;
            font-size: 0.84rem;
            color: #3d3a35;
        }
        .public-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #e9f3ed;
            border-left: 4px solid #2d7d53;
            border-radius: 6px;
            padding: 8px 14px;
            margin-bottom: 0.8rem;
            font-size: 0.84rem;
            color: #234734;
        }
        .eval-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #faeae6;
            border-left: 4px solid #c44536;
            border-radius: 6px;
            padding: 8px 14px;
            margin-bottom: 0.8rem;
            font-size: 0.84rem;
            color: #4f231d;
        }
        .signoff-banner {
            background-color: #fff6e5;
            border: 1px solid #e6b86a;
            border-left: 5px solid #d48817;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
            font-weight: 600;
            color: #4a3510;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def check_prerequisites(root: Path) -> tuple[bool, list[str]]:
    """Verify that saved risk models and saved agent reviews exist before any setup."""
    missing = []
    for name in ("gate.csv", "area_scores.csv", "calibration.csv"):
        path = root / "models" / name
        if not path.exists():
            missing.append(f"models/{name}")
    reviews_dir = root / "runs" / "reviews"
    if not reviews_dir.is_dir() or not any(reviews_dir.glob("*_result.json")):
        missing.append("runs/reviews/ (saved review records)")
    return (len(missing) == 0), missing


def restricted_header(title: str, subtitle: str = "") -> None:
    """Standard heading for restricted screens inside the twin."""
    st.markdown(
        '<div class="programme-banner">'
        "<span><strong>Inside the twin (restricted)</strong> · Operational view for programme staff</span>"
        "<span>Includes zones, areas, field teams, community consultation & access feed</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.header(title)
    if subtitle:
        st.markdown(f"<p style='color:#52514e;font-size:1.02rem;margin-top:-0.4rem;'>{subtitle}</p>", unsafe_allow_html=True)


def render_recommendation_box(rec: overlays.Recommendation, title: str = "", model_name: str = "") -> None:
    """Render an overlay's Recommendation contract fields in plain, human-friendly terms."""
    heading = title or f"Decision tool note: {rec.overlay} (week {rec.week})"
    ev_text = ", ".join(str(x) for x in rec.evidence) if rec.evidence else "—"
    unc_text = rec.uncertainty if rec.uncertainty else "—"

    if not rec.ok:
        st.warning(f"**{heading} — {rec.status}:** {rec.reason}")

    with st.expander(f"ℹ️ {heading} — status: {rec.status} (what this tool used & its limits)", expanded=not rec.ok):
        if model_name:
            st.caption(f"Method / model: {model_name}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Status:** `{rec.status}`" + (f" — {rec.reason}" if rec.reason else ""))
            st.markdown(f"**Information relied on:** {ev_text}")
        with c2:
            st.markdown(f"**Uncertainty:** {unc_text}")
            if rec.limits:
                st.markdown("**Key limitations to keep in mind:**")
                for lim in rec.limits:
                    st.markdown(f"- {lim}")
            else:
                st.markdown("**Key limitations:** —")


def humanize_zone_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format twin.zone_table() with plain-English column headers and values for programme staff."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index.name = "Zone"
    if "surveyed_week" in out.columns:
        out["surveyed_week"] = out["surveyed_week"].apply(lambda w: f"Week {int(w)}" if pd.notna(w) and int(w) > 0 else "—")
    if "free_week" in out.columns:
        out["free_week"] = out["free_week"].apply(lambda w: f"Week {int(w)}" if pd.notna(w) and int(w) > 0 else "—")
    if "km2" in out.columns:
        out["km2"] = out["km2"].round(2)
    rename_map = {
        "municipality": "Municipality",
        "state": "State at Season End (Week 52)",
        "areas": "Total Areas",
        "km2": "Size (km²)",
        "access": "Road Distance",
        "access_status": "Access Feed",
        "access_as_of": "Access Checked (Week)",
        "consultation": "Community Consultation",
        "surveyed_week": "Zone Survey Finished",
        "free_week": "Declared Free",
        "reopened": "Times Reopened",
        "teams": "Assigned Teams",
        "unknown": "Unknown Areas",
        "suspected": "Suspected Areas",
        "confirmed": "Confirmed Areas",
        "cancelled": "Cancelled Areas",
        "reduced": "Reduced Areas",
        "cleared": "Cleared Areas",
    }
    return out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})


def humanize_reviews_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format dossier reviews with plain-English labels while keeping all programme information."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "outcome" in out.columns:
        out["outcome"] = out["outcome"].map(lambda x: HUMAN_OUTCOME.get(str(x), str(x)))
    if "decision" in out.columns:
        out["decision"] = out["decision"].map(lambda x: HUMAN_OUTCOME.get(str(x), str(x)))
    if "confidence" in out.columns:
        out["confidence"] = out["confidence"].apply(
            lambda v: str(v).capitalize() if pd.notna(v) and str(v) != "" else "—"
        )
    if "vetoed at sign-off" in out.columns:
        out["vetoed at sign-off"] = out["vetoed at sign-off"].map(
            lambda v: "Vetoed by the rule check" if bool(v) else "Passed"
        )
    if "fallback" in out.columns:
        out["fallback"] = out["fallback"].map(
            lambda v: "Insufficient evidence (fallback)" if bool(v) else "Standard review"
        )
    rename_map = {
        "week": "Week",
        "area_id": "Area",
        "zone_id": "Zone",
        "municipality": "Municipality",
        "visit": "Visit #",
        "recorded": "Accident Recorded Inside (Desk Study)",
        "outcome": "Reviewer Recommendation",
        "confidence": "Reviewer Confidence",
        "decision": "Final Sign-Off Decision",
        "vetoed at sign-off": "Rule Check",
        "fallback": "Review Mode",
        "model": "Model Used",
        "summary": "Brief Summary",
    }
    return out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})


def humanize_log_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format season_log.csv with clean, human-readable column titles."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename_map = {
        "n": "Entry #",
        "week": "Week",
        "entry": "Record Type",
        "kind": "Category",
        "subject": "Subject",
        "decision": "Action / Decision",
        "before": "Before",
        "after": "After",
        "decided_by": "Decided By (Role and Sign-Off Policy)",
        "recommended_by": "Recommended By",
        "overlays": "Tools Consulted",
        "evidence": "Evidence Cited",
        "rule_check": "Rule Check",
        "source": "Reported By",
        "reason": "Details / Notes",
    }
    return out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})


@st.cache_data
def load_run_info(root_str: str) -> dict:
    """Read runs/run_info.json without touching restricted or truth data."""
    path = Path(root_str) / "runs" / "run_info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_gate_table(root_str: str) -> pd.DataFrame:
    """Read models/gate.csv and add human-readable model names from prioritise.RANKINGS."""
    path = Path(root_str) / "models" / "gate.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"municipality_code": str})
    if "chosen" in df.columns:
        df["chosen_label"] = df["chosen"].map(lambda k: prioritise.RANKINGS.get(k, str(k)))
    if "chosen by ranking alone" in df.columns:
        df["ranking_alone_label"] = df["chosen by ranking alone"].map(
            lambda k: prioritise.RANKINGS.get(k, str(k))
        )
    return df


@st.cache_data
def load_saved_log(root_str: str) -> pd.DataFrame:
    """Read runs/season_log.csv directly (fast, no rebuild needed)."""
    path = Path(root_str) / "runs" / "season_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.fillna("")


@st.cache_data
def load_saved_reviews_public(root_str: str) -> pd.DataFrame:
    """Read runs/season_reviews.csv keeping only programme-visible columns."""
    root = Path(root_str)
    path = root / "runs" / "season_reviews.csv"
    if not path.exists():
        return pd.DataFrame()
    full_cols = pd.read_csv(path, nrows=0).columns.tolist()
    keep = [c for c in PROGRAMME_REVIEW_COLS if c in full_cols]
    df = pd.read_csv(path, usecols=keep)
    models = []
    efforts = []
    summaries = []
    for _, row in df.iterrows():
        key = f"w{int(row['week']):02d}_{row['area_id']}_v{int(row['visit'])}"
        res_path = root / "runs" / "reviews" / f"{key}_result.json"
        if res_path.exists():
            rec = json.loads(res_path.read_text(encoding="utf-8"))
            models.append(rec.get("model", ""))
            efforts.append(rec.get("reasoning_effort", "") or "")
            summaries.append(rec.get("summary", ""))
        else:
            models.append("")
            efforts.append("")
            summaries.append("")
    df["model"] = models
    df["reasoning_effort"] = efforts
    df["summary"] = summaries
    return df


@st.cache_data
def load_review_detail(root_str: str, week: int, area_id: str, visit: int) -> dict:
    """Read one dossier review's sign-off sheet (.md) and result record (.json)."""
    root = Path(root_str)
    key = f"w{int(week):02d}_{area_id}_v{int(visit)}"
    md_path = root / "runs" / "reviews" / f"{key}_signoff.md"
    json_path = root / "runs" / "reviews" / f"{key}_result.json"
    signoff_md = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    record = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    return {"key": key, "signoff_md": signoff_md, "record": record}


@st.cache_data
def load_saved_sitreps(root_str: str) -> dict[int, dict]:
    """Read all saved weekly situation reports in runs/sitreps/ and run the coded figure check on each."""
    root = Path(root_str)
    sitreps_dir = root / "runs" / "sitreps"
    out: dict[int, dict] = {}
    if not sitreps_dir.is_dir():
        return out
    for week in range(1, 53):
        path = sitreps_dir / f"w{week:02d}_sitrep.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        draft = rec.get("draft", "")
        facts = rec.get("facts", {})
        model = rec.get("model", "model")
        kept, removed, unmatched = sitrep.figure_check(draft, facts)
        sentences = len([s for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()])
        insufficient = bool(sentences and len(removed) > sentences / 2)
        status = overlays.INSUFFICIENT if insufficient else "ok"
        reason = (
            f"{len(removed)} of {sentences} sentences had figures not in the facts"
            if insufficient
            else ""
        )
        recommendation = overlays.Recommendation(
            overlay=sitrep.SituationReport.name,
            week=week,
            status=status,
            answer={
                "text": kept,
                "draft": draft,
                "origin": f"saved draft ({model})",
                "removed": removed,
                "unmatched": unmatched,
                "facts": facts,
            },
            evidence=("the twin's facts of the week",),
            uncertainty=f"{len(removed)} sentence(s) removed by the figure check",
            limits=sitrep.SituationReport.limits,
            reason=reason,
        )
        out[week] = {
            "record": rec,
            "recommendation": recommendation,
            "sentences": sentences,
        }
    return out


class RecordingRun(season.SeasonRun):
    """The same season as `season.SeasonRun`; after each week it copies the twin's own area statuses and zone
    states, and the risk overlay's record in force that week, so that the screens show any week as the twin
    recorded it instead of rebuilding it. It only reads: the season, its log and its decisions are unchanged."""

    def end_week(self, week):
        out = super().end_week(week)
        if not hasattr(self, "snapshots"):
            self.snapshots = {}
        self.snapshots[week] = {
            "area_status": pd.Series(self.twin.status).copy(),
            "zone_state": self.twin.zone_table()["state"].copy(),
            "risk": self.risk_rec,   # the risk overlay's record in force this week (chances, status, limits)
        }
        return out


class _BaselineChances:
    """A stand-in for the risk models that only answers the baseline's chances, computed once: the programme
    screens need no model object, and the models (fitted on RELand's labels) stay with the evaluation."""

    def __init__(self, baseline: pd.DataFrame):
        self._baseline = baseline

    def chances(self, name: str = "baseline") -> pd.DataFrame:
        if name != "baseline":
            raise KeyError(f"only the baseline's chances are available outside the Evaluation screen, not {name!r}")
        return self._baseline.copy()


@st.cache_resource(show_spinner=False)
def _rebuild_twin_and_season(root_str: str) -> tuple[dict, dict]:
    """Build the twin, rebuild the agent's season from saved reviews/reports, and compute the first plan and forecast.

    Follows the sequence the build brief sets, once per server process (`st.cache_resource`). It returns two parts:
    the programme bundle, made of copies of tables and plain values with no truth in it, which every screen reads
    through `get_twin_bundle`; and the evaluation objects (the setup with the simulation's truth, and the season's
    result with its measures against the truth), which only the Evaluation screen reads, through
    `get_evaluation_objects`.
    """
    root = Path(root_str)
    s = evaluate.setup(False)
    twin = s.twin

    # The agent's season, rebuilt from the saved reviews and reports (no key)
    twin.reset()
    month_starts = tuple(twin.calendar.first_week(m) for m in range(1, 13))
    reviewer = review.AgentReviewer(root / "runs" / "reviews", client=None, label="main season")
    reports = sitrep.SituationReport(root / "runs" / "sitreps", client=None)
    run = RecordingRun(
        twin,
        field.Field(twin, s.truth, seed=0),
        s.models,
        s.factors,
        reviewer=reviewer,
        checker=s.checker,
        sitrep=reports,
        forecast_weeks=month_starts,
    )
    try:
        result = run.run()
        missing_review_error = None
        zones = twin.zone_table().copy()
        public = twin.municipal_summary().copy()
        snaps = getattr(run, "snapshots", {})
        weekly_area_status = {w: v["area_status"] for w, v in snaps.items()}
        weekly_zone_state = {w: v["zone_state"] for w, v in snaps.items()}
        weekly_risk = {w: v["risk"] for w, v in snaps.items()}
        area_flag_end = pd.Series(twin.flag).copy()   # the zone survey's flag for each area, at the season's end
        plan_recs = list(run.plan_recs)
        reviewer_stats = {
            "loaded": int(reviewer.loaded),
            "other_dossier": int(reviewer.other_dossier),
            "other_method": int(reviewer.other_method),
        }
        sitrep_stats = {
            "loaded": int(reports.loaded),
            "coded": int(reports.coded),
        }
    except review.MissingReview as exc:
        result = None
        missing_review_error = str(exc)
        zones = None
        public = None
        weekly_area_status, weekly_zone_state, weekly_risk = {}, {}, {}
        area_flag_end = pd.Series(dtype=object)
        plan_recs = []
        reviewer_stats = {}
        sitrep_stats = {}

    # The first plan and its forecast, as notebook sections 6.3 and 6.4 compute them
    twin.reset()
    risk = prioritise.RiskOverlay(s.models).recommend(twin.view(), 1)
    plan = planner.SeasonPlanner(s.factors, method="milp").recommend(twin.view(), 1, risk=risk)
    rule = planner.SeasonPlanner(s.factors, method="rule").recommend(twin.view(), 1, risk=risk)
    fc = forecast.CompletionForecaster(s.factors).recommend(twin.view(), 1, plan=plan.answer, risk=risk)

    # Extract plain copies of metadata from the reset twin so widgets never touch `twin`
    calendar_labels = {w: twin.calendar.label(w) for w in range(1, twin.season.weeks + 1)}
    month_of_week = dict(twin.calendar.month_of)
    weeks_of_month = {m: tuple(ws) for m, ws in twin.calendar.weeks_of.items()}
    zone_areas = {z: tuple(a_ids) for z, a_ids in twin.zone_areas.items()}
    neighbours = {a: tuple(ns) for a, ns in twin.neighbours.items()}
    zones_meta = twin.zones.copy()
    areas_meta = twin.areas[
        [
            "municipality_code",
            "municipality",
            "zone_id",
            "number",
            "cells",
            "km2",
            "main_land_use",
            "event_inside",
        ]
    ].copy()
    cells_meta = twin.cells[["municipality_code", "area_id", "zone_id", "row", "col"]].copy()
    victims_by_municipality = twin.municipal_summary()["recorded victims (all years)"].copy()
    # The open demining layer (datos.gov.co jbqm-s5tn, as P2 summed it), for the pilot's 15 municipalities only
    pilot_codes = set(config.NAME_OF)
    activity_df = twin.activity[twin.activity["municipality_code"].isin(pilot_codes)].copy()
    team_counts = twin.teams["kind"].value_counts().to_dict()
    criteria_text = twin.criteria.as_text()
    layer_df = layers.layer_table().copy()

    # Copy season result tables without truth columns for programme screens
    if result is not None:
        log_df = result.log.copy()
        tasks_df = result.tasks.copy()
        keep_rev = [c for c in PROGRAMME_REVIEW_COLS if c in result.reviews.columns]
        reviews_public_df = result.reviews[keep_rev].copy()
        zone_state_df = result.zone_state.copy()
        plans_list = copy.deepcopy(result.plans)
        forecasts_list = list(result.forecasts)
        sitreps_list = list(result.sitreps)
        replans_list = list(result.replans)
        insufficient_list = list(result.insufficient)
        week_days_dict = copy.deepcopy(result.week_days)
        idle_dict = copy.deepcopy(result.idle)
        # the injected fault is the simulation's (it never reaches the twin), so it stays out
        checks_df = result.checks.drop(columns=["features", "fault"], errors="ignore").copy()
    else:
        log_df = pd.DataFrame()
        tasks_df = pd.DataFrame()
        reviews_public_df = pd.DataFrame()
        zone_state_df = pd.DataFrame()
        plans_list = []
        forecasts_list = []
        sitreps_list = []
        replans_list = []
        insufficient_list = []
        week_days_dict = {}
        idle_dict = {}
        checks_df = pd.DataFrame()

    # A truth-free stand-in for the setup: copies of the twin's tables and settings that the screens read, and the
    # baseline's chances; no `s.truth`, no models, no field
    public_twin = SimpleNamespace(
        activity=twin.activity.copy(),
        cells=twin.cells.copy(),     # RELand's 63 features with the grid keys; no coordinates, no labels
        areas=twin.areas.drop(columns=["cell_list"], errors="ignore").copy(),
        zones=twin.zones.copy(),
        people=twin.people.copy(),
        season=twin.season,          # settings objects, read only
        calendar=twin.calendar,
        criteria=twin.criteria,
    )
    setup_public = SimpleNamespace(
        twin=public_twin,
        models=_BaselineChances(s.models.chances("baseline").copy()),
        checks=s.checks.copy() if hasattr(s.checks, "copy") else s.checks,   # the data files' SHA-256 checks
    )

    evaluation = {"setup": s, "result": result}
    bundle = {
        "setup": setup_public,
        "missing_review_error": missing_review_error,
        "zones": zones,
        "public": public,
        "risk": risk,
        "plan": plan,
        "rule": rule,
        "fc": fc,
        "weekly_area_status": weekly_area_status,
        "weekly_zone_state": weekly_zone_state,
        "weekly_risk": weekly_risk,
        "area_flag_end": area_flag_end,
        "activity_df": activity_df,
        "team_counts": team_counts,
        "plan_recs": plan_recs,
        "reviewer_stats": reviewer_stats,
        "sitrep_stats": sitrep_stats,
        "calendar_labels": calendar_labels,
        "month_of_week": month_of_week,
        "weeks_of_month": weeks_of_month,
        "zone_areas": zone_areas,
        "neighbours": neighbours,
        "zones_meta": zones_meta,
        "areas_meta": areas_meta,
        "cells_meta": cells_meta,
        "victims_by_municipality": victims_by_municipality,
        "criteria_text": criteria_text,
        "layer_df": layer_df,
        "log_df": log_df,
        "tasks_df": tasks_df,
        "reviews_public_df": reviews_public_df,
        "zone_state_df": zone_state_df,
        "plans": plans_list,
        "forecasts": forecasts_list,
        "sitreps": sitreps_list,
        "replans": replans_list,
        "insufficient": insufficient_list,
        "week_days": week_days_dict,
        "idle": idle_dict,
        "checks_df": checks_df,
        "month_names": list(config.MONTHS),
    }
    return bundle, evaluation


def get_twin_bundle(root: Path) -> dict:
    """Load the cached twin and rebuilt season, showing a plain note while it runs the first time."""
    root_str = str(root)
    if not st.session_state.get("_twin_bundle_ready", False):
        placeholder = st.empty()
        placeholder.info(
            "**Rebuilding the 52-week operational season from the saved reviews and weekly reports** "
            "(loads the risk models, prepares the field report checker, replays the 52 weeks of decisions "
            "and sign-offs, and computes the season's initial plan and completion forecast). "
            "This runs once on first load and takes about two minutes."
        )
        bundle, _ = _rebuild_twin_and_season(root_str)
        st.session_state["_twin_bundle_ready"] = True
        placeholder.empty()
    else:
        bundle, _ = _rebuild_twin_and_season(root_str)

    if bundle.get("missing_review_error"):
        st.warning(
            "**Saved season note:** "
            f"{bundle['missing_review_error']} Showing the saved files directly."
        )
    return bundle


def get_evaluation_objects(root: Path) -> dict:
    """For the Evaluation screen only: the setup with the simulation's truth (`s.truth`, `s.models`) and the rebuilt
    season's result (`result.mined`, `result.base_mined`, `result.metrics`). Call `get_twin_bundle` first."""
    _, evaluation = _rebuild_twin_and_season(str(root))
    return evaluation


# ---- The evidence the saved runs hold about each area (business level; no device types) ----------------------
@st.cache_data(show_spinner=False)
def load_area_ordnance_findings(root_str: str) -> dict[str, dict]:
    """What the saved runs show about each area: the clearance result in `runs/season_log.csv` (the number of mine
    cells the log says clearance found) and every item of the reviewer's evidence registers in
    `runs/reviews/*_result.json` (the statement as the agent quoted it, its evidence class and the source's
    reliability class). No device type is inferred. The hidden truth is not read.
    """
    root = Path(root_str)
    findings: dict[str, dict] = {}

    def _ensure(aid: str) -> dict:
        if aid not in findings:
            findings[aid] = {"area_id": aid, "cleared_cells": 0, "cleared_week": 0, "clearance_note": "",
                             "tags": set(), "quotes": []}
        return findings[aid]

    # 1. Clearance results from the log
    log_path = root / "runs" / "season_log.csv"
    if log_path.exists():
        log_df = pd.read_csv(log_path).fillna("")
        clr_rows = log_df[(log_df["kind"] == "area") & (log_df["decision"] == "CLEARED")]
        for _, r in clr_rows.iterrows():
            aid = str(r["subject"]).strip()
            reason = str(r.get("reason", "")).strip()
            rec = _ensure(aid)
            rec["cleared_week"] = int(r["week"])
            rec["clearance_note"] = reason
            m = re.search(r"(\d+)\s+mine\s+cell", reason)
            if m:
                rec["cleared_cells"] = int(m.group(1))
                if rec["cleared_cells"] > 0:
                    rec["tags"].add("mines_found")

    # 2. The reviewer's evidence registers
    rev_dir = root / "runs" / "reviews"
    if rev_dir.is_dir():
        for jpath in sorted(rev_dir.glob("*_result.json")):
            try:
                data = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            aid = str(data.get("area", "")).strip()
            if not aid:
                continue
            wk = int(data.get("week", 0) or 0)
            vis = int(data.get("visit", 1) or 1)
            for item_id, entry in (data.get("register") or {}).items():
                quote = str(entry.get("quote", "")).strip()
                if not quote:
                    continue
                ec = str(entry.get("evidence_class", "")).strip()
                rc = int(entry.get("reliability_class", 7) or 7)
                rec = _ensure(aid)
                rec["tags"].add(ec)
                if not any(q["quote"] == quote for q in rec["quotes"]):   # each statement once per area
                    rec["quotes"].append({
                        "week": wk, "visit": vis, "item_id": str(item_id), "tag": ec,
                        "tag_label": f"{ec.replace('_', ' ')}, reliability class {rc}",
                        "evidence_class": ec, "reliability_class": rc, "quote": quote,
                    })

    return {aid: {**info, "tags": sorted(info["tags"])} for aid, info in findings.items()}
