"""Screen 7: Evaluation (The simulation's hidden truth, for evaluation only)."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch

from twin import config, evaluate, prioritise, scenarios
import grid_map
from loaders import get_evaluation_objects, get_twin_bundle, load_run_info, restricted_header

MEASURES = [
    "mine areas found",
    "mine areas cleared",
    "zones declared free",
    "km² in zones declared free",
    "released with mines (critical)",
    "mine areas not flagged, in zones declared free (critical)",
    "clearances on land without mines",
    "share of clearance days on land without mines",
    "technical surveys on land without mines",
    "team-days with no task",
    "plans approved",
]

BETTER = ["mine areas found", "mine areas cleared", "zones declared free"]


@st.cache_data
def _load_replay_records(root_str: str) -> tuple[list[dict], list[dict]]:
    """Load saved replay JSON files from runs/replay/ without running any simulation."""
    replay_dir = Path(root_str) / "runs" / "replay"
    if not replay_dir.is_dir():
        return [], []
    all_records = []
    for path in sorted(replay_dir.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        all_records.append(rec)

    policy_order = {name: i for i, name in enumerate(evaluate.POLICIES)}
    scenario_order = {name: i for i, name in enumerate(evaluate.SCENARIO_NAMES)}

    policy_records = [r for r in all_records if r.get("scenario", "none") == "none"]
    policy_records.sort(key=lambda r: (policy_order.get(r["policy"], 99), int(r["seed"])))

    scenario_records = [r for r in all_records if r.get("scenario", "none") != "none"]
    scenario_records.sort(key=lambda r: (scenario_order.get(r["scenario"], 99), int(r["seed"])))

    return policy_records, scenario_records


@st.cache_data
def _load_full_season_reviews(root_str: str) -> pd.DataFrame:
    """Read runs/season_reviews.csv including mined and reference columns (Evaluation screen only)."""
    path = Path(root_str) / "runs" / "season_reviews.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def _load_review_json_stats(root_str: str) -> tuple[pd.DataFrame, list[dict]]:
    """Aggregate token, timing, and safeguard statistics across runs/reviews/*_result.json."""
    reviews_dir = Path(root_str) / "runs" / "reviews"
    if not reviews_dir.is_dir():
        return pd.DataFrame(), []
    rows = []
    flagged = []
    for path in sorted(reviews_dir.glob("*_result.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        stats = rec.get("stats", {})
        usage = rec.get("usage", {})
        rows.append(
            {
                "area": rec.get("area", ""),
                "week": rec.get("week", 0),
                "visit": rec.get("visit", 0),
                "model": rec.get("model", ""),
                "model calls": stats.get("model_calls", 0),
                "tool calls": stats.get("tool_calls", 0),
                "seconds": stats.get("seconds", 0.0),
                "input tokens": usage.get("input_tokens", 0),
                "cached input tokens": usage.get("cached_tokens", 0),
                "output tokens": usage.get("output_tokens", 0),
                "cost (USD)": rec.get("cost_usd") or 0.0,
                "vetoes": len(rec.get("vetoes", [])),
                "quote failures": len(rec.get("quote_failures", [])),
                "tool errors": len(rec.get("tool_errors", [])),
                "sentences removed": len(rec.get("handling_blocked", [])),
                "fallback": bool(rec.get("fallback", False)),
                "instruction lines found": len(rec.get("instruction_lines_found", [])),
            }
        )
        if rec.get("instruction_lines_found"):
            flagged.append(rec)
    return pd.DataFrame(rows), flagged


def _coverage_summary(records: list[dict], week: int) -> pd.DataFrame:
    """Compute the forecast interval coverage summary table as in notebook section 9.4."""
    coverage = evaluate.coverage_table(records, week)
    if coverage.empty:
        return pd.DataFrame()
    policy_names = list(evaluate.POLICIES)
    cov = coverage.groupby("policy").agg(
        **{
            "seasons": ("seed", "size"),
            "zones with a P90 week": ("zones with a P90 week", "sum"),
            "free by P90": ("free by P90", "sum"),
            "zones with a P50 week": ("zones with a P50 week", "sum"),
            "free by P50": ("free by P50", "sum"),
            "mean Brier score": ("brier score", "mean"),
        }
    )
    cov["share free by P90"] = cov["free by P90"] / cov["zones with a P90 week"]
    cov["share free by P50"] = cov["free by P50"] / cov["zones with a P50 week"]
    for total in ("zones free", "mine areas cleared"):
        actual = coverage[f"{total}: actual"]
        low = coverage[f"{total}: P10"]
        high = coverage[f"{total}: P90"]
        cov[f"{total}: seasons inside P10–P90"] = (
            ((actual >= low) & (actual <= high)).groupby(coverage["policy"]).sum()
        )
        cov[f"{total}: seasons below P10"] = (actual < low).groupby(coverage["policy"]).sum()
        cov[f"{total}: seasons above P90"] = (actual > high).groupby(coverage["policy"]).sum()
    for total in ("zones free", "mine areas cleared"):
        cov[f"{total} outside the forecast's zones (mean)"] = coverage.groupby("policy")[
            f"{total} outside the forecast's zones"
        ].mean()
    cov = cov.reindex([p for p in policy_names if p in cov.index])

    t = cov.T.astype(object)
    for row in t.index:
        digits = 2 if row.startswith(("share", "mean Brier")) or row.endswith("(mean)") else 0
        t.loc[row] = ["–" if pd.isna(v) else f"{v:.{digits}f}" for v in cov[row]]
    return t


POLICY_COLOURS = {
    "main policy": "#1f4e78",
    "rule-of-thumb plan": "#d97706",
    "risk by nearness to past accidents": "#64748b",
    "technical survey first": "#0d6e52",
    "no learning in the season": "#9333ea",
}

SCENARIO_COLOURS = {
    "funding shock": "#dc2626",
    "zone closure": "#d97706",
    "new contamination": "#7c3aed",
    "stale or corrupted feed": "#2563eb",
    "missing item": "#0d6e52",
}

OUTCOME_WORDS = {
    "CANCEL": "cancellation",
    "TECHNICAL_SURVEY": "technical survey",
    "CLEARANCE": "clearance",
    "FURTHER_NTS": "further survey",
}


def _weeks_text(weeks) -> str:
    weeks = sorted(int(w) for w in weeks)
    if not weeks:
        return ""
    return f"week {weeks[0]}" if len(weeks) == 1 else f"weeks {weeks[0]} to {weeks[-1]}"


def _range_text(values) -> str:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if not len(v):
        return "–"
    return f"{v.min():g}" if v.min() == v.max() else f"{v.min():g} to {v.max():g}"


def _weeks_count_text(values) -> str:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if not len(v):
        return "–"
    return f"{_range_text(v)} week" + ("" if v.min() == v.max() == 1 else "s")


def _scenario_doctrine(bundle: dict) -> dict:
    """Each injected scenario: its grounding, what is injected and what the twin should do (the notebook's section 10
    table, `twin/scenarios.py`, and the project's risk register, `docs/risk_register.md`). The settings are read from
    the scenario classes and the twin's calendar, so the texts follow the code."""
    weeks_of = bundle["weeks_of_month"]
    month_of = bundle["month_of_week"]
    month_names = bundle["month_names"]
    settings = bundle["setup"].twin.season
    teams = bundle["team_counts"]
    n_clear = int(teams.get("clearance", 0)) + int(teams.get("ts", 0))
    n_survey = int(teams.get("survey", 0))

    fs = scenarios.FundingShock()
    fs_start = weeks_of[fs.start_month][0]
    fs_back = weeks_of[fs.start_month + fs.months][0]
    fs_lost = fs.clearance_lost + fs.survey_lost
    zc = scenarios.ZoneClosure()
    nc = scenarios.NewContamination()
    ff = scenarios.FeedFaults()
    fault_weeks: dict[str, list[int]] = {}
    for w, fault in sorted(ff.FAULTS.items()):
        if w in ff.corrupt_weeks:
            fault_weeks.setdefault(fault, []).append(w)
    inflated = fault_weeks.get("days inflated", [])
    inflated_months = sorted({month_of[w] for w in inflated})
    months_text = " and ".join(month_names[m - 1] for m in inflated_months)
    if inflated_months and all(m in settings.rainy_months for m in inflated_months):
        months_text += ", both rainy months" if len(inflated_months) == 2 else ", rainy months"
    mi = scenarios.MissingItem()

    return {
        "funding shock": {
            "title": "1. Funding shock (`twin.scenarios.FundingShock`)",
            "grounding": (
                "As HALO in 2025, which lost \"up to two-thirds of clearance teams and half of NTS teams\" (Mine Action "
                "Review, 2025, on Colombia, quoted in `twin/scenarios.py`); the shock's size and length are a setting "
                "from the project's domain expert (project interview, question 35)."
            ),
            "injection": (
                f"From {month_names[fs.start_month - 1]} (week {fs_start}), {fs.clearance_lost} of {n_clear} clearance "
                f"teams and {fs.survey_lost} of {n_survey} survey teams are stood down for {fs.months} months; their "
                f"operator expects them back in {month_names[fs.start_month + fs.months - 1]} (week {fs_back})."
            ),
            "should_do": (
                "Pause their tasks and plan again the same week, with the teams expected back: the operations officer "
                f"pauses the stood-down teams' tasks, and the integer programme plans with the {n_clear + n_survey - fs_lost} "
                f"teams still active, counting the {fs_lost} stood-down teams from week {fs_back}, when their operator "
                "expects them back (`twin/season.py`, `twin/planner.py`)."
            ),
        },
        "zone closure": {
            "title": "2. Zone closure in mid-season (`twin.scenarios.ZoneClosure`)",
            "grounding": (
                "A setting from the project interview (question 36): zones close on the access and security feed, a "
                f"simulated restricted layer, where each zone has a 1-in-{round(1 / settings.closure_chance)} chance a "
                f"month of closing for {settings.closure_months[0]} to {settings.closure_months[1]} months "
                "(`twin/config.py`). Here one zone is closed on purpose."
            ),
            "injection": (
                f"In week {zc.week} ({month_names[month_of[zc.week] - 1]}), the zone in progress with the most open "
                f"tasks closes on the access feed for {zc.weeks} weeks."
            ),
            "should_do": (
                "Pause its tasks, move its teams, plan again: the operations officer pauses the zone's tasks, the "
                "integer programme plans again the same week, and the teams take their next tasks elsewhere "
                "(`twin/season.py`)."
            ),
        },
        "new contamination": {
            "title": "3. New contamination in a zone declared free (`twin.scenarios.NewContamination`)",
            "grounding": (
                "The project's risk register, Me2: an area with mines released with its zone (`docs/risk_register.md`). "
                "The scenario tests the control the register names: a zone reopened when new contamination is reported."
            ),
            "injection": (
                "Community leaders report an accident in one of the areas of the season's first zone declared free, "
                f"{nc.delay} weeks after its declaration, through the municipal liaison."
            ),
            "should_do": (
                "Reopen the zone, visit and clear the area first: the national authority reopens the zone in the week "
                "of the report, and the area goes back for a visit (`twin/core.py`, `reopen_zone`); the visit's review "
                "decides what follows, and the zone is declared free again once its flagged areas are resolved and no "
                "task is open."
            ),
        },
        "stale or corrupted feed": {
            "title": "4. Stale or corrupted feed (`twin.scenarios.FeedFaults`)",
            "grounding": (
                "P1's insight, check data as it arrives (`twin/checker.py`), and the project's risk register: Ma2, "
                "stale or missing data, and Me6, wrong team reports get through (`docs/risk_register.md`)."
            ),
            "injection": (
                f"One survey team's reports are lost in {_weeks_text(ff.stale_weeks)} and arrive late, in week "
                f"{ff.stale_weeks[-1] + 1}; one clearance team's reports are corrupted in {_weeks_text(ff.corrupt_weeks)}: "
                f"days that add up to more than a week ({_weeks_text(fault_weeks.get('impossible days', []))}), a "
                f"duplicate ({_weeks_text(fault_weeks.get('duplicate', []))}), a wrong task "
                f"({_weeks_text(fault_weeks.get('wrong task', []))}), and inflated days in {_weeks_text(inflated)} "
                f"({months_text})."
            ),
            "should_do": (
                "Flag the silent team as stale and query it; hold or query the bad reports: the twin's staleness check "
                "flags a team with no report for longer than the season allows, and the operations officer queries it; "
                "the report checker's coded checks hold an impossible, duplicate or wrong-task report; and the isolation "
                "forest, or the check on three weeks in a row with no day lost to rain or wind in a rainy month, queries "
                "an unusual one (`twin/core.py`, `twin/checker.py`)."
            ),
        },
        "missing item": {
            "title": "5. Dossier with a missing item (`twin.scenarios.MissingItem`)",
            "grounding": (
                "The twin's own completeness check (`twin/core.py`, `dossier_complete`), and the project's risk "
                "register, Ma2, stale or missing data (`docs/risk_register.md`). Without the desk study's record, rule "
                "R6 cannot see the recorded accident."
            ),
            "injection": (
                f"From week {mi.start_week}: (a) a visit report with two or more items loses its item of direct evidence "
                "in transmission; (b) the desk study's record of a past accident fails to load into the dossier of "
                "another area where one is recorded."
            ),
            "should_do": (
                "Hold both dossiers until the items arrive; never release on the incomplete one: the report checker "
                "accepts the visit report with a flag (it declares more items than arrived), and the twin's completeness "
                "check holds both dossiers, so that neither is reviewed on incomplete evidence (`twin/checker.py`, "
                "`twin/season.py`)."
            ),
        },
    }


def _scenario_results(name: str, part: pd.DataFrame) -> str:
    """What the twin did in the replay's seasons with one scenario, computed from their measures (`scenario: ...`
    columns of the saved records; notebook sections 10 and 11.1)."""
    n = len(part)
    if not n:
        return ""

    def col(c):
        return part[c] if c in part.columns else pd.Series(np.nan, index=part.index)

    def num(c):
        return pd.to_numeric(col(c), errors="coerce")

    def flag(c):
        return col(c).astype("boolean").fillna(False)

    def outcomes(series):
        counts = series.dropna().value_counts()
        return ", ".join(f"{OUTCOME_WORDS.get(k, k)} in {v}" for k, v in counts.items()) or "–"

    if name == "funding shock":
        same = int((num("scenario: weeks to a new plan") == 0).sum())
        return (
            f"In {same} of {n} seasons, the stood-down teams' tasks were paused "
            f"({_range_text(col('scenario: tasks paused'))} tasks) and a new plan was approved in the week of the "
            f"operator's notice (week {_range_text(col('scenario: noticed in week'))})."
        )
    if name == "zone closure":
        same = int((num("scenario: weeks to a new plan") == 0).sum())
        return (
            f"In {same} of {n} seasons, the closed zone's tasks were paused "
            f"({_range_text(col('scenario: tasks paused'))} tasks) and a new plan was approved in the week it closed "
            f"(week {_range_text(col('scenario: noticed in week'))}); "
            f"{_range_text(col('scenario: orders elsewhere in the week'))} tasks were ordered elsewhere that week."
        )
    if name == "new contamination":
        report_week = pd.to_numeric(col("scenario: incident").astype(str).str.extract(r"in week (\d+)")[0],
                                    errors="coerce")
        visit = num("scenario: weeks to the visit")
        clear = num("scenario: weeks to clearance")
        same = int((num("scenario: noticed in week") == report_week).sum())
        free_same = int((num("scenario: zone declared free again in week") == report_week + clear).sum())
        return (
            f"The zone was reopened in the week of the report in {same} of {n} seasons. The area was visited within "
            f"4 weeks of the report in {int((visit <= 4).sum())} ({_range_text(visit)} weeks after it) and cleared in "
            f"{int(flag('scenario: area cleared').sum())} ({_range_text(clear)} weeks after it); the zone was declared "
            f"free again in the week the area's clearance finished in {free_same}."
        )
    if name == "stale or corrupted feed":
        delay = num("scenario: weeks from the first lost report to the flag").dropna().astype(int)
        delay = delay.value_counts().sort_index()
        flag_parts = [
            f"{k} week{'s' if k != 1 else ''} after its first lost report in {v} of {n} seasons" if i == 0
            else f"{k} week{'s' if k != 1 else ''} after in {v}"
            for i, (k, v) in enumerate(delay.items())
        ]
        counts = Counter()
        for text in col("scenario: caught").dropna():
            for piece in str(text).split(", "):
                fault, _, action = piece.rpartition(": ")
                counts[(fault, action)] += 1
        coded = ("impossible days", "duplicate", "wrong task")
        n_coded = sum(v for (f, _), v in counts.items() if f in coded)
        held_coded = sum(v for (f, a), v in counts.items() if f in coded and a == "hold")
        caught_coded = sum(v for (f, a), v in counts.items() if f in coded and a in ("hold", "query"))
        n_inflated = sum(v for (f, _), v in counts.items() if f == "days inflated")
        caught_inflated = sum(v for (f, a), v in counts.items() if f == "days inflated" and a in ("hold", "query"))
        held_inflated = counts[("days inflated", "hold")]
        verb = "held" if held_coded == caught_coded else "held or queried"
        verb_inflated = "queried" if held_inflated == 0 else "held or queried"
        return (
            f"The silent team was flagged {' and '.join(flag_parts) or '–'}. Of the "
            f"{int(num('scenario: corrupted reports').sum())} corrupted reports, "
            f"{int(num('scenario: corrupted reports caught').sum())} were held or queried: the report checker {verb} "
            f"{caught_coded} of the {n_coded} with impossible days, a duplicate or a wrong task, and {verb_inflated} "
            f"{caught_inflated} of the {n_inflated} with inflated days; the other {n_inflated - caught_inflated} were "
            "accepted."
        )
    if name == "missing item":
        out = []
        for part_name, label in (("lost item", "lost-item"), ("desk record", "desk-record")):
            lost_week = pd.to_numeric(col(f"scenario: {part_name}").astype(str).str.extract(r"in week (\d+)")[0],
                                      errors="coerce")
            delay = num(f"scenario: {part_name} reviewed in week") - lost_week
            out.append(
                f"The {label} dossier was held in {int(flag(f'scenario: {part_name} held').sum())} of {n} seasons and "
                f"reviewed {_weeks_count_text(delay)} after the loss, once the missing item had arrived; decision after "
                f"the hold: {outcomes(col(f'scenario: {part_name} decision'))}."
            )
        without_desk = col("scenario: desk record decision without the hold")
        cancelled = without_desk == "CANCEL"
        n_cancelled = int(cancelled.sum())
        mined_cancelled = int((cancelled & flag("scenario: desk record area holds mines")).sum())
        if n_cancelled:
            out.append(
                f"Without the hold, the desk-record dossier would have been cancelled in {n_cancelled} of {n} "
                "seasons, which the national criteria do not allow for an area with a recorded accident "
                f"({mined_cancelled} of those {n_cancelled} areas held mines)."
            )
        else:
            out.append("Without the hold, the desk-record dossier would not have been cancelled in any season.")
        out.append(
            "Without the hold, the lost-item dossier would have gone to "
            f"{outcomes(col('scenario: lost item decision without the hold'))}."
        )
        return " ".join(out)
    return ""


def _plot_scenario_stress_test_studio(
    policy_records: list[dict],
    scenario_records: list[dict],
    policy_runs: pd.DataFrame,
    scenario_runs: pd.DataFrame,
    chosen_scen: str,
) -> plt.Figure:
    """Render a 4-panel stress-test comparison figure for the injected scenarios, over the saved field seeds."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15.6, 8.2), facecolor="#ffffff")
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    weeks = np.arange(1, 53)
    n_seeds = int(scenario_runs["seed"].nunique())
    main_recs = [r for r in policy_records if r.get("policy") == "main policy" and "weekly" in r]
    if main_recs:
        m_c = np.array([r["weekly"]["mine areas cleared"] for r in main_recs], dtype=float)
        m_f = np.array([r["weekly"]["zones declared free"] for r in main_recs], dtype=float)
        ax1.plot(weeks, m_c.mean(axis=0), color="#1f4e78", linewidth=2.5, label="main policy (no fault)")
        ax2.plot(weeks, m_f.mean(axis=0), color="#1f4e78", linewidth=2.5, label="main policy (no fault)")

    for s_name in evaluate.SCENARIO_NAMES:
        s_recs = [r for r in scenario_records if r.get("scenario") == s_name and "weekly" in r]
        if not s_recs:
            continue
        col = SCENARIO_COLOURS.get(s_name, "#64748b")
        is_sel = s_name == chosen_scen
        lw = 2.4 if is_sel else 1.25
        alpha = 1.0 if is_sel else 0.55
        ls = "--" if is_sel else ":"
        arr_c = np.array([r["weekly"]["mine areas cleared"] for r in s_recs], dtype=float)
        arr_f = np.array([r["weekly"]["zones declared free"] for r in s_recs], dtype=float)
        ax1.plot(weeks, arr_c.mean(axis=0), color=col, linewidth=lw, alpha=alpha, linestyle=ls, label=s_name)
        ax2.plot(weeks, arr_f.mean(axis=0), color=col, linewidth=lw, alpha=alpha, linestyle=ls, label=s_name)
        if is_sel:
            ax1.fill_between(
                weeks,
                np.percentile(arr_c, 10, axis=0),
                np.percentile(arr_c, 90, axis=0),
                color=col,
                alpha=0.12,
            )
            ax2.fill_between(
                weeks,
                np.percentile(arr_f, 10, axis=0),
                np.percentile(arr_f, 90, axis=0),
                color=col,
                alpha=0.12,
            )

    ax1.set_title(f"Weekly Mine Areas Cleared: No-Fault vs. Injected Scenarios ({n_seeds} Seeds)", fontsize=9.6, fontweight="bold")
    ax1.set_xlabel("Week of Season", fontsize=8.2)
    ax1.set_ylabel("Cumulative Mine Areas Cleared", fontsize=8.2)
    ax1.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    ax2.set_title(f"Weekly Zones Declared Free: No-Fault vs. Injected Scenarios ({n_seeds} Seeds)", fontsize=9.6, fontweight="bold")
    ax2.set_xlabel("Week of Season", fontsize=8.2)
    ax2.set_ylabel("Cumulative Zones Declared Free", fontsize=8.2)
    ax2.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel 3: Seed-by-seed paired impact for `chosen_scen` vs `main policy`
    main_df = policy_runs[policy_runs["policy"] == "main policy"].set_index("seed")
    scen_df = scenario_runs[scenario_runs["scenario"] == chosen_scen].set_index("seed")
    common = main_df.index.intersection(scen_df.index)
    if len(common) > 0:
        d_clr = (scen_df.loc[common, "mine areas cleared"] - main_df.loc[common, "mine areas cleared"]).to_numpy(dtype=float)
        d_free = (scen_df.loc[common, "zones declared free"] - main_df.loc[common, "zones declared free"]).to_numpy(dtype=float)
        x_s = common.to_numpy(dtype=int)
        w_bar = 0.38
        ax3.bar(x_s - w_bar / 2, d_clr, width=w_bar, color=SCENARIO_COLOURS.get(chosen_scen, "#dc2626"), alpha=0.88, label="Δ Mine Areas Cleared")
        ax3.bar(x_s + w_bar / 2, d_free, width=w_bar, color="#475569", alpha=0.82, label="Δ Zones Declared Free")
        ax3.axhline(0, color="#0f172a", linewidth=0.9)
        ax3.set_xticks(x_s)
        ax3.set_title(
            f"Seed-by-Seed Throughput Impact of `{chosen_scen}`\n(Scenario minus No-Fault Main Policy across {len(common)} Seeds)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax3.set_xlabel("Field Seed (0 to 19)", fontsize=8.2)
        ax3.set_ylabel("Δ Count vs. Unperturbed Season", fontsize=8.2)
        ax3.legend(loc="lower left", fontsize=7.6, frameon=True)
        ax3.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel 4: each scenario's mean change against the same seed with no fault (and in areas released with mines)
    scen_list = [s for s in evaluate.SCENARIO_NAMES if s in scenario_runs["scenario"].values]
    y_pos = np.arange(len(scen_list))
    mean_d_clr = []
    mean_d_free = []
    mean_d_crit = []
    for s_name in scen_list:
        s_part = scenario_runs[scenario_runs["scenario"] == s_name].set_index("seed")
        cm = main_df.index.intersection(s_part.index)
        mean_d_clr.append(float((s_part.loc[cm, "mine areas cleared"] - main_df.loc[cm, "mine areas cleared"]).mean()))
        mean_d_free.append(float((s_part.loc[cm, "zones declared free"] - main_df.loc[cm, "zones declared free"]).mean()))
        mean_d_crit.append(
            float((s_part.loc[cm, "released with mines (critical)"] - main_df.loc[cm, "released with mines (critical)"]).mean())
        )

    b_colors = [SCENARIO_COLOURS.get(s, "#64748b") for s in scen_list]
    ax4.barh(y_pos - 0.18, mean_d_clr, height=0.34, color=b_colors, alpha=0.90, label="Mean Δ Mine Areas Cleared")
    ax4.barh(y_pos + 0.18, mean_d_free, height=0.34, color="#94a3b8", edgecolor="#334155", label="Mean Δ Zones Declared Free")
    ax4.axvline(0, color="#0f172a", linewidth=0.9)
    for idx_s, (mc_v, mf_v, mcr_v) in enumerate(zip(mean_d_clr, mean_d_free, mean_d_crit)):
        ax4.text(
            max(mc_v, mf_v, 0.0) + 0.08,
            idx_s,
            f"ΔClr {mc_v:+.2f} · ΔFree {mf_v:+.2f} · ΔReleased with mines {mcr_v:+.2f}",
            va="center",
            ha="left",
            fontsize=7.4,
            fontweight="bold",
        )
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(scen_list, fontsize=8.0)
    ax4.set_xlim(min(mean_d_clr + mean_d_free + [-1.0]) * 1.45, max(mean_d_clr + mean_d_free + [0.0]) + 3.6)
    ax4.set_xlabel(f"Mean Change vs. Main Policy with No Fault, Same Seed ({n_seeds} Seeds)", fontsize=8.2)
    ax4.set_title(
        "Mean Change against the Same Seed with No Fault\n"
        "(ΔReleased with mines: areas released with mines by decision)",
        fontsize=9.5,
        fontweight="bold",
    )
    ax4.legend(loc="upper left", fontsize=7.5, frameon=True)
    ax4.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


def _plot_policy_ablation_studio(
    policy_records: list[dict],
    policy_runs: pd.DataFrame,
    chosen_alt: str,
) -> plt.Figure:
    """Render a 4-panel counterfactual policy & ablation comparison figure over the saved field seeds."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15.6, 8.2), facecolor="#ffffff")
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_facecolor("#ffffff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    weeks = np.arange(1, 53)
    n_seeds = int(policy_runs["seed"].nunique())

    # Panel 1 & 2: 52-week trajectories of `mine areas cleared` and `zones declared free`
    for p_name in evaluate.POLICIES:
        p_recs = [r for r in policy_records if r["policy"] == p_name and "weekly" in r]
        if not p_recs:
            continue
        col = POLICY_COLOURS.get(p_name, "#334155")
        lw = 2.4 if p_name in ("main policy", chosen_alt) else 1.25
        alpha = 1.0 if p_name in ("main policy", chosen_alt) else 0.55
        ls = "-" if p_name == "main policy" else ("--" if p_name == chosen_alt else ":")

        arr_c = np.array([r["weekly"]["mine areas cleared"] for r in p_recs], dtype=float)
        arr_f = np.array([r["weekly"]["zones declared free"] for r in p_recs], dtype=float)

        ax1.plot(weeks, arr_c.mean(axis=0), color=col, linewidth=lw, alpha=alpha, linestyle=ls, label=p_name)
        ax2.plot(weeks, arr_f.mean(axis=0), color=col, linewidth=lw, alpha=alpha, linestyle=ls, label=p_name)
        if p_name in ("main policy", chosen_alt):
            ax1.fill_between(
                weeks,
                np.percentile(arr_c, 10, axis=0),
                np.percentile(arr_c, 90, axis=0),
                color=col,
                alpha=0.11,
            )
            ax2.fill_between(
                weeks,
                np.percentile(arr_f, 10, axis=0),
                np.percentile(arr_f, 90, axis=0),
                color=col,
                alpha=0.11,
            )

    ax1.set_title(f"Weekly Mine Areas Cleared (Mean & P10–P90 Band, {n_seeds} Seeds)", fontsize=9.8, fontweight="bold")
    ax1.set_xlabel("Week of Season", fontsize=8.2)
    ax1.set_ylabel("Cumulative Mine Areas Cleared", fontsize=8.2)
    ax1.legend(loc="upper left", fontsize=7.6, frameon=True)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    ax2.set_title(f"Weekly Zones Declared Free (Mean & P10–P90 Band, {n_seeds} Seeds)", fontsize=9.8, fontweight="bold")
    ax2.set_xlabel("Week of Season", fontsize=8.2)
    ax2.set_ylabel("Cumulative Zones Declared Free", fontsize=8.2)
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel 3: Seed-by-seed paired difference (chosen_alt minus main policy) on `mine areas cleared`
    main_df = policy_runs[policy_runs["policy"] == "main policy"].set_index("seed")
    alt_df = policy_runs[policy_runs["policy"] == chosen_alt].set_index("seed")
    common = main_df.index.intersection(alt_df.index)
    if len(common) > 0:
        diff_clr = (alt_df.loc[common, "mine areas cleared"] - main_df.loc[common, "mine areas cleared"]).to_numpy(dtype=float)
        seeds_x = common.to_numpy(dtype=int)
        b_cols = ["#0d6e52" if d > 0 else ("#94a3b8" if d == 0 else "#dc2626") for d in diff_clr]
        ax3.bar(seeds_x, diff_clr, color=b_cols, edgecolor="#1e293b", linewidth=0.5, width=0.68)
        ax3.axhline(0, color="#0f172a", linewidth=0.9)
        ax3.axhline(float(diff_clr.mean()), color="#1f4e78", linestyle="--", linewidth=1.3, label=f"Mean Δ = {diff_clr.mean():+.2f}")
        ax3.set_xticks(seeds_x)
        ax3.set_title(
            f"Seed-by-Seed Paired Difference in Mine Areas Cleared\n(`{chosen_alt}` minus `main policy` across {len(common)} seeds)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax3.set_xlabel("Field Seed (0 to 19)", fontsize=8.2)
        ax3.set_ylabel("Δ Mine Areas Cleared", fontsize=8.2)
        ax3.legend(loc="upper right", fontsize=7.8, frameon=True)
        ax3.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel 4: clearance days on land without mines, and areas released with mines by decision, by policy
    pol_list = [p for p in evaluate.POLICIES if p in policy_runs["policy"].values]
    y_p = np.arange(len(pol_list))
    clean_land_share = [
        100.0 * float(policy_runs.loc[policy_runs["policy"] == p, "share of clearance days on land without mines"].mean())
        for p in pol_list
    ]
    crit_rel = [
        float(policy_runs.loc[policy_runs["policy"] == p, "released with mines (critical)"].mean())
        for p in pol_list
    ]
    bar_cols = [POLICY_COLOURS.get(p, "#64748b") for p in pol_list]
    ax4.barh(y_p, clean_land_share, color=bar_cols, alpha=0.86, height=0.58)
    for idx_p, (w_pct, cr_val) in enumerate(zip(clean_land_share, crit_rel)):
        ax4.text(
            w_pct + 0.8,
            idx_p,
            f"{w_pct:.1f}% · {cr_val:.2f} areas released with mines by decision a season",
            va="center",
            fontsize=7.6,
            fontweight="bold",
        )
    ax4.set_yticks(y_p)
    ax4.set_yticklabels(pol_list, fontsize=8.0)
    ax4.set_xlim(0, max(clean_land_share, default=50.0) * 2.3)
    ax4.set_xlabel("% of Clearance Days Spent on Land Without Mines", fontsize=8.2)
    ax4.set_title(
        "Clearance Days on Land Without Mines, and Areas Released with Mines by Decision",
        fontsize=9.8,
        fontweight="bold",
    )
    ax4.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


def render_evaluation(root: Path) -> None:
    restricted_header("Evaluation: The simulation's hidden truth, for evaluation only")
    st.markdown(
        '<div class="eval-banner">'
        "<span><strong>The simulation's hidden truth, for evaluation only</strong></span>"
        "<span>Benchmark & ground-truth evaluation — never visible in field operations</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This screen evaluates the policies, the completion forecasts, the critical failures, the injected scenarios "
        "and the dossier reviewer against the simulation's truth (RELand's labels, plus any contamination a scenario "
        "injects) and the saved replay records (`runs/replay/`). No one in a real programme has this truth, and the "
        "programme screens do not show it."
    )

    bundle = get_twin_bundle(root)
    policy_records, scenario_records = _load_replay_records(str(root))
    full_reviews = _load_full_season_reviews(str(root))
    run_info = load_run_info(str(root))

    policy_runs = evaluate.frame(policy_records) if policy_records else pd.DataFrame()
    scenario_runs = evaluate.frame(scenario_records) if scenario_records else pd.DataFrame()
    n_seeds = int(policy_runs["seed"].nunique()) if not policy_runs.empty else 0

    # 1. Policies compared over 20 seeds (notebook section 9.3)
    st.subheader("1. The policies compared over 20 seeds (notebook section 9.3)")
    if policy_runs.empty:
        st.info("No policy replay records found in `runs/replay/`.")
    else:
        st.caption(
            f"Loaded {len(policy_records)} saved replay seasons across {policy_runs['policy'].nunique()} policies "
            "from `runs/replay/` (reviewed by the reference reviewer on the true evidence classes)."
        )

        # Interactive Counterfactual Policy & Ablation Studio
        alt_policies = [p for p in evaluate.POLICIES if p != "main policy" and p in policy_runs["policy"].values]
        if alt_policies:
            st.markdown("#### 🔬 Counterfactual Policy & Ablation Comparison Studio")
            chosen_alt = st.selectbox(
                f"Select a counterfactual policy / ablation to compare head-to-head against `main policy` across all {n_seeds} field seeds:",
                options=alt_policies,
                index=alt_policies.index("technical survey first") if "technical survey first" in alt_policies else 0,
            )
            main_sub = policy_runs[policy_runs["policy"] == "main policy"].set_index("seed")
            alt_sub = policy_runs[policy_runs["policy"] == chosen_alt].set_index("seed")
            common_s = main_sub.index.intersection(alt_sub.index)

            if len(common_s) > 0:
                d_clr = alt_sub.loc[common_s, "mine areas cleared"] - main_sub.loc[common_s, "mine areas cleared"]
                d_free = alt_sub.loc[common_s, "zones declared free"] - main_sub.loc[common_s, "zones declared free"]
                d_waste = 100.0 * (
                    alt_sub.loc[common_s, "share of clearance days on land without mines"]
                    - main_sub.loc[common_s, "share of clearance days on land without mines"]
                )
                d_crit = (
                    alt_sub.loc[common_s, "released with mines (critical)"]
                    - main_sub.loc[common_s, "released with mines (critical)"]
                )

                p_k1, p_k2, p_k3, p_k4 = st.columns(4)
                p_k1.metric(
                    f"Mine Areas Cleared (`{chosen_alt}`)",
                    f"{alt_sub['mine areas cleared'].mean():.2f}",
                    f"{d_clr.mean():+.2f} vs main ({int((d_clr > 0).sum())}W / {int((d_clr == 0).sum())}T / {int((d_clr < 0).sum())}L)",
                )
                p_k2.metric(
                    f"Zones Declared Free (`{chosen_alt}`)",
                    f"{alt_sub['zones declared free'].mean():.2f}",
                    f"{d_free.mean():+.2f} vs main ({int((d_free > 0).sum())}W / {int((d_free == 0).sum())}T / {int((d_free < 0).sum())}L)",
                )
                p_k3.metric(
                    "Clearance Days on Land Without Mines",
                    f"{100.0 * alt_sub['share of clearance days on land without mines'].mean():.1f}%",
                    f"{d_waste.mean():+.1f} pp vs main policy",
                    delta_color="inverse",
                )
                p_k4.metric(
                    "Areas Released with Mines by Decision",
                    f"{alt_sub['released with mines (critical)'].mean():.2f} / season",
                    f"{d_crit.mean():+.2f} vs main policy",
                    delta_color="inverse",
                )

            fig_studio = _plot_policy_ablation_studio(policy_records, policy_runs, chosen_alt)
            st.pyplot(fig_studio)
            plt.close(fig_studio)

        present_measures = [m for m in MEASURES if m in policy_runs.columns]
        # one text column per policy: the seasons' count and the "mean (min–max)" texts, so no column mixes types
        ranges_tbl = evaluate.ranges(policy_runs, present_measures).T.astype(str)
        st.markdown(f"**Mean (min–max) over {n_seeds} seeds, by policy (`evaluate.ranges`):**")
        st.dataframe(ranges_tbl, width="stretch")

        st.markdown("**Seed-by-seed paired differences against the main policy (`evaluate.paired`):**")
        paired_tbl = evaluate.paired(policy_runs, BETTER)
        st.dataframe(paired_tbl, width="stretch", hide_index=True)

        # Weekly "mine areas cleared" and "zones declared free" curves from the saved replay records
        with st.expander("Weekly 'mine areas cleared' and 'zones declared free' curves (52 weeks)"):
            weeks = list(range(1, 53))
            mean_cleared = {}
            mean_free = {}
            for p_name in evaluate.POLICIES:
                p_recs = [r for r in policy_records if r["policy"] == p_name and "weekly" in r]
                if p_recs:
                    mean_cleared[p_name] = np.mean(
                        [r["weekly"]["mine areas cleared"] for r in p_recs], axis=0
                    ).round(2)
                    mean_free[p_name] = np.mean(
                        [r["weekly"]["zones declared free"] for r in p_recs], axis=0
                    ).round(2)
            if mean_cleared:
                st.markdown(f"**Mean weekly `mine areas cleared` over {n_seeds} seeds:**")
                st.line_chart(pd.DataFrame(mean_cleared, index=pd.Index(weeks, name="week")))
                st.markdown(f"**Mean weekly `zones declared free` over {n_seeds} seeds:**")
                st.line_chart(pd.DataFrame(mean_free, index=pd.Index(weeks, name="week")))

    # 2. Forecast intervals coverage (notebook section 9.4)
    st.divider()
    st.subheader("2. Do the forecast's intervals hold? (notebook section 9.4)")
    if not policy_records:
        st.info("No replay records available for forecast coverage.")
    else:
        col_w1, col_w27 = st.columns(2)
        with col_w1:
            st.markdown("**The forecast with the first plan (week 1)**")
            st.dataframe(_coverage_summary(policy_records, 1), width="stretch")
        with col_w27:
            st.markdown("**The forecast with the plan of week 27 (mid-season)**")
            st.dataframe(_coverage_summary(policy_records, 27), width="stretch")

        with st.expander("Inspect per-seed forecast coverage table (`evaluate.coverage_table`)"):
            fc_week_choice = st.radio("Forecast week", options=[1, 27], horizontal=True)
            cov_raw = evaluate.coverage_table(policy_records, week=fc_week_choice)
            st.dataframe(cov_raw.round(3), width="stretch", hide_index=True)

    # 3. Critical failures and what the hidden truth shows (notebook section 9.5)
    st.divider()
    st.subheader("3. Critical failures and what the hidden truth shows (notebook section 9.5)")
    if not policy_runs.empty:
        crit_cols = [
            "released with mines (critical)",
            "cancelled with mines",
            "reduced with mines",
            "mine areas not flagged, in zones declared free (critical)",
            "mine areas not flagged, in surveyed zones",
            "zones declared free",
        ]
        crit_cols = [c for c in crit_cols if c in policy_runs.columns]
        critical_df = policy_runs.groupby("policy", sort=False)[crit_cols].mean().round(2)
        st.markdown(f"**The critical failures and related measures, mean per season over {n_seeds} seeds:**")
        st.dataframe(critical_df, width="stretch")

        if "share of clearance days on land without mines" in policy_runs.columns:
            share_df = (
                policy_runs.groupby("policy", sort=False)["share of clearance days on land without mines"]
                .agg(["mean", "min", "max"])
                .round(3)
                .rename(columns={"mean": "share of clearance days on land without mines (mean)"})
            )
            st.markdown("**Share of clearance days spent on land without mines:**")
            st.dataframe(share_df, width="stretch")

    # 4. Injected scenarios (notebook section 10)
    st.divider()
    st.subheader("4. The injected scenarios: Fault-Injection Stress-Test Studio (notebook section 10)")
    if scenario_runs.empty or policy_runs.empty:
        st.info("No scenario replay records found in `runs/replay/`.")
    else:
        main_runs = policy_runs[policy_runs["policy"] == "main policy"].set_index("seed")
        avail_scenarios = [s for s in evaluate.SCENARIO_NAMES if s in scenario_runs["scenario"].values]

        if avail_scenarios:
            st.markdown(f"#### 🛡️ The {len(avail_scenarios)} injected scenarios and the twin's responses (`twin/scenarios.py`)")
            n_scen_seeds = int(scenario_runs["seed"].nunique())
            st.caption(
                f"Tests **P5's insight (*'synthetic data serve stress tests'*)** on **{len(scenario_records)} saved "
                f"seasons with an injected scenario** ({len(avail_scenarios)} things that go wrong in a season × "
                f"{n_scen_seeds} field seeds, under the main policy, on the same seeds as its seasons with no fault). "
                "Select a scenario below to see what is injected, what the twin should do and did, what it costs, "
                "and the change in areas released with mines."
            )
            chosen_scen = st.selectbox(
                f"Select an injected scenario to inspect across all {n_scen_seeds} field seeds:",
                options=avail_scenarios,
                index=0,
                key="eval_scen_stress_select",
            )
            s_sub = scenario_runs[scenario_runs["scenario"] == chosen_scen].set_index("seed")
            doc_meta = _scenario_doctrine(bundle).get(chosen_scen, {})
            if doc_meta:
                with st.container(border=True):
                    st.markdown(f"**{doc_meta['title']}**")
                    st.markdown(
                        f"- **Grounding:** {doc_meta['grounding']}\n"
                        f"- **What is injected:** {doc_meta['injection']}\n"
                        f"- **What the twin should do** (notebook section 10): {doc_meta['should_do']}\n"
                        f"- **What it did in the replay** ({len(s_sub)} seasons): {_scenario_results(chosen_scen, s_sub)}"
                    )

            cm_s = main_runs.index.intersection(s_sub.index)
            if len(cm_s) > 0:
                sd_clr = s_sub.loc[cm_s, "mine areas cleared"] - main_runs.loc[cm_s, "mine areas cleared"]
                sd_free = s_sub.loc[cm_s, "zones declared free"] - main_runs.loc[cm_s, "zones declared free"]
                sd_crit = s_sub.loc[cm_s, "released with mines (critical)"] - main_runs.loc[cm_s, "released with mines (critical)"]
                sd_idle = s_sub.loc[cm_s, "team-days with no task"] - main_runs.loc[cm_s, "team-days with no task"]

                sk1, sk2, sk3, sk4 = st.columns(4)
                sk1.metric(
                    f"Mine Areas Cleared (`{chosen_scen}`)",
                    f"{s_sub['mine areas cleared'].mean():.2f}",
                    f"{sd_clr.mean():+.2f} vs no-fault ({sd_clr.min():g} to {sd_clr.max():g})",
                )
                sk2.metric(
                    f"Zones Declared Free (`{chosen_scen}`)",
                    f"{s_sub['zones declared free'].mean():.2f}",
                    f"{sd_free.mean():+.2f} vs no-fault ({sd_free.min():g} to {sd_free.max():g})",
                )
                sk3.metric(
                    "Change in Areas Released with Mines by Decision",
                    f"{sd_crit.mean():+.2f} / season",
                    f"more than with no fault in {int((sd_crit > 0).sum())} of {len(sd_crit)} seasons",
                    delta_color="off",
                )
                sk4.metric(
                    "Δ Team-Days with No Task",
                    f"{sd_idle.mean():+.2f} days",
                    f"Range: {sd_idle.min():g} to {sd_idle.max():g} days",
                    delta_color="inverse",
                )

            fig_scen = _plot_scenario_stress_test_studio(
                policy_records, scenario_records, policy_runs, scenario_runs, chosen_scen
            )
            st.pyplot(fig_scen)
            plt.close(fig_scen)

        cost_rows = []
        for name, part in scenario_runs.groupby("scenario", sort=False):
            part = part.set_index("seed")
            common_seeds = part.index.intersection(main_runs.index)
            row = {"scenario": name}
            for m in [
                "mine areas cleared",
                "zones declared free",
                "released with mines (critical)",
                "team-days with no task",
            ]:
                if m in part.columns and m in main_runs.columns:
                    d = part.loc[common_seeds, m] - main_runs.loc[common_seeds, m]
                    row[f"{m}: mean change"] = round(float(d.mean()), 2)
                    row[f"{m}: range"] = f"{d.min():g} to {d.max():g}"
            cost_rows.append(row)
        st.markdown(f"**Scenario cost against the main policy on the same {scenario_runs['seed'].nunique()} seeds:**")
        # each scenario's column holds mean changes and "min to max" ranges: shown as text, so no column mixes types
        st.dataframe(pd.DataFrame(cost_rows).set_index("scenario").T.astype(str), width="stretch")

        with st.expander("Scenario detection and response measures (by scenario)"):
            for s_name in evaluate.SCENARIO_NAMES:
                part = scenario_runs[scenario_runs["scenario"] == s_name]
                if part.empty:
                    continue
                s_cols = [c for c in part.columns if c.startswith("scenario: ")]
                m_df = part[["seed"] + s_cols].dropna(axis=1, how="all").set_index("seed")
                st.markdown(f"**{s_name}**")
                flag_cols = [
                    c
                    for c in m_df.columns
                    if len(m_df[c].dropna()) > 0
                    and m_df[c].dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all()
                ]
                rest = m_df.drop(columns=flag_cols)
                numeric = rest.apply(pd.to_numeric, errors="coerce")
                numbers = numeric.loc[:, numeric.notna().any()]
                if numbers.shape[1]:
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "seasons with a value": numbers.notna().sum(),
                                "mean": numbers.mean(),
                                "min": numbers.min(),
                                "max": numbers.max(),
                            }
                        ).round(2),
                        width="stretch",
                    )
                if flag_cols:
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "seasons where it holds": m_df[flag_cols].astype("boolean").sum().astype(int),
                                "of": m_df[flag_cols].notna().sum(),
                            }
                        ),
                        width="stretch",
                    )
                text_cols = rest.loc[:, numeric.isna().all()]
                if text_cols.shape[1]:
                    st.dataframe(text_cols.head(5), width="stretch")

    # 5. The agent against the coded rules (notebook section 9.2)
    st.divider()
    st.subheader("5. The agent against the coded rules (notebook section 9.2)")
    agent_model = run_info.get("model", "saved model")
    if full_reviews.empty:
        st.info("No saved agent reviews found in `runs/season_reviews.csv`.")
    else:
        n_rev = len(full_reviews)
        n_agree = int(full_reviews["agrees"].sum())
        agree_rate = float(full_reviews["agrees"].mean())
        n_fb = int(full_reviews["fallback"].sum())
        n_veto = int(full_reviews["vetoed at sign-off"].sum())
        st.markdown(
            f"**Model:** `{agent_model}` | **{n_rev} dossier reviews:** the agent's recommendation matched "
            f"the coded rules' (`reference`) on the same dossier in **{n_agree} ({agree_rate:.0%})**; "
            f"fallbacks: **{n_fb}**; recommendations vetoed by the rule check at sign-off: **{n_veto}**."
        )
        ct = pd.crosstab(full_reviews["reference"], full_reviews["outcome"], margins=True).rename_axis(
            index="coded rules (reference)", columns=f"agent ({agent_model})"
        )
        st.markdown("**Confusion matrix (`reference` vs `outcome`):**")
        st.dataframe(ct, width="stretch")

        disagree = full_reviews[~full_reviews["agrees"]]
        if len(disagree):
            st.markdown(f"**Dossiers where the agent differed from the coded rules ({len(disagree)}):**")
            st.dataframe(
                disagree[
                    [
                        "week",
                        "area_id",
                        "visit",
                        "mined",
                        "recorded",
                        "branch",
                        "reference",
                        "outcome",
                        "decision",
                        "confidence",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("No dossier where the agent's recommendation differed from the coded rules'.")

        with st.expander(f"View all {n_rev} rows of `runs/season_reviews.csv` (including `mined` and `reference`)"):
            st.dataframe(full_reviews.fillna(""), width="stretch", hide_index=True)

        stats_df, flagged_recs = _load_review_json_stats(str(root))
        if not stats_df.empty:
            with st.expander("Agent review run statistics (calls, tokens, cost, safeguards)"):
                num_cols = [
                    "model calls",
                    "tool calls",
                    "seconds",
                    "input tokens",
                    "cached input tokens",
                    "output tokens",
                    "cost (USD)",
                    "vetoes",
                    "quote failures",
                    "tool errors",
                    "sentences removed",
                    "instruction lines found",
                ]
                desc = stats_df[num_cols].describe().loc[["mean", "min", "50%", "max"]].round(2)
                desc["cost (USD)"] = stats_df["cost (USD)"].describe().loc[["mean", "min", "50%", "max"]].round(4)
                st.dataframe(desc, width="stretch")
                totals = stats_df[num_cols].sum()
                st.markdown(
                    f"Totals over **{len(stats_df)}** reviews: **{int(totals['model calls'])}** model calls, "
                    f"**{int(totals['tool calls'])}** tool calls, **{totals['seconds'] / 60:.1f}** minutes, "
                    f"**USD {totals['cost (USD)']:.3f}** at the prices in `twin/agent.py` (cached input counted at the "
                    f"full price, so an upper bound), **{int(totals['vetoes'])}** vetoes during review, "
                    f"**{int(totals['quote failures'])}** quote failures, "
                    f"**{len(flagged_recs)}** dossier(s) with text addressed to the reviewer."
                )

    # Compare agent's season metrics vs coded rules' season metrics (and truth checks from the rebuild)
    evaluation = get_evaluation_objects(root)
    result = evaluation.get("result")
    s_obj = evaluation.get("setup")

    if result is not None:
        seed0_coded = next(
            (r for r in policy_records if r["policy"] == "main policy" and int(r["seed"]) == 0), None
        )
        if seed0_coded is not None:
            compare_df = pd.DataFrame(
                {
                    f"agent's season ({agent_model})": pd.Series(result.metrics),
                    "coded rules' season (seed 0)": pd.Series(seed0_coded["metrics"]),
                }
            )
            differ = compare_df[compare_df.iloc[:, 0] != compare_df.iloc[:, 1]]
            st.markdown("**Season measures: agent's season vs. coded rules' season on seed 0:**")
            if differ.empty:
                st.success("The agent's season and the coded rules' season have identical measures on seed 0.")
            else:
                st.dataframe(differ, width="stretch")
            with st.expander("Full season measures (`result.metrics` vs coded rules on seed 0)"):
                st.dataframe(compare_df, width="stretch")

        with st.expander("Simulated field truth on seed 0 (`result.mined`, `result.base_mined`, `s.truth`)", expanded=True):
            n_cells_mined = int(s_obj.truth.sum()) if s_obj is not None else 0
            n_areas_mined = int(result.mined.sum())
            n_base_mined = int(result.base_mined.sum())
            st.markdown(
                f"- **RELand cells with `mines == 1` (`s.truth`):** `{n_cells_mined:,}`\n"
                f"- **Areas with mines by RELand's labels (`result.base_mined`):** `{n_base_mined:,}` of `{len(result.base_mined):,}`\n"
                f"- **Areas that hold mines this season (`result.mined`: RELand's labels plus any injected "
                f"contamination; cleared areas still count):** `{n_areas_mined:,}`"
            )
            # Visual 15-municipality ground-truth grid map (no axes, no scale, no coordinates)
            cells_meta = bundle["cells_meta"]
            zones_meta = bundle["zones_meta"]
            areas_meta = bundle["areas_meta"]
            done_clear_areas = set(
                result.tasks[
                    (result.tasks["kind"] == "CLEARANCE") & (result.tasks["status"] == "done")
                ]["area"].dropna().astype(str)
            )
            truth_state = pd.Series(0.0, index=areas_meta.index)
            mined_idx = result.mined.index[result.mined.astype(bool)]
            truth_state.loc[mined_idx] = 1.0
            cleared_mined_idx = [a for a in mined_idx if a in done_clear_areas]
            truth_state.loc[cleared_mined_idx] = 2.0

            fig_truth, axes_t = plt.subplots(3, 5, figsize=(15.0, 8.6), facecolor=grid_map.SURFACE)
            truth_colours = ["#e6e4dd", "#d95f43", "#2b7bba"]
            for idx_m, (m_code, m_name) in enumerate(config.NAME_OF.items()):
                ax_t = axes_t.ravel()[idx_m]
                g_t = grid_map.compute_grid(cells_meta, truth_state, m_code, zones_meta.index, areas_meta.index)
                m_mask = areas_meta["municipality_code"] == m_code
                n_m = int((truth_state[m_mask] >= 1.0).sum())
                n_c = int((truth_state[m_mask] == 2.0).sum())
                grid_map._draw_cell_grid(
                    ax_t,
                    g_t,
                    colours=truth_colours,
                    title=m_name,
                    subtitle=f"True mined areas: {n_m} ({n_c} cleared)",
                )
            fig_truth.legend(
                handles=[
                    Patch(color="#e6e4dd", label=f"No mines ({int((truth_state == 0.0).sum())} areas)"),
                    Patch(
                        color="#d95f43",
                        label=f"True mined — not yet cleared ({int((truth_state == 1.0).sum())} areas)",
                    ),
                    Patch(
                        color="#2b7bba",
                        label=f"True mined — cleared in the season ({int((truth_state == 2.0).sum())} areas)",
                    ),
                ],
                loc="lower center",
                ncol=3,
                frameon=False,
                fontsize=8.5,
                bbox_to_anchor=(0.5, 0.01),
            )
            fig_truth.suptitle(
                "Simulation Hidden Truth (Seed 0) — 15-Municipality Twin Grid Map (no axes, no scale, no coordinates)",
                fontsize=11,
                fontweight="bold",
                color=grid_map.INK,
                y=0.98,
            )
            fig_truth.subplots_adjust(bottom=0.09, top=0.93, hspace=0.32, wspace=0.12)
            st.pyplot(fig_truth)
            plt.close(fig_truth)

            if s_obj is not None:
                ranking_eval = prioritise.evaluate_rankings(s_obj.models, s_obj.truth, bundle["cells_meta"])
                summary_rank = ranking_eval.groupby("ranking").agg(
                    **{
                        "mean PR-AUC": ("PR-AUC", "mean"),
                        "mine areas in the first 10%": ("mine areas in first 10%", "sum"),
                    }
                )
                summary_rank.index = summary_rank.index.map(
                    lambda r: "the gate's choice" if r == "gated" else prioritise.RANKINGS.get(r, r)
                )
                st.markdown("**Risk rankings evaluated against each municipality's own labels (section 6.1):**")
                st.dataframe(summary_rank.round(3), width="stretch")

    # 6. The expert review (runs/expert_review.csv)
    st.divider()
    st.subheader("6. Domain-Expert Review & Case-Study Audit Studio (`runs/expert_review.csv`)")
    expert_path = root / "runs" / "expert_review.csv"
    if expert_path.exists():
        expert_df = pd.read_csv(expert_path).fillna("")
        rev_exp = expert_df[expert_df["kind"].str.startswith("review")].copy()
        rep_exp = expert_df[expert_df["kind"].str.startswith("report")].copy()

        n_rev_exp = len(rev_exp)
        n_sign = int((rev_exp["verdict"] == "Would sign off").sum())
        n_acceptable = int((rev_exp["verdict"] == "Acceptable call").sum())
        n_rev_ok = n_sign + n_acceptable
        err_mask = rev_exp["verdict"].str.startswith("Error")
        n_rev_err = int(err_mask.sum())
        n_rev_differ = int(rev_exp["kind"].str.contains("differs").sum())
        n_rev_sampled = int(rev_exp["kind"].str.contains("sampled").sum())
        n_rep_exp = len(rep_exp)
        n_rep_accurate = int(rep_exp["verdict"].str.startswith("Accurate").sum())
        n_rep_useful = int((rep_exp["verdict"] == "Accurate and useful").sum())
        n_rep_edits = int((rep_exp["verdict"] == "Accurate, needs edits").sum())
        disc_mask = rev_exp["register vs key"] != "none"

        def _review_label(r) -> str:
            visit = int(r["visit"]) if str(r["visit"]) != "" else 1
            return str(r["area_id"]) + (f" (visit {visit})" if visit > 1 else "")

        st.caption(
            f"Loaded **{len(expert_df)} judgements by the project's domain expert** from `runs/expert_review.csv` "
            f"({n_rev_exp} dossier reviews and {n_rep_exp} weekly situation reports)."
        )
        ek1, ek2, ek3, ek4 = st.columns(4)
        ek1.metric(
            "Dossier Reviews Judged",
            f"{n_rev_ok} / {n_rev_exp} Acceptable",
            f"{n_sign} 'Would sign off' · {n_acceptable} 'Acceptable call'",
        )
        ek2.metric(
            "Expert-Caught Review Errors",
            f"{n_rev_err} / {n_rev_exp} Flagged",
            ", ".join(_review_label(r) for _, r in rev_exp[err_mask].iterrows()) or "none",
            delta_color="inverse",
        )
        ek3.metric(
            "Weekly Sitreps Judged",
            f"{n_rep_accurate} / {n_rep_exp} Judged Accurate",
            f"{n_rep_useful} 'Accurate and useful' · {n_rep_edits} 'Accurate, needs edits'",
        )
        ek4.metric(
            "Register vs. Key Discrepancies",
            f"{int(disc_mask.sum())} / {n_rev_exp} Reviews",
            ", ".join(_review_label(r) for _, r in rev_exp[disc_mask].iterrows()) or "none",
        )

        # 2-Panel Visual Summary of Expert Judgements
        fig_exp, (ax_er1, ax_er2) = plt.subplots(1, 2, figsize=(15.4, 4.3), facecolor="#ffffff")
        for ax in (ax_er1, ax_er2):
            ax.set_facecolor("#ffffff")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        v_counts = rev_exp["verdict"].value_counts()
        v_labels = list(v_counts.index)
        v_vals = v_counts.to_numpy(dtype=int)
        v_cols = [
            "#0d6e52" if v == "Would sign off" else ("#2563eb" if v == "Acceptable call" else "#dc2626")
            for v in v_labels
        ]
        y_er1 = np.arange(len(v_labels))
        ax_er1.barh(y_er1, v_vals, color=v_cols, height=0.55, alpha=0.88)
        for i_v, val in enumerate(v_vals):
            ax_er1.text(val + 0.18, i_v, f"{val} review(s)", va="center", fontsize=8.2, fontweight="bold")
        ax_er1.set_yticks(y_er1)
        ax_er1.set_yticklabels(v_labels, fontsize=8.3)
        ax_er1.set_xlim(0, max(v_vals, default=5) + 2.0)
        ax_er1.set_xlabel("Number of Dossier Reviews", fontsize=8.2)
        ax_er1.set_title(
            f"Expert Verdicts on {n_rev_exp} Dossier Reviews\n"
            f"(the {n_rev_differ} that differ from the coded rules and {n_rev_sampled} drawn at random)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax_er1.grid(axis="x", linestyle=":", alpha=0.45)

        rep_items = [f"Wk {int(r['week']):02d}" for _, r in rep_exp.iterrows()]
        rep_omits = [
            len([x for x in str(r["left out of the report"]).split(";") if x.strip()])
            for _, r in rep_exp.iterrows()
        ]
        rep_cols = [
            "#0d6e52" if str(r["verdict"]) == "Accurate and useful" else "#d97706"
            for _, r in rep_exp.iterrows()
        ]
        x_er2 = np.arange(len(rep_items))
        ax_er2.bar(x_er2, rep_omits, color=rep_cols, width=0.55, alpha=0.88)
        for i_r, (om_c, (_, r_row)) in enumerate(zip(rep_omits, rep_exp.iterrows())):
            ax_er2.text(
                i_r,
                om_c + 0.12,
                f"{om_c} left out\n({r_row['verdict']})",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
            )
        ax_er2.set_xticks(x_er2)
        ax_er2.set_xticklabels(rep_items, fontsize=8.3)
        ax_er2.set_ylim(0, max(rep_omits, default=5) + 1.6)
        ax_er2.set_ylabel("Facts of the Week Left Out", fontsize=8.2)
        ax_er2.set_title(
            f"{n_rep_exp} Weekly Situation Reports Drawn at Random: Facts Left Out & Verdict\n"
            "(Green = 'Accurate and useful', Amber = 'Accurate, needs edits')",
            fontsize=9.3,
            fontweight="bold",
        )
        ax_er2.grid(axis="y", linestyle=":", alpha=0.45)
        fig_exp.tight_layout()
        st.pyplot(fig_exp)
        plt.close(fig_exp)

        # Interactive Side-by-Side Expert Case-Study Inspector
        st.markdown("#### 🔍 Interactive Expert Case-Study & Primary Document Inspector")
        st.caption(
            f"Select any of the {len(expert_df)} judged items below to inspect the question asked, **the expert's "
            "answer, word for word**, the comparison against the simulated field's ground-truth key "
            "(`register vs key` / `mined`), and the saved sign-off sheet or weekly situation report."
        )
        exp_indices = list(range(len(expert_df)))

        def _fmt_exp_row(idx: int) -> str:
            r = expert_df.iloc[idx]
            if str(r["kind"]).startswith("review"):
                return (
                    f"[{idx+1:02d}] Dossier Review {r['item']} · {r['municipality']} (Area {r['area_id']}, Visit {int(r['visit'])}) "
                    f"→ Agent: {r['outcome']} | Verdict: {r['verdict']}"
                )
            return f"[{idx+1:02d}] Weekly Sitrep {r['item']} (Week {int(r['week'])}) | Verdict: {r['verdict']}"

        # Default to w46_SAB-0214_v1, one of the two reviews the expert judged an error
        default_exp_idx = next((i for i in exp_indices if expert_df.iloc[i]["item"] == "w46_SAB-0214_v1"), 0)
        sel_exp_idx = st.selectbox(
            "Select an expert-judged dossier review or situation report:",
            options=exp_indices,
            index=default_exp_idx,
            format_func=_fmt_exp_row,
            key="eval_expert_case_select",
        )
        erow = expert_df.iloc[sel_exp_idx]
        col_ej, col_edoc = st.columns([1.05, 1.15])

        with col_ej:
            with st.container(border=True):
                st.markdown(f"##### 👨‍✈️ Expert Judgement Card: `{erow['item']}`")
                st.markdown(
                    f"- **Category:** `{erow['kind']}` · **Week:** `{erow['week']}`\n"
                    f"- **Question Asked:** {erow['question']}\n"
                    f"- **Options Provided:** `{erow['options']}`\n"
                    f"- **The expert's answer (word for word):** **{erow['judgement']}**\n"
                    f"- **Verdict (for reports, Claude's reading of the answer):** `{erow['verdict']}`"
                )
                if str(erow["kind"]).startswith("review"):
                    st.markdown(
                        f"- **Area & Municipality:** `{erow['area_id']}` ({erow['municipality']}) · **Visit:** `{erow['visit']}`\n"
                        f"- **Hidden Ground Truth (`mined`):** `{erow['mined']}` · **Database Accident Recorded (`recorded`):** `{erow['recorded']}`\n"
                        f"- **Agent Outcome vs. Coded Reference:** Agent `{erow['outcome']}` vs. Reference `{erow['reference']}`\n"
                        f"- **Agent Register vs. Field Ground-Truth Key:** `{erow['register vs key']}`"
                    )
                else:
                    st.markdown(
                        f"- **Weekly Facts Left Out of Draft:** `{erow['left out of the report'] or 'None'}`"
                    )
                st.info(f"**Claude's note, checked against the saved files (`runs/expert_review.csv`):**\n\n{erow['note']}")

        with col_edoc:
            with st.container(border=True):
                item_key = str(erow["item"])
                if str(erow["kind"]).startswith("review"):
                    md_p = root / "runs" / "reviews" / f"{item_key}_signoff.md"
                    js_p = root / "runs" / "reviews" / f"{item_key}_result.json"
                    st.markdown(f"##### 📄 Saved Sign-Off Sheet (`runs/reviews/{item_key}_signoff.md`)")
                    if md_p.exists():
                        st.markdown(md_p.read_text(encoding="utf-8"))
                    if js_p.exists():
                        with st.expander(f"Inspect raw agent JSON register (`{item_key}_result.json`)"):
                            st.json(json.loads(js_p.read_text(encoding="utf-8")))
                else:
                    sr_p = root / "runs" / "sitreps" / f"{item_key}.json"
                    st.markdown(f"##### 📰 Saved Weekly Sitrep (`runs/sitreps/{item_key}.json`)")
                    if sr_p.exists():
                        sr_rec = json.loads(sr_p.read_text(encoding="utf-8"))
                        st.markdown(sr_rec.get("draft", ""))
                        with st.expander(f"Inspect week {erow['week']} facts JSON passed to model"):
                            st.json(sr_rec.get("facts", {}))

        with st.expander(f"View all {len(expert_df)} rows of `runs/expert_review.csv`"):
            # shown as text: the sheet's columns mix numbers, true/false and the blanks of the report rows
            st.dataframe(expert_df.astype(str), width="stretch", hide_index=True)
    else:
        st.info(
            "`runs/expert_review.csv` is not yet present in `runs/`. "
            "Once added, this table will load and display it automatically."
        )

    # 7. The notebook's figures (figures/*.png)
    st.divider()
    st.subheader("7. The notebook's figures (`figures/*.png`)")
    figures_dir = root / "figures"
    png_files = sorted(figures_dir.glob("*.png")) if figures_dir.is_dir() else []
    if not png_files:
        st.info("No PNG figures found in `figures/`.")
    else:
        fig_captions = {
            "architecture.png": "Architecture of the digital twin and its overlays (`figures/architecture.png`)",
            "capacity_check.png": "Humanitarian clearance in the 15 pilot municipalities by year vs simulated capacity (`figures/capacity_check.png`)",
            "zones.png": "Operational zones and initial suspected areas on a coordinate-free cell grid (`figures/zones.png`)",
            "season_seed0.png": "Seed 0 season week by week and team-days (`figures/season_seed0.png`)",
            "policies.png": "The five policies compared over 20 field seeds (`figures/policies.png`)",
        }
        selected_fig = st.selectbox(
            "Select a figure to view (or expand below to see all)",
            options=[p.name for p in png_files],
            format_func=lambda name: fig_captions.get(name, f"figures/{name}"),
        )
        chosen_fig_path = figures_dir / selected_fig
        st.image(str(chosen_fig_path), caption=fig_captions.get(selected_fig, selected_fig), width="stretch")

        with st.expander("Show all notebook figures together"):
            for p in png_files:
                st.markdown(f"**`figures/{p.name}`**")
                st.image(str(p), caption=fig_captions.get(p.name, p.name), width="stretch")

    # 8. Model cards, risk register, and datasheets
    st.divider()
    st.subheader("8. Model cards, risk register, and datasheets (`docs/`)")
    docs_dir = root / "docs"
    model_cards_dir = docs_dir / "model_cards"
    risk_register_path = docs_dir / "risk_register.md"
    datasheets_dir = docs_dir / "datasheets"

    mc_files = sorted(model_cards_dir.glob("*.md")) if model_cards_dir.is_dir() else []
    ds_files = sorted(datasheets_dir.glob("*.md")) if datasheets_dir.is_dir() else []

    doc_entries: list[tuple[str, Path]] = []
    if risk_register_path.exists():
        doc_entries.append(("Risk register (docs/risk_register.md)", risk_register_path))
    for p in mc_files:
        doc_entries.append((f"Model card: {p.name} (docs/model_cards/{p.name})", p))
    for p in ds_files:
        doc_entries.append((f"Datasheet: {p.name} (docs/datasheets/{p.name})", p))

    if not risk_register_path.exists():
        st.caption("- `docs/risk_register.md`: not yet present in `docs/` (will appear here automatically when added).")
    if not mc_files:
        st.caption("- `docs/model_cards/*.md`: not yet present in `docs/model_cards/` (will appear here automatically when added).")

    if doc_entries:
        labels = [label for label, _ in doc_entries]
        chosen_doc_label = st.selectbox("Select a documentation file to read", options=labels)
        chosen_doc_path = dict(doc_entries)[chosen_doc_label]
        with st.container(border=True):
            st.caption(f"Path: `{chosen_doc_path.relative_to(root).as_posix()}`")
            st.markdown(chosen_doc_path.read_text(encoding="utf-8"))
