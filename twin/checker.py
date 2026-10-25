"""The report checker: team reports that are impossible, inconsistent, incomplete or unusual, caught before they
change the twin (machine learning with coded checks; P1's insight: check data as it arrives).

Coded checks, on every team report:
- its fields are present, and its days are whole numbers from 0 to 5 that add up to at most a week;
- it reports on a task ordered to that team, still open, in that zone and area, in a week since the order;
- it is not a second copy of a report already received;
- no work is reported in a zone the access feed showed closed in the week the report covers;
- a finished task carries the result its kind needs (flagged areas in the zone; evidence items, as many as the
  report declares; a technical survey's finding; the mine cells cleared), and a technical survey or clearance is
  not reported done in under a quarter of its planned days;
- across a team's run of reports: three weeks in a row with work and no day lost to rain or wind, in months when
  about half the days are lost, is queried (an honest team reports that about once in 30,000 such runs).
An impossible or inconsistent report is held, and the team is asked to send it again; a report whose items did not
all arrive is accepted with a flag, since what arrived is real, and the dossier stays on hold until the rest
arrives (the twin's own completeness check).

The isolation forest reads every report as numbers (days worked and lost, the month's expected loss to weather, the
task's kind, its progress against the planned days, items and mine cells reported) and flags the unusual ones for
the operations officer to query. It is fitted on a season of reports with no fault injected, and says
"insufficient evidence" until it has been.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import SEED
from .overlays import Overlay

KINDS = ["ZONE_SURVEY", "VISIT", "TS", "CLEARANCE"]
FEATURES = ["days_worked", "days_lost_weather", "days_lost_access", "expected_loss", "complete", "progress",
            "items", "mine_cells", *[f"kind_{k}" for k in KINDS]]


def features(report, task, season, calendar):
    """A team report as numbers, for the isolation forest."""
    planned = season.task_days(task["kind"])
    result = report.get("result") or {}
    row = {"days_worked": report["days_worked"], "days_lost_weather": report["days_lost_weather"],
           "days_lost_access": report["days_lost_access"],
           "expected_loss": season.day_loss(calendar.month(report["week"])) * season.days_per_week,
           "complete": float(bool(report.get("complete"))),
           "progress": (task["days_done"] + report["days_worked"]) / planned,
           "items": float(len(result.get("items", []))) if task["kind"] == "VISIT" else 0.0,
           "mine_cells": float(result.get("mine_cells", 0) or 0)}
    for k in KINDS:
        row[f"kind_{k}"] = float(task["kind"] == k)
    return row


class ReportChecker(Overlay):
    name = "report checker"
    domain = "machine learning (isolation forest) with coded checks"
    insight = "P1: check data as it arrives"
    reads = ("teams", "rainfall")
    limits = ("the coded checks catch what is impossible or inconsistent; the isolation forest only flags what is "
              "unusual against a season of simulated reports, for a person to query",)

    def __init__(self, contamination=0.01, seed=SEED, min_reports=200):
        self.contamination = contamination
        self.seed = seed
        self.min_reports = min_reports
        self.model = None
        self.trained_on = 0
        self.seen = set()
        self.runs = {}

    def reset(self):
        self.seen = set()
        self.runs = {}         # team -> the weeks in a row it reported work with no day lost, in rainy months

    def fit(self, rows):
        """Fit the isolation forest on the feature rows of a season's reports with no fault injected."""
        table = pd.DataFrame(rows, columns=FEATURES)
        self.trained_on = len(table)
        if len(table) < self.min_reports:
            self.model = None
            return self
        self.model = IsolationForest(n_estimators=200, contamination=self.contamination, random_state=self.seed)
        self.model.fit(table.to_numpy())
        return self

    def check(self, view, report):
        """Check one report. Returns {"action": "accept", "hold" or "query", "problems", "flags", "anomaly",
        "score", "features"}: hold an impossible or inconsistent report; accept one with a flag; query an unusual one."""
        problems, flags = [], []
        season, calendar = view.season, view.calendar
        if report.get("kind") != "team report":
            return {"action": "accept", "problems": [], "flags": [], "anomaly": None, "score": None, "features": None}
        needed = ["week", "team", "task_id", "days_worked", "days_lost_weather", "days_lost_access", "complete"]
        missing = [k for k in needed if k not in report or report[k] is None]
        if missing:
            return self._hold([f"fields missing: {', '.join(missing)}"])
        days = [report["days_worked"], report["days_lost_weather"], report["days_lost_access"]]
        if any(not isinstance(d, (int, np.integer)) or d < 0 or d > season.days_per_week for d in days):
            problems.append("days must be whole numbers from 0 to 5")
        elif sum(days) > season.days_per_week:
            problems.append(f"{sum(days)} days reported in a week of {season.days_per_week}")
        task = view.tasks.get(report["task_id"])
        if task is None:
            return self._hold(problems + [f"no task {report['task_id']} was ordered"])
        if task["team"] != report["team"]:
            problems.append(f"{report['task_id']} was ordered to team {task['team']}, not {report['team']}")
        if task["status"] not in ("ordered", "in progress", "paused"):
            problems.append(f"{report['task_id']} is already {task['status']}")
        if report.get("zone") not in (None, task["zone"]) or report.get("area") not in (None, task["area"]):
            problems.append("the report's zone or area is not the task's")
        if report["week"] < task["ordered_week"] or report["week"] > view.week + 1:
            problems.append(f"week {report['week']} is outside the task's weeks")
        key = (report["team"], report["task_id"], report["week"], bool(report["complete"]))
        if key in self.seen:
            problems.append("a second copy of a report already received")
        if report["days_worked"]:
            # a late report is checked against the access feed of the week it reports on
            then = view.access_history.get(report["week"]) if report["week"] < view.week else None
            closed = task["zone"] in then if then is not None else view.access.loc[task["zone"], "status"] != "open"
            if closed:
                problems.append(f"work reported in {task['zone']}, which the access feed showed closed that week")
        if report["complete"]:
            result = report.get("result") or {}
            kind = task["kind"]
            if kind == "ZONE_SURVEY":
                flagged = result.get("flagged")
                if flagged is None or any(a not in view.zone_areas[task["zone"]] for a, _ in flagged):
                    problems.append("the zone survey's flagged areas are missing or not in the zone")
            elif kind == "VISIT":
                items = result.get("items")
                if not items:
                    problems.append("a visit report with no evidence items")
                elif result.get("declared_items", len(items)) != len(items):
                    flags.append(f"the report declares {result['declared_items']} item(s); {len(items)} arrived")
            elif kind == "TS" and not isinstance(result.get("found"), (bool, np.bool_)):
                problems.append("a technical survey report with no finding")
            elif kind == "CLEARANCE" and not isinstance(result.get("mine_cells"), (int, np.integer)):
                problems.append("a clearance report with no count of mine cells")
            if kind in ("TS", "CLEARANCE"):
                done = task["days_done"] + report["days_worked"]
                if done < 0.25 * season.task_days(kind):
                    problems.append(f"done in {done} working days, under a quarter of the {season.task_days(kind):.0f} "
                                    "planned")
        if problems:
            return self._hold(problems)
        self.seen.add(key)
        rainy = season.day_loss(calendar.month(report["week"])) >= 0.5
        if rainy and report["days_worked"] and not report["days_lost_weather"]:
            weeks = self.runs.setdefault(report["team"], set())
            weeks.add(report["week"])
            run_length = 1
            while report["week"] - run_length in weeks:
                run_length += 1
            if run_length >= 3:
                flags.append(f"{run_length} weeks in a row with no day lost to rain or wind in a rainy month")
        elif rainy and report["days_lost_weather"]:
            self.runs.pop(report["team"], None)
        row = features(report, task, season, calendar)
        anomaly, score = None, None
        if self.model is not None:
            x = np.array([[row[f] for f in FEATURES]])
            score = float(self.model.score_samples(x)[0])
            anomaly = bool(self.model.predict(x)[0] == -1)
        action = "query" if anomaly or any("weeks in a row" in f for f in flags) else "accept"
        return {"action": action, "problems": [], "flags": flags, "anomaly": anomaly, "score": score,
                "features": row}

    @staticmethod
    def _hold(problems):
        return {"action": "hold", "problems": problems, "flags": [], "anomaly": None, "score": None,
                "features": None}

    def recommend(self, view, week, report=None, **inputs):
        """The contract's form of `check`: the action on one report, or "insufficient evidence" for the isolation
        forest's part before it has been fitted (the coded checks always run)."""
        result = self.check(view, report)
        limits = () if self.model is not None else ("the isolation forest is not fitted yet: insufficient evidence "
                                                    "for unusual reports; the coded checks ran",)
        return self.answer(week, result, ("the report", "the twin's task orders", "access feed", "rainfall pattern"),
                           uncertainty=None if result["score"] is None else f"anomaly score {result['score']:.3f}",
                           limits=limits)
