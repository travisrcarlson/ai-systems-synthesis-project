"""The season: the twin's monthly planning loop and weekly operations loop, run against the simulated field.

Each week:
1. Feeds: the access feed (every zone's status), operators' notices about their teams, consultations held, and any
   incident. The twin's responses are decisions: tasks in a zone that has closed, or of a team stood down, are
   paused; a zone declared free where an incident is reported is reopened.
2. Planning: at the start of each month, and in any week when a feed changed the teams or the zones of the plan,
   the risk overlay's chances and the twin's state go to the planner; a person approves its plan (the season's
   sign-off policy: approved as proposed) and the consultations it needs are requested. The forecaster forecasts the
   plan in the weeks set.
3. Each working day: a team short of work gets its next task from the plan's lists (the operations officer's
   order); every team works its task unless rain or wind stops work; a finished task is reported at once, and the
   report goes through the report checker into the twin; a visit's dossier goes to the dossier reviewer, and the
   recommendation to sign-off; results of technical survey and clearance are signed off.
4. End of the week: each team reports the week on its task still running; held reports sent again are checked;
   zones whose flagged areas are all resolved are declared free, and municipalities whose zones all are; the
   situation report is drafted.

The teams' own state (the task each is working) is kept apart from the twin's knowledge of it, so a report that is
lost or late leaves the twin behind, as it would in the field. Each person's decision follows a stated sign-off
policy (ROLES); the twin's checks apply to every one.
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import agent, rules
from .checker import ReportChecker
from .config import PILOT
from .core import RELEASED, TASK_KINDS
from .forecast import CompletionForecaster
from .planner import SeasonPlanner
from .prioritise import RiskOverlay
from .review import ReferenceReviewer

ROLES = {
    "plan": "departmental office (season sign-off policy: approves the plan as proposed)",
    "ops": "operations officer (season sign-off policy: orders the next task in the plan)",
    "qc": "QC reviewer (season sign-off policy: signs off what passes the rule check)",
    "na": "national authority (season sign-off policy: declares free what passes the twin's check)",
}
WORK_KIND = {"survey": ("ZONE_SURVEY", "VISIT"), "clearance": ("CLEARANCE",), "ts": ("TS",)}


@dataclass(frozen=True)
class Policy:
    """How the season is planned: the planner, the risk ranking it plans with, and whether technical survey is used."""
    planner: str = "milp"            # "milp" (the integer programme) or "rule" (the rule of thumb)
    ranking: str = "gated"           # "gated" (the baseline-first gate) or "baseline" (nearness to past accidents)
    technical_survey: bool = False   # False: Colombia's practice; True: 2 of the 4 clearance teams do technical survey
    learn: bool = True               # the risk overlay learns from the season's field results, month by month

    def label(self):
        parts = []
        if not self.learn:
            parts.append("no learning in the season")
        if self.planner != "milp":
            parts.append("rule-of-thumb plan")
        if self.ranking != "gated":
            parts.append({"baseline": "risk by nearness to past accidents", "lr": "risk by logistic regression",
                          "hgb": "risk by gradient-boosted trees", "random": "risk in random order"}[self.ranking])
        if self.technical_survey:
            parts.append("technical survey first")
        return "; ".join(parts) if parts else "main policy"


class NoScenario:
    """A season with nothing injected. Scenarios (scenarios.py) override these hooks."""
    name = "none"

    def reset(self, run):
        pass

    def access(self, run, week, statuses):
        return statuses

    def before_week(self, run, week):
        return []

    def filter(self, run, report):
        return [report]

    def after_week(self, run, week):
        return []

    def measure(self, run, result):
        return {}


class SeasonRun:
    """One season of the twin's two loops, under a policy, a reviewer and a scenario."""

    def __init__(self, twin, field, models, factors, policy=None, reviewer=None, scenario=None, checker=None,
                 sitrep=None, forecast_weeks=(1,), forecast_draws=200, planner_options=None, verbose=False,
                 keep_sessions=False):
        self.twin, self.field = twin, field
        self.keep_sessions = keep_sessions   # True: keep each dossier as the reviewer read it (agent.ReviewSession)
        self.policy = policy or Policy()
        if self.policy.technical_survey != twin.criteria.technical_survey:
            raise ValueError("the twin's national criteria do not match the policy's use of technical survey")
        self.risk = RiskOverlay(models, self.policy.ranking, learn=self.policy.learn,
                                flag_mined=twin.season.flag_mined, flag_clean=twin.season.flag_clean)
        self.planner = SeasonPlanner(factors, method=self.policy.planner, **(planner_options or {}))
        self.forecaster = CompletionForecaster(factors, draws=forecast_draws)
        self.reference = ReferenceReviewer(field)
        self.reviewer = reviewer or self.reference
        self.scenario = scenario or NoScenario()
        self.checker = checker or ReportChecker()
        self.sitrep = sitrep
        self.forecast_weeks = set(forecast_weeks or ())
        self.verbose = verbose

    # ---- the season -------------------------------------------------------------------------------------------------
    def run(self):
        twin, field = self.twin, self.field
        twin.reset()
        field.reset()
        self.checker.reset()
        self.physical = {t: None for t in twin.teams.index}   # the task each team is working, as the team knows it
        self.tally = {}            # (team, task) -> this week's days: worked, weather, access
        self.idle = {}             # week -> team-days with no task
        self.idle_by_kind = {}     # kind of team -> team-days with no task
        self.idle_by_team = {}     # team -> week -> days with no task
        self.week_days = {}        # week -> team-days worked, lost to weather, lost to access
        self.checker_counts = {}   # week -> reports accepted, held and queried
        self.check_rows = []       # every check: the report's kind and team, the action, the problems
        self.held_reports = []     # reports held by the checker, to be sent again the next week
        self.lost_items = []       # items lost in transmission, to be sent again
        self.field_visits, self.field_items = {}, {}
        self.reviews = []          # every dossier review: the recommendation, the reference's, the decision
        self.sessions = {}         # (week, area, visit) -> the dossier as the reviewer read it, if keep_sessions
        self.holds = {}            # area -> the week its dossier was first held
        self.plan_recs, self.forecasts, self.sitreps = [], [], []
        self.latest_forecast = None
        self.replans = []          # (week, reason)
        self.insufficient = []     # (week, overlay, reason)
        self.events = []
        self.queried, self.stale = set(), {}
        self.timings = []
        self.scenario.reset(self)
        self.risk_rec = self.risk.recommend(twin.view(), 1)
        if not self.risk_rec.ok:
            raise RuntimeError(f"the risk overlay cannot answer: {self.risk_rec.reason}")
        for week in range(1, twin.season.weeks + 1):
            self.feeds(week)
            self.plan_step(week)
            for day in range(1, twin.season.days_per_week + 1):
                self.work_day(week, day)
            self.end_week(week)
            if self.verbose and week % 13 == 0:
                print(f"week {week}: {self.status_line()}", flush=True)
        return SeasonResult(self)

    def status_line(self):
        s, z = self.twin.status, self.twin.zone_state
        return (f"{int((z['surveyed_week'] > 0).sum())} zones surveyed, {int((z['free_week'] > 0).sum())} free, "
                f"{int(s.isin(RELEASED).sum())} areas released, {int((s == 'cleared').sum())} cleared")

    # ---- 1. feeds ---------------------------------------------------------------------------------------------------
    def plan_zones(self):
        plan = self.twin.plan or {}
        return set(plan.get("survey", [])) | set(plan.get("clearance", [])) | set(plan.get("ts", []))

    def feeds(self, week):
        twin, field = self.twin, self.field
        self.events = []
        statuses = {z: ("closed" if field.is_closed(z, week) else "open") for z in twin.zones.index}
        statuses = self.scenario.access(self, week, statuses)
        changed = [z for z, s in statuses.items() if s != twin.access.loc[z, "status"]]
        twin.receive({"kind": "access", "week": week, "zones": statuses, "changed": changed,
                      "source": "security coordination feed (simulated)"})
        planned = self.plan_zones()
        for z in changed:
            if statuses[z] == "closed":
                self.pause_zone(z, week, f"{z} closed by the access feed")
            if z in planned:
                self.events.append(f"{z} {statuses[z]}")
        c = twin.consultation
        due = c.index[(c["requested_week"] > 0) & (c["held_week"] == 0)
                      & (c["requested_week"] + twin.season.consultation_weeks <= week)]
        for z in due:
            twin.receive({"kind": "consultation", "week": week, "zone": z,
                          "source": "municipal liaison (simulated)"})
        for report in self.scenario.before_week(self, week):
            self.feed_report(report, week)

    def feed_report(self, report, week):
        """A report from outside the teams' weekly work (an operator's notice, an incident), and the response to it.
        Keys that start with "_" are the simulation's own (an incident's true class) and never reach the twin."""
        twin = self.twin
        twin.receive({k: v for k, v in report.items() if not k.startswith("_")}, week)
        if report["kind"] == "team status":
            team = report["team"]
            if report["status"] != "active":
                task_id = twin.team_state.loc[team, "task_id"]
                if task_id is not None:
                    twin.pause_task(task_id, week=week, decided_by=ROLES["ops"], recommended_by=report["source"],
                                    reason=f"team {team} stood down")
                self.physical[team] = None
            self.events.append(f"team {team} {report['status']}")
        elif report["kind"] == "incident":
            a = report["area"]
            item = twin.evidence[a][-1]
            self.field.key[(a, item.item_id)] = report["_key"]
            z = twin.areas.loc[a, "zone_id"]
            if twin.zone_state.loc[z, "free_week"]:
                twin.reopen_zone(z, a, week=week, decided_by=ROLES["na"], recommended_by=report["source"],
                                 reason="an incident reported in a zone declared free")
            elif twin.status[a] in RELEASED or twin.status[a] in ("unknown", "not_suspected"):
                twin.reopen_area(a, week=week, decided_by=ROLES["qc"], recommended_by=report["source"],
                                 reason="an incident reported in the area")
            self.events.append(f"incident in {z}")

    def pause_zone(self, zone, week, reason):
        twin = self.twin
        for t in list(twin.tasks.values()):
            if t["zone"] == zone and t["status"] in ("ordered", "in progress"):
                twin.pause_task(t["task_id"], week=week, decided_by=ROLES["ops"], recommended_by="access feed",
                                reason=reason)
                if self.physical.get(t["team"]) == t["task_id"]:
                    self.physical[t["team"]] = None

    # ---- 2. planning ------------------------------------------------------------------------------------------------
    def plan_step(self, week):
        twin, cal = self.twin, self.twin.calendar
        reasons = []
        if week == 1:
            reasons.append("the season's first plan")
        elif cal.is_month_start(week):
            reasons.append("the monthly plan")
        if self.events:
            reasons.append("changes in the feeds: " + ", ".join(self.events))
        if not reasons:
            return
        reason = "; ".join(reasons)
        started = time.perf_counter()
        if week > 1 and self.policy.learn:
            self.risk_rec = self.risk.recommend(twin.view(), week)       # the chances, updated from the season so far
        rec = self.planner.recommend(twin.view(), week, risk=self.risk_rec)
        self.timings.append(("plan", week, time.perf_counter() - started))
        self.plan_recs.append(rec)
        if not rec.ok:
            self.insufficient.append((week, self.planner.name, rec.reason))
            return
        twin.approve_plan(rec.answer, week=week, decided_by=ROLES["plan"], recommended_by=self.planner.name,
                          overlays=(self.risk.name, self.planner.name), reason=reason)
        self.replans.append((week, reason))
        for z in rec.answer["consultations"]:
            if twin.consultation.loc[z, "required"] and not twin.consultation.loc[z, "requested_week"]:
                twin.request_consultation(z, week=week, decided_by=ROLES["plan"], recommended_by=self.planner.name,
                                          reason="a planned zone where communities must first be consulted")
        if week in self.forecast_weeks:
            started = time.perf_counter()
            f = self.forecaster.recommend(twin.view(), week, plan=twin.plan, risk=self.risk_rec)
            self.timings.append(("forecast", week, time.perf_counter() - started))
            self.forecasts.append(f)
            self.latest_forecast = f
            if not f.ok:
                self.insufficient.append((week, self.forecaster.name, f.reason))

    # ---- 3. the working day ----------------------------------------------------------------------------------------------
    def work_day(self, week, day):
        twin, field = self.twin, self.field
        self.dispatch(week)
        for team in twin.teams.index:
            if twin.team_state.loc[team, "status"] != "active":
                continue
            task_id = self.physical[team]
            if task_id is None:
                self.idle[week] = self.idle.get(week, 0) + 1
                kind = twin.teams.loc[team, "kind"]
                self.idle_by_kind[kind] = self.idle_by_kind.get(kind, 0) + 1
                by_week = self.idle_by_team.setdefault(team, {})
                by_week[week] = by_week.get(week, 0) + 1
                continue
            task = twin.tasks[task_id]
            outcome = field.work_day(task, week, day)
            tally = self.tally.setdefault((team, task_id), {"worked": 0, "weather": 0, "access": 0})
            tally["worked" if outcome in ("worked", "complete") else outcome] += 1
            if outcome == "complete":
                self.complete(team, task, week, day)

    def dispatch(self, week):
        """Give every active team the twin shows without a task its next task, from the plan's lists."""
        twin = self.twin
        plan = twin.plan or {"survey": [], "clearance": [], "ts": [], "fallback": list(twin.zones.index)}
        state = twin.team_state
        idle = [t for t in twin.teams.index if state.loc[t, "status"] == "active" and state.loc[t, "task_id"] is None
                and self.physical[t] is None]
        if not idle:
            return
        work = self.work_index()
        survey_busy = {t["zone"] for t in twin.tasks.values()
                       if t["status"] in ("ordered", "in progress") and TASK_KINDS[t["kind"]] == "survey"}
        priority = plan.get("priority")
        for team in idle:
            kind = twin.teams.loc[team, "kind"]
            listed = plan.get(kind, [])
            order = listed + [z for z in plan.get("fallback", []) if z not in set(listed)]
            for tier, z in ((("plan" if i < len(listed) else "fallback"), z) for i, z in enumerate(order)):
                if z not in work[kind]:
                    continue
                if kind == "survey" and z in survey_busy:
                    continue
                if not twin.zone_ready_for_tasking(z)[0] or twin.zone_state.loc[z, "free_week"]:
                    continue
                task_id = self.order_in_zone(team, kind, z, week, priority, tier)
                if task_id is None:
                    continue
                if kind == "survey":
                    survey_busy.add(z)
                work = self.work_index()
                break

    def work_index(self):
        """The zones with work open for each kind of team (not yet ordered), or a paused task to resume."""
        twin = self.twin
        taken = {t["area"] or t["zone"] for t in twin.tasks.values() if t["status"] in ("ordered", "in progress",
                                                                                         "paused")}
        zone_of = twin.areas["zone_id"]
        out = {"survey": set(), "clearance": set(), "ts": set()}
        zs = twin.zone_state
        out["survey"] |= {z for z in zs.index[(zs["surveyed_week"] == 0) & (zs["free_week"] == 0)] if z not in taken}
        nt = twin.next_task
        for want, kind in (("NTS", "survey"), ("CLEARANCE", "clearance"), ("TS", "ts")):
            areas = nt.index[nt == want]
            out[kind] |= {zone_of[a] for a in areas if a not in taken and a not in twin.pending_results}
        for t in twin.tasks.values():
            if t["status"] == "paused":
                out[TASK_KINDS[t["kind"]]].add(t["zone"])
        return out

    def order_in_zone(self, team, kind, zone, week, priority, tier):
        """Order the team's next task in a zone: a paused task first; then the zone survey; then the areas in order of
        the plan's priority (confirmed areas first for clearance)."""
        twin = self.twin
        why = "the plan's next zone for this kind of team" if tier == "plan" else \
            "no work left in the plan's list: the zone with the most expected mine areas per working day"
        paused = twin.paused_work(zone, kind)
        if paused:
            t = twin.tasks[paused[0]]
            task_id = twin.order_task(team, t["kind"], zone, t["area"], week=week, decided_by=ROLES["ops"],
                                      recommended_by=f"plan {twin.plan['plan_id']}" if twin.plan else None,
                                      reason=why, resume=paused[0])
            self.physical[team] = task_id
            return task_id
        options = twin.open_work(zone, kind)
        if not options:
            return None
        if options[0][0] == "ZONE_SURVEY":
            task_kind, area = options[0]
        else:
            def rank(option):
                a = option[1]
                confirmed = twin.status[a] == "confirmed"
                chance = priority.loc[a, "chance"] if priority is not None else 0.0
                zone_rank = priority.loc[a, "zone_rank"] if priority is not None else 0
                return (not confirmed, -chance, zone_rank) if kind != "survey" else (zone_rank,)
            task_kind, area = min(options, key=rank)
        task_id = twin.order_task(team, task_kind, zone, area, week=week, decided_by=ROLES["ops"],
                                  recommended_by=f"plan {twin.plan['plan_id']}" if twin.plan else None, reason=why)
        self.physical[team] = task_id
        return task_id

    def complete(self, team, task, week, day):
        """A team finishes its task: the field's result, reported at once."""
        field, twin = self.field, self.twin
        kind, a = task["kind"], task["area"]
        if kind == "ZONE_SURVEY":
            result = {"flagged": list(field.zone_survey(task["zone"]).items())}
        elif kind == "VISIT":
            visit = self.field_visits.get(a, int(twin.visits[a])) + 1
            self.field_visits[a] = visit
            start = max(twin.highest_item(a), self.field_items.get(a, 0)) + 1
            items = field.nts(a, visit, week, start)
            self.field_items[a] = start + len(items) - 1
            result = {"items": items, "declared_items": len(items)}
        elif kind == "TS":
            result = {"found": bool(field.technical_survey(a))}
        else:
            result = {"mine_cells": int(field.clearance(a))}
        tally = self.tally.pop((team, task["task_id"]))
        self.physical[team] = None
        report = {"kind": "team report", "week": week, "day": day, "team": team, "task_id": task["task_id"],
                  "zone": task["zone"], "area": a, "task": kind, "days_worked": tally["worked"],
                  "days_lost_weather": tally["weather"], "days_lost_access": tally["access"], "complete": True,
                  "result": result, "source": f"team {team}"}
        self.add_days(week, tally)
        self.deliver(report, week)

    def add_days(self, week, tally):
        d = self.week_days.setdefault(week, {"worked": 0, "weather": 0, "access": 0})
        for k in d:
            d[k] += tally[k]

    # ---- reports into the twin ---------------------------------------------------------------------------------------
    def deliver(self, report, week):
        for r in self.scenario.filter(self, report):
            self.receive_checked(r, week)

    def receive_checked(self, report, week):
        twin = self.twin
        check = self.checker.check(twin.view(), report)
        counts = self.checker_counts.setdefault(week, {"accept": 0, "hold": 0, "query": 0})
        counts[check["action"]] += 1
        self.check_rows.append({"week": week, "team": report.get("team"), "task": report.get("task"),
                                "complete": report.get("complete"), "action": check["action"],
                                "problems": "; ".join(check["problems"]), "flags": "; ".join(check["flags"]),
                                "anomaly": check["anomaly"], "score": check["score"],
                                "fault": report.get("_fault", ""), "features": check["features"]})
        if check["action"] == "hold":
            twin.decide("report", f"{report.get('team')} week {report.get('week')}", "hold and ask the team to send "
                        "it again", week=week, decided_by=ROLES["ops"], recommended_by=self.checker.name,
                        reason="; ".join(check["problems"]))
            if report.get("_clean") is not None:
                self.held_reports.append(report["_clean"])
            return
        clean = {k: v for k, v in report.items() if not k.startswith("_")}
        twin.receive(clean, week)
        if check["action"] == "query":
            why = "; ".join(check["flags"]) or f"unusual report (anomaly score {check['score']:.3f})"
            twin.decide("report", f"{report['team']} week {report['week']}", "query the team", week=week,
                        decided_by=ROLES["ops"], recommended_by=self.checker.name, reason=why)
        self.after_report(clean, week)

    def after_report(self, report, week):
        twin = self.twin
        if not report.get("complete"):
            return
        task = twin.tasks[report["task_id"]]
        a = task["area"]
        if task["kind"] == "VISIT":
            self.review(a, week)
        elif task["kind"] == "TS":
            found = report["result"]["found"]
            twin.sign_off(a, "TS_FOUND" if found else "TS_NOTHING", week=week, decided_by=ROLES["qc"],
                          recommended_by=f"technical survey team {report['team']}",
                          reason="technical survey found mines" if found else "technical survey found nothing")
        elif task["kind"] == "CLEARANCE":
            n = report["result"]["mine_cells"]
            twin.sign_off(a, "CLEARED", week=week, decided_by=ROLES["qc"],
                          recommended_by=f"clearance team {report['team']}",
                          reason=f"{n} mine cell(s) found and cleared" if n else "clearance found no mines")

    def review(self, area, week):
        """The dossier reviewer's recommendation on an area after a visit, and its sign-off."""
        twin, field = self.twin, self.field
        complete, problems = twin.dossier_complete(area)
        if not complete:
            if area not in self.holds:
                self.holds[area] = week
                twin.hold_dossier(area, week=week, decided_by=ROLES["qc"], reason="; ".join(problems))
            return
        held_since = self.holds.pop(area, None)
        items = list(twin.evidence[area])
        if self.keep_sessions:
            self.sessions[(week, area, int(twin.visits[area]))] = agent.ReviewSession(twin.view(), area, week)
        review = self.reviewer.review(twin.view(), area, week)
        reference, _, _ = rules.reference_decision(items, field.key_for(area, items), twin.criteria)
        decision, reason = review.outcome, ""
        vetoed = review.rule_check["result"] != "pass"
        if vetoed:
            decision = "FURTHER_NTS"
            reason = ("The rule check vetoes the recommendation on its own register, so it is not signed off; the "
                      "area goes back for more survey.")
        twin.sign_off(area, decision, week=week, decided_by=ROLES["qc"], recommended_by=review.by,
                      recommendation=review.outcome, register=review.register, reason=reason,
                      overlays=(self.reviewer.name,))
        self.reviews.append({"week": week, "area_id": area, "zone_id": twin.areas.loc[area, "zone_id"],
                             "municipality": twin.areas.loc[area, "municipality"],
                             "visit": int(twin.visits[area]), "mined": bool(field.mined[area]),
                             "recorded": bool(twin.areas.loc[area, "event_inside"]),
                             "branch": field.branches.get((area, int(twin.visits[area])), ""),
                             "outcome": review.outcome, "decision": decision, "reference": reference,
                             "agrees": review.outcome == reference, "fallback": review.fallback,
                             "confidence": review.confidence, "vetoed at sign-off": vetoed,
                             "vetoes": len(review.record["vetoes"]) if review.record else 0,
                             "held since": held_since})

    # ---- 4. end of the week -----------------------------------------------------------------------------------------
    def end_week(self, week):
        twin = self.twin
        resend, self.held_reports = self.held_reports, []
        for report in resend:
            self.receive_checked({**report, "_resent": True}, week)
        # items lost in transmission arrive the week after their report, when the team sends them again
        due = [entry for entry in self.lost_items if entry[2] < week]
        self.lost_items = [entry for entry in self.lost_items if entry[2] >= week]
        for area, items, _ in due:
            twin.receive({"kind": "evidence", "week": week, "area": area, "items": items,
                          "source": "the survey team, sending again"}, week)
            if not twin.next_task[area] and twin.status[area] not in RELEASED:
                self.review(area, week)
        for (team, task_id), tally in list(self.tally.items()):
            task = twin.tasks[task_id]
            report = {"kind": "team report", "week": week, "team": team, "task_id": task_id, "zone": task["zone"],
                      "area": task["area"], "task": task["kind"], "days_worked": tally["worked"],
                      "days_lost_weather": tally["weather"], "days_lost_access": tally["access"],
                      "complete": False, "source": f"team {team}"}
            self.add_days(week, tally)
            del self.tally[(team, task_id)]
            self.deliver(report, week)
        for report in self.scenario.after_week(self, week):
            self.receive_checked(report, week)
        # a team with no report for longer than the season allows is queried, once while it stays silent
        teams = twin.team_table(week)
        for team in teams.index[teams["stale"]]:
            if team not in self.queried:
                self.queried.add(team)
                twin.decide("report", team, "query the team", week=week, decided_by=ROLES["ops"],
                            recommended_by="the twin's staleness check",
                            reason=f"no report for {int(teams.loc[team, 'weeks_since_report'])} weeks")
        self.queried &= set(teams.index[teams["stale"]])
        self.stale[week] = sorted(teams.index[teams["stale"]])
        self.declarations(week)
        if self.sitrep is not None:
            rec = self.sitrep.recommend(twin.view(), week, run=self)
            self.sitreps.append(rec)

    def declarations(self, week):
        twin = self.twin
        zs = twin.zone_state
        for z in zs.index[(zs["surveyed_week"] > 0) & (zs["free_week"] == 0)]:
            open_areas, open_tasks = twin.zone_unresolved(z)
            if not open_areas and not open_tasks:
                twin.declare_zone_free(z, week=week, decided_by=ROLES["na"],
                                       recommended_by="the twin's check: surveyed, every flagged area released",
                                       reason="Travis's setting (question 39): free once every flagged area is resolved")
        for code in PILOT:
            zones = twin.zones.index[twin.zones["municipality_code"] == code]
            if not twin.municipality_free[code] and (zs.loc[zones, "free_week"] > 0).all():
                twin.declare_municipality_free(code, week=week, decided_by=ROLES["na"],
                                               recommended_by="the twin's check: every zone declared free")


class SeasonResult:
    """What a season did, and how well, measured against the field's truth (RELand's labels and any injected
    contamination)."""

    def __init__(self, run):
        twin, field = run.twin, run.field
        self.policy = run.policy
        self.scenario = run.scenario.name
        self.status = twin.status.copy()
        self.log = twin.log_table()
        self.tasks = twin.task_table()
        self.reviews = pd.DataFrame(run.reviews)
        self.sessions = dict(run.sessions)
        # plans the solver stopped at its time limit (status 1), which would depend on the computer's speed
        self.solver_limited = sum(1 for r in run.plan_recs if r.ok and r.answer.get("solver", {}).get("status") == 1)
        self.checks = pd.DataFrame(run.check_rows)
        self.zone_state = twin.zone_state.copy()
        self.municipality_free = twin.municipality_free.copy()
        self.mined = field.mined.copy()
        self.base_mined = field.base_mined.copy()
        self.flag = twin.flag.copy()
        self.zone_of = twin.areas["zone_id"]
        self.km2 = twin.areas["km2"]
        self.replans = list(run.replans)
        self.insufficient = list(run.insufficient)
        self.forecasts = list(run.forecasts)
        self.week_days = dict(run.week_days)
        self.idle = dict(run.idle)
        self.idle_by_kind = dict(run.idle_by_kind)
        self.idle_by_team = {team: dict(weeks) for team, weeks in run.idle_by_team.items()}
        self.plans = [{k: v for k, v in p.items() if k not in ("workloads", "priority")} for p in twin.plans]
        self.sitreps = list(run.sitreps)
        self.metrics = self.measure(twin)
        self.metrics.update(run.scenario.measure(run, self))

    def measure(self, twin):
        s, mined, km2 = self.status, self.mined, self.km2
        released = s.isin(RELEASED)
        free_zones = self.zone_state.index[self.zone_state["free_week"] > 0]
        in_free = self.zone_of.isin(free_zones)
        surveyed = self.zone_of.isin(self.zone_state.index[self.zone_state["surveyed_week"] > 0])
        t = self.tasks
        done = t[t["status"] == "done"] if len(t) else t
        clear = done[done["kind"] == "CLEARANCE"] if len(done) else done
        ts = done[done["kind"] == "TS"] if len(done) else done
        clear_clean = clear[~mined[clear["area"]].to_numpy()] if len(clear) else clear
        ts_clean = ts[~mined[ts["area"]].to_numpy()] if len(ts) else ts
        days = {k: sum(w.get(k, 0) for w in self.week_days.values()) for k in ("worked", "weather", "access")}
        decided = self.log[(self.log["entry"] == "decision")] if len(self.log) else self.log
        area_dec = decided[decided["kind"] == "area"] if len(decided) else decided
        reviewed = area_dec[area_dec["decision"].isin(["CANCEL", "TECHNICAL_SURVEY", "CLEARANCE", "FURTHER_NTS"])]
        r = self.reviews
        return {
            "zones surveyed": int((self.zone_state["surveyed_week"] > 0).sum()),
            "zones declared free": int(len(free_zones)),
            "km² in zones declared free": float(km2[in_free].sum()),
            "municipalities declared free": int((self.municipality_free > 0).sum()),
            "areas flagged": int((self.flag != "").sum()),
            "visits": int((done["kind"] == "VISIT").sum()) if len(done) else 0,
            "mine areas in surveyed zones": int((mined & surveyed).sum()),
            "mine areas found": int((self.status.isin(["confirmed", "cleared"]) & mined).sum()),
            "mine areas cleared": int((s.eq("cleared") & mined).sum()),
            "areas released": int(released.sum()),
            "cancelled": int(s.eq("cancelled").sum()), "reduced": int(s.eq("reduced").sum()),
            "cleared": int(s.eq("cleared").sum()),
            "released with mines (critical)": int((s.isin(["cancelled", "reduced"]) & mined).sum()),
            "cancelled with mines": int((s.eq("cancelled") & mined).sum()),
            "reduced with mines": int((s.eq("reduced") & mined).sum()),
            "mine areas not flagged, in zones declared free (critical)": int((s.eq("not_suspected") & mined
                                                                             & in_free).sum()),
            "mine areas not flagged, in surveyed zones": int((s.eq("not_suspected") & mined & surveyed).sum()),
            "clearances": int(len(clear)), "clearances on land without mines": int(len(clear_clean)),
            "share of clearance days on land without mines": float(clear_clean["days_done"].sum()
                                                                   / max(1, clear["days_done"].sum()))
            if len(clear) else 0.0,
            "technical surveys": int(len(ts)), "technical surveys on land without mines": int(len(ts_clean)),
            "team-days worked": days["worked"], "team-days lost to weather": days["weather"],
            "team-days lost to access": days["access"], "team-days with no task": int(sum(self.idle.values())),
            **{f"team-days with no task ({k} teams)": int(v) for k, v in sorted(self.idle_by_kind.items())},
            "plans approved": len(self.replans),
            "plans stopped by the solver's time limit": int(self.solver_limited),
            "plans between the monthly plans, after changes in the feeds": sum(
                1 for w, _ in self.replans if w > 1 and not twin.calendar.is_month_start(w)),
            "review sign-offs": int(len(reviewed)),
            "sign-offs that differ from the recommendation": int((r["outcome"] != r["decision"]).sum()) if len(r)
            else 0,
            "vetoed at sign-off": int(r["vetoed at sign-off"].sum()) if len(r) else 0,
            "reviews agreeing with the reference": int(r["agrees"].sum()) if len(r) else 0,
            "fallbacks": int(r["fallback"].sum()) if len(r) else 0,
            "dossiers held": int((decided["kind"] == "dossier").sum()) if len(decided) else 0,
            "reports held": int((self.checks["action"] == "hold").sum()) if len(self.checks) else 0,
            "reports held with no fault injected": int(((self.checks["action"] == "hold")
                                                        & (self.checks["fault"] == "")).sum()) if len(self.checks) else 0,
            "reports queried": int((self.checks["action"] == "query").sum()) if len(self.checks) else 0,
        }

    def forecast_coverage(self, i=0):
        """For the season's forecast `i` (0, the first): each zone's forecast against when it was declared free (if it
        was)."""
        if len(self.forecasts) <= i or not self.forecasts[i].ok:
            return None
        f = self.forecasts[i].answer["zones"].copy()
        actual = self.zone_state["free_week"].reindex(f.index)
        f["declared free (week)"] = actual.where(actual > 0)
        f["by P50"] = f["declared free (week)"].le(f["P50 week"].fillna(np.inf))
        f["by P90"] = f["declared free (week)"].le(f["P90 week"].fillna(np.inf))
        return f


def summarise(results):
    """One row per season result: its policy, its scenario and its measures."""
    return pd.DataFrame([{"policy": r.policy.label(), "scenario": r.scenario, **r.metrics} for r in results])
