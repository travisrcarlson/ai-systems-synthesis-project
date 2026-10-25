"""The five injected scenarios: realistic things that go wrong in a season, and whether the twin notices, how fast,
and what it costs (P5's insight: synthetic data serve stress tests).

1. Funding shock: as HALO in 2025, which lost "up to two-thirds of clearance teams and half of NTS teams" (Mine
   Action Review, 2025), 3 of 4 clearance teams and 2 of 4 survey teams are stood down for three months from month
   4 (Travis's setting, question 35); their operator expects them back in month 7.
2. Zone closure: the zone in progress with the most open work closes mid-season for three months (question 36).
3. New contamination: an accident is reported in the first zone declared free, four weeks after its declaration.
4. Stale or corrupted feed: one survey team's reports are lost for three weeks and arrive late; one clearance
   team's reports are corrupted for four weeks (impossible days, a duplicate, a wrong task, inflated days).
5. Dossier with a missing item: a visit report with several items loses its item of direct evidence in
   transmission; and the desk study's record of a past accident fails to load into another area's dossier.
Each scenario's measures are in `measure`; what it costs is measured against the season with the same seed and no
scenario (evaluate.py).
"""
import numpy as np

from .core import desk_item
from .rules import reference_decision
from .season import NoScenario


def decisions(result, kind=None, subject=None):
    log = result.log
    d = log[log["entry"] == "decision"]
    if kind is not None:
        d = d[d["kind"] == kind]
    if subject is not None:
        d = d[d["subject"] == subject]
    return d


class FundingShock(NoScenario):
    name = "funding shock"

    def __init__(self, start_month=4, months=3, clearance_lost=3, survey_lost=2):
        self.start_month, self.months = start_month, months
        self.clearance_lost, self.survey_lost = clearance_lost, survey_lost

    def reset(self, run):
        cal = run.twin.calendar
        self.start = cal.first_week(self.start_month)
        self.back = cal.first_week(self.start_month + self.months)
        teams = run.twin.teams
        clearance = [t for t in teams.index if teams.loc[t, "kind"] in ("clearance", "ts")]
        survey = [t for t in teams.index if teams.loc[t, "kind"] == "survey"]
        self.lost = clearance[-self.clearance_lost:] + survey[-self.survey_lost:]

    def before_week(self, run, week):
        if week == self.start:
            return [{"kind": "team status", "week": week, "team": t, "status": "stood down", "back_week": self.back,
                     "source": "operator's notice (funding suspended)"} for t in self.lost]
        if week == self.back:
            return [{"kind": "team status", "week": week, "team": t, "status": "active",
                     "source": "operator's notice (funding restored)"} for t in self.lost]
        return []

    def measure(self, run, result):
        replans = [w for w, reason in result.replans if w >= self.start and "stood down" in reason]
        paused = decisions(result, "pause")
        paused = paused[(paused["week"] == self.start) & paused["reason"].str.contains("stood down")]
        return {"scenario: teams stood down": len(self.lost), "scenario: noticed in week": self.start,
                "scenario: tasks paused": int(len(paused)),
                "scenario: weeks to a new plan": (replans[0] - self.start) if replans else None}


class ZoneClosure(NoScenario):
    name = "zone closure"

    def __init__(self, week=20, weeks=13):
        self.week, self.weeks = week, weeks

    def reset(self, run):
        self.zone = None

    def access(self, run, week, statuses):
        if week == self.week and self.zone is None:
            twin = run.twin
            open_work = {}
            for t in twin.tasks.values():
                if t["status"] in ("ordered", "in progress"):
                    open_work[t["zone"]] = open_work.get(t["zone"], 0) + 1
            if open_work:
                self.zone = max(sorted(open_work), key=open_work.get)
            elif twin.plan:
                self.zone = twin.plan["survey"][0]
        if self.zone is not None and self.week <= week < self.week + self.weeks:
            statuses = dict(statuses)
            statuses[self.zone] = "closed"
        return statuses

    def measure(self, run, result):
        if self.zone is None:
            return {"scenario: zone closed": None}
        paused = decisions(result, "pause")
        paused = paused[(paused["week"] == self.week) & paused["reason"].str.contains(self.zone)]
        replans = [w for w, reason in result.replans if w >= self.week and self.zone in reason]
        orders = decisions(result, "task order")
        orders = orders[orders["week"] >= self.week]
        moved = orders[~orders["decision"].str.contains(self.zone)]
        return {"scenario: zone closed": self.zone, "scenario: noticed in week": self.week,
                "scenario: tasks paused": int(len(paused)),
                "scenario: weeks to a new plan": (replans[0] - self.week) if replans else None,
                "scenario: orders elsewhere in the week": int((moved["week"] == self.week).sum())}


class NewContamination(NoScenario):
    name = "new contamination in a zone declared free"

    def __init__(self, delay_weeks=4):
        self.delay = delay_weeks

    def reset(self, run):
        self.area = self.zone = self.week = None

    def before_week(self, run, week):
        if self.area is not None:
            return []
        zs = run.twin.zone_state
        free = zs[zs["free_week"] > 0]
        if not len(free) or week < free["free_week"].min() + self.delay:
            return []
        zone = free["free_week"].idxmin()
        twin = run.twin
        candidates = [a for a in twin.zone_areas[zone] if twin.status[a] == "not_suspected"] or \
            list(twin.zone_areas[zone])
        rng = np.random.default_rng([run.field.seed, 99])
        self.area, self.zone, self.week = candidates[int(rng.integers(len(candidates)))], zone, week
        statement, key = run.field.new_contamination(self.area, week)
        return [{"kind": "incident", "week": week, "area": self.area, "statement": statement,
                 "source": "community leaders, through the municipal liaison", "source_id": "community leaders",
                 "_key": key}]

    def measure(self, run, result):
        if self.area is None:
            return {"scenario: incident": None}
        reopened = decisions(result, "zone", self.zone)
        reopened = reopened[reopened["decision"] == "reopen"]
        t = result.tasks
        mine = t[(t["area"] == self.area) & (t["status"] == "done")] if len(t) else t
        visit = mine[mine["kind"] == "VISIT"]["finished_week"].min() if len(mine) else None
        clear = mine[mine["kind"] == "CLEARANCE"]["finished_week"].min() if len(mine) else None
        free_again = result.zone_state.loc[self.zone, "free_week"]
        return {"scenario: incident": f"{self.area} ({self.zone}) in week {self.week}",
                "scenario: noticed in week": int(reopened["week"].min()) if len(reopened) else None,
                "scenario: weeks to the visit": (int(visit) - self.week) if visit == visit and visit else None,
                "scenario: weeks to clearance": (int(clear) - self.week) if clear == clear and clear else None,
                "scenario: zone declared free again in week": int(free_again) if free_again else None,
                "scenario: area cleared": bool(result.status[self.area] == "cleared")}


class FeedFaults(NoScenario):
    name = "stale or corrupted feed"
    FAULTS = {40: "impossible days", 41: "duplicate", 42: "wrong task", 43: "days inflated", 44: "days inflated",
              45: "days inflated", 46: "days inflated"}

    def __init__(self, stale_team="S2", stale_weeks=(20, 21, 22), corrupt_team="C2", corrupt_weeks=tuple(range(40, 47))):
        self.stale_team, self.stale_weeks = stale_team, tuple(stale_weeks)
        self.corrupt_team, self.corrupt_weeks = corrupt_team, tuple(corrupt_weeks)

    def reset(self, run):
        self.lost = []
        self.injected = []

    def filter(self, run, report):
        if report.get("kind") != "team report" or report.get("_resent"):
            return [report]
        team, week = report["team"], report["week"]
        if team == self.stale_team and week in self.stale_weeks:
            self.lost.append(report)
            return []
        if team == self.corrupt_team and week in self.corrupt_weeks:
            fault = self.FAULTS.get(week)
            if fault is None or any(w == week for _, w in self.injected):
                return [report]
            self.injected.append((fault, week))
            if fault == "impossible days":       # the week's days add up to 7 in a week of 5
                lost = 7 - report["days_worked"] - report["days_lost_access"]
                return [{**report, "days_lost_weather": lost, "_fault": fault, "_clean": report}]
            if fault == "duplicate":
                return [report, {**report, "_fault": fault}]
            if fault == "wrong task":
                return [{**report, "task_id": "K9999", "_fault": fault, "_clean": report}]
            return [{**report, "days_worked": run.twin.season.days_per_week, "days_lost_weather": 0,
                     "days_lost_access": 0, "_fault": fault}]
        return [report]

    def after_week(self, run, week):
        if week == self.stale_weeks[-1] + 1 and self.lost:
            backlog, self.lost = self.lost, []
            return [{**r, "_resent": True} for r in backlog]
        return []

    def measure(self, run, result):
        c = result.checks
        faults = c[c["fault"] != ""] if len(c) else c
        caught = faults[faults["action"].isin(["hold", "query"])] if len(faults) else faults
        normal = c[(c["fault"] == "") & (c["team"].notna())] if len(c) else c
        flagged = [w for w, teams in sorted(run.stale.items()) if w >= self.stale_weeks[0] and self.stale_team in teams]
        first_stale = flagged[0] if flagged else None
        idle = sum(result.idle.get(w, 0) for w in self.stale_weeks)
        return {"scenario: stale team flagged in week": first_stale,
                "scenario: weeks from the first lost report to the flag": (first_stale - self.stale_weeks[0])
                if first_stale else None,
                "scenario: team-days idle during the outage": idle,
                "scenario: corrupted reports": int(len(faults)), "scenario: corrupted reports caught": int(len(caught)),
                "scenario: caught": ", ".join(f"{f}: {a}" for f, a in zip(faults["fault"], faults["action"]))
                if len(faults) else "",
                "scenario: normal reports held or queried": int(normal["action"].isin(["hold", "query"]).sum())
                if len(normal) else 0}


class MissingItem(NoScenario):
    name = "dossier with a missing item"

    def __init__(self, start_week=10):
        self.start_week = start_week

    def reset(self, run):
        self.lost_area = self.lost_item = self.lost_week = self.lost_counterfactual = None
        self.desk_area = self.desk_week = self.desk_counterfactual = None
        self.reload = []

    def filter(self, run, report):
        if report.get("task") != "VISIT" or not report.get("complete") or report.get("_resent"):
            return [report]
        twin, field = run.twin, run.field
        a = report["area"]
        items = report["result"]["items"]
        # (a) an item of direct evidence lost from a visit report in transmission
        if self.lost_area is None and report["week"] >= self.start_week and len(items) >= 2:
            direct = [i for i in items if field.key[(a, i.item_id)]["class"] == "direct"]
            if direct:
                drop = direct[0]
                kept = [i for i in items if i is not drop]
                self.lost_area, self.lost_item, self.lost_week = a, drop.item_id, report["week"]
                before = list(twin.evidence[a]) + kept
                self.lost_counterfactual = reference_decision(before, field.key_for(a, before), twin.criteria)[0]
                run.lost_items.append((a, [drop], report["week"]))
                return [{**report, "result": {**report["result"], "items": kept}}]
        # (b) the desk study's record of a past accident fails to load into the dossier of a recorded area
        if self.desk_area is None and report["week"] >= self.start_week and twin.areas.loc[a, "event_inside"] \
                and any(i.source_id == "national database" for i in twin.evidence[a]) and a != self.lost_area:
            desk = [i for i in twin.evidence[a] if i.source_id == "national database"][0]
            twin.evidence[a].remove(desk)          # the injected data fault: the record did not load
            self.desk_area, self.desk_week = a, report["week"]
            after = list(twin.evidence[a]) + list(items)
            self.desk_counterfactual = reference_decision(after, field.key_for(a, after), twin.criteria)[0]
            self.reload.append((a, report["week"]))
        return [report]

    def after_week(self, run, week):
        twin = run.twin
        for a, w in list(self.reload):
            if w < week:
                self.reload.remove((a, w))
                twin.receive({"kind": "evidence", "week": week, "area": a, "items": [desk_item()],
                              "source": "national mine action database (record reloaded)"}, week)
                if twin.next_task[a] is None and twin.status[a] not in ("cancelled", "reduced", "cleared"):
                    run.review(a, week)
        return []

    def measure(self, run, result):
        out = {}
        for part, area, week, counter in (("lost item", self.lost_area, self.lost_week, self.lost_counterfactual),
                                          ("desk record", self.desk_area, self.desk_week, self.desk_counterfactual)):
            if area is None:
                out[f"scenario: {part}"] = None
                continue
            held = decisions(result, "dossier", area)
            r = result.reviews
            after = r[(r["area_id"] == area) & (r["week"] >= week)] if len(r) else r
            out[f"scenario: {part}"] = f"{area} in week {week}"
            out[f"scenario: {part} held"] = bool(len(held))
            out[f"scenario: {part} reviewed in week"] = int(after["week"].min()) if len(after) else None
            out[f"scenario: {part} decision"] = after["decision"].iloc[0] if len(after) else None
            out[f"scenario: {part} decision without the hold"] = counter
            out[f"scenario: {part} area holds mines"] = bool(result.mined[area])
        return out


SCENARIOS = {"funding shock": FundingShock, "zone closure": ZoneClosure,
             "new contamination": NewContamination, "stale or corrupted feed": FeedFaults,
             "missing item": MissingItem}
