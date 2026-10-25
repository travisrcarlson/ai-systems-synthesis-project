"""The digital twin: one state model of land release in the pilot, kept in step with the reports that come in, on
which decisions are tested before they are written back.

The state model:
- place: municipality, operational zone, area (a block of 2 x 2 RELand cells) and cell, with RELand's 63 features;
- work: the teams, their task orders and the calendar of the season (52 weeks in 12 months);
- hazard: each area's status and the evidence behind it, starting from the national database's past accidents;
- the data layers registered with it (overlays.LAYERS), each with its source, licence and sensitivity.
Every record carries when it was last reported and by whom (`as_of`, `source`), so stale data shows as stale.

The twin never holds RELand's labels. It changes in two ways only, and logs both, in order, in one log:
- reports received: team reports (progress, and the results of zone surveys, visits, technical survey and
  clearance), the access feed, operators' notices about their teams, consultations held, incidents;
- decisions approved by a person: plans, task orders, pauses, consultation requests, sign-offs on areas, zones
  and municipalities declared free, zones reopened.
Reports change what the twin knows; only decisions change the status of land or what the teams are told to do. The
twin checks each decision before it accepts it: the rule check for decisions on areas, a complete dossier before a
review is signed off, a task order only for open work in an open zone, and every flagged area resolved before a zone
is declared free. Overlays read it through `view()`, which has no method that writes.
"""
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import AREA_CELLS, CELL_KM, EVENT_INSIDE_KM, MONTHS, NAME_OF, PILOT, SEED, Criteria, Season
from .rules import failed_rules, rule_check

STATUSES = ["unknown", "not_suspected", "suspected", "confirmed", "cancelled", "reduced", "cleared"]
RELEASED = {"cancelled", "reduced", "cleared"}
TASK_KINDS = {"ZONE_SURVEY": "survey", "VISIT": "survey", "TS": "ts", "CLEARANCE": "clearance"}
TASK_WORDS = {"ZONE_SURVEY": "zone survey", "VISIT": "visit", "TS": "technical survey", "CLEARANCE": "clearance"}

# What a signed-off review does to an area: its new status (None: unchanged) and its next task (None: none)
REVIEW_OUTCOMES = {
    "CANCEL": ("cancelled", None),
    "TECHNICAL_SURVEY": ("suspected", "TS"),
    "CLEARANCE": ("suspected", "CLEARANCE"),   # confirmed instead, where the register holds direct evidence
    "FURTHER_NTS": (None, "NTS"),
}
# What a signed-off result of technical survey or clearance does to an area
RESULT_OUTCOMES = {
    "TS_FOUND": ("confirmed", "CLEARANCE"),     # technical survey confirmed mines: clearance follows
    "TS_NOTHING": ("reduced", None),            # technical survey found nothing: the area is released
    "CLEARED": ("cleared", None),               # clearance done: the area is released
}

LAND_USE = {"land_use_Agroforestal": "agroforestry", "land_use_Agrícola": "farming",
            "land_use_Conservación de Suelos": "soil conservation", "land_use_Cuerpo de agua": "water",
            "land_use_Forestal": "forest", "land_use_Ganadera": "livestock", "land_use_Zonas urbanas": "urban"}
USED_LAND = ["land_use_Agroforestal", "land_use_Agrícola", "land_use_Ganadera", "land_use_Zonas urbanas"]
ROADS = ["dist_roads_t1", "dist_roads_t2", "dist_roads_t3"]


@dataclass(frozen=True)
class EvidenceItem:
    """One item of evidence on an area, as the twin holds it and a reviewer reads it."""
    item_id: str
    statement: str
    source: str
    when: str                  # "Desk study", or the visit and week, such as "Visit 1, week 12"
    visit: int = 0             # 0 for the desk study and incident reports
    week: int = 0
    years: int | None = None   # years of safe use the statement reports, if any (read by the rule check)
    source_id: str = ""        # who it comes from, such as "informant 3" (read by the rule check to count sources)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def desk_item():
    """The desk study's record of a past landmine accident inside an area: indirect evidence (IMAS 08.10, §6.1)."""
    return EvidenceItem("E1", "The national mine action database records one or more past landmine accidents inside "
                              "this area. The twin holds no date or other detail of the record, and its location has "
                              "not been checked on the ground.",
                        "National mine action database (desk study)", "Desk study", source_id="national database")


def event_band(km):
    """Where the nearest recorded past landmine accident lies, in words."""
    if km < EVENT_INSIDE_KM:
        return "one or more are recorded inside the area"
    if km < 1:
        return "the nearest is within 1 km of the area"
    if km < 3:
        return "the nearest is 1 to 3 km from the area"
    return "the nearest is more than 3 km from the area"


def settlement_band(km):
    for limit, text in [(1, "under 1 km"), (3, "1 to 3 km"), (5, "3 to 5 km")]:
        if km < limit:
            return text
    return "more than 5 km"


def access_band(km):
    """A zone's access, by the average distance of its areas to the nearest road."""
    if km < 2:
        return "under 2 km from a road"
    if km < 5:
        return "2 to 5 km from a road"
    return "more than 5 km from a road"


class Calendar:
    """The season's weeks and months: 52 weeks of 5 working days, in months of 5, 4 and 4 weeks."""

    def __init__(self, season):
        self.season = season
        month_of, weeks_of = {}, {}
        week = 1
        for month, n in enumerate(season.month_weeks, start=1):
            weeks_of[month] = tuple(range(week, week + n))
            for w in weeks_of[month]:
                month_of[w] = month
            week += n
        self.month_of = MappingProxyType(month_of)    # week -> month (1 to 12)
        self.weeks_of = MappingProxyType(weeks_of)    # month -> its weeks

    def month(self, week):
        return self.month_of[week]

    def first_week(self, month):
        return self.weeks_of[month][0]

    def is_month_start(self, week):
        return self.first_week(self.month(week)) == week

    def label(self, week):
        return f"week {week} ({MONTHS[self.month(week) - 1]})"

    def expected_days(self, week):
        """The working days a team can expect in a week of this month, after days lost to rain and wind."""
        return self.season.days_per_week * (1 - self.season.day_loss(self.month(week)))


class Twin:
    """The twin of the pilot: its places, its season and teams, the hazard, and the one log of changes."""

    def __init__(self, cells, people, activity, criteria=None, season=None, layers=None):
        self.criteria = criteria or Criteria()
        self.season = season or Season()
        self.calendar = Calendar(self.season)
        self.cells = cells.copy()
        self.people = people          # victims by municipality and month (data.victims_by_month)
        self.activity = activity      # demining results by municipality and month (data.load_demining)
        self.layers = MappingProxyType(dict(layers or {}))   # name -> layers.Layer, registered with the twin
        self._build_areas()
        self._build_zones()
        self._build_teams()
        self.reset()

    # ---- the places ---------------------------------------------------------------------------------------------
    def _build_areas(self):
        c = self.cells
        c["area_row"], c["area_col"] = c["row"] // AREA_CELLS, c["col"] // AREA_CELLS
        c["_used"] = c[USED_LAND].sum(axis=1)
        c["_road"] = c[ROADS].min(axis=1)
        groups = c.groupby(["municipality_code", "area_row", "area_col"], sort=True)
        areas = groups.agg(cells=("row", "size"), forest_share=("land_use_Forestal", "mean"),
                           used_share=("_used", "mean"), settlement_km=("settlement_dist", "mean"),
                           road_km=("_road", "mean"), event_km=("dist_old_mine", "min"),
                           population=("population_2012", "sum"))
        areas["km2"] = areas["cells"] * CELL_KM ** 2
        areas["main_land_use"] = groups[list(LAND_USE)].sum().idxmax(axis=1).map(LAND_USE)
        areas["cell_list"] = groups.apply(lambda part: list(part.index), include_groups=False)
        areas = areas.reset_index()
        c.drop(columns=["_used", "_road"], inplace=True)
        areas["number"] = areas.groupby("municipality_code").cumcount() + 1
        areas["area_id"] = [f"{PILOT[code][2]}-{n:04d}" for code, n in zip(areas["municipality_code"], areas["number"])]
        areas["municipality"] = areas["municipality_code"].map(NAME_OF)
        areas["event_inside"] = areas["event_km"] < EVENT_INSIDE_KM
        self.areas = areas.set_index("area_id")
        self.cells["area_id"] = pd.Series({cell: a for a, cl in self.areas["cell_list"].items() for cell in cl})
        # neighbours: areas whose blocks share an edge, or the other part of a block cut by a municipal border
        where = {}
        for a, ar, ac in self.areas[["area_row", "area_col"]].itertuples():
            where.setdefault((ar, ac), []).append(a)
        neighbours = {}
        for a, ar, ac in self.areas[["area_row", "area_col"]].itertuples():
            neighbours[a] = tuple(b for key in [(ar, ac), (ar - 1, ac), (ar + 1, ac), (ar, ac - 1), (ar, ac + 1)]
                                  for b in where.get(key, []) if b != a)
        self.neighbours = MappingProxyType(neighbours)

    def _build_zones(self):
        """Group each municipality's areas into operational zones of about `zone_areas` areas, by position and
        access, since Colombia's municipalities "are subdivided into operational zones based on accessibility"
        (Mine Action Review, 2025). K-means on each area's position in km and the log of 1 + its distance in km to
        the nearest road, weighted by `zone_access_weight`. The zones carry no coordinates."""
        a = self.areas
        position = np.c_[(a["area_col"] + 0.5) * AREA_CELLS * CELL_KM, (a["area_row"] + 0.5) * AREA_CELLS * CELL_KM]
        access = self.season.zone_access_weight * np.log1p(a["road_km"].to_numpy())
        zone_of = pd.Series("", index=a.index)
        for code in PILOT:
            here = np.flatnonzero(a["municipality_code"].to_numpy() == code)
            k = max(1, round(len(here) / self.season.zone_areas))
            labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(np.c_[position[here], access[here]])
            # number the zones from north-west to south-east, so the IDs do not depend on k-means' labels
            centre = pd.DataFrame({"label": labels, "row": a["area_row"].to_numpy()[here],
                                   "col": a["area_col"].to_numpy()[here]}).groupby("label").mean()
            order = centre.sort_values(["row", "col"], ascending=[False, True]).index
            number = {label: n for n, label in enumerate(order, start=1)}
            zone_of.iloc[here] = [f"{PILOT[code][2]}-Z{number[label]:02d}" for label in labels]
        a["zone_id"] = zone_of
        groups = a.groupby("zone_id")
        zones = groups.agg(municipality_code=("municipality_code", "first"), areas=("cells", "size"),
                           km2=("km2", "sum"), road_km=("road_km", "mean"), recorded=("event_inside", "sum"),
                           population=("population", "sum"))
        zones["municipality"] = zones["municipality_code"].map(NAME_OF)
        zones["access"] = zones["road_km"].map(access_band)
        # The consultation layer (simulated, restricted): the zones where communities must be consulted before any
        # tasking, a share set by this project
        rng = np.random.default_rng([SEED, 7])
        zones["consultation"] = rng.random(len(zones)) < self.season.consultation_share
        self.zones = zones
        self.zone_areas = MappingProxyType({z: tuple(ids) for z, ids in groups.groups.items()})
        self.cells["zone_id"] = self.cells["area_id"].map(a["zone_id"])

    def _build_teams(self):
        s = self.season
        rows = ([(f"S{n}", "survey") for n in range(1, s.survey_teams + 1)]
                + [(f"C{n}", "clearance") for n in range(1, s.clearance_teams + 1)]
                + [(f"T{n}", "ts") for n in range(1, s.ts_teams + 1)])
        self.teams = pd.DataFrame(rows, columns=["team", "kind"]).set_index("team")

    # ---- the state before the season ----------------------------------------------------------------------------
    def reset(self):
        """Return the twin to its state before the season: no survey done, no task ordered, nothing logged."""
        idx = self.areas.index
        self.status = pd.Series("unknown", index=idx, name="status")
        self.next_task = pd.Series(None, index=idx, name="next_task", dtype=object)
        self.visits = pd.Series(0, index=idx, name="visits")
        self.mines_found = pd.Series(False, index=idx, name="mines_found")   # by technical survey or clearance
        self.flag = pd.Series("", index=idx, name="flag")                     # why the zone survey flagged it
        self.evidence = {a: [] for a in idx}
        self.declared_items = {}       # (area, visit) -> the items the team's report said it sent
        self.item_high = {}            # area -> the highest item number received or declared so far
        zones = self.zones.index
        self.zone_state = pd.DataFrame({"surveyed_week": 0, "free_week": 0, "reopened": 0}, index=zones)
        self.municipality_free = pd.Series(0, index=list(PILOT), name="free_week")
        self.access = pd.DataFrame({"status": "open", "as_of": 0, "source": "access feed (simulated)"}, index=zones)
        self.access_history = {}       # week -> the zones the access feed showed closed that week
        self.consultation = pd.DataFrame({"required": self.zones["consultation"], "requested_week": 0,
                                          "held_week": 0}, index=zones)
        self.team_state = pd.DataFrame({"status": "active", "task_id": None, "zone": None, "as_of": 0,
                                        "back_week": np.nan, "source": [f"team {t}" for t in self.teams.index]},
                                       index=self.teams.index).astype({"task_id": object, "zone": object})
        self.tasks = {}                # task_id -> the task order and its progress, as reported
        self.pending_results = {}      # area -> the result of technical survey or clearance awaiting sign-off
        self.reports = []              # every report received, in order
        self.plan = None               # the approved plan
        self.plans = []                # every approved plan
        self.log = []                  # the one log: every report received and every decision, in order
        self.week = 0
        # The desk study: a past landmine accident recorded inside an area is indirect evidence, so the area is suspected
        for a in idx[self.areas["event_inside"]]:
            self.evidence[a].append(desk_item())
            self.status[a] = "suspected"
        self.initial_status = self.status.copy()

    # ---- the log ------------------------------------------------------------------------------------------------
    def _log(self, entry):
        self.log.append({"n": len(self.log) + 1, "time": utc_now(), **entry})
        return self.log[-1]["n"]

    def _tick(self, week):
        if week < self.week:
            raise ValueError(f"week {week} is before the twin's week {self.week}.")
        self.week = week

    def decide(self, kind, subject, decision, *, week, decided_by, recommended_by=None, overlays=(), evidence=(),
               rule_check="n/a", reason="", before=None, after=None):
        """Log one decision. Every write-back goes through this, after the checks of the method that calls it."""
        if not decided_by:
            raise ValueError("Every decision needs the role of the person who approved it.")
        self._tick(week)
        return self._log({"week": week, "entry": "decision", "kind": kind, "subject": subject, "decision": decision,
                          "before": before, "after": after, "decided_by": decided_by,
                          "recommended_by": recommended_by, "overlays": list(overlays), "evidence": list(evidence),
                          "rule_check": rule_check, "reason": reason})

    # ---- reports received ---------------------------------------------------------------------------------------
    def receive(self, report, week=None):
        """Record one report, received in `week` (by default the week it describes). Returns what it changed, in words.

        Report kinds: "team report" (a team's week on a task: days worked and lost, and the result when the task is
        done), "access" (the access feed, every zone's status), "team status" (an operator stands a team down, with
        the week it expects the team back, or brings it back), "consultation" (a consultation held in a zone),
        "evidence" (items a team re-sends after a report arrived without them), "incident" (a new accident or find
        reported in an area).
        """
        week = report["week"] if week is None else week
        self._tick(week)
        self.reports.append(report)
        kind = report["kind"]
        if kind == "team report":
            note = self._team_report(report)
        elif kind == "access":
            statuses = report["zones"]          # zone -> "open" or "closed"; the feed covers every zone each week
            zones = list(statuses)
            self.access.loc[zones, "status"] = [statuses[z] for z in zones]
            self.access.loc[zones, "as_of"] = week
            self.access.loc[zones, "source"] = report["source"]
            self.access_history[week] = frozenset(z for z in zones if statuses[z] != "open")
            changed = report.get("changed", [])
            note = "access feed: " + (", ".join(f"{z} {statuses[z]}" for z in changed) if changed else "no change")
        elif kind == "team status":
            team = report["team"]
            back = report.get("back_week", np.nan) if report["status"] != "active" else np.nan
            self.team_state.loc[team, ["status", "as_of", "source", "back_week"]] = [report["status"], week,
                                                                                    report["source"], back]
            note = f"team {team} {report['status']}" + (f", expected back in week {back}" if back == back else "")
        elif kind == "consultation":
            self.consultation.loc[report["zone"], "held_week"] = week
            note = f"consultation held in {report['zone']}"
        elif kind == "evidence":
            a = report["area"]
            self.evidence[a].extend(report["items"])
            self.evidence[a].sort(key=lambda i: int(i.item_id[1:]))
            note = f"{a}: {len(report['items'])} item(s) re-sent by the team"
        elif kind == "incident":
            a = report["area"]
            n = self.highest_item(a) + 1       # after every item received or declared, so no number is taken twice
            self.item_high[a] = n
            self.evidence[a].append(EvidenceItem(f"E{n}", report["statement"], report["source"],
                                                 f"Incident report, week {week}", week=week,
                                                 source_id=report.get("source_id", report["source"])))
            note = f"incident reported in {a} ({self.areas.loc[a, 'zone_id']})"
        else:
            raise ValueError(f"Unknown report kind {kind}.")
        subject = report.get("team") or report.get("zone") or report.get("area") or "access feed"
        self._log({"week": week, "entry": "report", "kind": kind, "subject": subject, "decision": None,
                   "source": report.get("source"), "reason": note})
        return note

    def highest_item(self, area_id):
        """The highest item number the area's evidence log holds or a report has declared."""
        return max([self.item_high.get(area_id, 0)] + [int(i.item_id[1:]) for i in self.evidence[area_id]])

    def _team_report(self, r):
        team, task = r["team"], self.tasks.get(r["task_id"])
        if task is None or task["team"] != team:
            raise ValueError(f"team {team} reports on {r['task_id']}, which is not its task order.")
        self.team_state.loc[team, ["as_of", "source"]] = [r["week"], r["source"]]
        task["days_done"] += r["days_worked"]
        task["days_lost_weather"] += r["days_lost_weather"]
        task["days_lost_access"] += r["days_lost_access"]
        task["weeks_reported"] += 1
        if task["status"] == "ordered" and r["days_worked"] > 0:
            task["status"], task["started_week"] = "in progress", r["week"]
        if not r.get("complete"):
            return f"{team}: {r['days_worked']} day(s) on {TASK_WORDS[task['kind']]} {task['area'] or task['zone']}"
        task["status"], task["finished_week"] = "done", r["week"]
        self.team_state.loc[team, ["task_id", "zone"]] = [None, None]
        result = r["result"]
        if task["kind"] == "ZONE_SURVEY":
            z = task["zone"]
            self.zone_state.loc[z, "surveyed_week"] = r["week"]
            flagged = dict(result["flagged"])
            for a in self.zone_areas[z]:
                if a in flagged:
                    self.flag[a] = flagged[a]
                    if self.status[a] in ("unknown", "suspected"):
                        self.status[a] = "suspected"
                        self.next_task[a] = "NTS"
                elif self.status[a] == "unknown":
                    self.status[a] = "not_suspected"
            return f"{z} surveyed: {len(flagged)} area(s) flagged for a visit"
        a = task["area"]
        if task["kind"] == "VISIT":
            before = self.highest_item(a)
            self.visits[a] += 1
            self.evidence[a].extend(result["items"])
            self.declared_items[(a, int(self.visits[a]))] = result["declared_items"]
            # a visit's items are numbered on from the area's last item, so an item still on its way keeps its number
            self.item_high[a] = max(self.highest_item(a), before + int(result["declared_items"] or 0))
            if self.next_task[a] == "NTS":
                self.next_task[a] = None     # awaiting review
            return f"{a}: visit {self.visits[a]}, {len(result['items'])} item(s) received"
        self.pending_results[a] = {"kind": task["kind"], "week": r["week"], "task_id": task["task_id"], **result}
        return f"{a}: {TASK_WORDS[task['kind']]} done, result awaiting sign-off"

    # ---- decisions ----------------------------------------------------------------------------------------------
    def approve_plan(self, plan, *, week, decided_by, recommended_by, overlays=(), reason=""):
        """Approve a season plan: from now on, task orders follow it."""
        plan = dict(plan)
        plan["plan_id"] = f"P{len(self.plans) + 1:02d}"
        plan["approved_week"] = week
        self.plans.append(plan)
        before = self.plan["plan_id"] if self.plan else None
        self.plan = plan
        self.decide("plan", plan["plan_id"], "approve", week=week, decided_by=decided_by,
                    recommended_by=recommended_by, overlays=overlays, before=before, after=plan["plan_id"],
                    reason=reason)
        return plan["plan_id"]

    def open_work(self, zone, kind):
        """The work open in a zone for a kind of team, not yet ordered (or paused): the zone survey, visits, TS or
        clearance."""
        ordered = {t["area"] or t["zone"] for t in self.tasks.values()
                   if t["status"] in ("ordered", "in progress", "paused")}
        if kind == "survey":
            if self.zone_state.loc[zone, "surveyed_week"] == 0:
                return [] if zone in ordered else [("ZONE_SURVEY", None)]
            want = "NTS"
            task = "VISIT"
        else:
            want = task = "TS" if kind == "ts" else "CLEARANCE"
        areas = self.zone_areas[zone]
        nt = self.next_task.loc[list(areas)]
        return [(task, a) for a in nt.index[nt == want] if a not in ordered and a not in self.pending_results]

    def paused_work(self, zone, kind):
        """The paused tasks in a zone for a kind of team, which a team of that kind can resume."""
        return [t["task_id"] for t in self.tasks.values() if t["zone"] == zone and t["status"] == "paused"
                and TASK_KINDS[t["kind"]] == kind]

    def zone_ready_for_tasking(self, zone):
        """Whether a zone can be tasked: open by the access feed, and consulted where consultation is required."""
        if self.access.loc[zone, "status"] != "open":
            return False, "the access feed shows the zone closed"
        c = self.consultation.loc[zone]
        if c["required"] and not c["held_week"]:
            return False, "the communities have not yet been consulted"
        return True, ""

    def order_task(self, team, kind, zone, area=None, *, week, decided_by, recommended_by, overlays=(), reason="",
                   resume=None):
        """Order a team to a task, or back to a paused one (`resume`). Returns the task's ID."""
        if self.team_state.loc[team, "status"] != "active":
            raise ValueError(f"team {team} is {self.team_state.loc[team, 'status']}.")
        if self.team_state.loc[team, "task_id"] is not None:
            raise ValueError(f"team {team} already has task {self.team_state.loc[team, 'task_id']}.")
        if TASK_KINDS[kind] != self.teams.loc[team, "kind"]:
            raise ValueError(f"team {team} is a {self.teams.loc[team, 'kind']} team and cannot do {TASK_WORDS[kind]}.")
        ready, why = self.zone_ready_for_tasking(zone)
        if not ready:
            raise ValueError(f"{zone} cannot be tasked: {why}.")
        if self.zone_state.loc[zone, "free_week"]:
            raise ValueError(f"{zone} is declared free.")
        if resume is not None:
            task = self.tasks[resume]
            if task["status"] != "paused" or (task["kind"], task["zone"], task["area"]) != (kind, zone, area):
                raise ValueError(f"{resume} is not a paused {TASK_WORDS[kind]} in {zone}.")
            task["status"], task["team"] = ("in progress" if task["days_done"] else "ordered"), team
            task_id = resume
        else:
            if (kind, area) not in self.open_work(zone, TASK_KINDS[kind]):
                raise ValueError(f"there is no open {TASK_WORDS[kind]} for {area or zone}.")
            task_id = f"K{len(self.tasks) + 1:04d}"
            self.tasks[task_id] = {"task_id": task_id, "kind": kind, "team": team, "zone": zone, "area": area,
                                   "ordered_week": week, "status": "ordered", "days_planned":
                                   self.season.task_days(kind), "days_done": 0, "days_lost_weather": 0,
                                   "days_lost_access": 0, "weeks_reported": 0, "started_week": 0, "finished_week": 0,
                                   "plan_id": self.plan["plan_id"] if self.plan else None}
        self.team_state.loc[team, ["task_id", "zone"]] = [task_id, zone]
        self.decide("task order", task_id, f"{team}: {TASK_WORDS[kind]} {area or zone}" + (" (resumed)" if resume
                    else ""), week=week, decided_by=decided_by, recommended_by=recommended_by, overlays=overlays,
                    reason=reason)
        return task_id

    def pause_task(self, task_id, *, week, decided_by, recommended_by=None, reason):
        """Pause a task (its team stood down, or its zone closed); the progress reported so far is kept."""
        task = self.tasks[task_id]
        if task["status"] not in ("ordered", "in progress"):
            raise ValueError(f"{task_id} is {task['status']}.")
        task["status"] = "paused"
        if self.team_state.loc[task["team"], "task_id"] == task_id:
            self.team_state.loc[task["team"], ["task_id", "zone"]] = [None, None]
        self.decide("pause", task_id, f"pause {TASK_WORDS[task['kind']]} {task['area'] or task['zone']}", week=week,
                    decided_by=decided_by, recommended_by=recommended_by, reason=reason)

    def request_consultation(self, zone, *, week, decided_by, recommended_by=None, reason=""):
        if not self.consultation.loc[zone, "required"] or self.consultation.loc[zone, "requested_week"]:
            raise ValueError(f"{zone} needs no new consultation request.")
        self.consultation.loc[zone, "requested_week"] = week
        self.decide("consultation", zone, "request consultation", week=week, decided_by=decided_by,
                    recommended_by=recommended_by, reason=reason)

    def dossier_complete(self, area_id):
        """Whether an area's dossier holds everything the reports said was sent. Returns (complete, problems).

        - Each visit's items arrived: as many as the team's report declared.
        - The items are numbered without a gap.
        - The desk study's record is there, if the hazard layer records an accident inside the area.
        """
        problems = []
        items = self.evidence[area_id]
        for v in range(1, int(self.visits[area_id]) + 1):
            got = sum(1 for i in items if i.visit == v)
            declared = self.declared_items.get((area_id, v))
            if declared is not None and got != declared:
                problems.append(f"visit {v}: the report declared {declared} item(s), {got} arrived")
        numbers = sorted(int(i.item_id[1:]) for i in items)
        if numbers != list(range(1, len(numbers) + 1)):
            missing = sorted(set(range(1, max(numbers, default=0) + 1)) - set(numbers))
            problems.append("missing item number(s): " + ", ".join(f"E{n}" for n in missing))
        if self.areas.loc[area_id, "event_inside"] and not any(i.source_id == "national database" for i in items):
            problems.append("the hazard layer records an accident inside the area, but the desk study's record is "
                            "not in the evidence log")
        return not problems, problems

    def hold_dossier(self, area_id, *, week, decided_by, reason):
        """Hold an area's dossier back from review until what is missing arrives."""
        self.decide("dossier", area_id, "hold for missing items", week=week, decided_by=decided_by, reason=reason)

    def sign_off(self, area_id, outcome, *, week, decided_by, recommended_by, recommendation=None, register=None,
                 evidence=None, reason="", overlays=()):
        """A decision on an area signed off by a person: its status and next task change, and the change is logged.

        A review decision (CANCEL, TECHNICAL_SURVEY, CLEARANCE, FURTHER_NTS) rests on a register of evidence, and the
        twin runs the rule check on it itself: it refuses a cancellation, technical survey or clearance that the rule
        check vetoes (further non-technical survey is always open), and any review of an incomplete dossier. A
        decision on direct evidence makes the area a confirmed hazardous area. A result of technical survey
        (TS_FOUND, TS_NOTHING) or clearance (CLEARED) is accepted only when the team's report of it is in. The twin
        also refuses a decision on an area already released, and an override of a recommendation with no written
        reason.
        """
        if self.status[area_id] in RELEASED:
            raise ValueError(f"{area_id} is already released ({self.status[area_id]}).")
        if recommendation is not None and outcome != recommendation and not reason:
            raise ValueError(f"{area_id}: overriding {recommendation} with {outcome} needs a written reason.")
        if outcome in REVIEW_OUTCOMES:
            complete, problems = self.dossier_complete(area_id)
            if not complete:
                raise ValueError(f"{area_id}: the dossier is incomplete ({'; '.join(problems)}), so no review of it "
                                 "can be signed off.")
            new_status, task = REVIEW_OUTCOMES[outcome]
            register = register or {}
            result = rule_check(outcome, register, self.evidence[area_id], self.criteria)
            if result["result"] != "pass":
                raise ValueError(f"{area_id}: the rule check vetoes {outcome} ({failed_rules(result)}), so it "
                                 "cannot be signed off.")
            check_result = result["result"]
            if outcome in ("TECHNICAL_SURVEY", "CLEARANCE") and any(e.evidence_class == "direct"
                                                                     for e in register.values()):
                new_status = "confirmed"
        elif outcome in RESULT_OUTCOMES:
            pending = self.pending_results.get(area_id)
            want = "CLEARANCE" if outcome == "CLEARED" else "TS"
            if pending is None or pending["kind"] != want:
                raise ValueError(f"{area_id}: there is no reported {TASK_WORDS[want]} result to sign off.")
            if outcome != "CLEARED" and pending["found"] != (outcome == "TS_FOUND"):
                raise ValueError(f"{area_id}: {outcome} does not match the team's report.")
            new_status, task = RESULT_OUTCOMES[outcome]
            if outcome == "TS_FOUND" or (outcome == "CLEARED" and pending.get("mine_cells")):
                self.mines_found[area_id] = True
            del self.pending_results[area_id]
            check_result = "n/a"
        else:
            raise ValueError(f"Unknown outcome {outcome}.")
        before = self.status[area_id]
        if new_status is not None:
            self.status[area_id] = new_status
        self.next_task[area_id] = task
        if evidence is None:
            evidence = [i.item_id for i in self.evidence[area_id]] if outcome in REVIEW_OUTCOMES else []
        self.decide("area", area_id, outcome, week=week, decided_by=decided_by, recommended_by=recommended_by,
                    overlays=overlays, evidence=evidence, rule_check=check_result, before=before,
                    after=self.status[area_id], reason=reason or (f"override of {recommendation}"
                                                                    if recommendation not in (None, outcome) else ""))

    def zone_unresolved(self, zone):
        """The flagged areas of a surveyed zone that are not yet released, and the IDs of its open tasks."""
        areas = self.zone_areas[zone]
        open_areas = [a for a in areas if self.status[a] in ("suspected", "confirmed")]
        open_tasks = [t["task_id"] for t in self.tasks.values() if t["zone"] == zone and t["status"] in
                      ("ordered", "in progress", "paused")]
        return open_areas, open_tasks

    def declare_zone_free(self, zone, *, week, decided_by, recommended_by=None, reason=""):
        """Declare a zone free: surveyed, every flagged area released, no task open (Travis's setting, question 39)."""
        if not self.zone_state.loc[zone, "surveyed_week"]:
            raise ValueError(f"{zone} has not been surveyed.")
        if self.zone_state.loc[zone, "free_week"]:
            raise ValueError(f"{zone} is already declared free.")
        open_areas, open_tasks = self.zone_unresolved(zone)
        if open_areas or open_tasks:
            raise ValueError(f"{zone} cannot be declared free: {len(open_areas)} area(s) unresolved, "
                             f"{len(open_tasks)} task(s) open.")
        self.zone_state.loc[zone, "free_week"] = week
        self.decide("zone", zone, "declare free", week=week, decided_by=decided_by, recommended_by=recommended_by,
                    before="open", after="free", reason=reason)

    def reopen_zone(self, zone, area_id, *, week, decided_by, recommended_by=None, reason):
        """Reopen a zone declared free, after new evidence in one of its areas: the area goes back for a visit."""
        if not self.zone_state.loc[zone, "free_week"]:
            raise ValueError(f"{zone} is not declared free.")
        self.zone_state.loc[zone, ["free_week", "reopened"]] = [0, self.zone_state.loc[zone, "reopened"] + 1]
        code = self.zones.loc[zone, "municipality_code"]
        if self.municipality_free[code]:
            self.municipality_free[code] = 0
            self.decide("municipality", code, "revoke free", week=week, decided_by=decided_by,
                        recommended_by=recommended_by, before="free", after="open",
                        reason=f"{zone} reopened")
        self._reopen_area(area_id)
        self.decide("zone", zone, "reopen", week=week, decided_by=decided_by, recommended_by=recommended_by,
                    evidence=[self.evidence[area_id][-1].item_id], before="free", after="open", reason=reason)

    def reopen_area(self, area_id, *, week, decided_by, recommended_by=None, reason):
        """Send an area back for a visit after new evidence, in a zone not declared free."""
        before = self.status[area_id]
        self._reopen_area(area_id)
        self.decide("area", area_id, "REOPEN", week=week, decided_by=decided_by, recommended_by=recommended_by,
                    evidence=[self.evidence[area_id][-1].item_id], before=before, after="suspected", reason=reason)

    def _reopen_area(self, area_id):
        self.status[area_id] = "suspected"
        self.next_task[area_id] = "NTS"
        self.flag[area_id] = self.flag[area_id] or "incident report"

    def declare_municipality_free(self, code, *, week, decided_by, recommended_by=None, reason=""):
        zones = self.zones.index[self.zones["municipality_code"] == code]
        if not (self.zone_state.loc[zones, "free_week"] > 0).all():
            raise ValueError(f"{NAME_OF[code]} has zones not yet declared free.")
        if self.municipality_free[code]:
            raise ValueError(f"{NAME_OF[code]} is already declared free.")
        self.municipality_free[code] = week
        self.decide("municipality", code, "declare free", week=week, decided_by=decided_by,
                    recommended_by=recommended_by, before="open", after="free", reason=reason)

    # ---- views --------------------------------------------------------------------------------------------------
    def view(self):
        """A read-only view of the twin, for the overlays."""
        return TwinView(self)

    def log_table(self):
        """The one log as a table, with its lists copied, so that nothing in the table reaches the log itself."""
        return pd.DataFrame([{k: (list(v) if isinstance(v, list) else v) for k, v in e.items()} for e in self.log])

    def task_table(self):
        return pd.DataFrame(list(self.tasks.values()))

    def team_table(self, week=None):
        """The teams as the twin knows them: status, task, zone, the week of the last report, and whether it is stale."""
        week = self.week if week is None else week
        t = self.teams.join(self.team_state)
        t["weeks_since_report"] = week - t["as_of"]
        t["stale"] = (t["status"] == "active") & t["task_id"].notna() & (
            t["weeks_since_report"] > self.season.stale_after_weeks)
        # a team that has not reported since its task order is stale too, once the order is old enough
        for team, task_id in t["task_id"].dropna().items():
            ordered = self.tasks[task_id]["ordered_week"]
            if t.loc[team, "as_of"] < ordered:
                t.loc[team, "weeks_since_report"] = week - ordered
                t.loc[team, "stale"] = week - ordered > self.season.stale_after_weeks
        return t

    def zone_table(self):
        """The operations picture by zone: status, access, consultation, and the areas by status."""
        z = self.zones[["municipality", "areas", "km2", "access", "recorded"]].copy()
        counts = self.status.groupby(self.areas["zone_id"]).value_counts().unstack(fill_value=0)
        z = z.join(counts.reindex(columns=STATUSES, fill_value=0))
        z["access_status"] = self.access["status"]
        z["access_as_of"] = self.access["as_of"]
        z["consultation"] = np.where(~self.consultation["required"], "not required",
                                     np.where(self.consultation["held_week"] > 0, "held",
                                              np.where(self.consultation["requested_week"] > 0, "requested",
                                                       "required")))
        z["surveyed_week"] = self.zone_state["surveyed_week"]
        z["free_week"] = self.zone_state["free_week"]
        z["reopened"] = self.zone_state["reopened"]
        teams = self.team_state["zone"].dropna()
        z["teams"] = [", ".join(sorted(teams.index[teams == zone])) for zone in z.index]
        z["state"] = np.select([z["free_week"] > 0, z["surveyed_week"] > 0, z["teams"] != ""],
                               ["free", "surveyed", "in survey"], "not surveyed")
        return z

    def municipal_summary(self):
        """The public picture, by municipality: open layers only, as counts. No zone, area or team is named."""
        z = self.zone_table()
        g = z.groupby("municipality")
        out = pd.DataFrame({"zones": g.size(), "zones surveyed": g["surveyed_week"].apply(lambda s: int((s > 0).sum())),
                            "zones declared free": g["free_week"].apply(lambda s: int((s > 0).sum())),
                            "areas": g["areas"].sum(), "km²": g["km2"].sum()})
        a = self.areas
        released = self.status.isin(RELEASED)
        out["areas released"] = released.groupby(a["municipality"]).sum()
        out["areas cleared"] = self.status.eq("cleared").groupby(a["municipality"]).sum()
        out["areas suspected or confirmed"] = self.status.isin(["suspected", "confirmed"]).groupby(
            a["municipality"]).sum()
        victims = self.people.groupby("municipality_code")["victims"].sum()
        out["recorded victims (all years)"] = [int(victims.get(c, 0)) for c in
                                               a.groupby("municipality")["municipality_code"].first()[out.index]]
        free = {NAME_OF[c]: w for c, w in self.municipality_free.items()}
        out["declared free (week)"] = [free[m] or "" for m in out.index]
        return out

    def area_dossier(self, area_id):
        """Everything the twin holds on one area, as the sections a reviewer reads."""
        a = self.areas.loc[area_id]
        z = self.zones.loc[a["zone_id"]]
        record = "\n".join([
            "| Field | Entry |", "|---|---|",
            f"| Area ID | {area_id} |",
            f"| Municipality | {a['municipality']}, Antioquia |",
            f"| Operational zone | {a['zone_id']} ({z['areas']} areas, {z['access']}) |",
            f"| Size | {a['cells']} cell(s) of 500 m × 500 m ({a['km2']:.2f} km²) |",
            f"| Recorded status | {self.status[area_id]} |",
            f"| Main land use (base layer) | {a['main_land_use']}; {a['forest_share']:.0%} forest, "
            f"{a['used_share']:.0%} farmed, grazed or built on |",
            f"| Distance to the nearest settlement (average over the cells) | {settlement_band(a['settlement_km'])} |",
            f"| Non-technical survey visits so far | {self.visits[area_id]} |",
        ])
        desk = [f"- Past landmine accidents in the national database: {event_band(a['event_km'])}."]
        surveyed = self.zone_state.loc[a["zone_id"], "surveyed_week"]
        flag = self.flag[area_id]
        if surveyed and flag:
            desk.append(f"- The zone survey of week {surveyed} flagged the area for a visit: {flag}. The flag says "
                        "where to visit. It is not an item of evidence: the evidence is in the evidence log.")
        priority = (self.plan or {}).get("priority")
        if priority is not None and area_id in priority.index:
            p = priority.loc[area_id]
            desk.append(f"- Survey priority in zone {a['zone_id']}: rank {int(p['zone_rank'])} of {z['areas']} "
                        f"areas, by {p['ranking']}. The priority says where to survey first. It is not evidence "
                        "about this area.")
        neighbour_status = pd.Series([self.status[b] for b in self.neighbours[area_id]]).value_counts()
        desk.append("- Neighbouring areas: " + (", ".join(f"{n} {s}" for s, n in neighbour_status.items())
                                                 if len(neighbour_status) else "none") + ".")
        log = ["| ID | Evidence | Source | When |", "|---|---|---|---|"]
        for item in self.evidence[area_id]:
            log.append(f"| {item.item_id} | {item.statement} | {item.source} | {item.when} |")
        return {"area_record": record, "desk_study": "\n".join(desk), "evidence_log": "\n".join(log),
                "national_criteria": self.criteria.as_text()}

    def map_layer(self, values, municipality_code):
        """A grid of one municipality's cells, with each cell's value (from a Series by area, zone or cell).

        Rows run north to south and columns west to east. The grid carries no coordinates: plotted with no axes,
        it shows the shape of the municipality and where values lie within it, and nothing that locates it.
        """
        part = self.cells[self.cells["municipality_code"] == municipality_code]
        if values.index.isin(self.zones.index).all():
            per_cell = values.reindex(part["zone_id"]).to_numpy()
        elif values.index.isin(self.areas.index).all():
            per_cell = values.reindex(part["area_id"]).to_numpy()
        else:
            per_cell = values.reindex(part.index).to_numpy()
        rows, cols = part["row"] - part["row"].min(), part["col"] - part["col"].min()
        grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
        grid[rows.max() - rows.to_numpy(), cols.to_numpy()] = per_cell
        return grid


def copy_plan(plan):
    """A copy of a plan that writing to cannot change: lists and dicts copied, tables copied (cheaply, as pandas
    copies on write)."""
    out = {}
    for k, v in plan.items():
        if isinstance(v, (pd.DataFrame, pd.Series)):
            out[k] = v.copy(deep=False)
        elif isinstance(v, (list, dict, set, tuple)):
            out[k] = copy.deepcopy(v)
        else:
            out[k] = v
    return out


class TwinView:
    """What an overlay may read: the twin's state and its views. It has no method that changes the twin, and
    assigning to it fails, so an overlay can recommend but never write (the overlay contract).

    Tables come as copies (cheap under pandas' copy-on-write: writing to one never reaches the twin), mappings as
    read-only proxies, evidence items are frozen, the small mutable records (tasks, plans) are copies, and the view
    methods return plain values (IDs, text, counts). The view guards against mistakes; it is not a security boundary,
    since Python code can always reach the twin behind it.
    """
    STATE = {"criteria", "season", "calendar", "areas", "zones", "zone_areas", "teams", "neighbours", "cells",
             "people", "activity", "layers", "status", "next_task", "visits", "mines_found", "flag", "evidence",
             "zone_state", "municipality_free", "access", "consultation", "team_state", "tasks", "pending_results",
             "plan", "plans", "week", "initial_status", "declared_items", "access_history"}
    VIEWS = {"zone_table", "team_table", "municipal_summary", "area_dossier", "map_layer", "log_table", "task_table",
             "open_work", "paused_work", "zone_ready_for_tasking", "zone_unresolved", "dossier_complete"}

    def __init__(self, twin):
        object.__setattr__(self, "_twin", twin)

    def __getattr__(self, name):
        if name in self.VIEWS:
            return getattr(self._twin, name)
        if name not in self.STATE:
            raise AttributeError(f"an overlay cannot use the twin's {name}: overlays read the twin, never write to it")
        value = getattr(self._twin, name)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.copy(deep=False)
        if isinstance(value, MappingProxyType) or isinstance(value, (int, Criteria, Season, Calendar)):
            return value
        if name == "evidence":
            return MappingProxyType({a: tuple(items) for a, items in value.items()})
        if name == "access_history":
            return MappingProxyType(dict(value))                # week -> a frozen set of zones
        if name in ("tasks", "pending_results"):
            return {k: dict(v) for k, v in value.items()}       # records of plain values: one level deep is a copy
        if name == "plan":
            return None if value is None else copy_plan(value)
        if name == "plans":
            return [copy_plan(p) for p in value]
        return copy.deepcopy(value)

    def __setattr__(self, name, value):
        raise AttributeError("overlays read the twin, never write to it")
