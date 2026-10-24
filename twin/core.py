"""The digital twin: a model of the pilot's state, kept in step with the evidence that comes in, on which
decisions are tested before they are written back.

Its keys are place (department, municipality, area, cell) and time (year and month for the national tables,
rounds of work in the replay). Its layers:
- base: RELand's cells and their 63 features, grouped into areas of 2 x 2 cells;
- hazard: each area's status and the evidence behind it, starting from the national database's past accidents;
- activity: national demining results by municipality and month (P2), and the pilot's tasks in the replay;
- people: victims by municipality and month (P1).

The twin never holds RELand's labels. It changes in two ways only, and logs both: evidence received from a
field team, and a decision signed off by a person. Before it accepts a decision, it runs the rule check itself.
"""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import AREA_CELLS, CELL_KM, EVENT_INSIDE_KM, NAME_OF, PILOT, Criteria
from .rules import failed_rules, rule_check

STATUSES = ["unknown", "suspected", "confirmed", "cancelled", "reduced", "cleared"]
RELEASED = {"cancelled", "reduced", "cleared"}

# What a signed-off decision does to an area: its new status (None: unchanged) and the next task (None: none).
REVIEW_OUTCOMES = {
    "CANCEL": ("cancelled", None),
    "TECHNICAL_SURVEY": ("suspected", "TS"),
    "CLEARANCE": ("confirmed", "CLEARANCE"),
    "FURTHER_NTS": (None, "NTS"),
}
RESULT_OUTCOMES = {
    "TS_FOUND": ("confirmed", "CLEARANCE"),   # technical survey confirmed mines: clearance follows
    "TS_NOTHING": ("reduced", None),          # technical survey found nothing: the area is released
    "CLEARED": ("cleared", None),             # clearance done: the area is released
}

LAND_USE = {"land_use_Agroforestal": "agroforestry", "land_use_Agrícola": "farming",
            "land_use_Conservación de Suelos": "soil conservation", "land_use_Cuerpo de agua": "water",
            "land_use_Forestal": "forest", "land_use_Ganadera": "livestock", "land_use_Zonas urbanas": "urban"}
USED_LAND = ["land_use_Agroforestal", "land_use_Agrícola", "land_use_Ganadera", "land_use_Zonas urbanas"]


@dataclass
class EvidenceItem:
    """One item of evidence on an area, as the twin holds it and a reviewer reads it."""
    item_id: str
    statement: str
    source: str
    when: str                  # "Desk study", or the visit and round, such as "Visit 1, round 3"
    visit: int = 0             # 0 for the desk study
    round: int = 0
    years: int | None = None   # years of safe use the statement reports, if any (read by the rule check)
    photo: bool = False        # whether a photo was taken (the screener's reading is held separately)
    source_id: str = ""        # who it comes from, such as "informant 3" (read by the rule check to count sources)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


class Twin:
    """The twin of the pilot, with the national people and activity layers beside it."""

    def __init__(self, cells, people, activity, criteria=None):
        self.criteria = criteria or Criteria()
        self.cells = cells.copy()
        self.people = people          # victims by municipality and month (data.victims_by_month)
        self.activity = activity      # demining results by municipality and month (data.load_demining)
        self._build_areas()
        self.priority = None          # set by the operational service
        self.reset()

    # ---- building the base layer ------------------------------------------------------------------------------
    def _build_areas(self):
        c = self.cells
        c["area_row"], c["area_col"] = c["row"] // AREA_CELLS, c["col"] // AREA_CELLS
        c["_used"] = c[USED_LAND].sum(axis=1)
        groups = c.groupby(["municipality_code", "area_row", "area_col"], sort=True)
        areas = groups.agg(cells=("row", "size"), forest_share=("land_use_Forestal", "mean"),
                           used_share=("_used", "mean"), settlement_km=("settlement_dist", "mean"),
                           event_km=("dist_old_mine", "min"), population=("population_2012", "sum"))
        areas["km2"] = areas["cells"] * CELL_KM ** 2
        areas["main_land_use"] = groups[list(LAND_USE)].sum().idxmax(axis=1).map(LAND_USE)
        areas["cell_list"] = groups.apply(lambda part: list(part.index), include_groups=False)
        areas = areas.reset_index()
        c.drop(columns="_used", inplace=True)
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
        self.neighbours = {}
        for a, ar, ac in self.areas[["area_row", "area_col"]].itertuples():
            near = [b for key in [(ar, ac), (ar - 1, ac), (ar + 1, ac), (ar, ac - 1), (ar, ac + 1)]
                    for b in where.get(key, []) if b != a]
            self.neighbours[a] = near

    def reset(self):
        """Return the hazard and activity layers of the pilot to their state before any survey."""
        self.status = pd.Series("unknown", index=self.areas.index, name="status")
        self.next_task = pd.Series(None, index=self.areas.index, name="next_task", dtype=object)
        self.visits = pd.Series(0, index=self.areas.index, name="visits")
        self.mines_found = pd.Series(False, index=self.areas.index, name="mines_found")   # by TS or clearance
        self.evidence = {a: [] for a in self.areas.index}
        self.screener_readings = {}   # (area, item) -> the screener's reading of the item's photo
        self.audit = []               # every change to the twin, in order
        self.tasks = []               # the pilot's activity in the replay: one row per task done
        # The desk study: a past landmine accident recorded inside an area is indirect evidence, so the area is suspected
        for a in self.areas.index[self.areas["event_inside"]]:
            self.evidence[a].append(EvidenceItem(
                "E1", "The national mine action database records one or more past landmine accidents inside this "
                      "area. The twin holds no date or other detail of the record, and its location has not been "
                      "checked on the ground.",
                "National mine action database (desk study)", "Desk study", source_id="national database"))
            self.status[a] = "suspected"
        self.initial_status = self.status.copy()

    # ---- the write-back: the only two ways the twin changes -----------------------------------------------------
    def record_visit(self, area_id, items, round_number, team, screener_readings=None):
        """Evidence received from a field team: the items are added to the area's evidence log."""
        self.visits[area_id] += 1
        self.evidence[area_id].extend(items)
        for item_id, reading in (screener_readings or {}).items():
            self.screener_readings[(area_id, item_id)] = reading
        self.tasks.append({"round": round_number, "area_id": area_id, "task": "NTS", "team": team,
                           "items": len(items)})
        self._log(round_number, area_id, "evidence received", None, self.status[area_id], self.status[area_id],
                  decided_by=None, recommended_by=team, recommendation=None, rule_check="n/a",
                  evidence=[i.item_id for i in items], reason=f"{len(items)} item(s) from visit {self.visits[area_id]}")
        if self.next_task[area_id] == "NTS":
            self.next_task[area_id] = None

    def sign_off(self, area_id, outcome, *, round_number, decided_by, recommended_by, recommendation=None,
                 register=None, evidence=None, reason="", found=None):
        """A decision signed off by a person: the area's status and next task change, and the change is logged.

        A review decision (CANCEL, TECHNICAL_SURVEY, CLEARANCE, FURTHER_NTS) rests on a register of evidence, and
        the twin runs the rule check on it itself: it refuses a cancellation, technical survey or clearance that the
        rule check vetoes (further non-technical survey is always open). Technical survey on direct evidence makes
        the area a confirmed hazardous area. A result of technical survey (TS_FOUND, TS_NOTHING) or clearance
        (CLEARED, with the number of mine cells found) is accepted only for an area with that task pending. The twin
        also refuses a decision on an area already released, and an override of a recommendation with no written
        reason.
        """
        if self.status[area_id] in RELEASED:
            raise ValueError(f"{area_id} is already released ({self.status[area_id]}).")
        if recommendation is not None and outcome != recommendation and not reason:
            raise ValueError(f"{area_id}: overriding {recommendation} with {outcome} needs a written reason.")
        if outcome in REVIEW_OUTCOMES:
            new_status, task = REVIEW_OUTCOMES[outcome]
            register = register or {}
            result = rule_check(outcome, register, self.evidence[area_id], self.criteria)
            if result["result"] != "pass":
                raise ValueError(f"{area_id}: the rule check vetoes {outcome} ({failed_rules(result)}), so it "
                                 "cannot be signed off.")
            check_result = result["result"]
            if outcome == "TECHNICAL_SURVEY" and any(e.evidence_class == "direct" for e in register.values()):
                new_status = "confirmed"
        elif outcome in RESULT_OUTCOMES:
            new_status, task = RESULT_OUTCOMES[outcome]
            pending = "CLEARANCE" if outcome == "CLEARED" else "TS"
            if self.next_task[area_id] != pending:
                raise ValueError(f"{area_id}: there is no {'clearance' if pending == 'CLEARANCE' else 'technical survey'}"
                                 f" task to report {outcome} for.")
            if outcome == "TS_FOUND" or (outcome == "CLEARED" and found):
                self.mines_found[area_id] = True
            check_result = "n/a"
        else:
            raise ValueError(f"Unknown outcome {outcome}.")
        before = self.status[area_id]
        if new_status is not None:
            self.status[area_id] = new_status
        self.next_task[area_id] = task
        if outcome in RESULT_OUTCOMES:
            self.tasks.append({"round": round_number, "area_id": area_id,
                               "task": "CLEARANCE" if outcome == "CLEARED" else "TS", "team": recommended_by,
                               "items": None, "mine cells found": found if outcome == "CLEARED" else None})
        if evidence is None:
            evidence = [i.item_id for i in self.evidence[area_id]] if outcome in REVIEW_OUTCOMES else []
        self._log(round_number, area_id, "decision", outcome, before, self.status[area_id], decided_by=decided_by,
                  recommended_by=recommended_by, recommendation=recommendation, rule_check=check_result,
                  evidence=list(evidence), reason=reason)

    def _log(self, round_number, area_id, action, outcome, before, after, **details):
        self.audit.append({"time": utc_now(), "round": round_number, "area_id": area_id, "action": action,
                           "outcome": outcome, "status_before": before, "status_after": after, **details})

    def audit_log(self):
        return pd.DataFrame(self.audit)

    # ---- views --------------------------------------------------------------------------------------------------
    def set_priority(self, table):
        """Store the operational service's ranking: one row per area with score, rank, ranking and baseline rank."""
        self.priority = table.reindex(self.areas.index)

    def operational_view(self, municipality_code=None):
        """The areas still to be resolved in one municipality (or all), in priority order, with their state."""
        view = self.areas[["municipality", "cells", "main_land_use", "event_inside"]].join(
            pd.DataFrame({"status": self.status, "next_task": self.next_task, "visits": self.visits}))
        if self.priority is not None:
            view = view.join(self.priority)
        if municipality_code is not None:
            view = view[self.areas["municipality_code"] == municipality_code]
        view = view[~view["status"].isin(RELEASED)]
        return view.sort_values("rank") if "rank" in view else view

    def municipal_table(self):
        """The national people and activity layers joined by municipality: one row per municipality."""
        people = self.people.groupby(["municipality_code"]).agg(
            department=("department", "first"), municipality=("municipality", "first"),
            victims=("victims", "sum"), first_victim_year=("year", "min"), last_victim_year=("year", "max"))
        activity = self.activity.groupby("municipality_code").agg(
            first_demining_year=("year", "min"), last_demining_year=("year", "max"),
            area_cleared_m2=("area_cleared_m2", "sum"), items_destroyed=("items_total", "sum"),
            non_technical_surveys=("non_technical_surveys", "sum"), technical_surveys=("technical_surveys", "sum"))
        table = people.join(activity, how="outer")
        names = self.activity.groupby("municipality_code")[["department", "municipality"]].first()
        table["department"] = table["department"].fillna(names["department"])
        table["municipality"] = table["municipality"].fillna(names["municipality"])
        table["in_pilot"] = table.index.isin(list(PILOT))
        return table

    def area_dossier(self, area_id):
        """Everything the twin holds on one area, as the sections a reviewer reads."""
        a = self.areas.loc[area_id]
        visits = self.visits[area_id]
        record = "\n".join([
            "| Field | Entry |", "|---|---|",
            f"| Area ID | {area_id} |",
            f"| Municipality | {a['municipality']}, Antioquia |",
            f"| Size | {a['cells']} cell(s) of 500 m × 500 m ({a['km2']:.2f} km²) |",
            f"| Recorded status | {self.status[area_id]} |",
            f"| Main land use (base layer) | {a['main_land_use']}; {a['forest_share']:.0%} forest, "
            f"{a['used_share']:.0%} farmed, grazed or built on |",
            f"| Distance to the nearest settlement (average over the cells) | {settlement_band(a['settlement_km'])} |",
            f"| Non-technical survey visits so far | {visits} |",
        ])
        neighbour_status = pd.Series([self.status[b] for b in self.neighbours[area_id]]).value_counts()
        desk = [f"- Past landmine accidents in the national database: {event_band(a['event_km'])}."]
        if self.priority is not None and not pd.isna(self.priority.loc[area_id, "rank"]):
            p = self.priority.loc[area_id]
            desk.append(f"- Survey priority in {a['municipality']}: rank {int(p['rank'])} of "
                        f"{int((self.areas['municipality_code'] == a['municipality_code']).sum())} areas, by "
                        f"{p['ranking']}; rank {int(p['baseline_rank'])} by nearness to past accidents alone.")
            desk.append("- The priority says where to survey first. It is not evidence about this area.")
        desk.append("- Neighbouring areas: " + (", ".join(f"{n} {s}" for s, n in neighbour_status.items())
                                                 if len(neighbour_status) else "none") + ".")
        items = self.evidence[area_id]
        log = ["| ID | Evidence | Source | When | Photo |", "|---|---|---|---|---|"]
        for item in items:
            log.append(f"| {item.item_id} | {item.statement} | {item.source} | {item.when} | "
                       f"{'yes' if item.photo else '-'} |")
        photos = []
        for item in items:
            reading = self.screener_readings.get((area_id, item.item_id))
            if item.photo and reading is not None:
                photos.append(f"- {item.item_id}: {reading}")
        photo_text = ("The photo screener gives a first indication of what a photo shows, for an explosive ordnance "
                      "disposal (EOD) technician to confirm. It is not evidence, and it decides nothing.\n"
                      + ("\n".join(photos) if photos else "- No photos."))
        return {"area_record": record, "desk_study": "\n".join(desk), "evidence_log": "\n".join(log),
                "photos": photo_text, "national_criteria": self.criteria.as_text()}

    def map_layer(self, values, municipality_code):
        """A grid of one municipality's cells, with each cell's value (from a Series by area or by cell).

        Rows run north to south and columns west to east. The grid carries no coordinates: plotted with no axes,
        it shows the shape of the municipality and where values lie within it, and nothing that locates it.
        """
        part = self.cells[self.cells["municipality_code"] == municipality_code]
        per_cell = values.reindex(part["area_id"]).to_numpy() if values.index.isin(self.areas.index).all() \
            else values.reindex(part.index).to_numpy()
        rows, cols = part["row"] - part["row"].min(), part["col"] - part["col"].min()
        grid = np.full((rows.max() + 1, cols.max() + 1), np.nan)
        grid[rows.max() - rows.to_numpy(), cols.to_numpy()] = per_cell
        return grid
