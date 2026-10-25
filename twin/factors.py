"""Planning factors: what a flagged area is expected to cost and yield, for the planner and the forecaster.

A planner knows the programme's figures, not the truth: how many visits a flagged area takes, and how often a
flagged area that holds mines, or one that does not, ends in cancellation, technical survey or clearance. Here
those figures come from simulated visits under Travis's settings (questions 13 to 23), reviewed by the coded rules
under the national criteria. No RELand label is read: each simulated area is assumed to hold mines, or assumed
clean, and its land use and desk study are its own. The factors are by whether the area holds mines and whether
an accident is recorded inside it; the risk overlay's chances then weigh the two.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SEED, Assumptions
from .core import desk_item
from .field import Field
from .rules import reference_decision

OUTCOMES = ["CANCEL", "TECHNICAL_SURVEY", "CLEARANCE", "FURTHER_NTS"]


@dataclass
class Factors:
    """The planning factors, one row per (mined, recorded): the mean visits until a decision, the chances of each
    outcome, and the chance that technical survey finds mines (where it is used); `sims` keeps every simulated area
    (its visits, outcome, whether its evidence held direct evidence, and the technical survey's finding), so the
    forecaster can draw them together."""
    table: pd.DataFrame
    sims: pd.DataFrame
    max_visits: int
    flag_mined: float
    flag_clean: float

    def __post_init__(self):
        # the table as plain lookups and arrays indexed [mined][recorded], so the planner's loops stay fast
        self._rows = {key: {k: float(v) for k, v in self.table.loc[key].items()} for key in self.table.index}
        self._arrays = {col: np.array([[self.table.loc[(m, r), col] for r in (False, True)] for m in (False, True)])
                        for col in self.table.columns}

    def row(self, mined, recorded):
        return self._rows[(bool(mined), bool(recorded))]

    def expected(self, chance_mined, recorded):
        """For flagged areas with these chances of holding mines: expected visits, technical surveys, clearances and
        mine areas cleared, and the chance a flagged area with mines is released without an intrusive check."""
        r = np.asarray(recorded).astype(int)
        q = np.asarray(chance_mined, dtype=float)
        a = self._arrays
        m = {k: a[k][1][r] for k in ("visits", "TECHNICAL_SURVEY", "CLEARANCE", "CANCEL", "TS_FOUND")}
        c = {k: a[k][0][r] for k in ("visits", "TECHNICAL_SURVEY", "CLEARANCE", "CANCEL")}
        # clearance follows a decision of clearance, or technical survey that found mines
        mine_cleared = q * m["CLEARANCE"] + q * m["TECHNICAL_SURVEY"] * m["TS_FOUND"]
        return {"visits": q * m["visits"] + (1 - q) * c["visits"],
                "ts": q * m["TECHNICAL_SURVEY"] + (1 - q) * c["TECHNICAL_SURVEY"],
                "clearance": q * m["CLEARANCE"] + (1 - q) * c["CLEARANCE"] + q * m["TECHNICAL_SURVEY"] * m["TS_FOUND"],
                "mine_cleared": mine_cleared, "mine_released": q * m["CANCEL"]}


def planning_factors(twin, criteria, assumptions=None, season=None, n=300, seed=SEED, max_visits=6):
    """Simulate `n` flagged areas of each kind (holding mines or not; an accident recorded inside or not), each
    visited until the coded rules decide (at most `max_visits` visits), and return the planning factors."""
    assumptions = assumptions or Assumptions()
    areas = twin.areas
    pick = np.random.default_rng([seed, 17])
    rows = []
    for mined in (True, False):
        truth = pd.Series(int(mined), index=twin.cells.index, name="mines")
        field = Field(twin, truth, assumptions, season, seed=10_000 + seed)
        for recorded in (True, False):
            pool = areas.index[areas["event_inside"] == recorded]
            for a in pick.choice(pool, size=min(n, len(pool)), replace=False):
                items = [desk_item()] if recorded else []
                visits, outcome = 0, "FURTHER_NTS"
                while outcome == "FURTHER_NTS" and visits < max_visits:
                    visits += 1
                    items += field.nts(a, visits, 0, len(items) + 1)
                    outcome, _, _ = reference_decision(items, field.key_for(a, items), criteria)
                found = field.technical_survey(a) if outcome == "TECHNICAL_SURVEY" else np.nan
                # a decision of clearance on direct evidence makes the area a confirmed hazardous area, which the
                # weekly loop clears first
                direct = any(k["class"] == "direct" for k in field.key_for(a, items).values())
                rows.append({"mined": mined, "recorded": recorded, "visits": visits, "outcome": outcome,
                             "found": found, "direct": direct})
    sims = pd.DataFrame(rows)
    g = sims.groupby(["mined", "recorded"])
    table = pd.DataFrame({"areas": g.size(), "visits": g["visits"].mean()})
    for o in OUTCOMES:
        table[o] = g["outcome"].apply(lambda s, o=o: float((s == o).mean()))
    table["TS_FOUND"] = g["found"].mean().fillna(0.0)
    season = season or twin.season
    return Factors(table, sims, max_visits, season.flag_mined, season.flag_clean)


def flag_chance(chance, recorded, factors):
    """The chance a zone survey flags an area: 1 with a recorded accident; else 85% if it holds mines, 5% if not."""
    return np.where(recorded, 1.0, factors.flag_mined * chance + factors.flag_clean * (1 - chance))


def mined_if_flagged(chance, recorded, factors):
    """The chance a flagged area holds mines (Bayes' rule on the flagging rates; a recorded area is always flagged)."""
    flagged = factors.flag_mined * chance / (factors.flag_mined * chance + factors.flag_clean * (1 - chance))
    return np.where(recorded, chance, flagged)


def zone_workloads(view, chances, factors, criteria, week):
    """The work left in each zone not declared free, in working days, and its expected yield, from the twin's state,
    the risk overlay's chances and the planning factors.

    For each zone: the survey days left (the zone survey, then the visits), the days of technical survey and of
    clearance known now (areas already sent to them) and expected later (from areas still to be visited or flagged),
    the expected mine areas cleared by that work, and how many teams of each kind could work there at once.
    """
    s = view.season
    areas = view.areas
    status, next_task, visits, flags = view.status, view.next_task, view.visits, view.flag
    zone_state, tasks = view.zone_state, view.tasks
    pending = view.pending_results
    done = {}                                    # (zone or area, kind) -> days reported on open tasks
    for t in tasks.values():
        if t["status"] in ("ordered", "in progress", "paused"):
            done[(t["area"] or t["zone"], t["kind"])] = t["days_done"]
    p = chances.reindex(areas.index).to_numpy()
    rec = areas["event_inside"].to_numpy()
    ts = criteria.technical_survey
    rows = []
    zone_ids = view.zones.index[zone_state["free_week"] == 0]
    position = {a: i for i, a in enumerate(areas.index)}
    for z in zone_ids:
        ids = list(view.zone_areas[z])
        idx = np.array([position[a] for a in ids])
        pz, rz = p[idx], rec[idx]
        surveyed = zone_state.loc[z, "surveyed_week"] > 0
        survey_left = 0.0 if surveyed else max(0.0, s.task_days("ZONE_SURVEY") - done.get((z, "ZONE_SURVEY"), 0))
        visit_days = ts_known = ts_future = clr_known = clr_future = 0.0
        value_known = value_future = 0.0
        n_clear = n_ts = 0.0
        if not surveyed:
            flag = flag_chance(pz, rz, factors)
            q = mined_if_flagged(pz, rz, factors)
            e = factors.expected(q, rz)
            visit_days = float((flag * e["visits"]).sum()) * s.task_days("VISIT")
            n_ts_f = float((flag * e["ts"]).sum())
            n_clr_f = float((flag * e["clearance"]).sum())
            ts_future = n_ts_f * s.task_days("TS")
            clr_future = n_clr_f * s.task_days("CLEARANCE")
            value_future = float((flag * e["mine_cleared"]).sum())
            n_clear, n_ts = n_clr_f, n_ts_f
        else:
            for a, pa, ra in zip(ids, pz, rz):
                st, nt = status[a], next_task[a]
                if st not in ("suspected", "confirmed") or a in pending:
                    continue
                q = float(mined_if_flagged(np.array([pa]), np.array([ra]), factors)[0])
                if flags[a] == "incident report":
                    q = 1.0           # an accident reported with its place is direct evidence (IMAS 08.10, §6.1)
                row_m, row_c = factors.row(True, ra), factors.row(False, ra)
                if nt in ("NTS", None):
                    # still to be visited, or awaiting review: the rest of its expected visits, then its outcome
                    left_m = max(1.0 if nt == "NTS" else 0.0, row_m["visits"] - visits[a])
                    left_c = max(1.0 if nt == "NTS" else 0.0, row_c["visits"] - visits[a])
                    visit_days += (q * left_m + (1 - q) * left_c) * s.task_days("VISIT") - done.get((a, "VISIT"), 0)
                    e = factors.expected(np.array([q]), np.array([ra]))
                    ts_future += float(e["ts"][0]) * s.task_days("TS")
                    clr_future += float(e["clearance"][0]) * s.task_days("CLEARANCE")
                    value_future += float(e["mine_cleared"][0])
                    n_clear += float(e["clearance"][0])
                    n_ts += float(e["ts"][0])
                elif nt == "TS":
                    qt = q * row_m["TECHNICAL_SURVEY"] / max(1e-9, q * row_m["TECHNICAL_SURVEY"]
                                                                + (1 - q) * row_c["TECHNICAL_SURVEY"])
                    ts_known += max(0.0, s.task_days("TS") - done.get((a, "TS"), 0))
                    clr_future += qt * row_m["TS_FOUND"] * s.task_days("CLEARANCE")
                    value_future += qt * row_m["TS_FOUND"]
                    n_ts += 1
                    n_clear += qt * row_m["TS_FOUND"]
                elif nt == "CLEARANCE":
                    if st == "confirmed":
                        qc = 1.0                     # direct evidence, or technical survey found mines
                    else:
                        qc = q * row_m["CLEARANCE"] / max(1e-9, q * row_m["CLEARANCE"]
                                                          + (1 - q) * row_c["CLEARANCE"])
                    clr_known += max(0.0, s.task_days("CLEARANCE") - done.get((a, "CLEARANCE"), 0))
                    value_known += qc
                    n_clear += 1
        urgent = any(flags[a] == "incident report" and status[a] in ("suspected", "confirmed") for a in ids)
        rows.append({"zone_id": z, "surveyed": surveyed, "urgent": urgent,
                     "survey_days": survey_left + max(0.0, visit_days),
                     "ts_known": ts_known, "ts_future": ts_future if ts else 0.0,
                     "clearance_known": clr_known, "clearance_future": clr_future,
                     "value_known": value_known, "value_future": value_future,
                     "clearance_tasks": n_clear, "ts_tasks": n_ts if ts else 0.0,
                     "expected_mine_areas": float(pz[[status[a] in ("unknown", "suspected", "confirmed")
                                                       for a in ids]].sum())})
    out = pd.DataFrame(rows).set_index("zone_id")
    out["value"] = out["value_known"] + out["value_future"]
    out["days"] = out["survey_days"] + out["ts_known"] + out["ts_future"] + out["clearance_known"] + \
        out["clearance_future"]
    out["value_per_day"] = out["value"] / out["days"].clip(lower=1)
    return out
