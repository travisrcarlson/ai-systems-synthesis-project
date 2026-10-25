"""The season planner: which zones each kind of team works, month by month, for the rest of the season
(optimisation: integer programming with SciPy's `milp`), and the rule-of-thumb plan it is compared with (P3's
insight: a simple baseline first; P2's: compare plans by their outcomes, with ranges).

The integer programme, from the twin's state, the risk overlay's chances and the planning factors:
- decides, for each candidate zone and each month left, whether a survey team works there (at most one), and how
  many technical survey and clearance teams do;
- within the teams available each month (a team stood down is available again from the week its operator gave),
  and only in months the zone can be tasked (open by the access feed; consulted, where consultation is required,
  after the consultation's lead time);
- with the working days a team can expect in each month (rain and wind take about half of them in March, April,
  October and November, one in ten otherwise);
- where a zone's clearance and technical survey become available as its survey goes on (in step with the share of
  the survey done; work already known can start at once);
- to maximise the expected mine areas found and cleared this season, with zones declared free as a small tie-break
  (this project's setting, `Season.zone_free_value`).
It is solved by HiGHS through SciPy, with a time limit and a 1% optimality gap. The candidates are the zones in
progress and the zones with the most expected mine areas per working day, up to `candidates` in all.

The rule of thumb works the zones with the most expected mine areas first, finishing what is in progress, with no
weighing of effort, timing or access. Both plans become ordered lists of zones for each kind of team, which the
weekly loop follows; a team with nothing left in its list takes the zone with the most expected mine areas per day.
"""
import contextlib
import ctypes
import math
import os
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .factors import zone_workloads
from .overlays import Overlay

KINDS = ("survey", "ts", "clearance")


@contextlib.contextmanager
def quiet_solver():
    """Send the process's C-level standard output to the null device while the solver runs. The HiGHS solver that
    SciPy 1.18 bundles prints a stray debug line ("HighsMipSolverData::transformNewIntegerFeasibleSolution
    tmpSolver.run();") on some problems, whatever SciPy's `disp` option says. Python's own output in a notebook is not
    affected, and where the redirection is not possible the solver runs as it is."""
    try:
        libc = ctypes.CDLL("ucrtbase") if sys.platform == "win32" else ctypes.CDLL(None)
        saved = os.dup(1)
    except Exception:
        libc = saved = None
    if saved is None:
        yield
        return
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        libc.fflush(None)
        os.dup2(null, 1)
        yield
    finally:
        try:
            libc.fflush(None)
            sys.stdout.flush()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.close(null)


def months_left(view, week):
    """The months from this week to the season's end, with the weeks left in each."""
    cal = view.calendar
    first = cal.month(week)
    return [(m, sum(1 for w in cal.weeks_of[m] if w >= week)) for m in range(first, 13)]


def team_months(view, week, months):
    """The teams of each kind available in each month: active teams, and stood-down teams from the month their
    operator expects them back."""
    t = view.teams.join(view.team_state)
    cal = view.calendar
    out = {k: [] for k in KINDS}
    for m, _ in months:
        first = max(week, cal.first_week(m))
        for k in KINDS:
            mine = t[t["kind"] == k]
            back = mine["back_week"].fillna(10 ** 6)
            n = int(((mine["status"] == "active") | ((mine["status"] == "stood down") & (back <= first))).sum())
            out[k].append(n)
    return out


def first_open_month(view, week, zones):
    """The first month each zone can be tasked: after the consultation's lead time where communities must first be
    consulted (counting from now, if not yet requested), and not this month if the access feed shows it closed."""
    cal, season = view.calendar, view.season
    c, access = view.consultation, view.access
    out = {}
    for z in zones:
        month = cal.month(week)
        if c.loc[z, "required"] and not c.loc[z, "held_week"]:
            asked = c.loc[z, "requested_week"] or week
            ready = min(season.weeks, asked + season.consultation_weeks)
            month = max(month, cal.month(ready) + (0 if cal.first_week(cal.month(ready)) == ready else 1))
        closed_now = access.loc[z, "status"] != "open"
        out[z] = (month, closed_now)
    return out


class SeasonPlanner(Overlay):
    name = "season planner"
    domain = "optimisation (integer programming with SciPy's milp)"
    insight = "P3: a simple plan first; P2: plans compared on their outcomes, with ranges"
    reads = ("hazard", "land_use", "access", "consultation", "rainfall", "teams")
    limits = ("plans with expected workloads and yields from the planning factors, not with the truth",
              "moving between zones takes no time in the plan or the weekly loop",
              "in the plan, a zone's clearance work appears in step with its survey; in the weekly loop it appears "
              "area by area, as each visit is reviewed")

    def __init__(self, factors, method="milp", candidates=60, time_limit=20.0, gap=0.01):
        self.factors = factors
        self.method = method
        self.candidates = candidates
        self.time_limit = time_limit
        self.gap = gap

    def recommend(self, view, week, risk=None, **inputs):
        reason = self.check_inputs(view, week)
        if reason:
            return self.insufficient(week, reason)
        if risk is None or not risk.ok:
            return self.insufficient(week, "no risk estimate" + (f" ({risk.reason})" if risk is not None else ""))
        chances = risk.answer["areas"]["chance"]
        work = zone_workloads(view, chances, self.factors, view.criteria, week)
        months = months_left(view, week)
        limits = []
        if self.method == "rule":
            plan = self.rule_plan(view, week, work, months)
        else:
            plan = self.milp_plan(view, week, work, months)
            if plan is None:
                limits.append("the integer programme found no plan in its time limit; the rule of thumb stands")
                plan = self.rule_plan(view, week, work, months)
        # a zone with an accident newly reported in it comes first in every list: an urgent response
        urgent = list(work.index[work["urgent"]])
        if urgent:
            for kind in KINDS:
                plan[kind] = urgent + [z for z in plan[kind] if z not in urgent]
            limits.append("first: the zone(s) with an accident newly reported: " + ", ".join(urgent))
        plan["urgent"] = urgent
        plan["priority"] = risk.answer["areas"][["zone_rank", "ranking", "chance"]].copy()
        plan["ranking"] = risk.answer["areas"]["ranking"].value_counts().to_dict()
        evidence = ("risk prioritisation (week %d)" % risk.week, "planning factors", "access feed",
                    "consultation layer", "rainfall pattern", "team reports")
        uncertainty = "expected values only; the completion forecaster gives the ranges"
        return self.answer(week, plan, evidence, uncertainty, limits)

    # ---- the rule of thumb ------------------------------------------------------------------------------------------
    def rule_plan(self, view, week, work, months):
        """Zones in progress first, then the zones with the most expected mine areas, for every kind of team."""
        in_progress = work["surveyed"] | work.index.isin([t["zone"] for t in view.tasks.values()
                                                          if t["status"] in ("ordered", "in progress", "paused")])
        order = work.assign(in_progress=in_progress).sort_values(
            ["in_progress", "expected_mine_areas", "value"], ascending=[False, False, False]).index
        lists = {"survey": [z for z in order if work.loc[z, "survey_days"] > 0],
                 "ts": [z for z in order if work.loc[z, ["ts_known", "ts_future"]].sum() > 0],
                 "clearance": [z for z in order if work.loc[z, ["clearance_known", "clearance_future"]].sum() > 0]}
        teams = team_months(view, week, months)
        n = 2 * (teams["survey"][0] + teams["clearance"][0] + teams["ts"][0])
        consult = [z for z in order[:n] if view.consultation.loc[z, "required"]
                   and not view.consultation.loc[z, "requested_week"]]
        return {"planner": "rule of thumb", "week": week, **lists, "fallback": self.fallback(work),
                "months": {}, "expected": {}, "workloads": work, "consultations": consult,
                "solver": {"status": "rule"}}

    def fallback(self, work):
        return list(work.sort_values("value_per_day", ascending=False).index)

    # ---- the integer programme --------------------------------------------------------------------------------------
    def milp_plan(self, view, week, work, months):
        season = view.season
        ts_policy = view.criteria.technical_survey
        tasks_zones = {t["zone"] for t in view.tasks.values() if t["status"] in ("ordered", "in progress", "paused")}
        progress = work.index[work["surveyed"] | work.index.isin(tasks_zones)]
        rest = work.drop(progress).sort_values("value_per_day", ascending=False)
        cand = list(progress) + list(rest.index[:max(0, self.candidates - len(progress))])
        W = work.loc[cand]
        teams = team_months(view, week, months)
        open_from = first_open_month(view, week, cand)
        L = len(months)
        D = np.array([n * season.days_per_week * (1 - season.day_loss(m)) for m, n in months])
        cols = {}

        def col(*key):
            if key not in cols:
                cols[key] = len(cols)
            return cols[key]

        lb, ub, integ, obj = [], [], [], []

        def var(key, low, high, integer, cost=0.0):
            j = col(*key)
            lb.append(low), ub.append(high), integ.append(1 if integer else 0), obj.append(cost)
            return j

        avail = {}
        for z in cand:
            start, closed_now = open_from[z]
            avail[z] = [(m >= start) and not (closed_now and i == 0) for i, (m, _) in enumerate(months)]
        cmax = max(teams["clearance"]) if teams["clearance"] else 0
        tmax = max(teams["ts"]) if teams["ts"] else 0
        has = {}
        for z in cand:
            w = W.loc[z]
            Ws, Wck, Wcf = w["survey_days"], w["clearance_known"], w["clearance_future"]
            Wtk, Wtf = w["ts_known"], w["ts_future"]
            has[z] = {"s": Ws > 0.5, "c": (Wck + Wcf) > 0.5 and cmax > 0, "t": ts_policy and (Wtk + Wtf) > 0.5 and tmax > 0}
            cap_c = int(min(cmax, max(1, math.ceil(w["clearance_tasks"])))) if has[z]["c"] else 0
            cap_t = int(min(tmax, max(1, math.ceil(w["ts_tasks"])))) if has[z]["t"] else 0
            for i in range(L):
                ok = avail[z][i]
                if has[z]["s"]:
                    var(("s", z, i), 0, 1 if ok else 0, True)
                    var(("u", z, i), 0, 1, False)        # the share of the zone's survey done by the month's end
                if has[z]["c"]:
                    var(("c", z, i), 0, cap_c if ok else 0, True)
                if has[z]["t"]:
                    var(("t", z, i), 0, cap_t if ok else 0, True)
            if has[z]["c"]:
                var(("f", z), 0, 1, False, w["value"])
            if has[z]["t"]:
                var(("g", z), 0, 1, False, 0.0 if has[z]["c"] else w["value"])
            var(("y", z), 0, 1, True, season.zone_free_value)
        rows, cidx, vals, lo, hi = [], [], [], [], []

        def constraint(terms, low, high):
            r = len(lo)
            for j, v in terms:
                rows.append(r), cidx.append(j), vals.append(v)
            lo.append(low), hi.append(high)

        # the teams available each month
        for i in range(L):
            for kind, name in (("survey", "s"), ("clearance", "c"), ("ts", "t")):
                terms = [(cols[(name, z, i)], 1.0) for z in cand if (name, z, i) in cols]
                if terms:
                    constraint(terms, -np.inf, teams[kind][i])
        for z in cand:
            w = W.loc[z]
            Ws, Wck, Wcf = w["survey_days"], w["clearance_known"], w["clearance_future"]
            Wtk, Wtf = w["ts_known"], w["ts_future"]
            for i in range(L):
                if has[z]["s"]:
                    # the share of the survey done by the end of month i is at most the survey days worked by then
                    constraint([(cols[("u", z, i)], Ws)] + [(cols[("s", z, k)], -D[k]) for k in range(i + 1)],
                               -np.inf, 0.0)
                for name, known, future, flag in (("c", Wck, Wcf, "c"), ("t", Wtk, Wtf, "t")):
                    if not has[z][flag]:
                        continue
                    terms = [(cols[(name, z, k)], D[k]) for k in range(i + 1)]
                    if has[z]["s"]:
                        # work found by the survey becomes available as the survey goes on: work already known
                        # at once, the rest in step with the share of the survey done
                        terms.append((cols[("u", z, i)], -future))
                        constraint(terms, -np.inf, known)
                    else:
                        constraint(terms, -np.inf, known + future)
            if has[z]["c"]:
                total = Wck + Wcf
                constraint([(cols[("f", z)], total)] + [(cols[("c", z, k)], -D[k]) for k in range(L)], -np.inf, 0.0)
                if has[z]["t"] and Wcf > 0:
                    # under technical survey first, clearance found by technical survey waits for it
                    constraint([(cols[("f", z)], total), (cols[("g", z)], -Wcf)], -np.inf, Wck)
            if has[z]["t"]:
                constraint([(cols[("g", z)], Wtk + Wtf)] + [(cols[("t", z, k)], -D[k]) for k in range(L)], -np.inf,
                           0.0)
            y = cols[("y", z)]
            if has[z]["s"]:
                constraint([(y, 1.0), (cols[("u", z, L - 1)], -1.0)], -np.inf, 0.0)
            if has[z]["c"]:
                constraint([(y, 1.0), (cols[("f", z)], -1.0)], -np.inf, 0.0)
            if has[z]["t"]:
                constraint([(y, 1.0), (cols[("g", z)], -1.0)], -np.inf, 0.0)
        n = len(cols)
        A = coo_matrix((vals, (rows, cidx)), shape=(len(lo), n)).tocsr()
        with quiet_solver():
            res = milp(c=-np.array(obj), integrality=np.array(integ), bounds=Bounds(np.array(lb), np.array(ub)),
                       constraints=LinearConstraint(A, np.array(lo), np.array(hi)),
                       options={"time_limit": self.time_limit, "mip_rel_gap": self.gap, "disp": False})
        if res.x is None:
            return None
        x = res.x
        plan_months = {z: {"survey": [], "clearance": [], "ts": []} for z in cand}
        for (name, *key), j in cols.items():
            if name in ("s", "c", "t") and x[j] > 0.5:
                z, i = key
                kind = {"s": "survey", "c": "clearance", "t": "ts"}[name]
                plan_months[z][kind].append(months[i][0])
        first = {kind: {z: min(v[kind]) for z, v in plan_months.items() if v[kind]} for kind in KINDS}
        lists = {}
        for kind in KINDS:
            planned = sorted(first[kind], key=lambda z: (first[kind][z], -W.loc[z, "value_per_day"]))
            lists[kind] = planned
        value = sum(W.loc[z, "value"] * x[cols[("f", z)]] for z in cand if ("f", z) in cols) + \
            sum(W.loc[z, "value"] * x[cols[("g", z)]] for z in cand if ("g", z) in cols and ("f", z) not in cols)
        zones_free = sum(x[cols[("y", z)]] > 0.5 for z in cand)
        # consultations are requested now for the planned zones that need them (the programme plans such a zone
        # only from the month its consultation can be held)
        consult = [z for z in cand if any(plan_months[z].values()) and view.consultation.loc[z, "required"]
                   and not view.consultation.loc[z, "requested_week"]]
        return {"planner": "integer programme", "week": week, **lists, "fallback": self.fallback(work),
                "months": {z: v for z, v in plan_months.items() if any(v.values())},
                "expected": {"mine areas cleared": float(value), "zones declared free": int(zones_free)},
                "workloads": work, "consultations": consult,
                "solver": {"status": res.status, "message": res.message, "gap": getattr(res, "mip_gap", None),
                           "variables": n, "constraints": len(lo), "candidates": len(cand)}}
