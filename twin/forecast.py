"""The completion forecaster: when each zone in the plan could be declared free, as P50 and P90 weeks, and what the
season could deliver, as ranges (statistics: Monte Carlo simulation; P2's insight: show uncertainty with intervals).

Each draw plays the rest of the season at the level of zones and areas, following the plan's lists the way the
weekly loop does, with what the twin does not know drawn afresh:
- which areas hold mines (the risk overlay's chances, updated by the zone survey's flag where an area is flagged)
  and which areas the zone survey flags (Travis's rates);
- each flagged area's visits and outcome, drawn together from the planning factors' simulated areas;
- the working days each team gets each week (days lost to rain and wind at the month's rate), the time each
  technical survey and clearance takes (lognormal around the planned time), and which zones close (1 in 50 a month,
  for one to three months; a zone closed now stays closed for one to three months);
- the teams available: the active teams, and stood-down teams from the week their operator expects them back.
It answers with each zone's P50 and P90 weeks of completion (or "after the season"), the chance it completes this
season, and the season's mine areas cleared and zones declared free as P10, P50 and P90. A zone the plan does not
reach this season gets no date.
"""
import math

import numpy as np
import pandas as pd

from .config import SEED
from .factors import mined_if_flagged
from .overlays import Overlay


class CompletionForecaster(Overlay):
    name = "completion forecaster"
    domain = "statistics (Monte Carlo simulation)"
    insight = "P2: uncertainty shown as intervals"
    reads = ("hazard", "land_use", "access", "consultation", "rainfall", "teams")
    limits = ("draws the unknowns from the planning factors and the risk overlay's chances, so it is only as good as "
              "they are; its coverage is measured in the season replay",
              "no new funding shock or other event is drawn; a forecast is remade when one happens")

    def __init__(self, factors, draws=200, seed=SEED, zones=60):
        self.factors = factors
        self.draws = draws
        self.seed = seed
        self.max_zones = zones
        sims = factors.sims
        self.rows = {key: part[["visits", "outcome", "found"]].to_numpy()
                     for key, part in sims.groupby(["mined", "recorded"])}

    def recommend(self, view, week, plan=None, risk=None, **inputs):
        reason = self.check_inputs(view, week)
        if reason:
            return self.insufficient(week, reason)
        if plan is None:
            return self.insufficient(week, "no approved plan to forecast")
        if risk is None or not risk.ok:
            return self.insufficient(week, "no risk estimate")
        chances = risk.answer["areas"]["chance"]
        order = []
        for z in plan["survey"] + plan["clearance"] + plan["ts"] + plan["fallback"]:
            if z not in order and not view.zone_state.loc[z, "free_week"]:
                order.append(z)
        zones = order[:self.max_zones]
        result = self.simulate(view, week, plan, zones, chances)
        weeks = result["completion"]          # draws x zones, np.inf where not complete
        # percentiles without interpolation, so that a zone done in fewer than half the draws has no P50 week
        table = pd.DataFrame({"P50 week": np.percentile(weeks, 50, axis=0, method="inverted_cdf"),
                              "P90 week": np.percentile(weeks, 90, axis=0, method="inverted_cdf"),
                              "chance this season": np.isfinite(weeks).mean(axis=0)}, index=zones)
        for col in ("P50 week", "P90 week"):
            table[col] = table[col].where(np.isfinite(table[col]))
        table["municipality"] = view.zones.loc[zones, "municipality"].to_numpy()
        totals = {}
        for name, values in (("mine areas cleared", result["cleared"]), ("zones declared free", result["free"])):
            totals[name] = {f"P{q}": float(np.percentile(values, q, method="inverted_cdf")) for q in (10, 50, 90)}
        by_municipality = pd.DataFrame(np.isfinite(weeks), columns=zones).T.groupby(table["municipality"]).sum().T
        municipal = pd.DataFrame({f"zones free P{q}": np.percentile(by_municipality, q, axis=0, method="inverted_cdf")
                                  for q in (10, 50, 90)}, index=by_municipality.columns)
        n_zones = view.zones.groupby("municipality").size()
        free_now = (view.zone_state["free_week"] > 0).groupby(view.zones["municipality"]).sum()
        municipal["zones"] = n_zones.reindex(municipal.index)
        municipal["already free"] = free_now.reindex(municipal.index)
        complete = (by_municipality.to_numpy() + free_now.reindex(by_municipality.columns).to_numpy()
                    >= n_zones.reindex(by_municipality.columns).to_numpy())
        municipal["chance declared free this season"] = complete.mean(axis=0)
        return self.answer(week, {"zones": table, "totals": totals, "municipalities": municipal,
                                  "draws": self.draws},
                           evidence=(f"plan {plan.get('plan_id', '(proposed)')}", "risk prioritisation",
                                     "planning factors", "rainfall pattern", "access feed", "team reports"),
                           uncertainty="P50 and P90 weeks by zone; P10 to P90 for the season's totals")

    # ---- the simulation ---------------------------------------------------------------------------------------------
    def simulate(self, view, week, plan, zones, chances):
        season, cal = view.season, view.calendar
        rng = np.random.default_rng([self.seed, week, len(view.log_table())])
        access, consultation = view.access, view.consultation
        teams = view.teams.join(view.team_state)
        weeks = list(range(week, season.weeks + 1))
        loss = np.array([season.day_loss(cal.month(w)) for w in weeks])
        sigma = season.duration_spread
        in_plan = set(zones)
        kinds = {k: [z for z in plan[k] + [z for z in plan["fallback"] if z not in plan[k]] if z in in_plan]
                 for k in ("survey", "clearance", "ts")}
        zone_index = {z: i for i, z in enumerate(zones)}
        ready = {}
        for z in zones:
            r = week
            if consultation.loc[z, "required"] and not consultation.loc[z, "held_week"]:
                r = (consultation.loc[z, "requested_week"] or week) + season.consultation_weeks
            ready[z] = r
        static = self._static(view, zones, chances)
        team_rows = []
        for _, row in teams.iterrows():
            back = row["back_week"]
            start = week if row["status"] == "active" else (int(back) if back == back else None)
            if start is not None:
                team_rows.append((row["kind"], start))
        completion = np.full((self.draws, len(zones)), np.inf)
        cleared = np.zeros(self.draws)
        free = np.zeros(self.draws)
        for d in range(self.draws):
            work = self._draw_work(rng, static, season, sigma)
            closed = self._draw_closures(rng, zones, weeks, access, season, cal)
            team_list = [{"kind": kind, "start": start, "zone": None, "item": None} for kind, start in team_rows]
            done_week = {}
            for wi, w in enumerate(weeks):
                for team in team_list:
                    if w < team["start"]:
                        continue
                    days = int(rng.binomial(season.days_per_week, 1 - loss[wi]))
                    while days > 0:
                        if team["item"] is None or closed[zone_index[team["zone"]]][wi]:
                            if team["item"] is not None:
                                team["item"]["taken"] = False      # paused: the progress stays with the item
                            team["zone"], team["item"] = self._dispatch(team["kind"], kinds, work, closed, wi,
                                                                        zone_index, ready, w, team_list)
                            if team["item"] is None:
                                break
                        item = team["item"]
                        use = min(days, item["left"])
                        item["left"] -= use
                        days -= use
                        if item["left"] <= 0:
                            z = team["zone"]
                            self._finish(item, work[z])
                            team["item"] = None
                            if z not in done_week and self._complete(work[z]):
                                done_week[z] = w
                for z in zones:
                    if z not in done_week and self._complete(work[z]):
                        done_week[z] = w
            for z, w in done_week.items():
                completion[d, zone_index[z]] = w
            cleared[d] = sum(zw["cleared_mines"] for zw in work.values())
            free[d] = len(done_week)
        return {"completion": completion, "cleared": cleared, "free": free}

    def _static(self, view, zones, chances):
        """What the twin knows about each zone's areas, prepared once for all the draws."""
        season = view.season
        areas = view.areas
        status, next_task, visits = view.status, view.next_task, view.visits
        zstate = view.zone_state
        done = {}
        for t in view.tasks.values():
            if t["status"] in ("ordered", "in progress", "paused"):
                done[(t["area"] or t["zone"], t["kind"])] = t["days_done"]
        static = {}
        for z in zones:
            ids = list(view.zone_areas[z])
            surveyed = zstate.loc[z, "surveyed_week"] > 0
            if surveyed:
                ids = [a for a in ids if status[a] in ("suspected", "confirmed")]
            else:
                ids = [a for a in ids if status[a] in ("unknown", "suspected")]
            p = chances.reindex(ids).to_numpy(dtype=float)
            rec = areas.loc[ids, "event_inside"].to_numpy(dtype=bool)
            q = mined_if_flagged(p, rec, self.factors)
            if surveyed:
                # the chance each flagged area holds mines, given what has been decided on it so far
                for i, a in enumerate(ids):
                    if status[a] == "confirmed":
                        q[i] = 1.0
                    elif next_task[a] in ("CLEARANCE", "TS"):
                        key = "CLEARANCE" if next_task[a] == "CLEARANCE" else "TECHNICAL_SURVEY"
                        m, c = self.factors.row(True, rec[i])[key], self.factors.row(False, rec[i])[key]
                        q[i] = q[i] * m / max(1e-9, q[i] * m + (1 - q[i]) * c)
            static[z] = {"surveyed": surveyed, "ids": ids, "p": p, "rec": rec, "q": q,
                         "status": [status[a] for a in ids], "next": [next_task[a] for a in ids],
                         "visits": [int(visits[a]) for a in ids], "done": {a: {k: done.get((a, k), 0)
                                                                              for k in ("VISIT", "TS", "CLEARANCE")}
                                                                          for a in ids},
                         "survey_left": max(1, season.task_days("ZONE_SURVEY") - int(done.get((z, "ZONE_SURVEY"), 0)))}
        return static

    def _sample(self, rng, mined, recorded):
        rows = self.rows[(bool(mined), bool(recorded))]
        return rows[rng.integers(len(rows))]

    def _duration(self, rng, kind, season, sigma, done=0):
        planned = season.task_days(kind)
        if kind in ("TS", "CLEARANCE"):
            planned = planned * math.exp(sigma * rng.standard_normal() - sigma ** 2 / 2)
        return max(1, int(round(planned)) - int(done))

    def _draw_work(self, rng, static, season, sigma):
        """One draw of the work left in each zone: the zone survey, and each flagged area's queue of tasks."""
        work = {}
        for z, st in static.items():
            n = len(st["ids"])
            items = []
            zw = {"survey": None, "areas": items, "cleared_mines": 0}
            if not st["surveyed"]:
                zw["survey"] = {"kind": "ZONE_SURVEY", "left": st["survey_left"], "taken": False}
                mined = rng.random(n) < st["p"]
                flag_p = np.where(mined, self.factors.flag_mined, self.factors.flag_clean)
                flagged = st["rec"] | (rng.random(n) < flag_p)
            else:
                mined = rng.random(n) < st["q"]
                flagged = np.ones(n, dtype=bool)
            for i in np.flatnonzero(flagged):
                a, rec, m = st["ids"][i], bool(st["rec"][i]), bool(mined[i])
                v, outcome, _ = self._sample(rng, m, rec)
                queue = ["VISIT"] * int(v)
                if st["surveyed"]:
                    nt = st["next"][i]
                    if nt == "CLEARANCE":
                        queue, outcome = [], "CLEARANCE"
                    elif nt == "TS":
                        queue, outcome = [], "TECHNICAL_SURVEY"
                    else:
                        queue = ["VISIT"] * max(1 if nt == "NTS" else 0, int(v) - st["visits"][i])
                if outcome == "TECHNICAL_SURVEY":
                    queue.append("TS")
                    if m and rng.random() < self.factors.row(True, rec)["TS_FOUND"]:
                        queue.append("CLEARANCE")
                elif outcome == "CLEARANCE":
                    queue.append("CLEARANCE")
                elif outcome == "FURTHER_NTS":
                    queue.append("VISIT")         # still open after the factors' last visit: one more
                tasks = []
                for kind in queue:
                    done = st["done"].get(a, {}).get(kind, 0) if not tasks else 0
                    left = (self._duration(rng, kind, season, sigma, done) if kind in ("TS", "CLEARANCE")
                            else max(1, season.task_days(kind) - int(done)))
                    tasks.append({"kind": kind, "left": left, "taken": False})
                items.append({"area": a, "mined": m, "tasks": tasks, "needs_survey": not st["surveyed"]})
            work[z] = zw
        return work

    def _draw_closures(self, rng, zones, weeks, access, season, cal):
        closed = np.zeros((len(zones), len(weeks)), dtype=bool)
        first = weeks[0]
        for i, z in enumerate(zones):
            if access.loc[z, "status"] != "open":
                length = int(rng.integers(season.closure_months[0], season.closure_months[1] + 1))
                end = min(season.weeks, first + 4 * length)
                closed[i, : end - first + 1] = True
            month = cal.month(first) + 1
            while month <= 12:
                if rng.random() < season.closure_chance:
                    length = int(rng.integers(season.closure_months[0], season.closure_months[1] + 1))
                    for m in range(month, min(12, month + length - 1) + 1):
                        for w in cal.weeks_of[m]:
                            if w >= first:
                                closed[i, w - first] = True
                    month += length
                else:
                    month += 1
        return closed

    def _dispatch(self, kind, kinds, work, closed, wi, zone_index, ready, w, team_list):
        """The next item of work for a team of this kind, from the plan's list for its kind."""
        task_kind = {"survey": ("ZONE_SURVEY", "VISIT"), "clearance": ("CLEARANCE",), "ts": ("TS",)}[kind]
        busy = {t["zone"] for t in team_list if t["kind"] == "survey" and t["item"] is not None}
        for z in kinds[kind]:
            if closed[zone_index[z]][wi] or ready[z] > w:
                continue
            zw = work[z]
            if kind == "survey":
                if z in busy:
                    continue
                if zw["survey"] is not None:
                    if not zw["survey"]["taken"] and zw["survey"]["left"] > 0:
                        zw["survey"]["taken"] = True
                        return z, zw["survey"]
                    continue
            for item in zw["areas"]:
                if item["needs_survey"] and zw["survey"] is not None:
                    break
                if item["tasks"] and item["tasks"][0]["kind"] in task_kind and not item["tasks"][0]["taken"]:
                    task = item["tasks"][0]
                    task["taken"] = True
                    task["owner"] = item
                    return z, task
        return None, None

    def _finish(self, task, zw):
        if task["kind"] == "ZONE_SURVEY":
            zw["survey"] = None
            for item in zw["areas"]:
                item["needs_survey"] = False
            return
        item = task["owner"]
        item["tasks"].pop(0)
        if task["kind"] == "CLEARANCE" and item["mined"]:
            zw["cleared_mines"] += 1

    @staticmethod
    def _complete(zw):
        return zw["survey"] is None and all(not item["tasks"] for item in zw["areas"])
