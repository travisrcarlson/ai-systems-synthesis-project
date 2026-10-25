"""The weekly situation report: a short brief for the operations officer, drafted by an LLM from the twin's facts,
with every figure checked against the facts before it is shown (generative AI; P5's insight: generated output is
checked against its source before anyone relies on it).

The twin assembles the week's facts (work done, teams, access, reports held, the plan, the forecast's range). The
model drafts at most 150 words from them (prompts/sitrep_prompt.md). The figure check then reads every number in the
draft, digits or words, and removes each sentence with a number that is not among the facts; if more than half the
sentences go, the draft is not shown, and the answer is "insufficient evidence" with the facts table instead.

Without a key, a coded draft is written from the same facts (it always passes the check), and saved drafts made with
a key are read when their inputs match. Each draft made with a key is saved in `runs/sitreps/`.
"""
import hashlib
import json
import re
from pathlib import Path

from .agent import read_text, redact, utc_now, write_whole
from .config import MONTHS, SITREP_PROMPT_FILE
from .overlays import Overlay

SITREP_PROMPT = read_text(SITREP_PROMPT_FILE)
DEFAULTS = {"model": "gpt-6-luna", "reasoning_effort": "low", "max_output_tokens": 4000}
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19, "twenty": 20, "half": None, "dozen": 12}
NUMBER = re.compile(r"(?<![\w.\-/])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w\-/])")
WORD = re.compile(r"\b(" + "|".join(w for w, v in WORDS.items() if v is not None) + r")\b", re.I)


def facts_numbers(facts):
    """Every number among the facts, as the check accepts it (each float also rounded to 0 and 1 decimal)."""
    found = set()

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.add(float(value))
            found.add(float(round(value)))
            found.add(round(float(value), 1))
        elif isinstance(value, str):
            for m in NUMBER.finditer(value):
                found.add(float(m.group(1).replace(",", "")))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)
    walk(facts)
    return found


def numbers_in(sentence):
    out = [float(m.group(1).replace(",", "")) for m in NUMBER.finditer(sentence)]
    out += [float(WORDS[m.group(1).lower()]) for m in WORD.finditer(sentence)]
    return out


def figure_check(text, facts):
    """Keep the sentences whose every number is among the facts. Returns (kept text, removed sentences, numbers not
    in the facts)."""
    allowed = facts_numbers(facts)
    kept, removed, unmatched = [], [], []
    for paragraph in (text or "").split("\n"):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s]
        keep = []
        for sentence in sentences:
            bad = [n for n in numbers_in(sentence) if n not in allowed]
            if bad:
                removed.append(sentence)
                unmatched += bad
            else:
                keep.append(sentence)
        kept.append(" ".join(keep))
    return "\n".join(k for k in kept if k).strip(), removed, unmatched


def listed(codes, noun):
    """A list of codes with its count, so that a draft may give either and the check finds both among the facts."""
    codes = sorted(codes)
    return {"count": len(codes), noun: codes}


def week_facts(view, week, run=None):
    """The week's facts, from the twin's view (and the season's own records of the week, where a run gives them)."""
    cal = view.calendar
    teams = view.team_table(week)
    log = view.log_table()
    this_week = log[log["week"] == week] if len(log) else log
    done = {}
    for t in view.tasks.values():
        if t["finished_week"] == week:
            done[t["kind"]] = done.get(t["kind"], 0) + 1
    status = view.status
    zones = view.zone_state
    facts = {
        "week": week, "month": MONTHS[cal.month(week) - 1],
        "teams active": {k: int(((teams["kind"] == k) & (teams["status"] == "active")).sum())
                         for k in ("survey", "clearance", "ts") if (teams["kind"] == k).any()},
        "teams stood down": listed(teams.index[teams["status"] != "active"], "teams"),
        "teams with no report for more than a week": listed(teams.index[teams["stale"]], "teams"),
        "tasks finished this week": {"zone surveys": done.get("ZONE_SURVEY", 0), "visits": done.get("VISIT", 0),
                                     "technical surveys": done.get("TS", 0), "clearances": done.get("CLEARANCE", 0)},
        "zones surveyed so far": int((zones["surveyed_week"] > 0).sum()),
        "zones declared free so far": int((zones["free_week"] > 0).sum()),
        "zones declared free this week": listed(zones.index[zones["free_week"] == week], "zones"),
        "areas released so far": int(status.isin(["cancelled", "reduced", "cleared"]).sum()),
        "areas cleared so far": int((status == "cleared").sum()),
        "areas where mines were found so far": int(view.mines_found.sum()),
        "areas suspected or confirmed, still open": int(status.isin(["suspected", "confirmed"]).sum()),
        "zones closed by the access feed": listed(view.access.index[view.access["status"] != "open"], "zones"),
        "plan": (view.plan or {}).get("plan_id", "none"),
        "new plan approved this week": bool(len(this_week) and ((this_week["kind"] == "plan")
                                                                & (this_week["entry"] == "decision")).any()),
        "dossiers held for missing items": listed(this_week.loc[(this_week["kind"] == "dossier"), "subject"]
                                                  if len(this_week) else [], "areas"),
    }
    if run is not None:
        facts["team-days worked this week"] = run.week_days.get(week, {}).get("worked", 0)
        facts["team-days lost to rain or wind this week"] = run.week_days.get(week, {}).get("weather", 0)
        facts["reports held by the report checker this week"] = run.checker_counts.get(week, {}).get("hold", 0)
        facts["reports queried by the report checker this week"] = run.checker_counts.get(week, {}).get("query", 0)
        forecast = run.latest_forecast
        if forecast is not None and forecast.ok:
            t = forecast.answer["totals"]
            facts["forecast"] = {
                "made in week": forecast.week,
                "mine areas cleared this season, P10 to P90": [t["mine areas cleared"]["P10"],
                                                                t["mine areas cleared"]["P90"]],
                "zones declared free this season, P10 to P90": [t["zones declared free"]["P10"],
                                                                 t["zones declared free"]["P90"]]}
        elif forecast is not None:
            facts["forecast"] = "insufficient evidence"
    return facts


def coded_draft(facts):
    """The report written by code from the facts: what the notebook shows without a key."""
    f = facts
    done = f["tasks finished this week"]
    parts = [f"Week {f['week']} ({f['month']}): {done['zone surveys']} zone survey(s), {done['visits']} visit(s), "
             f"{done['technical surveys']} technical survey(s) and {done['clearances']} clearance(s) finished."]
    if "team-days worked this week" in f:
        parts.append(f"Teams worked {f['team-days worked this week']} team-days and lost "
                     f"{f['team-days lost to rain or wind this week']} to rain or wind.")
    parts.append(f"So far {f['zones surveyed so far']} zone(s) surveyed, {f['zones declared free so far']} declared "
                 f"free, {f['areas released so far']} area(s) released and mines found in "
                 f"{f['areas where mines were found so far']}.")
    attention = []
    stale = f["teams with no report for more than a week"]
    if stale["count"]:
        attention.append("No report for more than a week from " + ", ".join(stale["teams"]) + ".")
    if f["teams stood down"]["count"]:
        attention.append("Stood down: " + ", ".join(f["teams stood down"]["teams"]) + ".")
    if f["zones closed by the access feed"]["count"]:
        attention.append(f"Closed by the access feed: {f['zones closed by the access feed']['count']} zone(s).")
    if f.get("reports held by the report checker this week"):
        attention.append(f"Reports held by the checker: {f['reports held by the report checker this week']}.")
    if f["dossiers held for missing items"]["count"]:
        attention.append("Dossiers held for missing items: " + ", ".join(f["dossiers held for missing items"]["areas"])
                         + ".")
    if f["new plan approved this week"]:
        attention.append(f"A new plan was approved: {f['plan']}.")
    text = " ".join(parts) + "\n" + (" ".join(attention) if attention else "Nothing needs attention this week.")
    forecast = f.get("forecast")
    if isinstance(forecast, dict):
        low, high = forecast["mine areas cleared this season, P10 to P90"]
        zl, zh = forecast["zones declared free this season, P10 to P90"]
        text += (f"\nThe forecast of week {forecast['made in week']} gives {low:g} to {high:g} mine areas cleared this "
                 f"season and {zl:g} to {zh:g} zones declared free (P10 to P90).")
    elif forecast is not None:
        text += "\nForecast: insufficient evidence."
    return text


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SituationReport(Overlay):
    name = "weekly situation report"
    domain = "generative AI (an LLM drafts, and a coded check verifies every figure)"
    insight = "P5: generated output is checked against its source before it is used"
    reads = ("teams", "access")
    limits = ("the figure check verifies numbers, not the wording around them; a person reads the report",)

    def __init__(self, runs_dir=None, client=None, settings=None, limit=None):
        self.runs_dir = Path(runs_dir) if runs_dir else None
        self.client = client
        self.settings = {**DEFAULTS, **(settings or {})}
        self.limit = limit          # the smoke test: this many model drafts, then the coded draft
        self.ran = self.loaded = self.coded = 0

    def inputs_sha(self, facts):
        return sha256_text(json.dumps({"facts": facts, "prompt": SITREP_PROMPT,
                                       "settings": {k: self.settings[k] for k in sorted(DEFAULTS)}},
                                      sort_keys=True, ensure_ascii=False, default=str))

    def draft(self, facts):
        """The draft and where it came from: a saved draft with the same inputs, the model, or the coded draft."""
        sha = self.inputs_sha(facts)
        path = self.runs_dir / f"w{facts['week']:02d}_sitrep.json" if self.runs_dir else None
        if path is not None and path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("inputs_sha") == sha and not record.get("error"):
                self.loaded += 1
                return record["draft"], f"saved draft ({record['model']})", record
        if self.client is None or (self.limit is not None and self.ran >= self.limit):
            self.coded += 1
            why = "no model" if self.client is None else f"the smoke test's {self.limit} model drafts are done"
            return coded_draft(facts), f"coded draft ({why})", None
        record = {"week": facts["week"], "model": self.settings["model"], "inputs_sha": sha, "time": utc_now(),
                  "facts": facts, "draft": "", "error": "", "usage": None}
        try:
            request = {"model": self.settings["model"], "instructions": SITREP_PROMPT, "store": False,
                       "input": json.dumps(facts, ensure_ascii=False, default=str),
                       "max_output_tokens": self.settings["max_output_tokens"]}
            if self.settings.get("reasoning_effort"):
                request["reasoning"] = {"effort": self.settings["reasoning_effort"]}
            response = self.client.responses.create(**request)
            record["draft"] = getattr(response, "output_text", "") or ""
            usage = getattr(response, "usage", None)
            record["usage"] = usage if isinstance(usage, dict) or usage is None else usage.model_dump(mode="json")
            self.ran += 1
        except Exception as error:   # a failed call leaves the coded draft in its place, and is not reused
            record["error"] = redact(f"{type(error).__name__}: {error}")
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_whole(path, json.dumps(record, ensure_ascii=False, indent=1, default=str))
        if record["error"]:
            self.coded += 1
            return coded_draft(facts), f"coded draft (the model call failed: {record['error']})", record
        return record["draft"], f"model draft ({self.settings['model']})", record

    def recommend(self, view, week, run=None, facts=None, **inputs):
        reason = self.check_inputs(view, week)
        if reason:
            return self.insufficient(week, reason)
        facts = facts or week_facts(view, week, run)
        text, origin, _ = self.draft(facts)
        kept, removed, unmatched = figure_check(text, facts)
        sentences = len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()])
        answer = {"text": kept, "draft": text, "origin": origin, "removed": removed, "unmatched": unmatched,
                  "facts": facts}
        if sentences and len(removed) > sentences / 2:
            rec = self.insufficient(week, f"{len(removed)} of {sentences} sentences had figures not in the facts",
                                    evidence=("the twin's facts of the week",))
            rec.answer = answer
            return rec
        return self.answer(week, answer, ("the twin's facts of the week",),
                           uncertainty=f"{len(removed)} sentence(s) removed by the figure check")
