"""The replay: the closed loop of the pilot, round after round, with known outcomes to measure it against.

Each round: the strategic service's shares set how many areas each municipality's teams visit; the operational
service's ranking picks the areas; the simulated field returns evidence; the photo screener reads any photo; a
reviewer recommends an outcome, the rule check tests it, and a person signs it off under the replay's sign-off
policy (accept the recommendation; one the rule check vetoes is not signed off, and the area goes back for more
survey); the twin, which runs the rule check again itself, is updated. At the end of the round, technical survey
and clearance teams work through the tasks the decisions created, and their results are signed off too. RELand's
labels decide what the field finds, and afterwards how well the loop did.
"""
from dataclasses import dataclass, field as dataclass_field

import pandas as pd

from . import agent, rules
from .config import NAME_OF, PILOT
from .core import RELEASED, RESULT_OUTCOMES, REVIEW_OUTCOMES
from .strategic import capacity_shares, round_allocation

SIGN_OFF = "QC reviewer (replay sign-off policy)"


@dataclass(frozen=True)
class Policy:
    """How the replay tasks its survey teams."""
    ranking: str = "gated"        # "gated" (the baseline-first gate), "baseline", "lr", "hgb" or "random"
    allocation: str = "victims"   # how the capacity is shared: "victims", "suspected" or "areas"
    follow_up: bool = True        # neighbours of an area where mines were found are surveyed next

    def label(self):
        """The policy in words, by how it differs from the main policy."""
        parts = []
        if self.ranking != "gated":
            parts.append({"baseline": "ranking by nearness to past accidents", "lr": "ranking by logistic regression",
                          "hgb": "ranking by gradient-boosted trees", "random": "random order"}[self.ranking])
        if self.allocation != "victims":
            parts.append({"suspected": "shares by suspected areas", "areas": "shares by number of areas"}[self.allocation])
        if not self.follow_up:
            parts.append("no follow-up of areas with mines")
        return "; ".join(parts) if parts else "main policy"


@dataclass
class Review:
    """A reviewer's recommendation for one area."""
    outcome: str
    rule_check: dict
    register: dict
    by: str
    summary: str = ""
    confidence: str = ""
    flags: list = dataclass_field(default_factory=list)
    fallback: bool = False
    record: dict | None = None


class ReferenceReviewer:
    """The coded rules alone, applied to the true class of every item (no LLM): what a reviewer who read every
    statement exactly as the field meant it would recommend."""
    by = "reference reviewer (coded rules on the true classes)"

    def __init__(self, field):
        self.field = field

    def review(self, twin, area_id, round_number):
        items = twin.evidence[area_id]
        outcome, register, result = rules.reference_decision(items, self.field.key_for(area_id, items), twin.criteria)
        return Review(outcome, result, register, self.by)


class MissingReview(RuntimeError):
    """No saved review matches, and there is no client to run the agent."""


class AgentRunError(RuntimeError):
    """A review stopped on an API error. It is not saved, so a rerun with a key resumes from that review."""


class AgentReviewer:
    """The area reviewer agent.

    With a client (a key), it reads a saved review only if its checksum is the same as now and the review did not
    end on an API error; otherwise it runs the agent and saves the review. An API error that concerns one review
    (its request refused as too long or invalid) ends that review with the fallback, which is saved so that the run
    goes on, and is run again on the next run with a key; three such errors in a row stop the run. Any other API
    error (the connection, a time-out, the key, the account, the rate limit, the server) stops the run at once, and
    that review is not saved, so a rerun resumes from it.

    Without a client, it reads the saved review of each round, area and visit, and counts the saved reviews whose
    dossier differs from the one rebuilt here, and those made with a different prompt, rules, code or settings.
    """
    by = "area reviewer agent"
    STOP_AFTER = 3

    def __init__(self, runs_dir, client=None, settings=None, label="replay"):
        self.runs_dir, self.client = runs_dir, client
        self.settings = {**agent.DEFAULTS, **(settings or {})}
        self.label = label
        self.ran = self.loaded = self.review_errors = 0
        self.other_dossier = self.other_method = 0
        self._errors_in_a_row = 0

    def review(self, twin, area_id, round_number):
        session = agent.ReviewSession(twin, area_id, round_number)
        live = self.client is not None
        record = agent.load_review(self.runs_dir, session, self.settings, require_same_inputs=live)
        if record is not None and live and record["stats"].get("error"):
            record = None     # a review that ended on an API error is run again when there is a key
        if record is None:
            if not live:
                raise MissingReview(f"there is no saved review {agent.review_key(round_number, area_id, session.visit)}"
                                    ", and the agent's replay needs a key to run it.")
            trace, usage, stats = agent.run_review(session, self.client, self.settings)
            if stats["error"] and stats["error_scope"] != "review":
                raise AgentRunError(f"round {round_number}, {area_id}: {stats['error']}")
            record = agent.save_review(self.runs_dir, session, trace, usage, stats, self.settings, self.label)
            self.ran += 1
            if stats["error"]:
                self.review_errors += 1
                self._errors_in_a_row += 1
                if self._errors_in_a_row >= self.STOP_AFTER:
                    raise AgentRunError(f"{self.STOP_AFTER} reviews in a row ended on an API error, the last at round "
                                        f"{round_number}, {area_id}: {stats['error']}. They are saved as fallbacks and "
                                        "will be run again with a key.")
            else:
                self._errors_in_a_row = 0
        else:
            self.loaded += 1
            hashes = agent.input_hashes(session, self.settings)
            self.other_dossier += record.get("dossier_sha") != hashes["dossier"]
            self.other_method += record.get("method_sha") != hashes["method"]
        register = {k: rules.RegisterEntry(**v) for k, v in record["register"].items()}
        result = rules.rule_check(record["outcome"], register, session.items, twin.criteria)
        return Review(record["outcome"], result, register, f"{self.by} ({self.settings['model']})", record["summary"],
                      record["confidence"], record["flags"], record["fallback"], record)


class Replay:
    """One run of the closed loop under a policy, assumptions and a reviewer."""

    def __init__(self, twin, field, service, reviewer, policy=None, assumptions=None, photo_readings=None,
                 verbose=False):
        self.twin, self.field, self.service, self.reviewer = twin, field, service, reviewer
        self.policy = policy or Policy()
        self.a = assumptions or field.a
        self.photo_readings = photo_readings or {}
        self.verbose = verbose

    def tasking(self, round_number, plan):
        """The areas each municipality's teams visit this round, in the order they are visited and reviewed."""
        twin, chosen = self.twin, []
        rank = twin.priority["rank"]
        open_area = ~twin.status.isin(RELEASED)
        # areas where mines were confirmed: by direct evidence or technical survey, or found by clearance
        confirmed = set(twin.status.index[twin.status.eq("confirmed") | twin.mines_found])
        for code in PILOT:
            k = int(plan.loc[code, round_number])
            if k == 0:
                continue
            here = (twin.areas["municipality_code"] == code) & open_area
            revisit = twin.areas.index[here & (twin.next_task == "NTS")]
            fresh = here & (twin.visits == 0) & twin.next_task.isna()
            follow = []
            if self.policy.follow_up:
                near = {b for a in confirmed for b in twin.neighbours[a]}
                follow = [a for a in twin.areas.index[fresh] if a in near]
            first = sorted(revisit, key=rank.get) + sorted(follow, key=rank.get)
            rest = [a for a in rank[fresh].sort_values().index if a not in set(first)]
            chosen += (first + rest)[:k]
        return chosen

    def run(self):
        twin, field = self.twin, self.field
        twin.reset()
        field.reset()
        twin.set_priority(self.service.priority(self.policy.ranking))
        plan = round_allocation(capacity_shares(twin, self.policy.allocation), self.a.visits_per_round, self.a.rounds)
        rows, round_rows = [], []
        for r in range(1, self.a.rounds + 1):
            areas = self.tasking(r, plan)
            for a in areas:
                visit = int(twin.visits[a]) + 1
                items = field.nts(a, visit, r)
                readings = {i.item_id: self.photo_readings.get(field.photos[(a, i.item_id)], "no reading")
                            for i in items if i.photo}
                twin.record_visit(a, items, r, f"NTS team, round {r}", readings)
                review = self.reviewer.review(twin, a, r)
                reference, _, _ = rules.reference_decision(twin.evidence[a], field.key_for(a, twin.evidence[a]),
                                                           twin.criteria)
                decision, reason = review.outcome, ""
                vetoed = review.rule_check["result"] != "pass"
                if vetoed:
                    decision = "FURTHER_NTS"
                    reason = ("The rule check vetoes the recommendation on its own register, so it is not signed "
                              "off; the area goes back for more survey.")
                twin.sign_off(a, decision, round_number=r, decided_by=SIGN_OFF, recommended_by=review.by,
                              recommendation=review.outcome, register=review.register, reason=reason)
                rows.append({"round": r, "area_id": a, "municipality": NAME_OF[twin.areas.loc[a, "municipality_code"]],
                             "visit": visit, "rank": int(twin.priority.loc[a, "rank"]),
                             "mined": bool(field.mined[a]), "branch": field.branches[(a, visit)],
                             "outcome": review.outcome, "decision": decision, "reference": reference,
                             "agrees": review.outcome == reference,
                             "fallback": review.fallback, "confidence": review.confidence,
                             "vetoes": len(review.record["vetoes"]) if review.record else 0,
                             "vetoed at sign-off": vetoed,
                             "photos": len(readings), "event_inside": bool(twin.areas.loc[a, "event_inside"])})
            # the follow-up teams: technical survey, then clearance of what it confirms
            ts_done = cleared = 0
            for a in twin.next_task.index[twin.next_task == "TS"]:
                found = field.technical_survey(a)
                twin.sign_off(a, "TS_FOUND" if found else "TS_NOTHING", round_number=r, decided_by=SIGN_OFF,
                              recommended_by="technical survey team", reason="")
                ts_done += 1
            for a in twin.next_task.index[twin.next_task == "CLEARANCE"]:
                found = field.clearance(a)
                twin.sign_off(a, "CLEARED", round_number=r, decided_by=SIGN_OFF, recommended_by="clearance team",
                              found=found, reason=f"{found} mine cell(s) found and cleared" if found
                              else "clearance found no mines")
                cleared += 1
            round_rows.append({"round": r, "visits": len(areas), "technical surveys": ts_done, "clearances": cleared,
                               **twin.status.value_counts().reindex(["suspected", "confirmed", "cancelled", "reduced",
                                                                     "cleared"], fill_value=0).to_dict()})
            if self.verbose:
                done = (f"; reviews run {self.reviewer.ran}, read {self.reviewer.loaded}"
                        if hasattr(self.reviewer, "ran") else "")
                print(f"round {r}: {len(areas)} visits, {ts_done} technical surveys, {cleared} clearances{done}",
                      flush=True)
        return ReplayResult(self, pd.DataFrame(rows), pd.DataFrame(round_rows).set_index("round"), plan)


class ReplayResult:
    """What a replay did, and how well, measured against RELand's labels."""

    def __init__(self, replay, visits, rounds, plan):
        twin, field = replay.twin, replay.field
        self.policy, self.visits, self.rounds, self.plan = replay.policy, visits, rounds, plan
        self.status = twin.status.copy()
        self.audit = twin.audit_log()
        self.tasks = pd.DataFrame(twin.tasks)
        self.mined = field.mined.copy()
        self.km2 = twin.areas["km2"]
        self.metrics = self.measure()

    def measure(self):
        s, mined, km2 = self.status, self.mined, self.km2
        decided = self.audit[self.audit["action"] == "decision"]
        reviewed = decided[decided["outcome"].isin(list(REVIEW_OUTCOMES))]
        released = s.isin(RELEASED)
        false_release = released & mined & s.isin(["cancelled", "reduced"])
        ts = self.tasks[self.tasks["task"] == "TS"]["area_id"] if len(self.tasks) else pd.Series(dtype=object)
        clear = self.tasks[self.tasks["task"] == "CLEARANCE"]["area_id"] if len(self.tasks) else pd.Series(dtype=object)
        visited = s.index[self.visits.groupby("area_id").size().reindex(s.index).fillna(0) > 0]
        return {
            "visits": len(self.visits), "areas visited": len(visited),
            "mine areas visited": int(mined[visited].sum()),
            "technical surveys": len(ts), "technical surveys on clean land": int((~mined[ts]).sum()),
            "clearances": len(clear), "clearances on clean land": int((~mined[clear]).sum()),
            "mine areas found and cleared": int((s.eq("cleared") & mined).sum()),
            "areas released": int(released.sum()), "km² released": float(km2[released].sum()),
            "cancelled": int(s.eq("cancelled").sum()), "reduced": int(s.eq("reduced").sum()),
            "cleared": int(s.eq("cleared").sum()),
            "released with mines (critical)": int(false_release.sum()),
            "cancelled with mines": int((s.eq("cancelled") & mined).sum()),
            "reduced with mines": int((s.eq("reduced") & mined).sum()),
            "visited, still open": int((~released[visited]).sum()),
            "review sign-offs": len(reviewed),
            "result sign-offs": int(decided["outcome"].isin(list(RESULT_OUTCOMES)).sum()),
            "overrides": int((reviewed["recommendation"].notna()
                              & (reviewed["recommendation"] != reviewed["outcome"])).sum()),
            "vetoes during reviews": int(self.visits["vetoes"].sum()) if len(self.visits) else 0,
            "vetoed at sign-off": int(self.visits["vetoed at sign-off"].sum()) if len(self.visits) else 0,
            "reviews agreeing with the reference": int(self.visits["agrees"].sum()) if len(self.visits) else 0,
            "fallbacks": int(self.visits["fallback"].sum()) if len(self.visits) else 0,
        }


def summarise(results):
    """One row per replay result: its policy and its measures."""
    return pd.DataFrame([{"policy": r.policy.label(), **r.metrics} for r in results])
