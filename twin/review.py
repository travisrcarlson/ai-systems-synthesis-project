"""The dossier reviewers: who recommends the outcome for an area after a visit, before a person signs it off.

- `AgentReviewer` is the decision tool: the LLM agent of `agent.py` (agentic AI, P6's insight), which reads the
  dossier through the twin's read-only view, records an evidence register, tests its decision against the coded
  rule check, and recommends. With a key it runs the agent and saves each review in `runs/`; without one it reads
  the saved reviews. Its fallback, further survey when no recommendation passes the rule check, is its
  "insufficient evidence".
- `ReferenceReviewer` applies the coded rules alone to the true class of every item (no LLM). It reads the
  field's key, so it exists only in the simulation: it is what a reviewer who read every statement exactly as the
  field meant it would recommend, the yardstick for the agent and the reviewer of the season replay's 20 seeds.
"""
from dataclasses import dataclass, field as dataclass_field

from . import agent, rules
from .overlays import Overlay


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


class ReferenceReviewer(Overlay):
    """The coded rules alone, on the true class of every item, read from the simulated field's key."""
    name = "reference reviewer"
    domain = "coded rules (simulation only)"
    insight = "P6"
    by = "reference reviewer (coded rules on the true classes)"

    def __init__(self, field):
        self.field = field

    def review(self, view, area_id, week):
        items = list(view.evidence[area_id])
        outcome, register, result = rules.reference_decision(items, self.field.key_for(area_id, items),
                                                             view.criteria)
        return Review(outcome, result, register, self.by)


class MissingReview(RuntimeError):
    """No saved review matches, and there is no client to run the agent."""


class AgentRunError(RuntimeError):
    """A review stopped on an API error. It is not saved, so a rerun with a key resumes from that review."""


class AgentReviewer(Overlay):
    """The dossier reviewer agent.

    With a client (a key), it reads a saved review only if its checksum is the same as now and the review did not
    end on an API error; otherwise it runs the agent and saves the review. An API error that concerns one review (its
    request refused as too long or invalid) ends that review with the fallback, which is saved so that the run goes
    on, and is run again on the next run with a key; three such errors in a row stop the run. Any other API error
    (the connection, a time-out, the key, the account, the rate limit, the server) stops the run at once, and that
    review is not saved, so a rerun resumes from it.

    Without a client, it reads the saved review of each week, area and visit, and counts the saved reviews whose
    dossier differs from the one rebuilt here, and those made with a different prompt, rules, code or settings.
    """
    name = "dossier reviewer"
    domain = "agentic AI (LLM agent with tools)"
    insight = "P6"
    reads = ("hazard", "land_use")
    by = "dossier reviewer agent"
    STOP_AFTER = 3

    def __init__(self, runs_dir, client=None, settings=None, label="season", limit=None, after_limit=None):
        self.runs_dir, self.client = runs_dir, client
        self.settings = {**agent.DEFAULTS, **(settings or {})}
        self.label = label
        self.limit, self.after_limit = limit, after_limit     # the smoke test: this many live reviews, then another
        self.ran = self.loaded = self.review_errors = self.handed_on = 0
        self.other_dossier = self.other_method = 0
        self._errors_in_a_row = 0

    def review(self, view, area_id, week):
        session = agent.ReviewSession(view, area_id, week)
        live = self.client is not None
        record = agent.load_review(self.runs_dir, session, self.settings, require_same_inputs=live)
        if record is not None and live and record["stats"].get("error"):
            record = None     # a review that ended on an API error is run again when there is a key
        if record is None and self.limit is not None and self.ran >= self.limit and self.after_limit is not None:
            self.handed_on += 1
            return self.after_limit.review(view, area_id, week)
        if record is None:
            if not live:
                raise MissingReview(f"there is no saved review {agent.review_key(week, area_id, session.visit)}, and "
                                    "the agent's season needs a key to run it.")
            trace, usage, stats = agent.run_review(session, self.client, self.settings)
            if stats["error"] and stats["error_scope"] != "review":
                raise AgentRunError(f"week {week}, {area_id}: {stats['error']}")
            record = agent.save_review(self.runs_dir, session, trace, usage, stats, self.settings, self.label)
            self.ran += 1
            if stats["error"]:
                self.review_errors += 1
                self._errors_in_a_row += 1
                if self._errors_in_a_row >= self.STOP_AFTER:
                    raise AgentRunError(f"{self.STOP_AFTER} reviews in a row ended on an API error, the last in week "
                                        f"{week}, {area_id}: {stats['error']}. They are saved as fallbacks and will be "
                                        "run again with a key.")
            else:
                self._errors_in_a_row = 0
        else:
            self.loaded += 1
            hashes = agent.input_hashes(session, self.settings)
            self.other_dossier += record.get("dossier_sha") != hashes["dossier"]
            self.other_method += record.get("method_sha") != hashes["method"]
        register = {k: rules.RegisterEntry(**v) for k, v in record["register"].items()}
        result = rules.rule_check(record["outcome"], register, session.items, view.criteria)
        return Review(record["outcome"], result, register, f"{self.by} ({self.settings['model']})", record["summary"],
                      record["confidence"], record["flags"], record["fallback"], record)
