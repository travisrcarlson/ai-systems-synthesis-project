"""The overlay contract: how every decision tool reads the twin and answers.

An overlay reads the twin through its read-only view (`Twin.view()`) and returns a `Recommendation`:
- an answer, with the evidence it relied on (the layers and records it read), its uncertainty and its limits; or
- "insufficient evidence", with the reason, instead of a weak answer (P4's insight: a model should say "unknown"
  when it is unsure).
No overlay can write to the twin: the view has no method that writes, and assigning to it fails. What the twin then
does is a decision a person approves, recorded in its log with the overlays the decision relied on. A new overlay
plugs in by following this contract, without changing the core.
"""
from dataclasses import dataclass

INSUFFICIENT = "insufficient evidence"


@dataclass
class Recommendation:
    overlay: str
    week: int
    status: str                 # "ok", or "insufficient evidence"
    answer: object = None
    evidence: tuple = ()        # what the answer relied on: layers, records, other overlays' answers
    uncertainty: object = None
    limits: tuple = ()
    reason: str = ""            # why there is insufficient evidence, when there is

    @property
    def ok(self):
        return self.status == "ok"

    def describe(self):
        if not self.ok:
            return f"{self.overlay}, week {self.week}: {INSUFFICIENT} ({self.reason})"
        return f"{self.overlay}, week {self.week}: relied on {', '.join(self.evidence) or 'the twin'}"


class Overlay:
    """The base of every decision tool. A subclass sets its name, AI domain, the earlier project whose insight it
    carries, the layers it reads and its standing limits, and implements `recommend`."""
    name = "overlay"
    domain = ""
    insight = ""
    reads = ()
    limits = ()

    def recommend(self, view, week, **inputs):
        raise NotImplementedError

    # ---- the contract's checks and answers ------------------------------------------------------------------------
    def check_inputs(self, view, week):
        """The reason the overlay cannot answer from the twin's layers now, or "" if it can: a layer it reads is not
        registered, or a live feed it reads has not been updated for longer than the layer allows."""
        missing = [n for n in self.reads if n not in view.layers]
        if missing:
            return "the twin has no " + " or ".join(missing) + " layer"
        for name in self.reads:
            layer = view.layers[name]
            age = layer_age(view, name, week)
            if layer.stale_after_weeks is not None and age is not None and age > layer.stale_after_weeks:
                return f"the {name} layer was last updated {age} weeks ago (stale after {layer.stale_after_weeks})"
        return ""

    def insufficient(self, week, reason, evidence=()):
        return Recommendation(self.name, week, INSUFFICIENT, evidence=tuple(evidence), limits=tuple(self.limits),
                              reason=reason)

    def answer(self, week, answer, evidence=(), uncertainty=None, limits=()):
        return Recommendation(self.name, week, "ok", answer, tuple(evidence), uncertainty,
                              tuple(self.limits) + tuple(limits))


def layer_age(view, name, week):
    """Weeks since a live layer was last updated (None for a layer that is not a feed). Before the season's first
    feed, a layer counts as fresh."""
    if name == "access":
        last = int(view.access["as_of"].max())
    elif name == "teams":
        last = int(view.team_state["as_of"].max())
    else:
        return None
    return 0 if last == 0 and week <= 1 else week - last


def overlay_table(overlays):
    """The overlays as a table, for the notebook."""
    import pandas as pd
    return pd.DataFrame([{"overlay": o.name, "AI domain": o.domain, "insight from": o.insight,
                          "reads": ", ".join(o.reads)} for o in overlays])
