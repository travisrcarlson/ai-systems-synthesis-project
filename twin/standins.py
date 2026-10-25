"""Stand-ins for the OpenAI client, so the agent's season and the situation reports can be run end to end without a
key or a network call. None of them is a model; each is a test of the code around one.

- `OracleModel` plays the reference reviewer through the agent's own tools: it reads every section, records each
  item with its true class (from the simulated field's key) and a quote from its statement, and submits the coded
  rules' outcome. A season reviewed by it must match the reference season exactly, which tests the tools, the
  register, the rule check, the saved-review records and the sign-off path.
- `ScriptedWriter` returns a fixed text as a situation report's draft, to test the figure check.
"""
import json
import re
from types import SimpleNamespace

from . import agent, rules


class OracleModel:
    def __init__(self, twin, field, fail=None):
        self.twin, self.field = twin, field
        self.responses = self       # so that client.responses.create(...) works as with the real client
        self.calls = 0
        self.plans = {}
        self.fail = fail or {}      # call number -> the exception to raise, to test error handling

    def create(self, **request):
        self.calls += 1
        if self.calls in self.fail:
            raise self.fail[self.calls]
        opening = request["input"][0]["content"]
        area = re.search(r"Review area (\S+) ", opening).group(1)
        done = sum(1 for i in request["input"] if isinstance(i, dict) and i.get("type") == "function_call_output")
        if done == 0:
            items = list(self.twin.evidence[area])
            outcome, register, _ = rules.reference_decision(items, self.field.key_for(area, items), self.twin.criteria)
            quote = {i.item_id: i.statement[:50] for i in items}
            self.plans[area] = [
                [("read_dossier", {"section": s}) for s in agent.SECTIONS],
                [("record_evidence", {"item_id": e.item_id, "quote": quote[e.item_id], "evidence_class":
                                      e.evidence_class, "reliability_class": e.reliability_class,
                                      "explains": e.explains, "note": "Oracle stand-in: the item's true class."})
                 for e in register.values()],
                [("check_decision", {"outcome": outcome})],
                [("submit", {"outcome": outcome, "summary": "Oracle stand-in review (a test, not a model).",
                             "confidence": "medium", "flags": []})]]
        turn = self.plans[area].pop(0) if self.plans[area] else []
        return SimpleNamespace(output=[{"type": "function_call", "call_id": f"oracle_{self.calls}_{n}", "name": name,
                                        "arguments": json.dumps(args)} for n, (name, args) in enumerate(turn)],
                               usage=None)


class ScriptedWriter:
    def __init__(self, text):
        self.text = text
        self.responses = self

    def create(self, **request):
        return SimpleNamespace(output_text=self.text, usage=None)
