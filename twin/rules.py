"""The rule check, and the reference reviewer that applies the same rules to the true classes of evidence.

The rule check is code, independent of any model, adapted from P6's rules R1 to R7 to this project's areas and
national criteria. It tests a decision against a register of evidence (each item's class, the reliability class
of its source, and the items it explains) and the twin's own record of each item (who it comes from, and the years
of safe use it reports). It vetoes a decision if any rule fails:
- R1 (cancellation): there is no direct evidence, and every item of indirect evidence is explained by evidence of
  absence from a source of reliability class 1 to 4. Evidence of absence never explains direct evidence away:
  direct evidence leads to technical survey or clearance (IMAS 07.11, §4.4).
- R2 (cancellation): the national criteria are met: enough different reliable first-hand sources (class 1 to 4)
  each report enough years of safe use. Sources are counted from the twin's record, so a person counts once
  however many items they gave.
- R3 (clearance): where technical survey is available, there is direct evidence of class 1 to 4 to define the
  hazard; where the national criteria provide no technical survey (Colombia's practice for its baseline areas),
  there is direct or indirect evidence to target, since clearance takes the place of technical survey.
- R4 (technical survey): technical survey is available, and there is direct or indirect evidence to target.
- R5 (cancellation, technical survey, clearance): every item of the evidence log is in the register.
- R6 (cancellation): where the national criteria need an intrusive check before an area with a past accident
  recorded inside is released, there is no such record (read from the twin's record of the desk study).
Further non-technical survey is always open, whatever the register holds. The rule check tests the register, not
the world: if an item is classed wrongly, it checks the wrong class.
"""
from dataclasses import dataclass, field

OUTCOMES = ["CANCEL", "TECHNICAL_SURVEY", "CLEARANCE", "FURTHER_NTS"]
EVIDENCE_CLASSES = ["direct", "indirect", "absence", "not_evidence"]


@dataclass
class RegisterEntry:
    """One item of evidence as a reviewer recorded it."""
    item_id: str
    quote: str
    evidence_class: str
    reliability_class: int
    explains: list = field(default_factory=list)
    note: str = ""
    step: int = 0


def check(rule, passed, detail):
    return {"rule": rule, "passed": bool(passed), "detail": detail}


def source_of(item):
    """Who an item comes from, as the twin records it: the informant's ID where there is one, else the source."""
    return getattr(item, "source_id", "") or item.source


def recorded_accident(items):
    """Whether the twin's desk study records a past landmine accident inside the area."""
    return any(source_of(item) == "national database" for item in items)


def explained(entry, register, limit=4):
    """Whether evidence of absence from a source of class 1 to `limit` explains this item."""
    return any(a.evidence_class == "absence" and entry.item_id in a.explains and a.reliability_class <= limit
               for a in register.values())


def rule_check(outcome, register, items, criteria):
    """Test one outcome for an area. Returns {"result": "pass" or "veto", "outcome", "checks"}."""
    if outcome not in OUTCOMES:
        return {"result": "veto", "outcome": outcome,
                "checks": [check("-", False, f"{outcome} is not an outcome; the outcomes are {', '.join(OUTCOMES)}.")]}
    unrecorded = [item.item_id for item in items if item.item_id not in register]
    if outcome == "FURTHER_NTS":
        detail = "Further non-technical survey is always open."
        if unrecorded:
            detail += f" Not yet in the register: {', '.join(unrecorded)}."
        return {"result": "pass", "outcome": outcome, "checks": [check("-", True, detail)]}
    checks = [check("R5", not unrecorded, "Every item of the evidence log is in the register." if not unrecorded
                    else f"Not yet in the register: {', '.join(unrecorded)}.")]
    by_id = {item.item_id: item for item in items}
    entries = [register[i.item_id] for i in items if i.item_id in register]
    if outcome == "CANCEL":
        direct = [e.item_id for e in entries if e.evidence_class == "direct"]
        unexplained = [e.item_id for e in entries if e.evidence_class == "indirect" and not explained(e, register)]
        if not direct and not unexplained:
            detail = ("There is no direct evidence, and every item of indirect evidence is explained by evidence of "
                      "absence of class 1 to 4.")
        else:
            parts = []
            if direct:
                parts.append(f"direct evidence, which evidence of absence cannot explain away: {', '.join(direct)}")
            if unexplained:
                parts.append("indirect evidence not explained by evidence of absence of class 1 to 4: "
                             + ", ".join(unexplained))
            detail = "Cancellation is not allowed with " + "; and ".join(parts) + "."
        checks.append(check("R1", not direct and not unexplained, detail))
        sources = {}
        for e in entries:
            item = by_id[e.item_id]
            if e.evidence_class == "absence" and e.reliability_class <= 4 and (item.years or 0) >= criteria.min_years:
                sources.setdefault(source_of(item), []).append(e.item_id)
        listed = "; ".join(" and ".join(ids) + (" (one source)" if len(ids) > 1 else "") for ids in sources.values())
        checks.append(check("R2", len(sources) >= criteria.sources,
                            f"{len(sources)} different reliable source(s) report at least {criteria.min_years} years "
                            f"of safe use" + (f" ({listed})" if sources else "") + f"; the criteria need "
                            f"{criteria.sources}."))
        if criteria.recorded_accident_needs_ts:
            recorded = recorded_accident(items)
            needed = "technical survey" if criteria.technical_survey else "clearance"
            checks.append(check("R6", not recorded, "A past landmine accident is recorded inside the area, so the "
                                f"national criteria need {needed} before it is released." if recorded
                                else "No past landmine accident is recorded inside the area."))
    elif outcome == "CLEARANCE":
        if criteria.technical_survey:
            direct = [e.item_id for e in entries if e.evidence_class == "direct" and e.reliability_class <= 4]
            checks.append(check("R3", direct, f"Direct evidence of class 1 to 4: {', '.join(direct)}." if direct
                                else "No direct evidence of class 1 to 4, so clearance without technical survey "
                                     "is not justified."))
        else:
            target = [e.item_id for e in entries if e.evidence_class in ("direct", "indirect")]
            checks.append(check("R3", target, "The national criteria provide no technical survey, so clearance "
                                f"takes its place; evidence to target: {', '.join(target)}." if target
                                else "There is no direct or indirect evidence to target, so clearance is not "
                                     "justified."))
    else:
        target = [e.item_id for e in entries if e.evidence_class in ("direct", "indirect")]
        if not criteria.technical_survey:
            checks.append(check("R4", False, "The national criteria provide no technical survey in this programme; "
                                "a hazardous area defined by non-technical survey goes to clearance."))
        else:
            checks.append(check("R4", target, f"Evidence to target: {', '.join(target)}." if target
                                else "There is no direct or indirect evidence to target."))
    return {"result": "pass" if all(c["passed"] for c in checks) else "veto", "outcome": outcome, "checks": checks}


def failed_rules(result):
    return ", ".join(c["rule"] for c in result["checks"] if not c["passed"])


# ---- the reference reviewer -------------------------------------------------------------------------------------
def reference_register(items, key, criteria):
    """The register a reviewer would build who read every item exactly as the field wrote it.

    Each source that reports enough years of safe use from class 1 to 4 explains away the indirect evidence of
    class 5 to 7 (the database record, second-hand reports and rumours): safe human activity over time can
    cancel an area recorded on old reports (IMAS 07.11, §4.3). It explains no direct evidence, and no indirect
    evidence that the survey team saw or that a first-hand source reported.
    """
    weak = [i.item_id for i in items if key[i.item_id]["class"] == "indirect" and key[i.item_id]["reliability"] >= 5]
    register = {}
    for item in items:
        k = key[item.item_id]
        strong_absence = (k["class"] == "absence" and k["reliability"] <= 4
                          and (item.years or 0) >= criteria.min_years)
        register[item.item_id] = RegisterEntry(item.item_id, item.statement[:60], k["class"], k["reliability"],
                                               weak if strong_absence else [], f"reference: {k['template']}")
    return register


def reference_decision(items, key, criteria):
    """The coded rules alone, as the rules file sets them out (rules/land_release_rules.md):
    - cancellation, where the rule check allows it;
    - otherwise an intrusive check, where there is a target: direct evidence; indirect evidence that the survey team
      saw or a first-hand source reported (class 1 to 4); a past accident recorded inside the area, where the
      national criteria need an intrusive check before such an area is released; or weaker indirect evidence (class
      5 to 7) still unexplained after the visits the national criteria set for following it up
      (criteria.follow_up_visits). The check is technical survey where the national criteria provide it (it is then
      the default after direct evidence too: the standards do not make moving straight to clearance the default,
      IMAS 07.11, §4.2), and clearance where they do not;
    - otherwise further non-technical survey.
    """
    register = reference_register(items, key, criteria)
    result = rule_check("CANCEL", register, items, criteria)
    if result["result"] == "pass":
        return "CANCEL", register, result
    visits = max((item.visit for item in items), default=0)
    entries = list(register.values())
    target = any(e.evidence_class == "direct" or (e.evidence_class == "indirect" and e.reliability_class <= 4)
                 for e in entries)
    weak_unexplained = any(e.evidence_class == "indirect" and e.reliability_class >= 5 and not explained(e, register)
                           for e in entries)
    recorded = criteria.recorded_accident_needs_ts and recorded_accident(items)
    if target or recorded or (weak_unexplained and visits >= criteria.follow_up_visits):
        outcome = "TECHNICAL_SURVEY" if criteria.technical_survey else "CLEARANCE"
        result = rule_check(outcome, register, items, criteria)
        if result["result"] != "pass":
            raise AssertionError(f"there is a target here, so {outcome} passes the rule check")
        return outcome, register, result
    return "FURTHER_NTS", register, rule_check("FURTHER_NTS", register, items, criteria)
