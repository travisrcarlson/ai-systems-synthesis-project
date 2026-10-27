# Model card: the dossier reviewer agent

Sections follow Mitchell et al. (2019), *Model cards for model reporting*, shortened, with a section on replacing the
model.

**Code:** `twin/agent.py`, `twin/review.py` and `twin/rules.py`; the prompt `prompts/system_prompt.md`; the rules
`rules/land_release_rules.md`; the saved reviews `runs/reviews/`. Notebook sections 8 and 9.2.

## Model details
- An LLM agent: OpenAI's gpt-6-luna through the Responses API (reasoning effort medium; at most 40 steps; no state
  stored with OpenAI between calls), in the role of an area reviewer in the national authority's quality-control
  team, who recommends and never decides.
- Five tools: `read_dossier`; `record_evidence` (an exact quote, the evidence class, the reliability class of the
  source from 1 to 7 as in IMAS 08.10, and the items an item of absence explains); `lookup_rules`; `check_decision`
  (the coded rule check, R1 to R6); and `submit`, accepted only if the recommendation passes the rule check.
- Safeguards: the quote check; a scan for text in the dossier that addresses the reviewer; a filter that removes
  advice on handling items; the step limit and the fallback to further survey; API errors handled by scope (one review,
  or the run); and the twin's own checks at sign-off.

## Intended use
For each area's dossier after a visit, a recommendation of cancellation, clearance or further survey, with an evidence
register and a short summary, for a quality-control reviewer to sign off. Not for deciding; not for dossiers under
other criteria than the pilot's; never for advice on handling items.

## Factors
How each source is classed (first-hand or second-hand, reliability 1 to 7) and what kind of evidence each item is
(direct, indirect, absence, not evidence); the national criteria, which are Travis's settings.

## Metrics
Agreement with the coded rules applied to the true classes, on the same dossier; fallbacks, vetoes and quote failures;
cost and time; an expert's judgement of a sample.

## Evaluation data
The 96 dossiers of seed 0's season under the main policy, from the simulated field (section 9.2), and Travis's expert
review of 14 of them (`runs/expert_review.csv`).

## Quantitative analyses (the full run)
- 94 of 96 recommendations the same as the coded rules' (98%); no fallback, veto, quote failure or tool error; 2
  sentences removed by the handling-advice filter, both false alarms (register notes that say "second-hand" or
  "first-hand", which it read as the verb "hand"); 518 model calls, 33.2 minutes, USD 0.246 (cached input counted at
  the full price, so an upper bound).
- The 2 that differ, NAR-0283 and SAB-0225: a teacher's report that people stopped using the land after an armed
  group camped there, classed second-hand (class 5) where its true class is first-hand (class 4), so further survey
  instead of clearance. Travis judged both acceptable calls; the second visit sent each to clearance.
- 12 more, drawn at random, 4 of each outcome: 7 he would sign off, 3 acceptable calls and 2 errors. SLU-0447: a
  hunter's report of a mule killed by an explosion, at a place he saw, recorded as indirect, class 5, where its true
  class is direct, class 4; the outcome, clearance, was right; sent back for the register to be corrected. SAB-0214:
  an object the team saw that looked like explosive ordnance, sent to clearance as the criteria say; in practice,
  Travis answered, it would generally go first to an EOD spot task, an outcome the twin does not have.
- One more slip among the 12, with no effect on the outcome: at ABE-0324's second visit, the agent classed the
  president of the community action board as second-hand (class 5), where it had given her class 4 at the first visit,
  as her true class is.

## Ethical considerations
- Automation bias: a person signing off may follow the agent's errors (Parasuraman & Manzey, 2010). Its reading errors
  pass the rule check when they are consistent with the outcome, so only a person catches them.
- Dual use: the twin holds no technical data on munitions, and the filter removes advice on handling items; its 2
  removals in the full run were false alarms, which cost the reader the agent's note on those items.
- Text in a dossier that instructs the reviewer is flagged and never followed (section 8.3); the full run's dossiers
  had none.

## Caveats and recommendations
- One model, 96 simulated dossiers, one seed. The statements are simulated, in English and from a fixed set of
  patterns; a programme's survey records would be in Spanish and varied.
- In the 2 cases that differ, and in SLU-0447, it classed evidence below its true class. In the 2 cases, that
  delayed the clearance of land with mines by a visit; had the second visit brought reports of safe use, its register
  would have allowed cancellation, which the true classes never do.
- A person signs every review; the sign-off sheet shows the recommendation, then the evidence register and the rule
  check.

## Replacing this model
A new model, or a new prompt or rules, changes the review's checksum, so every dossier is reviewed again; the old
reviews stay in the repository's history to compare with. The new reviews are compared with the coded rules review by
review, and with the old model's on the same dossiers, and a fresh sample goes to an expert. The rule check, the twin's
refusals and the person's sign-off do not change with the model (section 11.5).

## References
Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T.
(2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and
Transparency* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596

Parasuraman, R., & Manzey, D. H. (2010). Complacency and bias in human use of automation: An attentional integration.
*Human Factors, 52*(3), 381–410. https://doi.org/10.1177/0018720810376055
