# Risk register

The risks of using this system in a land release programme, by the four functions of the NIST AI Risk Management
Framework (National Institute of Standards and Technology [NIST], 2023): **Govern**, the roles, policies and
accountability that run through the others; **Map**, the context in which risks arise; **Measure**, testing and
tracking them; and **Manage**, acting on what is mapped and measured. Each risk has the evidence from this
repository's full run, the control in the twin, what remains, and the role that would own it in a programme. The
evidence comes from simulated seasons (P5's lesson): a programme would measure again on its own records. The sections
cited are the notebook's.

## Govern

| ID | Risk | Evidence from the full run | Control in the twin | What remains | Owner |
|---|---|---|---|---|---|
| G1 | People take a recommendation for the decision | In the replay, every sign-off followed a stated policy: 19,853 sign-offs, none different from the recommendation (11.1) | A person in a stated role decides every change to land, plans and tasks; the twin refuses a decision with no role, an override with no written reason, and a release its rule check vetoes; one log (5.3, 8.4) | How real reviewers would use it is untested | Departmental office of the national authority |
| G2 | A model changes under the system: a newer model, a new prompt, or a change by the provider to the hosted model | Both LLM tools use one hosted model (gpt-6-luna); each saved review and report carries a checksum of the model settings (3) | A new model passes the same gate, replay, scenarios and expert sample before use; its runs are new records beside the old (11.5) | A provider can change or retire a model behind the same name, and prices and terms can change | Information management officer |
| G3 | Data on victims or restricted layers reach people who should not have them (IASC, 2023) | Victims shown only as counts by municipality; the restricted layers inside the twin; the situation report's model sent counts only; no coordinates shown (4.2, 7.3) | The layer registry's sensitivity and handling for each layer; public outputs by municipality | Small counts can point to people in a small place; a programme needs its own data-sharing agreements | Information management officer |

## Map

| ID | Risk | Evidence from the full run | Control in the twin | What remains | Owner |
|---|---|---|---|---|---|
| M1 | The simulation is taken for the field | Every result comes from a simulated field under stated settings: the flagging rates, the visits' outcomes, weather and the teams' output (5, 9) | The settings stated in the notebook and `twin/config.py`; results described as behaviour under those settings (1.5, 11.4) | Performance in the field is unknown until tested on a programme's records | National authority |
| M2 | The labels are a partial truth | RELand marks cells with a confirmed hazardous area or a cleared area where mines were found, in 15 municipalities since declared free of mines (`docs/datasheets/03_reland.md`) | Each municipality's chances fitted on the other 14; the gate; the datasheet | Other regions differ; the labels also reflect where surveys and accidents happened | Information management officer |
| M3 | The plan leaves out the impact on people | The planner maximises the expected mine areas found and cleared this season; victims are shown beside the plan, not weighed in it (6.3) | The victims recorded in each zone's municipality shown beside the plan, for the departmental office | Weighing impact needs weights the national authority would set (11.4) | Departmental office |
| M4 | Data or text that helps someone handle or misuse explosive items | The twin holds no technical data on munitions; the handling-advice filter's 2 removals in the full run were both false alarms, register notes that say "second-hand" or "first-hand" (11.2) | Texts at business level; the filter; no coordinates | The filter works on wording: new phrasings may pass, and harmless notes can be removed | Information management officer |
| M5 | An item the team sees needs an EOD spot task, which the twin does not model | SAB-0214, judged an error in the expert review (11.2) | A person decides; the agent flagged the find for qualified personnel | The call-out and the EOD team's assessment are outside the twin (11.4) | Operations officer |

## Measure

| ID | Risk | Evidence from the full run | Control in the twin | What remains | Owner |
|---|---|---|---|---|---|
| Me1 | An area with mines released by decision | 1.4 a season under the main policy (0 to 4), all by cancellation after visits that found nothing and looked like safe use; none in the agent's season (11.1) | The national criteria (two first-hand sources with five years of safe use each); the rule check; the rate measured in the replay | No reviewer can see what a visit did not; the criteria are the programme's | National authority |
| Me2 | An area with mines released with its zone | 1.15 a season (0 to 5); 3.1 mine areas a season in surveyed zones never flagged (9.5) | A zone declared free only once its flagged areas are resolved; new contamination reopens the zone (noticed the same week and cleared in 20 of 20 seasons) | The zone survey misses about one area with mines in seven (Travis's setting) | National authority |
| Me3 | The agent's reading errors pass the rule check | SLU-0447, and ABE-0324's second visit; the 2 reviews that differ from the coded rules (11.2; `docs/model_cards/03_dossier_reviewer.md`) | The evidence register on the sign-off sheet; the expert review of a sample | From 14 judged reviews, the error rate is not known | Quality-control reviewer |
| Me4 | Chances far off in some municipalities | Sonsón 4.49 times as many mine areas as expected, Chigorodó 0.12 times; 27 zones "insufficient evidence" (6.1) | The gate on ranking and calibration; learning in the season; the flag | The planner does not use the flag, and the twin cannot learn where no team goes (11.4) | Information management officer |
| Me5 | The forecast too sure of how soon zones finish | 71% of zones free by their P90 week with the first plan's forecast, 73% with week 27's, against about 90% (9.4) | Forecasts made at set weeks (each month in the agent's season), with ranges, not points | A P90 week reads as safer than it is | Departmental office |
| Me6 | Wrong team reports get through | 42 of 80 reports with inflated days accepted; 4 false alarms in 514 reports (7.2, 10) | Coded holds for impossible, duplicate and wrong-task reports; the isolation forest's queries | Inflated days within what a week allows pass | Operations officer |

## Manage

| ID | Risk | Evidence from the full run | Control in the twin | What remains | Owner |
|---|---|---|---|---|---|
| Ma1 | Automation bias at sign-off (Parasuraman & Manzey, 2010) | No sign-off in the replay overrode a recommendation (11.1) | Every recommendation with its evidence register and rule check; an override needs a written reason and is logged; chances with their spread; "insufficient evidence" in place of a guess | The sign-off sheet puts the recommendation before the evidence register; real reviewers untested (11.6) | Quality-control reviewer |
| Ma2 | Stale or missing data | The silent team flagged a week after its first lost report in 16 of 20 seasons, two weeks after in 4; both dossiers with a missing item held in 20 of 20 (10) | Staleness shown on every record; overlays answer "insufficient evidence" on a stale feed; incomplete dossiers held | A two-week delay in 4 of 20 seasons | Operations officer |
| Ma3 | Text in a dossier that tries to instruct the agent | Flagged and not followed in section 8.3's tests; none in the full run's dossiers | The scan for text addressed to the reviewer; tools that cannot change the twin; the rule check in code | New wording may pass the scan; the rule check and the person still stand | Information management officer |
| Ma4 | An outage of the model's service | The full run's agent season stopped twice on short network drops and resumed from its saved reviews; the client now retries for longer (9.2) | A run-scope error stops the run without saving the review; a review-scope error falls back to further survey; every review saved as it finishes | A long outage stops the agent's reviews until the service returns | Information management officer |
| Ma5 | Cost, and plans that depend on a computer's speed | USD 0.246 for 96 reviews; 6 of 200 seasons had a plan stopped by the planner's two-minute limit with 8 seasons running at once on one computer, and were run again with 2; the saved seasons have none (11.3) | Costs reported each run; plans stopped by the limit counted in each season | Costs grow with the dossiers; a plan stopped by the limit can differ between computers | Information management officer |

## References
Inter-Agency Standing Committee. (2023). *IASC operational guidance on data responsibility in humanitarian action*.
https://interagencystandingcommittee.org/system/files/2023-04/IASC%20Operational%20Guidance%20on%20Data%20Responsibility%20in%20Humanitarian%20Action%2C%202023.pdf

National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF
1.0)* (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1

Parasuraman, R., & Manzey, D. H. (2010). Complacency and bias in human use of automation: An attentional integration.
*Human Factors, 52*(3), 381–410. https://doi.org/10.1177/0018720810376055
