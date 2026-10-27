# Model card: the report checker

Sections follow Mitchell et al. (2019), *Model cards for model reporting*, shortened, with a section on replacing the
model.

**Code:** `twin/checker.py`. Notebook sections 7.2 and 10.

## Model details
- **Coded checks** on every team report: its fields are present; its days are whole numbers from 0 to 5 that fit in a
  week; it reports on a task ordered to that team, still open, in that zone and area; it is not a second copy; no work
  is reported in a zone the access feed showed closed; a finished task carries the result its kind needs; a technical
  survey or clearance is not reported done in under a quarter of its planned days. Across a team's run of reports,
  three weeks in a row with work and no day lost to rain or wind, in a rainy month, is queried.
- **An isolation forest** (scikit-learn; 200 trees; contamination 0.01) reads each report as numbers: the days worked
  and lost, the month's expected loss to weather, the task's kind, its progress against the planned days, and the
  items and mine cells reported. It is fitted on the 503 reports of a season with no fault injected (field seed 1000,
  not one of the replay's seeds), flags unusual reports for the operations officer to query, and answers
  "insufficient evidence" until it has been fitted.
- An impossible or inconsistent report is held, and the team asked to send it again; a report whose evidence items
  did not all arrive is accepted with a flag, and the twin holds the dossier until the rest arrives.

## Intended use
Catching reports that are impossible, inconsistent, incomplete or unusual before they change the twin (P1's insight:
check data as it arrives). Not for judging a team's honesty or performance: a query is a question to the team, not a
finding.

## Metrics
Faults caught (held or queried) among those injected, by kind; false alarms among reports with no fault injected.

## Evaluation data
Seed 0's season under the main policy (514 reports), and the stale or corrupted feed scenario over 20 seasons, each
with 7 corrupted reports from one clearance team in weeks 40 to 46: an impossible week, a duplicate, a wrong task and
four with inflated days.

## Quantitative analyses (the full run)
- Every impossible week, duplicate and wrong task held, in 20 of 20 seasons (60 of 60).
- Inflated days: 38 of 80 queried, by the isolation forest or the check on three weeks in a row with no day lost to
  rain or wind; 42 accepted.
- All corrupted reports: 98 of 140 held or queried.
- False alarms: in seed 0's season, 4 of 514 reports queried with no fault injected (0.8%); in the scenario, 5.85
  reports a season with no fault held or queried (0 to 11).
- The survey team whose reports were lost was flagged as stale one week after its first lost report in 16 seasons,
  two weeks after in 4.

## Ethical considerations
A query costs a team time and trust, so the false alarms are counted and shown with the catches. The checker reads no
personal data: teams are codes.

## Caveats and recommendations
- Days inflated within what a week allows look like an honest report; only a pattern over time, or the team's own
  records, would show them.
- The forest's idea of a usual report comes from one simulated clean season: fit it again on a programme's own
  reports before use.

## Replacing this model
A new detector keeps the coded checks, which are rules, not learned. It is compared with the isolation forest on the
same scenario, faults caught by kind, and on a clean season, false alarms, seed by seed; it answers "insufficient
evidence" until it has been fitted (section 11.5).

## Reference
Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T.
(2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and
Transparency* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596
