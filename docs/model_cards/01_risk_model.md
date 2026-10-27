# Model card: risk prioritisation

Sections follow Mitchell et al. (2019), *Model cards for model reporting*, shortened, with a section on replacing the
model.

**Code:** `twin/prioritise.py` (`RiskModels` fits the rankings, the gate and the calibration; `RiskOverlay` is the
decision tool); the fitted scores, gate and calibration in `models/`.
Notebook section 6.1.

## Model details
- Three rankings of the pilot's 7,280 areas, each scoring RELand's 500 m cells, an area taking its highest cell's
  score: nearness to past landmine accidents (the baseline: minus the distance to the nearest recorded accident), and
  P3's two models, logistic regression and gradient-boosted trees (scikit-learn 1.9.1), on 63 features.
- Each ranking's scores become chances, one held-out municipality at a time, calibrated on the other 14
  municipalities' areas (25 bins of equal size, made monotone).
- The gate chooses one ranking for each municipality: a model is used only where it beats the baseline on both
  ranking (mean area-level PR-AUC) and calibration, on the other 14 municipalities. In the fitted files: the baseline
  in 11 municipalities, the trees in 3 (El Carmen de Viboral, San Luis, San Roque) and logistic regression in 1
  (Sonsón); on ranking alone, logistic regression would have been chosen in 13.
- In the season, each municipality's odds are multiplied by a factor learned each month from the season's own flags
  and outcomes, with no label read (a log-normal prior around 1, spread 0.6: this project's setting), rounded to 6
  significant digits so that the same season runs on any computer.
- Fitted on the first run and saved, so later runs load the same chances.

## Intended use
- Where to survey first, and the expected mine areas of each zone for the season planner and the forecaster.
- Not for releasing or declaring land, or any decision about an area on its own: the dossier states that the
  survey priority "is not evidence about this area".

## Factors
The municipality: the chances of each are calibrated on the other 14, and they can be far off in a given one. A zone
most of whose areas score outside the range the chances were calibrated on is flagged "insufficient evidence".

## Metrics
Area-level PR-AUC within each municipality (ranking); the mean of |log((observed + 1) / (expected + 1))| over
municipalities (calibration); and, in the season, the expected against the observed mine areas in the zones each plan
picks.

## Training and evaluation data
RELand's grid of 26,761 cells in 15 municipalities of Antioquia (Dulce Rubio et al., 2024), 430 of them mine cells
(`docs/datasheets/03_reland.md`); each municipality held out in turn. Only the fit and the evaluation read the labels.

## Quantitative analyses (the full run)
- Mean area-level PR-AUC on each municipality's own labels: the gate's choice 0.274, nearness 0.275, the trees 0.302,
  logistic regression 0.328 (random order 0.058).
- Expected against observed mine areas under the gate's chances, by municipality: from 8.68 expected and 1 observed in
  Chigorodó (0.12 times) to 14.69 expected and 66 observed in Sonsón (4.49 times).
- 27 zones flagged "insufficient evidence": 18 in San Roque, 9 in Sonsón.
- The integer programme's first 25 zones for survey: 21.5 mine areas expected, 9 observed (section 6.3).
- In the season replay, against nearness alone: 3.4 more mine areas found and 1.3 more cleared a season (higher in 17
  of 20 seeds), and 1.75 fewer zones declared free. Learning in the season: 1.2 more zones declared free a season, and
  0.35 more areas released with mines (28 against 21 over the 20 seasons, with no consistent effect seed by seed;
  section 11.3).

## Ethical considerations
- Chances that are too low leave people waiting: in seed 0's season, no team went to Sonsón, the largest
  municipality, so the season's learning could not correct its chances.
- The labels mark cells with a confirmed hazardous area or a cleared area where mines were found, so they also
  reflect where surveys and accidents happened, and a model learns that pattern too.
- No coordinates are shown; the chances appear by area and zone, inside the twin.

## Caveats and recommendations
- Fit and gate again on a programme's own records before use, and check the calibration again whenever a
  municipality is added.
- Treat an "insufficient evidence" zone as unknown, not as low risk; in this build the planner does not use the flag
  (section 11.4).

## Replacing this model
A new ranking, such as another model, more data or a remote-sensing layer, joins as one more candidate for the gate:
it is used in a municipality only where it beats the baseline on ranking and calibration there, on the other
municipalities' labels. The 20-seed replay then compares the season with and without it, seed by seed. The replay's
saved seasons carry a checksum of the fitted models, so a new model's seasons are run anew and the old ones stay in
the repository's history to compare with (section 11.5).

## References
Dulce Rubio, M., Zeng, S., Wang, Q., Alvarado, D., Moreno Rivera, F., Heidari, H., & Fang, F. (2024). RELand: Risk
estimation of landmines via interpretable invariant risk minimization. *ACM Journal on Computing and Sustainable
Societies, 2*(2), Article 23. https://doi.org/10.1145/3648437

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T.
(2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and
Transparency* (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596
