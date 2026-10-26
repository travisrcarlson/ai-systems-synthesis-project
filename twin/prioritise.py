"""Risk prioritisation: where in the pilot mines are likeliest, as calibrated chances by area and expected mine
areas by zone (machine learning; P3's insight: a simple baseline first, tested on held-out municipalities).

The default ranking is nearness to past landmine accidents, the rule that beat both of P3's models in 9 of its 15
held-out municipalities. P3's two models, logistic regression and gradient-boosted trees, are trained on other
municipalities' clearance results and used for a municipality only if they beat the baseline there on evidence
that does not use that municipality's labels (the baseline-first gate). Areas are scored by their highest cell.

A score only orders areas; the season planner needs chances, and compares them across municipalities. Each
ranking's scores are calibrated, one held-out municipality at a time, on the other 14 municipalities' areas: the
share of mined areas among those with similar scores, in 25 bins of equal size, made monotone and read between the
bins' centres. For the models, the calibration uses the gate's inner scores, from models that never saw the
municipality, so no municipality's labels shape its own chances.

The gate therefore tests both things the planner relies on, on the other 14 municipalities only: ranking (mean
area-level PR-AUC, as in P3) and calibration (how far each municipality's expected mine areas fall from its
observed ones, as the mean of |log((observed + 1) / (expected + 1))|, with each group of municipalities calibrated
on the other groups). A model is used for a municipality only if it beats the baseline on both. The calibration
test was added after the first season replays showed the planner choosing the zones where a model's chances were
furthest too high (the optimiser's curse): a model can rank areas well within a municipality and still give
chances that do not carry from one municipality to another.

As the season goes on, the twin learns: each month the chances of each municipality are updated from the season's
own field results there, with no label read (P2's insight: watch change over time). The update is Bayesian, on one
multiplier k of the odds of every area in the municipality: its prior is log-normal around 1, with a spread set
from the gate's own measure of how far chances miss between municipalities (`LEARN_SD`, this project's setting);
the evidence is which areas of the surveyed zones people pointed to (Travis's flagging rates: 85% of the areas with
mines, 5% of the others), and which areas an intrusive check or direct evidence showed to hold mines or not.

Two parts are kept apart:
- `RiskModels` fits the rankings, the gate and the calibration (it reads labels: for a municipality, only the
  others'), and saves them in `models/`, so later runs give exactly the same chances.
- `RiskOverlay` is the decision tool: it reads the twin's view and the fitted models, and answers with the chances,
  the zones' expected mine areas with their spread, the evidence, and the limits; or "insufficient evidence".
- `evaluate_rankings` and `calibration_table` score the rankings against each municipality's own labels, for the
  evaluation only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .agent import write_whole
from .config import NAME_OF, PILOT, SEED
from .data import features_of
from .overlays import Overlay

# P3's preprocessing: log(1 + x) for the skewed distances and counts, then standardising; one-hot columns as they are
LOG_FEATURES = ["airports_dist", "seaport_dist", "settlement_dist", "finance_dist", "edu_dist", "buildings_dist",
                "waterways_dist", "coca_dist", "dist_roads_t1", "dist_roads_t2", "dist_roads_t3", "dist_old_mine",
                "discapital", "population_2012", "nighttime_lights_2012"]
ONEHOT_PREFIXES = ("land_use_", "weather_", "relief_")
# P3 tuned these settings in each round; its tuning chose C = 0.001 in 8 of 15 rounds, and a learning rate of
# 0.03 or 0.1 in 12 of 15 (6 each). This project fixes C = 0.001 and keeps the library's default learning rate,
# 0.1, so that the gate's evaluation on other municipalities stays affordable.
LR_C = 0.001
HGB_LEARNING_RATE = 0.1
CALIBRATION_BINS = 25
# This project's setting: the prior spread (sd of log k) of the in-season update, from the gate's calibration error
# between municipalities (about 0.5 as a mean absolute log ratio, so an sd of about 0.6)
LEARN_SD = 0.6
LEARN_GRID = np.exp(np.linspace(-2.5, 2.5, 101))

RANKINGS = {"baseline": "nearness to past accidents", "lr": "logistic regression",
            "hgb": "gradient-boosted trees", "random": "random order"}


def make_model(name, features):
    if name == "lr":
        log_columns = [c for c in LOG_FEATURES if c in features]
        onehot = [c for c in features if c.startswith(ONEHOT_PREFIXES)]
        other = [c for c in features if c not in log_columns and c not in onehot]
        preprocess = ColumnTransformer(
            [("log_then_scale", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                          ("scale", StandardScaler())]), log_columns),
             ("scale", StandardScaler(), other), ("one_hot", "passthrough", onehot)],
            verbose_feature_names_out=False)
        model = LogisticRegression(solver="newton-cholesky", tol=1e-6, class_weight="balanced", max_iter=1000,
                                   C=LR_C)
        return Pipeline([("preprocess", preprocess), ("model", model)])
    if name == "hgb":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED,
                                              learning_rate=HGB_LEARNING_RATE)
    raise ValueError(name)


def area_scores(areas_of_cells, cell_scores, index):
    """An area's score is its highest cell's score."""
    return cell_scores.groupby(areas_of_cells).max().reindex(index)


def mean_area_ap(municipality_of, labels, scores, municipalities):
    """Mean area-level PR-AUC (average precision) over municipalities, each ranked within itself."""
    values = []
    for code in municipalities:
        part = municipality_of.index[municipality_of == code]
        values.append(average_precision_score(labels[part], scores[part]))
    return float(np.mean(values))


def fit_calibration(scores, labels, bins=CALIBRATION_BINS):
    """A monotone map from scores to chances, from labelled areas: (score at each bin's centre, chance, areas).

    The areas are sorted by score and cut into bins of equal size; each bin's chance is its share of mined areas,
    with half an area added to each side so that no bin says 0 or 1; the chances are then made monotone in the
    score, weighted by the bins' sizes."""
    order = np.argsort(scores.to_numpy(), kind="stable")
    s, y = scores.to_numpy()[order], labels.to_numpy()[order].astype(float)
    parts = np.array_split(np.arange(len(s)), bins)
    centre = np.array([s[p].mean() for p in parts])
    rate = np.array([(y[p].sum() + 0.5) / (len(p) + 1) for p in parts])
    size = np.array([len(p) for p in parts])
    rate = IsotonicRegression(increasing=True).fit(centre, rate, sample_weight=size).predict(centre)
    return pd.DataFrame({"score": centre, "chance": rate, "areas": size, "low": [s[p].min() for p in parts],
                         "high": [s[p].max() for p in parts]})


def calibration_error(scores, labels, municipality_of, folds):
    """How far expected mine areas fall from observed ones, by municipality, when each group of municipalities is
    calibrated on the other groups: the mean over municipalities of |log((observed + 1) / (expected + 1))|."""
    errors = []
    for g in np.unique(folds):
        here, other = folds == g, folds != g
        table = fit_calibration(scores[other], labels[other])
        chance, _ = apply_calibration(table, scores[here])
        part = pd.DataFrame({"expected": chance, "observed": labels[here], "m": municipality_of[here]})
        by = part.groupby("m").sum()
        errors += list(np.abs(np.log((by["observed"] + 1) / (by["expected"] + 1))))
    return float(np.mean(errors))


def apply_calibration(table, scores):
    """Chances for scores, read between the bins' centres (flat beyond the end bins), and whether each score lies
    outside the range of the scores the calibration was fitted on."""
    chance = np.interp(scores.to_numpy(), table["score"].to_numpy(), table["chance"].to_numpy())
    outside = (scores < table["low"].min()) | (scores > table["high"].max())
    return pd.Series(chance, index=scores.index), outside


class RiskModels:
    """Fits the rankings, the baseline-first gate and the calibration, one held-out municipality at a time."""

    def __init__(self, twin, truth, inner_folds=3, seed=SEED):
        self.areas = twin.areas[["municipality_code", "zone_id", "number", "event_inside"]].copy()
        self.cells = twin.cells
        self.features = features_of(twin.cells)
        self._truth = truth          # cell labels; for a municipality, only the others' labels are ever read
        self.inner = GroupKFold(n_splits=inner_folds)
        self.seed = seed
        self.scores = {}             # ranking -> area scores, every municipality scored by a model blind to it
        self.gate = None             # one row per municipality: inner evidence and the ranking chosen
        self.calibration = None      # one row per ranking, municipality and bin

    def _area_labels(self, cell_mask):
        return self._truth[cell_mask].groupby(self.cells.loc[cell_mask, "area_id"]).max()

    def fit(self, verbose=True):
        cells, X = self.cells, self.cells[self.features]
        codes = cells["municipality_code"].to_numpy()
        index = self.areas.index
        self.scores["baseline"] = area_scores(cells["area_id"], -cells["dist_old_mine"], index)
        rng = np.random.default_rng(self.seed)
        self.scores["random"] = pd.Series(rng.random(len(index)), index=index)
        all_labels = self._area_labels(np.ones(len(cells), dtype=bool)).reindex(index)
        gate_rows, calibration = [], []
        for code in PILOT:
            others = index[self.areas["municipality_code"] != code]
            for name in ("baseline", "random"):
                calibration.append(fit_calibration(self.scores[name][others], all_labels[others])
                                   .assign(ranking=name, municipality_code=code))
        for name in ("lr", "hgb"):
            held_out = pd.Series(np.nan, index=cells.index)
            for code in PILOT:
                train = codes != code
                model = make_model(name, self.features).fit(X[train], self._truth[train])
                held_out[~train] = model.predict_proba(X[~train])[:, 1]
                # the gate's evidence: the other 14 municipalities, in GroupKFold groups, each scored by a model
                # trained on the remaining groups; the baseline is scored on the same municipalities
                inner_scores = pd.Series(np.nan, index=cells.index)
                fold_of = {}
                train_index = np.flatnonzero(train)
                for g, (fit_rows, score_rows) in enumerate(self.inner.split(train_index, groups=codes[train])):
                    fit_index, score_index = train_index[fit_rows], train_index[score_rows]
                    inner_model = make_model(name, self.features).fit(X.iloc[fit_index], self._truth.iloc[fit_index])
                    inner_scores.iloc[score_index] = inner_model.predict_proba(X.iloc[score_index])[:, 1]
                    fold_of.update({m: g for m in set(codes[score_index])})
                other_codes = [c for c in PILOT if c != code]
                labels = self._area_labels(train)
                model_areas = area_scores(cells["area_id"], inner_scores, index).dropna()
                municipality_of = self.areas["municipality_code"][labels.index]
                folds = municipality_of.map(fold_of).to_numpy()
                baseline_areas = self.scores["baseline"][labels.index]
                gate_rows.append({"municipality_code": code, "model": name,
                                  "model mean PR-AUC (others)": mean_area_ap(municipality_of, labels, model_areas,
                                                                             other_codes),
                                  "baseline mean PR-AUC (others)": mean_area_ap(
                                      municipality_of, labels, baseline_areas, other_codes),
                                  "model calibration error (others)": calibration_error(
                                      model_areas[labels.index], labels, municipality_of, folds),
                                  "baseline calibration error (others)": calibration_error(
                                      baseline_areas, labels, municipality_of, folds)})
                calibration.append(fit_calibration(model_areas[labels.index], labels)
                                   .assign(ranking=name, municipality_code=code))
            self.scores[name] = area_scores(cells["area_id"], held_out, index)
            if verbose:
                print(f"{RANKINGS[name]}: fitted for 15 municipalities, with the gate's inner evaluation", flush=True)
        gate = pd.DataFrame(gate_rows).pivot(index="municipality_code", columns="model")
        gate.columns = [f"{model}: {measure}" for measure, model in gate.columns]
        chosen, by_ranking = [], []
        for code, row in gate.iterrows():
            margins = {m: row[f"{m}: model mean PR-AUC (others)"] - row[f"{m}: baseline mean PR-AUC (others)"]
                       for m in ("lr", "hgb")}
            calibrated = {m: row[f"{m}: model calibration error (others)"]
                          < row[f"{m}: baseline calibration error (others)"] for m in ("lr", "hgb")}
            passing = [m for m in ("lr", "hgb") if margins[m] > 0 and calibrated[m]]
            best_rank = max(margins, key=margins.get)
            by_ranking.append(best_rank if margins[best_rank] > 0 else "baseline")
            chosen.append(max(passing, key=margins.get) if passing else "baseline")
        gate["chosen by ranking alone"] = by_ranking
        gate["chosen"] = chosen
        gate.insert(0, "municipality", gate.index.map(NAME_OF))
        self.gate = gate
        self.calibration = pd.concat(calibration, ignore_index=True)[
            ["ranking", "municipality_code", "score", "chance", "areas", "low", "high"]]
        return self

    def save(self, directory):
        """Save every area's scores, the gate's table and the calibration, so that later runs give exactly the same
        chances (a refit on another computer can differ in the last digits, enough to swap two close areas)."""
        directory.mkdir(parents=True, exist_ok=True)
        # each file is written whole (to a temporary file, then moved into place), so a stop part-way through never
        # leaves a file that looks saved
        write_whole(directory / "area_scores.csv",
                    pd.DataFrame(self.scores).rename_axis("area_id").to_csv(lineterminator="\n"))
        write_whole(directory / "gate.csv", self.gate.to_csv(lineterminator="\n"))
        write_whole(directory / "calibration.csv", self.calibration.to_csv(index=False, lineterminator="\n"))

    def load(self, directory):
        """Load saved scores, gate and calibration instead of fitting."""
        scores = pd.read_csv(directory / "area_scores.csv", index_col="area_id",
                             float_precision="round_trip").reindex(self.areas.index)
        if scores.isna().any().any():
            raise ValueError(f"{directory / 'area_scores.csv'} does not hold a score for every area.")
        self.scores = {name: scores[name] for name in scores.columns}
        self.gate = pd.read_csv(directory / "gate.csv", index_col="municipality_code", dtype={"municipality_code": str},
                                float_precision="round_trip")
        self.calibration = pd.read_csv(directory / "calibration.csv", dtype={"municipality_code": str},
                                       float_precision="round_trip")
        return self

    def saved(self, directory):
        return all((directory / f).exists() for f in ("area_scores.csv", "gate.csv", "calibration.csv"))

    def chances(self, ranking="gated"):
        """One row per area: the ranking used, its score, its calibrated chance of holding mines, and whether its score
        lies outside the range the calibration was fitted on.

        ranking: "gated" (the gate's choice for each municipality), or one of "baseline", "lr", "hgb", "random".
        """
        a = self.areas
        chosen = (a["municipality_code"].map(self.gate["chosen"]) if ranking == "gated"
                  else pd.Series(ranking, index=a.index))
        rows = []
        for (code, name), part in a.groupby([a["municipality_code"], chosen]):
            table = self.calibration[(self.calibration["ranking"] == name)
                                     & (self.calibration["municipality_code"] == code)]
            score = self.scores[name][part.index]
            chance, outside = apply_calibration(table, score)
            rows.append(pd.DataFrame({"ranking": name, "score": score, "chance": chance, "outside": outside}))
        return pd.concat(rows).reindex(a.index)


class RiskOverlay(Overlay):
    """The decision tool: chances by area, expected mine areas by zone, and the order to survey in."""
    name = "risk prioritisation"
    domain = "machine learning (P3's models, gated by a baseline, calibrated on held-out municipalities)"
    insight = "P3: a simple baseline first, tested on held-out municipalities"
    reads = ("hazard", "land_use")
    limits = ("RELand's labels mark where clearance found mines, so the models also learn where people looked and "
              "reported, not only where mines are",
              "a chance says where to look first; it is not evidence about an area, and no area is released on it",
              "calibrated on the other 14 municipalities of one region; it does not carry to other regions")

    def __init__(self, models, ranking="gated", learn=True, flag_mined=0.85, flag_clean=0.05):
        self.models = models
        self.ranking = ranking
        self.learn = learn
        self.flag_mined, self.flag_clean = flag_mined, flag_clean
        self.prior = models.chances(ranking)

    def evidence_by_municipality(self, view):
        """What the season has shown so far in each municipality, without any label: for the areas of surveyed zones,
        whether the zone survey flagged them, and whether an intrusive check or direct evidence showed mines."""
        areas = view.areas
        surveyed = view.zone_state.index[view.zone_state["surveyed_week"] > 0]
        in_surveyed = areas["zone_id"].isin(surveyed)
        status, found = view.status, view.mines_found
        known = pd.Series(np.nan, index=areas.index)
        known[(status == "confirmed") | found] = 1.0
        known[((status == "cleared") & ~found) | (status == "reduced")] = 0.0
        flagged = view.flag.isin(["people at the community meetings pointed to it",
                                  "a past accident is recorded inside it"])
        return pd.DataFrame({"code": areas["municipality_code"], "surveyed": in_surveyed, "flagged": flagged,
                             "recorded": areas["event_inside"], "known": known})

    def multipliers(self, view, prior):
        """The posterior mean of each municipality's odds multiplier k, and the evidence behind it."""
        ev = self.evidence_by_municipality(view)
        log_prior = -0.5 * (np.log(LEARN_GRID) / LEARN_SD) ** 2
        rows = {}
        for code, part in ev[ev["surveyed"] | ev["known"].notna()].groupby("code"):
            p = prior.loc[part.index].to_numpy()
            k = LEARN_GRID[:, None]
            pk = np.clip(k * p / (1 - p + k * p), 1e-9, 1 - 1e-9)
            fm, fc = self.flag_mined, self.flag_clean
            rec, flag, known = part["recorded"].to_numpy(), part["flagged"].to_numpy(), part["known"].to_numpy()
            ll = np.zeros(len(LEARN_GRID))
            unrec = ~rec & part["surveyed"].to_numpy()
            # an area not recorded, in a surveyed zone: its flag, and its outcome where it is known
            m = unrec & np.isnan(known)
            ll += np.where(flag[m], np.log(fm * pk[:, m] + fc * (1 - pk[:, m])),
                           np.log((1 - fm) * pk[:, m] + (1 - fc) * (1 - pk[:, m]))).sum(axis=1)
            m = unrec & ~np.isnan(known)
            ll += np.where(known[m] == 1, np.log(fm * pk[:, m]), np.log(fc * (1 - pk[:, m]))).sum(axis=1)
            # a recorded area is always flagged; only its outcome tells
            m = rec & ~np.isnan(known)
            ll += np.where(known[m] == 1, np.log(pk[:, m]), np.log(1 - pk[:, m])).sum(axis=1)
            post = np.exp(ll + log_prior - (ll + log_prior).max())
            post /= post.sum()
            # to 6 significant digits: the last digits of log and exp differ between computers, and the season
            # planner's choice among nearly equal plans can turn on them; rounded, the same season runs on any computer
            rows[code] = {"multiplier": float(f"{(LEARN_GRID * post).sum():.6g}"),
                          "areas surveyed": int(part["surveyed"].sum()),
                          "flagged, not recorded": int((part["flagged"] & ~part["recorded"]).sum()),
                          "outcomes known": int(part["known"].notna().sum())}
        return pd.DataFrame.from_dict(rows, orient="index")

    def recommend(self, view, week, **inputs):
        reason = self.check_inputs(view, week)
        if reason:
            return self.insufficient(week, reason)
        chances = self.prior.copy()
        learned = pd.DataFrame()
        if self.learn:
            learned = self.multipliers(view, chances["chance"])
            if len(learned):
                k = view.areas["municipality_code"].map(learned["multiplier"]).fillna(1.0).to_numpy()
                p = chances["chance"].to_numpy()
                chances["prior chance"] = p
                chances["chance"] = k * p / (1 - p + k * p)
        areas = view.areas
        table = chances.join(areas[["municipality_code", "zone_id", "number", "event_inside"]])
        # rank 1 is surveyed first; ties go to the lower area number, so every ranking is a strict order
        for level, key in (("zone_rank", "zone_id"), ("municipality_rank", "municipality_code")):
            order = table.sort_values([key, "score", "number"], ascending=[True, False, True])
            table[level] = order.groupby(key).cumcount().reindex(table.index) + 1
        base = self.models.scores["baseline"]
        order = pd.DataFrame({"score": base, "zone_id": areas["zone_id"], "number": areas["number"]}).sort_values(
            ["zone_id", "score", "number"], ascending=[True, False, True])
        table["baseline_zone_rank"] = order.groupby("zone_id").cumcount().reindex(table.index) + 1
        table["ranking"] = table["ranking"].map(RANKINGS)
        g = table.groupby("zone_id")
        zones = pd.DataFrame({"expected mine areas": g["chance"].sum(),
                              "sd": g["chance"].apply(lambda p: float(np.sqrt((p * (1 - p)).sum()))),
                              "areas": g.size(), "outside": g["outside"].mean()})
        # a zone most of whose areas score outside the calibrated range is flagged "insufficient evidence": its
        # chances are read flat from the end of the calibration and are not reliable. The planner does not use the
        # flag yet (a limitation the notebook states, section 11)
        zones["insufficient"] = zones["outside"] > 0.5
        limits = []
        n_out = int(zones["insufficient"].sum())
        if n_out:
            limits.append(f"{n_out} zone(s) score mostly outside the calibrated range: insufficient evidence for them")
        gate = self.models.gate["chosen"].value_counts().to_dict() if self.ranking == "gated" else {self.ranking: 15}
        evidence = ["hazard layer", "land-use layer",
                    "held-out calibration on the other 14 municipalities",
                    "gate: " + ", ".join(f"{RANKINGS[k]} in {v}" for k, v in gate.items())]
        if len(learned):
            evidence.append(f"the season's field results in {len(learned)} municipalit"
                            f"{'y' if len(learned) == 1 else 'ies'} (zone-survey flags and outcomes)")
        return self.answer(week, {"areas": table, "zones": zones, "learned": learned}, evidence,
                           uncertainty="expected mine areas by zone, with the spread (sd) of a sum of independent "
                                       "chances", limits=limits)


def evaluate_rankings(models, truth, cells, k_share=0.1):
    """Score each ranking in each municipality against that municipality's own labels (evaluation only): area-level
    PR-AUC, and the mine areas among the first 10% of its areas."""
    labels = truth.groupby(cells["area_id"]).max().reindex(models.areas.index)
    rows = []
    for ranking in ("baseline", "lr", "hgb", "random", "gated"):
        table = models.chances(ranking)
        for code in PILOT:
            part = table[models.areas["municipality_code"] == code]
            y = labels[part.index]
            k = max(1, round(k_share * len(part)))
            top = part.sort_values("score", ascending=False, kind="stable").index[:k]
            rows.append({"municipality_code": code, "municipality": NAME_OF[code], "ranking": ranking,
                         "areas": len(part), "mine areas": int(y.sum()),
                         "PR-AUC": average_precision_score(y, part["score"]),
                         "first 10% of areas": k, "mine areas in first 10%": int(y[top].sum())})
    return pd.DataFrame(rows)


def calibration_table(models, truth, cells, ranking="gated"):
    """Expected against observed mine areas by zone, in ten groups of zones by expected mine areas (evaluation
    only): whether the chances are calibrated where the planner uses them."""
    labels = truth.groupby(cells["area_id"]).max().reindex(models.areas.index)
    chances = models.chances(ranking)
    zone = models.areas["zone_id"]
    z = pd.DataFrame({"expected": chances["chance"].groupby(zone).sum(), "observed": labels.groupby(zone).sum()})
    z["group"] = pd.qcut(z["expected"].rank(method="first"), 10, labels=False) + 1
    out = z.groupby("group").agg(zones=("expected", "size"), expected=("expected", "sum"),
                                 observed=("observed", "sum"))
    return out
