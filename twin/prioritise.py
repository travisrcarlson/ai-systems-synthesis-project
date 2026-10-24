"""The operational service: which areas of each municipality to survey first (P3).

The default ranking is nearness to past landmine accidents, the rule that beat both of P3's models in 9 of its 15
held-out municipalities. P3's two models, logistic regression and gradient-boosted trees, are trained on other
municipalities' clearance results and shown for a municipality only if they beat the baseline there on evidence
that does not use that municipality's labels (the baseline-first gate). Areas are scored by their highest cell.

Two parts are kept apart:
- `OperationalService` makes the rankings. For a municipality, it reads only the other municipalities' labels.
- `evaluate_rankings` scores the rankings against each municipality's own labels, for the evaluation only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import NAME_OF, PILOT, SEED
from .data import features_of

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


def area_scores(twin, cell_scores):
    """An area's score is its highest cell's score."""
    return cell_scores.groupby(twin.cells["area_id"]).max().reindex(twin.areas.index)


def mean_area_ap(twin, labels, scores, municipalities):
    """Mean area-level PR-AUC (average precision) over municipalities, each ranked within itself."""
    values = []
    for code in municipalities:
        part = twin.areas.index[twin.areas["municipality_code"] == code]
        values.append(average_precision_score(labels[part], scores[part]))
    return float(np.mean(values))


class OperationalService:
    """Fits the rankings and applies the baseline-first gate, one municipality at a time."""

    def __init__(self, twin, truth, inner_folds=3, seed=SEED):
        self.twin = twin
        self.features = features_of(twin.cells)
        self._truth = truth          # cell labels; for a municipality, only the others' labels are ever read
        self.inner = GroupKFold(n_splits=inner_folds)
        self.seed = seed
        self.scores = {}             # ranking -> area scores, every municipality scored by a model blind to it
        self.gate = None             # one row per municipality: inner evidence and the ranking chosen

    def _area_labels(self, cell_mask):
        return self._truth[cell_mask].groupby(self.twin.cells.loc[cell_mask, "area_id"]).max()

    def fit(self, verbose=True):
        cells, X = self.twin.cells, self.twin.cells[self.features]
        codes = cells["municipality_code"].to_numpy()
        baseline_cells = -cells["dist_old_mine"]
        self.scores["baseline"] = area_scores(self.twin, baseline_cells)
        rng = np.random.default_rng(self.seed)
        self.scores["random"] = pd.Series(rng.random(len(self.twin.areas)), index=self.twin.areas.index)
        gate_rows = []
        for name in ("lr", "hgb"):
            held_out = pd.Series(np.nan, index=cells.index)
            for code in PILOT:
                train = codes != code
                model = make_model(name, self.features).fit(X[train], self._truth[train])
                held_out[~train] = model.predict_proba(X[~train])[:, 1]
                # the gate's evidence: the other 14 municipalities, in GroupKFold groups, each scored by a model
                # trained on the remaining groups; the baseline is scored on the same municipalities
                inner_scores = pd.Series(np.nan, index=cells.index)
                train_index = np.flatnonzero(train)
                for fit_rows, score_rows in self.inner.split(train_index, groups=codes[train]):
                    fit_index, score_index = train_index[fit_rows], train_index[score_rows]
                    inner_model = make_model(name, self.features).fit(X.iloc[fit_index], self._truth.iloc[fit_index])
                    inner_scores.iloc[score_index] = inner_model.predict_proba(X.iloc[score_index])[:, 1]
                others = [c for c in PILOT if c != code]
                labels = self._area_labels(train)
                model_areas = area_scores(self.twin, inner_scores).dropna()
                gate_rows.append({"municipality_code": code, "model": name,
                                  "model mean PR-AUC (others)": mean_area_ap(self.twin, labels, model_areas, others),
                                  "baseline mean PR-AUC (others)": mean_area_ap(
                                      self.twin, labels, self.scores["baseline"][labels.index], others)})
            self.scores[name] = area_scores(self.twin, held_out)
            if verbose:
                print(f"{RANKINGS[name]}: fitted for 15 municipalities, with the gate's inner evaluation", flush=True)
        gate = pd.DataFrame(gate_rows).pivot(index="municipality_code", columns="model")
        gate.columns = [f"{model}: {measure}" for measure, model in gate.columns]
        chosen = []
        for code, row in gate.iterrows():
            margins = {m: row[f"{m}: model mean PR-AUC (others)"] - row[f"{m}: baseline mean PR-AUC (others)"]
                       for m in ("lr", "hgb")}
            best = max(margins, key=margins.get)
            chosen.append(best if margins[best] > 0 else "baseline")
        gate["chosen"] = chosen
        gate.insert(0, "municipality", gate.index.map(NAME_OF))
        self.gate = gate
        return self

    def save(self, scores_file, gate_file):
        """Save the area scores of every ranking and the gate's table, so that later runs rank exactly the same way
        (a refit on another computer can differ in the last digits, enough to swap two close areas)."""
        pd.DataFrame(self.scores).rename_axis("area_id").to_csv(scores_file, lineterminator="\n")
        self.gate.to_csv(gate_file, lineterminator="\n")

    def load(self, scores_file, gate_file):
        """Load saved area scores and the gate's table instead of fitting."""
        scores = pd.read_csv(scores_file, index_col="area_id", float_precision="round_trip").reindex(
            self.twin.areas.index)
        if scores.isna().any().any():
            raise ValueError(f"{scores_file} does not hold a score for every area.")
        self.scores = {name: scores[name] for name in scores.columns}
        self.gate = pd.read_csv(gate_file, index_col="municipality_code", dtype={"municipality_code": str},
                                float_precision="round_trip")
        return self

    def priority(self, ranking="gated"):
        """One row per area: its score and rank within its municipality under a ranking, and its baseline rank.

        ranking: "gated" (the gate's choice for each municipality), or one of "baseline", "lr", "hgb", "random".
        """
        areas = self.twin.areas
        if ranking == "gated":
            chosen = areas["municipality_code"].map(self.gate["chosen"])
        else:
            chosen = pd.Series(ranking, index=areas.index)
        score = pd.Series([self.scores[r][a] for a, r in chosen.items()], index=areas.index)
        table = pd.DataFrame({"score": score, "ranking": chosen.map(RANKINGS), "municipality_code":
                              areas["municipality_code"], "number": areas["number"]})
        # rank 1 is surveyed first; ties go to the lower area number, so every ranking is a strict order
        order = table.sort_values(["municipality_code", "score", "number"], ascending=[True, False, True])
        table["rank"] = order.groupby("municipality_code").cumcount().reindex(table.index) + 1
        base = pd.DataFrame({"score": self.scores["baseline"], "municipality_code": areas["municipality_code"],
                             "number": areas["number"]})
        order = base.sort_values(["municipality_code", "score", "number"], ascending=[True, False, True])
        table["baseline_rank"] = order.groupby("municipality_code").cumcount().reindex(table.index) + 1
        return table[["score", "rank", "ranking", "baseline_rank"]]


def evaluate_rankings(service, truth, visits=None):
    """Score each ranking in each municipality against that municipality's own labels (evaluation only).

    Returns one row per municipality and ranking: area-level PR-AUC, and the mine areas among the first k areas,
    where k is the municipality's visits in the replay (if given) or 10% of its areas.
    """
    twin = service.twin
    labels = truth.groupby(twin.cells["area_id"]).max().reindex(twin.areas.index)
    rows = []
    for ranking in ("baseline", "lr", "hgb", "random", "gated"):
        table = service.priority(ranking)
        for code in PILOT:
            part = table[twin.areas["municipality_code"] == code]
            y = labels[part.index]
            k = int(visits[code]) if visits is not None else max(1, round(0.1 * len(part)))
            top = part.sort_values("rank").index[:k]
            rows.append({"municipality_code": code, "municipality": NAME_OF[code], "ranking": ranking,
                         "areas": len(part), "mine areas": int(y.sum()), "PR-AUC": average_precision_score(y, part["score"]),
                         "first k areas": k, "mine areas in first k": int(y[top].sum())})
    return pd.DataFrame(rows)
