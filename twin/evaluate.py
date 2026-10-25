"""The evaluation: the season replay over 20 field seeds, the policies compared, the five scenarios, and forecast
coverage (P2's insight: results as ranges over replicates; P5's: synthetic results test how the system behaves,
not how it would do in the field).

Each case is one season under a policy, a scenario (or none) and a field seed; its record keeps the season's
measures, the weekly totals, and the forecasts made in weeks 1 and 27 against what happened. Records are saved in
`runs/replay/`, one file per case, as soon as the case is done, with a checksum of everything that shapes a season
(the package's code, the rules, the prompts, the fitted models and the data); a saved record is reused only if its
checksum is the same as now, so a changed method always reruns, and a stopped replay resumes. Cases run in parallel
worker processes when `n_jobs` > 1, each building its own copy of the twin, and one after another if that fails.
"""
import hashlib
import json
import os
import platform
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import factors as factors_mod
from . import field as field_mod
from . import layers, prioritise, scenarios, season
from .agent import write_whole
from .checker import ReportChecker
from .config import DATA_FILES, MODELS_DIR, PROMPT_FILE, ROOT, RULES_FILE, SITREP_PROMPT_FILE, Criteria, Season

SEEDS = tuple(range(20))
CHECKER_SEED = 1000            # the clean season the report checker learns from: not one of the 20
POLICIES = {
    "main policy": season.Policy(),
    "rule-of-thumb plan": season.Policy(planner="rule"),
    "risk by nearness to past accidents": season.Policy(ranking="baseline"),
    "technical survey first": season.Policy(technical_survey=True),
    "no learning in the season": season.Policy(learn=False),
}
SCENARIO_NAMES = tuple(scenarios.SCENARIOS)
_SETUP = {}


def setup(technical_survey=False, models_dir=MODELS_DIR, verbose=False):
    """The twin, the truth, the fitted risk models, the planning factors and the trained report checker, for the
    main policy's national criteria or technical survey first; built once per process."""
    key = (bool(technical_survey), str(models_dir))
    if key in _SETUP:
        return _SETUP[key]
    criteria = replace(Criteria(), technical_survey=bool(technical_survey))
    settings = Season().with_technical_survey() if technical_survey else Season()
    twin, truth, checks = layers.build_twin(criteria, settings)
    models = prioritise.RiskModels(twin, truth)
    if models.saved(Path(models_dir)):
        models.load(Path(models_dir))
    else:
        models.fit(verbose=verbose)
        models.save(Path(models_dir))
    factors = factors_mod.planning_factors(twin, criteria)
    policy = season.Policy(technical_survey=bool(technical_survey))
    clean = season.SeasonRun(twin, field_mod.Field(twin, truth, seed=CHECKER_SEED), models, factors, policy=policy,
                             forecast_weeks=()).run()
    checker = ReportChecker().fit([r for r in clean.checks["features"] if r is not None])
    _SETUP[key] = SimpleNamespace(twin=twin, truth=truth, models=models, factors=factors, checker=checker,
                                  checks=checks, criteria=criteria, season=settings)
    return _SETUP[key]


def method_sha(models_dir=MODELS_DIR):
    """A checksum of everything that shapes a season: the package's code, the rules, the prompts, the fitted
    models, the data files' checksums, the policies, the checker's seed and the libraries' versions."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "twin").glob("*.py")) + [RULES_FILE, PROMPT_FILE, SITREP_PROMPT_FILE]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    for path in sorted(Path(models_dir).glob("*.csv")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    digest.update(json.dumps({k: v[1] for k, v in DATA_FILES.items()}, sort_keys=True).encode())
    digest.update(repr(sorted(POLICIES.items())).encode())
    digest.update(str(CHECKER_SEED).encode())
    # the libraries that compute a season: after an upgrade, saved seasons are run again rather than reused
    digest.update(json.dumps(library_versions(), sort_keys=True).encode())
    return digest.hexdigest()


def library_versions():
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def plain(value):
    """A value as JSON can hold it."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def forecast_record(result, i):
    """Forecast `i` of a season against what happened: by zone, and for the season's totals. The totals are compared
    with what happened within the forecast's scope (what was done before it, and what its zones added after it);
    what zones outside its scope added is kept apart."""
    cov = result.forecast_coverage(i)
    if cov is None:
        return None
    f = result.forecasts[i]
    p90 = cov[cov["P90 week"].notna()]
    p50 = cov[cov["P50 week"].notna()]
    done = cov["declared free (week)"].notna()
    scope = set(cov.index)
    before = f.answer["done"]
    t = result.tasks
    cleared = t[(t["kind"] == "CLEARANCE") & (t["status"] == "done")] if len(t) else t
    cleared = cleared[result.mined[cleared["area"]].to_numpy()] if len(cleared) else cleared
    after = cleared[(cleared["finished_week"] >= f.week) & cleared["zone"].isin(scope)] if len(cleared) else cleared
    free = result.zone_state["free_week"]
    within = {"mine areas cleared": int(before["mine areas cleared"]) + int(len(after)),
              "zones declared free": int(before["zones declared free"]) + int(((free > 0) & free.index.isin(scope)).sum())}
    season_total = {k: int(result.metrics[k]) for k in within}
    return {"week": int(f.week), "zones forecast": int(len(cov)), "zones with a P90 week": int(len(p90)),
            "of those, free by their P90 week": int(p90["by P90"].sum()),
            "zones with a P50 week": int(len(p50)), "of those, free by their P50 week": int(p50["by P50"].sum()),
            "sum of chances this season": float(cov["chance this season"].sum()),
            "zones of the forecast declared free": int(done.sum()),
            "brier score": float(((cov["chance this season"] - done.astype(float)) ** 2).mean()),
            "totals": f.answer["totals"], "actual": within,
            "outside the forecast's zones": {k: season_total[k] - within[k] for k in within}}


def record(result, policy_name, scenario_name, seed):
    """What the evaluation keeps of one season: its measures, weekly totals, and each forecast against what
    happened (the first, made in week 1, and the mid-season one, made in week 27)."""
    weeks = range(1, 53)
    free = result.zone_state["free_week"]
    t = result.tasks
    cleared = t[(t["kind"] == "CLEARANCE") & (t["status"] == "done")] if len(t) else t
    cleared_mined = cleared[result.mined[cleared["area"]].to_numpy()] if len(cleared) else cleared
    weekly = {"zones declared free": [int(((free > 0) & (free <= w)).sum()) for w in weeks],
              "mine areas cleared": [int((cleared_mined["finished_week"] <= w).sum()) for w in weeks],
              "team-days worked": [int(result.week_days.get(w, {}).get("worked", 0)) for w in weeks],
              "team-days lost to weather": [int(result.week_days.get(w, {}).get("weather", 0)) for w in weeks]}
    forecasts = [forecast_record(result, i) for i in range(len(result.forecasts))]
    return {"policy": policy_name, "scenario": scenario_name or "none", "seed": int(seed),
            "metrics": {k: plain(v) for k, v in result.metrics.items()}, "weekly": weekly,
            "forecasts": [f for f in forecasts if f is not None]}


FORECAST_WEEKS = (1, 27)       # the season's first plan, and the plan at the start of July, half way through


def run_case(policy_name, scenario_name, seed, models_dir=MODELS_DIR, cache_dir=None, method=None):
    """One season of the replay, as a record; saved in `cache_dir` (with the method's checksum) as soon as it is
    done, so that a stopped replay keeps the seasons it finished."""
    policy = POLICIES[policy_name]
    s = setup(policy.technical_survey, models_dir)
    scenario = scenarios.SCENARIOS[scenario_name]() if scenario_name and scenario_name != "none" else None
    field = field_mod.Field(s.twin, s.truth, seed=seed)
    result = season.SeasonRun(s.twin, field, s.models, s.factors, policy=policy, scenario=scenario,
                              checker=s.checker, forecast_weeks=FORECAST_WEEKS).run()
    rec = record(result, policy_name, scenario_name, seed)
    rec["method_sha"] = method
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        write_whole(case_path(cache_dir, (policy_name, scenario_name, seed)),
                    json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def case_path(cache_dir, case):
    policy, scenario, seed = case
    name = f"{policy}__{scenario or 'none'}__seed{seed:02d}".replace(" ", "_")
    return Path(cache_dir) / f"{name}.json"


def replay(cases, n_jobs=1, cache_dir=None, models_dir=MODELS_DIR, verbose=True):
    """Run (or read back) the cases, each (policy name, scenario name or None, seed). Returns their records."""
    method = method_sha(models_dir)
    found, todo = {}, []
    for case in cases:
        path = case_path(cache_dir, case) if cache_dir else None
        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("method_sha") == method:
                found[case] = saved
                continue
        todo.append(case)
    if verbose:
        print(f"{len(found)} of {len(cases)} season(s) read from saved records; {len(todo)} to run", flush=True)
    results = []
    if todo:
        if n_jobs > 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
                    delayed(run_case)(*case, models_dir, cache_dir, method) for case in todo)
            except Exception as error:     # parallel workers unavailable: run the rest one after another
                if verbose:
                    print(f"Parallel workers failed ({type(error).__name__}: {error}); running one after another.")
                results = []
        if not results:
            results = []
            for i, case in enumerate(todo, start=1):
                path = case_path(cache_dir, case) if cache_dir else None
                saved = json.loads(path.read_text(encoding="utf-8")) if path is not None and path.exists() else None
                if saved is not None and saved.get("method_sha") == method:
                    results.append(saved)       # finished by a worker before the parallel run failed
                else:
                    results.append(run_case(*case, models_dir, cache_dir, method))
                if verbose and (i % 10 == 0 or i == len(todo)):
                    print(f"  {i} of {len(todo)} season(s) run", flush=True)
        for case, rec in zip(todo, results):
            found[case] = rec
    return [found[case] for case in cases]


def frame(records):
    """One row per season: its policy, scenario and seed, and its measures."""
    return pd.DataFrame([{"policy": r["policy"], "scenario": r["scenario"], "seed": r["seed"], **r["metrics"]}
                         for r in records])


def ranges(df, measures, by="policy"):
    """Mean, minimum and maximum of each measure, by policy (or scenario): counts as a mean to one decimal with the
    whole-number range, shares to two decimals, other measures to one."""
    rows = []
    for name, part in df.groupby(by, sort=False):
        row = {by: name, "seasons": len(part)}
        for m in measures:
            v = pd.to_numeric(part[m], errors="coerce").dropna()
            if not len(v):
                row[m] = "-"
            elif (v == v.round()).all():
                row[m] = f"{v.mean():.1f} ({v.min():.0f}–{v.max():.0f})"
            else:
                d = 2 if v.abs().max() <= 1 else 1
                row[m] = f"{v.mean():.{d}f} ({v.min():.{d}f}–{v.max():.{d}f})"
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


def paired(df, measures, base="main policy", by="policy"):
    """Each policy's difference from the base, seed by seed: the mean difference, its range, and in how many seeds
    the policy did better, the same, or worse (higher is better for every measure given)."""
    base_rows = df[df[by] == base].set_index("seed")
    rows = []
    for name, part in df.groupby(by, sort=False):
        if name == base:
            continue
        part = part.set_index("seed")
        seeds = part.index.intersection(base_rows.index)
        for m in measures:
            d = pd.to_numeric(part.loc[seeds, m]) - pd.to_numeric(base_rows.loc[seeds, m])
            rows.append({by: name, "measure": m, "mean difference": round(float(d.mean()), 2),
                         "range": f"{d.min():g} to {d.max():g}", "better": int((d > 0).sum()),
                         "same": int((d == 0).sum()), "worse": int((d < 0).sum())})
    return pd.DataFrame(rows)


def coverage_table(records, week=1):
    """Each season's forecast made in `week` (1, the first; 27, the mid-season one) against what happened within its
    scope, with what zones outside its scope added."""
    rows = []
    for r in records:
        c = next((f for f in r.get("forecasts", []) if f["week"] == week), None)
        if c is None:
            continue
        t, a = c["totals"], c["actual"]
        rows.append({"policy": r["policy"], "seed": r["seed"], "zones with a P90 week": c["zones with a P90 week"],
                     "free by P90": c["of those, free by their P90 week"],
                     "zones with a P50 week": c["zones with a P50 week"],
                     "free by P50": c["of those, free by their P50 week"],
                     "expected zones free (sum of chances)": c["sum of chances this season"],
                     "zones of the forecast free": c["zones of the forecast declared free"],
                     "brier score": c["brier score"],
                     "zones free: P10": t["zones declared free"]["P10"], "zones free: P90": t["zones declared free"]["P90"],
                     "zones free: actual": a["zones declared free"],
                     "mine areas cleared: P10": t["mine areas cleared"]["P10"],
                     "mine areas cleared: P90": t["mine areas cleared"]["P90"],
                     "mine areas cleared: actual": a["mine areas cleared"],
                     "zones free outside the forecast's zones": c["outside the forecast's zones"]["zones declared free"],
                     "mine areas cleared outside the forecast's zones":
                         c["outside the forecast's zones"]["mine areas cleared"]})
    return pd.DataFrame(rows)


def default_jobs():
    """Worker processes for the replay: one fewer than the computer's processors, at most 8."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))
