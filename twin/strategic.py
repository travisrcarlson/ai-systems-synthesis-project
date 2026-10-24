"""The strategic service: where national effort goes, from the people and activity layers (P1 and P2).

It reads the twin's national tables only: victims by municipality and month (P1's data) and demining results by
municipality and month (P2's data). It reports where harm continues without recorded clearance work, with the
uncertainty of small counts, and sets how the pilot's survey capacity is shared among its municipalities.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .config import NAME_OF, PILOT


def poisson_interval(count, confidence=0.95):
    """Exact (Garwood) interval for the mean of a Poisson count."""
    alpha = 1 - confidence
    count = np.asarray(count, dtype=float)
    lower = np.where(count > 0, stats.chi2.ppf(alpha / 2, 2 * count) / 2, 0.0)
    upper = stats.chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2
    return lower, upper


def national_series(twin):
    """Victims (by condition) and demining results per year, for the whole country."""
    victims = twin.people.pivot_table(index="year", columns="condition", values="victims", aggfunc="sum",
                                      fill_value=0)
    victims["All victims"] = victims.sum(axis=1)
    demining = twin.activity.groupby("year")[["area_cleared_m2", "items_total", "non_technical_surveys"]].sum()
    return victims.join(demining, how="outer").fillna(0)


def harm_without_clearance(twin, first_year=2019, last_year=2023):
    """Municipalities with victims in the recent years, against the demining recorded there in the same years.

    The yearly rate of victims has an exact 95% Poisson interval, because most counts are small. A municipality
    is flagged where victims were recorded but no land was cleared in those years.
    """
    years = last_year - first_year + 1
    recent = twin.people[twin.people["year"].between(first_year, last_year)]
    victims = recent.groupby("municipality_code").agg(department=("department", "first"),
                                                       municipality=("municipality", "first"),
                                                       victims=("victims", "sum"))
    work = twin.activity[twin.activity["year"].between(first_year, last_year)].groupby("municipality_code")[
        ["area_cleared_m2", "non_technical_surveys", "technical_surveys"]].sum()
    table = victims.join(work, how="left").fillna({"area_cleared_m2": 0, "non_technical_surveys": 0,
                                                   "technical_surveys": 0})
    lower, upper = poisson_interval(table["victims"])
    table["victims per year"] = table["victims"] / years
    table["95% interval"] = [f"{lo / years:.1f} to {hi / years:.1f}" for lo, hi in zip(lower, upper)]
    table["no land cleared"] = table["area_cleared_m2"] == 0
    return table.sort_values("victims", ascending=False)


def pilot_summary(twin):
    """The pilot's municipalities: recorded victims and demining, from the national layers."""
    table = twin.municipal_table().loc[list(PILOT)]
    table.index.name = "municipality_code"
    table.insert(0, "name", table.index.map(NAME_OF))
    return table.drop(columns=["department", "municipality", "in_pilot"])


def capacity_shares(twin, by="victims"):
    """Each pilot municipality's share of the survey capacity.

    by="victims": in proportion to the victims recorded there since 1990 (the people layer), so effort goes
    first where harm was greatest; by="suspected": in proportion to its areas with a past accident recorded inside
    (the hazard layer at the start); by="areas": in proportion to its number of areas, a neutral comparison.
    """
    if by == "victims":
        weights = twin.people[twin.people["municipality_code"].isin(list(PILOT))].groupby(
            "municipality_code")["victims"].sum().reindex(list(PILOT)).fillna(0)
    elif by == "suspected":
        weights = twin.areas.groupby("municipality_code")["event_inside"].sum().reindex(list(PILOT))
    elif by == "areas":
        weights = twin.areas.groupby("municipality_code").size().reindex(list(PILOT))
    else:
        raise ValueError(by)
    return (weights / weights.sum()).rename("share")


def round_allocation(shares, visits_per_round, rounds):
    """Visits per municipality in each round, as close to the shares as whole visits allow.

    Each round, every municipality earns credit in proportion to its share; visits go one at a time to the
    municipality with the most credit, which then spends one. Totals stay within one visit of the shares.
    """
    credit = pd.Series(0.0, index=shares.index)
    plan = []
    for _ in range(rounds):
        credit += shares * visits_per_round
        visits = pd.Series(0, index=shares.index)
        for _ in range(visits_per_round):
            top = credit.idxmax()
            visits[top] += 1
            credit[top] -= 1
        plan.append(visits)
    return pd.DataFrame(plan, index=pd.RangeIndex(1, rounds + 1, name="round")).T
