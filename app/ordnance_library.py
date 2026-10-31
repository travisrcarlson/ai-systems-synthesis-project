"""Contamination categories: what Colombia's national demining register records in the pilot's 15 municipalities,
and what the season's dossiers report, in the twin's own terms.

Business level only, as the project's rules require ("Stay at business level: add no technical details about
munitions", `rules/land_release_rules.md`): no device models, mechanisms or calibres.
- The categories are the register's own three, MAP, MUSE and AEI, read from the twin's open demining layer
  (`twin.activity`: datos.gov.co `jbqm-s5tn`, as P2 summed it by municipality and month). The app never reads `data/`.
- In Colombia, all remaining mines are said to have been laid by non-state armed groups and all are of an improvised
  nature (Mine Action Review, 2024, *Clearing the Mines 2024: Colombia*, p. 118).
- The dossiers' statements are simulated by the twin's field and quoted by the reviewer agent in its evidence
  register (`runs/reviews/*_result.json`). They are grouped by the register's evidence class and by the source's
  reliability class, 1 to 7 (IMAS 08.10, §8.2, as `rules/land_release_rules.md` paraphrases it).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from twin import config
from loaders import HUMAN_OUTCOME, REVIEW_OUTCOMES, load_area_ordnance_findings, load_saved_reviews_public

SURFACE = "#faf9f5"
INK = "#181816"
INK_2 = "#52514e"
MUTED = "#85827a"
GRID = "#e4e1d9"

MAR_2024 = (
    "Mine Action Review (2024), *Clearing the Mines 2024: Colombia*, p. 118: \"All the mines remaining in Colombia "
    "are said to have been laid by NSAGs and all are of an improvised nature.\""
)

# The register's three categories (column names as `twin/data.py` renames them)
CATEGORIES = [
    {
        "code": "MAP",
        "column": "items_mines",
        "name": "Minas antipersonal (MAP)",
        "meaning": "Anti-personnel mines. In Colombia, all remaining mines are said to be improvised, laid by "
                   "non-state armed groups (Mine Action Review, 2024, p. 118).",
        "colour": "#a8322b",
    },
    {
        "code": "MUSE",
        "column": "items_uxo",
        "name": "Munición sin explotar (MUSE)",
        "meaning": "Unexploded ordnance: munitions that were used but did not explode.",
        "colour": "#d9a23a",
    },
    {
        "code": "AEI",
        "column": "items_ieds",
        "name": "Artefactos explosivos improvisados (AEI)",
        "meaning": "Improvised explosive devices.",
        "colour": "#3f6aa8",
    },
]

# The register's evidence classes (`twin/rules.py`), as `rules/land_release_rules.md` defines them
EVIDENCE = {
    "direct": ("Direct evidence", "confirms the presence of explosive ordnance with certainty and can be verified",
               "#a8322b"),
    "indirect": ("Indirect evidence", "suggests explosive ordnance may be present but does not confirm it",
                 "#d9a23a"),
    "not_evidence": ("Not evidence", "neither evidence of explosive ordnance nor of its absence", "#3f6aa8"),
    "absence": ("Evidence of absence", "supports that there is no explosive ordnance, such as safe use of the land "
                "over time", "#2f8f5b"),
}
EVIDENCE_ORDER = list(EVIDENCE)

# Reliability classes 1 to 7 (IMAS 08.10, §8.2, as rules/land_release_rules.md paraphrases them)
RELIABILITY = {
    1: "direct physical evidence seen by the survey team",
    2: "indirect physical evidence seen by the survey team",
    3: "records shown to be reliable",
    4: "first-hand sources",
    5: "second-hand sources",
    6: "records not assessed, or found poor",
    7: "people told by others who are not first-hand sources",
}

# What the app can say about each area, from the twin and the saved runs
AREA_TAGS = {
    "mines_found": "mines found in clearance (log)",
    "direct": "direct evidence (register)",
    "indirect": "indirect evidence (register)",
    "absence": "evidence of absence (register)",
    "recorded_accident": "past accident recorded inside (desk study)",
}


# ---- data ----------------------------------------------------------------------------------------------------------
def register_by_municipality(activity_df: pd.DataFrame) -> pd.DataFrame:
    """Items destroyed by category in each of the 15 municipalities, from the twin's demining layer."""
    cols = [c["column"] for c in CATEGORIES]
    if activity_df is None or activity_df.empty:
        return pd.DataFrame(columns=["municipality"] + [c["code"] for c in CATEGORIES] + ["total"])
    g = activity_df.groupby("municipality_code")[cols].sum()
    g = g.reindex(list(config.NAME_OF), fill_value=0)
    out = g.rename(columns={c["column"]: c["code"] for c in CATEGORIES}).astype(int)
    out.insert(0, "municipality", [config.NAME_OF[c] for c in out.index])
    out["total"] = out[[c["code"] for c in CATEGORIES]].sum(axis=1)
    return out


def register_by_year(activity_df: pd.DataFrame) -> pd.DataFrame:
    """Items destroyed by category and year across the 15 municipalities."""
    cols = [c["column"] for c in CATEGORIES]
    if activity_df is None or activity_df.empty:
        return pd.DataFrame(columns=[c["code"] for c in CATEGORIES])
    g = activity_df.groupby("year")[cols].sum().astype(int)
    return g.rename(columns={c["column"]: c["code"] for c in CATEGORIES})


def register_items(findings: dict[str, dict], areas_meta: pd.DataFrame) -> pd.DataFrame:
    """Every item of the reviewer's evidence registers, one row each (simulated statements, quoted)."""
    rows = []
    for aid, info in findings.items():
        for q in info.get("quotes", []):
            rows.append({
                "area_id": aid,
                "municipality": areas_meta.loc[aid, "municipality"] if aid in areas_meta.index else "",
                "week": int(q["week"]),
                "visit": int(q["visit"]),
                "item": q["item_id"],
                "evidence_class": q["evidence_class"],
                "reliability_class": int(q["reliability_class"]),
                "quote": q["quote"],
            })
    return pd.DataFrame(rows)


def classify_area_evidence(bundle: dict, findings: dict[str, dict], gate_df: pd.DataFrame) -> pd.DataFrame:
    """One row per area: what the twin and the saved runs show about it, with no device type assigned.

    Tags: mines found in clearance (the log's clearance result), the evidence classes in the reviewer's registers
    (direct, indirect, absence), and a past accident recorded inside (the desk study, `event_inside`).
    """
    areas = bundle["areas_meta"]
    gate_map = {}
    if gate_df is not None and not gate_df.empty and "municipality_code" in gate_df.columns:
        label = "chosen_label" if "chosen_label" in gate_df.columns else "chosen"
        gate_map = dict(zip(gate_df["municipality_code"].astype(str), gate_df[label].astype(str)))
    tags_col, quotes_col, cells_col, week_col = [], [], [], []
    for aid, arow in areas.iterrows():
        info = findings.get(str(aid), {})
        tags = []
        if int(info.get("cleared_cells", 0)) > 0:
            tags.append("mines_found")
        classes = {q["evidence_class"] for q in info.get("quotes", [])}
        tags += [t for t in ("direct", "indirect", "absence") if t in classes]
        if bool(arow["event_inside"]):
            tags.append("recorded_accident")
        tags_col.append(tags)
        quotes_col.append(" | ".join(
            f"[week {q['week']}, visit {q['visit']}, {q['evidence_class']}, class {q['reliability_class']}] "
            f"\"{q['quote']}\"" for q in info.get("quotes", [])))
        cells_col.append(int(info.get("cleared_cells", 0)))
        week_col.append(int(info.get("cleared_week", 0)))
    out = areas[["municipality_code", "municipality", "zone_id", "event_inside"]].copy()
    out["tags"] = tags_col
    out["tag_str"] = [", ".join(AREA_TAGS[t] for t in ts) if ts else "—" for ts in tags_col]
    out["has_any_finding"] = [len(ts) > 0 for ts in tags_col]
    out["cleared_cells"] = cells_col
    out["cleared_week"] = week_col
    out["risk_model_family"] = out["municipality_code"].astype(str).map(gate_map).fillna("—")
    out["verbatim_evidence"] = quotes_col
    return out


# ---- figures -------------------------------------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def plot_register_by_municipality(reg: pd.DataFrame) -> plt.Figure:
    """Stacked horizontal bars: items destroyed by category in each municipality, largest total first."""
    df = reg.sort_values("total", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    _style(ax)
    left = np.zeros(len(df))
    for c in CATEGORIES:
        vals = df[c["code"]].to_numpy()
        ax.barh(df["municipality"], vals, left=left, color=c["colour"], height=0.62, label=c["name"],
                edgecolor=SURFACE, linewidth=1.0)
        left += vals
    for y, total in enumerate(df["total"].to_numpy()):
        ax.text(total + max(df["total"].max() * 0.01, 1), y, f"{int(total):,}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Items destroyed, 2008 to 2023 (national register)", fontsize=8.5, color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Items destroyed by category and municipality", fontsize=10, fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    return fig


def plot_register_by_year(by_year: pd.DataFrame) -> plt.Figure:
    """Stacked bars: items destroyed by category and year across the 15 municipalities."""
    fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor=SURFACE)
    _style(ax)
    years = by_year.index.to_numpy()
    bottom = np.zeros(len(by_year))
    for c in CATEGORIES:
        vals = by_year[c["code"]].to_numpy()
        ax.bar(years, vals, bottom=bottom, color=c["colour"], width=0.7, label=c["name"],
               edgecolor=SURFACE, linewidth=1.0)
        bottom += vals
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=7.5, rotation=0)
    ax.set_ylabel("Items destroyed", fontsize=8.5, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Items destroyed by category and year, the 15 municipalities", fontsize=10, fontweight="bold",
                 color=INK, loc="left")
    fig.tight_layout()
    return fig


def plot_evidence_classes(items: pd.DataFrame) -> plt.Figure:
    """Grouped counts: register items by evidence class and reliability class."""
    fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor=SURFACE)
    _style(ax)
    classes = list(range(1, 8))
    width = 0.2
    for i, ev in enumerate(EVIDENCE_ORDER):
        sub = items[items["evidence_class"] == ev]
        counts = [int((sub["reliability_class"] == k).sum()) for k in classes]
        xs = np.array(classes) + (i - 1.5) * width
        ax.bar(xs, counts, width=width * 0.92, color=EVIDENCE[ev][2], label=f"{EVIDENCE[ev][0]} ({len(sub)})")
        for x, n in zip(xs, counts):
            if n:
                ax.text(x, n + 0.8, str(n), ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(classes)
    ax.set_xticklabels([f"{k}\n{RELIABILITY[k][:28]}{'…' if len(RELIABILITY[k]) > 28 else ''}" for k in classes],
                       fontsize=6.8)
    ax.set_ylabel("Register items", fontsize=8.5, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title("What the dossiers report: register items by evidence class and source reliability class",
                 fontsize=10, fontweight="bold", color=INK, loc="left")
    fig.tight_layout()
    return fig


# ---- the section ---------------------------------------------------------------------------------------------------
def render_catalogue(root: Path, bundle: dict) -> None:
    """Contamination categories: the national register's MAP, MUSE and AEI, and what the dossiers report."""
    st.subheader("Contamination categories: the national register and the season's dossiers")
    st.caption(
        "Business level only: no device models, mechanisms or calibres. The categories are the national demining "
        "register's own (datos.gov.co `jbqm-s5tn`, the twin's open demining layer). The dossier statements are "
        "simulated by the twin and quoted by the reviewer agent."
    )

    # 1. The categories
    activity = bundle.get("activity_df", pd.DataFrame())
    reg = register_by_municipality(activity)
    cols = st.columns(len(CATEGORIES))
    for col, c in zip(cols, CATEGORIES):
        with col:
            with st.container(border=True):
                st.markdown(f"**{c['name']}**")
                st.metric("Items destroyed in the 15 municipalities", f"{int(reg[c['code']].sum()):,}")
                st.caption(c["meaning"])
    years = sorted(activity["year"].unique()) if not activity.empty else []
    st.caption(
        (f"Register rows for the 15 municipalities cover {years[0]} to {years[-1]}. " if years else "")
        + "The register counts items destroyed by municipality and month; it does not record where in a "
        "municipality they were found. " + MAR_2024
    )

    # 2. By municipality, 3. by year
    fig = plot_register_by_municipality(reg)
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Table: items destroyed by category and municipality"):
        st.dataframe(reg.set_index("municipality"), width="stretch")
    by_year = register_by_year(activity)
    if not by_year.empty:
        fig = plot_register_by_year(by_year)
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Table: items destroyed by category and year"):
            st.dataframe(by_year, width="stretch")

    # 4. What the dossiers report, and 5. the outcomes they led to
    st.markdown("#### What the season's dossiers report")
    findings = load_area_ordnance_findings(str(root))
    items = register_items(findings, bundle["areas_meta"])
    if items.empty:
        st.info("No saved evidence registers were found.")
        return
    st.caption(
        f"{len(items)} items in the reviewer's evidence registers for {items['area_id'].nunique()} areas "
        "(each statement counted once per area). Evidence classes as `rules/land_release_rules.md` defines them; "
        "reliability classes 1 to 7 from IMAS 08.10, §8.2, as that file paraphrases them."
    )
    fig = plot_evidence_classes(items)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(
        pd.DataFrame([{"Evidence class": v[0], "Meaning": v[1], "Items": int((items["evidence_class"] == k).sum())}
                      for k, v in EVIDENCE.items()]),
        width="stretch", hide_index=True,
    )

    reviews = load_saved_reviews_public(str(root))
    if not reviews.empty:
        keyed = items.merge(reviews[["week", "area_id", "visit", "outcome"]], on=["week", "area_id", "visit"],
                            how="inner")
        if not keyed.empty:
            per_review = keyed.groupby(["week", "area_id", "visit", "outcome"])["evidence_class"].apply(set)
            rows = []
            for ev in EVIDENCE_ORDER:
                has = per_review[per_review.apply(lambda s, e=ev: e in s)]
                outs = has.index.get_level_values("outcome")
                rows.append({"Register holds": EVIDENCE[ev][0], "Dossier reviews": int(len(has)),
                             **{HUMAN_OUTCOME[o]: int((outs == o).sum()) for o in REVIEW_OUTCOMES}})
            st.markdown("**The outcomes of the dossier reviews whose register holds each evidence class**")
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("A review's register usually holds more than one class, so a review can count in several rows.")

    with st.expander("Read the statements (simulated, quoted by the reviewer agent)"):
        f1, f2 = st.columns(2)
        with f1:
            ev_sel = st.multiselect("Evidence class", options=EVIDENCE_ORDER, default=["direct", "indirect"],
                                    format_func=lambda k: EVIDENCE[k][0], key="catalogue_ev_sel")
        with f2:
            mun_sel = st.multiselect("Municipality (leave empty for all)",
                                     options=sorted(items["municipality"].unique()), default=[],
                                     key="catalogue_mun_sel")
        show = items[items["evidence_class"].isin(ev_sel)] if ev_sel else items
        if mun_sel:
            show = show[show["municipality"].isin(mun_sel)]
        show = show.assign(evidence_class=show["evidence_class"].map(lambda k: EVIDENCE[k][0]),
                           reliability=show["reliability_class"].map(lambda k: f"{k}: {RELIABILITY[k]}"))
        st.dataframe(
            show[["area_id", "municipality", "week", "visit", "item", "evidence_class", "reliability", "quote"]]
            .rename(columns={"area_id": "Area", "municipality": "Municipality", "week": "Week", "visit": "Visit",
                             "item": "Item", "evidence_class": "Evidence class", "reliability": "Reliability class",
                             "quote": "Statement (simulated)"})
            .sort_values(["Week", "Area"]),
            width="stretch", hide_index=True,
        )
