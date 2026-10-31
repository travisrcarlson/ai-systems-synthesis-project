"""Real-world grounding panels for the Humanitarian Mine Action Digital Twin.

Sets the twin's season beside the sources and settings already built into `twin/` and `rules/`:
1. Terrain, land use and population (`twin.areas`, RELand's cell features)
2. The demining layer as a check on the simulated capacity, with victims shown beside it (`twin.activity`,
   `twin.people`; notebook §4.3)
3. The rainfall season (simulated from IDEAM's two Andean rainy seasons, at Travis's rates), road access tiers and
   community consultation (`twin.season`, `twin.zones`)
4. The IMAS 08.10 reliability classes and the saved rule check (R1 to R6) on the pilot's national criteria, invented
   for this project (`rules/land_release_rules.md`, `twin/rules.py`)
5. Data checksums and the layer registry (`setup.checks`, where present; `twin.layers`)

Strictly follows all build-brief guardrails:
- Read-only; no external network calls; no coordinates, axes or scale bars on any map (`ax.axis("off")`).
- Victims shown only as counts by municipality (Rule 4).
- No hidden truth labels read or displayed (Rule 6).
- Business-level terminology only (Rule 7).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection

from twin import config, rules
import grid_map
from loaders import HUMAN_OUTCOME

SURFACE = "#ffffff"
BG_WARM = "#f6f5f1"
INK = "#1c1e21"
INK_2 = "#52514e"
INK_3 = "#7c7a75"
BORDER_SUBTLE = "#d8d5ce"

LAND_USE_PALETTE = {
    "farming": ("#d97706", "Farming (Agrícola)"),
    "livestock": ("#ca8a04", "Livestock Pasture (Ganadera)"),
    "agroforestry": ("#65a30d", "Agroforestry (Agroforestal)"),
    "forest": ("#166534", "Forest land use (Forestal)"),
    "soil conservation": ("#0d9488", "Soil Conservation (Conservación)"),
    "water": ("#0284c7", "Water Body (Cuerpo de agua)"),
    "urban": ("#7c3aed", "Urban / Settlement (Zonas urbanas)"),
}

# The twin's used land: farmed, grazed or built on (`core.USED_LAND`: farming, livestock, agroforestry, urban)
PRODUCTIVE_USES = {"farming", "livestock", "agroforestry", "urban"}

# The pilot's national criteria as the season used them (`config.Criteria()`, invented for this project)
_CRITERIA = config.Criteria()
_INTRUSIVE = "technical survey" if _CRITERIA.technical_survey else "clearance"

IMAS_RELIABILITY_LADDER = [
    (
        1,
        "Class 1 — Direct physical evidence (survey team)",
        "Direct physical evidence of explosive ordnance seen and recorded by the survey team itself.",
        "Direct target (cannot be explained away)",
        "#b91c1c",
    ),
    (
        2,
        "Class 2 — Indirect physical / team-observed safe use",
        "Indirect physical evidence seen by the survey team, or safe land use observed directly by the team.",
        "Target (if hazard) / High-confidence absence",
        "#d97706",
    ),
    (
        3,
        "Class 3 — Validated historical records",
        "Historical sources and records shown to be reliable and accurate by comparison with direct evidence.",
        "Target (if hazard)",
        "#ca8a04",
    ),
    (
        4,
        "Class 4 — First-hand community informants",
        "Women, girls, boys and men of the community, land users, victims or witnesses with first-hand knowledge.",
        f"Counts toward ≥{_CRITERIA.sources} sources with ≥{_CRITERIA.min_years} yrs safe use (R2); "
        "target if reporting hazard (R1)",
        "#15803d",
    ),
    (
        5,
        "Class 5 — Second-hand community reports",
        "Sources who did not see what they report directly, but were told by first-hand sources.",
        "Weaker indirect evidence: followed up by NTS; can be explained by Class 1–4 safe use",
        "#2563eb",
    ),
    (
        6,
        "Class 6 — Unchecked historical / national database record",
        "Records whose location and reliability have not been checked on the ground (e.g., desk-study past accident E1).",
        "Weaker indirect evidence, followed up by NTS. A desk-study record of an accident inside the area blocks "
        f"cancellation (R6) and, after the first visit, is a target for {_INTRUSIVE} (R3)",
        "#7c3aed",
    ),
    (
        7,
        "Class 7 — Unattributed rumours",
        "People who were told about a hazard by others who cannot be identified as first-hand sources.",
        "Weaker indirect evidence: becomes a target only if still unexplained after "
        f"{_CRITERIA.follow_up_visits} NTS visits",
        "#64748b",
    ),
]


def get_enriched_areas(bundle: dict) -> pd.DataFrame:
    """Return the full 7,280-area table from `setup.twin.areas` with all terrain and demographic columns."""
    setup = bundle.get("setup")
    if setup is not None and hasattr(setup, "twin") and hasattr(setup.twin, "areas"):
        df = setup.twin.areas.drop(columns=["cell_list"], errors="ignore").copy()
        return df
    return bundle["areas_meta"].copy()


def get_enriched_zones(bundle: dict) -> pd.DataFrame:
    """Return the full 292-zone table from `setup.twin.zones`."""
    setup = bundle.get("setup")
    if setup is not None and hasattr(setup, "twin") and hasattr(setup.twin, "zones"):
        return setup.twin.zones.copy()
    return bundle["zones_meta"].copy()


def get_people_df(bundle: dict) -> pd.DataFrame:
    """Return `setup.twin.people` (victims counted by municipality, year, month, condition, and outcome)."""
    setup = bundle.get("setup")
    if setup is not None and hasattr(setup, "twin") and hasattr(setup.twin, "people"):
        return setup.twin.people.copy()
    return pd.DataFrame()


def get_checks_df(bundle: dict) -> pd.DataFrame:
    """Return the SHA-256 dataset verification table from `setup.checks` (empty where the setup holds none)."""
    setup = bundle.get("setup")
    if setup is not None and hasattr(setup, "checks") and setup.checks is not None:
        return setup.checks.reset_index().copy()
    return pd.DataFrame()


# =====================================================================================================================
# 1. TERRAIN, LAND USE & LAND RELEASED
# =====================================================================================================================
def plot_terrain_and_land_use_grounding(
    bundle: dict,
    log_df: pd.DataFrame,
    week: int = 52,
) -> plt.Figure:
    """Render a 3-panel figure of the 15 pilot municipalities: RELand's land use, road access, and the land
    released by week `week` (land mainly farmed, grazed or built on among it).
    No geographic coordinates, axes or scale bars are used.
    """
    areas_df = get_enriched_areas(bundle)
    area_status = grid_map.area_status_at_week(bundle, week)
    canvas_df = grid_map._build_regional_canvas_index(bundle["cells_meta"])

    released_mask = area_status.isin(["cleared", "reduced", "cancelled"])
    areas_df = areas_df.copy()
    areas_df["status"] = area_status.reindex(areas_df.index).fillna("unknown")
    areas_df["is_released"] = released_mask.reindex(areas_df.index).fillna(False)

    fig = plt.figure(figsize=(16.2, 7.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.05, 0.95], hspace=0.36, wspace=0.22)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_lu = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 1])

    for ax in (ax_map, ax_lu, ax_acc):
        ax.set_facecolor(SURFACE)

    # ---- Left Panel: Schematic Regional Land-Use Map + Released Areas Outlined ----
    ax_map.set_xlim(-2, grid_map.REGIONAL_CANVAS_W + 2)
    ax_map.set_ylim(grid_map.REGIONAL_CANVAS_H + 2, -2)
    ax_map.axis("off")
    grid_map._draw_regional_inset_decorations(ax_map)

    pr = canvas_df["plot_r"].to_numpy(dtype=int)
    pc = canvas_df["plot_c"].to_numpy(dtype=int)

    muni_codes = list(config.PILOT.keys())
    muni_to_int = {c: idx for idx, c in enumerate(muni_codes)}
    zone_ids = sorted(canvas_df["zone_id"].unique().tolist())
    zone_to_int = {z: idx for idx, z in enumerate(zone_ids)}

    muni_grid = np.full((grid_map.REGIONAL_CANVAS_H, grid_map.REGIONAL_CANVAS_W), -1, dtype=int)
    zone_grid = np.full((grid_map.REGIONAL_CANVAS_H, grid_map.REGIONAL_CANVAS_W), -1, dtype=int)
    muni_grid[pr, pc] = canvas_df["municipality_code"].map(muni_to_int).to_numpy(dtype=int)
    zone_grid[pr, pc] = canvas_df["zone_id"].map(zone_to_int).to_numpy(dtype=int)

    outer_segs = grid_map._extract_grid_boundary_segments(muni_grid, mode="outer")
    muni_segs = grid_map._extract_grid_boundary_segments(muni_grid, mode="inner")
    zone_segs = grid_map._extract_grid_boundary_segments(zone_grid, mode="inner")

    rgb_img = np.ones((grid_map.REGIONAL_CANVAS_H, grid_map.REGIONAL_CANVAS_W, 4), dtype=float)
    rgb_img[:, :, 3] = 0.0

    lu_series = canvas_df["area_id"].map(areas_df["main_land_use"]).fillna("forest")
    st_series = canvas_df["area_id"].map(area_status).fillna("unknown")

    for r_i, c_i, lu_val, st_val in zip(pr, pc, lu_series, st_series):
        hex_col = LAND_USE_PALETTE.get(str(lu_val), ("#64748b", ""))[0]
        rgb = mpatches.colors.to_rgb(hex_col)
        if st_val in ("cleared", "reduced", "cancelled"):
            # Highlight released land with full saturation
            rgb_img[r_i, c_i] = (rgb[0], rgb[1], rgb[2], 0.96)
        else:
            # Slightly muted base for unreleased land
            rgb_img[r_i, c_i] = (rgb[0] * 0.85 + 0.15, rgb[1] * 0.85 + 0.15, rgb[2] * 0.85 + 0.15, 0.72)

    ax_map.imshow(
        rgb_img,
        interpolation="nearest",
        aspect="equal",
        extent=(-0.5, grid_map.REGIONAL_CANVAS_W - 0.5, grid_map.REGIONAL_CANVAS_H - 0.5, -0.5),
        zorder=2,
    )

    # Outline released areas on top of the land-use map
    rel_grid = np.full((grid_map.REGIONAL_CANVAS_H, grid_map.REGIONAL_CANVAS_W), -1, dtype=int)
    is_rel_cell = st_series.isin(["cleared", "reduced", "cancelled"]).to_numpy()
    rel_grid[pr[is_rel_cell], pc[is_rel_cell]] = 1
    rel_segs = grid_map._extract_grid_boundary_segments(rel_grid, mode="outer")

    if zone_segs:
        ax_map.add_collection(LineCollection(zone_segs, colors="#ffffff", linewidths=0.35, alpha=0.55, zorder=3))
    if muni_segs:
        ax_map.add_collection(LineCollection(muni_segs, colors="#1e293b", linewidths=1.1, alpha=0.85, zorder=4))
    if outer_segs:
        ax_map.add_collection(LineCollection(outer_segs, colors="#0f172a", linewidths=1.6, alpha=0.95, zorder=5))
    if rel_segs:
        ax_map.add_collection(LineCollection(rel_segs, colors="#0f172a", linewidths=1.25, alpha=0.95, zorder=6))

    # Municipality labels
    for code in muni_codes:
        sub = canvas_df[canvas_df["municipality_code"] == code]
        if sub.empty:
            continue
        cy = float(sub["plot_r"].mean())
        cx = float(sub["plot_c"].mean())
        ax_map.text(
            cx,
            cy,
            config.NAME_OF[code],
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.86, lw=0.6),
            zorder=8,
        )

    ax_map.set_title(
        f"Main Land Use of the {len(areas_df):,} Areas, Released Areas Outlined (Week {week})\n"
        "RELand's land-use classes (no coordinates, axes or scale)",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
        pad=8,
    )

    legend_handles = [
        mpatches.Patch(facecolor=col, edgecolor="#334155", linewidth=0.6, label=lbl)
        for _, (col, lbl) in LAND_USE_PALETTE.items()
    ]
    legend_handles.append(
        mpatches.Patch(facecolor="#ffffff", edgecolor="#0f172a", linewidth=1.4, label="Released areas (outlined)")
    )
    ax_map.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=7.2,
        ncol=2,
    )

    # ---- Top-Right Panel: Land-Use Composition (km² Released vs. Remaining) ----
    lu_grp = (
        areas_df.groupby("main_land_use")
        .agg(
            total_km2=("km2", "sum"),
            released_km2=("km2", lambda s: s[areas_df.loc[s.index, "is_released"]].sum()),
            pop_total=("population", "sum"),
        )
        .reindex(list(LAND_USE_PALETTE.keys()), fill_value=0.0)
    )
    lu_grp["remaining_km2"] = (lu_grp["total_km2"] - lu_grp["released_km2"]).clip(lower=0.0)
    lu_grp = lu_grp.sort_values("total_km2", ascending=True)

    y_lu = np.arange(len(lu_grp))
    labels_lu = [LAND_USE_PALETTE.get(k, ("#64748b", k))[1] for k in lu_grp.index]
    colors_lu = [LAND_USE_PALETTE.get(k, ("#64748b", k))[0] for k in lu_grp.index]

    ax_lu.barh(y_lu, lu_grp["total_km2"], color=colors_lu, alpha=0.30, height=0.66, label="Total Area (km²)")
    ax_lu.barh(
        y_lu,
        lu_grp["released_km2"],
        color=colors_lu,
        edgecolor="#0f172a",
        linewidth=0.7,
        height=0.66,
        label=f"Released by Week {week} (km²)",
    )
    for idx, (_, row) in enumerate(lu_grp.iterrows()):
        tot_v = float(row["total_km2"])
        rel_v = float(row["released_km2"])
        pct = 100.0 * rel_v / max(tot_v, 1e-6)
        ax_lu.text(
            tot_v + 25,
            idx,
            f"{rel_v:,.0f} / {tot_v:,.0f} km² ({pct:.0f}% released)",
            va="center",
            fontsize=7.6,
            color=INK,
            fontweight="bold",
        )

    ax_lu.set_yticks(y_lu)
    ax_lu.set_yticklabels(labels_lu, fontsize=8.0, color=INK)
    ax_lu.set_xlim(0, float(lu_grp["total_km2"].max()) * 1.34)
    ax_lu.set_xlabel("Area (km²)", fontsize=8.2, color=INK_2)
    ax_lu.set_title(
        f"Land Use: Total km² and km² Released by Week {week}",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
    )
    ax_lu.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax_lu.spines["top"].set_visible(False)
    ax_lu.spines["right"].set_visible(False)
    ax_lu.grid(axis="x", linestyle=":", alpha=0.45)

    # ---- Bottom-Right Panel: Road Access Tiers & Community Consultation Profile ----
    zones_df = get_enriched_zones(bundle).copy()
    zone_st = grid_map.zone_state_at_week(bundle, week)
    zones_df["is_free"] = zone_st.reindex(zones_df.index).fillna("not surveyed") == "free"

    acc_order = ["under 2 km from a road", "2 to 5 km from a road", "more than 5 km from a road"]
    acc_grp = (
        zones_df.groupby("access")
        .agg(
            total_zones=("areas", "size"),
            free_zones=("is_free", "sum"),
            consult_zones=("consultation", "sum"),
            total_pop=("population", "sum"),
        )
        .reindex(acc_order, fill_value=0)
    )
    consult_weeks = int(bundle["setup"].twin.season.consultation_weeks)
    y_acc = np.arange(len(acc_grp))
    ax_acc.barh(
        y_acc,
        acc_grp["total_zones"],
        color="#cbd5e1",
        edgecolor="#64748b",
        height=0.58,
        label="Total Operational Zones",
    )
    ax_acc.barh(
        y_acc,
        acc_grp["free_zones"],
        color="#0d6e52",
        height=0.58,
        label=f"Zones Declared Free (Week {week})",
    )
    for idx, (acc_lbl, row) in enumerate(acc_grp.iterrows()):
        tz = int(row["total_zones"])
        fz = int(row["free_zones"])
        cz = int(row["consult_zones"])
        ax_acc.text(
            tz + 2.5,
            idx,
            f"{fz}/{tz} zones free · {cz} need consultation",
            va="center",
            fontsize=7.7,
            color=INK,
            fontweight="bold",
        )

    ax_acc.set_yticks(y_acc)
    ax_acc.set_yticklabels([a.capitalize() for a in acc_grp.index], fontsize=8.2, color=INK)
    ax_acc.set_xlim(0, float(acc_grp["total_zones"].max()) * 1.55)
    ax_acc.set_xlabel("Number of Operational Zones", fontsize=8.2, color=INK_2)
    ax_acc.set_title(
        f"Operational Zones by Road Access & Community Consultation ({consult_weeks}-Week Lead Time)",
        fontsize=9.8,
        fontweight="bold",
        color=INK,
    )
    ax_acc.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)
    ax_acc.grid(axis="x", linestyle=":", alpha=0.45)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.07)
    return fig


# =====================================================================================================================
# 2. PAST OUTPUT AND VICTIMS (P1 VICTIMS & P2 DEMINING), AND THE NOTEBOOK'S CHECK OF THE SIMULATED CAPACITY (§4.3)
# =====================================================================================================================
# Antioquia's known hazardous areas average about 4,700 m²: Claude's calculation from Mine Action Review's (2025)
# table, 300,079 m² in 64 areas. The notebook (§4.3) multiplies the season's clearances by this size.
MAR_ANTIOQUIA_M2, MAR_ANTIOQUIA_AREAS = 300_079, 64
HAZARDOUS_AREA_M2 = MAR_ANTIOQUIA_M2 / MAR_ANTIOQUIA_AREAS


def capacity_check(bundle: dict) -> dict:
    """The notebook's check of the simulated capacity against the pilot's past clearance (§4.3): a clearance team's
    expected working days in the season (after the days rain and wind take), the clearances the clearance teams can
    finish at the planned days per area, their m² at the average hazardous area's size, and the pilot's clearance by
    year (P2's data, the years with clearance only)."""
    s_cfg = bundle["setup"].twin.season
    days = sum(n * s_cfg.days_per_week * (1 - s_cfg.day_loss(m)) for m, n in enumerate(s_cfg.month_weeks, start=1))
    clearances = s_cfg.clearance_teams * days / s_cfg.task_days("CLEARANCE")
    act_df = grid_map.get_activity_df(bundle)
    if act_df.empty:
        by_year = pd.Series(dtype=float)
    else:
        pilot_a = act_df[act_df["municipality_code"].isin(list(config.PILOT))]
        by_year = pilot_a.groupby("year")["area_cleared_m2"].sum()
        by_year = by_year[by_year > 0]
    return {
        "teams": int(s_cfg.clearance_teams),
        "days": float(days),
        "days_per_area": float(s_cfg.task_days("CLEARANCE")),
        "clearances": float(clearances),
        "capacity_m2": float(clearances * HAZARDOUS_AREA_M2),
        "by_year": by_year,
    }


def plot_historical_calibration_panels(
    bundle: dict,
    log_df: pd.DataFrame,
    week: int = 52,
) -> plt.Figure:
    """Plot the past victim counts and humanitarian demining output in the 15 pilot municipalities (`twin.people`
    and `twin.activity`, Presidencia de la República / datos.gov.co), and the notebook's check of the simulated
    clearance capacity against the pilot's past clearance (§4.3). The check is the season's expected capacity, the
    same for every week.
    """
    act_df = grid_map.get_activity_df(bundle)
    people_df = get_people_df(bundle)
    muni_codes = list(config.PILOT.keys())

    fig, (ax_vict, ax_dem, ax_comp) = plt.subplots(1, 3, figsize=(16.2, 4.9), facecolor=SURFACE)
    for ax in (ax_vict, ax_dem, ax_comp):
        ax.set_facecolor(SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---- Panel 1: Historical Annual Victim Counts in the 15 Pilot Municipalities (Civilian vs. Security Forces) ----
    if not people_df.empty:
        pilot_p = people_df[people_df["municipality_code"].isin(muni_codes)]
        yr_cond = (
            pilot_p.groupby(["year", "condition"])["victims"]
            .sum()
            .unstack(fill_value=0)
            .sort_index()
        )
        years = yr_cond.index.to_numpy(dtype=int)
        civ = yr_cond.get("Civilian", pd.Series(0, index=yr_cond.index)).to_numpy(dtype=float)
        sec = yr_cond.get("Security forces", pd.Series(0, index=yr_cond.index)).to_numpy(dtype=float)

        ax_vict.bar(years, civ, color="#dc2626", alpha=0.86, label="Civilian victims")
        ax_vict.bar(years, sec, bottom=civ, color="#475569", alpha=0.86, label="Security forces victims")
        ax_vict.set_title(
            "Historical Victim Counts in the 15 Pilot Municipalities\n"
            "(datos.gov.co yhxn-eqqw, municipal counts by year)",
            fontsize=9.4,
            fontweight="bold",
            color=INK,
        )
        ax_vict.set_xlabel("Year", fontsize=8.2, color=INK_2)
        ax_vict.set_ylabel("Recorded Victims (count)", fontsize=8.2, color=INK_2)
        ax_vict.legend(loc="upper right", fontsize=7.6, frameon=True)
        ax_vict.grid(axis="y", linestyle=":", alpha=0.45)

    # ---- Panel 2: Historical Annual Demining Output in the 15 Pilot Municipalities ----
    if not act_df.empty:
        pilot_a = act_df[act_df["municipality_code"].isin(muni_codes)]
        yr_act = (
            pilot_a.groupby("year")[
                ["non_technical_surveys", "clearance_operations", "items_total", "area_cleared_m2"]
            ]
            .sum()
            .sort_index()
        )
        yrs_a = yr_act.index.to_numpy(dtype=int)
        ax_dem.plot(
            yrs_a,
            yr_act["non_technical_surveys"],
            marker="o",
            linewidth=2.0,
            color="#2563eb",
            label="Non-Technical Surveys (estudiosnotecnicos)",
        )
        ax_dem.plot(
            yrs_a,
            yr_act["clearance_operations"],
            marker="s",
            linewidth=2.0,
            color="#0d6e52",
            label="Clearance Operations (operacionesdespeje)",
        )
        ax_dem.plot(
            yrs_a,
            yr_act["items_total"],
            marker="^",
            linewidth=1.8,
            linestyle="--",
            color="#b91c1c",
            label="Items Destroyed (MAP, MUSE, AEI)",
        )
        ax_dem.set_title(
            "Historical Humanitarian Demining in the 15 Municipalities\n"
            "(datos.gov.co jbqm-s5tn, annual totals)",
            fontsize=9.4,
            fontweight="bold",
            color=INK,
        )
        ax_dem.set_xlabel("Year", fontsize=8.2, color=INK_2)
        ax_dem.set_ylabel("Annual Count", fontsize=8.2, color=INK_2)
        ax_dem.legend(loc="upper left", fontsize=7.4, frameon=True)
        ax_dem.grid(axis="y", linestyle=":", alpha=0.45)

    # ---- Panel 3: The notebook's capacity check (§4.3): the pilot's clearance by year vs. the teams' season ----
    cap = capacity_check(bundle)
    by_year = cap["by_year"]
    if not by_year.empty:
        ax_comp.bar(
            by_year.index.to_numpy(dtype=int),
            by_year.to_numpy(dtype=float) / 1000,
            color="#94a3b8",
            edgecolor="#475569",
            width=0.8,
            label="Cleared in the year (P2's data)",
        )
    ax_comp.axhline(
        cap["capacity_m2"] / 1000,
        color="#1f4e78",
        linewidth=2.0,
        label=(
            f"The simulated teams' capacity in a season: about {cap['clearances']:.0f} clearances "
            f"× {HAZARDOUS_AREA_M2:,.0f} m²"
        ),
    )
    ax_comp.set_xlabel("Year", fontsize=8.2, color=INK_2)
    ax_comp.set_ylabel("Thousand m² Cleared", fontsize=8.2, color=INK_2)
    top = max(float(by_year.max()) if not by_year.empty else 0.0, cap["capacity_m2"]) / 1000
    ax_comp.set_ylim(0, top * 1.32)
    ax_comp.set_title(
        "Capacity Check (Notebook §4.3)\n"
        "Clearance in the 15 Municipalities by Year (P2's data)",
        fontsize=9.4,
        fontweight="bold",
        color=INK,
    )
    ax_comp.legend(loc="upper left", fontsize=7.2, frameon=True)
    ax_comp.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


# =====================================================================================================================
# 3. THE RAINFALL SEASON (SIMULATED LAYER) AND THE SEASON'S SETTINGS
# =====================================================================================================================
def _month_list(months) -> str:
    """Month numbers as words: 'March, April, October and November'."""
    names = [config.MONTHS[int(m) - 1] for m in months]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def plot_ideam_and_logistics_strip(bundle: dict) -> plt.Figure:
    """Plot the rainfall season (a simulated layer: IDEAM's two Andean rainy seasons, at Travis's rates) as the
    share of working days lost each month, beside the team-days the simulated field worked and lost to rain or wind.
    """
    setup = bundle["setup"]
    season_cfg = setup.twin.season
    cal = setup.twin.calendar
    week_days = bundle.get("week_days", {})

    months = list(range(1, 13))
    month_names = [m[:3] for m in config.MONTHS]
    expected_loss_pct = [100.0 * season_cfg.day_loss(m) for m in months]

    worked_by_month = []
    lost_by_month = []
    for m in months:
        wks = cal.weeks_of[m]
        w_sum = 0
        l_sum = 0
        for w in wks:
            wd = week_days.get(int(w), {})
            if isinstance(wd, dict):
                if "worked" in wd or "weather" in wd:
                    w_sum += int(wd.get("worked", 0) or 0)
                    l_sum += int(wd.get("weather", 0) or 0)
                else:
                    for val in wd.values():
                        if isinstance(val, (tuple, list)) and len(val) >= 2:
                            w_sum += int(val[0])
                            l_sum += int(val[1])
        worked_by_month.append(w_sum)
        lost_by_month.append(l_sum)

    fig, (ax_rain, ax_real) = plt.subplots(1, 2, figsize=(15.6, 3.9), facecolor=SURFACE)
    for ax in (ax_rain, ax_real):
        ax.set_facecolor(SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = np.arange(12)
    bar_colors = ["#2563eb" if m in season_cfg.rainy_months else "#93c5fd" for m in months]
    ax_rain.bar(x, expected_loss_pct, color=bar_colors, edgecolor="#1e3a8a", width=0.62)
    for i, (m, pct) in enumerate(zip(months, expected_loss_pct)):
        wks_n = len(cal.weeks_of[m])
        exp_days = wks_n * season_cfg.days_per_week * (1.0 - season_cfg.day_loss(m))
        ax_rain.text(
            i,
            pct + 1.8,
            f"{pct:.0f}%\n({exp_days:.1f}d)",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold" if m in season_cfg.rainy_months else "normal",
            color="#1e3a8a" if m in season_cfg.rainy_months else INK_2,
        )
    ax_rain.set_xticks(x)
    ax_rain.set_xticklabels(month_names, fontsize=8.2, color=INK)
    ax_rain.set_ylim(0, 68)
    ax_rain.set_ylabel("% Working Days Lost to Rain/Wind", fontsize=8.2, color=INK_2)
    ax_rain.set_title(
        "Rainfall Season (Simulated Layer): IDEAM's Two Andean Rainy Seasons, at Travis's Rates\n"
        f"Dark blue: {_month_list(season_cfg.rainy_months)}, {season_cfg.rainy_day_loss:.0%} of working days lost; "
        f"other months {season_cfg.other_day_loss:.0%}\n"
        "(in brackets: a team's expected working days in the month)",
        fontsize=9.3,
        fontweight="bold",
        color=INK,
    )
    ax_rain.grid(axis="y", linestyle=":", alpha=0.45)

    ax_real.bar(x, worked_by_month, color="#0d6e52", width=0.62, label="Team-days worked on tasks")
    ax_real.bar(
        x,
        lost_by_month,
        bottom=worked_by_month,
        color="#d97706",
        alpha=0.85,
        width=0.62,
        label="Team-days lost to heavy rain / high wind",
    )
    ax_real.set_xticks(x)
    ax_real.set_xticklabels(month_names, fontsize=8.2, color=INK)
    ax_real.set_ylabel("Total Team-Days Across Active Teams", fontsize=8.2, color=INK_2)
    ax_real.set_title(
        f"Team-Days Worked and Lost by Month, Across the {season_cfg.weeks}-Week Season\n"
        "Team-days the simulated field lost to rain or wind, drawn at Travis's rates",
        fontsize=9.3,
        fontweight="bold",
        color=INK,
    )
    ax_real.legend(loc="upper right", fontsize=7.8, frameon=True)
    ax_real.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


# =====================================================================================================================
# 4. THE IMAS 08.10 RELIABILITY CLASSES AND THE SAVED RULE CHECK (`twin/rules.py`, R1 TO R6)
# =====================================================================================================================
def saved_rule_check_rows(record: dict) -> list[dict]:
    """The rule check saved with a dossier review (`rule_check` in its `*_result.json`), for the outcome the reviewer
    submitted: one row per check, with its rule, whether it passed and its detail. Nothing is re-run: the saved
    records hold the reviewer's register but not the evidence log the rule check reads (who each item comes from, and
    the years of safe use it reports), so the other outcomes are not tested here.
    """
    saved_rc = record.get("rule_check") or {}
    return [
        {
            "Rule": str(c.get("rule", "")),
            "Result": "PASS (rule check)" if c.get("passed") else "VETO (rule check)",
            "Detail": str(c.get("detail", "")),
        }
        for c in saved_rc.get("checks", [])
    ]


def rule_treatment(evidence_class: str, reliability_class: int, criteria: config.Criteria) -> str:
    """How the rule check (`twin/rules.py`) treats an item of the register, by its evidence class and the reliability
    class of its source. Whether an item is the desk study's record (R6) is read from the twin's record of the item,
    which the saved register does not hold, so that case is stated as a condition.
    """
    check = "technical survey" if criteria.technical_survey else "clearance"
    rule = "R4" if criteria.technical_survey else "R3"
    if evidence_class == "direct":
        return (f"Direct evidence: blocks cancellation, and evidence of absence never explains it away (R1); a target "
                f"for {check} ({rule}).")
    if evidence_class == "indirect" and reliability_class <= 4:
        return (f"Indirect evidence the survey team saw or a first-hand source reported: blocks cancellation (R1); a "
                f"target for {check} ({rule}).")
    if evidence_class == "indirect":
        text = (f"Weaker indirect evidence: blocks cancellation unless evidence of absence of class 1 to 4 explains it "
                f"(R1); followed up by non-technical survey, and a target for {check} only if still unexplained after "
                f"{criteria.follow_up_visits} visits ({rule}).")
        if reliability_class == 6 and criteria.recorded_accident_needs_ts:
            text += (f" If it is the desk study's record of an accident inside the area, it blocks cancellation in any "
                     f"case (R6) and is a target for {check} after the first visit ({rule}).")
        return text
    if evidence_class == "absence" and reliability_class <= 4:
        return (f"Evidence of absence of class 1 to 4: explains the weaker indirect evidence (class 5 to 7) it names "
                f"(R1); counts toward the {criteria.sources} sources the criteria need if it reports at least "
                f"{criteria.min_years} years of safe use, one person counting once (R2).")
    if evidence_class == "absence":
        return ("Evidence of absence from a source of class 5 to 7: explains no item (R1) and does not count toward "
                "the criteria (R2).")
    return "Not evidence either way: the rule check only needs it in the register (R5)."


def render_imas_and_rule_check_inspector(record: dict, criteria: config.Criteria | None = None) -> None:
    """Render a dossier review's saved rule check (R1 to R6, for the outcome the reviewer submitted) and its
    evidence register, each item with its IMAS 08.10 (§8.2) reliability class and how the rule check treats it.
    """
    criteria = criteria or config.Criteria()
    reg = record.get("register", {})
    saved_rc = record.get("rule_check") or {}
    outcome = str(saved_rc.get("outcome") or record.get("outcome", ""))

    st.markdown("#### ⚖️ Rule check (R1 to R6) on the pilot's national criteria")
    st.caption(
        "The rule check saved with this review (`rule_check` in its result record; the code is `twin/rules.py`), for "
        "the outcome the reviewer submitted. The saved records hold the reviewer's register but not the evidence log "
        "the rule check reads, so the other outcomes are not tested here. The pilot's national criteria are invented "
        "for this project."
    )
    if saved_rc:
        verdict = "PASS (rule check)" if saved_rc.get("result") == "pass" else "VETO (rule check)"
        failed = rules.failed_rules(saved_rc) if saved_rc.get("checks") else ""
        st.markdown(
            f"**Submitted outcome:** {HUMAN_OUTCOME.get(outcome, outcome)} (`{outcome}`) · "
            f"**Saved rule check:** {verdict}" + (f" (failed: {failed})" if failed else "")
        )
        st.dataframe(pd.DataFrame(saved_rule_check_rows(record)), width="stretch", hide_index=True)
    else:
        st.info("No rule check is saved with this review.")

    if reg:
        rel_lookup = {cls_num: title for cls_num, title, _, _, _ in IMAS_RELIABILITY_LADDER}
        reg_rows = []
        for k, v in reg.items():
            r_cls = int(v.get("reliability_class", 7) or 7)
            e_cls = str(v.get("evidence_class", ""))
            reg_rows.append(
                {
                    "Item ID": v.get("item_id", k),
                    "Evidence Class": e_cls,
                    "IMAS 08.10 Reliability Class": rel_lookup.get(r_cls, f"Class {r_cls}"),
                    "Explains Item(s)": ", ".join(v.get("explains", [])) or "—",
                    "How the Rule Check Treats It (R1 to R6)": rule_treatment(e_cls, r_cls, criteria),
                    "Dossier Quote": v.get("quote", ""),
                }
            )
        st.markdown("**The reviewer's evidence register, with IMAS 08.10 (§8.2) reliability classes (1 to 7):**")
        st.dataframe(pd.DataFrame(reg_rows), width="stretch", hide_index=True)


# =====================================================================================================================
# 5. MAIN DASHBOARD SECTION: REAL-WORLD GROUNDING, TERRAIN, PAST OUTPUT & IMAS-BASED RULES
# =====================================================================================================================
def render_grounding_and_calibration_section(
    root: Path,
    bundle: dict,
    log_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    week: int,
) -> None:
    """Render the Real-World Grounding section on the main dashboard: terrain and land use, past output and victims
    (with the notebook's capacity check), the rainfall season and the season's settings, the reliability classes and
    the saved rule check, and the data checksums and layer registry.
    """
    s_cfg = bundle["setup"].twin.season
    st.subheader("🌍 Real-World Grounding: Terrain, Land Use, Past Output & IMAS-Based Rules")
    st.caption(
        f"Sets the twin's {s_cfg.weeks}-week season beside its sources and settings: "
        "**(1) RELand's terrain, land-use and population features**, "
        "**(2) the past demining output and victims published by Presidencia de la República (`datos.gov.co`)**, "
        "**(3) the rainfall season (IDEAM's two Andean rainy seasons, at Travis's rates) and the season's settings**, "
        "**(4) the IMAS 08.10 reliability classes and the rule check on the pilot's national criteria**, and "
        "**(5) SHA-256 dataset verification and the layer registry**."
    )

    areas_df = get_enriched_areas(bundle)
    zones_df = get_enriched_zones(bundle)
    area_status = grid_map.area_status_at_week(bundle, week)
    is_released = area_status.isin(["cleared", "reduced", "cancelled"])

    areas_w = areas_df.copy()
    areas_w["status"] = area_status.reindex(areas_w.index).fillna("unknown")
    areas_w["is_released"] = is_released.reindex(areas_w.index).fillna(False)

    prod_mask = areas_w["main_land_use"].isin(PRODUCTIVE_USES)
    total_prod_km2 = float(areas_w.loc[prod_mask, "km2"].sum())
    freed_prod_km2 = float(areas_w.loc[prod_mask & areas_w["is_released"], "km2"].sum())

    total_pop = int(round(float(areas_w["population"].sum())))
    freed_pop = int(round(float(areas_w.loc[areas_w["is_released"], "population"].sum())))

    remote_mask = areas_w["road_km"] >= 5.0
    remote_total = int(remote_mask.sum())
    remote_freed = int((remote_mask & areas_w["is_released"]).sum())

    forest_mask = areas_w["forest_share"] >= 0.50
    forest_total = int(forest_mask.sum())
    forest_freed = int((forest_mask & areas_w["is_released"]).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Used Land Released",
        f"{freed_prod_km2:,.1f} km²",
        f"of {total_prod_km2:,.1f} km²",
    )
    k2.metric(
        "Population Score Released",
        f"{freed_pop:,}",
        f"of {total_pop:,} (not residents)",
    )
    k3.metric(
        "Areas ≥5 km from a Road",
        f"{remote_freed:,} / {remote_total:,}",
        "released",
    )
    k4.metric(
        "Areas ≥50% Forest Land Use",
        f"{forest_freed:,} / {forest_total:,}",
        "released",
    )
    k5.metric(
        "Zones Needing Consultation",
        f"{int(zones_df['consultation'].sum())} / {len(zones_df)}",
        f"{s_cfg.consultation_weeks}-week lead time",
    )
    st.caption(
        f"At the end of week {week}. **Used land:** areas whose main land use is farming, livestock, agroforestry or "
        "urban (the twin's land farmed, grazed or built on). **Population score:** RELand's 2012 population estimates "
        "per cell, summed over the areas; a relative score (the simulated field uses it in its measure of how much an "
        "area is used), not a count of residents. **Distance to a road:** averaged over each area's cells. **Forest "
        "land use:** the share of an area's cells in the forest land-use class. **Consultation:** a simulated layer; "
        "the communities of these zones are consulted before any tasking."
    )

    g_tab1, g_tab2, g_tab3, g_tab4, g_tab5 = st.tabs(
        [
            "🌾 1. Terrain, Land Use & Land Released",
            "📊 2. Past Output and Victims (`datos.gov.co`, P1 and P2)",
            "🌧️ 3. Rainfall Season and Season Settings",
            "⚖️ 4. Reliability Classes and the Saved Rule Check",
            "🔐 5. SHA-256 Data Provenance & Layer Registry",
        ]
    )

    with g_tab1:
        st.markdown(f"##### RELand Terrain and Land-Use Features, and Land Released by Week {week}, by Municipality")
        st.caption(
            "In the simulated field (`twin.config.Assumptions`), land use shapes what a non-technical survey visit "
            "returns: signs of safe use are likelier on farmed or grazed land near settlements, and rarer in forest or "
            f"remote land (`land_use_effect = {config.Assumptions().land_use_effect:g}` log-odds per SD of the "
            "land-use score, this project's setting). Each municipality's areas are grouped into zones of about "
            f"{s_cfg.zone_areas} (Travis's setting, question 29) by k-means on their position in km and their access, "
            f"log(1 + km to the nearest road), weighted {s_cfg.zone_access_weight:g} (this project's setting)."
        )
        fig_lu = plot_terrain_and_land_use_grounding(bundle, log_df=log_df, week=week)
        st.pyplot(fig_lu)
        plt.close(fig_lu)

        # Municipality terrain, land-use and land-released table
        muni_rows = []
        for code in config.PILOT:
            sub = areas_w[areas_w["municipality_code"] == code]
            sub_rel = sub[sub["is_released"]]
            sub_prod = sub[sub["main_land_use"].isin(PRODUCTIVE_USES)]
            sub_prod_rel = sub_prod[sub_prod["is_released"]]
            muni_rows.append(
                {
                    "Municipality": config.NAME_OF[code],
                    "DANE Code": code,
                    "Total Area (km²)": round(float(sub["km2"].sum()), 1),
                    "Dominant Land Use": str(sub["main_land_use"].mode().iloc[0]).capitalize() if not sub.empty else "—",
                    "Farmed, Grazed or Built-On Share": f"{100.0 * float(sub['used_share'].mean()):.1f}%",
                    "Forest Land-Use Share": f"{100.0 * float(sub['forest_share'].mean()):.1f}%",
                    "Mean Distance to Road (km)": round(float(sub["road_km"].mean()), 2),
                    "Mean Distance to Settlement (km)": round(float(sub["settlement_km"].mean()), 2),
                    f"Used Land Released / Total (km², Wk {week})": f"{float(sub_prod_rel['km2'].sum()):.1f} / {float(sub_prod['km2'].sum()):.1f}",
                    f"Population Score Released / All (Wk {week}; Not Residents)": f"{int(round(float(sub_rel['population'].sum()))):,} / {int(round(float(sub['population'].sum()))):,}",
                }
            )
        st.dataframe(pd.DataFrame(muni_rows), width="stretch", hide_index=True)

    with g_tab2:
        st.markdown(
            f"##### Past Demining Output and Victims in the {len(config.PILOT)} Municipalities, Beside the Simulated Season"
        )
        st.caption(
            "Sets the twin's season beside the records published by **Presidencia de la República de Colombia "
            f"(`datos.gov.co` `yhxn-eqqw` and `jbqm-s5tn`)** for the {len(config.PILOT)} pilot municipalities. As "
            "registered in `twin/layers.py`, the `demining` layer is *'a check of the simulated capacity against the "
            "pilot's past output'*: the right-hand chart is the notebook's check (§4.3)."
        )
        fig_cal = plot_historical_calibration_panels(bundle, log_df=log_df, week=week)
        st.pyplot(fig_cal)
        plt.close(fig_cal)

        cap = capacity_check(bundle)
        by_year = cap["by_year"]
        if not by_year.empty:
            st.caption(
                f"**The capacity check (notebook §4.3):** a clearance team can expect {cap['days']:.0f} working days "
                f"in the season; {cap['teams']} clearance teams at {cap['days_per_area']:.0f} days per area finish "
                f"about {cap['clearances']:.0f} clearances, or {cap['capacity_m2']:,.0f} m² at "
                f"{HAZARDOUS_AREA_M2:,.0f} m² per hazardous area (Antioquia's known hazardous areas: Claude's "
                f"calculation from Mine Action Review's (2025) table, {MAR_ANTIOQUIA_M2:,} m² in "
                f"{MAR_ANTIOQUIA_AREAS} areas), against the pilot's median of {by_year.median():,.0f} m² cleared a "
                f"year (middle half {by_year.quantile(0.25):,.0f} to {by_year.quantile(0.75):,.0f}; {len(by_year)} "
                f"years with clearance, {int(by_year.index.min())}–{int(by_year.index.max())}). It is the season's "
                "expected capacity, the same for every week. A cleared area of the twin is a 1 km² block in which a "
                "hazardous area is cleared, so the twin's areas and the register's operations in the table below "
                "are not like for like."
            )

        # Municipality-by-municipality past victims (by condition & outcome), past demining, and the twin's areas
        people_df = get_people_df(bundle)
        act_df = grid_map.get_activity_df(bundle)
        cal_rows = []
        for code in config.PILOT:
            name = config.NAME_OF[code]
            p_sub = people_df[people_df["municipality_code"] == code] if not people_df.empty else pd.DataFrame()
            a_sub = act_df[act_df["municipality_code"] == code] if not act_df.empty else pd.DataFrame()
            w_sub = areas_w[areas_w["municipality_code"] == code]

            civ_v = int(p_sub.loc[p_sub["condition"] == "Civilian", "victims"].sum()) if not p_sub.empty else 0
            sec_v = int(p_sub.loc[p_sub["condition"] == "Security forces", "victims"].sum()) if not p_sub.empty else 0
            inj_v = int(p_sub.loc[p_sub["outcome"] == "Injured", "victims"].sum()) if not p_sub.empty else 0
            kil_v = int(p_sub.loc[p_sub["outcome"] == "Killed", "victims"].sum()) if not p_sub.empty else 0
            tot_v = int(p_sub["victims"].sum()) if not p_sub.empty else 0

            h_nts = int(a_sub["non_technical_surveys"].sum()) if not a_sub.empty else 0
            h_ts = int(a_sub["technical_surveys"].sum()) if not a_sub.empty else 0
            h_clr = int(a_sub["clearance_operations"].sum()) if not a_sub.empty else 0
            h_km2 = float(a_sub["area_cleared_m2"].sum()) / 1e6 if not a_sub.empty else 0.0

            cal_rows.append(
                {
                    "Municipality": name,
                    "Recorded Victims (All Years)": tot_v,
                    "Civilian Victims": civ_v,
                    "Security Forces Victims": sec_v,
                    "Injured / Killed": f"{inj_v:,} / {kil_v:,}",
                    "Historical NTS (`datos.gov.co`)": h_nts,
                    "Historical TS (`datos.gov.co`)": h_ts,
                    "Historical Clearance Ops": h_clr,
                    "Historical Cleared (km²)": round(h_km2, 3),
                    f"Twin Released Areas (Wk {week})": int(w_sub["is_released"].sum()),
                    f"Twin Cleared Areas (Wk {week})": int((w_sub["status"] == "cleared").sum()),
                }
            )
        st.dataframe(
            pd.DataFrame(cal_rows).sort_values("Recorded Victims (All Years)", ascending=False),
            width="stretch",
            hide_index=True,
        )

    with g_tab3:
        st.markdown("##### Season Settings (`twin/config.py`)")
        st.caption(
            f"In the twin, heavy rain or high wind stops work on {s_cfg.rainy_day_loss:.0%} of the working days in "
            f"{_month_list(s_cfg.rainy_months)}, the two rainy seasons IDEAM (2025) describes for the Andean region, "
            f"and on {s_cfg.other_day_loss:.0%} in other months (Travis's settings, questions 34 and 37); the simulated "
            "field draws each zone's lost days at those rates. The season planner and the completion forecaster allow "
            "for these losses."
        )
        fig_ideam = plot_ideam_and_logistics_strip(bundle)
        st.pyplot(fig_ideam)
        plt.close(fig_ideam)

        c_cfg = bundle["setup"].twin.criteria
        n_consult = int(zones_df["consultation"].sum())
        intrusive = "technical survey" if c_cfg.technical_survey else "clearance"
        mw = s_cfg.month_weeks
        if s_cfg.ts_teams == 0:
            ts_text = (
                "no technical survey teams, since in Colombia TS 'is not foreseen' for the baseline areas "
                "(Mine Action Review, 2025; question 38)"
            )
        else:
            ts_text = f"{s_cfg.ts_teams} technical survey teams, {s_cfg.ts_weeks:g} weeks per area (questions 30 and 38)"
        param_rows = [
            {
                "Operational Parameter": "Season Calendar (`Season.weeks`, `Season.month_weeks`)",
                "Configured Value": f"{s_cfg.weeks} weeks across 12 months {s_cfg.month_weeks} ({s_cfg.days_per_week} working days/week)",
                "Set By / Source": (
                    f"{s_cfg.weeks} weeks, planned monthly and run weekly (Travis's setting, question 28); months of "
                    f"{mw[0]}, {mw[1]} and {mw[2]} weeks: this project's setting"
                ),
            },
            {
                "Operational Parameter": "Rainy Months (`Season.rainy_months`, `rainy_day_loss`, `other_day_loss`)",
                "Configured Value": (
                    f"{_month_list(s_cfg.rainy_months)}: {s_cfg.rainy_day_loss:.0%} of working days lost; "
                    f"other months: {s_cfg.other_day_loss:.0%}"
                ),
                "Set By / Source": (
                    "Travis's settings (questions 34 and 37): no work on days of heavy rain or high wind, at these "
                    "rates; the rainy months are the two rainy seasons IDEAM (2025) describes for the Andean region "
                    "(a simulated layer, not data)"
                ),
            },
            {
                "Operational Parameter": "Operational Zone Subdivision (`Season.zone_areas`, `zone_access_weight`)",
                "Configured Value": f"~{s_cfg.zone_areas} areas/zone ({len(zones_df)} zones total), K-means weight {s_cfg.zone_access_weight:g} on log(1 + km to road)",
                "Set By / Source": (
                    "Colombia subdivides municipalities into operational zones by accessibility (Mine Action Review, "
                    f"2025; question 29); the access weight of {s_cfg.zone_access_weight:g} is this project's setting"
                ),
            },
            {
                "Operational Parameter": "Field Teams & Task Durations (`survey_teams`, `clearance_teams`)",
                "Configured Value": (
                    f"{s_cfg.survey_teams} survey teams (Zone survey: {s_cfg.zone_survey_weeks:g} wks, Area visit: {s_cfg.visit_weeks:g} wk) · "
                    f"{s_cfg.clearance_teams} clearance teams (Clearance: {s_cfg.clearance_weeks:g} wks/area, spread σ={s_cfg.duration_spread})"
                ),
                "Set By / Source": (
                    f"Travis's settings: {s_cfg.survey_teams} survey and {s_cfg.clearance_teams} clearance teams "
                    f"(question 31); zone survey {s_cfg.zone_survey_weeks:g} weeks (question 39), visit "
                    f"{s_cfg.visit_weeks:g} week{'s' if s_cfg.visit_weeks != 1 else ''} (question 32), clearance "
                    f"{s_cfg.clearance_weeks:g} weeks per area (question 33); {ts_text}; spread "
                    f"{s_cfg.duration_spread:g}: this project's setting"
                ),
            },
            {
                "Operational Parameter": "Community Consultation (`consultation_share`, `consultation_weeks`)",
                "Configured Value": (
                    f"{n_consult} of {len(zones_df)} zones (each drawn with a {s_cfg.consultation_share:.0%} chance) "
                    f"need consultation, with a {s_cfg.consultation_weeks}-week lead time before tasking"
                ),
                "Set By / Source": (
                    "Simulated consultation layer: this project's settings (share and lead time), for Travis to review"
                ),
            },
            {
                "Operational Parameter": "Security Access Closures (`closure_chance`, `closure_months`)",
                "Configured Value": f"1-in-{int(round(1.0 / s_cfg.closure_chance))} monthly chance per zone of closing for {s_cfg.closure_months[0]}–{s_cfg.closure_months[1]} months",
                "Set By / Source": (
                    "Travis's setting (question 36). The simulated weekly access feed pauses tasks in a closed zone; "
                    "a change in a planned zone brings a new plan that week."
                ),
            },
            {
                "Operational Parameter": "The Pilot's National Criteria (`Criteria.sources`, `min_years`)",
                "Configured Value": (
                    f"≥{c_cfg.sources} different first-hand sources (class 1–4) each reporting ≥{c_cfg.min_years} years of safe use; "
                    + (
                        f"an area with a past accident recorded inside is released only after {intrusive} "
                        if c_cfg.recorded_accident_needs_ts
                        else "a past accident recorded inside does not by itself rule out cancellation "
                    )
                    + f"(`recorded_accident_needs_ts={c_cfg.recorded_accident_needs_ts}`)"
                ),
                "Set By / Source": (
                    "The pilot's national criteria, invented for this project (Travis's choices, questions 19 to 23), "
                    "since IMAS leaves such numbers to each national authority; rules paraphrased from IMAS 07.11 "
                    "(§3.2, §4.3, §4.4) and IMAS 08.10 (§6.1, §8.2)"
                ),
            },
        ]
        st.dataframe(pd.DataFrame(param_rows), width="stretch", hide_index=True)

    with g_tab4:
        st.markdown("##### IMAS 08.10 (§8.2) Reliability Classes and the Saved Rule Check")
        st.caption(
            "The reviewer reads `rules/land_release_rules.md`, which paraphrases parts of **IMAS 07.11 (*Land release*, "
            "2nd ed., 2026)**, **IMAS 08.10 (*Non-technical survey*)** and **IMAS 08.20 (*Technical survey*)** and adds "
            "project rules; `twin/rules.py` checks each decision against R1 to R6 and the pilot's national criteria, "
            "invented for this project."
        )
        ladder_df = pd.DataFrame(
            [
                {
                    "Class": f"Class {num}",
                    "IMAS 08.10 (§8.2) Source Category": title,
                    "Definition (after `rules/land_release_rules.md`)": desc,
                    "Treatment by `twin.rules.rule_check` (R1–R6)": effect,
                }
                for num, title, desc, effect, _ in IMAS_RELIABILITY_LADDER
            ]
        )
        st.dataframe(ladder_df, width="stretch", hide_index=True)

        if not reviews_df.empty:
            st.markdown("##### The Saved Rule Check of a Dossier Review")
            option_indices = reviews_df.index.tolist()

            def _fmt_rev(idx: int) -> str:
                r = reviews_df.loc[idx]
                return (
                    f"Week {int(r['week']):02d} · Area {r['area_id']} ({r['municipality']}) · "
                    f"Visit {int(r['visit'])} · Submitted: {HUMAN_OUTCOME.get(str(r['outcome']), str(r['outcome']))}"
                )

            sel_idx = st.selectbox(
                f"Pick one of the {len(reviews_df)} dossier reviews to see its saved rule check:",
                options=option_indices,
                index=0,
                format_func=_fmt_rev,
                key="grounding_rule_sim_select",
            )
            sel_row = reviews_df.loc[sel_idx]
            from loaders import load_review_detail

            det = load_review_detail(
                str(root),
                week=int(sel_row["week"]),
                area_id=str(sel_row["area_id"]),
                visit=int(sel_row["visit"]),
            )
            if det.get("record"):
                render_imas_and_rule_check_inspector(det["record"], bundle["setup"].twin.criteria)

    with g_tab5:
        st.markdown("##### Cryptographic Data Integrity (`twin.data.verify()`) & Registered Layer Governance (`twin.layers`)")
        st.caption(
            "When the twin is built (`twin.layers.build_twin`), `twin.data.verify()` computes the SHA-256 digest of each "
            "of the three upstream public datasets in 1 MB chunks and checks it against the checksum published in "
            "`data/README.md` (P1, P2 and P3); the build stops if any file differs."
        )
        checks_df = get_checks_df(bundle)
        if not checks_df.empty:
            upstream_meta = {
                "victims": (
                    "Presidencia de la República de Colombia — Datos Abiertos (`datos.gov.co` `yhxn-eqqw`)",
                    "CC BY-SA 4.0",
                    "Municipal victim counts by year, month, condition & outcome (coordinates & site names dropped on load per P1)",
                ),
                "demining": (
                    "Presidencia de la República de Colombia — Datos Abiertos (`datos.gov.co` `jbqm-s5tn`)",
                    "CC BY-SA 4.0",
                    "Municipal humanitarian demining surveys, clearance operations, m² cleared & MAP, MUSE and AEI counts (summed per P2)",
                ),
                "reland": (
                    "Dulce Rubio et al. (2024) — RELand Hugging Face (`mdulcer10/landmine-detection`)",
                    "CC BY 4.0",
                    "26,761 cells (500 m × 500 m) across 15 Antioquia municipalities with 63 features (terrain, climate, land use, relief, population, distances); coordinates set aside on load, after placing each cell on the grid",
                ),
            }
            disp_checks = checks_df.copy()
            disp_checks["Upstream Publisher / Portal"] = disp_checks["dataset"].map(
                lambda k: upstream_meta.get(str(k), ("—", "—", "—"))[0]
            )
            disp_checks["Licence"] = disp_checks["dataset"].map(
                lambda k: upstream_meta.get(str(k), ("—", "—", "—"))[1]
            )
            disp_checks["Governance & Privacy Handling"] = disp_checks["dataset"].map(
                lambda k: upstream_meta.get(str(k), ("—", "—", "—"))[2]
            )
            disp_checks["Integrity Verified"] = disp_checks["matches"].map(
                lambda m: "✅ SHA-256 Verified" if m else "❌ Checksum Mismatch"
            )
            st.dataframe(
                disp_checks[
                    [
                        "dataset",
                        "file",
                        "bytes",
                        "sha256 (first 12)",
                        "Integrity Verified",
                        "Upstream Publisher / Portal",
                        "Licence",
                        "Governance & Privacy Handling",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        st.markdown("##### Registered Data Layer Catalogue (`twin.layers.LAYERS`)")
        layer_df = bundle.get("layer_df", pd.DataFrame())
        if not layer_df.empty:
            st.dataframe(layer_df, width="stretch", hide_index=True)
