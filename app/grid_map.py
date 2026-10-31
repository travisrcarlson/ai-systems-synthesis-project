"""Visual schematic & twin grid map helpers (no axes, no scale, no coordinates)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from twin import config
from loaders import (
    HUMAN_AREA_STATUS,
    HUMAN_TASK_KIND,
    humanize_log_table,
    humanize_reviews_table,
    humanize_zone_table,
    load_review_detail,
    load_saved_reviews_public,
    render_recommendation_box,
)

SURFACE = "#faf9f5"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID_BG = "#e6e4dd"

ZONE_COLOURS = ["#cfe0f5", "#f6d6c4", "#cdeee0", "#e2d6f0", "#efe6bf", "#dcdcd6"]

AREA_STATUS_ORDER = ["unknown", "not_suspected", "suspected", "confirmed", "cancelled", "reduced", "cleared"]
AREA_STATUS_COLOURS = {
    "unknown": "#e6e4dd",
    "not_suspected": "#c9c4b6",
    "suspected": "#f0a25c",
    "confirmed": "#d95f43",
    "cancelled": "#b8d8be",
    "reduced": "#7ec4a5",
    "cleared": "#2b7bba",
}

ZONE_STATE_ORDER = ["not surveyed", "in survey", "surveyed", "free"]
ZONE_STATE_COLOURS = {
    "not surveyed": "#e6e4dd",
    "in survey": "#f2c97d",
    "surveyed": "#8cb8e8",
    "free": "#3b9a72",
}

PLAN_ROLE_ORDER = ["not in plan", "survey only", "survey + clearance"]
PLAN_ROLE_COLOURS = {
    "not in plan": "#e6e4dd",
    "survey only": "#9ecae1",
    "survey + clearance": "#2171b5",
}


def compute_grid(
    cells_meta: pd.DataFrame,
    values: pd.Series,
    municipality_code: str,
    zones_index: pd.Index,
    areas_index: pd.Index,
) -> np.ndarray:
    """Pure-DataFrame equivalent of twin.map_layer(values, municipality_code).

    Rows run north to south and columns west to east. The grid carries no coordinates:
    plotted with ax.axis('off'), it shows the shape of the municipality and where values
    lie within it, with no axes, no scale and no coordinates.
    """
    part = cells_meta[cells_meta["municipality_code"] == municipality_code]
    if part.empty:
        return np.full((1, 1), np.nan)
    if values.index.isin(zones_index).all():
        per_cell = values.reindex(part["zone_id"]).to_numpy(dtype=float)
    elif values.index.isin(areas_index).all():
        per_cell = values.reindex(part["area_id"]).to_numpy(dtype=float)
    else:
        per_cell = values.reindex(part.index).to_numpy(dtype=float)
    rows = (part["row"] - part["row"].min()).to_numpy()
    cols = (part["col"] - part["col"].min()).to_numpy()
    grid = np.full((int(rows.max()) + 1, int(cols.max()) + 1), np.nan)
    grid[int(rows.max()) - rows, cols] = per_cell
    return grid


def zone_colouring(
    municipality_code: str,
    zones_meta: pd.DataFrame,
    areas_meta: pd.DataFrame,
    neighbours: dict,
) -> pd.Series:
    """Greedy 6-colour assignment for each zone in a municipality, as in notebook section 5.1."""
    zones = zones_meta.index[zones_meta["municipality_code"] == municipality_code]
    adj = {zn: set() for zn in zones}
    for area, near in neighbours.items():
        if area not in areas_meta.index:
            continue
        za = areas_meta.loc[area, "zone_id"]
        if za in adj:
            for b in near:
                if b in areas_meta.index:
                    zb = areas_meta.loc[b, "zone_id"]
                    if zb != za and zb in adj:
                        adj[za].add(zb)
    colour = {}
    for zn in zones:
        used = {colour[n] for n in adj[zn] if n in colour}
        colour[zn] = next((c for c in range(len(ZONE_COLOURS)) if c not in used), 0)
    return pd.Series(colour, dtype=float)


def zone_label_positions(cells_meta: pd.DataFrame, municipality_code: str) -> dict[str, tuple[float, float]]:
    """Compute the median (plot_row, plot_col) inside the municipality grid for each zone to place short labels."""
    part = cells_meta[cells_meta["municipality_code"] == municipality_code]
    if part.empty:
        return {}
    r_min = int(part["row"].min())
    c_min = int(part["col"].min())
    r_max = int(part["row"].max() - r_min)
    positions: dict[str, tuple[float, float]] = {}
    for zone_id, grp in part.groupby("zone_id"):
        plot_r = float(np.median(r_max - (grp["row"] - r_min)))
        plot_c = float(np.median(grp["col"] - c_min))
        positions[str(zone_id)] = (plot_r, plot_c)
    return positions


def area_status_at_week(bundle: dict, week: int) -> pd.Series:
    """Every area's status at the end of `week`, as the twin recorded it (copied each week while the season was
    rebuilt; see `loaders.RecordingRun`): unknown, not_suspected, suspected, confirmed, cancelled, reduced or cleared."""
    weekly = bundle.get("weekly_area_status") or {}
    areas = bundle["areas_meta"].index
    if not weekly:
        return pd.Series("unknown", index=areas, dtype=object)
    week = int(min(max(int(week), min(weekly)), max(weekly)))
    return weekly[week].reindex(areas).fillna("unknown").astype(object)


def zone_state_at_week(bundle: dict, week: int) -> pd.Series:
    """Every zone's state at the end of `week` ('not surveyed', 'in survey', 'surveyed', 'free'), as the twin's own
    zone table recorded it (copied each week while the season was rebuilt; see `loaders.RecordingRun`)."""
    weekly = bundle.get("weekly_zone_state") or {}
    zones = bundle["zones_meta"].index
    if not weekly:
        return pd.Series("not surveyed", index=zones, dtype=object)
    week = int(min(max(int(week), min(weekly)), max(weekly)))
    return weekly[week].reindex(zones).fillna("not surveyed").astype(object)


def chance_in_force(bundle: dict, week: int) -> tuple[pd.Series, object]:
    """The risk overlay's chances in force in `week` (updated at each monthly plan), and its record."""
    weekly = bundle.get("weekly_risk") or {}
    areas = bundle["areas_meta"].index
    rec = None
    if weekly:
        week = int(min(max(int(week), min(weekly)), max(weekly)))
        rec = weekly[week]
    if rec is None:
        rec = bundle.get("risk")
    if rec is not None and rec.ok:
        return rec.answer["areas"]["chance"].reindex(areas).astype(float), rec
    return pd.Series(np.nan, index=areas), rec


def rule_first_n(rule_answer: dict, milp_answer: dict) -> dict:
    """The rule of thumb answers with every zone that has work, in its order (twin/planner.py); to compare it with
    the integer programme's plan, take its first N zones of each kind, N being the number the programme schedules."""
    if not rule_answer:
        return {}
    out = dict(rule_answer)
    for kind in ("survey", "clearance", "ts"):
        n = len(milp_answer.get(kind, [])) if milp_answer else 0
        out[kind] = list(rule_answer.get(kind, []))[:n]
    return out


def plan_role_series(zones_meta: pd.DataFrame, plan_dict: dict) -> pd.Series:
    """Map each zone to its role in a plan: 0 = not in plan, 1 = survey only, 2 = survey + clearance."""
    role = pd.Series(0.0, index=zones_meta.index)
    if not plan_dict:
        return role
    survey_set = set(plan_dict.get("survey", []))
    clear_set = set(plan_dict.get("clearance", []))
    for z in zones_meta.index:
        if z in clear_set:
            role.loc[z] = 2.0
        elif z in survey_set:
            role.loc[z] = 1.0
    return role


def _draw_cell_grid(
    ax: plt.Axes,
    grid: np.ndarray,
    *,
    colours: list[str] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "",
    subtitle: str = "",
) -> plt.Axes.imshow:
    """Draw a municipality's cells as a grid of squares with no axes, no scale, and no coordinates."""
    ax.set_facecolor(SURFACE)
    if colours is not None:
        cmap_obj = ListedColormap(colours)
        cmap_obj.set_bad(color=SURFACE)
        img = ax.imshow(
            grid,
            cmap=cmap_obj,
            vmin=-0.5,
            vmax=len(colours) - 0.5,
            interpolation="nearest",
        )
    else:
        cmap_obj = plt.get_cmap(cmap or "YlOrRd").copy()
        cmap_obj.set_bad(color=SURFACE)
        img = ax.imshow(
            grid,
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
    if title:
        ax.set_title(title, fontsize=9.0, fontweight="bold", color=INK, pad=4)
    if subtitle:
        ax.text(
            0.5,
            -0.03,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            color=INK_2,
        )
    ax.axis("off")
    return img


# ---- Unified regional map and the risk model's chance of mines (no coordinates, no scale, no axes) ------------
REGIONAL_CANVAS_H = 262
REGIONAL_CANVAS_W = 304

SHORT_MUNI_LABELS = {
    "05002": "Abejorral",
    "05021": "Alejandría",
    "05148": "El Carmen\nde Viboral",
    "05172": "Chigorodó",
    "05197": "Cocorná",
    "05313": "Granada",
    "05400": "La Unión",
    "05483": "Nariño",
    "05628": "Sabanalarga",
    "05649": "San Carlos",
    "05652": "San Francisco",
    "05660": "San Luis",
    "05667": "San Rafael",
    "05670": "San Roque",
    "05756": "Sonsón",
}


def _build_regional_canvas_index(cells_meta: pd.DataFrame) -> pd.DataFrame:
    """Place all 26,761 cells on a single composite regional canvas at 1:1 cell resolution.

    - The 13 contiguous municipalities (12 in Oriente, San Roque in Nordeste) form one block in the twin's
      row/col grid and keep their relative layout on the main right-hand canvas.
    - The 2 outlying municipalities (Chigorodó in Urabá and Sabanalarga in Occidente) are placed
      in framed inset boxes on the left at the exact same 1:1 cell scale.
    - Plotted with ax.axis('off'), no coordinates, axes or scale are ever used.
    """
    code = cells_meta["municipality_code"].to_numpy()
    r = cells_meta["row"].to_numpy(dtype=int)
    c = cells_meta["col"].to_numpy(dtype=int)

    plot_r = 5 + (250 - r)
    plot_c = 92 + (c - 281)

    chig = code == "05172"
    plot_r[chig] = 15 + (523 - r[chig])
    plot_c[chig] = 8 + (c[chig] - 0)

    saba = code == "05628"
    plot_r[saba] = 110 + (364 - r[saba])
    plot_c[saba] = 30 + (c[saba] - 212)

    out = cells_meta.copy()
    out["plot_r"] = plot_r
    out["plot_c"] = plot_c
    return out


def _extract_grid_boundary_segments(
    id_grid: np.ndarray,
    mode: str = "outer",
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract vector line segments along pixel boundaries of a 2D integer grid (`-1` = background).

    - mode == 'outer': boundary between background (-1) and landmass (>= 0)
    - mode == 'inner': boundary between two different landmass IDs (both >= 0 and a != b)
    - mode == 'mask': boundary where a binary 0/1 mask changes between 0 and 1
    """
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    h, w = id_grid.shape

    # Horizontal edges between row r and r+1
    a_h = id_grid[:-1, :]
    b_h = id_grid[1:, :]
    if mode == "outer":
        mask_h = ((a_h == -1) & (b_h >= 0)) | ((a_h >= 0) & (b_h == -1))
    elif mode == "inner":
        mask_h = (a_h >= 0) & (b_h >= 0) & (a_h != b_h)
    else:
        mask_h = a_h != b_h
    rs, cs = np.nonzero(mask_h)
    for r, c in zip(rs, cs):
        y = float(r) + 0.5
        segments.append(((float(c) - 0.5, y), (float(c) + 0.5, y)))

    # Vertical edges between col c and c+1
    a_v = id_grid[:, :-1]
    b_v = id_grid[:, 1:]
    if mode == "outer":
        mask_v = ((a_v == -1) & (b_v >= 0)) | ((a_v >= 0) & (b_v == -1))
    elif mode == "inner":
        mask_v = (a_v >= 0) & (b_v >= 0) & (a_v != b_v)
    else:
        mask_v = a_v != b_v
    rs, cs = np.nonzero(mask_v)
    for r, c in zip(rs, cs):
        x = float(c) + 0.5
        segments.append(((x, float(r) - 0.5), (x, float(r) + 0.5)))

    return segments


def _draw_regional_inset_decorations(ax: plt.Axes, public_df: pd.DataFrame | None = None,
                                     counts: tuple[int, int, int] | None = None) -> None:
    """Draw clean cartographic inset frames for Chigorodó (Urabá) and Sabanalarga (Occidente) and a regional key."""
    # Inset 1: Urabá sub-region (Chigorodó)
    ax.add_patch(
        FancyBboxPatch(
            (4, 6),
            80,
            82,
            boxstyle="round,pad=0.3,rounding_size=3.0",
            facecolor="#f5f3ec",
            edgecolor="#9c9990",
            linewidth=0.9,
            linestyle="--",
            zorder=0,
        )
    )
    ax.text(
        44,
        11.0,
        "URABÁ SUB-REGION (INSET)",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=INK_2,
        zorder=5,
    )

    # Inset 2: Occidente sub-region (Sabanalarga)
    ax.add_patch(
        FancyBboxPatch(
            (14, 100),
            62,
            74,
            boxstyle="round,pad=0.3,rounding_size=3.0",
            facecolor="#f5f3ec",
            edgecolor="#9c9990",
            linewidth=0.9,
            linestyle="--",
            zorder=0,
        )
    )
    ax.text(
        45,
        105.0,
        "OCCIDENTE SUB-REGION (INSET)",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=INK_2,
        zorder=5,
    )

    # Header note for the 13 contiguous municipalities
    ax.text(
        195,
        6.5,
        "13 CONTIGUOUS MUNICIPALITIES (12 IN ORIENTE, SAN ROQUE IN NORDESTE)",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=INK_2,
        zorder=5,
    )

    # Bottom-left schematic regional key box
    ax.add_patch(
        FancyBboxPatch(
            (4, 184),
            80,
            71,
            boxstyle="round,pad=0.3,rounding_size=3.0",
            facecolor="#f4f1e8",
            edgecolor="#c5c1b6",
            linewidth=0.9,
            zorder=1,
        )
    )
    ax.text(
        44,
        193,
        "ANTIOQUIA PILOT REGION",
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=INK,
        zorder=5,
    )
    ax.text(
        44,
        211,
        (f"15 Municipalities · {counts[0]:,} Zones\n{counts[1]:,} Areas · {counts[2]:,} Grid Cells\n\n"
         if counts else "15 Municipalities\n\n")
        + "• Contiguous block: 13 (12 Oriente, 1 Nordeste)\n"
        "• Urabá inset: Chigorodó\n"
        "• Occidente inset: Sabanalarga",
        ha="center",
        va="center",
        fontsize=6.3,
        color=INK_2,
        zorder=5,
    )
    ax.text(
        44,
        246,
        "Schematic twin grid (no coordinates)",
        ha="center",
        va="center",
        fontsize=6.0,
        style="italic",
        color=MUTED,
        zorder=5,
    )


def plot_regional_hero_dashboard_map(
    bundle: dict,
    week: int,
    log_df: pd.DataFrame,
    risk_source: str = "updated",
    show_cleaned_contours: bool = True,
    show_muni_labels: bool = True,
) -> plt.Figure:
    """Render a unified 2-panel cartographic dashboard map across all 15 pilot municipalities (no coordinates/axes):

    - Left panel: the risk overlay's chance of mines by area (where to look first; not evidence about an area),
      each area's own chance with no smoothing, and the areas released by `week` drawn on top.
    - Right panel: Operational Land Release & Clearance Status Map at `week`, combining zone-level progress and
      area-level clearance/release status on the same unified regional canvas.
    """
    cells_meta = bundle["cells_meta"]
    zones_meta = bundle["zones_meta"]
    areas_meta = bundle["areas_meta"]
    tasks_df = bundle.get("tasks_df", pd.DataFrame())

    canvas_df = _build_regional_canvas_index(cells_meta)
    pr = canvas_df["plot_r"].to_numpy(dtype=int)
    pc = canvas_df["plot_c"].to_numpy(dtype=int)

    muni_codes = list(config.NAME_OF.keys())
    muni_to_int = {c: idx for idx, c in enumerate(muni_codes)}
    zone_ids = list(zones_meta.index)
    zone_to_int = {z: idx for idx, z in enumerate(zone_ids)}

    muni_grid = np.full((REGIONAL_CANVAS_H, REGIONAL_CANVAS_W), -1, dtype=int)
    zone_grid = np.full((REGIONAL_CANVAS_H, REGIONAL_CANVAS_W), -1, dtype=int)
    muni_grid[pr, pc] = canvas_df["municipality_code"].map(muni_to_int).to_numpy(dtype=int)
    zone_grid[pr, pc] = canvas_df["zone_id"].map(zone_to_int).to_numpy(dtype=int)

    outer_segs = _extract_grid_boundary_segments(muni_grid, mode="outer")
    muni_segs = _extract_grid_boundary_segments(muni_grid, mode="inner")
    zone_segs = _extract_grid_boundary_segments(zone_grid, mode="inner")

    # The risk overlay's chance of mines by area: where to look first (not evidence about an area)
    n_areas = len(areas_meta)
    if risk_source == "initial":
        risk_rec = bundle.get("risk")
        chance_ser = (
            risk_rec.answer["areas"]["chance"].astype(float)
            if (risk_rec is not None and risk_rec.ok)
            else pd.Series(np.nan, index=areas_meta.index)
        )
        heat_subtitle = f"Start of season (week 1): the gated models' calibrated chance for each of the {n_areas:,} areas"
    elif risk_source == "baseline":
        models_obj = bundle.get("setup").models if bundle.get("setup") is not None else None
        if models_obj is not None:
            chance_ser = models_obj.chances("baseline")["chance"].astype(float)
        else:
            chance_ser = pd.Series(np.nan, index=areas_meta.index)
        heat_subtitle = f"The baseline: nearness to past accidents, calibrated, for each of the {n_areas:,} areas"
    else:
        chance_ser, _rec = chance_in_force(bundle, week)
        heat_subtitle = (f"Chances in force in week {week} (updated at each monthly plan) for each of the "
                         f"{n_areas:,} areas")

    # Area and zone statuses at selected week
    status_ser = area_status_at_week(bundle, week)
    z_state_ser = zone_state_at_week(bundle, week)

    # Build the left panel grid: the risk model's chance of mines for each area
    raw_heat_grid = np.full((REGIONAL_CANVAS_H, REGIONAL_CANVAS_W), np.nan, dtype=float)
    raw_heat_grid[pr, pc] = chance_ser.reindex(canvas_df["area_id"]).to_numpy(dtype=float)
    heat_grid = raw_heat_grid   # each area's own chance, with no smoothing

    # Mask of released areas (cancelled, reduced, cleared) and of cleared areas at the selected week
    cell_status = status_ser.reindex(canvas_df["area_id"]).to_numpy(dtype=object)
    cleaned_mask = np.zeros((REGIONAL_CANVAS_H, REGIONAL_CANVAS_W), dtype=int)
    cleared_mask = np.zeros((REGIONAL_CANVAS_H, REGIONAL_CANVAS_W), dtype=int)
    cleaned_mask[pr, pc] = np.isin(cell_status, ["cancelled", "reduced", "cleared"]).astype(int)
    cleared_mask[pr, pc] = (cell_status == "cleared").astype(int)
    cleaned_segs = _extract_grid_boundary_segments(cleaned_mask, mode="mask") if show_cleaned_contours else []
    cleared_segs = _extract_grid_boundary_segments(cleared_mask, mode="mask") if show_cleaned_contours else []

    # Build Right Panel Grid: Combined Zone Progress + Area Operational Status at selected week
    # Categories:
    # 0 = Zone not surveyed (unknown area)
    # 1 = Zone in survey (unknown area)
    # 2 = Zone surveyed, open (unknown area)
    # 3 = Zone declared free (background area)
    # 4 = Cancelled area (released by survey)
    # 5 = Reduced area (released by technical survey)
    # 6 = Cleared area (released by physical clearance)
    # 7 = Suspected hazard area
    # 8 = Confirmed hazard area (awaiting / in clearance)
    ops_colours = [
        "#e8e5dc",  # 0: Zone not yet surveyed
        "#f5e4be",  # 1: Zone currently in survey
        "#d9e7f5",  # 2: Zone surveyed (open)
        "#d5efe0",  # 3: Zone declared free
        "#8fd1a8",  # 4: Area cancelled (released)
        "#48b078",  # 5: Area reduced (released)
        "#1b6ca8",  # 6: Area physically cleared
        "#f09436",  # 7: Suspected hazard area
        "#cf3a2b",  # 8: Confirmed hazard area
    ]
    z_map_code = {"not surveyed": 0.0, "in survey": 1.0, "surveyed": 2.0, "free": 3.0}
    cell_z_code = z_state_ser.reindex(canvas_df["zone_id"]).map(z_map_code).fillna(0.0).to_numpy(dtype=float)
    cell_ops_code = cell_z_code.copy()
    for st_name, code_val in (
        ("cancelled", 4.0),
        ("reduced", 5.0),
        ("cleared", 6.0),
        ("suspected", 7.0),
        ("confirmed", 8.0),
    ):
        cell_ops_code[cell_status == st_name] = code_val

    ops_grid = np.full((REGIONAL_CANVAS_H, REGIONAL_CANVAS_W), np.nan, dtype=float)
    ops_grid[pr, pc] = cell_ops_code

    # Create 2-panel side-by-side figure
    fig, (ax_heat, ax_ops) = plt.subplots(1, 2, figsize=(16.2, 7.8), facecolor=SURFACE)
    for ax in (ax_heat, ax_ops):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-2, REGIONAL_CANVAS_W + 2)
        ax.set_ylim(REGIONAL_CANVAS_H + 2, -2)
        ax.axis("off")
        _draw_regional_inset_decorations(ax, counts=(len(zones_meta), len(areas_meta), len(cells_meta)))

        # Subtle drop shadow behind landmass
        shadow_grid = np.where(muni_grid >= 0, 1.0, np.nan)
        shadow_cmap = ListedColormap(["#dcd8cc"])
        shadow_cmap.set_bad(color=(0, 0, 0, 0))
        ax.imshow(
            shadow_grid,
            cmap=shadow_cmap,
            extent=(1.8, REGIONAL_CANVAS_W + 1.8, REGIONAL_CANVAS_H + 1.8, 1.8),
            interpolation="nearest",
            zorder=1,
        )

    # Render Left Panel: the risk model's chance of mines, with released areas drawn on top
    heat_cmap = LinearSegmentedColormap.from_list(
        "chance_heat",
        ["#fbf6e9", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"],
    )
    heat_cmap.set_bad(color=(0, 0, 0, 0))
    vmax_heat = float(np.nanpercentile(heat_grid, 99.2)) if np.isfinite(heat_grid).any() else 0.30
    vmax_heat = max(0.18, min(0.45, vmax_heat))
    heat_img = ax_heat.imshow(
        heat_grid,
        cmap=heat_cmap,
        vmin=0.0,
        vmax=vmax_heat,
        interpolation="nearest",
        zorder=2,
    )

    # Draw the areas released by the selected week on top of the heat map
    is_released_area = status_ser.isin(["cancelled", "reduced", "cleared"])

    if show_cleaned_contours:
        # Solid overlay on top of the heat map for the areas released by `week`:
        # 0 = Cancelled (released by non-technical survey), 1 = Reduced (by technical survey), 2 = Cleared
        removed_overlay_grid = np.full((REGIONAL_CANVAS_H, REGIONAL_CANVAS_W), np.nan, dtype=float)
        rem_codes = np.full(len(canvas_df), np.nan, dtype=float)
        rem_codes[cell_status == "cancelled"] = 0.0
        rem_codes[cell_status == "reduced"] = 1.0
        rem_codes[cell_status == "cleared"] = 2.0
        removed_overlay_grid[pr, pc] = rem_codes

        removed_cmap = ListedColormap(["#8fd1a8", "#48b078", "#1b6ca8"])
        removed_cmap.set_bad(color=(0, 0, 0, 0))
        ax_heat.imshow(
            removed_overlay_grid,
            cmap=removed_cmap,
            vmin=-0.5,
            vmax=2.5,
            interpolation="nearest",
            zorder=2.8,
        )
        if cleaned_segs:
            ax_heat.add_collection(
                LineCollection(cleaned_segs, colors="#0d8a72", linewidths=0.70, alpha=0.92, zorder=3.5)
            )
        if cleared_segs:
            ax_heat.add_collection(
                LineCollection(cleared_segs, colors="#0b4f6c", linewidths=1.00, alpha=0.96, zorder=3.7)
            )

    # Render Right Panel: Operational Land Release & Clearance Status Map
    ops_cmap = ListedColormap(ops_colours)
    ops_cmap.set_bad(color=(0, 0, 0, 0))
    ax_ops.imshow(
        ops_grid,
        cmap=ops_cmap,
        vmin=-0.5,
        vmax=len(ops_colours) - 0.5,
        interpolation="nearest",
        zorder=2,
    )

    # Add vector boundaries (zone borders, municipal borders, outer perimeter) and municipality name callouts
    for ax in (ax_heat, ax_ops):
        ax.add_collection(LineCollection(zone_segs, colors="#1f1e1b", linewidths=0.22, alpha=0.20, zorder=3))
        ax.add_collection(LineCollection(muni_segs, colors="#faf9f5", linewidths=1.55, alpha=0.75, zorder=4))
        ax.add_collection(LineCollection(muni_segs, colors="#2b2a27", linewidths=0.80, alpha=0.92, zorder=4.1))
        ax.add_collection(LineCollection(outer_segs, colors="#181816", linewidths=1.15, alpha=0.95, zorder=4.2))

        if show_muni_labels:
            for code, grp in canvas_df.groupby("municipality_code"):
                med_r = float(np.median(grp["plot_r"]))
                med_c = float(np.median(grp["plot_c"]))
                lbl = SHORT_MUNI_LABELS.get(str(code), config.NAME_OF.get(str(code), str(code)))
                ax.text(
                    med_c,
                    med_r,
                    lbl,
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    fontweight="bold",
                    color=INK,
                    bbox={
                        "boxstyle": "round,pad=0.14,rounding_size=0.2",
                        "facecolor": "#faf9f5d9",
                        "edgecolor": "#b8b4a9",
                        "linewidth": 0.45,
                    },
                    zorder=6,
                )

    ax_heat.set_title(
        "1. The Risk Model's Chance of Mines (Where to Look First)",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    ax_heat.text(
        0.5,
        -0.015,
        heat_subtitle + f"; the {int(is_released_area.sum())} areas released by week {week} drawn on top",
        transform=ax_heat.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        color=INK_2,
    )

    n_free_w = int((z_state_ser == "free").sum())
    n_rel_w = int(is_released_area.sum())
    n_clr_w = int((status_ser == "cleared").sum())
    ax_ops.set_title(
        f"2. Operational Land Release & Clearance Status (Week {week} of 52)",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
        pad=8,
    )
    ax_ops.text(
        0.5,
        -0.015,
        f"Week {week}: {n_free_w}/{len(zones_meta)} zones declared free · {n_rel_w} areas released ({n_clr_w} cleared)",
        transform=ax_ops.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        color=INK_2,
    )

    # Bottom colour bar for the left panel, and the legend for the released areas drawn on top
    cbar_ax = fig.add_axes([0.04, 0.055, 0.22, 0.020])
    cb = fig.colorbar(heat_img, cax=cbar_ax, orientation="horizontal")
    cb.set_label(f"Chance of mines (darkest colour: {vmax_heat:.2f} or more)", fontsize=8.0, color=INK)
    cb.ax.tick_params(labelsize=7.5)

    if show_cleaned_contours:
        heat_leg_handles = [
            Patch(color="#1b6ca8", label=f"Released: cleared ({n_clr_w})"),
            Patch(color="#48b078", label=f"Released: reduced ({int((status_ser == 'reduced').sum())})"),
            Patch(color="#8fd1a8", label=f"Released: cancelled ({int((status_ser == 'cancelled').sum())})"),
        ]
        fig.legend(
            handles=heat_leg_handles,
            loc="lower left",
            bbox_to_anchor=(0.28, 0.024),
            ncol=1,
            frameon=False,
            fontsize=7.6,
        )

    # Bottom legend for Right Panel
    ops_legend_handles = [
        Patch(color="#d5efe0", label="Zone declared free"),
        Patch(color="#d9e7f5", label="Zone surveyed (open)"),
        Patch(color="#f5e4be", label="Zone in survey"),
        Patch(color="#e8e5dc", label="Zone not surveyed"),
        Patch(color="#1b6ca8", label="Area cleared"),
        Patch(color="#48b078", label="Area reduced"),
        Patch(color="#8fd1a8", label="Area cancelled"),
        Patch(color="#f09436", label="Suspected area"),
        Patch(color="#cf3a2b", label="Confirmed area"),
    ]
    fig.legend(
        handles=ops_legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.022),
        ncol=3,
        frameon=False,
        fontsize=7.8,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.13, wspace=0.04)
    return fig


def get_activity_df(bundle: dict) -> pd.DataFrame:
    """Return the twin's registered humanitarian demining activity layer (`s.twin.activity`)."""
    setup_obj = bundle.get("setup")
    if setup_obj is not None and hasattr(setup_obj, "twin") and hasattr(setup_obj.twin, "activity"):
        return setup_obj.twin.activity.copy()
    return pd.DataFrame()


def plot_all_municipalities_grid(
    bundle: dict,
    layer_name: str,
    week: int,
    log_df: pd.DataFrame,
) -> plt.Figure:
    """Render all 15 pilot municipalities in a 3x5 grid of twin cell maps (no axes, no scale, no coordinates)."""
    cells_meta = bundle["cells_meta"]
    zones_meta = bundle["zones_meta"]
    areas_meta = bundle["areas_meta"]
    neighbours = bundle["neighbours"]
    tasks_df = bundle.get("tasks_df", pd.DataFrame())
    zones_end = bundle.get("zones")

    fig, axes = plt.subplots(3, 5, figsize=(15.5, 9.2), facecolor=SURFACE)
    axes_flat = axes.ravel()

    # Prepare layer values across all municipalities
    legend_handles = []
    cbar_img = None
    cbar_label = ""

    if layer_name == "Area status in selected week":
        status_ser = area_status_at_week(bundle, week)
        code_map = {k: float(i) for i, k in enumerate(AREA_STATUS_ORDER)}
        val_ser = status_ser.map(code_map)
        colours = [AREA_STATUS_COLOURS[k] for k in AREA_STATUS_ORDER]
        legend_handles = [
            Patch(color=AREA_STATUS_COLOURS[k], label=f"{HUMAN_AREA_STATUS.get(k, k)} ({int((status_ser == k).sum())})")
            for k in AREA_STATUS_ORDER
        ]
    elif layer_name == "Zone state in selected week":
        z_ser = zone_state_at_week(bundle, week)
        code_map = {k: float(i) for i, k in enumerate(ZONE_STATE_ORDER)}
        val_ser = z_ser.map(code_map)
        colours = [ZONE_STATE_COLOURS[k] for i, k in enumerate(ZONE_STATE_ORDER)]
        legend_handles = [
            Patch(color=ZONE_STATE_COLOURS[k], label=f"{k} ({int((z_ser == k).sum())} zones)")
            for k in ZONE_STATE_ORDER
        ]
    elif layer_name == "Area risk chance (week 1 gated model)":
        risk_rec = bundle.get("risk")
        val_ser = (
            risk_rec.answer["areas"]["chance"].astype(float)
            if (risk_rec is not None and risk_rec.ok)
            else pd.Series(0.0, index=areas_meta.index)
        )
        colours = None
        cbar_label = "Calibrated chance of mines, week 1 (darkest colour: 0.35 or more)"
    elif layer_name == "Area risk chance (in force in the selected week)":
        val_ser, _rec = chance_in_force(bundle, week)
        colours = None
        cbar_label = f"Calibrated chance of mines in force in week {week} (darkest colour: 0.35 or more)"
    elif layer_name == "Week-1 plan: integer programme (MILP)":
        plan_rec = bundle.get("plan")
        val_ser = plan_role_series(zones_meta, plan_rec.answer if (plan_rec and plan_rec.ok) else {})
        colours = [PLAN_ROLE_COLOURS[k] for k in PLAN_ROLE_ORDER]
        legend_handles = [
            Patch(color=PLAN_ROLE_COLOURS[k], label=f"{k} ({int((val_ser == i).sum())} zones)")
            for i, k in enumerate(PLAN_ROLE_ORDER)
        ]
    elif layer_name == "Week-1 plan: rule of thumb":
        rule_rec = bundle.get("rule")
        plan_rec = bundle.get("plan")
        val_ser = plan_role_series(zones_meta, rule_first_n(
            rule_rec.answer if (rule_rec and rule_rec.ok) else {},
            plan_rec.answer if (plan_rec and plan_rec.ok) else {}))
        colours = [PLAN_ROLE_COLOURS[k] for k in PLAN_ROLE_ORDER]
        legend_handles = [
            Patch(color=PLAN_ROLE_COLOURS[k], label=f"{k} ({int((val_ser == i).sum())} zones)")
            for i, k in enumerate(PLAN_ROLE_ORDER)
        ]
    elif layer_name == "Completion forecast: P50 week":
        fc_rec = bundle.get("fc")
        if fc_rec is not None and fc_rec.ok and "zones" in fc_rec.answer:
            val_ser = fc_rec.answer["zones"]["P50 week"].reindex(zones_meta.index).astype(float)
        else:
            val_ser = pd.Series(np.nan, index=zones_meta.index)
        colours = None
        cbar_label = "Forecast P50 completion week (1–52)"
    else:
        # Operational zones (6-colour greedy colouring per municipality)
        val_ser = None
        colours = ZONE_COLOURS
        legend_handles = [
            Patch(color=ZONE_COLOURS[i], label=f"Zone colour slot {i + 1}") for i in range(len(ZONE_COLOURS))
        ]

    for idx, (code, name) in enumerate(config.NAME_OF.items()):
        ax = axes_flat[idx]
        n_zones = int((zones_meta["municipality_code"] == code).sum())
        n_areas = int((areas_meta["municipality_code"] == code).sum())
        free_note = ""
        if zones_end is not None and not zones_end.empty:
            m_zones = zones_end[zones_end["municipality"] == name]
            n_free = int((m_zones["free_week"] > 0).sum())
            free_note = f"{n_free}/{n_zones} zones free at w52"
        else:
            free_note = f"{n_zones} zones · {n_areas} areas"

        if val_ser is None:
            m_val = zone_colouring(code, zones_meta, areas_meta, neighbours)
        else:
            m_val = val_ser

        grid = compute_grid(cells_meta, m_val, code, zones_meta.index, areas_meta.index)
        if colours is not None:
            _draw_cell_grid(
                ax,
                grid,
                colours=colours,
                title=f"{name}",
                subtitle=free_note,
            )
        else:
            vmin = 1.0 if "P50" in layer_name else 0.0
            vmax = 52.0 if "P50" in layer_name else 0.35
            cmap_name = "viridis_r" if "P50" in layer_name else "YlOrRd"
            cbar_img = _draw_cell_grid(
                ax,
                grid,
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax,
                title=f"{name}",
                subtitle=free_note,
            )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 6),
            frameon=False,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.subplots_adjust(bottom=0.09, top=0.93, hspace=0.32, wspace=0.12)
    elif cbar_img is not None:
        cbar_ax = fig.add_axes([0.30, 0.03, 0.40, 0.02])
        cb = fig.colorbar(cbar_img, cax=cbar_ax, orientation="horizontal")
        cb.set_label(cbar_label, fontsize=8.5, color=INK)
        cb.ax.tick_params(labelsize=8)
        fig.subplots_adjust(bottom=0.10, top=0.93, hspace=0.32, wspace=0.12)
    else:
        fig.subplots_adjust(bottom=0.05, top=0.93, hspace=0.32, wspace=0.12)

    fig.suptitle(
        f"15 Pilot Municipalities of Antioquia — {layer_name}",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    return fig


def plot_municipality_detail_grids(
    bundle: dict,
    municipality_code: str,
    week: int,
    log_df: pd.DataFrame,
) -> plt.Figure:
    """Render 4 side-by-side twin cell grids for one municipality:
    1) Operational zones (with short zone labels),
    2) The risk model's chance of mines (all areas; areas released by the week outlined in teal),
    3) Area status in week W,
    4) Initial season plan.
    """
    cells_meta = bundle["cells_meta"]
    zones_meta = bundle["zones_meta"]
    areas_meta = bundle["areas_meta"]
    neighbours = bundle["neighbours"]
    name = config.NAME_OF[municipality_code]

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), facecolor=SURFACE)

    # Extract municipality cell grid for zone & perimeter boundary lines
    part = cells_meta[cells_meta["municipality_code"] == municipality_code]
    zone_ids_m = list(zones_meta.index[zones_meta["municipality_code"] == municipality_code])
    z_to_int = {z: i for i, z in enumerate(zone_ids_m)}
    z_int_ser = pd.Series(z_to_int, dtype=float)
    g_zid_float = compute_grid(cells_meta, z_int_ser, municipality_code, zones_meta.index, areas_meta.index)
    g_zid = np.nan_to_num(g_zid_float, nan=-1.0).astype(int)
    outer_segs = _extract_grid_boundary_segments(g_zid, mode="outer")
    zone_segs = _extract_grid_boundary_segments(g_zid, mode="inner")

    # Panel 1: Operational zones with short zone labels (Z01, Z02...)
    z_col = zone_colouring(municipality_code, zones_meta, areas_meta, neighbours)
    g_zones = compute_grid(cells_meta, z_col, municipality_code, zones_meta.index, areas_meta.index)
    n_zones = int((zones_meta["municipality_code"] == municipality_code).sum())
    _draw_cell_grid(
        axes[0],
        g_zones,
        colours=ZONE_COLOURS,
        title=f"1. {name}: {n_zones} Zones",
        subtitle="Zone numbers & boundaries",
    )
    pos_map = zone_label_positions(cells_meta, municipality_code)
    for zone_id, (pr, pc) in pos_map.items():
        short_lbl = zone_id.split("-")[-1]
        axes[0].text(
            pc,
            pr,
            short_lbl,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=INK,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "#ffffffcc", "edgecolor": "none"},
        )

    # Panel 2: the risk model's chance of mines in force in the selected week (each area's own value)
    chance_ser, _rec = chance_in_force(bundle, week)
    g_heat = compute_grid(cells_meta, chance_ser, municipality_code, zones_meta.index, areas_meta.index)
    vmax_m = float(np.nanpercentile(g_heat, 98.5)) if np.isfinite(g_heat).any() else 0.25
    vmax_m = max(0.15, min(0.45, vmax_m))

    status_ser = area_status_at_week(bundle, week)
    m_areas = areas_meta["municipality_code"] == municipality_code
    m_status = status_ser[m_areas]
    n_cleaned_m = int(m_status.isin(["cancelled", "reduced", "cleared"]).sum())

    _draw_cell_grid(
        axes[1],
        g_heat,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax_m,
        title=f"2. Chance of Mines (Week {week})",
        subtitle=f"darkest: {vmax_m:.2f} or more · {n_cleaned_m} released areas on top",
    )
    rem_map = {"cancelled": 0.0, "reduced": 1.0, "cleared": 2.0}
    rem_ser = status_ser.map(rem_map)
    g_rem_overlay = compute_grid(cells_meta, rem_ser, municipality_code, zones_meta.index, areas_meta.index)
    rem_cmap = ListedColormap(["#8fd1a8", "#48b078", "#1b6ca8"])
    rem_cmap.set_bad(color=(0, 0, 0, 0))
    axes[1].imshow(g_rem_overlay, cmap=rem_cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    cleaned_ser = status_ser.isin(["cancelled", "reduced", "cleared"]).astype(float)
    g_cleaned_float = compute_grid(cells_meta, cleaned_ser, municipality_code, zones_meta.index, areas_meta.index)
    g_cleaned_mask = np.where(np.isfinite(g_cleaned_float) & (g_cleaned_float > 0.5), 1, 0)
    cleaned_segs = _extract_grid_boundary_segments(g_cleaned_mask, mode="mask")
    if cleaned_segs:
        axes[1].add_collection(LineCollection(cleaned_segs, colors="#0d8a72", linewidths=1.1, alpha=0.95))

    # Panel 3: Area status at selected week
    code_map = {k: float(i) for i, k in enumerate(AREA_STATUS_ORDER)}
    g_status = compute_grid(
        cells_meta,
        status_ser.map(code_map),
        municipality_code,
        zones_meta.index,
        areas_meta.index,
    )
    counts_str = ", ".join(
        f"{k.replace('_', ' ')}: {int((m_status == k).sum())}" for k in AREA_STATUS_ORDER
        if int((m_status == k).sum()) > 0
    )
    _draw_cell_grid(
        axes[2],
        g_status,
        colours=[AREA_STATUS_COLOURS[k] for k in AREA_STATUS_ORDER],
        title=f"3. Area Status (Week {week})",
        subtitle=counts_str or "all unknown",
    )

    # Panel 4: Initial season plan
    plan_rec = bundle.get("plan")
    p_roles = plan_role_series(zones_meta, plan_rec.answer if (plan_rec and plan_rec.ok) else {})
    g_plan = compute_grid(cells_meta, p_roles, municipality_code, zones_meta.index, areas_meta.index)
    m_zones_idx = zones_meta.index[zones_meta["municipality_code"] == municipality_code]
    m_roles = p_roles.reindex(m_zones_idx)
    n_surv = int((m_roles >= 1.0).sum())
    n_clr = int((m_roles == 2.0).sum())
    _draw_cell_grid(
        axes[3],
        g_plan,
        colours=[PLAN_ROLE_COLOURS[k] for k in PLAN_ROLE_ORDER],
        title="4. Initial Season Plan",
        subtitle=f"Planned: {n_surv} survey, {n_clr} clearance zones",
    )

    for ax in axes:
        if zone_segs:
            ax.add_collection(LineCollection(zone_segs, colors="#1f1e1b", linewidths=0.35, alpha=0.32))
        if outer_segs:
            ax.add_collection(LineCollection(outer_segs, colors="#181816", linewidths=0.95, alpha=0.90))

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.88, wspace=0.10)
    return fig


def plot_public_municipality_tiles(public_df: pd.DataFrame) -> plt.Figure:
    """Render a 3x5 schematic tile board of the 15 municipalities showing ONLY public municipal summary figures.

    No zone, area, or team is named, and no geographic coordinates, axes or scale are used.
    """
    fig, axes = plt.subplots(3, 5, figsize=(15.0, 6.8), facecolor=SURFACE)
    axes_flat = axes.ravel()

    status_palette = {
        "declared free": ("#d9f0e3", "#2d7d53"),
        "work in progress": ("#e6f0fb", "#2b6cb0"),
        "not started": ("#f2f1ed", "#6e6d68"),
    }

    for idx, (muni_name, row) in enumerate(public_df.iterrows()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        z_free = int(row.get("zones declared free", 0))
        z_surv = int(row.get("zones surveyed", 0))
        z_tot = int(row.get("zones", 0))
        a_rel = int(row.get("areas released", 0))
        a_clr = int(row.get("areas cleared", 0))
        victims = int(row.get("recorded victims (all years)", 0))
        free_wk = str(row.get("declared free (week)", "")).strip()

        if free_wk:
            status = f"declared free (w{free_wk})"
            fill_col, edge_col = status_palette["declared free"]
        elif z_surv > 0 or z_free > 0 or a_rel > 0:
            status = "work in progress"
            fill_col, edge_col = status_palette["work in progress"]
        else:
            status = "not started"
            fill_col, edge_col = status_palette["not started"]

        ax.add_patch(
            FancyBboxPatch(
                (0.03, 0.05),
                0.94,
                0.90,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.6,
                edgecolor=edge_col,
                facecolor=fill_col,
            )
        )

        ax.text(0.50, 0.82, str(muni_name), ha="center", va="center", fontsize=10.0, fontweight="bold", color=INK)
        ax.text(0.50, 0.66, status.upper(), ha="center", va="center", fontsize=7.8, fontweight="bold", color=edge_col)
        ax.text(
            0.50,
            0.45,
            f"Zones declared free: {z_free} / {z_tot} ({z_surv} surveyed)\n"
            f"Areas released: {a_rel} ({a_clr} cleared)",
            ha="center",
            va="center",
            fontsize=8.0,
            color=INK,
        )
        ax.text(
            0.50,
            0.18,
            f"Recorded victims (all years): {victims:,}",
            ha="center",
            va="center",
            fontsize=8.0,
            color=INK_2,
        )

    for j in range(len(public_df), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Public Municipal Summary Schematic — 15 Pilot Municipalities of Antioquia (no zone, area or team named)",
        fontsize=11,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.08, hspace=0.14)
    return fig


def render_municipality_map_explorer(
    root: Path,
    bundle: dict,
    log_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    saved_sitreps: dict,
    week: int,
) -> None:
    """Interactive visual map explorer combining the 15-municipality grid map and a single-municipality
    all-in-one dashboard (zones, tasks, plan, forecast, dossier reviews with sign-off sheets, log, and sitrep).
    """
    st.subheader("Regional Grid Map & Municipality Explorer")
    st.caption(
        "Compare all 15 municipalities side by side on the schematic grid map (shown without geographic coordinates, "
        "axes or scale), then select any municipality to inspect its zones, field tasks, plan, completion forecast, "
        "sign-off sheets, and the week's situation report (for the whole pilot)."
    )

    layer_label_map = {
        "Area status in selected week": "Area status in selected week",
        "Zone progress in selected week": "Zone state in selected week",
        "Operational zone layout": "Operational zones (6-colour layout)",
        "Estimated hazard chance (start of season)": "Area risk chance (week 1 gated model)",
        "Estimated hazard chance (in force in the selected week)": "Area risk chance (in force in the selected week)",
        "Initial season plan (optimised programme)": "Week-1 plan: integer programme (MILP)",
        "Initial season plan (rule of thumb, first N zones)": "Week-1 plan: rule of thumb",
        "Expected completion week (P50 forecast)": "Completion forecast: P50 week",
    }
    col_l, col_m = st.columns([3, 2])
    with col_l:
        chosen_display_layer = st.selectbox(
            "Colour all 15 municipalities by:",
            options=list(layer_label_map.keys()),
            index=0,
        )
        chosen_layer = layer_label_map[chosen_display_layer]
    muni_items = [(code, name) for code, name in config.NAME_OF.items()]
    with col_m:
        selected_muni_idx = st.selectbox(
            "Focus on municipality:",
            options=range(len(muni_items)),
            format_func=lambda i: muni_items[i][1],
            index=9,  # San Carlos by default
        )
    selected_code = muni_items[selected_muni_idx][0]
    selected_name = config.NAME_OF[selected_code]

    # 1. All-15-municipalities overview grid map
    with st.expander("All 15 Municipalities — Regional Grid Map", expanded=True):
        fig_all = plot_all_municipalities_grid(bundle, chosen_layer, week, log_df)
        st.pyplot(fig_all)
        plt.close(fig_all)

    # 2. Selected municipality 4-panel map + unified dashboard
    st.markdown(f"#### Municipality Focus: **{selected_name}** — Week {week}")
    fig_muni = plot_municipality_detail_grids(bundle, selected_code, week, log_df)
    st.pyplot(fig_muni)
    plt.close(fig_muni)

    # Key municipality metrics bar
    zones_meta = bundle["zones_meta"]
    areas_meta = bundle["areas_meta"]
    victims_by_muni = bundle["victims_by_municipality"]
    zones_end = bundle.get("zones", pd.DataFrame())

    m_zones_ids = zones_meta.index[zones_meta["municipality_code"] == selected_code].tolist()
    m_areas_ids = areas_meta.index[areas_meta["municipality_code"] == selected_code].tolist()
    status_w = area_status_at_week(bundle, week).reindex(m_areas_ids)
    z_state_w = zone_state_at_week(bundle, week).reindex(m_zones_ids)

    gate_choice_label = "—"
    if not gate_df.empty and "municipality_code" in gate_df.columns:
        g_row = gate_df[gate_df["municipality_code"] == selected_code]
        if not g_row.empty:
            gate_choice_label = str(g_row.iloc[0].get("chosen_label", g_row.iloc[0].get("chosen", "—")))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Zones Declared Free", f"{int((z_state_w == 'free').sum())} of {len(m_zones_ids)}")
    k2.metric(
        "Areas Released",
        int(status_w.isin(["cancelled", "reduced", "cleared"]).sum()),
    )
    k3.metric(
        "Suspected / Confirmed Areas",
        f"{int((status_w == 'suspected').sum())} / {int((status_w == 'confirmed').sum())}",
    )
    k4.metric("Recorded Victims (All Years)", f"{int(victims_by_muni.get(selected_name, 0)):,}")
    k5.metric("Selected Risk Method", gate_choice_label)

    # Unified Dashboard Tabs for the selected municipality
    tab_ops, tab_plan, tab_rev, tab_log, tab_sitrep = st.tabs(
        [
            f"Zones & Field Tasks ({selected_name})",
            f"Plan & Forecast ({selected_name})",
            f"Sign-Off Reviews ({selected_name})",
            f"Activity History ({selected_name})",
            f"Week {week} Situation Report",
        ]
    )

    with tab_ops:
        st.markdown(f"**Operational zones in {selected_name}** (status in week {week} and at season's end)")
        if zones_end is not None and not zones_end.empty:
            m_ztable = zones_end.loc[[z for z in m_zones_ids if z in zones_end.index]].copy()
            m_ztable.insert(1, f"State in Week {week}", z_state_w.reindex(m_ztable.index))
            st.dataframe(humanize_zone_table(m_ztable), width="stretch")
        tasks_df = bundle.get("tasks_df", pd.DataFrame())
        if not tasks_df.empty:
            # Tasks ordered by the selected week; weeks after it are left blank (the twin records 0 until set)
            m_tasks = tasks_df[
                (tasks_df["zone"].isin(m_zones_ids)) & (tasks_df["ordered_week"] <= week)
            ].copy()
            st.markdown(f"**Field team tasks ordered by week {week} in {selected_name} ({len(m_tasks)}):**")
            if m_tasks.empty:
                st.write("No field tasks ordered in this municipality on or before this week.")
            else:
                show_t = m_tasks.copy()
                for col in ("started_week", "finished_week"):
                    if col in show_t.columns:
                        show_t[col] = show_t[col].map(
                            lambda w: "" if (pd.isna(w) or int(w) == 0 or int(w) > week) else f"Week {int(w)}")
                if "kind" in show_t.columns:
                    show_t["kind"] = show_t["kind"].map(lambda k: HUMAN_TASK_KIND.get(str(k), str(k)))
                keep = [c for c in ("task_id", "kind", "zone", "area", "team", "ordered_week", "started_week",
                                    "finished_week", "days_planned", "days_done", "days_lost_weather",
                                    "days_lost_access", "status") if c in show_t.columns]
                show_t = show_t[keep].rename(
                    columns={
                        "task_id": "Task #",
                        "kind": "Task Type",
                        "zone": "Zone",
                        "area": "Area",
                        "team": "Team",
                        "ordered_week": "Week Ordered",
                        "started_week": "Week Started",
                        "finished_week": "Week Finished",
                        "days_planned": "Days Planned",
                        "days_done": "Days Worked",
                        "days_lost_weather": "Days Lost to Weather",
                        "days_lost_access": "Days Lost to Access",
                        "status": "Status at Season's End",
                    }
                )
                st.dataframe(show_t, width="stretch", hide_index=True)

    with tab_plan:
        st.caption(
            f"**Recorded victims in {selected_name}: {int(victims_by_muni.get(selected_name, 0)):,}** "
            "(shown beside the plan as municipal context; **not weighed in the integer programme**)."
        )
        plan_rec = bundle.get("plan")
        rule_rec = bundle.get("rule")
        fc_rec = bundle.get("fc")
        risk_rec = bundle.get("risk")

        rows = []
        milp_surv = set(plan_rec.answer.get("survey", [])) if (plan_rec and plan_rec.ok) else set()
        milp_clr = set(plan_rec.answer.get("clearance", [])) if (plan_rec and plan_rec.ok) else set()
        rule_n = rule_first_n(rule_rec.answer if (rule_rec and rule_rec.ok) else {},
                              plan_rec.answer if (plan_rec and plan_rec.ok) else {})
        rule_surv = set(rule_n.get("survey", []))
        rule_clr = set(rule_n.get("clearance", []))
        fc_zones = fc_rec.answer.get("zones", pd.DataFrame()) if (fc_rec and fc_rec.ok) else pd.DataFrame()
        rz = risk_rec.answer.get("zones", pd.DataFrame()) if (risk_rec and risk_rec.ok) else pd.DataFrame()

        for z in m_zones_ids:
            z_row = zones_meta.loc[z]
            p50 = fc_zones.loc[z, "P50 week"] if (not fc_zones.empty and z in fc_zones.index) else np.nan
            p90 = fc_zones.loc[z, "P90 week"] if (not fc_zones.empty and z in fc_zones.index) else np.nan
            p_in = fc_zones.loc[z, "chance this season"] if (not fc_zones.empty and z in fc_zones.index) else np.nan
            exp_m = rz.loc[z, "expected mine areas"] if (not rz.empty and z in rz.index) else np.nan
            insuff_flag = bool(rz.loc[z, "insufficient"]) if (not rz.empty and z in rz.index) else False
            rows.append(
                {
                    "Zone": z,
                    "Total Areas": int(z_row["areas"]),
                    "Road Access": str(z_row["access"]),
                    "Prior Consultation Needed": "Yes" if bool(z_row["consultation"]) else "No",
                    "Expected Mine Areas (Week 1)": round(float(exp_m), 2) if pd.notna(exp_m) else np.nan,
                    "Calibration Note": (
                        "Insufficient evidence (most areas outside the calibrated range)"
                        if insuff_flag else "Within the calibrated range"
                    ),
                    "Integer Programme Plan (Week 1)": (
                        "Survey + clearance"
                        if z in milp_clr
                        else ("Survey only" if z in milp_surv else "Not in the week-1 plan")
                    ),
                    "Rule of Thumb, First N Zones (Week 1)": (
                        "Survey + clearance"
                        if z in rule_clr
                        else ("Survey only" if z in rule_surv else "Not among its first zones")
                    ),
                    "Week-1 Forecast P50 Week (Assumes the Week-1 Plan Holds)": (
                        "Not forecast (outside the forecast's 60 zones)"
                        if (fc_zones.empty or z not in fc_zones.index)
                        else (f"Week {int(p50)}" if pd.notna(p50) else "After the season")
                    ),
                    "Week-1 Forecast P90 Week (Assumes the Week-1 Plan Holds)": (
                        "Not forecast (outside the forecast's 60 zones)"
                        if (fc_zones.empty or z not in fc_zones.index)
                        else (f"Week {int(p90)}" if pd.notna(p90) else "After the season")
                    ),
                    "Chance Completed This Season": f"{float(p_in):.0%}" if pd.notna(p_in) else "—",
                    "Recorded Victims in Municipality (Not Weighed)": int(victims_by_muni.get(selected_name, 0)),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tab_rev:
        rev_df = load_saved_reviews_public(str(root))
        m_rev = rev_df[rev_df["municipality"] == selected_name].copy() if not rev_df.empty else pd.DataFrame()
        if m_rev.empty:
            st.info(f"No area dossier reviews took place in {selected_name} during the season.")
        else:
            st.markdown(
                f"**{len(m_rev)} area dossier review(s) in {selected_name}** "
                "(the agent recommends; in this saved season the QC reviewer's stated sign-off policy, "
                "\"signs off what passes the rule check\", stood in for the person who signs off):"
            )
            st.dataframe(
                humanize_reviews_table(
                    m_rev[
                        [
                            "week",
                            "area_id",
                            "zone_id",
                            "visit",
                            "outcome",
                            "confidence",
                            "decision",
                            "vetoed at sign-off",
                            "model",
                        ]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            rev_options = [
                f"Week {int(r['week']):02d} · Area {r['area_id']} (Visit {int(r['visit'])}) — Final decision: {r['decision']}"
                for _, r in m_rev.iterrows()
            ]
            chosen_idx = st.selectbox(
                f"Open sign-off sheet in {selected_name}:",
                options=range(len(rev_options)),
                format_func=lambda i: rev_options[i],
                key=f"map_rev_{selected_code}",
            )
            sel_row = m_rev.iloc[chosen_idx]
            detail = load_review_detail(
                str(root), int(sel_row["week"]), str(sel_row["area_id"]), int(sel_row["visit"])
            )
            st.markdown(
                '<div class="signoff-banner">Recommendation only: a person decides</div>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown(detail.get("signoff_md", "_Sign-off sheet not found._"))

    with tab_log:
        subjects = set(m_zones_ids) | set(m_areas_ids) | {selected_code, selected_name}
        m_log = log_df[
            (log_df["subject"].isin(subjects)) & (log_df["week"] <= week)
        ].copy()
        st.markdown(
            f"**Log entries whose subject is one of {selected_name}'s zones or areas, through week {week} "
            f"({len(m_log)} entries; task orders and team reports name their task or team instead):**"
        )
        if m_log.empty:
            st.write("No log entries recorded for this municipality on or before the selected week.")
        else:
            st.dataframe(
                humanize_log_table(
                    m_log[
                        [
                            "n",
                            "week",
                            "entry",
                            "kind",
                            "subject",
                            "decision",
                            "before",
                            "after",
                            "decided_by",
                            "recommended_by",
                            "rule_check",
                            "reason",
                        ]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    with tab_sitrep:
        sr = saved_sitreps.get(week)
        if not sr:
            st.info(f"No saved situation report found for week {week}.")
        else:
            rec = sr["recommendation"]
            st.markdown(f"**Week {week} Situation Report**")
            st.write(rec.answer.get("text", ""))
            render_recommendation_box(
                rec,
                title=f"Week {week} situation report check ({rec.answer.get('origin', 'saved draft')})",
                model_name=str(sr["record"].get("model", "")),
            )
