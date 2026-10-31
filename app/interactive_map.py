"""Self-contained local HTML5 Canvas interactive map for exploring mine action across the 15 pilot municipalities.

Complies strictly with all build brief rules:
- Read-only, 100% local and offline (no external scripts, fonts, tiles, or network calls).
- No geographic coordinates, axes, or scale bars ever shown or used (uses the twin's schematic cell grid).
- Shows only programme-visible data (no hidden truth labels).
- Victims shown only as counts by municipality.
"""
from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path
from twin import config
from loaders import load_area_ordnance_findings
import ordnance_library
from grid_map import (
    REGIONAL_CANVAS_H,
    REGIONAL_CANVAS_W,
    SHORT_MUNI_LABELS,
    _build_regional_canvas_index,
    chance_in_force,
)

# Subregion of Antioquia for each pilot municipality: San Roque is in the Nordeste (Wikipedia, "San Roque,
# Antioquia"), Chigorodó in Urabá, Sabanalarga in Occidente, the other twelve in the Oriente
SUBREGION_OF = {"05172": "Urabá", "05628": "Occidente", "05670": "Nordeste"}

# Area status codes for the page: released areas keep codes 3 to 5; "not suspected" (zone surveyed, area not
# flagged) shares code 0 with "unknown", as neither is flagged as hazardous
AREA_CODE = {"unknown": 0, "not_suspected": 0, "suspected": 1, "confirmed": 2, "cancelled": 3, "reduced": 4,
             "cleared": 5}
ZONE_CODE = {"not surveyed": 0, "in survey": 1, "surveyed": 2, "free": 3}


def _build_interactive_payload(
    bundle: dict,
    log_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> dict:
    """Build a compact JSON-serialisable dictionary of the 15 municipalities and the twin's zones, areas and cells."""
    root_path = Path(__file__).resolve().parent.parent
    ord_findings = load_area_ordnance_findings(str(root_path))

    cells_meta = bundle["cells_meta"]
    setup_obj = bundle.get("setup")
    zones_meta = (
        setup_obj.twin.zones
        if (setup_obj is not None and hasattr(setup_obj, "twin") and hasattr(setup_obj.twin, "zones"))
        else bundle["zones_meta"]
    )
    areas_meta = (
        setup_obj.twin.areas
        if (setup_obj is not None and hasattr(setup_obj, "twin") and hasattr(setup_obj.twin, "areas"))
        else bundle["areas_meta"]
    )
    tasks_df = bundle.get("tasks_df", pd.DataFrame())
    victims_by_muni = bundle["victims_by_municipality"]
    zone_state_df = bundle.get("zone_state_df", pd.DataFrame())

    canvas_df = _build_regional_canvas_index(cells_meta)
    pr = canvas_df["plot_r"].to_numpy(dtype=int)
    pc = canvas_df["plot_c"].to_numpy(dtype=int)

    muni_codes = list(config.NAME_OF.keys())
    muni_to_idx = {c: i for i, c in enumerate(muni_codes)}

    zone_ids = list(zones_meta.index)
    zone_to_idx = {z: i for i, z in enumerate(zone_ids)}

    area_ids = list(areas_meta.index)
    area_to_idx = {a: i for i, a in enumerate(area_ids)}

    # The risk overlay's chance of mines for every area: week 1, in force at week 52, and the baseline
    risk_rec = bundle.get("risk")
    _p52, final_risk = chance_in_force(bundle, 52)
    p1_ser = (
        risk_rec.answer["areas"]["chance"].reindex(area_ids).fillna(0.0).astype(float)
        if (risk_rec is not None and risk_rec.ok)
        else pd.Series(0.0, index=area_ids)
    )
    p52_ser = (
        final_risk.answer["areas"]["chance"].reindex(area_ids).fillna(0.0).astype(float)
        if (final_risk is not None and final_risk.ok)
        else p1_ser.copy()
    )
    models_obj = bundle.get("setup").models if bundle.get("setup") is not None else None
    if models_obj is not None:
        pb_ser = models_obj.chances("baseline")["chance"].reindex(area_ids).fillna(0.0).astype(float)
    else:
        pb_ser = p1_ser.copy()

    # Each cell carries its area's own chance (no smoothing)
    cell_area_ids = canvas_df["area_id"]
    cell_sm52 = np.round(p52_ser.reindex(cell_area_ids).to_numpy(dtype=float), 4).tolist()
    cell_sm1 = np.round(p1_ser.reindex(cell_area_ids).to_numpy(dtype=float), 4).tolist()
    cell_smb = np.round(pb_ser.reindex(cell_area_ids).to_numpy(dtype=float), 4).tolist()

    # Week-by-week area statuses and zone states, as the twin recorded them at the end of each week
    # (loaders.RecordingRun), sent as changes: [week, code]
    weekly_a = bundle.get("weekly_area_status") or {}
    weekly_z = bundle.get("weekly_zone_state") or {}
    area_events: dict[int, list[list[int]]] = {}
    zone_events: dict[str, list[list[int]]] = {}
    if weekly_a:
        prev = pd.Series([1 if bool(e) else 0 for e in areas_meta["event_inside"]], index=areas_meta.index)
        for w in sorted(weekly_a):
            cur = weekly_a[w].reindex(areas_meta.index).map(AREA_CODE).fillna(0).astype(int)
            for aid in cur.index[cur != prev]:
                area_events.setdefault(area_to_idx[aid], []).append([int(w), int(cur[aid])])
            prev = cur
    if weekly_z:
        prev_z = pd.Series(0, index=zones_meta.index)
        for w in sorted(weekly_z):
            cur_z = weekly_z[w].reindex(zones_meta.index).map(ZONE_CODE).fillna(0).astype(int)
            for zid in cur_z.index[cur_z != prev_z]:
                zone_events.setdefault(str(zid), []).append([int(w), int(cur_z[zid])])
            prev_z = cur_z

    # Dossier review summaries by area_id
    area_reviews: dict[int, list[dict]] = {}
    if not reviews_df.empty:
        for _, r in reviews_df.iterrows():
            aid = str(r["area_id"])
            if aid in area_to_idx:
                a_idx = area_to_idx[aid]
                area_reviews.setdefault(a_idx, []).append(
                    {
                        "w": int(r["week"]),
                        "v": int(r["visit"]),
                        "out": str(r["outcome"]),
                        "dec": str(r["decision"]),
                        "conf": str(r["confidence"]) if pd.notna(r["confidence"]) else "—",
                        "veto": bool(r.get("vetoed at sign-off", False)),
                    }
                )

    # Median canvas coordinates per area for quick jump & callout pins
    area_med_coords = canvas_df.groupby("area_id")[["plot_r", "plot_c"]].median()

    # Build compact areas array
    areas_list = []
    for idx, aid in enumerate(area_ids):
        arow = areas_meta.loc[aid]
        init_st = 1 if bool(arow["event_inside"]) else 0
        f_info = ord_findings.get(aid, {})
        # Evidence tags: mines found in clearance (log), evidence classes in the reviewer's registers, and a past
        # accident recorded inside (desk study); no device type is assigned
        ot_set = {t for t in f_info.get("tags", []) if t in ("mines_found", "direct", "indirect", "absence")}
        mc_int = int(f_info.get("cleared_cells", 0))
        cn_str = str(f_info.get("clearance_note", ""))
        if bool(arow["event_inside"]):
            ot_set.add("recorded_accident")
        ar_val = float(area_med_coords.loc[aid, "plot_r"]) if aid in area_med_coords.index else 0.0
        ac_val = float(area_med_coords.loc[aid, "plot_c"]) if aid in area_med_coords.index else 0.0
        oq_compact = [
            [int(q["week"]), int(q["reliability_class"]), str(q["tag_label"]), str(q["quote"])]
            for q in f_info.get("quotes", [])
        ]
        areas_list.append(
            {
                "id": aid,
                "m": muni_to_idx[str(arow["municipality_code"])],
                "z": zone_to_idx[str(arow["zone_id"])],
                "r": round(ar_val, 1),
                "c": round(ac_val, 1),
                "km2": round(float(arow["km2"]), 2),
                "lu": str(arow["main_land_use"]),
                "uSh": round(float(arow.get("used_share", 0.0)) * 100.0, 0),
                "fSh": round(float(arow.get("forest_share", 0.0)) * 100.0, 0),
                "pop": int(round(float(arow.get("population", 0.0)))),
                "rKm": round(float(arow.get("road_km", 0.0)), 2),
                "sKm": round(float(arow.get("settlement_km", 0.0)), 2),
                "eKm": round(float(arow.get("event_km", 0.0)), 2),
                "ev": 1 if bool(arow["event_inside"]) else 0,
                "s0": init_st,
                "p52": round(float(p52_ser.iloc[idx]), 4),
                "p1": round(float(p1_ser.iloc[idx]), 4),
                "pb": round(float(pb_ser.iloc[idx]), 4),
                "ch": area_events.get(idx, []),
                "rev": area_reviews.get(idx, []),
                "mc": mc_int,
                "cw": int(f_info.get("cleared_week", 0)),
                "cn": cn_str,
                "ot": sorted(ot_set),
                "oq": oq_compact,
            }
        )

    # Build zones array
    plan_rec = bundle.get("plan")
    rule_rec = bundle.get("rule")
    fc_rec = bundle.get("fc")
    plan_ans = plan_rec.answer if (plan_rec is not None and plan_rec.ok) else {}
    rule_ans = rule_rec.answer if (rule_rec is not None and rule_rec.ok) else {}
    fc_zones = fc_rec.answer.get("zones", pd.DataFrame()) if (fc_rec is not None and fc_rec.ok) else pd.DataFrame()
    rz_w1 = risk_rec.answer.get("zones", pd.DataFrame()) if (risk_rec is not None and risk_rec.ok) else pd.DataFrame()
    rz_w52 = (
        final_risk.answer.get("zones", pd.DataFrame())
        if (final_risk is not None and final_risk.ok)
        else pd.DataFrame()
    )

    milp_surv = set(plan_ans.get("survey", []))
    milp_clr = set(plan_ans.get("clearance", []))
    milp_months = plan_ans.get("months", {})
    n_s_milp = len(plan_ans.get("survey", []))
    n_c_milp = len(plan_ans.get("clearance", []))
    rule_surv = set(rule_ans.get("survey", [])[:n_s_milp])
    rule_clr = set(rule_ans.get("clearance", [])[:n_c_milp])

    # The first week each zone was in survey, from the twin's weekly zone states
    nts_start_by_zone: dict[str, int] = {
        z: next((w for w, code in ev if code >= 1), 0) for z, ev in zone_events.items()
    }

    # The access feed's weekly reports in the log ("access feed: ABE-Z14 closed, SAB-Z02 open" or "no change"),
    # kept as changes per zone: [week, 1 = closed / 0 = open]
    access_events: dict[str, list[list[int]]] = {}
    if log_df is not None and not log_df.empty and {"kind", "reason", "week"} <= set(log_df.columns):
        for _, arow in log_df[log_df["kind"].astype(str) == "access"].iterrows():
            body = str(arow["reason"]).split("access feed:", 1)[-1].strip()
            if not body or body == "no change":
                continue
            for part in body.split(","):
                m_acc = re.match(r"\s*([A-Z]{3}-Z\d+)\s+(open|closed)\s*$", part)
                if m_acc:
                    access_events.setdefault(m_acc.group(1), []).append(
                        [int(arow["week"]), 1 if m_acc.group(2) == "closed" else 0]
                    )

    zones_end = bundle.get("zones", pd.DataFrame())
    zones_list = []
    for idx, zid in enumerate(zone_ids):
        zrow = zones_meta.loc[zid]
        surv_w = 0
        free_w = 0
        acc_status = "open"
        if not zone_state_df.empty and zid in zone_state_df.index:
            surv_w = int(zone_state_df.loc[zid, "surveyed_week"])
            free_w = int(zone_state_df.loc[zid, "free_week"])
        elif zones_end is not None and not zones_end.empty and zid in zones_end.index:
            surv_w = int(zones_end.loc[zid, "surveyed_week"])
            free_w = int(zones_end.loc[zid, "free_week"])
        if zones_end is not None and not zones_end.empty and zid in zones_end.index:
            acc_status = str(zones_end.loc[zid, "access_status"])

        start_w = nts_start_by_zone.get(zid, surv_w if surv_w > 0 else 0)
        m_role = 2 if zid in milp_clr else (1 if zid in milp_surv else 0)
        r_role = 2 if zid in rule_clr else (1 if zid in rule_surv else 0)

        p50_val = (
            int(fc_zones.loc[zid, "P50 week"])
            if (not fc_zones.empty and zid in fc_zones.index and pd.notna(fc_zones.loc[zid, "P50 week"]))
            else 0
        )
        p90_val = (
            int(fc_zones.loc[zid, "P90 week"])
            if (not fc_zones.empty and zid in fc_zones.index and pd.notna(fc_zones.loc[zid, "P90 week"]))
            else 0
        )
        fc_prob = (
            round(float(fc_zones.loc[zid, "chance this season"]), 2)
            if (not fc_zones.empty and zid in fc_zones.index and pd.notna(fc_zones.loc[zid, "chance this season"]))
            else 0.0
        )
        exp1 = (
            round(float(rz_w1.loc[zid, "expected mine areas"]), 2)
            if (not rz_w1.empty and zid in rz_w1.index and pd.notna(rz_w1.loc[zid, "expected mine areas"]))
            else 0.0
        )
        exp52 = (
            round(float(rz_w52.loc[zid, "expected mine areas"]), 2)
            if (not rz_w52.empty and zid in rz_w52.index and pd.notna(rz_w52.loc[zid, "expected mine areas"]))
            else exp1
        )
        m_sched = milp_months.get(zid, {})

        # Median canvas position of zone
        z_cells = canvas_df[canvas_df["zone_id"] == zid]
        zr = float(np.median(z_cells["plot_r"])) if not z_cells.empty else 0.0
        zc = float(np.median(z_cells["plot_c"])) if not z_cells.empty else 0.0

        zones_list.append(
            {
                "id": zid,
                "short": zid.split("-")[-1],
                "m": muni_to_idx[str(zrow["municipality_code"])],
                "na": int(zrow["areas"]),
                "km2": round(float(zrow["km2"]), 2),
                "acc": str(zrow["access"]),
                "acs": acc_status,
                "ach": access_events.get(str(zid), []),
                "fcin": 1 if (not fc_zones.empty and zid in fc_zones.index) else 0,
                "con": bool(zrow["consultation"]),
                "sw": start_w,
                "fw_s": surv_w,
                "fw": free_w,
                "zch": zone_events.get(zid, []),
                "mp": m_role,
                "rp": r_role,
                "p50": p50_val,
                "p90": p90_val,
                "fcp": fc_prob,
                "e1": exp1,
                "e52": exp52,
                "ms": [int(x) for x in m_sched.get("survey", [])],
                "mc": [int(x) for x in m_sched.get("clearance", [])],
                "r": round(zr, 1),
                "c": round(zc, 1),
            }
        )

    # Gate model label per municipality
    gate_by_code = {}
    if not gate_df.empty and "municipality_code" in gate_df.columns:
        for _, grow in gate_df.iterrows():
            gate_by_code[str(grow["municipality_code"])] = str(
                grow.get("chosen_label", grow.get("chosen", "nearness to past accidents"))
            )

    # The national demining register per municipality, from the twin's open demining layer (s.twin.activity)
    setup_obj = bundle.get("setup")
    act_df = (
        setup_obj.twin.activity
        if (setup_obj is not None and hasattr(setup_obj, "twin") and hasattr(setup_obj.twin, "activity"))
        else pd.DataFrame()
    )
    muni_act_totals: dict[str, dict[str, float]] = {}
    if not act_df.empty:
        pilot_act = act_df[act_df["municipality_code"].isin(muni_codes)]
        for code_k, grp_a in pilot_act.groupby("municipality_code"):
            muni_act_totals[str(code_k)] = {
                "map": int(grp_a["items_mines"].sum()),
                "uxo": int(grp_a["items_uxo"].sum()),
                "ied": int(grp_a["items_ieds"].sum()),
                "tot": int(grp_a["items_total"].sum()),
                "ha": int(grp_a["hazardous_areas"].sum()),
                "cha": int(grp_a["confirmed_hazardous_areas"].sum()),
                "m2": round(float(grp_a["area_cleared_m2"].sum()), 0),
            }

    # Municipalities array
    munis_list = []
    for idx, code in enumerate(muni_codes):
        name = config.NAME_OF[code]
        m_cells = canvas_df[canvas_df["municipality_code"] == code]
        r_min = int(m_cells["plot_r"].min())
        r_max = int(m_cells["plot_r"].max())
        c_min = int(m_cells["plot_c"].min())
        c_max = int(m_cells["plot_c"].max())
        med_r = round(float(np.median(m_cells["plot_r"])), 1)
        med_c = round(float(np.median(m_cells["plot_c"])), 1)
        subregion = SUBREGION_OF.get(code, "Oriente")
        act_info = muni_act_totals.get(
            code,
            {"map": 0, "uxo": 0, "ied": 0, "tot": 0, "ha": 0, "cha": 0, "m2": 0.0},
        )
        munis_list.append(
            {
                "idx": idx,
                "code": code,
                "name": name,
                "short": SHORT_MUNI_LABELS.get(code, name).replace("\n", " "),
                "sub": subregion,
                "vic": int(victims_by_muni.get(name, 0)),
                "gate": gate_by_code.get(code, "nearness to past accidents"),
                "nz": int((zones_meta["municipality_code"] == code).sum()),
                "na": int((areas_meta["municipality_code"] == code).sum()),
                "map": act_info["map"],
                "uxo": act_info["uxo"],
                "ied": act_info["ied"],
                "tot": act_info["tot"],
                "ha": act_info["ha"],
                "cha": act_info["cha"],
                "m2": act_info["m2"],
                "bbox": [r_min, c_min, r_max, c_max],
                "r": med_r,
                "c": med_c,
            }
        )

    # Cells compact array: [plot_r, plot_c, area_idx]
    area_idx_arr = canvas_df["area_id"].map(area_to_idx).to_numpy(dtype=int)
    cells_compact = [
        [int(r), int(c), int(a), cell_sm52[i], cell_sm1[i], cell_smb[i]]
        for i, (r, c, a) in enumerate(zip(pr, pc, area_idx_arr))
    ]

    def _p99(values):
        arr = np.asarray(values, dtype=float)
        return float(np.nanpercentile(arr, 99.2)) if np.isfinite(arr).any() else 0.30

    vmax_52, vmax_1, vmax_b = _p99(cell_sm52), _p99(cell_sm1), _p99(cell_smb)

    return {
        "H": REGIONAL_CANVAS_H,
        "W": REGIONAL_CANVAS_W,
        "vmax": {
            "p52": round(max(0.18, min(0.45, vmax_52)), 3),
            "p1": round(max(0.18, min(0.45, vmax_1)), 3),
            "pb": round(max(0.18, min(0.45, vmax_b)), 3),
        },
        "months": [m[:3] for m in config.MONTHS],
        "munis": munis_list,
        "zones": zones_list,
        "areas": areas_list,
        "cells": cells_compact,
    }


@st.cache_data(show_spinner=False)
def _cached_payload_json(
    _bundle: dict,
    _log_df: pd.DataFrame,
    _reviews_df: pd.DataFrame,
    _gate_df: pd.DataFrame,
    cache_version: int = 1,
) -> str:
    """Cache the serialized JSON string of the interactive map payload."""
    payload = _build_interactive_payload(_bundle, _log_df, _reviews_df, _gate_df)
    return json.dumps(payload, separators=(",", ":"))


def render_interactive_country_explorer(
    bundle: dict,
    log_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    initial_week: int = 52,
) -> None:
    """Render the interactive HTML5 Canvas map explorer for the 15 pilot municipalities."""
    data_json = _cached_payload_json(bundle, log_df, reviews_df, gate_df, cache_version=11)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --bg: #faf9f5;
    --panel: #ffffff;
    --panel-alt: #f3f1ea;
    --border: #dcd9cf;
    --ink: #181816;
    --ink-2: #52514e;
    --muted: #85827a;
    --blue: #235789;
    --green: #2d7d53;
    --teal: #0d8a72;
    --orange: #d97236;
    --red: #c44536;
    --gold: #d9a73e;
  }}
  * {{
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  body {{
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--ink);
    user-select: none;
  }}
  .explorer-shell {{
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  /* Top Control Bar */
  .top-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 14px;
    background: var(--panel-alt);
    border-bottom: 1px solid var(--border);
  }}
  .layer-group {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
  }}
  .layer-btn {{
    border: 1px solid #c9c5b9;
    background: #ffffff;
    color: var(--ink);
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.12s ease;
  }}
  .layer-btn:hover {{
    background: #eef3f8;
    border-color: var(--blue);
  }}
  .layer-btn.active {{
    background: var(--blue);
    color: #ffffff;
    border-color: var(--blue);
  }}
  .week-controls {{
    display: flex;
    align-items: center;
    gap: 8px;
    background: #ffffff;
    padding: 4px 10px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }}
  .play-btn {{
    border: none;
    background: var(--green);
    color: #ffffff;
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }}
  .play-btn.playing {{
    background: var(--red);
  }}
  .week-slider {{
    width: 135px;
    cursor: pointer;
  }}
  .week-badge {{
    font-size: 12.5px;
    font-weight: 700;
    min-width: 78px;
    color: var(--ink);
  }}
  /* Second Sub-toolbar: Sub-options + Municipality Quick Jump */
  .sub-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 14px;
    background: #f8f7f2;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
  }}
  .sub-left, .sub-right {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
  }}
  select.ctrl-select {{
    padding: 4px 8px;
    border-radius: 5px;
    border: 1px solid #c9c5b9;
    background: #ffffff;
    font-size: 12px;
    color: var(--ink);
    font-weight: 500;
  }}
  label.chk-label {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    font-weight: 500;
    color: var(--ink-2);
  }}
  /* Main 2-column body: Map Canvas Left + Live Inspector Right */
  .main-grid {{
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    min-height: 610px;
  }}
  @media (max-width: 980px) {{
    .main-grid {{
      grid-template-columns: 1fr;
    }}
  }}
  .canvas-wrap {{
    position: relative;
    background: var(--bg);
    border-right: 1px solid var(--border);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  #mapCanvas {{
    width: 100%;
    height: 555px;
    display: block;
    cursor: grab;
  }}
  #mapCanvas:active {{
    cursor: grabbing;
  }}
  /* Floating Zoom & Country Context Overlay on Canvas */
  .zoom-Readout {{
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    gap: 4px;
    z-index: 10;
  }}
  .z-btn {{
    background: rgba(255,255,255,0.94);
    border: 1px solid #c5c1b6;
    border-radius: 6px;
    padding: 4px 9px;
    font-size: 12px;
    font-weight: 700;
    color: var(--ink);
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .z-btn:hover {{
    background: #eef3f8;
    border-color: var(--blue);
  }}
  /* Colombia -> Antioquia Mini Context Locator Card */
  .country-card {{
    position: absolute;
    bottom: 54px;
    left: 10px;
    width: 192px;
    background: rgba(250, 249, 245, 0.95);
    border: 1px solid #c5c1b6;
    border-radius: 8px;
    padding: 8px 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    z-index: 9;
  }}
  .country-card-title {{
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--ink);
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .region-pill-row {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-top: 4px;
  }}
  .region-pill {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10.5px;
    padding: 3px 6px;
    border-radius: 4px;
    background: #ffffff;
    border: 1px solid #dfdco3;
    cursor: pointer;
    color: var(--ink);
  }}
  .region-pill:hover {{
    border-color: var(--blue);
    background: #f0f5fa;
  }}
  /* Floating Hover Tooltip */
  #hoverTip {{
    position: absolute;
    pointer-events: none;
    background: rgba(24, 24, 22, 0.93);
    color: #ffffff;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    line-height: 1.38;
    max-width: 265px;
    z-index: 20;
    display: none;
    box-shadow: 0 3px 10px rgba(0,0,0,0.22);
  }}
  /* Legend footer under canvas */
  .legend-bar {{
    min-height: 52px;
    padding: 8px 12px;
    background: #f5f3ec;
    border-top: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: 11.5px;
  }}
  .leg-items {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
  }}
  .leg-swatch {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }}
  .swatch-box {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
    display: inline-block;
    border: 1px solid rgba(0,0,0,0.18);
  }}
  .grad-bar {{
    width: 120px;
    height: 10px;
    border-radius: 4px;
    background: linear-gradient(90deg, #fbf6e9, #fee391, #fec44f, #fe9929, #ec7014, #cc4c02, #8c2d04);
    border: 1px solid rgba(0,0,0,0.2);
    display: inline-block;
  }}
  /* Right-Hand Inspector Panel */
  .inspector {{
    padding: 12px 14px;
    background: var(--panel);
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 610px;
    overflow-y: auto;
  }}
  .insp-card {{
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    background: #fcfbf8;
  }}
  .insp-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }}
  .insp-title {{
    font-size: 14px;
    font-weight: 700;
    color: var(--ink);
  }}
  .insp-sub {{
    font-size: 11.5px;
    color: var(--ink-2);
  }}
  .badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 700;
  }}
  .kv-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 10px;
    margin-top: 6px;
    font-size: 12px;
  }}
  .kv-item {{
    background: #ffffff;
    border: 1px solid #e7e4dc;
    border-radius: 6px;
    padding: 5px 8px;
  }}
  .kv-lbl {{
    font-size: 10.5px;
    color: var(--muted);
    font-weight: 600;
  }}
  .kv-val {{
    font-size: 13px;
    font-weight: 700;
    color: var(--ink);
    margin-top: 1px;
  }}
  .zone-table-wrap {{
    max-height: 195px;
    overflow-y: auto;
    border: 1px solid #e5e2d9;
    border-radius: 6px;
    background: #ffffff;
    margin-top: 6px;
  }}
  table.z-tbl {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11.5px;
  }}
  table.z-tbl th {{
    position: sticky;
    top: 0;
    background: #f3f1ea;
    color: var(--ink-2);
    text-align: left;
    padding: 5px 7px;
    font-size: 10.5px;
    border-bottom: 1px solid var(--border);
  }}
  table.z-tbl td {{
    padding: 4px 7px;
    border-bottom: 1px solid #f0eee8;
    cursor: pointer;
  }}
  table.z-tbl tr:hover td {{
    background: #eef4fa;
  }}
  table.z-tbl tr.selected-row td {{
    background: #dceaf7;
    font-weight: 600;
  }}
  .month-strip {{
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 2px;
    margin-top: 6px;
  }}
  .m-cell {{
    text-align: center;
    font-size: 9.5px;
    padding: 3px 1px;
    border-radius: 3px;
    background: #efede6;
    color: var(--ink-2);
    font-weight: 600;
  }}
</style>
</head>
<body>
<div class="explorer-shell">
  <!-- Top Layer & Timeline Bar -->
  <div class="top-bar">
    <div class="layer-group">
      <button class="layer-btn active" data-layer="heat_ops">🔥 Chance of Mines + Released Areas</button>
      <button class="layer-btn" data-layer="ordnance_dist">💣 National Register (MAP / MUSE / AEI)</button>
      <button class="layer-btn" data-layer="land_use">🌾 Land Use (Released Areas in Full Colour)</button>
      <button class="layer-btn" data-layer="road_access">🛣️ Road Access &amp; Consultation</button>
      <button class="layer-btn" data-layer="area_status">🚜 Area Clearance Status</button>
      <button class="layer-btn" data-layer="zone_state">🏁 Zone Release State</button>
      <button class="layer-btn" data-layer="plan_milp">📅 Optimised Season Plan</button>
      <button class="layer-btn" data-layer="forecast_p50">⏳ Forecast Completion (P50)</button>
    </div>
    <div class="week-controls">
      <button id="playBtn" class="play-btn">▶ Play Season</button>
      <input id="weekSlider" class="week-slider" type="range" min="1" max="52" value="{int(initial_week)}" />
      <span id="weekBadge" class="week-badge">Week {int(initial_week)} / 52</span>
    </div>
  </div>

  <!-- Secondary Controls: Heat Map Source, Contours, Municipality Jump -->
  <div class="sub-bar">
    <div class="sub-left">
      <span><strong>Chance of mines:</strong></span>
      <select id="heatSourceSel" class="ctrl-select">
        <option value="p52">In force at week 52 (season end; updated at each monthly plan)</option>
        <option value="p1">Start of season (week 1)</option>
        <option value="pb">Nearness to past accidents (the baseline)</option>
        <option value="delta">Change, week 52 minus week 1 (monthly update by municipality)</option>
      </select>
      <label class="chk-label">
        <input type="checkbox" id="chkCleanedOutline" checked />
        Draw released areas on top
      </label>
      <label class="chk-label">
        <input type="checkbox" id="chkMuniLabels" checked />
        Municipality labels
      </label>
    </div>
    <div class="sub-right">
      <span><strong>Jump to municipality:</strong></span>
      <select id="muniJumpSel" class="ctrl-select">
        <option value="-1">All 15 Pilot Municipalities (Full Region)</option>
      </select>
      <button id="clearPinBtn" class="layer-btn" style="padding:3px 8px;">Unpin Selection</button>
    </div>
  </div>

  <!-- Ordnance, Sub-Type & Multi-Criteria Spatial Query Bar -->
  <div class="sub-bar" style="background:#f3efe4;border-bottom:1px solid #d5d0c3;">
    <div class="sub-left">
      <span><strong>🔍 Evidence:</strong></span>
      <select id="ordSearchSel" class="ctrl-select" style="border-color:#9b5b28;font-weight:600;max-width:300px;">
        <option value="all">All areas (no filter)</option>
        <optgroup label="What the twin and the saved runs show about an area">
          <option value="mines_found" data-tag="mines_found">🎯 Mines found in clearance (log)</option>
          <option value="direct" data-tag="direct">Direct evidence in the reviewer's register</option>
          <option value="indirect" data-tag="indirect">Indirect evidence in the reviewer's register</option>
          <option value="absence" data-tag="absence">Evidence of absence in the reviewer's register</option>
          <option value="recorded_accident" data-tag="recorded_accident">📋 Past accident recorded inside (desk study)</option>
        </optgroup>
        <optgroup label="National demining register by municipality (items destroyed, 2008 to 2023)">
          <option value="reg_map" data-reg="map">MAP: anti-personnel mines</option>
          <option value="reg_muse" data-reg="uxo">MUSE: unexploded ordnance</option>
          <option value="reg_aei" data-reg="ied">AEI: improvised explosive devices</option>
        </optgroup>
      </select>
      <span><strong>Land Use:</strong></span>
      <select id="luFilterSel" class="ctrl-select">
        <option value="all">All Land Uses</option>
        <option value="farming">Farming</option>
        <option value="livestock">Livestock</option>
        <option value="agroforestry">Agroforestry</option>
        <option value="forest">Forest</option>
        <option value="soil conservation">Soil Conservation</option>
      </select>
      <span><strong>Status:</strong></span>
      <select id="stFilterSel" class="ctrl-select">
        <option value="all">All Statuses</option>
        <option value="5">Cleared</option>
        <option value="2">Confirmed</option>
        <option value="1">Suspected</option>
        <option value="rel">Released (cleared, reduced or cancelled)</option>
      </select>
      <span id="ordMatchCountBadge" class="badge" style="background:#e7e2d3;color:#3d3a33;">Showing all areas</span>
    </div>
    <div class="sub-right">
      <span><strong>📍 Area Jump:</strong></span>
      <select id="areaMatchJumpSel" class="ctrl-select" style="max-width:255px;">
        <option value="-1">Select matched area to zoom…</option>
      </select>
      <button id="exportCsvBtn" class="layer-btn" style="padding:4px 9px;background:#235789;color:#fff;border-color:#183d61;">⬇ Export Matched (.csv, restricted)</button>
    </div>
  </div>

  <!-- Main 2-Column Interactive Area -->
  <div class="main-grid">
    <!-- Left Column: Interactive HTML5 Map Canvas -->
    <div class="canvas-wrap">
      <canvas id="mapCanvas" width="820" height="555"></canvas>
      <div id="hoverTip"></div>

      <!-- Floating Zoom Controls -->
      <div class="zoom-Readout">
        <button class="z-btn" id="zoomInBtn" title="Zoom in">＋</button>
        <button class="z-btn" id="zoomOutBtn" title="Zoom out">－</button>
        <button class="z-btn" id="zoomResetBtn" title="Reset full regional view">Reset View</button>
      </div>

      <!-- Schematic Colombia / Antioquia Sub-Region Locator -->
      <div class="country-card">
        <div class="country-card-title">
          <span>Colombia · Antioquia Pilot</span>
          <span style="font-weight:500;color:#6e6b63;font-size:9.5px;">No coordinates</span>
        </div>
        <div style="font-size:10.5px;color:#52514e;line-height:1.3;">
          Click a sub-region or any municipality on the map to zoom &amp; inspect:
        </div>
        <div class="region-pill-row">
          <div class="region-pill" data-sub="all">
            <span><strong>Full Pilot Region</strong></span>
            <span>15 munis · {len(bundle["zones_meta"])} zones</span>
          </div>
          <div class="region-pill" data-sub="east">
            <span>• Oriente and San Roque</span>
            <span>12 Oriente + San Roque (Nordeste)</span>
          </div>
          <div class="region-pill" data-sub="uraba">
            <span>• Urabá Sub-Region</span>
            <span>Chigorodó</span>
          </div>
          <div class="region-pill" data-sub="occ">
            <span>• Occidente Sub-Region</span>
            <span>Sabanalarga</span>
          </div>
        </div>
      </div>

      <!-- Dynamic Legend Footer -->
      <div class="legend-bar" id="legendBar"></div>
    </div>

    <!-- Right Column: Live Interactive Inspector -->
    <div class="inspector">
      <!-- 1. National / Regional Summary Strip at Selected Week -->
      <div class="insp-card">
        <div class="insp-header">
          <div>
            <div class="insp-title" id="natTitle">15 Pilot Municipalities — Week {int(initial_week)}</div>
            <div class="insp-sub">Drag the week slider or press Play ▶ to watch the season unfold</div>
          </div>
          <span class="badge" style="background:#e9f3ed;color:#234734;" id="natFreeBadge">— Free</span>
        </div>
        <div class="kv-grid">
          <div class="kv-item">
            <div class="kv-lbl">Zones Declared Free</div>
            <div class="kv-val" id="kpiZonesFree">—</div>
          </div>
          <div class="kv-item">
            <div class="kv-lbl">Areas Released (Cleared)</div>
            <div class="kv-val" id="kpiAreasRel">—</div>
          </div>
          <div class="kv-item">
            <div class="kv-lbl">Dossier Reviews Signed Off</div>
            <div class="kv-val" id="kpiHotspots">—</div>
          </div>
          <div class="kv-item">
            <div class="kv-lbl">Suspected / Confirmed Areas</div>
            <div class="kv-val" id="kpiActiveHazard">—</div>
          </div>
        </div>
      </div>

      <!-- 2. Hovered / Pinned Area & Zone Inspector Card -->
      <div class="insp-card" id="focusCard">
        <div class="insp-header">
          <div>
            <div class="insp-title" id="focusTitle">Hover or click any area on the map</div>
            <div class="insp-sub" id="focusSub">Click to pin an area, zone, or municipality for detailed inspection</div>
          </div>
          <span class="badge" id="focusBadge" style="background:#eee;color:#333;">Ready</span>
        </div>
        <div id="focusBody" style="font-size:12px;color:#52514e;line-height:1.45;">
          Move your cursor over the map to inspect an area: its status in the selected week, the risk model's
          chance of mines (where to look first; not evidence about the area), its zone's progress and schedule,
          and its sign-off reviews.
        </div>
      </div>

      <!-- 3. Selected Municipality Breakdown & Clickable Zone Table -->
      <div class="insp-card">
        <div class="insp-header">
          <div>
            <div class="insp-title" id="muniCardTitle">Municipality Zones &amp; Progress</div>
            <div class="insp-sub" id="muniCardSub">Click any zone row below to highlight &amp; zoom on the map</div>
          </div>
        </div>
        <div class="zone-table-wrap">
          <table class="z-tbl">
            <thead>
              <tr>
                <th>Zone</th>
                <th>State (W<span id="tblWkHdr">{int(initial_week)}</span>)</th>
                <th>Released</th>
                <th>Expected Mine Areas (Wk 52)</th>
                <th>Plan (Wk 1)</th>
                <th>P50 Wk (Wk-1 Fcst)</th>
              </tr>
            </thead>
            <tbody id="zoneTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};

// Status & colour constants
const AREA_STATUS_NAMES = ["Unknown or not suspected", "Suspected", "Confirmed", "Cancelled (released)", "Reduced (released)", "Cleared (released)"];
const AREA_STATUS_COLORS = ["#e6e4dd", "#f09436", "#cf3a2b", "#8fd1a8", "#48b078", "#1b6ca8"];
const ZONE_STATE_NAMES = ["Not surveyed", "In survey", "Surveyed (open)", "Declared free"];
const ZONE_STATE_COLORS = ["#e8e5dc", "#f5e4be", "#8cb8e8", "#3b9a72"];
const PLAN_ROLE_NAMES = ["Not in initial plan", "Survey only", "Survey + clearance"];
const PLAN_ROLE_COLORS = ["#e6e4dd", "#9ecae1", "#2171b5"];

// Colour ramp for the risk model's chance of mines
const HEAT_STOPS = [
  [0.00, [251, 246, 233]],
  [0.18, [254, 227, 145]],
  [0.36, [254, 196, 79]],
  [0.54, [254, 153, 41]],
  [0.72, [236, 112, 20]],
  [0.88, [204, 76, 2]],
  [1.00, [140, 45, 4]]
];

function sampleRamp(stops, t) {{
  const c = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {{
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (c <= t1) {{
      const u = (c - t0) / Math.max(1e-6, t1 - t0);
      return [
        Math.round(c0[0] + u * (c1[0] - c0[0])),
        Math.round(c0[1] + u * (c1[1] - c0[1])),
        Math.round(c0[2] + u * (c1[2] - c0[2]))
      ];
    }}
  }}
  return stops[stops.length - 1][1];
}}

function hexToRgb(hex) {{
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}}

const AREA_RGB = AREA_STATUS_COLORS.map(hexToRgb);
const ZONE_RGB = ZONE_STATE_COLORS.map(hexToRgb);
const PLAN_RGB = PLAN_ROLE_COLORS.map(hexToRgb);
const LAND_USE_RGB = {{
  "farming": hexToRgb("#d97706"),
  "livestock": hexToRgb("#ca8a04"),
  "agroforestry": hexToRgb("#65a30d"),
  "forest": hexToRgb("#166534"),
  "soil conservation": hexToRgb("#0d9488"),
  "water": hexToRgb("#0284c7"),
  "urban": hexToRgb("#7c3aed")
}};
const ACCESS_RGB = {{
  "under 2 km from a road": hexToRgb("#2d8a68"),
  "2 to 5 km from a road": hexToRgb("#d99429"),
  "more than 5 km from a road": hexToRgb("#b84a39")
}};

// Build fast lookup grids (H x W)
const H = DATA.H;
const W = DATA.W;
const cellAreaGrid = new Int32Array(H * W).fill(-1);
const cellZoneGrid = new Int32Array(H * W).fill(-1);
const cellMuniGrid = new Int32Array(H * W).fill(-1);

for (let i = 0; i < DATA.cells.length; i++) {{
  const [r, c, aIdx] = DATA.cells[i];
  const pos = r * W + c;
  cellAreaGrid[pos] = aIdx;
  const area = DATA.areas[aIdx];
  cellZoneGrid[pos] = area.z;
  cellMuniGrid[pos] = area.m;
}}

// Populate municipality jump select
const muniJumpSel = document.getElementById("muniJumpSel");
DATA.munis.forEach(m => {{
  const opt = document.createElement("option");
  opt.value = String(m.idx);
  opt.textContent = `${{m.name}} (${{m.nz}} zones, ${{m.vic.toLocaleString()}} victims)`;
  muniJumpSel.appendChild(opt);
}});

// Interactive state
let state = {{
  layer: "heat_ops",
  ordFilter: "all",
  luFilter: "all",
  stFilter: "all",
  week: {int(initial_week)},
  heatSource: "p52",
  showCleanedOutline: true,
  showMuniLabels: true,
  playing: false,
  playTimer: null,
  camX: W / 2,
  camY: H / 2,
  zoom: 1.95,
  hoverAreaIdx: -1,
  pinnedAreaIdx: -1,
  selectedMuniIdx: 9 // San Carlos default for zone table
}};

// Check if an area satisfies the combinable Land Use and Area Status filters
function passesExtraFilters(aIdx, a) {{
  if (state.luFilter !== "all" && a.lu !== state.luFilter) return false;
  if (state.stFilter !== "all") {{
    const st = curAreaStatus[aIdx];
    if (state.stFilter === "rel") {{
      if (st < 3) return false;
    }} else if (st !== parseInt(state.stFilter, 10)) {{
      return false;
    }}
  }}
  return true;
}}

// Does an area match the selected evidence or register filter? (no device type is assigned to any area)
const REG_MAX = {{
  map: Math.max(1, ...DATA.munis.map(m => m.map || 0)),
  uxo: Math.max(1, ...DATA.munis.map(m => m.uxo || 0)),
  ied: Math.max(1, ...DATA.munis.map(m => m.ied || 0))
}};
const TAG_STYLE = {{
  mines_found: {{ rgb: [27, 108, 168], col: "#1b6ca8", icon: "🎯" }},
  direct: {{ rgb: [168, 50, 43], col: "#a8322b", icon: "●" }},
  indirect: {{ rgb: [217, 162, 58], col: "#d9a23a", icon: "◐" }},
  absence: {{ rgb: [47, 143, 91], col: "#2f8f5b", icon: "○" }},
  recorded_accident: {{ rgb: [15, 118, 110], col: "#0f766e", icon: "📋" }}
}};
// Whether the access feed showed a zone closed in a week (the log's weekly access reports)
function zoneClosedAt(z, wk) {{
  let closed = 0;
  for (const ev of (z.ach || [])) {{
    if (ev[0] <= wk) closed = ev[1]; else break;
  }}
  return closed === 1;
}}

function getAreaOrdMatch(a, m, filter) {{
  const ot = a.ot || [];
  if (filter.startsWith("reg_")) {{
    const key = filter === "reg_map" ? "map" : (filter === "reg_muse" ? "uxo" : "ied");
    const n = m[key] || 0;
    if (n <= 0) return {{ matched: false, exactPin: false, rgb: [238, 236, 229], pin: "", col: "#999" }};
    const t = Math.log1p(n) / Math.log1p(REG_MAX[key]);
    return {{
      matched: true,
      exactPin: false,
      rgb: [Math.round(248 - t * 70), Math.round(240 - t * 120), Math.round(228 - t * 130)],
      pin: "",
      col: "#9b5b28"
    }};
  }}
  const sty = TAG_STYLE[filter];
  if (!sty) return {{ matched: true, exactPin: false, rgb: [230, 228, 221], pin: "", col: "#235789" }};
  const hit = ot.includes(filter);
  const pin = hit && filter !== "recorded_accident" && filter !== "absence"
    ? (filter === "mines_found" ? `${{sty.icon}} ${{a.id}} (${{a.mc}} cell(s))` : `${{sty.icon}} ${{a.id}}`)
    : "";
  return {{ matched: hit, exactPin: pin !== "", rgb: sty.rgb, pin: pin, col: sty.col }};
}}

const TAG_TEXT = {{
  mines_found: "mines found in clearance",
  direct: "direct evidence",
  indirect: "indirect evidence",
  absence: "evidence of absence",
  recorded_accident: "past accident recorded inside"
}};
function describeAreaFindingsShort(a) {{
  const ot = a.ot || [];
  const parts = ot.filter(t => t !== "mines_found").map(t => TAG_TEXT[t] || t);
  if ((a.mc || 0) > 0) parts.unshift(`${{a.mc}} mine cell(s) found in clearance (W${{a.cw}})`);
  return parts.length > 0 ? parts.join(" · ") : `${{a.lu}} (${{a.km2}} km²)`;
}}

// Add the counts to the filter's options, computed from the data
function labelFilterCounts() {{
  document.querySelectorAll("#ordSearchSel option[data-tag]").forEach(opt => {{
    const tag = opt.getAttribute("data-tag");
    const n = DATA.areas.filter(a => (a.ot || []).includes(tag)).length;
    opt.textContent = `${{opt.textContent}} (${{n.toLocaleString()}} areas)`;
  }});
  document.querySelectorAll("#ordSearchSel option[data-reg]").forEach(opt => {{
    const key = opt.getAttribute("data-reg");
    const n = DATA.munis.reduce((s, m) => s + (m[key] || 0), 0);
    opt.textContent = `${{opt.textContent}} (${{n.toLocaleString()}} items)`;
  }});
}}
labelFilterCounts();

function populateAreaMatchJump() {{
  const sel = document.getElementById("areaMatchJumpSel");
  const badge = document.getElementById("ordMatchCountBadge");
  sel.innerHTML = `<option value="-1">Select matched area to zoom…</option>`;
  const flt = state.ordFilter;

  const candidates = [];
  let totalAreaMatches = 0;
  for (let i = 0; i < DATA.areas.length; i++) {{
    const a = DATA.areas[i];
    if (!passesExtraFilters(i, a)) continue;
    const m = DATA.munis[a.m];
    const ot = a.ot || [];
    const hasSpecificFieldFinding = (a.mc || 0) > 0 || ot.includes("direct") || ot.includes("indirect");
    if (flt === "all") {{
      totalAreaMatches++;
      if (hasSpecificFieldFinding || state.luFilter !== "all" || state.stFilter !== "all") candidates.push(i);
    }} else {{
      const res = getAreaOrdMatch(a, m, flt);
      if (res.matched) {{
        totalAreaMatches++;
        if (res.exactPin || flt === "recorded_accident" || flt === "absence" || hasSpecificFieldFinding) {{
          candidates.push(i);
        }}
      }}
    }}
  }}

  candidates.sort((ia, ib) => {{
    const aa = DATA.areas[ia], ab = DATA.areas[ib];
    if ((ab.mc || 0) !== (aa.mc || 0)) return (ab.mc || 0) - (aa.mc || 0);
    const va = (aa.ot || []).includes("direct") ? 1 : 0;
    const vb = (ab.ot || []).includes("direct") ? 1 : 0;
    if (vb !== va) return vb - va;
    return aa.id.localeCompare(ab.id);
  }});

  const maxList = Math.min(candidates.length, 150);
  for (let k = 0; k < maxList; k++) {{
    const idx = candidates[k];
    const a = DATA.areas[idx];
    const m = DATA.munis[a.m];
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = `${{a.id}} (${{m.name}}) — ${{describeAreaFindingsShort(a)}}`;
    sel.appendChild(opt);
  }}

  if (flt === "all" && state.luFilter === "all" && state.stFilter === "all") {{
    badge.textContent = `All ${{DATA.areas.length.toLocaleString()}} areas (${{candidates.length}} with mines found, or direct or indirect evidence)`;
    badge.style.background = "#e7e2d3";
    badge.style.color = "#3d3a33";
  }} else if (flt.startsWith("reg_")) {{
    const key = flt === "reg_map" ? "map" : (flt === "reg_muse" ? "uxo" : "ied");
    const nM = DATA.munis.filter(mm => (mm[key] || 0) > 0).length;
    badge.textContent = `${{nM}} of ${{DATA.munis.length}} municipalities with such items in the register (areas shaded by their municipality's count)`;
    badge.style.background = "#9b5b28";
    badge.style.color = "#ffffff";
  }} else {{
    badge.textContent = `${{totalAreaMatches.toLocaleString()}} matched areas`;
    badge.style.background = "#235789";
    badge.style.color = "#ffffff";
  }}
}}

function exportMatchedAreasCsv() {{
  const flt = state.ordFilter;
  const rows = [
    ["Area_ID", "Zone_ID", "Municipality", "Land_Use", "Area_km2", "Status_Week_" + state.week, "Mine_Cells_Found_In_Clearance", "Evidence_Tags", "Chance_Week52_Pct", "Chance_Week1_Pct", "Clearance_And_Dossier_Notes_Simulated"]
  ];
  for (let i = 0; i < DATA.areas.length; i++) {{
    const a = DATA.areas[i];
    if (!passesExtraFilters(i, a)) continue;
    const m = DATA.munis[a.m];
    const z = DATA.zones[a.z];
    if (flt !== "all") {{
      const res = getAreaOrdMatch(a, m, flt);
      if (!res.matched) continue;
    }}
    const aSt = curAreaStatus[i];
    const tags = (a.ot || []).join(";");
    const p52 = (a.p52 * 100).toFixed(1);
    const p1 = (a.p1 * 100).toFixed(1);
    const notes = [];
    if (a.cn) notes.push(`W${{a.cw}}: ${{a.cn}}`);
    (a.oq || []).forEach(q => notes.push(`W${{q[0]}} ${{q[2]}}: ${{q[3]}}`));
    const noteStr = notes.join(" | ").replace(/"/g, '""');
    rows.push([
      a.id,
      z.id,
      `"${{m.name}}"`,
      `"${{a.lu}}"`,
      a.km2,
      `"${{AREA_STATUS_NAMES[aSt]}}"`,
      a.mc || 0,
      `"${{tags}}"`,
      p52,
      p1,
      `"${{noteStr}}"`
    ]);
  }}
  const csvContent = ["RESTRICTED: inside the twin; not for public release"].concat(rows.map(r => r.join(","))).join("\\n");
  const blob = new Blob([csvContent], {{ type: "text/csv;charset=utf-8;" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `RESTRICTED_antioquia_areas_${{flt}}_w${{state.week}}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}}

// Pre-compute area status & zone state at state.week
const curAreaStatus = new Uint8Array(DATA.areas.length);
const curZoneState = new Uint8Array(DATA.zones.length);

function recomputeWeekState(week) {{
  for (let i = 0; i < DATA.areas.length; i++) {{
    const a = DATA.areas[i];
    let st = a.s0;
    const ch = a.ch;
    for (let j = 0; j < ch.length; j++) {{
      if (ch[j][0] <= week) st = ch[j][1];
    }}
    curAreaStatus[i] = st;
  }}
  for (let i = 0; i < DATA.zones.length; i++) {{
    const z = DATA.zones[i];
    let zs = 0;
    const zch = z.zch || [];
    for (let j = 0; j < zch.length; j++) {{
      if (zch[j][0] <= week) zs = zch[j][1];
    }}
    curZoneState[i] = zs;
  }}
}}

// Offscreen image buffer (H x W) for crisp fast rendering
const offCanvas = document.createElement("canvas");
offCanvas.width = W;
offCanvas.height = H;
const offCtx = offCanvas.getContext("2d");
const imgData = offCtx.createImageData(W, H);

function renderOffscreenCells() {{
  const buf = imgData.data;
  buf.fill(0);
  const vmax = DATA.vmax[state.heatSource] || 0.30;
  const srcIdx = state.heatSource === "p52" ? 3 : (state.heatSource === "p1" ? 4 : 5);
  const maxMuniTot = Math.max(1, ...DATA.munis.map(m => m.tot || 0));

  for (let i = 0; i < DATA.cells.length; i++) {{
    const cell = DATA.cells[i];
    const r = cell[0];
    const c = cell[1];
    const aIdx = cell[2];
    const a = DATA.areas[aIdx];
    const zIdx = a.z;
    const z = DATA.zones[zIdx];
    const m = DATA.munis[a.m];
    const aSt = curAreaStatus[aIdx];
    const zSt = curZoneState[zIdx];

    let rgb = [230, 228, 221];
    if (state.layer === "heat_ops") {{
      if (state.heatSource === "delta") {{
        const dp = cell[3] - cell[4]; // p52 - p1
        if (dp >= 0) {{
          const tUp = Math.min(1, dp / 0.08);
          rgb = [
            Math.round(243 - tUp * 58),
            Math.round(241 - tUp * 213),
            Math.round(233 - tUp * 205)
          ];
        }} else {{
          const tDn = Math.min(1, (-dp) / 0.06);
          rgb = [
            Math.round(243 - tDn * 228),
            Math.round(241 - tDn * 123),
            Math.round(233 - tDn * 123)
          ];
        }}
      }} else {{
        const hVal = cell[srcIdx];
        rgb = sampleRamp(HEAT_STOPS, hVal / vmax);
      }}
      // Draw the areas released by the selected week on top of the heat map
      if (state.showCleanedOutline && aSt >= 3) {{
        rgb = AREA_RGB[aSt];
      }}
    }} else if (state.layer === "land_use") {{
      const baseLu = LAND_USE_RGB[a.lu] || [120, 120, 120];
      if (aSt >= 3) {{
        rgb = baseLu;
      }} else {{
        rgb = [
          Math.round(baseLu[0] * 0.65 + 245 * 0.35),
          Math.round(baseLu[1] * 0.65 + 243 * 0.35),
          Math.round(baseLu[2] * 0.65 + 235 * 0.35)
        ];
      }}
    }} else if (state.layer === "road_access") {{
      const baseAcc = ACCESS_RGB[z.acc] || [160, 160, 160];
      if (zoneClosedAt(z, state.week)) {{
        rgb = [148, 42, 42]; // closed by the access feed in the selected week
      }} else if (z.con) {{
        rgb = [
          Math.round(baseAcc[0] * 0.78 + 55),
          Math.round(baseAcc[1] * 0.78 + 20),
          Math.round(baseAcc[2] * 0.78 + 55)
        ];
      }} else {{
        rgb = baseAcc;
      }}
    }} else if (state.layer === "ordnance_dist") {{
      // Shade background by log-scaled recorded items in municipality, with active/cleared areas on top
      if (aSt === 2) rgb = [196, 69, 54];      // Confirmed hazard
      else if (aSt === 1) rgb = [240, 148, 54]; // Suspected hazard
      else if (aSt === 5) rgb = [27, 108, 168]; // Released: cleared
      else if (aSt >= 3) rgb = [102, 187, 138]; // Released: cancelled or reduced
      else {{
        const t = Math.log1p(m.tot || 0) / Math.log1p(maxMuniTot);
        rgb = [
          Math.round(244 - t * 20),
          Math.round(241 - t * 65),
          Math.round(232 - t * 115)
        ];
      }}
    }} else if (state.layer === "area_status") {{
      if (aSt === 0) {{
        // Background tinted by zone state
        if (zSt === 3) rgb = [213, 239, 224];
        else if (zSt === 2) rgb = [217, 231, 245];
        else if (zSt === 1) rgb = [245, 228, 190];
        else rgb = [232, 229, 220];
      }} else {{
        rgb = AREA_RGB[aSt];
      }}
    }} else if (state.layer === "zone_state") {{
      rgb = ZONE_RGB[zSt];
    }} else if (state.layer === "plan_milp") {{
      rgb = PLAN_RGB[z.mp];
    }} else if (state.layer === "forecast_p50") {{
      if (z.p50 > 0) {{
        const u = Math.max(0, Math.min(1, (z.p50 - 1) / 51));
        rgb = [
          Math.round(43 + u * (235 - 43)),
          Math.round(131 + u * (190 - 131)),
          Math.round(186 + u * (110 - 186))
        ];
      }} else {{
        rgb = [228, 226, 218];
      }}
    }}

    // If an evidence, register, land use or status filter is active, highlight matched areas and dim others
    if (state.ordFilter !== "all" || state.luFilter !== "all" || state.stFilter !== "all") {{
      const extraOk = passesExtraFilters(aIdx, a);
      if (!extraOk) {{
        rgb = [239, 237, 230];
      }} else if (state.ordFilter !== "all") {{
        const matchRes = getAreaOrdMatch(a, m, state.ordFilter);
        if (matchRes.matched) {{
          rgb = matchRes.rgb;
        }} else {{
          rgb = [239, 237, 230];
        }}
      }}
    }}

    const p = (r * W + c) * 4;
    buf[p] = rgb[0];
    buf[p + 1] = rgb[1];
    buf[p + 2] = rgb[2];
    buf[p + 3] = 255;
  }}
  offCtx.putImageData(imgData, 0, 0);
}}

const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");

function gridToScreen(gx, gy) {{
  const sx = canvas.width / 2 + (gx - state.camX) * state.zoom;
  const sy = canvas.height / 2 + (gy - state.camY) * state.zoom;
  return [sx, sy];
}}

function screenToGrid(sx, sy) {{
  const gx = (sx - canvas.width / 2) / state.zoom + state.camX;
  const gy = (sy - canvas.height / 2) / state.zoom + state.camY;
  return [gx, gy];
}}

function drawBoundaryPass() {{
  const z = state.zoom;
  const ox = canvas.width / 2 - state.camX * z;
  const oy = canvas.height / 2 - state.camY * z;

  const activeAreaIdx = state.pinnedAreaIdx >= 0 ? state.pinnedAreaIdx : state.hoverAreaIdx;
  const activeZoneIdx = activeAreaIdx >= 0 ? DATA.areas[activeAreaIdx].z : -1;
  const activeMuniIdx = activeAreaIdx >= 0 ? DATA.areas[activeAreaIdx].m : state.selectedMuniIdx;

  // Draw pixel-edge boundaries in a single fast pass
  const outerPath = new Path2D();
  const muniPath = new Path2D();
  const zonePath = new Path2D();
  const cleanedPath = new Path2D();
  const clearedPath = new Path2D();
  const activeZonePath = new Path2D();
  const activeAreaPath = new Path2D();

  for (let i = 0; i < DATA.cells.length; i++) {{
    const r = DATA.cells[i][0];
    const c = DATA.cells[i][1];
    const pos = r * W + c;
    const aIdx = cellAreaGrid[pos];
    const zIdx = cellZoneGrid[pos];
    const mIdx = cellMuniGrid[pos];
    const aSt = curAreaStatus[aIdx];
    const isCleaned = aSt >= 3;
    const isCleared = aSt === 5;

    const x0 = ox + c * z;
    const y0 = oy + r * z;
    const x1 = x0 + z;
    const y1 = y0 + z;

    // Top neighbour
    const topPos = r > 0 ? (r - 1) * W + c : -1;
    const topM = topPos >= 0 ? cellMuniGrid[topPos] : -1;
    if (topM === -1) {{
      outerPath.moveTo(x0, y0); outerPath.lineTo(x1, y0);
    }}

    // Left neighbour
    const leftPos = c > 0 ? r * W + (c - 1) : -1;
    const leftM = leftPos >= 0 ? cellMuniGrid[leftPos] : -1;
    if (leftM === -1) {{
      outerPath.moveTo(x0, y0); outerPath.lineTo(x0, y1);
    }}

    // Bottom neighbour
    const botPos = r + 1 < H ? (r + 1) * W + c : -1;
    const botM = botPos >= 0 ? cellMuniGrid[botPos] : -1;
    const botZ = botPos >= 0 ? cellZoneGrid[botPos] : -1;
    const botA = botPos >= 0 ? cellAreaGrid[botPos] : -1;
    const botCleaned = botA >= 0 ? curAreaStatus[botA] >= 3 : false;
    const botCleared = botA >= 0 ? curAreaStatus[botA] === 5 : false;

    if (botM === -1) {{
      outerPath.moveTo(x0, y1); outerPath.lineTo(x1, y1);
    }} else if (botM !== mIdx) {{
      muniPath.moveTo(x0, y1); muniPath.lineTo(x1, y1);
    }} else if (botZ !== zIdx) {{
      zonePath.moveTo(x0, y1); zonePath.lineTo(x1, y1);
    }}

    if (state.showCleanedOutline && (state.layer === "heat_ops" || state.layer === "land_use")) {{
      if (isCleaned !== botCleaned) {{
        cleanedPath.moveTo(x0, y1); cleanedPath.lineTo(x1, y1);
      }}
      if (isCleared !== botCleared) {{
        clearedPath.moveTo(x0, y1); clearedPath.lineTo(x1, y1);
      }}
      if (topM === -1 && isCleaned) {{
        cleanedPath.moveTo(x0, y0); cleanedPath.lineTo(x1, y0);
      }}
    }}

    if (activeZoneIdx >= 0 && ((zIdx === activeZoneIdx) !== (botZ === activeZoneIdx))) {{
      activeZonePath.moveTo(x0, y1); activeZonePath.lineTo(x1, y1);
    }}
    if (activeAreaIdx >= 0 && ((aIdx === activeAreaIdx) !== (botA === activeAreaIdx))) {{
      activeAreaPath.moveTo(x0, y1); activeAreaPath.lineTo(x1, y1);
    }}

    // Right neighbour
    const rightPos = c + 1 < W ? r * W + (c + 1) : -1;
    const rightM = rightPos >= 0 ? cellMuniGrid[rightPos] : -1;
    const rightZ = rightPos >= 0 ? cellZoneGrid[rightPos] : -1;
    const rightA = rightPos >= 0 ? cellAreaGrid[rightPos] : -1;
    const rightCleaned = rightA >= 0 ? curAreaStatus[rightA] >= 3 : false;
    const rightCleared = rightA >= 0 ? curAreaStatus[rightA] === 5 : false;

    if (rightM === -1) {{
      outerPath.moveTo(x1, y0); outerPath.lineTo(x1, y1);
    }} else if (rightM !== mIdx) {{
      muniPath.moveTo(x1, y0); muniPath.lineTo(x1, y1);
    }} else if (rightZ !== zIdx) {{
      zonePath.moveTo(x1, y0); zonePath.lineTo(x1, y1);
    }}

    if (state.showCleanedOutline && (state.layer === "heat_ops" || state.layer === "land_use")) {{
      if (isCleaned !== rightCleaned) {{
        cleanedPath.moveTo(x1, y0); cleanedPath.lineTo(x1, y1);
      }}
      if (isCleared !== rightCleared) {{
        clearedPath.moveTo(x1, y0); clearedPath.lineTo(x1, y1);
      }}
      if (leftM === -1 && isCleaned) {{
        cleanedPath.moveTo(x0, y0); cleanedPath.lineTo(x0, y1);
      }}
    }}

    if (activeZoneIdx >= 0 && ((zIdx === activeZoneIdx) !== (rightZ === activeZoneIdx))) {{
      activeZonePath.moveTo(x1, y0); activeZonePath.lineTo(x1, y1);
    }}
    if (activeAreaIdx >= 0 && ((aIdx === activeAreaIdx) !== (rightA === activeAreaIdx))) {{
      activeAreaPath.moveTo(x1, y0); activeAreaPath.lineTo(x1, y1);
    }}
  }}

  // Stroke zone borders
  ctx.strokeStyle = "rgba(31, 30, 27, 0.22)";
  ctx.lineWidth = Math.max(0.4, Math.min(1.1, z * 0.14));
  ctx.stroke(zonePath);

  // Stroke cleaned / cleared contours on heat map or land_use map
  if (state.showCleanedOutline && (state.layer === "heat_ops" || state.layer === "land_use")) {{
    ctx.strokeStyle = "rgba(13, 138, 114, 0.92)";
    ctx.lineWidth = Math.max(1.1, Math.min(2.4, z * 0.35));
    ctx.stroke(cleanedPath);

    ctx.strokeStyle = "rgba(11, 79, 108, 0.96)";
    ctx.lineWidth = Math.max(1.4, Math.min(2.8, z * 0.45));
    ctx.stroke(clearedPath);
  }}

  // Stroke municipal boundaries (halo + dark line)
  ctx.strokeStyle = "rgba(250, 249, 245, 0.80)";
  ctx.lineWidth = Math.max(1.8, Math.min(3.5, z * 0.55));
  ctx.stroke(muniPath);

  ctx.strokeStyle = "#2b2a27";
  ctx.lineWidth = Math.max(1.0, Math.min(2.0, z * 0.30));
  ctx.stroke(muniPath);

  // Stroke outer perimeter
  ctx.strokeStyle = "#181816";
  ctx.lineWidth = Math.max(1.3, Math.min(2.4, z * 0.38));
  ctx.stroke(outerPath);

  // Highlight active zone & active area
  if (activeZoneIdx >= 0) {{
    ctx.strokeStyle = "#1e56a0";
    ctx.lineWidth = Math.max(1.8, Math.min(3.4, z * 0.50));
    ctx.stroke(activeZonePath);
  }}
  if (activeAreaIdx >= 0) {{
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = Math.max(2.0, Math.min(3.8, z * 0.60));
    ctx.stroke(activeAreaPath);
  }}
}}

function drawInsetFramesAndLabels() {{
  const z = state.zoom;
  const [u0x, u0y] = gridToScreen(4, 6);
  const [u1x, u1y] = gridToScreen(84, 88);
  ctx.save();
  ctx.fillStyle = "#f4f2ea";
  ctx.strokeStyle = "#9c9990";
  ctx.lineWidth = 1.1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.roundRect(u0x, u0y, u1x - u0x, u1y - u0y, 6);
  ctx.fill();
  ctx.stroke();

  const [o0x, o0y] = gridToScreen(14, 100);
  const [o1x, o1y] = gridToScreen(76, 174);
  ctx.beginPath();
  ctx.roundRect(o0x, o0y, o1x - o0x, o1y - o0y, 6);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}}

function drawTextLabels() {{
  const z = state.zoom;
  const [uTx, uTy] = gridToScreen(44, 11);
  const [oTx, oTy] = gridToScreen(45, 105);
  const [eTx, eTy] = gridToScreen(195, 6);

  ctx.save();
  ctx.font = "bold 10px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillStyle = "#52514e";
  ctx.textAlign = "center";
  ctx.fillText("CHIGORODÓ (URABÁ), INSET", uTx, uTy);
  ctx.fillText("SABANALARGA (OCCIDENTE), INSET", oTx, oTy);
  ctx.fillText("ORIENTE (12 MUNICIPALITIES) AND SAN ROQUE (NORDESTE)", eTx, eTy);

  // If zoomed in nicely (z >= 3.2), show short zone labels!
  if (z >= 3.2) {{
    ctx.font = "600 9.5px -apple-system, BlinkMacSystemFont, sans-serif";
    DATA.zones.forEach(zn => {{
      const [zx, zy] = gridToScreen(zn.c + 0.5, zn.r + 0.5);
      if (zx >= 0 && zx <= canvas.width && zy >= 0 && zy <= canvas.height) {{
        ctx.fillStyle = "rgba(255,255,255,0.78)";
        ctx.fillRect(zx - 12, zy - 6, 24, 12);
        ctx.fillStyle = "#181816";
        ctx.fillText(zn.short, zx, zy + 3);
      }}
    }});
  }}

  // Municipality pill labels + Ordnance Category Callouts when on "ordnance_dist" layer
  if (state.showMuniLabels) {{
    ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
    DATA.munis.forEach(m => {{
      const [mx, my] = gridToScreen(m.c + 0.5, m.r + 0.5);
      if (mx < -40 || mx > canvas.width + 40 || my < -20 || my > canvas.height + 20) return;
      const isSel = m.idx === state.selectedMuniIdx;
      const lblText = state.layer === "ordnance_dist" ? `${{m.short}} (${{m.tot || 0}})` : m.short;
      const w = Math.max(ctx.measureText(lblText).width + 10, state.layer === "ordnance_dist" ? 64 : 20);
      const h = state.layer === "ordnance_dist" && (m.tot || 0) > 0 ? 25 : 17;
      ctx.fillStyle = isSel ? "#235789" : "rgba(250, 249, 245, 0.92)";
      ctx.strokeStyle = isSel ? "#143554" : "#9c9990";
      ctx.lineWidth = isSel ? 1.5 : 0.8;
      ctx.beginPath();
      ctx.roundRect(mx - w / 2, my - h / 2, w, h, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = isSel ? "#ffffff" : "#181816";
      ctx.fillText(lblText, mx, my - h / 2 + 12);

      // Draw proportional 3-category bar (MAP / UXO / IED) inside the badge on the ordnance_dist layer
      if (state.layer === "ordnance_dist" && (m.tot || 0) > 0) {{
        const bw = w - 10;
        const bh = 5;
        const bx = mx - bw / 2;
        const by = my + h / 2 - 7.5;
        const wMap = bw * ((m.map || 0) / m.tot);
        const wUxo = bw * ((m.uxo || 0) / m.tot);
        const wIed = bw * ((m.ied || 0) / m.tot);
        ctx.fillStyle = "#a8322b"; // MAP (anti-personnel mines), as in the catalogue
        if (wMap > 0) ctx.fillRect(bx, by, wMap, bh);
        ctx.fillStyle = "#d9a23a"; // MUSE (unexploded ordnance)
        if (wUxo > 0) ctx.fillRect(bx + wMap, by, wUxo, bh);
        ctx.fillStyle = "#3f6aa8"; // AEI (improvised explosive devices)
        if (wIed > 0) ctx.fillRect(bx + wMap + wUxo, by, wIed, bh);
      }}
    }});
  }}

  // Draw high-contrast callout pins for exact area hits when an Ordnance / Finding Type filter is active
  if (state.ordFilter !== "all" || state.layer === "ordnance_dist") {{
    const activeFlt = state.ordFilter !== "all" ? state.ordFilter : "cm";
    ctx.font = "bold 9.5px -apple-system, BlinkMacSystemFont, sans-serif";
    for (let i = 0; i < DATA.areas.length; i++) {{
      const a = DATA.areas[i];
      if (!passesExtraFilters(i, a)) continue;
      const m = DATA.munis[a.m];
      const res = getAreaOrdMatch(a, m, activeFlt);
      if (!res.exactPin || !res.pin) continue;
      const [ax, ay] = gridToScreen(a.c + 0.5, a.r + 0.5);
      if (ax < -30 || ax > canvas.width + 30 || ay < -20 || ay > canvas.height + 20) continue;

      // Target dot on the area
      ctx.beginPath();
      ctx.arc(ax, ay, Math.max(3.5, Math.min(6.0, z * 0.9)), 0, Math.PI * 2);
      ctx.fillStyle = res.col;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();

      // Callout label pill above the dot
      if (state.ordFilter !== "all" || z >= 2.2) {{
        const pw = ctx.measureText(res.pin).width + 8;
        const ph = 14;
        const px = ax - pw / 2;
        const py = ay - ph - 5;
        ctx.fillStyle = res.col;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.roundRect(px, py, pw, ph, 4);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#ffffff";
        ctx.fillText(res.pin, ax, py + 10);
      }}
    }}
  }}
  ctx.restore();
}}

function renderMap() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#faf9f5";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawInsetFramesAndLabels();

  const z = state.zoom;
  const ox = canvas.width / 2 - state.camX * z;
  const oy = canvas.height / 2 - state.camY * z;

  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offCanvas, ox, oy, W * z, H * z);
  ctx.restore();

  drawBoundaryPass();
  drawTextLabels();
}}

function updateLegendBar() {{
  const bar = document.getElementById("legendBar");
  if (state.layer === "heat_ops" && state.heatSource === "delta") {{
    bar.innerHTML = `
      <div class="leg-items">
        <span><strong>Change in the chance of mines, week 52 minus week 1:</strong></span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#0f7676;"></span> lower (full colour at −6 points)</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#f3f1e9;"></span> no change</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#b91c1c;"></span> higher (full colour at +8 points)</span>
      </div>
      <div style="color:#52514e;">The risk overlay rescales each municipality's odds by one multiplier at each monthly plan; a chance is not evidence about an area.</div>
    `;
  }} else if (state.layer === "heat_ops") {{
    const vmaxPct = Math.round((DATA.vmax[state.heatSource] || 0.30) * 100);
    bar.innerHTML = `
      <div class="leg-items">
        <span><strong>Chance of mines:</strong> 0%</span>
        <span class="grad-bar"></span>
        <span>${{vmaxPct}}% or more</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#1b6ca8;"></span> Released: cleared</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#48b078;"></span> Released: reduced</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#8fd1a8;"></span> Released: cancelled</span>
      </div>
      <div style="color:#52514e;">Where to look first, not evidence about an area. Released areas are drawn on top; a released area's chance is not set by its own result.</div>
    `;
  }} else if (state.layer === "ordnance_dist") {{
    bar.innerHTML = `
      <div class="leg-items">
        <span class="leg-swatch"><span class="swatch-box" style="background:#c44536;"></span> Confirmed area</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#f09436;"></span> Suspected area</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#1b6ca8;"></span> Released: cleared</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#66bb8a;"></span> Released: cancelled or reduced</span>
        <span>Other areas shaded by the municipality's register total</span>
      </div>
      <div style="color:#52514e;">National demining register (datos.gov.co jbqm-s5tn): items destroyed by municipality, 2008 to 2023, in its categories MAP, MUSE and AEI. The register does not say where in a municipality they were found.</div>
    `;
  }} else if (state.layer === "land_use") {{
    bar.innerHTML = `
      <div class="leg-items">
        <span class="leg-swatch"><span class="swatch-box" style="background:#d97706;"></span> Farming</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#ca8a04;"></span> Livestock</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#65a30d;"></span> Agroforestry</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#166534;"></span> Forest</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#0d9488;"></span> Soil Cons.</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#0284c7;"></span> Water</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#7c3aed;"></span> Urban</span>
      </div>
      <div style="color:#52514e;">Land use: RELand's cell features (Dulce Rubio et al., 2024) · vivid colour = released land</div>
    `;
  }} else if (state.layer === "road_access") {{
    bar.innerHTML = `
      <div class="leg-items">
        <span class="leg-swatch"><span class="swatch-box" style="background:#2d8a68;"></span> &lt; 2 km from road</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#d99429;"></span> 2–5 km from road</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#b84a39;"></span> &gt; 5 km from road</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#942a2a;"></span> Closed by the access feed (week ${{state.week}})</span>
      </div>
      <div style="color:#52514e;">Zones grouped by position and road access (Mine Action Review, 2025: zones based on accessibility) · tinted: community consultation required (4-week lead time)</div>
    `;
  }} else if (state.layer === "area_status") {{
    bar.innerHTML = `
      <div class="leg-items">
        <span class="leg-swatch"><span class="swatch-box" style="background:#1b6ca8;"></span> Cleared</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#48b078;"></span> Reduced</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#8fd1a8;"></span> Cancelled</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#cf3a2b;"></span> Confirmed</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#f09436;"></span> Suspected</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#d5efe0;"></span> Declared-free zone</span>
      </div>
      <div style="color:#52514e;">Scroll to zoom · Drag to pan · Click any area/zone to pin</div>
    `;
  }} else if (state.layer === "zone_state") {{
    bar.innerHTML = `
      <div class="leg-items">
        <span class="leg-swatch"><span class="swatch-box" style="background:#3b9a72;"></span> Zone declared free</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#8cb8e8;"></span> Surveyed (open)</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#f5e4be;"></span> In survey</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#e8e5dc;"></span> Not surveyed</span>
      </div>
      <div style="color:#52514e;">${{DATA.zones.length}} operational zones across 15 municipalities</div>
    `;
  }} else if (state.layer === "plan_milp") {{
    bar.innerHTML = `
      <div class="leg-items">
        <span class="leg-swatch"><span class="swatch-box" style="background:#2171b5;"></span> Planned: Survey + clearance</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#9ecae1;"></span> Planned: Survey only</span>
        <span class="leg-swatch"><span class="swatch-box" style="background:#e6e4dd;"></span> Not in initial plan</span>
      </div>
      <div style="color:#52514e;">Week-1 integer programme season plan</div>
    `;
  }} else {{
    bar.innerHTML = `
      <div class="leg-items">
        <span><strong>Forecast P50 completion week:</strong> early season (blue) → late season (amber) · grey = no P50 week (after the season, or not among the forecast's 60 zones)</span>
      </div>
      <div style="color:#52514e;">The week-1 forecast of the week-1 plan; it assumes that plan holds</div>
    `;
  }}
}}

function updateInspector() {{
  const wk = state.week;
  document.getElementById("weekBadge").textContent = `Week ${{wk}} / 52`;
  document.getElementById("natTitle").textContent = `15 Pilot Municipalities — Week ${{wk}}`;
  document.getElementById("tblWkHdr").textContent = String(wk);

  // Compute national summary counters at week
  let zFree = 0, zSurv = 0;
  for (let i = 0; i < DATA.zones.length; i++) {{
    if (curZoneState[i] === 3) zFree++;
    else if (curZoneState[i] === 2) zSurv++;
  }}
  let aRel = 0, aClr = 0, aSusp = 0, aConf = 0, nRev = 0;
  for (let i = 0; i < DATA.areas.length; i++) {{
    const st = curAreaStatus[i];
    if (st >= 3) {{
      aRel++;
      if (st === 5) aClr++;
    }} else {{
      if (st === 1) aSusp++;
      if (st === 2) aConf++;
    }}
    nRev += (DATA.areas[i].rev || []).filter(r => r.w <= state.week).length;
  }}

  document.getElementById("natFreeBadge").textContent = `${{zFree}} / ${{DATA.zones.length}} Free`;
  document.getElementById("kpiZonesFree").textContent = `${{zFree}} free (${{zSurv}} surveyed open)`;
  document.getElementById("kpiAreasRel").textContent = `${{aRel}} released (${{aClr}} cleared)`;
  document.getElementById("kpiHotspots").textContent = `${{nRev}} by week ${{state.week}}`;
  document.getElementById("kpiActiveHazard").textContent = `${{aSusp}} suspected · ${{aConf}} confirmed`;

  // Update Focus Card (Pinned or Hovered Area)
  const fIdx = state.pinnedAreaIdx >= 0 ? state.pinnedAreaIdx : state.hoverAreaIdx;
  const fTitle = document.getElementById("focusTitle");
  const fSub = document.getElementById("focusSub");
  const fBadge = document.getElementById("focusBadge");
  const fBody = document.getElementById("focusBody");

  if (fIdx >= 0) {{
    const a = DATA.areas[fIdx];
    const z = DATA.zones[a.z];
    const m = DATA.munis[a.m];
    const aSt = curAreaStatus[fIdx];
    const zSt = curZoneState[a.z];
    const isPinned = state.pinnedAreaIdx >= 0;

    fTitle.textContent = `${{m.name}} · Zone ${{z.id}} · Area ${{a.id}}`;
    fSub.textContent = `${{m.sub}} · Recorded victims in municipality (all years): ${{m.vic.toLocaleString()}} (not weighed in plan)`;
    fBadge.textContent = isPinned ? "📌 Pinned" : AREA_STATUS_NAMES[aSt];
    fBadge.style.background = isPinned ? "#235789" : AREA_STATUS_COLORS[aSt];
    fBadge.style.color = isPinned || aSt === 5 || aSt === 2 ? "#ffffff" : "#181816";

    // Build 12-month schedule strip for this zone
    const sSet = new Set(z.ms);
    const cSet = new Set(z.mc);
    let mStripHtml = `<div class="month-strip">`;
    DATA.months.forEach((mName, idx) => {{
      const mNum = idx + 1;
      const inS = sSet.has(mNum);
      const inC = cSet.has(mNum);
      let bg = "#efede6", col = "#6e6b63";
      if (inS && inC) {{ bg = "#235789"; col = "#fff"; }}
      else if (inC) {{ bg = "#d97236"; col = "#fff"; }}
      else if (inS) {{ bg = "#8cb8e8"; col = "#181816"; }}
      mStripHtml += `<div class="m-cell" style="background:${{bg}};color:${{col}};" title="Month ${{mNum}} (${{mName}})">${{mName}}</div>`;
    }});
    mStripHtml += `</div>`;

    // Check reviews up to week
    const revs = a.rev.filter(r => r.w <= wk);
    let revHtml = `<div style="margin-top:6px;font-size:11.5px;color:#52514e;"><strong>Dossier sign-off reviews through Week ${{wk}}:</strong> None recorded for this area.</div>`;
    if (revs.length > 0) {{
      const lastR = revs[revs.length - 1];
      revHtml = `<div style="margin-top:6px;padding:6px 8px;background:#fff6e5;border-left:3px solid #d48817;border-radius:4px;font-size:11.5px;">
        <strong>Week ${{lastR.w}} Sign-Off Review (Visit ${{lastR.v}}):</strong> Reviewer recommended <code>${{lastR.out}}</code>
        (${{lastR.conf}} confidence) → <strong>Signed off by the QC reviewer (season sign-off policy: signs off what passes the rule check): ${{lastR.dec}}</strong>
        ${{lastR.veto ? " <em>(vetoed by the rule check)</em>" : ""}}
      </div>`;
    }}

    const cleanedNote = aSt >= 3
      ? `<span style="color:#0d8a72;font-weight:700;">Released (${{AREA_STATUS_NAMES[aSt]}})</span>`
      : `<span style="color:#b55016;font-weight:700;">${{AREA_STATUS_NAMES[aSt]}}</span>`;

    const chanceStr = `${{(a.p52 * 100).toFixed(1)}}% at week 52 (week 1: ${{(a.p1 * 100).toFixed(1)}}%)`;

    // Build Ordnance, Clearance & Field Evidence box for this area
    let ordHtml = "";
    const oqList = a.oq || [];
    const hasClrNote = (a.cn || "").length > 0;
    if (hasClrNote || oqList.length > 0 || a.ev === 1) {{
      let itemsHtml = "";
      if (hasClrNote) {{
        const clrCol = (a.mc || 0) > 0 ? "#1b6ca8" : "#2d7d53";
        itemsHtml += `<div style="margin-top:3px;color:${{clrCol}};font-weight:700;">🎯 Week ${{a.cw}} Clearance Outcome: ${{a.cn}}</div>`;
      }}
      oqList.forEach(q => {{
        itemsHtml += `<div style="margin-top:3px;font-size:11px;color:#2c2a26;">• <strong>[W${{q[0]}} · ${{q[2]}}]:</strong> <em>"${{q[3]}}"</em></div>`;
      }});
      if (oqList.length === 0 && a.ev === 1) {{
        itemsHtml += `<div style="margin-top:3px;font-size:11px;color:#2c2a26;">• <strong>[Desk Study · Class 6]:</strong> <em>"The national mine action database records one or more past landmine accidents inside this area."</em></div>`;
      }}
      ordHtml = `<div style="margin-top:6px;padding:6px 8px;background:#fdf2f0;border-left:3px solid #c44536;border-radius:4px;font-size:11.5px;">
        <strong>Area evidence and clearance record (dossier statements are simulated):</strong>
        ${{itemsHtml}}
      </div>`;
    }}

    fBody.innerHTML = `
      <div style="margin-bottom:4px;"><strong>Area Status in Week ${{wk}}:</strong> ${{cleanedNote}}</div>
      <div class="kv-grid">
        <div class="kv-item">
          <div class="kv-lbl">Risk Model's Chance of Mines</div>
          <div class="kv-val">${{chanceStr}}</div>
        </div>
        <div class="kv-item">
          <div class="kv-lbl">Area Land Use &amp; Size</div>
          <div class="kv-val">${{a.lu}} (${{a.km2}} km² · ${{a.uSh || 0}}% of cells farmed, grazed, agroforestry or built on)</div>
        </div>
        <div class="kv-item">
          <div class="kv-lbl">Zone ${{z.short}} State (Week ${{wk}})</div>
          <div class="kv-val">${{ZONE_STATE_NAMES[zSt]}} (${{z.na}} areas)</div>
        </div>
        <div class="kv-item">
          <div class="kv-lbl">Zone P50 / P90 Week (Week-1 Forecast)</div>
          <div class="kv-val">${{!z.fcin ? "Not forecast (outside its 60 zones)" : ((z.p50 > 0 ? "W" + z.p50 : "after the season") + " / " + (z.p90 > 0 ? "W" + z.p90 : "after the season"))}}</div>
        </div>
      </div>
      <div style="margin-top:6px;font-size:11.5px;background:#f7f5ee;padding:5px 8px;border-radius:5px;border:1px solid #e3dfd4;">
        <strong>RELand cell features (Dulce Rubio et al., 2024):</strong>
        Population score (2012 cell estimates, summed; not a count of residents): <strong>${{(a.pop || 0).toLocaleString()}}</strong> ·
        Forest land use: <strong>${{a.fSh || 0}}% of cells</strong> ·
        Mean distance to the nearest road: <strong>${{a.rKm ?? 0}} km</strong> ·
        Mean distance to the nearest settlement: <strong>${{a.sKm ?? 0}} km</strong> ·
        Nearest past accident: <strong>${{a.ev ? "inside the area (under 0.25 km): no cancellation (R6); a target for clearance after the first visit (R3)" : (a.eKm ?? 0) + " km"}}</strong>
      </div>
      <div style="margin-top:5px;font-size:11.5px;">
        <strong>Zone Access &amp; Consultation:</strong> Road distance: <code>${{z.acc}}</code> (access feed, week ${{wk}}: <code>${{zoneClosedAt(z, wk) ? "closed" : "open"}}</code>) ·
        Community consultation (4-week lead time): <code>${{z.con ? "Required" : "No"}}</code> ·
        Week-1 plan: <strong>${{PLAN_ROLE_NAMES[z.mp]}}</strong>
      </div>
      <div style="margin-top:4px;font-size:11px;color:#52514e;"><strong>Planned Monthly Schedule (Jan–Dec, week-1 plan):</strong> (light blue = survey, orange = clearance, dark blue = both)</div>
      ${{mStripHtml}}
      ${{ordHtml}}
      ${{revHtml}}
    `;
  }}

  // Update Municipality Zone Table for state.selectedMuniIdx
  const m = DATA.munis[state.selectedMuniIdx] || DATA.munis[0];
  document.getElementById("muniCardTitle").textContent = `${{m.name}} — ${{m.nz}} Zones`;
  document.getElementById("muniCardSub").textContent = `National demining register (2008–2023): ${{m.map || 0}} MAP · ${{m.uxo || 0}} MUSE · ${{m.ied || 0}} AEI items destroyed · ${{m.cha || 0}} confirmed hazardous areas · Recorded victims (all years): ${{m.vic.toLocaleString()}}`;

  const tbody = document.getElementById("zoneTableBody");
  tbody.innerHTML = "";
  const activeZoneIdx = fIdx >= 0 ? DATA.areas[fIdx].z : -1;

  DATA.zones.forEach((z, zIdx) => {{
    if (z.m !== m.idx) return;
    // Count released areas in this zone at week
    let relInZ = 0;
    for (let i = 0; i < DATA.areas.length; i++) {{
      if (DATA.areas[i].z === zIdx && curAreaStatus[i] >= 3) relInZ++;
    }}
    const tr = document.createElement("tr");
    if (zIdx === activeZoneIdx) tr.className = "selected-row";
    const zSt = curZoneState[zIdx];
    tr.innerHTML = `
      <td><strong>${{z.id}}</strong></td>
      <td>${{ZONE_STATE_NAMES[zSt]}}</td>
      <td>${{relInZ}} / ${{z.na}}</td>
      <td>${{z.e52.toFixed(1)}}</td>
      <td>${{z.mp === 2 ? "Surv+Clr" : (z.mp === 1 ? "Survey" : "—")}}</td>
      <td>${{z.p50 > 0 ? "W" + z.p50 : "—"}}</td>
    `;
    tr.addEventListener("click", () => {{
      // Find first area in this zone and pin it + center camera
      const aIdx = DATA.areas.findIndex(ar => ar.z === zIdx);
      if (aIdx >= 0) {{
        state.pinnedAreaIdx = aIdx;
        state.selectedMuniIdx = m.idx;
        state.camX = z.c;
        state.camY = z.r;
        state.zoom = Math.max(state.zoom, 4.5);
        renderMap();
        updateInspector();
      }}
    }});
    tbody.appendChild(tr);
  }});
}}

// Mouse / Pointer interaction on Canvas
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let camStartX = 0, camStartY = 0;
let movedDuringDrag = false;

canvas.addEventListener("mousedown", e => {{
  isDragging = true;
  movedDuringDrag = false;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  camStartX = state.camX;
  camStartY = state.camY;
}});

window.addEventListener("mouseup", () => {{
  isDragging = false;
}});

canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const sy = (e.clientY - rect.top) * (canvas.height / rect.height);

  if (isDragging) {{
    const dx = (e.clientX - dragStartX) * (canvas.width / rect.width);
    const dy = (e.clientY - dragStartY) * (canvas.height / rect.height);
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) movedDuringDrag = true;
    state.camX = camStartX - dx / state.zoom;
    state.camY = camStartY - dy / state.zoom;
    renderMap();
    return;
  }}

  const [gx, gy] = screenToGrid(sx, sy);
  const c = Math.floor(gx);
  const r = Math.floor(gy);
  const tip = document.getElementById("hoverTip");

  if (r >= 0 && r < H && c >= 0 && c < W) {{
    const aIdx = cellAreaGrid[r * W + c];
    if (aIdx >= 0) {{
      if (state.hoverAreaIdx !== aIdx) {{
        state.hoverAreaIdx = aIdx;
        if (state.pinnedAreaIdx < 0) {{
          state.selectedMuniIdx = DATA.areas[aIdx].m;
        }}
        renderMap();
        updateInspector();
      }}
      const a = DATA.areas[aIdx];
      const z = DATA.zones[a.z];
      const m = DATA.munis[a.m];
      const aSt = curAreaStatus[aIdx];
      const pRaw = state.heatSource === "delta" ? (a.p52 - a.p1) : a[state.heatSource];
      const pVal = (pRaw * 100).toFixed(1);
      const resTxt = state.heatSource === "delta"
        ? `Change in chance, week 52 minus week 1: <strong>${{pVal}} points</strong>`
        : `Risk model's chance of mines: <strong>${{pVal}}%</strong>`;
      tip.style.display = "block";
      tip.style.left = Math.min(canvas.width - 265, e.clientX - rect.left + 14) + "px";
      tip.style.top = Math.max(10, e.clientY - rect.top - 18) + "px";
      tip.innerHTML = `
        <strong>${{m.name}}</strong> · Zone ${{z.short}} · Area ${{a.id}}<br/>
        Status (W${{state.week}}): <strong>${{AREA_STATUS_NAMES[aSt]}}</strong><br/>
        ${{resTxt}}<br/>
        Municipality items found: <strong>${{m.map || 0}} MAP · ${{m.uxo || 0}} UXO · ${{m.ied || 0}} IED</strong>
      `;
      return;
    }}
  }}
  if (state.hoverAreaIdx !== -1) {{
    state.hoverAreaIdx = -1;
    tip.style.display = "none";
    renderMap();
    updateInspector();
  }}
}});

canvas.addEventListener("mouseleave", () => {{
  document.getElementById("hoverTip").style.display = "none";
}});

canvas.addEventListener("click", e => {{
  if (movedDuringDrag) return;
  const rect = canvas.getBoundingClientRect();
  const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
  const [gx, gy] = screenToGrid(sx, sy);
  const c = Math.floor(gx);
  const r = Math.floor(gy);
  if (r >= 0 && r < H && c >= 0 && c < W) {{
    const aIdx = cellAreaGrid[r * W + c];
    if (aIdx >= 0) {{
      state.pinnedAreaIdx = aIdx;
      state.selectedMuniIdx = DATA.areas[aIdx].m;
      muniJumpSel.value = String(state.selectedMuniIdx);
      renderMap();
      updateInspector();
      return;
    }}
  }}
  state.pinnedAreaIdx = -1;
  renderMap();
  updateInspector();
}});

canvas.addEventListener("dblclick", e => {{
  const rect = canvas.getBoundingClientRect();
  const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
  const [gx, gy] = screenToGrid(sx, sy);
  state.camX = gx;
  state.camY = gy;
  state.zoom = Math.min(10.0, state.zoom * 1.85);
  renderMap();
}});

canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
  const [gxBefore, gyBefore] = screenToGrid(sx, sy);
  const factor = e.deltaY < 0 ? 1.18 : 1 / 1.18;
  state.zoom = Math.max(1.3, Math.min(12.0, state.zoom * factor));
  state.camX = gxBefore - (sx - canvas.width / 2) / state.zoom;
  state.camY = gyBefore - (sy - canvas.height / 2) / state.zoom;
  renderMap();
}}, {{ passive: false }});

// Toolbar bindings
document.querySelectorAll(".layer-btn[data-layer]").forEach(btn => {{
  btn.addEventListener("click", () => {{
    document.querySelectorAll(".layer-btn[data-layer]").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.layer = btn.getAttribute("data-layer");
    renderOffscreenCells();
    renderMap();
    updateLegendBar();
  }});
}});

const weekSlider = document.getElementById("weekSlider");
weekSlider.addEventListener("input", () => {{
  state.week = parseInt(weekSlider.value, 10);
  recomputeWeekState(state.week);
  if (state.stFilter !== "all") populateAreaMatchJump();
  renderOffscreenCells();
  renderMap();
  updateInspector();
}});

const playBtn = document.getElementById("playBtn");
playBtn.addEventListener("click", () => {{
  state.playing = !state.playing;
  if (state.playing) {{
    playBtn.textContent = "⏸ Pause";
    playBtn.classList.add("playing");
    if (state.week >= 52) state.week = 1;
    state.playTimer = setInterval(() => {{
      state.week = state.week < 52 ? state.week + 1 : 1;
      weekSlider.value = String(state.week);
      recomputeWeekState(state.week);
      renderOffscreenCells();
      renderMap();
      updateInspector();
      if (state.week === 52) {{
        clearInterval(state.playTimer);
        state.playing = false;
        playBtn.textContent = "▶ Play Season";
        playBtn.classList.remove("playing");
      }}
    }}, 220);
  }} else {{
    clearInterval(state.playTimer);
    playBtn.textContent = "▶ Play Season";
    playBtn.classList.remove("playing");
  }}
}});

document.getElementById("heatSourceSel").addEventListener("change", e => {{
  state.heatSource = e.target.value;
  renderOffscreenCells();
  renderMap();
  updateLegendBar();
  updateInspector();
}});

document.getElementById("chkCleanedOutline").addEventListener("change", e => {{
  state.showCleanedOutline = e.target.checked;
  renderMap();
}});

document.getElementById("chkMuniLabels").addEventListener("change", e => {{
  state.showMuniLabels = e.target.checked;
  renderMap();
}});

muniJumpSel.addEventListener("change", e => {{
  const idx = parseInt(e.target.value, 10);
  if (idx < 0) {{
    state.camX = W / 2;
    state.camY = H / 2;
    state.zoom = 1.95;
  }} else {{
    const m = DATA.munis[idx];
    state.selectedMuniIdx = idx;
    const [r0, c0, r1, c1] = m.bbox;
    state.camX = (c0 + c1) / 2;
    state.camY = (r0 + r1) / 2;
    const span = Math.max(r1 - r0, c1 - c0, 35);
    state.zoom = Math.max(2.5, Math.min(7.0, 380 / span));
  }}
  renderMap();
  updateInspector();
}});

document.getElementById("clearPinBtn").addEventListener("click", () => {{
  state.pinnedAreaIdx = -1;
  renderMap();
  updateInspector();
}});

document.getElementById("zoomInBtn").addEventListener("click", () => {{
  state.zoom = Math.min(12.0, state.zoom * 1.35);
  renderMap();
}});
document.getElementById("zoomOutBtn").addEventListener("click", () => {{
  state.zoom = Math.max(1.3, state.zoom / 1.35);
  renderMap();
}});
document.getElementById("zoomResetBtn").addEventListener("click", () => {{
  state.camX = W / 2;
  state.camY = H / 2;
  state.zoom = 1.95;
  muniJumpSel.value = "-1";
  renderMap();
}});

document.getElementById("ordSearchSel").addEventListener("change", e => {{
  state.ordFilter = e.target.value;
  populateAreaMatchJump();
  renderOffscreenCells();
  renderMap();
  updateInspector();
}});

document.getElementById("luFilterSel").addEventListener("change", e => {{
  state.luFilter = e.target.value;
  populateAreaMatchJump();
  renderOffscreenCells();
  renderMap();
  updateInspector();
}});

document.getElementById("stFilterSel").addEventListener("change", e => {{
  state.stFilter = e.target.value;
  populateAreaMatchJump();
  renderOffscreenCells();
  renderMap();
  updateInspector();
}});

document.getElementById("exportCsvBtn").addEventListener("click", () => {{
  exportMatchedAreasCsv();
}});

document.getElementById("areaMatchJumpSel").addEventListener("change", e => {{
  const aIdx = parseInt(e.target.value, 10);
  if (aIdx >= 0 && aIdx < DATA.areas.length) {{
    const a = DATA.areas[aIdx];
    state.pinnedAreaIdx = aIdx;
    state.selectedMuniIdx = a.m;
    muniJumpSel.value = String(a.m);
    state.camX = a.c;
    state.camY = a.r;
    state.zoom = Math.max(state.zoom, 5.2);
    renderMap();
    updateInspector();
  }}
}});

document.querySelectorAll(".region-pill").forEach(pill => {{
  pill.addEventListener("click", () => {{
    const sub = pill.getAttribute("data-sub");
    if (sub === "all") {{
      state.camX = W / 2; state.camY = H / 2; state.zoom = 1.95;
    }} else if (sub === "east") {{
      state.camX = 192; state.camY = 128; state.zoom = 2.25;
    }} else if (sub === "uraba") {{
      const chig = DATA.munis.find(m => m.code === "05172");
      if (chig) {{ state.selectedMuniIdx = chig.idx; state.camX = chig.c; state.camY = chig.r; state.zoom = 4.6; }}
    }} else if (sub === "occ") {{
      const saba = DATA.munis.find(m => m.code === "05628");
      if (saba) {{ state.selectedMuniIdx = saba.idx; state.camX = saba.c; state.camY = saba.r; state.zoom = 4.8; }}
    }}
    renderMap();
    updateInspector();
  }});
}});

// Initial render
recomputeWeekState(state.week);
populateAreaMatchJump();
renderOffscreenCells();
updateLegendBar();
updateInspector();
renderMap();
</script>
</body>
</html>
"""
    # st.iframe replaces the deprecated st.components.v1.html; the page is built here from the twin's records
    st.iframe(html_doc, height=780)
