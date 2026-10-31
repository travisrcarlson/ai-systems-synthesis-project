"""Operational analytics, visual charts, and the public summary download for the 4 platform upgrades:

1. Rule check (R1–R6), safeguards and agent telemetry (`screen_reviews.py`)
2. Field Team Deployment (S1–S4, C1–C4), Days Lost to Weather and Zone Closures & Clearance Timeline
   (`screen_operations.py`)
3. Bayesian Risk Learning (Week-1 Prior vs. Week-52 Posterior) & Spatial Gate Diagnostics (`screen_plan.py`)
4. Public municipal summary download (`screen_public.py`; `twin.municipal_summary()`: open layers only, by
   municipality)

Read-only: it reads `runs/`, `models/` and the season rebuilt from them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from twin import config
import grid_map
from grid_map import area_status_at_week, zone_state_at_week
from loaders import HUMAN_OUTCOME

SURFACE = "#faf9f5"
INK = "#181816"
INK_2 = "#52514e"
MUTED = "#898781"


# ====================================================================================================
# UPGRADE 1: RULE CHECK (R1–R6), SAFEGUARDS AND AGENT TELEMETRY
# ====================================================================================================

# The rule check's rules, as `twin/rules.py` states them (adapted from P6's R1 to R7 to this project's areas and
# national criteria), and the agent's two other safeguards in code (`twin/agent.py`).
RULE_DEFINITIONS = {
    "R1": {
        "code": "R1",
        "title": "R1 (cancellation): no target for an intrusive check, and weaker indirect evidence explained",
        "standard": "twin/rules.py; IMAS 07.11 §4.4",
        "summary": "There is no direct evidence and no indirect evidence that the survey team saw or a first-hand source "
                   "reported (class 1 to 4), and every item of weaker indirect evidence (class 5 to 7) is explained by "
                   "evidence of absence from a source of class 1 to 4.",
        "colour": "#b91c1c",
    },
    "R2": {
        "code": "R2",
        "title": "R2 (cancellation): the national criteria are met",
        "standard": "twin/rules.py; the pilot's national criteria (twin/config.py)",
        "summary": "Enough different reliable first-hand sources (class 1 to 4) each report enough years of safe use. "
                   "Sources are counted from the twin's record, so a person counts once.",
        "colour": "#0f766e",
    },
    "R3": {
        "code": "R3",
        "title": "R3 (clearance): a target to clear",
        "standard": "twin/rules.py; Mine Action Review (2025): technical survey is not foreseen",
        "summary": "Where the national criteria provide no technical survey, as here, clearance takes its place and "
                   "needs a target (direct evidence, class 1 to 4 indirect evidence, a recorded accident inside after "
                   "the first visit, or weaker evidence still unexplained after the set visits).",
        "colour": "#1d4ed8",
    },
    "R4": {
        "code": "R4",
        "title": "R4 (technical survey): technical survey is available, and there is a target",
        "standard": "twin/rules.py; IMAS 08.20",
        "summary": "Not met in this season: the national criteria provide no technical survey.",
        "colour": "#7c3aed",
    },
    "R5": {
        "code": "R5",
        "title": "R5 (cancellation, technical survey, clearance): every item of the evidence log is in the register",
        "standard": "twin/rules.py",
        "summary": "The reviewer must record every item of the area's evidence log before it can recommend any outcome "
                   "but further survey.",
        "colour": "#6d28d9",
    },
    "R6": {
        "code": "R6",
        "title": "R6 (cancellation): no past accident recorded inside the area",
        "standard": "twin/rules.py; the pilot's national criteria",
        "summary": "Where the national criteria need an intrusive check before an area with a past accident recorded "
                   "inside is released, there is no such record in the desk study.",
        "colour": "#047857",
    },
    "FURTHER_NTS": {
        "code": "-",
        "title": "Further non-technical survey (no rule applies)",
        "standard": "twin/rules.py",
        "summary": "Further non-technical survey is always open, whatever the register holds; the rule check records "
                   "\"-\" for it.",
        "colour": "#b45309",
    },
    "HANDLING": {
        "code": "filter",
        "title": "Handling-advice filter (safeguard in twin/agent.py; rules file section `safety`)",
        "standard": "rules/land_release_rules.md, section safety",
        "summary": "Removes any sentence that advises handling, moving, disarming or destroying an item from the "
                   "reviewer's summary, flags and register notes.",
        "colour": "#c2410c",
    },
    "FALLBACK": {
        "code": "fallback",
        "title": "Step-limit fallback (safeguard in twin/agent.py)",
        "standard": "twin/agent.py",
        "summary": "When no recommendation passes the rule check within the step limit, or a call fails, the area goes "
                   "back for further non-technical survey.",
        "colour": "#6b7280",
    },
}


@st.cache_data(show_spinner=False)
def load_all_review_audit_records(root_str: str) -> pd.DataFrame:
    """Parse all 96 saved `runs/reviews/*_result.json` files into a rich audit DataFrame."""
    reviews_dir = Path(root_str) / "runs" / "reviews"
    if not reviews_dir.is_dir():
        return pd.DataFrame()

    rows = []
    for path in sorted(reviews_dir.glob("*_result.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        aid = str(rec.get("area", ""))
        mcode = aid.split("-")[0] if "-" in aid else ""
        mname = config.NAME_OF.get(mcode, mcode)
        rc = rec.get("rule_check", {}) or {}
        checks = rc.get("checks", []) or []
        rules_checked = [str(c.get("rule", "")) for c in checks if c.get("rule")]
        check_details = " | ".join(
            f"[{c.get('rule', '?')} {'PASS' if c.get('passed') else 'FAIL'}]: {c.get('detail', '')}"
            for c in checks
        )
        reg = rec.get("register") or {}
        reg_items = list(reg.values()) if isinstance(reg, dict) else (reg if isinstance(reg, list) else [])
        # the register records each item's evidence class (twin/rules.py: direct, indirect, absence, not_evidence)
        n_pres = sum(1 for e in reg_items if isinstance(e, dict) and e.get("evidence_class") in ("direct", "indirect"))
        n_abs = sum(1 for e in reg_items if isinstance(e, dict) and e.get("evidence_class") == "absence")
        n_neu = sum(1 for e in reg_items if isinstance(e, dict) and e.get("evidence_class") == "not_evidence")
        flags = rec.get("flags", []) or []
        hb = rec.get("handling_blocked", []) or []
        vetoes = rec.get("vetoes", []) or []
        qf = rec.get("quote_failures", []) or []
        te = rec.get("tool_errors", []) or []
        inj = rec.get("instruction_lines_found", []) or []
        stats = rec.get("stats", {}) or {}
        usage = rec.get("usage", {}) or {}

        rows.append(
            {
                "week": int(rec.get("week", 0)),
                "area_id": aid,
                "municipality_code": mcode,
                "municipality": mname,
                "visit": int(rec.get("visit", 1)),
                "label": str(rec.get("label", "")),
                "outcome": str(rec.get("outcome", "")),
                "outcome_label": HUMAN_OUTCOME.get(str(rec.get("outcome", "")), str(rec.get("outcome", ""))),
                "confidence": str(rec.get("confidence", "")),
                "fallback": bool(rec.get("fallback", False)),
                "rule_check_result": str(rc.get("result", "")),
                "rules_checked_list": rules_checked,
                "rules_checked": ", ".join(rules_checked) if rules_checked else "—",
                "rule_check_details": check_details,
                "evidence_count": len(reg),
                "supports_presence": n_pres,
                "supports_absence": n_abs,
                "neutral_evidence": n_neu,
                "uncertainty_flags_count": len(flags),
                "uncertainty_flags": " | ".join(str(f) for f in flags),
                "handling_blocked_count": len(hb),
                "handling_blocked_text": " | ".join(str(x) for x in hb),
                "vetoes_count": len(vetoes),
                "quote_failures_count": len(qf),
                "tool_errors_count": len(te),
                "injection_lines_count": len(inj),
                "sections_read": ", ".join(rec.get("sections_read", []) or []),
                "rules_read": ", ".join(rec.get("rules_read", []) or []),
                "model_calls": int(stats.get("model_calls", 0)),
                "tool_calls": int(stats.get("tool_calls", 0)),
                "seconds": round(float(stats.get("seconds", 0.0)), 2),
                "input_tokens": int(usage.get("input_tokens", 0)),
                "cached_tokens": int(usage.get("cached_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "cost_usd": round(float(rec.get("cost_usd") or 0.0), 4),
                "summary": str(rec.get("summary", "")),
            }
        )
    return pd.DataFrame(rows)


def plot_safety_rule_and_telemetry_studio(audit_df: pd.DataFrame) -> plt.Figure:
    """Render a 2-panel diagnostic figure for the saved dossier reviews:
    - Left panel: how many reviews each rule of the rule check (R1 to R6) tested, and the safeguards' counts.
    - Right panel: the reviewer's recommendations by confidence, with mean tool calls and time.
    """
    fig, (ax_rules, ax_out) = plt.subplots(1, 2, figsize=(15.6, 5.4), facecolor=SURFACE)
    for ax in (ax_rules, ax_out):
        ax.set_facecolor(SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel 1: Rule checks & safeguard triggers across the 96 reviews
    def tested(rule):
        return int(audit_df["rules_checked_list"].apply(lambda x: rule in x).sum())

    r_items = [
        ("R5: every log item in the register", tested("R5"), "#6d28d9"),
        ("R3: a target to clear", tested("R3"), "#1d4ed8"),
        ("Further survey (no rule applies, \"-\")", tested("-"), "#b45309"),
        ("R1: no target; weaker evidence explained", tested("R1"), "#b91c1c"),
        ("R2: national criteria met", tested("R2"), "#0f766e"),
        ("R6: no past accident recorded inside", tested("R6"), "#047857"),
        ("R4: technical survey (not available)", tested("R4"), "#7c3aed"),
        ("Reviews with flags for the sign-off", int((audit_df["uncertainty_flags_count"] > 0).sum()), "#4338ca"),
        ("Handling-advice filter removed a sentence", int((audit_df["handling_blocked_count"] > 0).sum()), "#c2410c"),
        ("Fallbacks, vetoes or quote failures", int(audit_df["fallback"].sum() + audit_df["vetoes_count"].sum() + audit_df["quote_failures_count"].sum()), "#6b7280"),
    ]
    r_items = list(reversed(r_items))
    y_pos = np.arange(len(r_items))
    vals = [x[1] for x in r_items]
    cols = [x[2] for x in r_items]
    lbls = [x[0] for x in r_items]

    ax_rules.barh(y_pos, vals, color=cols, alpha=0.88, edgecolor=INK, linewidth=0.5)
    for yi, v in enumerate(vals):
        pct = 100.0 * v / max(1, len(audit_df))
        ax_rules.text(v + 1.2, yi, f"{v} reviews ({pct:.0f}%)", va="center", fontsize=8.2, fontweight="bold", color=INK)
    ax_rules.set_yticks(y_pos)
    ax_rules.set_yticklabels(lbls, fontsize=8.4, color=INK)
    ax_rules.set_xlim(0, max(max(vals), 1) * 1.28)
    ax_rules.grid(axis="x", linestyle=":", alpha=0.45)
    ax_rules.set_xlabel(f"Number of saved dossier reviews (out of {len(audit_df)})", fontsize=8.5, color=INK)
    ax_rules.set_title(
        f"1. Rules the Rule Check Tested (R1–R6) and the Safeguards, Across {len(audit_df)} Dossier Reviews",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )

    # Panel 2: Outcome x Confidence stacked bar + mean latency & tool calls annotation
    outcomes_order = ["CLEARANCE", "CANCEL", "FURTHER_NTS", "TECHNICAL_SURVEY"]
    outcomes_present = [o for o in outcomes_order if o in audit_df["outcome"].values]
    conf_levels = [("high", "#15803d"), ("medium", "#d97706"), ("low", "#b91c1c")]

    x_pos = np.arange(len(outcomes_present))
    bottoms = np.zeros(len(outcomes_present))
    for c_name, c_col in conf_levels:
        c_vals = [
            int(((audit_df["outcome"] == o) & (audit_df["confidence"] == c_name)).sum())
            for o in outcomes_present
        ]
        ax_out.bar(
            x_pos,
            c_vals,
            bottom=bottoms,
            color=c_col,
            alpha=0.88,
            edgecolor=INK,
            linewidth=0.5,
            width=0.52,
            label=f"Confidence: {c_name.capitalize()}",
        )
        bottoms += np.array(c_vals, dtype=float)

    for xi, o in enumerate(outcomes_present):
        sub_o = audit_df[audit_df["outcome"] == o]
        tot_o = len(sub_o)
        mean_sec = float(sub_o["seconds"].mean()) if not sub_o.empty else 0.0
        mean_tc = float(sub_o["tool_calls"].mean()) if not sub_o.empty else 0.0
        ax_out.text(
            xi,
            bottoms[xi] + 1.2,
            f"{tot_o} reviews\n({mean_tc:.1f} tools · {mean_sec:.0f}s)",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color=INK,
        )

    ax_out.set_xticks(x_pos)
    ax_out.set_xticklabels([HUMAN_OUTCOME.get(o, o).capitalize() for o in outcomes_present], fontsize=8.4, color=INK)
    ax_out.set_ylim(0, max(bottoms) * 1.26 if len(bottoms) else 10)
    ax_out.grid(axis="y", linestyle=":", alpha=0.45)
    ax_out.set_ylabel("Number of dossier reviews", fontsize=8.5, color=INK)
    ax_out.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#dcd9cf", fontsize=8.0)
    ax_out.set_title(
        "2. Dossier Review Recommendations by Confidence, Tool Calls & Latency",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )

    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.12, wspace=0.25)
    return fig


def render_safety_rule_audit_studio(root: Path) -> None:
    """Render the rule check (R1–R6) and agent safeguard audit inside `screen_reviews.py`."""
    audit_df = load_all_review_audit_records(str(root))
    if audit_df.empty:
        st.info("No saved review JSON files found in `runs/reviews/`.")
        return

    st.subheader("🛡️ Rule Check (R1–R6) and the Agent's Safeguards")
    st.caption(
        f"Reads all **{len(audit_df)} saved dossier reviews** (`runs/reviews/*_result.json`): the saved result of the "
        "coded rule check (`twin.rules.rule_check`, R1 to R6), the **handling-advice filter** (`handling_blocked`), the **quote check** "
        "(`quote_failures`), the **scan for text that addresses the reviewer** (`instruction_lines_found`), and the "
        "**step-limit fallback** (`fallback`). The rule check tests the register, not the world: if an item is classed "
        "wrongly, it checks the wrong class."
    )

    # Top safeguard metrics bar (strictly honest counts from saved files)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Rule Checks Passed", f"{int((audit_df['rule_check_result'] == 'pass').sum())} / {len(audit_df)}")
    m2.metric("Re-Survey (`FURTHER_NTS`)", int((audit_df["outcome"] == "FURTHER_NTS").sum()))
    m3.metric("Sentences Removed by the Handling-Advice Filter", int(audit_df["handling_blocked_count"].sum()))
    m4.metric("Flags for the Person Signing Off", int(audit_df["uncertainty_flags_count"].sum()))
    m5.metric("Fallbacks / Vetoes / Quote Err", f"{int(audit_df['fallback'].sum())} / {int(audit_df['vetoes_count'].sum())} / {int(audit_df['quote_failures_count'].sum())}")
    m6.metric(f"Total Reviewer Cost ({len(audit_df)} reviews)", f"${float(audit_df['cost_usd'].sum()):.2f}")

    fig_audit = plot_safety_rule_and_telemetry_studio(audit_df)
    st.pyplot(fig_audit)
    plt.close(fig_audit)

    # Rule reference cards expander
    with st.expander("📖 The rule check's rules (R1–R6, `twin/rules.py`) and the agent's safeguards", expanded=False):
        rule_rows = []
        for rcode, rinfo in RULE_DEFINITIONS.items():
            if rcode == "HANDLING":
                cnt = int((audit_df["handling_blocked_count"] > 0).sum())
            elif rcode == "FALLBACK":
                cnt = int(audit_df["fallback"].sum())
            elif rcode == "FURTHER_NTS":
                cnt = int(audit_df["rules_checked_list"].apply(lambda x: "-" in x).sum())
            else:
                cnt = int(audit_df["rules_checked_list"].apply(lambda x: rcode in x).sum())
            rule_rows.append(
                {
                    "Rule Code": rinfo["code"],
                    "Rule / Safeguard": rinfo["title"],
                    "Source": rinfo["standard"],
                    f"Reviews (of {len(audit_df)})": cnt,
                    "What It Tests or Does": rinfo["summary"],
                }
            )
        st.dataframe(pd.DataFrame(rule_rows), width="stretch", hide_index=True)

    # Show each review where the handling-advice filter removed a sentence
    hb_df = audit_df[audit_df["handling_blocked_count"] > 0]
    if not hb_df.empty:
        with st.expander(
            f"✂️ Inspect the {len(hb_df)} review(s) where the handling-advice filter removed a sentence (`handling_blocked`)",
            expanded=True,
        ):
            for _, hrow in hb_df.iterrows():
                st.markdown(
                    f"- **Week {int(hrow['week'])} · Area `{hrow['area_id']}` ({hrow['municipality']}, Visit {int(hrow['visit'])})** "
                    f"-> Outcome: `{hrow['outcome']}` · **Removed sentence:** *\"{hrow['handling_blocked_text']}\"*"
                )

    # Filterable Audit Register across all 96 reviews
    fc1, fc2, fc3 = st.columns([1.6, 1.4, 1.5])
    with fc1:
        rule_filter = st.selectbox(
            "Filter reviews by rule or safeguard:",
            options=[
                "ALL",
                "R1",
                "R2",
                "R3",
                "R5",
                "R6",
                "-",
                "HANDLING_BLOCKED",
                "HAS_FLAGS",
            ],
            format_func=lambda x: {
                "ALL": f"All {len(audit_df)} dossier reviews",
                "R1": f"R1: no target; weaker evidence explained ({int(audit_df['rules_checked_list'].apply(lambda l: 'R1' in l).sum())})",
                "R2": f"R2: national criteria met ({int(audit_df['rules_checked_list'].apply(lambda l: 'R2' in l).sum())})",
                "R3": f"R3: a target to clear ({int(audit_df['rules_checked_list'].apply(lambda l: 'R3' in l).sum())})",
                "R5": f"R5: every log item in the register ({int(audit_df['rules_checked_list'].apply(lambda l: 'R5' in l).sum())})",
                "R6": f"R6: no past accident recorded inside ({int(audit_df['rules_checked_list'].apply(lambda l: 'R6' in l).sum())})",
                "-": f"Further survey, no rule applies (\"-\", {int(audit_df['rules_checked_list'].apply(lambda l: '-' in l).sum())})",
                "HANDLING_BLOCKED": f"Handling-advice filter removed a sentence ({int((audit_df['handling_blocked_count'] > 0).sum())})",
                "HAS_FLAGS": f"Flags for the sign-off ({int((audit_df['uncertainty_flags_count'] > 0).sum())})",
            }[x],
            key="audit_studio_rule_filter",
        )
    with fc2:
        muni_filter = st.multiselect(
            "Filter by municipality:",
            options=sorted(audit_df["municipality"].unique().tolist()),
            default=[],
            key="audit_studio_muni_filter",
        )
    with fc3:
        kw_filter = st.text_input(
            "Search Area ID, rule detail, or flag text:",
            value="",
            placeholder="e.g. ECV-0050, R1, E1, woodcutter…",
            key="audit_studio_kw",
        ).strip()

    sub_a = audit_df.copy()
    if rule_filter in ("R1", "R2", "R3", "R5", "R6", "-"):
        sub_a = sub_a[sub_a["rules_checked_list"].apply(lambda l: rule_filter in l)]
    elif rule_filter == "HANDLING_BLOCKED":
        sub_a = sub_a[sub_a["handling_blocked_count"] > 0]
    elif rule_filter == "HAS_FLAGS":
        sub_a = sub_a[sub_a["uncertainty_flags_count"] > 0]

    if muni_filter:
        sub_a = sub_a[sub_a["municipality"].isin(muni_filter)]
    if kw_filter:
        q_low = kw_filter.lower()
        sub_a = sub_a[
            sub_a["area_id"].str.lower().str.contains(q_low, na=False)
            | sub_a["rule_check_details"].str.lower().str.contains(q_low, na=False)
            | sub_a["uncertainty_flags"].str.lower().str.contains(q_low, na=False)
            | sub_a["summary"].str.lower().str.contains(q_low, na=False)
        ]

    show_cols = [
        "week",
        "area_id",
        "municipality",
        "visit",
        "outcome_label",
        "confidence",
        "rules_checked",
        "rule_check_details",
        "evidence_count",
        "supports_presence",
        "supports_absence",
        "uncertainty_flags",
        "handling_blocked_text",
        "tool_calls",
        "seconds",
        "cost_usd",
    ]
    dl_c1, dl_c2 = st.columns([3.5, 1.2])
    with dl_c1:
        st.caption(f"Showing **{len(sub_a)}** of **{len(audit_df)}** saved dossier review audit records.")
    with dl_c2:
        st.download_button(
            label=f"⬇ Download {len(sub_a)} Audit Rows (.csv, restricted)",
            data=sub_a[show_cols].to_csv(index=False).encode("utf-8"),
            file_name="RESTRICTED_dossier_review_audit_register.csv",
            mime="text/csv",
            key="dl_safety_rule_audit_csv",
        )
    st.dataframe(
        sub_a[show_cols].rename(
            columns={
                "week": "Week",
                "area_id": "Area ID",
                "municipality": "Municipality",
                "visit": "Visit #",
                "outcome_label": "Recommendation",
                "confidence": "Confidence",
                "rules_checked": "Rules Checked",
                "rule_check_details": "Rule Check Verbatim Detail",
                "evidence_count": "Evidence Items",
                "supports_presence": "Evidence of EO (Direct or Indirect)",
                "supports_absence": "Evidence of Absence",
                "uncertainty_flags": "Flags for the Sign-Off",
                "handling_blocked_text": "Sentence Removed by the Handling-Advice Filter",
                "tool_calls": "Tool Calls",
                "seconds": "Latency (s)",
                "cost_usd": "Cost ($)",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# ====================================================================================================
# UPGRADE 2: FIELD TEAM DEPLOYMENT (S1–S4, C1–C4), DAYS LOST TO WEATHER AND ZONE CLOSURES & CLEARANCE TIMELINE
# ====================================================================================================

def plot_team_gantt_and_attrition(
    bundle: dict,
    ord_findings: dict[str, dict],
    up_to_week: int = 52,
) -> plt.Figure:
    """Render a 2-panel figure of the 8 field teams (`S1–S4` Survey, `C1–C4` Clearance):
    - Left panel: the tasks started by `up_to_week`, by team and colour-coded by task type (`ZONE_SURVEY`, `VISIT`,
      `CLEARANCE`). Each bar ends at the task's finish week, or before its pause; a task not finished by
      `up_to_week` is hatched. Markers show the clearances whose result in the season log (`CLEARED`) found mine
      cells by `up_to_week`.
    - Right panel: each team's days worked, lost to weather and lost to zone closures, at the season's end (week 52):
      the saved tasks hold each task's days for the whole season only.
    """
    tasks_df = bundle.get("tasks_df", pd.DataFrame()).copy()
    log_df = bundle.get("log_df", pd.DataFrame())
    fig, (ax_gantt, ax_bar) = plt.subplots(
        1, 2, figsize=(16.0, 5.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    for ax in (ax_gantt, ax_bar):
        ax.set_facecolor(SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if tasks_df.empty:
        return fig

    # the tasks a team had started (first day worked) by the selected week
    sub_t = tasks_df[(tasks_df["started_week"] > 0) & (tasks_df["started_week"] <= up_to_week)].copy()
    teams_order = ["S1", "S2", "S3", "S4", "C1", "C2", "C3", "C4"]
    y_map = {t: idx for idx, t in enumerate(reversed(teams_order))}

    kind_colours = {
        "ZONE_SURVEY": "#2563eb",
        "VISIT": "#0d9488",
        "CLEARANCE": "#d97706",
    }

    # Pauses and resumptions from the season log (decisions of kind "pause", and task orders "(resumed)"), up to the
    # selected week. A pause comes with the week's feeds, before its working days, so a paused task's last week of
    # work is the week before its pause.
    pause_weeks: dict[str, list[int]] = {}
    resume_weeks: dict[str, list[int]] = {}
    if not log_df.empty:
        dec = log_df[(log_df["entry"] == "decision") & (log_df["week"] <= up_to_week)]
        for _, r in dec[dec["kind"] == "pause"].iterrows():
            pause_weeks.setdefault(str(r["subject"]), []).append(int(r["week"]))
        resumed = dec[(dec["kind"] == "task order") & dec["decision"].astype(str).str.contains("(resumed)", regex=False)]
        for _, r in resumed.iterrows():
            resume_weeks.setdefault(str(r["subject"]), []).append(int(r["week"]))

    # Draw Gantt bars for each task
    mine_markers_x = []
    mine_markers_y = []
    mined_areas: dict[str, int] = {}
    n_by_kind = {"ZONE_SURVEY": 0, "VISIT": 0}
    n_clr_none = n_clr_mines = n_unfinished = n_unfinished_clr = n_paused = 0

    for _, row in sub_t.iterrows():
        tm = str(row["team"])
        if tm not in y_map:
            continue
        yi = y_map[tm]
        tid = str(row["task_id"])
        w_start = float(row["started_week"])
        f_wk = row["finished_week"]
        finished = pd.notna(f_wk) and 0 < float(f_wk) <= up_to_week
        last_pause = max(pause_weeks.get(tid, []), default=0)
        last_resume = max(resume_weeks.get(tid, []), default=0)
        paused = (not finished) and last_pause > last_resume
        if finished:
            w_end = float(f_wk)
        elif paused:
            w_end = max(w_start, float(last_pause - 1))
            n_paused += 1
        else:
            w_end = float(up_to_week)   # still running in the selected week
        dur = max(0.65, w_end - w_start + 0.85)
        k_str = str(row["kind"])
        col = kind_colours.get(k_str, "#6b7280")
        if k_str in n_by_kind:
            n_by_kind[k_str] += 1

        aid = str(row.get("area", ""))
        if k_str == "CLEARANCE":
            # the clearance's result, from its CLEARED entry in the season log, if signed off by the selected week
            fnd = ord_findings.get(aid, {})
            res_week = int(fnd.get("cleared_week", 0) or 0)
            if 0 < res_week <= up_to_week:
                mc = int(fnd.get("cleared_cells", 0))
                if mc > 0:
                    col = "#b91c1c"
                    n_clr_mines += 1
                    mined_areas[aid] = mc
                    mine_markers_x.append(w_start + dur * 0.5)
                    mine_markers_y.append(yi)
                else:
                    n_clr_none += 1
        if not finished:
            n_unfinished += 1
            n_unfinished_clr += int(k_str == "CLEARANCE")

        ax_gantt.barh(
            yi,
            dur,
            left=w_start - 0.4,
            height=0.56,
            color=col,
            alpha=0.88 if finished else 0.55,
            edgecolor=INK,
            linewidth=0.45,
            hatch=None if finished else "////",
        )

    mine_sc = None
    if mine_markers_x:
        mine_sc = ax_gantt.scatter(
            mine_markers_x,
            mine_markers_y,
            s=48,
            marker="D",
            color="#fef08a",
            edgecolors="#7f1d1d",
            linewidths=1.1,
            zorder=5,
            label=f"Cleared, mines found ({len(mined_areas)} area{'s' if len(mined_areas) != 1 else ''}, "
                  f"{sum(mined_areas.values())} mine cells, by week {up_to_week})",
        )

    ax_gantt.axvline(up_to_week + 0.45, color="#181816", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gantt.set_yticks(range(len(teams_order)))
    ax_gantt.set_yticklabels(
        [
            f"{t} ({'Survey Team' if t.startswith('S') else 'Clearance Team'})"
            for t in reversed(teams_order)
        ],
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )
    ax_gantt.set_xlim(0.5, 53.0)
    ax_gantt.set_xlabel("Week of the 52-Week Season", fontsize=8.5, color=INK)
    ax_gantt.grid(axis="x", linestyle=":", alpha=0.45)
    legend_patches = [
        Patch(facecolor="#2563eb", edgecolor=INK,
              label=f"Zone Survey (`ZONE_SURVEY` · {n_by_kind['ZONE_SURVEY']} tasks started)"),
        Patch(facecolor="#0d9488", edgecolor=INK,
              label=f"Area Survey Visit (`VISIT` · {n_by_kind['VISIT']} tasks started)"),
        Patch(facecolor="#d97706", edgecolor=INK,
              label=f"Clearance, No Mines Found ({n_clr_none} task{'s' if n_clr_none != 1 else ''})"),
        Patch(facecolor="#b91c1c", edgecolor=INK,
              label=f"Clearance, Mines Found ({n_clr_mines} task{'s' if n_clr_mines != 1 else ''} · "
                    f"{sum(mined_areas.values())} mine cells found and cleared)"),
        Patch(facecolor="#ffffff", edgecolor=INK, hatch="////",
              label=f"Hatched: Not Finished by Week {up_to_week} ({n_unfinished} tasks: {n_unfinished_clr} "
                    f"clearances with no result yet; {n_paused} paused)"),
    ]
    if mine_sc is not None:
        legend_patches.append(mine_sc)
    ax_gantt.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#dcd9cf",
        fontsize=7.6,
    )
    ax_gantt.set_title(
        f"1. Field Team Deployment & Clearance Timeline (Weeks 1–{up_to_week})",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )

    # Panel 2: Days Worked vs Days Lost to Weather vs Days Lost to Zone Closures per Team, at the season's end (week 52):
    # the task table holds each task's days for the whole season, not week by week
    y_pos = np.arange(len(teams_order))
    rev_teams = list(reversed(teams_order))
    d_work = []
    d_weath = []
    d_acc = []
    n_tasks_t = []
    for tm in rev_teams:
        t_rows = tasks_df[tasks_df["team"] == tm]
        d_work.append(int(t_rows["days_done"].sum()))
        d_weath.append(int(t_rows["days_lost_weather"].sum()))
        d_acc.append(int(t_rows["days_lost_access"].sum()))
        n_tasks_t.append(len(t_rows))

    d_work_arr = np.array(d_work, dtype=float)
    d_weath_arr = np.array(d_weath, dtype=float)
    d_acc_arr = np.array(d_acc, dtype=float)

    ax_bar.barh(y_pos, d_work_arr, height=0.56, color="#15803d", alpha=0.88, edgecolor=INK, linewidth=0.45, label="Days Worked on Task")
    ax_bar.barh(
        y_pos,
        d_weath_arr,
        left=d_work_arr,
        height=0.56,
        color="#0284c7",
        alpha=0.88,
        edgecolor=INK,
        linewidth=0.45,
        label="Days Lost to Heavy Rain or High Wind",
    )
    ax_bar.barh(
        y_pos,
        d_acc_arr,
        left=d_work_arr + d_weath_arr,
        height=0.56,
        color="#dc2626",
        alpha=0.88,
        edgecolor=INK,
        linewidth=0.45,
        label="Days Lost to Zone Closures (Access Feed)",
    )

    totals_t = d_work_arr + d_weath_arr + d_acc_arr
    for yi, tot_v in enumerate(totals_t):
        ax_bar.text(
            tot_v + 3.0,
            yi,
            f"{int(d_work_arr[yi])}d work · {int(d_weath_arr[yi])}d weather · {int(d_acc_arr[yi])}d closures ({n_tasks_t[yi]} tasks)",
            va="center",
            fontsize=7.6,
            color=INK,
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(rev_teams, fontsize=8.5, fontweight="bold", color=INK)
    ax_bar.set_xlim(0, max(100.0, float(totals_t.max()) * 2.05))
    ax_bar.set_xlabel("Team-Days at the Season's End (Worked + Lost to Weather + Lost to Zone Closures)", fontsize=8.5, color=INK)
    ax_bar.grid(axis="x", linestyle=":", alpha=0.45)
    ax_bar.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#dcd9cf",
        fontsize=7.6,
    )
    ax_bar.set_title(
        "2. Team-Days by Team, Season's End (Week 52)",
        fontsize=10.2,
        fontweight="bold",
        color=INK,
    )

    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.22, wspace=0.18)
    return fig


def render_team_deployment_and_attrition_studio(
    bundle: dict,
    ord_findings: dict[str, dict],
    week: int,
) -> None:
    """Render the interactive Field Team Deployment, Days Lost to Weather and Zone Closures & Clearance Timeline studio."""
    tasks_df = bundle.get("tasks_df", pd.DataFrame()).copy()
    zones_meta = bundle.get("zones_meta", pd.DataFrame())
    areas_meta = bundle.get("areas_meta", pd.DataFrame())
    if tasks_df.empty:
        st.info("No field team tasks found in bundle.")
        return

    kind_counts = tasks_df["kind"].value_counts()
    n_teams = int(sum(bundle.get("team_counts", {}).values())) or int(tasks_df["team"].nunique())
    season_cfg = bundle["setup"].twin.season
    rainy = [config.MONTHS[m - 1] for m in season_cfg.rainy_months]
    rainy_txt = ", ".join(rainy[:-1]) + " and " + rainy[-1] if len(rainy) > 1 else "".join(rainy)
    st.subheader("🚜 Field Team Deployment (`S1–S4`, `C1–C4`), Days Lost to Weather and Zone Closures & Clearance Timeline")
    st.caption(
        f"Tracks the **{n_teams} field teams** (survey teams `S1–S4`, clearance teams `C1–C4`) across the season's "
        f"**{len(tasks_df)} tasks** ({int(kind_counts.get('ZONE_SURVEY', 0))} zone surveys, "
        f"{int(kind_counts.get('VISIT', 0))} area visits, {int(kind_counts.get('CLEARANCE', 0))} clearances): "
        "**days worked (`days_done`)**, **days lost to heavy rain or high wind (`days_lost_weather`**; simulated: "
        f"a {season_cfg.rainy_day_loss:.0%} chance that a working day is lost in {rainy_txt}, "
        f"{season_cfg.other_day_loss:.0%} in other months), **days lost to zone closures by the access feed "
        f"(`days_lost_access`**; {int(tasks_df['days_lost_access'].sum())} in this season), and the clearances whose "
        "result in the season log found mine cells."
    )

    # Team-days through the selected week, from the season's own weekly tally (`week_days`: worked, weather, access)
    week_days = bundle.get("week_days", {}) or {}
    tot_work = int(sum(v.get("worked", 0) for w, v in week_days.items() if int(w) <= week))
    tot_rain = int(sum(v.get("weather", 0) for w, v in week_days.items() if int(w) <= week))
    tot_acc = int(sum(v.get("access", 0) for w, v in week_days.items() if int(w) <= week))
    tot_cal = tot_work + tot_rain + tot_acc
    rain_pct = 100.0 * (tot_rain + tot_acc) / max(1, tot_cal)
    n_started = int(((tasks_df["started_week"] > 0) & (tasks_df["started_week"] <= week)).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Field Teams", f"{n_teams} (S1–S4 · C1–C4)")
    k2.metric(f"Tasks Started (≤ W{week})", f"{n_started} of {len(tasks_df)}")
    k3.metric(f"Team-Days Worked (Weeks 1–{week})", f"{tot_work:,} d")
    k4.metric(f"Team-Days Lost (Weeks 1–{week})", f"{tot_rain:,} d weather · {tot_acc:,} d zone closures")
    k5.metric(f"Share of Team-Days Lost (Weeks 1–{week})", f"{rain_pct:.1f}% of field days")

    fig_gantt = plot_team_gantt_and_attrition(bundle, ord_findings, up_to_week=week)
    st.pyplot(fig_gantt)
    plt.close(fig_gantt)

    # Enrich tasks_df with municipality, distance to a road, consultation, and the clearance result in the season log
    z_muni = zones_meta["municipality"].to_dict() if not zones_meta.empty else {}
    z_access = zones_meta["access"].to_dict() if not zones_meta.empty else {}
    z_cons = zones_meta["consultation"].to_dict() if not zones_meta.empty else {}

    enriched_rows = []
    for _, r in tasks_df.iterrows():
        zid = str(r.get("zone", ""))
        aid = str(r.get("area", "")) if pd.notna(r.get("area")) and str(r.get("area")).strip() else "—"
        k_raw = str(r.get("kind", ""))
        mc = int(ord_findings.get(aid, {}).get("cleared_cells", 0)) if k_raw == "CLEARANCE" and aid != "—" else 0
        cn = str(ord_findings.get(aid, {}).get("clearance_note", "")) if k_raw == "CLEARANCE" and aid != "—" else ""
        d_done = int(r.get("days_done", 0))
        d_rain = int(r.get("days_lost_weather", 0))
        d_acc = int(r.get("days_lost_access", 0))
        d_tot = d_done + d_rain + d_acc
        enriched_rows.append(
            {
                "Task #": str(r["task_id"]),
                "Team": str(r["team"]),
                "Team Role": "Survey Team" if str(r["team"]).startswith("S") else "Clearance Team",
                "Task Type": k_raw,
                "Municipality": z_muni.get(zid, "—"),
                "Zone": zid,
                "Area": aid,
                "Distance to a Road": str(z_access.get(zid, "—")),
                "Consultation Required (Simulated Layer)": "Yes" if bool(z_cons.get(zid, False)) else "No",
                "Plan ID": str(r.get("plan_id", "")),
                "Ordered Wk": int(r.get("ordered_week", 0)),
                "Started Wk": int(r.get("started_week", 0)),
                "Finished Wk": int(r["finished_week"]) if pd.notna(r["finished_week"]) and float(r["finished_week"]) > 0 else None,
                "Status": str(r.get("status", "")),
                "Days Planned": int(r.get("days_planned", 0)),
                "Days Worked": d_done,
                "Days Lost (Weather)": d_rain,
                "Days Lost (Zone Closures)": d_acc,
                "Share of Days Lost %": round(100.0 * (d_rain + d_acc) / max(1, d_tot), 1),
                "Mine Cells Found and Cleared": mc,
                "Clearance Result (Season Log)": cn if cn else ("—" if k_raw != "CLEARANCE" else "not finished by week 52"),
            }
        )
    edf = pd.DataFrame(enriched_rows)

    # Summary by Team & Distance to a Road, at the season's end
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("##### 📊 Summary by Field Team (`S1–S4` & `C1–C4`), Season's End (Week 52)")
        team_sum = (
            edf.groupby(["Team", "Team Role"], as_index=False)
            .agg(
                Tasks=("Task #", "count"),
                Days_Worked=("Days Worked", "sum"),
                Days_Lost_Rain=("Days Lost (Weather)", "sum"),
                Days_Lost_Access=("Days Lost (Zone Closures)", "sum"),
                Cleared_Mine_Cells=("Mine Cells Found and Cleared", "sum"),
            )
            .rename(
                columns={
                    "Days_Worked": "Days Worked",
                    "Days_Lost_Rain": "Days Lost (Weather)",
                    "Days_Lost_Access": "Days Lost (Zone Closures)",
                    "Cleared_Mine_Cells": "Mine Cells Found and Cleared",
                }
            )
        )
        st.dataframe(team_sum, width="stretch", hide_index=True)

    with col_s2:
        st.markdown(
            "##### 🛣️ Days Lost to Weather and Zone Closures by the Zone's Distance to a Road (Under 2 km, 2 to 5 km, "
            "More Than 5 km), Season's End (Week 52)"
        )
        acc_sum = (
            edf.assign(_lost=edf["Days Lost (Weather)"] + edf["Days Lost (Zone Closures)"])
            .groupby("Distance to a Road", as_index=False)
            .agg(
                Tasks=("Task #", "count"),
                Mean_Days_Worked=("Days Worked", "mean"),
                Mean_Days_Weather=("Days Lost (Weather)", "mean"),
                Mean_Days_Zone_Closures=("Days Lost (Zone Closures)", "mean"),
                Total_Lost_Days=("_lost", lambda s: int(s.sum())),
            )
        )
        acc_sum["Mean_Days_Worked"] = acc_sum["Mean_Days_Worked"].round(1)
        acc_sum["Mean_Days_Weather"] = acc_sum["Mean_Days_Weather"].round(1)
        acc_sum["Mean_Days_Zone_Closures"] = acc_sum["Mean_Days_Zone_Closures"].round(1)
        st.dataframe(acc_sum, width="stretch", hide_index=True)

    # Filterable ledger of every task, as at the season's end
    st.markdown(f"##### 🔍 Filterable Ledger of All {len(edf)} Tasks (Status and Days at the Season's End, Week 52)")
    tf1, tf2, tf3 = st.columns(3)
    with tf1:
        sel_teams = st.multiselect(
            "Filter by Team:",
            options=["S1", "S2", "S3", "S4", "C1", "C2", "C3", "C4"],
            default=[],
            key="ops_studio_team_filter",
        )
    with tf2:
        sel_kinds = st.multiselect(
            "Filter by Task Type:",
            options=["ZONE_SURVEY", "VISIT", "CLEARANCE"],
            default=[],
            key="ops_studio_kind_filter",
        )
    with tf3:
        mined_rows = edf[edf["Mine Cells Found and Cleared"] > 0]
        only_mined_clr = st.checkbox(
            f"Show only the clearances that found mine cells ({len(mined_rows)} tasks · "
            f"{int(mined_rows['Mine Cells Found and Cleared'].sum())} mine cells found and cleared)",
            value=False,
            key="ops_studio_only_mined",
        )

    f_edf = edf.copy()
    if sel_teams:
        f_edf = f_edf[f_edf["Team"].isin(sel_teams)]
    if sel_kinds:
        f_edf = f_edf[f_edf["Task Type"].isin(sel_kinds)]
    if only_mined_clr:
        f_edf = f_edf[f_edf["Mine Cells Found and Cleared"] > 0]

    st.dataframe(f_edf, width="stretch", hide_index=True)


# ====================================================================================================
# UPGRADE 3: BAYESIAN RISK LEARNING (WEEK-1 PRIOR vs. WEEK-52 POSTERIOR) & SPATIAL GATE DIAGNOSTICS
# ====================================================================================================

def build_bayesian_risk_shift_dataframe(
    bundle: dict,
    log_df: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the area-level Bayesian risk shift table (`Week-1 Prior` vs `Week-52 Posterior`) across all areas."""
    areas_meta = bundle["areas_meta"]
    risk_w1 = bundle.get("risk")
    _p52, risk_w52 = grid_map.chance_in_force(bundle, 52)
    risk_w52 = risk_w52 if risk_w52 is not None else risk_w1

    p1 = (
        risk_w1.answer["areas"]["chance"].reindex(areas_meta.index).astype(float)
        if (risk_w1 is not None and risk_w1.ok)
        else pd.Series(0.0, index=areas_meta.index)
    )
    p52 = (
        risk_w52.answer["areas"]["chance"].reindex(areas_meta.index).astype(float)
        if (risk_w52 is not None and risk_w52.ok)
        else p1.copy()
    )
    status_52 = area_status_at_week(bundle, 52)

    gate_map = {}
    if not gate_df.empty and "municipality_code" in gate_df.columns:
        for _, r in gate_df.iterrows():
            gate_map[str(r["municipality_code"])] = str(r.get("chosen_label", r.get("chosen", "—")))

    df = pd.DataFrame(
        {
            "area_id": areas_meta.index.astype(str),
            "zone_id": areas_meta["zone_id"].astype(str).to_numpy(),
            "municipality_code": areas_meta["municipality_code"].astype(str).to_numpy(),
            "municipality": areas_meta["municipality"].astype(str).to_numpy(),
            "gated_model": [gate_map.get(str(c), "—") for c in areas_meta["municipality_code"]],
            "main_land_use": areas_meta["main_land_use"].astype(str).to_numpy(),
            "event_inside": areas_meta["event_inside"].astype(bool).to_numpy(),
            "status_w52": status_52.astype(str).to_numpy(),
            "chance_w1": p1.to_numpy(dtype=float),
            "chance_w52": p52.to_numpy(dtype=float),
            "delta_chance": (p52 - p1).to_numpy(dtype=float),
            "delta_pct_pts": (100.0 * (p52 - p1)).to_numpy(dtype=float),
        },
        index=areas_meta.index.astype(str),
    )
    return df


def plot_bayesian_risk_shift_and_gate_diagnostics(
    bundle: dict,
    shift_df: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> plt.Figure:
    """Render a 2-panel figure:
    - Left panel: Regional Cartographic Map of the Week-1 -> Week-52 Bayesian Risk Shift (`Δp = p_W52 − p_W1`, no axes/coordinates).
    - Right panel: Spatial Cross-Validation Gate Diagnostics (`PR-AUC` vs `Calibration Error` across all 15 municipalities).
    """
    cells_meta = bundle["cells_meta"]
    zones_meta = bundle["zones_meta"]
    canvas_df = grid_map._build_regional_canvas_index(cells_meta)
    pr = canvas_df["plot_r"].to_numpy(dtype=int)
    pc = canvas_df["plot_c"].to_numpy(dtype=int)

    muni_codes = list(config.NAME_OF.keys())
    muni_to_int = {c: idx for idx, c in enumerate(muni_codes)}
    zone_ids = list(zones_meta.index)
    zone_to_int = {z: idx for idx, z in enumerate(zone_ids)}

    muni_grid = np.full((grid_map.REGIONAL_CANVAS_H, grid_map.REGIONAL_CANVAS_W), -1, dtype=int)
    zone_grid = np.full((grid_map.REGIONAL_CANVAS_H, grid_map.REGIONAL_CANVAS_W), -1, dtype=int)
    muni_grid[pr, pc] = canvas_df["municipality_code"].map(muni_to_int).to_numpy(dtype=int)
    zone_grid[pr, pc] = canvas_df["zone_id"].map(zone_to_int).to_numpy(dtype=int)

    outer_segs = grid_map._extract_grid_boundary_segments(muni_grid, mode="outer")
    muni_segs = grid_map._extract_grid_boundary_segments(muni_grid, mode="inner")
    zone_segs = grid_map._extract_grid_boundary_segments(zone_grid, mode="inner")

    fig = plt.figure(figsize=(16.0, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.18)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_gate = fig.add_subplot(gs[0, 1])

    ax_map.set_facecolor(SURFACE)
    ax_map.set_xlim(-2, grid_map.REGIONAL_CANVAS_W + 2)
    ax_map.set_ylim(grid_map.REGIONAL_CANVAS_H + 2, -2)
    ax_map.axis("off")
    grid_map._draw_regional_inset_decorations(ax_map)

    delta_ser = shift_df["delta_pct_pts"]
    delta_grid = np.full((grid_map.REGIONAL_CANVAS_H, grid_map.REGIONAL_CANVAS_W), np.nan, dtype=float)
    delta_grid[pr, pc] = delta_ser.reindex(canvas_df["area_id"]).to_numpy(dtype=float)

    div_cmap = LinearSegmentedColormap.from_list(
        "risk_shift_div",
        ["#0f766e", "#99f6e4", "#f3f1e9", "#fed7aa", "#b91c1c"],
    )
    div_cmap.set_bad(color=(0, 0, 0, 0))
    norm = TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=12.0)
    im = ax_map.imshow(delta_grid, cmap=div_cmap, norm=norm, interpolation="nearest", zorder=2)

    ax_map.add_collection(LineCollection(zone_segs, colors="#1f1e1b", linewidths=0.22, alpha=0.22, zorder=3))
    ax_map.add_collection(LineCollection(muni_segs, colors="#faf9f5", linewidths=1.5, alpha=0.75, zorder=4))
    ax_map.add_collection(LineCollection(muni_segs, colors="#2b2a27", linewidths=0.82, alpha=0.92, zorder=4.1))
    ax_map.add_collection(LineCollection(outer_segs, colors="#181816", linewidths=1.15, alpha=0.95, zorder=4.2))

    for code, grp in canvas_df.groupby("municipality_code"):
        med_r = float(np.median(grp["plot_r"]))
        med_c = float(np.median(grp["plot_c"]))
        short_lbl = grid_map.SHORT_MUNI_LABELS.get(str(code), config.NAME_OF.get(str(code), str(code))).replace("\n", " ")
        m_sub = shift_df[shift_df["municipality_code"] == str(code)]
        n_up = int((m_sub["delta_pct_pts"] > 0.05).sum())
        n_dn = int((m_sub["delta_pct_pts"] < -0.05).sum())
        lbl = f"{short_lbl}\n(+{n_up} / -{n_dn})" if (n_up + n_dn) > 0 else short_lbl
        ax_map.text(
            med_c,
            med_r,
            lbl,
            ha="center",
            va="center",
            fontsize=5.9,
            fontweight="bold" if (n_up + n_dn) > 0 else "normal",
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.12,rounding_size=0.2",
                "facecolor": "#faf9f5d8",
                "edgecolor": "#9c988d",
                "linewidth": 0.45,
            },
            zorder=5.5,
        )

    cbar = fig.colorbar(im, ax=ax_map, orientation="horizontal", fraction=0.038, pad=0.02)
    cbar.set_label("Shift in the Chance of Holding Mines (Week 52 − Week 1, percentage points)", fontsize=8.0, color=INK)
    cbar.ax.tick_params(labelsize=7.5)

    n_up_tot = int((shift_df["delta_pct_pts"] > 0.01).sum())
    n_dn_tot = int((shift_df["delta_pct_pts"] < -0.01).sum())
    ax_map.set_title(
        "1. Bayesian Risk Learning Map (Week 52 Posterior − Week 1 Prior)\n"
        f"{n_up_tot:,} areas revised upward (crimson) · {n_dn_tot:,} areas revised downward (teal)",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
        pad=6,
    )

    # Right panel: Spatial Cross-Validation Gate Comparison across the 15 municipalities
    ax_gate.set_facecolor(SURFACE)
    ax_gate.spines["top"].set_visible(False)
    ax_gate.spines["right"].set_visible(False)

    if not gate_df.empty:
        gdf = gate_df.sort_values("lr: model mean PR-AUC (others)", ascending=True).copy()
        y_pos = np.arange(len(gdf))
        h = 0.26
        ax_gate.barh(
            y_pos + h,
            gdf["lr: model mean PR-AUC (others)"],
            height=h,
            color="#1d4ed8",
            alpha=0.88,
            edgecolor=INK,
            linewidth=0.4,
            label="Logistic Regression (63 RELand Features)",
        )
        ax_gate.barh(
            y_pos,
            gdf["hgb: model mean PR-AUC (others)"],
            height=h,
            color="#7c3aed",
            alpha=0.88,
            edgecolor=INK,
            linewidth=0.4,
            label="Boosted Trees (HGB)",
        )
        ax_gate.barh(
            y_pos - h,
            gdf["lr: baseline mean PR-AUC (others)"],
            height=h,
            color="#d97706",
            alpha=0.88,
            edgecolor=INK,
            linewidth=0.4,
            label="Baseline (Nearness to Past Accidents)",
        )
        for yi, (_, r_g) in enumerate(gdf.iterrows()):
            chosen_s = str(r_g.get("chosen_label", r_g.get("chosen", "")))
            max_pr = max(
                float(r_g["lr: model mean PR-AUC (others)"]),
                float(r_g["hgb: model mean PR-AUC (others)"]),
                float(r_g["lr: baseline mean PR-AUC (others)"]),
            )
            # the calibration error of the ranking the gate chose (the baseline's is the same in both columns)
            cal_col = {
                "hgb": "hgb: model calibration error (others)",
                "lr": "lr: model calibration error (others)",
            }.get(str(r_g.get("chosen", "")), "lr: baseline calibration error (others)")
            cal_chosen = float(r_g.get(cal_col, float("nan")))
            ax_gate.text(
                max_pr + 0.008,
                yi,
                f"Gate: {chosen_s} (cal err {cal_chosen:.3f})",
                va="center",
                fontsize=7.3,
                color=INK,
            )

        ax_gate.set_yticks(y_pos)
        ax_gate.set_yticklabels(gdf["municipality"].tolist(), fontsize=8.0, color=INK)
        ax_gate.set_xlim(0, 0.58)
        ax_gate.set_xlabel(
            f"Mean PR-AUC on the Other {len(gate_df) - 1} Municipalities (the Gate's Inner Evaluation)",
            fontsize=8.3,
            color=INK,
        )
        ax_gate.grid(axis="x", linestyle=":", alpha=0.45)
        ax_gate.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#dcd9cf", fontsize=7.6)
        ax_gate.set_title(
            "2. Spatial Cross-Validation Model Gate (`models/gate.csv`)\n"
            f"Ranking (PR-AUC) + Calibration Gate Across {len(gate_df)} Municipalities",
            fontsize=10.0,
            fontweight="bold",
            color=INK,
        )

    fig.subplots_adjust(left=0.01, right=0.98, top=0.90, bottom=0.10)
    return fig


def render_bayesian_learning_and_gate_studio(
    bundle: dict,
    log_df: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> None:
    """Render the interactive Week-1 vs. Week-52 Bayesian Risk Shift & Spatial Gate Explorer inside `screen_plan.py`."""
    shift_df = build_bayesian_risk_shift_dataframe(bundle, log_df, gate_df)
    n_shifted = int((shift_df["delta_chance"].abs() > 1e-6).sum())
    n_up = int((shift_df["delta_chance"] > 1e-4).sum())
    n_down = int((shift_df["delta_chance"] < -1e-4).sum())
    max_up_row = shift_df.sort_values("delta_pct_pts", ascending=False).iloc[0]
    max_dn_row = shift_df.sort_values("delta_pct_pts", ascending=True).iloc[0]
    n_munis_changed = int(shift_df.loc[shift_df["delta_chance"].abs() > 1e-6, "municipality_code"].nunique())

    st.subheader("📈 Bayesian Risk Learning (Week 1 Prior vs. Week 52 Posterior) & Spatial Gate Explorer")
    st.caption(
        f"Compares the risk overlay's calibrated chance that each area holds mines at the start of the season (week 1) "
        f"with the chances in force at week 52, for all {len(bundle['areas_meta']):,} areas. Each month the overlay "
        "updates one odds multiplier per municipality from the season's own field results there (zone-survey flags, "
        "and the areas an intrusive check or direct evidence showed to hold mines or not; `twin/prioritise.py`), so "
        "every area in a municipality moves together; no area's chance is set by its own result. In this season the "
        f"chances changed in {n_munis_changed} of the {shift_df['municipality_code'].nunique()} municipalities. Beside "
        "it, the baseline-first gate (`models/gate.csv`)."
    )

    gate_counts = (
        gate_df["chosen"].value_counts() if not gate_df.empty and "chosen" in gate_df.columns else pd.Series(dtype=int)
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Areas Updated by Bayesian Learning", f"{n_shifted:,} of {len(shift_df):,}")
    k2.metric("Revised Upward (+Δp)", f"{n_up:,} areas", f"Max: +{max_up_row['delta_pct_pts']:.2f} pp ({max_up_row['area_id']})")
    k3.metric("Revised Downward (−Δp)", f"{n_down:,} areas", f"Min: {max_dn_row['delta_pct_pts']:.2f} pp ({max_dn_row['area_id']})")
    k4.metric(
        f"Gated Risk Models ({len(gate_df)} Munis)",
        f"{int(gate_counts.get('baseline', 0))} Nearness · {int(gate_counts.get('hgb', 0))} Trees · "
        f"{int(gate_counts.get('lr', 0))} Logistic",
    )

    fig_shift = plot_bayesian_risk_shift_and_gate_diagnostics(bundle, shift_df, gate_df)
    st.pyplot(fig_shift)
    plt.close(fig_shift)

    # Filterable Area Risk Shift Table
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        dir_mode = st.selectbox(
            "Filter by risk shift direction:",
            options=["UP", "DOWN", "ALL_SHIFTED", "ALL"],
            format_func=lambda x: {
                "UP": f"🔺 Largest Upward Risk Revisions (+Δp · {n_up:,} areas)",
                "DOWN": f"🔻 Largest Downward Risk Revisions (−Δp · {n_down:,} areas)",
                "ALL_SHIFTED": f"All Shifted Areas ({n_shifted:,} areas)",
                "ALL": f"All {len(shift_df):,} Areas",
            }[x],
            key="bayesian_shift_dir_mode",
        )
    with sc2:
        m_sel = st.multiselect(
            "Filter by municipality:",
            options=list(config.NAME_OF.values()),
            default=[],
            key="bayesian_shift_muni_sel",
        )
    with sc3:
        st_sel = st.multiselect(
            "Filter by Week-52 operational status:",
            options=sorted(shift_df["status_w52"].unique().tolist()),
            default=[],
            key="bayesian_shift_status_sel",
        )

    sub_s = shift_df.copy()
    if dir_mode == "UP":
        sub_s = sub_s[sub_s["delta_chance"] > 1e-4].sort_values("delta_pct_pts", ascending=False)
    elif dir_mode == "DOWN":
        sub_s = sub_s[sub_s["delta_chance"] < -1e-4].sort_values("delta_pct_pts", ascending=True)
    elif dir_mode == "ALL_SHIFTED":
        sub_s = sub_s[sub_s["delta_chance"].abs() > 1e-6].sort_values("delta_pct_pts", key=lambda s: s.abs(), ascending=False)
    else:
        sub_s = sub_s.sort_values("chance_w52", ascending=False)

    if m_sel:
        sub_s = sub_s[sub_s["municipality"].isin(m_sel)]
    if st_sel:
        sub_s = sub_s[sub_s["status_w52"].isin(st_sel)]

    disp_s = pd.DataFrame(
        {
            "Area ID": sub_s["area_id"],
            "Zone ID": sub_s["zone_id"],
            "Municipality": sub_s["municipality"],
            "Gated Risk Model": sub_s["gated_model"],
            "Land Use": sub_s["main_land_use"],
            "Past Accident Inside (DB)": sub_s["event_inside"],
            "Status (Week 52)": sub_s["status_w52"],
            "Week-1 Prior Chance": (sub_s["chance_w1"] * 100.0).round(2).astype(str) + "%",
            "Week-52 Posterior Chance": (sub_s["chance_w52"] * 100.0).round(2).astype(str) + "%",
            "Shift (pp)": sub_s["delta_pct_pts"].round(2),
        }
    )
    st.dataframe(disp_s.head(500), width="stretch", hide_index=True)


# ====================================================================================================
# PUBLIC MUNICIPAL SUMMARY DOWNLOAD
# ====================================================================================================

def render_public_release_pack_exporter(public_df: pd.DataFrame, bundle: dict) -> None:
    """Render the download of the public municipal summary (open layers, by municipality) inside `screen_public.py`."""
    n_weeks = int(bundle["setup"].twin.season.weeks)
    st.markdown("---")
    st.subheader("📑 Download the Public Municipal Summary (Open Layers, by Municipality: `.md` / `.html` / `.csv`)")
    st.caption(
        "Built only from the twin's public summary, `twin.municipal_summary()`: open layers, as counts by "
        f"municipality, at the season's end (week {n_weeks}). A check looks for zone IDs, area IDs and team codes in "
        "the summary text."
    )

    lines = [
        f"# Public Municipal Land-Release Summary — {len(public_df)} Pilot Municipalities of Antioquia",
        "**Source:** the twin's public summary (`twin.municipal_summary()`): open layers only, as counts by "
        f"municipality, at the season's end (week {n_weeks}).  ",
        "**Contents:** no zone, area or team is named; recorded victims are municipal totals for all years.",
        "",
        "| Municipality | Zones | Zones Surveyed | Zones Declared Free | Areas | Area (km²) | Areas Released | Areas Cleared | Areas Suspected or Confirmed | Recorded Victims (All Years) | Municipality Declared Free (Week) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m_name, row in public_df.iterrows():
        lines.append(
            f"| {m_name} | {int(row['zones'])} | {int(row['zones surveyed'])} | {int(row['zones declared free'])} | "
            f"{int(row['areas'])} | {float(row['km²']):.1f} | {int(row['areas released'])} | {int(row['areas cleared'])} | "
            f"{int(row['areas suspected or confirmed'])} | {int(row['recorded victims (all years)']):,} | "
            f"{str(row.get('declared free (week)', '')).strip() or 'Not declared free'} |"
        )
    md_pub = "\n".join(lines)

    # Check for zone IDs, area IDs and team codes in the summary text
    forbidden_match = re.search(r"\b([A-Z]{3}-\d{4}|[A-Z]{3}-Z\d{2}|S[1-4]|C[1-4])\b", md_pub)
    is_clean = forbidden_match is None

    if is_clean:
        st.success("✅ No zone ID, area ID or team code found in the summary text.")

    html_pub = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Antioquia Public Municipal Summary</title>
<style>
body {{ font-family: Georgia, serif; background: #faf9f5; color: #181816; max-width: 960px; margin: 28px auto; padding: 24px 32px; border: 1px solid #d5d2c8; border-radius: 8px; line-height: 1.55; }}
.pub-banner {{ background: #1e603c; color: #ffffff; padding: 8px 14px; border-radius: 6px; font-weight: bold; margin-bottom: 16px; }}
</style></head><body>
<div class="pub-banner">PUBLIC MUNICIPAL SUMMARY — OPEN LAYERS ONLY, BY MUNICIPALITY</div>
<pre style="white-space: pre-wrap; font-family: inherit;">{md_pub}</pre>
</body></html>"""

    p1, p2, p3 = st.columns(3)
    with p1:
        st.download_button(
            label="⬇ Download Public Summary (.md)",
            data=md_pub.encode("utf-8"),
            file_name="antioquia_public_municipal_summary.md",
            mime="text/markdown",
            key="dl_public_pack_md",
        )
    with p2:
        st.download_button(
            label="⬇ Download Printable Public Fact Sheet (.html)",
            data=html_pub.encode("utf-8"),
            file_name="antioquia_public_municipal_summary.html",
            mime="text/html",
            key="dl_public_pack_html",
        )
    with p3:
        st.download_button(
            label="⬇ Download Public Municipal Table (.csv)",
            data=public_df.to_csv(index=True).encode("utf-8"),
            file_name="antioquia_public_municipal_summary.csv",
            mime="text/csv",
            key="dl_public_pack_csv",
        )
