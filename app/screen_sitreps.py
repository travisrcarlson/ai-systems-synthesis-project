"""Screen 5: Weekly Situation Reports (Inside the twin, restricted).

Includes a visual sentence-by-sentence `sitrep.figure_check` highlighter and a 52-week overview of the
drafts' length and the figure check.
"""
import html
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from twin import sitrep
from loaders import (
    load_saved_sitreps,
    nav_to_operations_week,
    nav_to_pipeline,
    render_recommendation_box,
    restricted_header,
)

# The prompt asks for at most 150 words (prompts/sitrep_prompt.md); nothing in the code enforces it (Model Card 04)
PROMPT_WORD_LIMIT = 150


def _format_fact_value(v: object) -> str:
    """Format a weekly fact value cleanly in plain words while preserving counts."""
    if isinstance(v, dict):
        if list(v.keys()) == ["count"]:
            return f"{v['count']} (count only — no code or zone named)"
        return ", ".join(f"{k}: {val}" for k, val in v.items())
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else "None"
    return str(v)


def _render_sentence_diff_highlighter(draft: str, removed: list[str], unmatched: list) -> None:
    """Render a sentence-by-sentence figure-check card: figures in digits that are among the week's facts in green,
    and each sentence the figure check removed, with its figures not in the facts, in red. Figures are found with
    the figure check's own pattern (`sitrep.NUMBER`), so codes such as S2 or ABE-Z16 are not read as numbers.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", draft) if s.strip()]
    removed_set = {s.strip() for s in removed}
    unmatched_vals = {float(u) for u in unmatched}

    if not sentences:
        st.info("Empty draft.")
        return

    html_parts = [
        '<div style="border:1px solid #d8d5ce;border-radius:8px;padding:12px 14px;background:#ffffff;line-height:1.6;">'
    ]
    for idx, sent in enumerate(sentences, start=1):
        is_rem = sent in removed_set
        esc_sent = html.escape(sent)

        def _highlight_num(m: re.Match) -> str:
            tok = m.group(0)
            value = float(m.group(1).replace(",", ""))
            if is_rem and value in unmatched_vals:
                return (
                    f'<span style="background:#fee2e2;color:#991b1b;padding:1px 5px;border-radius:4px;'
                    f'font-weight:800;border:1px solid #f87171;" title="Figure not in the week\'s facts">{tok}</span>'
                )
            return (
                f'<span style="background:#dcfce7;color:#166534;padding:1px 5px;border-radius:4px;'
                f'font-weight:700;" title="Among the week\'s facts">{tok}</span>'
            )

        annotated = sitrep.NUMBER.sub(_highlight_num, esc_sent)
        if is_rem:
            html_parts.append(
                f'<div style="margin-bottom:7px;padding:6px 10px;background:#fef2f2;border-left:4px solid #dc2626;border-radius:4px;">'
                f'<span style="font-size:0.74rem;font-weight:700;color:#dc2626;text-transform:uppercase;margin-right:6px;">'
                f'🛑 Sentence {idx} (Removed by Figure Check):</span>'
                f'<span style="color:#7f1d1d;text-decoration:line-through;">{annotated}</span>'
                f"</div>"
            )
        else:
            html_parts.append(
                f'<div style="margin-bottom:7px;padding:6px 10px;background:#f8fafc;border-left:4px solid #16a34a;border-radius:4px;">'
                f'<span style="font-size:0.74rem;font-weight:700;color:#15803d;text-transform:uppercase;margin-right:6px;">'
                f'✅ Sentence {idx} (kept: every figure is in the facts):</span>'
                f'<span style="color:#1c1e21;">{annotated}</span>'
                f"</div>"
            )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _plot_52week_sitrep_audit(sitreps_map: dict) -> plt.Figure:
    """Plot each week's draft length against the prompt's word limit, with the drafts over it in amber."""
    weeks = sorted(sitreps_map.keys())
    word_counts = [len(str(sitreps_map[w]["recommendation"].answer.get("draft", "")).split()) for w in weeks]

    fig, ax_w = plt.subplots(figsize=(15.4, 3.6), facecolor="#ffffff")
    ax_w.set_facecolor("#ffffff")
    ax_w.spines["top"].set_visible(False)
    ax_w.spines["right"].set_visible(False)

    w_arr = np.array(weeks, dtype=int)
    wc_arr = np.array(word_counts, dtype=int)
    over = wc_arr[wc_arr > PROMPT_WORD_LIMIT]
    bar_cols = ["#d97706" if wc > PROMPT_WORD_LIMIT else "#1f4e78" for wc in wc_arr]
    ax_w.bar(w_arr, wc_arr, color=bar_cols, width=0.72, alpha=0.88)
    ax_w.axhline(
        PROMPT_WORD_LIMIT,
        color="#dc2626",
        linestyle="--",
        linewidth=1.3,
        label=f"{PROMPT_WORD_LIMIT}-Word Prompt Limit (`prompts/sitrep_prompt.md`)",
    )
    ax_w.set_ylim(0, max(wc_arr, default=160) + 18)
    ax_w.set_xlabel(f"Week of Season ({w_arr.min()} to {w_arr.max()})", fontsize=8.2)
    ax_w.set_ylabel("Word Count", fontsize=8.2)
    if len(over):
        span = f"{over.min()}" if over.min() == over.max() else f"{over.min()} to {over.max()}"
        over_text = (
            f"amber: the {len(over)} draft{'s' if len(over) != 1 else ''} over {PROMPT_WORD_LIMIT} words, {span}"
        )
    else:
        over_text = f"no draft over {PROMPT_WORD_LIMIT} words"
    ax_w.set_title(
        f"Word Count of Each Week's Draft Against the {PROMPT_WORD_LIMIT}-Word Prompt Limit\n"
        f"({over_text}; all {len(wc_arr)} drafts run {wc_arr.min()} to {wc_arr.max()} words)",
        fontsize=9.3,
        fontweight="bold",
    )
    ax_w.legend(loc="lower right", fontsize=7.6, frameon=True)
    ax_w.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


def render_situation_reports(root: Path) -> None:
    restricted_header(
        "Weekly Situation Reports",
        "Inspect each week's operational situation report: the draft as the model wrote it, "
        "the weekly facts it was given (with teams and closed zones as counts only), and the figure check result.",
    )

    sitreps_map = load_saved_sitreps(str(root))
    if not sitreps_map:
        st.error("No saved situation reports were found.")
        return

    weeks_available = sorted(sitreps_map.keys())
    all_removed = sum(len(v["recommendation"].answer["removed"]) for v in sitreps_map.values())
    all_insuff = sum(0 if v["recommendation"].ok else 1 for v in sitreps_map.values())
    weeks_with_removals = [w for w in weeks_available if len(sitreps_map[w]["recommendation"].answer["removed"]) > 0]
    models_seen = sorted({str(v["record"].get("model", "")) for v in sitreps_map.values() if v["record"].get("model")})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weekly Reports Available", len(weeks_available))
    c2.metric("Report Model", ", ".join(models_seen) if models_seen else "saved model")
    c3.metric(
        "Sentences Removed by Figure Check",
        all_removed,
        f"Across {len(weeks_with_removals)} week(s)"
        if weeks_with_removals
        else f"100% figure match across all {len(weeks_available)} weeks",
    )
    c4.metric("Reports Flagged Insufficient Evidence", all_insuff)

    with st.expander(
        f"📊 Draft Length & Figure-Check Ledger (All {len(weeks_available)} Weeks)", expanded=False
    ):
        fig_sr_audit = _plot_52week_sitrep_audit(sitreps_map)
        st.pyplot(fig_sr_audit)
        plt.close(fig_sr_audit)
        st.caption(
            f"The prompt asks for at most {PROMPT_WORD_LIMIT} words; nothing in the code enforces the limit "
            "(Model Card 04, `docs/model_cards/04_situation_report.md`)."
        )

        ledger_rows = []
        for w in weeks_available:
            e_w = sitreps_map[w]
            r_w = e_w["recommendation"]
            a_w = r_w.answer
            draft_w = a_w.get("draft", "")
            wc_w = len(str(draft_w).split())
            ledger_rows.append(
                {
                    "Week": w,
                    "Month": a_w["facts"].get("month", ""),
                    f"Word Count (Prompt Limit {PROMPT_WORD_LIMIT})": (
                        f"{wc_w} {f'⚠️ (>{PROMPT_WORD_LIMIT}w)' if wc_w > PROMPT_WORD_LIMIT else ''}".strip()
                    ),
                    "Total Sentences": int(e_w["sentences"]),
                    "Sentences Kept": int(e_w["sentences"]) - len(a_w["removed"]),
                    "Sentences Removed": len(a_w["removed"]),
                    "Unmatched Figures": ", ".join(str(x) for x in a_w["unmatched"]) or "—",
                    "Status": r_w.status,
                }
            )
        st.dataframe(pd.DataFrame(ledger_rows), width="stretch", hide_index=True)

    week = st.slider(
        f"Select week of the season ({min(weeks_available)} to {max(weeks_available)})",
        min_value=min(weeks_available),
        max_value=max(weeks_available),
        value=min(20, max(weeks_available)),
        step=1,
        key="sitreps_week_slider",
    )

    if week not in sitreps_map:
        st.warning(f"No saved situation report found for week {week}.")
        return

    entry = sitreps_map[week]
    record = entry["record"]
    rec = entry["recommendation"]
    ans = rec.answer
    facts = ans["facts"]
    model_name = record.get("model", "model")

    st.subheader(f"Week {week} — {facts.get('month', '')}")
    sj1, sj2 = st.columns(2)
    with sj1:
        st.button(
            f"📋 Open Week {week} in Weekly Operations",
            key=f"sitrep_jump_ops_{week}",
            width="stretch",
            on_click=nav_to_operations_week,
            args=(week,),
        )
    with sj2:
        st.button(
            f"🔄 Open Land Release Pipeline at Week {week}",
            key=f"sitrep_jump_pipe_{week}",
            width="stretch",
            on_click=nav_to_pipeline,
            args=(None, week, 4),
        )

    col_draft, col_facts = st.columns([5.4, 5.6])

    with col_draft:
        tab_diff, tab_audit, tab_raw = st.tabs(
            [
                "🔍 Sentence-by-Sentence Figure-Check Highlighter",
                "⚖️ Length & Coded Draft (`coded_draft`)",
                "📝 Raw Model Draft & Kept Text",
            ]
        )
        with tab_diff:
            st.caption(
                "`sitrep.figure_check` reads every number in each sentence (in digits or in words) and checks that it "
                "is among the week's facts as sent to the model. Figures in digits that are among the facts are "
                "highlighted in **green**; a sentence with a figure not among them is removed (**red**). The check "
                "tests the numbers, not what the words claim about them."
            )
            _render_sentence_diff_highlighter(ans["draft"], ans["removed"], ans["unmatched"])

        with tab_audit:
            wc_this = len(str(ans["draft"]).split())
            st.caption(
                f"**Length (week {week}):** {wc_this} words (prompt limit {PROMPT_WORD_LIMIT}"
                + ("; over the limit" if wc_this > PROMPT_WORD_LIMIT else "")
                + ")."
            )

            st.markdown("##### Side-by-Side: LLM Draft vs. Deterministic Coded Draft (`twin.sitrep.coded_draft`)")
            c_llm, c_cod = st.columns(2)
            with c_llm:
                st.markdown(f"**Saved LLM Draft (`{model_name}`, {wc_this} words):**")
                with st.container(border=True):
                    st.markdown(ans["draft"] or "*(Empty)*")
            with c_cod:
                try:
                    coded_txt = sitrep.coded_draft(facts)
                except KeyError:
                    coded_txt = ""
                if coded_txt:
                    st.markdown(f"**Deterministic Coded Draft (`coded_draft`, {len(coded_txt.split())} words):**")
                    with st.container(border=True):
                        st.markdown(coded_txt)
                else:
                    st.info(
                        "The coded draft names the teams that this week's saved facts give as counts only, so it "
                        "cannot be written from them."
                    )

        with tab_raw:
            st.markdown("#### 1. The draft as the model wrote it")
            with st.container(border=True):
                st.markdown(ans["draft"] if ans["draft"] else "*(Empty draft)*")

        st.markdown("#### 2. Figure check result")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Sentences in Draft", int(entry["sentences"]))
        fc2.metric("Sentences Removed", int(len(ans["removed"])))
        fc3.metric("Unmatched Figures", int(len(ans["unmatched"])))

        if ans["removed"]:
            st.warning(
                "**Sentences removed by the figure check:**\n"
                + "\n".join(f"- {s}" for s in ans["removed"])
                + f"\n\n**Unmatched numbers:** `{ans['unmatched']}`"
            )
            if not rec.ok:
                st.error(
                    f"**{rec.status}:** {rec.reason}. Because more than half the sentences had figures not in "
                    "the facts, the draft is not shown as the report; the facts table stands in its place."
                )
            with st.container(border=True):
                st.markdown("**Kept text:**\n\n" + (ans["text"] if ans["text"] else "*(No sentences kept)*"))
        else:
            st.success(
                "Every number in the draft matched the weekly facts: **0 sentences removed**."
            )

        render_recommendation_box(
            rec,
            title=f"Week {week} situation report check",
            model_name=f"{model_name}",
        )

    with col_facts:
        st.markdown("#### 3. The facts the model was given")
        st.caption(
            "Restricted layers (`teams stood down`, `teams with no report for more than a week`, and "
            "`zones closed by the access feed`) are passed to the model as counts only, so no team or "
            "closed zone is named."
        )
        fact_rows = [
            {"Weekly Fact": str(k).capitalize(), "Value Given to Model": _format_fact_value(v)}
            for k, v in facts.items()
        ]
        st.dataframe(pd.DataFrame(fact_rows), width="stretch", hide_index=True)

        with st.expander(f"Inspect raw saved record (week {week})"):
            st.json(record)
