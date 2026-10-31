"""Screen 6: Public Summary (open layers only, by municipality).

Strictly follows Hard Rules 3, 4, and 5:
- Shows only `twin.municipal_summary()`.
- Never names any operational zone, hazardous area, or field team.
- Shows victims solely as counts by municipality.
- Uses no geographic coordinates, axes, or scale bars.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from twin import config
import grid_map
import operational_analytics
from loaders import get_twin_bundle


def _render_municipality_public_card(
    muni_name: str,
    row: pd.Series,
) -> None:
    """Render a public-facing Municipal Release Fact Sheet card from the municipality's row of
    `twin.municipal_summary()` only."""
    z_free = int(row.get("zones declared free", 0))
    z_surv = int(row.get("zones surveyed", 0))
    z_tot = int(row.get("zones", 0))
    a_tot = int(row.get("areas", 0))
    km2 = float(row.get("km²", 0.0))
    a_rel = int(row.get("areas released", 0))
    a_clr = int(row.get("areas cleared", 0))
    a_susp = int(row.get("areas suspected or confirmed", 0))
    victims = int(row.get("recorded victims (all years)", 0))
    free_wk = str(row.get("declared free (week)", "")).strip()

    pct_zones_free = 100.0 * z_free / max(z_tot, 1)
    if free_wk:
        badge_bg, badge_fg = "#d9f0e3", "#1e603c"
        status_text = f"DECLARED FREE (WEEK {free_wk})"
    elif z_surv > 0 or z_free > 0 or a_rel > 0:
        badge_bg, badge_fg = "#e6f0fb", "#1e4e8c"
        status_text = f"LAND RELEASE IN PROGRESS ({pct_zones_free:.0f}% OF ZONES FREE)"
    else:
        badge_bg, badge_fg = "#f2f1ed", "#52514e"
        status_text = "NOT SURVEYED THIS SEASON"

    st.markdown(
        f"""
        <div style="border:1px solid #d5d2c8;border-radius:10px;padding:16px 18px;background:#ffffff;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
            <div>
              <span style="font-size:1.22rem;font-weight:800;color:#181816;">{muni_name}, Antioquia</span>
              <span style="font-size:0.85rem;color:#6e6b63;margin-left:8px;">Public Municipal Release Fact Sheet (Open Layers Only)</span>
            </div>
            <span style="background:{badge_bg};color:{badge_fg};padding:4px 12px;border-radius:999px;font-size:0.78rem;font-weight:700;">
              {status_text}
            </span>
          </div>
          <hr style="border:none;border-top:1px solid #eceae2;margin:10px 0;" />
          <div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:10px;font-size:0.88rem;">
            <div style="background:#faf9f5;padding:8px 10px;border-radius:6px;border:1px solid #eae7de;">
              <div style="font-size:0.74rem;color:#6e6b63;font-weight:600;">ZONES DECLARED FREE</div>
              <div style="font-size:1.08rem;font-weight:700;color:#181816;margin-top:2px;">{z_free} / {z_tot} <span style="font-size:0.8rem;font-weight:500;color:#52514e;">({z_surv} surveyed)</span></div>
            </div>
            <div style="background:#faf9f5;padding:8px 10px;border-radius:6px;border:1px solid #eae7de;">
              <div style="font-size:0.74rem;color:#6e6b63;font-weight:600;">AREAS RELEASED (SEASON)</div>
              <div style="font-size:1.08rem;font-weight:700;color:#181816;margin-top:2px;">{a_rel} of {a_tot} <span style="font-size:0.8rem;font-weight:500;color:#52514e;">areas released ({a_clr} cleared); its areas cover {km2:.0f} km²</span></div>
            </div>
            <div style="background:#faf9f5;padding:8px 10px;border-radius:6px;border:1px solid #eae7de;">
              <div style="font-size:0.74rem;color:#6e6b63;font-weight:600;">REMAINING SUSPECTED / CONFIRMED</div>
              <div style="font-size:1.08rem;font-weight:700;color:#181816;margin-top:2px;">{a_susp} areas</div>
            </div>
            <div style="background:#faf9f5;padding:8px 10px;border-radius:6px;border:1px solid #eae7de;">
              <div style="font-size:0.74rem;color:#6e6b63;font-weight:600;">RECORDED VICTIMS (ALL YEARS)</div>
              <div style="font-size:1.08rem;font-weight:700;color:#181816;margin-top:2px;">{victims:,}</div>
            </div>
          </div>
          <div style="margin-top:10px;font-size:0.82rem;color:#52514e;display:flex;justify-content:flex-end;flex-wrap:wrap;gap:6px;">
            <span style="color:#6e6b63;">Disclosure check: No zone ID, area ID, team code or coordinate shown</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_public_summary(root: Path) -> None:
    st.markdown(
        '<div class="public-banner">'
        "<span><strong>Public Summary</strong> · Open information layers only</span>"
        "<span>No operational zone, area or field team is named · Victims shown only as municipal counts</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.header("Public Summary by Municipality")
    st.markdown(
        f"The public picture of the {len(config.PILOT)} pilot municipalities of Antioquia at the end of the season. "
        "Only municipal counts are shown: no operational zone, hazardous area or field team is named, "
        "no geographic coordinates are kept, and victims appear solely as counts by municipality."
    )

    bundle = get_twin_bundle(root)
    public_df = bundle.get("public")

    if public_df is None or public_df.empty:
        st.warning("The season-end municipal summary is unavailable.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pilot Municipalities", int(len(public_df)))
    c2.metric("Zones Declared Free", f"{int(public_df['zones declared free'].sum())} of {int(public_df['zones'].sum())}")
    c3.metric("Areas Released", f"{int(public_df['areas released'].sum())} ({int(public_df['areas cleared'].sum())} cleared)")
    c4.metric("Recorded Victims (All Years)", f"{int(public_df['recorded victims (all years)'].sum()):,}")

    tab_board, tab_cards, tab_table = st.tabs(
        [
            "🗺️ Municipal Overview Board",
            "📄 Public Municipal Fact-Sheet Cards",
            "📋 Municipal Summary & Pilot Totals Table",
        ]
    )

    with tab_board:
        st.subheader("Municipal Overview Board")
        fig_tiles = grid_map.plot_public_municipality_tiles(public_df)
        st.pyplot(fig_tiles)
        plt.close(fig_tiles)

    with tab_cards:
        st.subheader("Public Municipal Fact-Sheet Cards (Open Layers Only)")
        st.caption(
            f"Select a municipality (or view all {len(public_df)}) to read its row of `twin.municipal_summary()`. "
            "No zone ID, area ID, team code, or coordinate appears."
        )
        all_label = f"All {len(public_df)} Pilot Municipalities"
        muni_options = [all_label] + list(public_df.index)
        chosen_muni = st.selectbox("Select municipality fact sheet:", options=muni_options, index=0)
        munis_to_show = list(public_df.index) if chosen_muni == all_label else [chosen_muni]

        for m_name in munis_to_show:
            _render_municipality_public_card(str(m_name), public_df.loc[m_name])

    with tab_table:
        st.subheader("Municipal Summary Table (`twin.municipal_summary()`)")
        st.dataframe(public_df, width="stretch")

        numeric_cols = [
            "zones",
            "zones surveyed",
            "zones declared free",
            "areas",
            "km²",
            "areas released",
            "areas cleared",
            "areas suspected or confirmed",
            "recorded victims (all years)",
        ]
        present_num = [c for c in numeric_cols if c in public_df.columns]
        totals = public_df[present_num].sum().to_frame(f"The pilot ({len(public_df)} municipalities)").T
        totals["declared free (week)"] = f"{int((public_df['declared free (week)'] != '').sum())} of {len(public_df)}"
        st.subheader("Pilot Totals")
        st.dataframe(totals, width="stretch")

    operational_analytics.render_public_release_pack_exporter(public_df, bundle)
