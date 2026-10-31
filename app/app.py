"""Read-only local Streamlit viewer for the Antioquia humanitarian mine action digital twin."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib
import matplotlib
matplotlib.use("Agg")
import streamlit as st

import loaders
import clearance_pipeline
import grid_map
import grounding_panels
import interactive_map
import literature_library
import operational_analytics
import ordnance_library
import screen_about
import screen_dashboard
import screen_evaluation
import screen_log
import screen_operations
import screen_plan
import screen_public
import screen_reviews
import screen_sitreps

for _mod in (
    loaders,
    clearance_pipeline,
    grid_map,
    grounding_panels,
    interactive_map,
    literature_library,
    operational_analytics,
    ordnance_library,
    screen_about,
    screen_dashboard,
    screen_evaluation,
    screen_log,
    screen_operations,
    screen_plan,
    screen_public,
    screen_reviews,
    screen_sitreps,
):
    importlib.reload(_mod)

st.set_page_config(
    page_title="Antioquia Land Release Programme",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    loaders.apply_editorial_style()

    ok, missing = loaders.check_prerequisites(ROOT)
    if not ok:
        st.error(
            "**Required saved files are missing.** This read-only viewer never fits risk models "
            "or calls a language model. Please ensure the following paths exist before starting the viewer:\n\n"
            + "\n".join(f"- `{item}`" for item in missing)
        )
        st.stop()

    run_info = loaders.load_run_info(str(ROOT))
    model_name = run_info.get("model", "saved model")
    reasoning_effort = run_info.get("reasoning_effort", "")
    finished = str(run_info.get("finished", ""))[:10]

    with st.sidebar:
        st.markdown("### Antioquia Land Release")
        st.caption("52-Week Programme Viewer · 15 Municipalities")

        screens = [
            "Command & Planning Overview",
            "🔄 Land Release & Clearance Pipeline",
            "📚 Literature, Standards & Governance Library",
            "1. Weekly Operations",
            "2. Plan & Completion Forecast",
            "3. Operational Log",
            "4. Dossier Sign-Off Reviews",
            "5. Weekly Situation Reports",
            "6. Public Summary",
            "7. Evaluation (Hidden Truth)",
            "8. About & Data Rules",
        ]
        if st.session_state.get("app_nav_screen") not in screens:
            st.session_state["app_nav_screen"] = screens[0]
        choice = st.radio("View", options=screens, key="app_nav_screen")

        st.divider()
        st.caption(
            f"**Saved Season Run**\n\n"
            f"Reviewer & report model: **{model_name}**"
            + (f" ({reasoning_effort})" if reasoning_effort else "")
            + f"\n\nRun date: **{finished or '—'}**\n\n"
            "Local read-only viewer (simulated season, not for field use)."
        )

    if choice.startswith("Command"):
        screen_dashboard.render_total_dashboard(ROOT)
    elif choice.startswith("🔄"):
        clearance_pipeline.render_clearance_pipeline_studio(ROOT)
    elif choice.startswith("📚"):
        literature_library.render_literature_library_studio(ROOT, key_prefix="sidebar_lit")
    elif choice.startswith("1."):
        screen_operations.render_operations(ROOT)
    elif choice.startswith("2."):
        screen_plan.render_plan_and_forecast(ROOT)
    elif choice.startswith("3."):
        screen_log.render_log(ROOT)
    elif choice.startswith("4."):
        screen_reviews.render_dossier_reviews(ROOT)
    elif choice.startswith("5."):
        screen_sitreps.render_situation_reports(ROOT)
    elif choice.startswith("6."):
        screen_public.render_public_summary(ROOT)
    elif choice.startswith("7."):
        screen_evaluation.render_evaluation(ROOT)
    elif choice.startswith("8."):
        screen_about.render(ROOT)


if __name__ == "__main__":
    main()
