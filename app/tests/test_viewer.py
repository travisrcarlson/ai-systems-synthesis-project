"""Read-only regression suite for the local Streamlit viewer in app/."""
import gc
import io
import logging
import os
import py_compile
import sys
from pathlib import Path

for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from streamlit.testing.v1 import AppTest

import clearance_pipeline
import grid_map
import interactive_map
import literature_library
import loaders
import ordnance_library
import screen_log


def main() -> None:
    # 1. Compile all 18 top-level modules in app/ plus this test file
    py_files = sorted(APP.glob("*.py"))
    assert len(py_files) == 18, f"Expected 18 py files in app/, got {len(py_files)}"
    for p in py_files + [Path(__file__)]:
        py_compile.compile(str(p), doraise=True)
        if p in py_files and p.name != "screen_about.py":
            assert "importlib.reload" not in p.read_text(encoding="utf-8"), p.name

    # 2. Check key literature catalogue, citation, and cached resource invariants
    assert len(literature_library.LITERATURE_CATALOGUE) == 17
    n_refs = sum(1 for x in literature_library.LITERATURE_CATALOGUE if x["kind"] == "reference")
    n_data = sum(1 for x in literature_library.LITERATURE_CATALOGUE if x["kind"] == "dataset")
    assert (n_refs, n_data) == (15, 2), (n_refs, n_data)
    assert "Mine Action Review (2025)" in ordnance_library.MAR_NOTE
    assert "IMAS 09.30" in ordnance_library.MAR_NOTE

    fig_t = literature_library.plot_literature_traceability_diagram()
    plt.close(fig_t)

    log_df = loaders.load_saved_log(str(ROOT))
    fig_r = screen_log._plot_roles_and_writeback_ledger(log_df)
    plt.close(fig_r)

    sitreps = loaders.load_saved_sitreps(str(ROOT))
    assert len(sitreps) == 52, len(sitreps)
    assert loaders.peek_twin_bundle(ROOT) is None
    print("STATIC_AND_STRUCTURAL_CHECKS_OK")

    # 3. Live AppTest traversal across screens & segmented sections
    warn_buf = io.StringIO()
    handler = logging.StreamHandler(warn_buf)
    handler.setLevel(logging.WARNING)
    st_logger = logging.getLogger("streamlit")
    st_logger.addHandler(handler)
    try:
        at = AppTest.from_file(str(APP / "app.py"), default_timeout=240)

        # Screen 8: runs the live self-check scans from screen_about.py
        at.session_state["app_nav_screen"] = "8. About & Data Rules"
        at.run()
        assert not at.exception, f"Screen 8 exception: {at.exception}"
        self_check_df = at.dataframe[-1].value
        assert not self_check_df["Result"].astype(str).str.contains("❌").any(), self_check_df
        print("SCREEN_8_OK")

        # Screen 0: Command & Planning Overview (warms cached bundle once)
        at.sidebar.radio[0].set_value("Command & Planning Overview").run()
        assert not at.exception, f"Overview exception: {at.exception}"
        metric_labels = [m.label for m in at.metric]
        assert "Plans Approved" in metric_labels, metric_labels
        assert "Monthly Plans Approved" not in metric_labels
        for m in at.metric:
            if m.label in (
                "Zones Declared Free",
                "Areas Released",
                "Sign-Off Reviews Completed",
                "Plans Approved",
                "Recorded Victims (All Years)",
            ):
                assert m.proto.color == 2, f"{m.label} proto.color={m.proto.color}"
                assert m.proto.direction == 2, f"{m.label} proto.direction={m.proto.direction}"
        print("SCREEN_OVERVIEW_OK")

        # Interactive map payload & week-dependent grid subtitle invariants
        bundle = loaders.peek_twin_bundle(ROOT)
        assert bundle is not None, "Expected cached bundle after running Command & Planning Overview"
        assert interactive_map.AREA_CODE["not_suspected"] == 1
        assert interactive_map.AREA_CODE["cleared"] == 6
        reviews_df = bundle["reviews_public_df"]
        gate_df = loaders.load_gate_table(str(ROOT))
        payload = interactive_map._build_interactive_payload(bundle, log_df, reviews_df, gate_df)
        assert len(payload["muni_k"]) == 52 and len(payload["muni_k"][0]) == 15
        k52 = payload["muni_k"][51]
        max_diff = 0.0
        for a in payload["areas"]:
            p1 = a["p1"]
            k = k52[a["m"]]
            pw52 = p1 if (k == 1.0 or p1 <= 0 or p1 >= 1) else (k * p1) / (1.0 - p1 + k * p1)
            max_diff = max(max_diff, abs(pw52 - a["p52"]))
        assert max_diff < 1e-3, f"pw(52) vs p52 max diff={max_diff}"
        assert any(any(ev[1] == 1 for ev in a["ch"]) for a in payload["areas"])

        fig_g = grid_map.plot_all_municipalities_grid(bundle, "Area status in selected week", 10, log_df)
        subtitles = [t.get_text() for ax in fig_g.axes for t in ax.texts]
        plt.close(fig_g)
        assert len(subtitles) == 15
        assert all("at w10" in s for s in subtitles) and not any("at w52" in s for s in subtitles), subtitles[:3]
        print("MAP_AND_GRID_OK")
        gc.collect()

        # Pipeline screen at week 10: verify no future dossier review leak
        at.sidebar.radio[0].set_value("🔄 Land Release & Clearance Pipeline").run()
        assert not at.exception, f"Pipeline exception: {at.exception}"
        week_sliders = [s for s in at.slider if "week" in s.label.lower()]
        assert week_sliders, "Expected week slider on Pipeline screen"
        week_sliders[0].set_value(10).run()
        assert not at.exception, f"Pipeline week 10 exception: {at.exception}"
        pipe_10 = clearance_pipeline.build_area_pipeline_profiles(str(ROOT), week=10)
        active_10 = pipe_10[pipe_10["has_dossier_or_task"]].index.astype(str).tolist()
        area_sel = at.selectbox(key="pipeline_area_sel_main")
        no_rev_w10_aid = next(
            (
                aid
                for aid in active_10
                if any(opt.startswith(aid + " ") for opt in area_sel.options)
                and reviews_df[(reviews_df["area_id"].astype(str) == aid) & (reviews_df["week"] <= 10)].empty
                and not reviews_df[reviews_df["area_id"].astype(str) == aid].empty
            ),
            None,
        )
        assert no_rev_w10_aid is not None
        area_sel.set_value(no_rev_w10_aid).run()
        assert not at.exception
        disabled_btns = [b for b in at.button if f"No Dossier Review Yet for {no_rev_w10_aid}" in b.label]
        assert disabled_btns and disabled_btns[0].disabled is True
        print(f"PIPELINE_NO_FUTURE_LEAK_OK: area={no_rev_w10_aid}")

        # Screen 1: Weekly Operations segmented persistence + Municipality Explorer segments
        at.sidebar.radio[0].set_value("1. Weekly Operations").run()
        assert not at.exception, f"Screen 1 exception: {at.exception}"
        ops_seg = at.segmented_control(key="ops_main_sec")
        assert len(ops_seg.options) == 4
        target_ops = "🚜 Field Teams, Attrition & Clearance Timeline"
        ops_seg.set_value(target_ops).run()
        assert not at.exception
        assert at.session_state["ops_main_sec"] == target_ops
        at.slider(key="ops_week_slider").set_value(26).run()
        assert not at.exception
        assert at.session_state["ops_main_sec"] == target_ops

        at.segmented_control(key="ops_main_sec").set_value("🗺️ Regional & Municipality Map").run()
        assert not at.exception
        assert "muni_exp_sec" in at.session_state
        muni_seg = at.segmented_control(key="muni_exp_sec")
        assert len(muni_seg.options) == 5
        muni_seg.set_value("📅 Plan & Forecast").run()
        assert not at.exception
        print("SCREEN_1_SEGMENTS_OK")

        # Screen 2: Plan & Completion Forecast
        at.sidebar.radio[0].set_value("2. Plan & Completion Forecast").run()
        assert not at.exception, f"Screen 2 exception: {at.exception}"
        plan_seg = at.segmented_control(key="plan_main_sec")
        assert len(plan_seg.options) == 6
        plan_seg.set_value("⚖️ Optimised Plan vs. Rule of Thumb").run()
        assert not at.exception
        at.segmented_control(key="plan_main_sec").set_value("⏳ Completion Forecast (P10 / P50 / P90)").run()
        assert not at.exception
        print("SCREEN_2_SEGMENTS_OK")
        gc.collect()

        # Screen 3: Operational Log, the ledger of decisions by role. In runs/season_log.csv, 108 of the 299 decisions
        # cite a tool (96 the dossier reviewer; 12 the risk prioritisation and the season planner); the other 191
        # cite none, saved as "[]".
        at.sidebar.radio[0].set_value("3. Operational Log").run()
        assert not at.exception, f"Screen 3 exception: {at.exception}"
        overlay_metric = next(m for m in at.metric if m.label == "Decisions Citing an Overlay Tool")
        assert overlay_metric.value == "108", f"decisions citing a tool: {overlay_metric.value}"
        at.selectbox(key="log_role_filter_rc").set_value("Only decisions citing an overlay tool").run()
        assert not at.exception, f"Screen 3 filter exception: {at.exception}"
        shown = [c.value for c in at.caption if c.value.startswith("Showing **") and "of **299**" in c.value]
        assert shown and shown[0].startswith("Showing **108** of **299**"), shown
        print("SCREEN_3_ROLE_LEDGER_OK")
        gc.collect()

        # Screen 6: Public Summary
        at.sidebar.radio[0].set_value("6. Public Summary").run()
        assert not at.exception, f"Screen 6 exception: {at.exception}"
        print("SCREEN_6_OK")

        # Screen 7: Evaluation (Hidden Truth) — iterate through all 5 segmented sections
        at.sidebar.radio[0].set_value("7. Evaluation (Hidden Truth)").run()
        assert not at.exception, f"Screen 7 exception: {at.exception}"
        assert "eval_main_sec" in at.session_state
        eval_seg = at.segmented_control(key="eval_main_sec")
        assert len(eval_seg.options) == 5
        for opt in list(eval_seg.options):
            at.segmented_control(key="eval_main_sec").set_value(opt).run()
            assert not at.exception, f"Screen 7 segment {opt!r} exception: {at.exception}"
        print("SCREEN_7_ALL_5_SEGMENTS_OK")

        warnings_text = warn_buf.getvalue().strip()
        assert "widget with key" not in warnings_text.lower(), f"Session state warning found:\n{warnings_text}"
    finally:
        st_logger.removeHandler(handler)

    print("ALL_VIEWER_TESTS_PASSED")


if __name__ == "__main__":
    main()
