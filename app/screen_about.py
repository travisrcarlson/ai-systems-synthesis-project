"""Screen 8: About — what the application is and is not, the data rules, how it was made, and the run it shows."""
from pathlib import Path

import pandas as pd
import streamlit as st

from twin import layers
from loaders import load_gate_table, load_run_info

# The criteria and season settings `runs/run_info.json` records, in plain words (meanings from twin/config.py)
CRITERIA_LABELS = {
    "min_years": "Years of safe use each source must report (cancellation)",
    "sources": "Reliable first-hand sources needed (cancellation)",
    "follow_up_visits": "Visits before weaker indirect evidence becomes a target for an intrusive check",
    "recorded_accident_needs_ts": "A recorded accident inside needs an intrusive check before release",
    "technical_survey": "Technical survey available (false in the main policy: hazardous areas go to clearance)",
}
SEASON_LABELS = {
    "weeks": "Season weeks",
    "days_per_week": "Working days per week",
    "survey_teams": "Survey teams",
    "clearance_teams": "Clearance teams",
    "ts_teams": "Technical survey teams",
    "rainy_months": "Rainy months",
}


def render(root: Path) -> None:
    st.caption("Application disclosure, rules and run provenance")
    st.header("About this application")

    n_reviews = len(list((root / "runs" / "reviews").glob("*_result.json")))
    n_sitreps = len(list((root / "runs" / "sitreps").glob("w*_sitrep.json")))

    st.markdown(
        f"""
### What this application is and is not
- **What it is:** A small, local, **read-only** viewer for the simulated season of land release across 15 municipalities of Antioquia, Colombia. It lets a reader inspect the season the way the people in the programme would: the weekly operations picture, the land release pipeline, the first plan and its completion forecast, the single operational log, the {n_reviews} dossier reviews with their sign-off sheets, the {n_sitreps} weekly situation reports, the public municipal summary, a library of the project's sources and documents, and, on a screen of its own, the evaluation against the simulation's hidden truth.
- **What it is not:** This is a viewer of a **simulated** season and is **not for operational or field use** in mine action. In this project, land is released only on evidence that survey or clearance gathers, as IMAS 07.11 sets out, and a person signs off every decision. A model's chance or recommendation is not evidence, and no area is released on it.
- **Relationship to the graded work:** The repository's notebook (`integrated_system.ipynb`) is the graded work and is complete on its own. This application in `app/` is a disclosed extra built on top of the frozen `twin/` package and the saved outputs in `runs/` and `models/`.
"""
    )

    st.divider()
    st.subheader("Data, safety and disclosure rules (in plain words)")
    st.markdown(
        """
1. **Write only inside `app/`.** Nothing outside `app/` is changed.
2. **Read-only.** The application never writes to `runs/`, `models/`, `figures/` or `data/`, never calls a language model, and never reads an API key. It never fits the risk models: it loads them from `models/`. Rebuilding the season fits the report checker's isolation forest on a clean simulated season, as the twin does (`setup` in `twin/evaluate.py`). When rebuilding the season, it passes `client=None` so that `twin` reads only the saved dossier reviews and situation reports on disk.
3. **No coordinates, ever.** The application never reads `data/` itself: it uses the twin's own loaders, which set the coordinates aside on loading. It shows no maps with axes, scale or geographic coordinates.
4. **Victims only as counts by municipality.** Victims appear only as counts by municipality (some panels split them by year, condition or outcome), never by person or place, and are never used as weights inside the integer programme.
5. **Restricted layers stay inside the twin.** Operational zones, hazardous areas, field teams, community consultation and the access and security feed appear only on screens headed *"Inside the twin (restricted)"*. The **Public summary** screen shows only `twin.municipal_summary()`, naming no zone, area or team.
6. **The simulation's hidden truth is read only by the Evaluation screen.** The simulation's truth comes from RELand's labels (cells that intersect a confirmed hazardous area or a cleared area where landmines were found), plus any contamination a scenario injects. The truth itself, the multi-seed replay records, the measures against the truth and the expert review sheet are read only by the **Evaluation** screen, under *"The simulation's hidden truth, for evaluation only"*; the other screens get the rebuilt season without them.
7. **Business level.** No technical data on munitions and no newly generated descriptions of explosive items: the application shows the saved texts and records as they were written. The contamination categories are the national demining register's own (MAP, MUSE and AEI), with no device models, mechanisms or calibres.
8. **Local and private.** The application runs on `localhost` offline, with browser usage statistics disabled and no outside scripts, fonts or network requests.
"""
    )

    st.divider()
    st.subheader("How it was made")
    st.markdown(
        """
- **The `twin/` package, rules, prompts and `integrated_system.ipynb`:** Written by **Claude (Anthropic)** under **Travis's** direction, as disclosed in the repository's root `README.md`.
- **The web application (`app/`):** Built by **Travis** with **Google Antigravity**, an AI coding tool, which generated the code from the project's build briefs at Travis's direction. The application reads the frozen `twin/` package and the saved season records without changing any file outside `app/`. **Reviewed and corrected by Claude (Anthropic)** at Travis's request: Claude rebuilt the contamination categories on the national register, took the week-by-week states from the twin's own records, corrected labels and counts, and checked the application's statements against the twin and its sources.
"""
    )

    st.divider()
    st.subheader("The run this application shows (`runs/run_info.json`)")
    st.markdown(
        "The project is designed with **room for better models** (notebook section 11.5). On this screen, the model "
        "names are read from the saved run records (`runs/run_info.json` for the reviewer and report model, "
        "`models/gate.csv` for the risk models), so a later run with a newer model appears here without changing "
        "this screen."
    )

    run_info = load_run_info(str(root))
    if not run_info:
        st.warning("`runs/run_info.json` was not found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reviewer and report model", str(run_info.get("model", "—")))
        c2.metric("Reasoning effort", str(run_info.get("reasoning_effort", "—")))
        c3.metric("Run finished (UTC)", str(run_info.get("finished", "—"))[:19].replace("T", " "))
        c4.metric("Saved reviews / reports", f"{n_reviews} / {n_sitreps}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Policy, season and national criteria (`runs/run_info.json`)**")
            season_cfg = run_info.get("season", {})
            criteria_cfg = run_info.get("criteria", {})
            rows = [{"Setting": "Policy", "Value": str(run_info.get("policy", "—"))}]
            for key, label in SEASON_LABELS.items():
                val = season_cfg.get(key, "—")
                rows.append({"Setting": label, "Value": ", ".join(str(v) for v in val) if isinstance(val, list) else str(val)})
            for key, label in CRITERIA_LABELS.items():
                rows.append({"Setting": label, "Value": str(criteria_cfg.get(key, "—"))})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("The national criteria are invented for this project, since IMAS leaves such numbers to each "
                       "national authority (`twin/config.py`).")

        with col_b:
            st.markdown("**SHA-256 checksums of the prompts and rules the run used (`runs/run_info.json`)**")
            sha = run_info.get("sha256", {})
            if sha:
                st.dataframe(pd.DataFrame([{"File": k, "SHA-256": str(v)} for k, v in sha.items()]),
                             width="stretch", hide_index=True)

    st.divider()
    st.subheader("Area risk model gate by municipality (`models/gate.csv`)")
    st.caption(
        "In each municipality, the baseline-first gate compares P3's two models, logistic regression and "
        "gradient-boosted trees, with the baseline, nearness to past landmine accidents, on the other 14 "
        "municipalities only: ranking (mean area-level PR-AUC) and calibration (how far expected mine areas fall "
        "from observed ones). A model is used only if it beats the baseline on both; otherwise the baseline is used."
    )
    gate_df = load_gate_table(str(root))
    if not gate_df.empty:
        st.dataframe(gate_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Information layers of the digital twin (`twin.layers.layer_table()`)")
    st.dataframe(layers.layer_table(), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Self-check of the application's code against the build brief's rules")
    st.caption(
        "A text search of the application's own files, run each time this screen opens. Each row says which files "
        "it searches and for which strings. Finding none of them does not show that a rule holds; it shows only that "
        "these strings do not appear. Rules that a text search cannot test are marked \"not tested here\"."
    )
    app_dir = root / "app"
    py_files = sorted(app_dir.glob("*.py"))
    texts = {p.name: p.read_text(encoding="utf-8") for p in py_files}
    others = [p for p in py_files if p.name != "screen_about.py"]
    # loaders.py rebuilds the season and holds its truth for the Evaluation screen (`get_evaluation_objects`)
    other_screens = [p for p in others if p.name not in ("screen_evaluation.py", "loaders.py")]
    all_but_this = "all but this one, which holds the search strings"

    def check_row(rule, files, which, tokens):
        found = [p.name for p in files if any(tok in texts[p.name] for tok in tokens)]
        return {
            "Rule": rule,
            "What the scan searches": f"{len(files)} Python files in `app/` ({which}), for "
                                      + ", ".join(f"`{tok}`" for tok in tokens),
            "Result": "✅ none found" if not found else f"❌ found in {', '.join(found)}",
        }

    cfg_path = app_dir / ".streamlit" / "config.toml"
    cfg_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else ""
    cfg_ok = "gatherUsageStats = false" in cfg_text and 'address = "localhost"' in cfg_text

    rows = [
        check_row("Read-only: no language model, API key or replay", others, all_but_this,
                  ("OPENAI_API_KEY", "import openai", "evaluate.replay(", "evaluate.run_case(")),
        check_row("No coordinates: no direct reads of `data/`", others, all_but_this,
                  ("landmine_detection.csv", "LATITUD_Y", "LONGITUD_X", "latitudcabecera", "longitudcabecera",
                   "read_csv(DATA", "DATA_DIR", "data_dir")),
        check_row("No writes", others, all_but_this,
                  (".to_csv(\"", ".to_csv('", "savefig(", "write_text(", "write_bytes(", "open(\"")),
        check_row("Local: no network calls", others, all_but_this,
                  ("requests.", "urllib", "http.client", "socket.")),
        check_row("The hidden truth read only by the Evaluation screen", other_screens,
                  "all but this one, the Evaluation screen and `loaders.py`, which rebuilds the season and holds "
                  "its truth for the Evaluation screen",
                  ("runs/replay", "\"replay\"", "expert_review.csv", "result.mined", "result.base_mined",
                   "get_evaluation_objects")),
        {"Rule": "Usage statistics off, localhost only",
         "What the scan searches": "`app/.streamlit/config.toml`, for `gatherUsageStats = false` and "
                                   "`address = \"localhost\"`",
         "Result": "✅ set" if cfg_ok else "❌ not set"},
        {"Rule": "Victims as counts; restricted layers on restricted screens; business level; no fit of the risk "
                 "models; written only in `app/`",
         "What the scan searches": "—", "Result": "not tested here"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
