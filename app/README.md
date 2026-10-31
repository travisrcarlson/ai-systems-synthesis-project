# Antioquia Land Release Digital Twin: local viewer (`app/`)

## What this application is and is not

A small, local, **read-only** Streamlit application for the repository `ai-systems-synthesis-project`. It lets a reader
inspect the simulated 52-week season of humanitarian land release in 15 municipalities of Antioquia, Colombia, the way
the people in the programme would.

- **It is** a viewer of a **simulated** season, built on the frozen `twin/` package and the saved outputs in `runs/`
  and `models/`.
- **It is not** for field or operational use. The repository's notebook (`integrated_system.ipynb`) is the graded work
  and is complete on its own; this application is a disclosed extra.

## Screens

Screens marked *restricted* carry the heading "Inside the twin (restricted)".

- **Command & Planning Overview** (*restricted*): the season at a chosen week (zones declared free, areas released,
  sign-offs, plans), an interactive map of the twin's cell grid, the risk model's chance of mines beside each week's
  land release status, the contamination categories, the real-world grounding, the municipality explorer, the
  planning dashboard and the season's progress.
- **Land Release & Clearance Pipeline** (*restricted*): each area's path through the twin's process, from the desk
  study to the decision, with its log entries, reviews and sign-off sheet.
- **Literature, Standards & Governance Library**: the project's references, standards, datasets, model cards,
  datasheets and risk register.
- **1. Weekly Operations** (*restricted*): the plan in force, task orders, team reports, decisions on areas and zones
  with who decided and the rule check, "insufficient evidence" notices, the season's end, and team deployment.
- **2. Plan & Completion Forecast** (*restricted*): the week-1 integer programme beside the rule of thumb, the
  completion forecast (P10, P50, P90; it assumes its plan holds), and the risk overlay's monthly learning.
- **3. Operational Log** (*restricted*): the one log of the season, filterable, and the story of any zone.
- **4. Dossier Sign-Off Reviews** (*restricted*): the dossier reviews, each sign-off sheet headed "Recommendation
  only: a person decides", and the rule check's results.
- **5. Weekly Situation Reports** (*restricted*): each week's draft, the facts it was given and the figure check.
- **6. Public Summary**: `twin.municipal_summary()` only; no zone, area or team named; victims as municipal counts.
- **7. Evaluation (Hidden Truth)**: the 20-seed replay, forecast coverage, critical failures, scenarios, the agent
  against the coded rules and the expert review, under "The simulation's hidden truth, for evaluation only".
- **8. About & Data Rules**: what the application is, the data rules, how it was made, the run it shows, and a scan
  of its own code.

## How to run it

From the repository's root in PowerShell, with Python 3.13:

```powershell
python -m venv app\.venv
.\app\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
cd app
streamlit run app.py
```

Run `streamlit run app.py` from inside `app/` so that `app/.streamlit/config.toml` (`address = "localhost"`,
`gatherUsageStats = false`) applies. The first screen rebuilds the season from the saved reviews and reports once
(`client=None`, no key), which takes about two minutes; after that, screens open at once.

## Data, safety and privacy rules

1. **Write only inside `app/`.** Nothing outside `app/` is changed.
2. **Read-only.** The application never writes to `runs/`, `models/`, `figures/` or `data/`, never calls a language
   model, never reads an API key, and never calls `evaluate.replay` or `evaluate.run_case`. If `models/gate.csv` or
   `runs/reviews/` is missing, it stops with a message and never fits the risk models.
3. **No coordinates.** The application never reads `data/` itself; it uses the twin's own loaders, which set the
   coordinates aside. No map has axes, scale or coordinates.
4. **Victims only as counts by municipality.**
5. **Restricted layers stay inside the twin.** Operational zones, hazardous areas, field teams, community consultation
   and the access feed appear only on screens headed "Inside the twin (restricted)". Tables exported from restricted
   screens are named `RESTRICTED_…`.
6. **The hidden truth stays on the Evaluation screen**: the simulation's truth labels, the `mined` and `reference`
   columns of `runs/season_reviews.csv`, `result.mined`, `result.base_mined`, `result.metrics`, `runs/replay/` and
   `runs/expert_review.csv`.
7. **Business level.** No technical data on munitions and no new text about explosive items; saved texts are shown as
   they are. The contamination categories are the national demining register's own (MAP, MUSE, AEI), with no device
   models, mechanisms or calibres.
8. **Local and private.** It runs on `localhost`, offline, with usage statistics off and no outside fonts, scripts or
   network calls.

## How it was made

- **The `twin/` package, rules, prompts and `integrated_system.ipynb`:** written by **Claude (Anthropic)** under
  **Travis's** direction, as the repository's root `README.md` discloses.
- **This application (`app/`):** built by **Travis** with **Google Antigravity**, reading the frozen `twin/` package
  and the saved season outputs. **Reviewed and corrected by Claude (Anthropic)** at Travis's request: Claude rebuilt
  the contamination categories on the national register, took the week-by-week states from the twin's own records,
  corrected labels and counts, and checked the application's statements against the twin and its sources.
