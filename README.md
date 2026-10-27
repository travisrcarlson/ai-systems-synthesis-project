# A Digital Twin That Plans and Runs Land Release in Antioquia, Colombia

A digital twin of land release in the 15 Antioquia municipalities of the RELand study, which helps people plan a season of survey and clearance and run it week by week. One state model of the pilot (municipality, operational zone, area and task, with the teams and the calendar) is kept in step with the teams' reports. The twin changes what the teams do, and the status of land, only through decisions a person approves, and it records every report and every decision in one log. Overlays read the twin and recommend: data layers, each registered with its source, licence and sensitivity, and six decision tools across machine learning, optimisation, statistics, generative AI and agentic AI, each of which can answer "insufficient evidence". A season replay against a simulated field, 20 seeds for each policy, measures how the system behaves, and five injected scenarios test it when things go wrong. The six earlier projects of my AI capstone on humanitarian mine action appear as insights. This is the seventh project.

> **Status (27 October 2026):** built, reviewed, smoke-tested and run in full on my computer with my OpenAI API key; I have judged a sample of the agent's reviews and of the model's situation reports (`runs/expert_review.csv`). Next: sections 11 and 12 of the notebook, written from the results.

## What's in this repository
- `integrated_system.ipynb`: the notebook, in 12 sections: the industry context, problem, users and boundaries; the architecture; setup; the data and data layers; the twin; planning; operations; release (the dossier reviewer, the rule check and sign-off); the season replay and its evaluation; the injected scenarios; failure cases, limitations and governance; and the summary. Sections 11 and 12 are written after the full run.
- `twin/`: the package that holds every component; the notebook calls it and shows its outputs.

  | Module | What it holds |
  |---|---|
  | `config.py` | the pilot, the national criteria, the season's settings and the simulation's assumptions, each marked as my setting or the project's |
  | `data.py` | loading and checking the three datasets |
  | `core.py` | the twin: the state model, the calendar, reports, decisions, the one log and the read-only view |
  | `layers.py` | the layer registry, and the twin built from it |
  | `overlays.py` | the overlay contract: a recommendation with its evidence, uncertainty and limits, or "insufficient evidence" |
  | `prioritise.py` | risk prioritisation: the baseline-first gate on ranking and calibration, and learning in the season |
  | `factors.py` | the planning factors, from simulated flagged areas |
  | `planner.py` | the season planner: the integer programme and the rule of thumb |
  | `forecast.py` | the completion forecaster (Monte Carlo) |
  | `checker.py` | the report checker: coded checks and an isolation forest |
  | `agent.py`, `review.py`, `rules.py` | the dossier reviewer agent with its tools and safeguards; the reviewers; the rule check |
  | `sitrep.py` | the weekly situation report and its figure check |
  | `season.py` | the monthly and weekly loops of a season, and the season's measures |
  | `field.py` | the simulated field, which stands in for the teams; with the risk models' fit before the season (each municipality's chances from the other 14 municipalities' labels) and the evaluation, the only code that reads RELand's labels |
  | `scenarios.py` | the five injected scenarios |
  | `evaluate.py` | the season replay: the policies, the seeds, the parallel runs and the saved records |
  | `standins.py` | the stand-in models that test the agent's code path without a key |

- `rules/land_release_rules.md`: the rules the agent looks up, paraphrased from IMAS 07.11, 08.10 and 08.20 with their section numbers, and the project's own rules, marked as such.
- `prompts/system_prompt.md` (the dossier reviewer) and `prompts/sitrep_prompt.md` (the situation report).
- `data/`: three public datasets, copied unchanged from P1, P2 and P3, with sources, licences and checksums in `data/README.md`; `docs/datasheets/` has a datasheet for each.
- `figures/`: the notebook's figures, saved when it runs.
- `models/`: the risk models' scores, the gate's table and the calibration, fitted by the first run and saved so that later runs give the same results.
- `runs/`: saved by the full run: the agent's reviews (`runs/reviews/`, a trace, a result and a sign-off sheet for each), the situation reports (`runs/sitreps/`), the season replay's records (`runs/replay/`), and the agent's season's reviews, log and run information. `runs/expert_review.csv` holds my expert review: the 2 reviews that differ from the coded rules, and a sample drawn at random with a fixed seed, 12 of the 94 that match them (4 of each outcome) and 6 of the 52 situation reports; for each, the question I was asked and its options, my answer word for word, and Claude's notes, checked against the saved files.
- `requirements.txt`: the Python environment, from `pip freeze`.

## How to run
Needs Python 3.13 and Git. `requirements.txt` was generated on Python 3.13.0.

```
git clone https://github.com/travisrcarlson/ai-systems-synthesis-project.git
cd ai-systems-synthesis-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook integrated_system.ipynb
```

On macOS or Linux, activate with `source .venv/bin/activate`. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. In Jupyter, choose **Kernel > Restart Kernel and Run All Cells**, or run the whole notebook from the terminal:

```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 integrated_system.ipynb
```

**The first run** fits the risk models and their gate and saves them in `models/` (about 3 minutes on two processors in this project's tests; a refit there gave identical files, while on another computer the last digits can differ, which is why the fit is saved), and runs the season replay: 200 seasons (5 policies and 5 scenarios, 20 field seeds each) in parallel worker processes, one fewer than the computer's processors and at most 8 (about 40 minutes with two worker processes in this project's tests). Each season is saved in `runs/replay/` as soon as it finishes, so a stopped run resumes where it stopped, and a later run reads the seasons back unless the code, the rules, the prompts, the fitted models, the data or the libraries' versions have changed. To choose the number of worker processes, set the environment variable `N_JOBS` before starting (PowerShell: `$env:N_JOBS = "4"`); `1` runs the seasons one after another.

**Without an API key**, the notebook runs everything except the model's calls: the agent's tools and safeguards are tested with a scripted stand-in model (section 8), and section 9.2 reads the agent's saved reviews from `runs/`; before the full run there are none, and section 9.2 says so.

**With an OpenAI API key**, set it as an environment variable named `OPENAI_API_KEY` before starting Jupyter (PowerShell, for one window: `$env:OPENAI_API_KEY = "..."`). Never put the key in the notebook or the repository: the notebook only checks whether it is set, and never prints or saves it. Section 9.2 then runs the agent on every dossier of seed 0's season (about 100 reviews), and the model drafts each week's situation report (52); both are saved in `runs/`. A saved review or report with the same inputs is reused, so a stopped run resumes where it stopped; to run the agent again from the start, delete `runs/reviews/`. Section 9.2 reports the run's time and cost.

**The smoke test.** To try it first, set the environment variable `SMOKE_TEST` to `1` before starting (PowerShell: `$env:SMOKE_TEST = "1"`). The agent then makes 5 reviews and the model drafts 2 reports, the coded rules and the coded draft take over after that, and the season replay runs on 2 seeds instead of 20; everything is saved in `runs_smoke/`, which is not committed. Without a key, the whole smoke test took about 10 minutes on two processors in this project's tests, including the models' fit; with a key, on my computer with 8 worker processes, about 20 minutes, and the model's calls cost about 1.5 US cents.

## How this project is made
I chose the use case, the country and the pilot, the digital-twin design (a digital shadow with approved write-back, for planning and operations), and every setting of the simulation that the notebook marks as mine, one question at a time; I approved the design. Claude (Anthropic) researched the sources, wrote the `twin/` package, the rules, the prompts and the notebook, and tested them in its workspace with scripted stand-in models, without a key. I run the smoke test and the full run on my own computer with my OpenAI API key, judge a sample of the agent's reviews and of the model's situation reports, and review every change before it is pushed.
