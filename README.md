# A Digital Twin of Land Release in Antioquia, with Decision Services from Six Projects

A digital twin of land release in the 15 Antioquia municipalities of the RELand study, with decision services at three levels above it: a strategic view of where harm continues without recorded clearance work (from P1 and P2), an operational ranking of where to survey first, with a simple baseline first (P3), a photo screener that answers "unknown" when unsure, stress-tested with generated images (P4 and P5), and a review agent that recommends what happens to each area after a survey visit, checked by coded rules and signed off by a person before the twin changes (P6). A replay of eight rounds of survey in the pilot, where RELand's labels show what each area held, measures what the system would have released, found and missed. This is the seventh project of my AI capstone on humanitarian mine action.

> **Status (24 October 2026):** built and tested in Claude's workspace without an API key; the smoke test and the full run with the agent are next. The results, the expert review, the model cards, the risk register and the paper come after the full run.

## What's in this repository
- `integrated_system.ipynb`: the notebook. Context and boundaries; the architecture; setup; the data; the twin; the strategic, operational, field-evidence and tactical services; the replay and its evaluation; failure cases. Sections 11 and 12 are written after the full run.
- `twin/`: the package that holds every component, used by the notebook: `data.py` (loading and checking the data), `core.py` (the twin), `strategic.py`, `prioritise.py`, `screener.py`, `generator.py`, `agent.py` (the review agent), `rules.py` (the rule check and the reference reviewer), `field.py` (the simulated field of the replay) and `replay.py`.
- `rules/land_release_rules.md`: the rules the agent looks up, paraphrased from IMAS 07.11, 08.10 and 08.20 with their section numbers, and the project's own rules, marked as such.
- `prompts/system_prompt.md`: the agent's persona and instructions.
- `data/`: the five public datasets, copied unchanged from P1 to P5, with sources, licences and checksums in `data/README.md`; `docs/datasheets/` has a datasheet for each.
- `models/`: what the first run fits and trains, saved so that later runs give the same results: the screener's and the generator's weights, every area's ranking scores and the gate's table, and the screener's readings of the replay's stand-in photos.
- `runs/`: the agent's reviews in the full run (a trace, a result and a sign-off sheet for each), saved by the full run.
- `requirements.txt`: the Python environment, from `pip freeze`.

## How to run
Needs Python 3.13 and Git. `requirements.txt` was generated on Python 3.13.0, with the CPU build of PyTorch.

```
git clone https://github.com/travisrcarlson/ai-systems-synthesis-project.git
cd ai-systems-synthesis-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
jupyter notebook integrated_system.ipynb
```

On macOS or Linux, activate with `source .venv/bin/activate`; on macOS, also change `torch==2.14.1+cpu` in `requirements.txt` to `torch==2.14.1` before installing, since PyTorch publishes no `+cpu` build for macOS. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. In Jupyter, choose **Kernel > Restart Kernel and Run All Cells**, or run the whole notebook from the terminal:

```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 integrated_system.ipynb
```

**The first run** fits the operational models and their gate (about 3 minutes on two CPU threads in this project's tests) and trains the photo screener and the image generator (about 15 minutes each), and saves the results in `models/`; later runs load them.

**Without an API key**, the notebook runs everything except the agent's calls to the model. It tests the agent's tools and safeguards with a scripted stand-in model, and section 10 reads the agent's saved reviews from `runs/`; before the full run there are none, and section 10 shows the coded rules alone.

**With an OpenAI API key**, set it as an environment variable named `OPENAI_API_KEY` before starting Jupyter (in PowerShell, for one window: `$env:OPENAI_API_KEY = "..."`). Never put the key in the notebook or the repository; the notebook reads it from the environment and never prints or saves it. Section 10 then runs the agent on each of the replay's 160 survey visits and saves every review in `runs/`. A saved review with the same inputs (the dossier, the prompt, the rules, the code and the settings) is reused, so a stopped run resumes where it stopped; to run the agent again from the start, delete `runs/`. Section 10.8 reports the run's time and cost; the full run's figures will be added here. To try it first on five reviews, into `runs_smoke/`, set the environment variable `SMOKE_TEST` to `1` before starting (PowerShell: `$env:SMOKE_TEST = "1"`).

## How this project is made
I chose the use case, the country and the pilot, the digital-twin design and its three levels, and every setting of the simulation that the notebook marks as mine, one question at a time; I approved the design. Claude (Anthropic) researched the sources, wrote the `twin/` package, the rules, the prompt and the notebook, and tested them in its workspace with a scripted stand-in model, without a key. I run the smoke test and the full run on my own computer with my OpenAI API key, judge a sample of the agent's reviews, and review every change before it is pushed.
