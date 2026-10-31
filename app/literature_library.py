"""The literature, standards and governance library (read-only).

Six tabs in one studio:
1. The references the notebook cites (its section 12) and the two datasets documented in `docs/datasheets/`,
   searchable, with APA and BibTeX export, and a Matplotlib diagram linking the sources, the earlier projects'
   insights and where they show in the twin.
2. The risk register's table (`docs/risk_register.md`), as written, without its evidence column.
3. The report checker (`twin/checker.py`, Model Card 02): its coded checks, the isolation forest's 12 features, and a
   re-run here of some coded checks on sample reports.
4. RELand's 63 features by municipality, and the baseline-first gate (`models/gate.csv`, `models/calibration.csv`).
5. A reader for the datasheets, the rules file, the prompts and the application's README.
6. The earlier projects as insights (notebook section 2.4).

Data rules (`app/README.md`):
- No coordinates; no map or chart of places has axes, a scale or coordinates (rule 3).
- Victims only as counts by municipality (rule 4).
- No zone, area or team is named (rule 5).
- No hidden truth (rule 6): RELand's cells come from the twin's shared bundle, which holds no labels; the model cards
  and the risk register's evidence column, which include evaluation results, are not shown.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from twin import checker, config, core, data, prioritise
import loaders

SURFACE = "#ffffff"
BG_WARM = "#f6f5f1"
INK = "#1c1e21"
INK_2 = "#52514e"
INK_3 = "#7c7a75"


# =====================================================================================================================
# 1. THE NOTEBOOK'S REFERENCES (SECTION 12) AND THE TWO DATASETS OF `docs/datasheets/`
# =====================================================================================================================
# The notebook's reference list (cell c86) is the source for every citation below. The two datasets are not in that
# list; their entries follow `docs/datasheets/01_victims.md`, `02_demining.md` and `twin/layers.py`.
LITERATURE_CATALOGUE: list[dict] = [
    {
        "ref_id": "LIT-01",
        "kind": "reference",
        "short_cite": "Dulce Rubio et al. (2024)",
        "year": "2024",
        "category": "Peer-reviewed research",
        "title": "RELand: Risk estimation of landmines via interpretable invariant risk minimization",
        "authors": "Dulce Rubio, M., Zeng, S., Wang, Q., Alvarado, D., Moreno Rivera, F., Heidari, H., & Fang, F.",
        "venue": "ACM Journal on Computing and Sustainable Societies, 2(2), Article 23",
        "doi_or_url": "https://doi.org/10.1145/3648437",
        "license_or_edition": "Journal article; its dataset on Hugging Face (mdulcer10/landmine-detection), CC BY 4.0",
        "key_findings": (
            "The study's dataset was released to support research on predicting where landmines are, so that survey "
            "can be prioritised. The public file covers 26,761 grid cells of 500 m × 500 m in 15 municipalities of "
            "Antioquia, with 63 open features (27 continuous, 36 one-hot) on terrain, climate, land use, relief, "
            "population and distances to roads, settlements, services, waterways, coca fields and past events, and a "
            "label from the records of Colombia's national mine action authority (`docs/datasheets/03_reland.md`)."
        ),
        "codebase_grounding": (
            "The twin's hazard and land-use layers. `twin/data.py` loads the 26,761 cells (the coordinates only place "
            "each cell on a grid and are not kept; the labels are kept apart), and `twin/core.py` groups them into "
            "7,280 areas of 2 × 2 cells and 292 operational zones. `twin/prioritise.py` fits P3's two models one "
            "held-out municipality at a time and uses one for a municipality only if it beats the baseline, nearness "
            "to past accidents, on both ranking and calibration (the baseline-first gate, `models/gate.csv`)."
        ),
        "modules": "twin/data.py, twin/core.py, twin/prioritise.py, models/gate.csv, docs/datasheets/03_reland.md",
        "apa_citation": (
            "Dulce Rubio, M., Zeng, S., Wang, Q., Alvarado, D., Moreno Rivera, F., Heidari, H., & Fang, F. (2024). "
            "RELand: Risk estimation of landmines via interpretable invariant risk minimization. ACM Journal on "
            "Computing and Sustainable Societies, 2(2), Article 23. https://doi.org/10.1145/3648437"
        ),
        "bibtex": (
            "@article{dulcerubio2024reland,\n"
            "  author    = {Dulce Rubio, Mateo and Zeng, Siqi and Wang, Qi and Alvarado, Didier and Moreno Rivera, Francisco and Heidari, Hoda and Fang, Fei},\n"
            "  title     = {{RELand}: Risk estimation of landmines via interpretable invariant risk minimization},\n"
            "  journal   = {ACM Journal on Computing and Sustainable Societies},\n"
            "  volume    = {2},\n"
            "  number    = {2},\n"
            "  articleno = {23},\n"
            "  year      = {2024},\n"
            "  doi       = {10.1145/3648437}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-02",
        "kind": "reference",
        "short_cite": "UNMAS (2026): IMAS 07.11",
        "year": "2026",
        "category": "Mine action standards and reports",
        "title": "IMAS 07.11: Land release",
        "authors": "United Nations Mine Action Service",
        "venue": "International Mine Action Standards",
        "doi_or_url": "https://www.mineactionstandards.org/standards/07-11/",
        "license_or_edition": (
            "2nd ed. (16 January 2026, as the rules file gives it); the notebook and the rules file cite "
            "§§3.2–3.20 and 4.1–4.4"
        ),
        "key_findings": (
            "Land release is \"an evidence-based decision-making process\" (§4.1): land is cancelled by non-technical "
            "survey, reduced by technical survey, or cleared (§3.2, §3.3, §3.16). Areas with only indirect evidence "
            "are suspected hazardous areas, and areas with direct evidence should be confirmed hazardous areas; "
            "decisions rest on evidence, \"not the lack thereof\" (§4.2). Land on old reports may be cancelled on "
            "evidence of safe human activity over time (§4.3); indirect evidence supports a suspected hazardous area "
            "and needs technical survey before any decision to clear, and direct evidence supports a confirmed one "
            "and leads to technical survey or clearance (§4.4)."
        ),
        "codebase_grounding": (
            "Paraphrased in `rules/land_release_rules.md`, beside the project's own rules, which are marked as such. "
            "Enforced in code by the rule check (`twin/rules.py`, `rule_check`, R1–R6) and by the twin's own checks "
            "at sign-off (`twin/core.py`, which runs the rule check again and refuses a release it vetoes); the "
            "dossier reviewer agent reads the rules through its `lookup_rules` tool (`twin/agent.py`). Under the "
            "standard, people decide to release land: the twin changes land only through a decision a person "
            "approves (notebook sections 1.4 and 2.1)."
        ),
        "modules": "rules/land_release_rules.md, twin/rules.py, twin/core.py, twin/agent.py",
        "apa_citation": (
            "United Nations Mine Action Service. (2026). IMAS 07.11: Land release (2nd ed.). International Mine Action "
            "Standards. https://www.mineactionstandards.org/standards/07-11/"
        ),
        "bibtex": (
            "@misc{unmas2026imas0711,\n"
            "  author       = {{United Nations Mine Action Service}},\n"
            "  title        = {{IMAS} 07.11: Land release},\n"
            "  howpublished = {International Mine Action Standards},\n"
            "  note         = {2nd ed.},\n"
            "  year         = {2026},\n"
            "  url          = {https://www.mineactionstandards.org/standards/07-11/}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-03",
        "kind": "reference",
        "short_cite": "UNMAS (2019a): IMAS 08.10",
        "year": "2019a",
        "category": "Mine action standards and reports",
        "title": "IMAS 08.10: Non-technical survey",
        "authors": "United Nations Mine Action Service",
        "venue": "International Mine Action Standards",
        "doi_or_url": "https://www.mineactionstandards.org/standards/08-10/",
        "license_or_edition": "1st ed., Amendment 4; the rules file cites §§6.1, 6.2, 7, 8.2 and 8.3",
        "key_findings": (
            "Suggests seven broad classes of sources, from the most confidence to the least (§8.2): from class 1, "
            "direct physical evidence seen and recorded by the survey team, to class 7, people told about the hazard "
            "by others who cannot be identified as first-hand sources; it lists women, girls, boys and men of the "
            "community, victims and other witnesses among the first-hand sources (class 4). It gives examples of "
            "indirect and direct evidence (§6.1); cancellation through non-technical survey needs all reasonable "
            "effort (§6.2); desk studies use all relevant sources (§7); the use of land may be weighed when "
            "considering cancellation (§8.3)."
        ),
        "codebase_grounding": (
            "The reliability class (1 to 7) that the dossier reviewer records for each item (`prompts/system_prompt.md`, "
            "`rules/land_release_rules.md`, `twin/agent.py`) and that the rule check reads (`twin/rules.py`): R1 (no "
            "direct evidence, no indirect evidence of class 1 to 4, and no unexplained weaker indirect evidence before "
            "a cancellation), R2 (at least two different sources of class 1 to 4, each reporting at least five years "
            "of safe use), R5 (every item of the evidence log is in the register) and R6 (no cancellation where a past "
            "accident is recorded inside the area; the criteria need an intrusive check first). That evidence of "
            "absence from class 1 to 4 can explain weaker indirect evidence (class 5 to 7) is this project's rule, not "
            "IMAS (`rules/land_release_rules.md`); weaker evidence becomes a target for an intrusive check after 2 "
            "visits (`Criteria.follow_up_visits`, `twin/config.py`)."
        ),
        "modules": "rules/land_release_rules.md, twin/rules.py, twin/agent.py, prompts/system_prompt.md, twin/config.py",
        "apa_citation": (
            "United Nations Mine Action Service. (2019a). IMAS 08.10: Non-technical survey (1st ed., Amendment 4). "
            "International Mine Action Standards. https://www.mineactionstandards.org/standards/08-10/"
        ),
        "bibtex": (
            "@misc{unmas2019imas0810,\n"
            "  author       = {{United Nations Mine Action Service}},\n"
            "  title        = {{IMAS} 08.10: Non-technical survey},\n"
            "  howpublished = {International Mine Action Standards},\n"
            "  note         = {1st ed., Amendment 4},\n"
            "  year         = {2019},\n"
            "  url          = {https://www.mineactionstandards.org/standards/08-10/}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-04",
        "kind": "reference",
        "short_cite": "UNMAS (2019b): IMAS 08.20",
        "year": "2019b",
        "category": "Mine action standards and reports",
        "title": "IMAS 08.20: Technical survey",
        "authors": "United Nations Mine Action Service",
        "venue": "International Mine Action Standards",
        "doi_or_url": "https://www.mineactionstandards.org/standards/08-20/",
        "license_or_edition": "1st ed., Amendment 4; the rules file cites §§4.1, 4.2 and 5.3",
        "key_findings": (
            "A suspected hazardous area should normally go through non-technical survey before technical survey "
            "assets are committed, and targeted technical survey is preferred to systematic technical survey (§4.1, "
            "§4.2); land can be reduced by technical survey when the efforts applied could reasonably have found "
            "evidence of contamination, had it been present (§5.3)."
        ),
        "codebase_grounding": (
            "In the rule check (`twin/rules.py`): R4, a technical survey passes only where the national criteria "
            "provide it and there is a target; R3, where they provide it, a clearance needs direct evidence of class "
            "1 to 4 to define the hazard. The main policy follows Colombia, with no technical survey "
            "(`Criteria.technical_survey = False`, `twin/config.py`); the season replay compares the policy "
            "\"technical survey first\" (`twin/evaluate.py`)."
        ),
        "modules": "rules/land_release_rules.md, twin/rules.py, twin/config.py, twin/evaluate.py",
        "apa_citation": (
            "United Nations Mine Action Service. (2019b). IMAS 08.20: Technical survey (1st ed., Amendment 4). "
            "International Mine Action Standards. https://www.mineactionstandards.org/standards/08-20/"
        ),
        "bibtex": (
            "@misc{unmas2019imas0820,\n"
            "  author       = {{United Nations Mine Action Service}},\n"
            "  title        = {{IMAS} 08.20: Technical survey},\n"
            "  howpublished = {International Mine Action Standards},\n"
            "  note         = {1st ed., Amendment 4},\n"
            "  year         = {2019},\n"
            "  url          = {https://www.mineactionstandards.org/standards/08-20/}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-05",
        "kind": "reference",
        "short_cite": "Mine Action Review (2025)",
        "year": "2025",
        "category": "Mine action standards and reports",
        "title": "Clearing the mines 2025: Colombia",
        "authors": "Mine Action Review",
        "venue": "Norwegian People's Aid",
        "doi_or_url": "https://www.mineactionreview.org/assets/downloads/Colombia_Clearing_the_Mines_2025.pdf",
        "license_or_edition": "Report (2025)",
        "key_findings": (
            "Reports that Colombia's municipalities \"are subdivided into operational zones based on accessibility\", "
            "with \"the zone as the unit of measurement\", and that technical survey \"is not foreseen\" for the "
            "baseline areas, which go to clearance after non-technical survey (notebook section 1.1)."
        ),
        "codebase_grounding": (
            "The main policy's `Criteria.technical_survey = False` (`twin/config.py`), under which a hazardous area "
            "that non-technical survey defines goes to clearance (R3 in `twin/rules.py`; the rules file's "
            "`technical_survey` section). The operational zones (`twin/core.py`): each municipality's areas are "
            "grouped into zones of about 25 areas by position and access (K-means on the areas' position and "
            "log(1 + km to the nearest road), weighted by `zone_access_weight = 6.0`, the project's setting), 292 "
            "zones in all."
        ),
        "modules": "twin/config.py, twin/rules.py, twin/core.py, rules/land_release_rules.md",
        "apa_citation": (
            "Mine Action Review. (2025). Clearing the mines 2025: Colombia. Norwegian People's Aid. "
            "https://www.mineactionreview.org/assets/downloads/Colombia_Clearing_the_Mines_2025.pdf"
        ),
        "bibtex": (
            "@techreport{mineactionreview2025colombia,\n"
            "  author      = {{Mine Action Review}},\n"
            "  title       = {Clearing the mines 2025: {Colombia}},\n"
            "  institution = {Norwegian People's Aid},\n"
            "  year        = {2025},\n"
            "  url         = {https://www.mineactionreview.org/assets/downloads/Colombia_Clearing_the_Mines_2025.pdf}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-06",
        "kind": "reference",
        "short_cite": "Parasuraman & Manzey (2010)",
        "year": "2010",
        "category": "Peer-reviewed research",
        "title": "Complacency and bias in human use of automation: An attentional integration",
        "authors": "Parasuraman, R., & Manzey, D. H.",
        "venue": "Human Factors, 52(3), 381–410",
        "doi_or_url": "https://doi.org/10.1177/0018720810376055",
        "license_or_edition": "Journal article",
        "key_findings": (
            "People make errors of omission and of commission when a decision aid is imperfect, experts as well as "
            "novices, and neither training nor instructions prevents it (as the notebook's section 11.6 summarises it)."
        ),
        "codebase_grounding": (
            "Cited for risk Ma1, automation bias at sign-off, in `docs/risk_register.md`, and in "
            "`docs/model_cards/03_dossier_reviewer.md`. The twin's answers (notebook section 11.6): every "
            "recommendation comes with its evidence register and rule check; an override needs a written reason and "
            "is logged (`twin/core.py`); the overlays say \"insufficient evidence\" rather than guess. The rule check "
            "tests the register, not the world: a reading error that fits the outcome passes it (`twin/rules.py`)."
        ),
        "modules": "docs/risk_register.md, docs/model_cards/03_dossier_reviewer.md, twin/core.py, twin/rules.py",
        "apa_citation": (
            "Parasuraman, R., & Manzey, D. H. (2010). Complacency and bias in human use of automation: An attentional "
            "integration. Human Factors, 52(3), 381–410. https://doi.org/10.1177/0018720810376055"
        ),
        "bibtex": (
            "@article{parasuraman2010complacency,\n"
            "  author  = {Parasuraman, Raja and Manzey, Dietrich H.},\n"
            "  title   = {Complacency and Bias in Human Use of Automation: An Attentional Integration},\n"
            "  journal = {Human Factors},\n"
            "  volume  = {52},\n"
            "  number  = {3},\n"
            "  pages   = {381--410},\n"
            "  year    = {2010},\n"
            "  doi     = {10.1177/0018720810376055}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-07",
        "kind": "reference",
        "short_cite": "Gebru et al. (2021)",
        "year": "2021",
        "category": "AI documentation and governance",
        "title": "Datasheets for datasets",
        "authors": "Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K.",
        "venue": "Communications of the ACM, 64(12), 86–92",
        "doi_or_url": "https://doi.org/10.1145/3458723",
        "license_or_edition": "Journal article",
        "key_findings": (
            "Proposes that every dataset be documented with a datasheet: its motivation, composition, collection "
            "process, preprocessing, uses, distribution and maintenance."
        ),
        "codebase_grounding": (
            "The three datasheets in `docs/datasheets/` follow its sections, shortened: they document the victims "
            "table's labels that do not match their values (`estado`, `genero`, `tipoevento`) and its repeated rows "
            "that are real (P1), the demining table's paired rows, summed (P2), RELand's coordinates, set aside once "
            "the cells are placed on a grid, and each file's SHA-256 checksum, which `twin/data.py` checks before "
            "loading it."
        ),
        "modules": "docs/datasheets/01_victims.md, docs/datasheets/02_demining.md, docs/datasheets/03_reland.md, twin/data.py",
        "apa_citation": (
            "Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. "
            "(2021). Datasheets for datasets. Communications of the ACM, 64(12), 86–92. https://doi.org/10.1145/3458723"
        ),
        "bibtex": (
            "@article{gebru2021datasheets,\n"
            "  author  = {Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Daum{\\'e} III, Hal and Crawford, Kate},\n"
            "  title   = {Datasheets for Datasets},\n"
            "  journal = {Communications of the ACM},\n"
            "  volume  = {64},\n"
            "  number  = {12},\n"
            "  pages   = {86--92},\n"
            "  year    = {2021},\n"
            "  doi     = {10.1145/3458723}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-08",
        "kind": "reference",
        "short_cite": "Mitchell et al. (2019)",
        "year": "2019",
        "category": "AI documentation and governance",
        "title": "Model cards for model reporting",
        "authors": "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T.",
        "venue": (
            "Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 220–229), "
            "Association for Computing Machinery"
        ),
        "doi_or_url": "https://doi.org/10.1145/3287560.3287596",
        "license_or_edition": "Conference paper",
        "key_findings": (
            "Proposes model cards that report a model's details, intended use, factors, metrics, evaluation data, "
            "training data, quantitative analyses, ethical considerations, and caveats and recommendations."
        ),
        "codebase_grounding": (
            "The four model cards in `docs/model_cards/` follow its sections, shortened, with a section on replacing "
            "each model: `01_risk_model.md` (risk prioritisation), `02_report_checker.md` (the report checker), "
            "`03_dossier_reviewer.md` (the dossier reviewer agent) and `04_situation_report.md` (the weekly situation "
            "report). They include the full run's evaluation results, so this screen lists them without showing them."
        ),
        "modules": "docs/model_cards/01_risk_model.md, docs/model_cards/02_report_checker.md, docs/model_cards/03_dossier_reviewer.md, docs/model_cards/04_situation_report.md",
        "apa_citation": (
            "Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., "
            "& Gebru, T. (2019). Model cards for model reporting. In Proceedings of the Conference on Fairness, "
            "Accountability, and Transparency (pp. 220–229). Association for Computing Machinery. "
            "https://doi.org/10.1145/3287560.3287596"
        ),
        "bibtex": (
            "@inproceedings{mitchell2019modelcards,\n"
            "  author    = {Mitchell, Margaret and Wu, Simone and Zaldivar, Andrew and Barnes, Parker and Vasserman, Lucy and Hutchinson, Ben and Spitzer, Elena and Raji, Inioluwa Deborah and Gebru, Timnit},\n"
            "  title     = {Model Cards for Model Reporting},\n"
            "  booktitle = {Proceedings of the Conference on Fairness, Accountability, and Transparency},\n"
            "  publisher = {Association for Computing Machinery},\n"
            "  pages     = {220--229},\n"
            "  year      = {2019},\n"
            "  doi       = {10.1145/3287560.3287596}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-09",
        "kind": "reference",
        "short_cite": "NIST (2023): AI RMF 1.0",
        "year": "2023",
        "category": "AI documentation and governance",
        "title": "Artificial intelligence risk management framework (AI RMF 1.0)",
        "authors": "National Institute of Standards and Technology",
        "venue": "NIST AI 100-1",
        "doi_or_url": "https://doi.org/10.6028/NIST.AI.100-1",
        "license_or_edition": "NIST AI 100-1",
        "key_findings": (
            "Organises AI risk management in four functions: Govern, the roles, policies and accountability that run "
            "through the others; Map, the context in which risks arise; Measure, testing and tracking them; and "
            "Manage, acting on what is mapped and measured (as `docs/risk_register.md` puts them)."
        ),
        "codebase_grounding": (
            "Structures `docs/risk_register.md`: 19 risks under Govern (G1–G3), Map (M1–M5), Measure (Me1–Me6) and "
            "Manage (Ma1–Ma5), each with its control in the twin, what remains and the role that would own it (tab 2)."
        ),
        "modules": "docs/risk_register.md",
        "apa_citation": (
            "National Institute of Standards and Technology. (2023). Artificial intelligence risk management "
            "framework (AI RMF 1.0) (NIST AI 100-1). https://doi.org/10.6028/NIST.AI.100-1"
        ),
        "bibtex": (
            "@techreport{nist2023airmf,\n"
            "  author      = {{National Institute of Standards and Technology}},\n"
            "  title       = {Artificial intelligence risk management framework ({AI RMF} 1.0)},\n"
            "  institution = {National Institute of Standards and Technology},\n"
            "  number      = {NIST AI 100-1},\n"
            "  year        = {2023},\n"
            "  doi         = {10.6028/NIST.AI.100-1}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-10",
        "kind": "reference",
        "short_cite": "IASC (2023)",
        "year": "2023",
        "category": "AI documentation and governance",
        "title": "IASC operational guidance on data responsibility in humanitarian action",
        "authors": "Inter-Agency Standing Committee",
        "venue": "Inter-Agency Standing Committee",
        "doi_or_url": (
            "https://interagencystandingcommittee.org/system/files/2023-04/IASC%20Operational%20Guidance%20on%20Data"
            "%20Responsibility%20in%20Humanitarian%20Action%2C%202023.pdf"
        ),
        "license_or_edition": "2023",
        "key_findings": (
            "Operational guidance on data responsibility in humanitarian action: principles and actions, at "
            "system-wide, cluster and organisation level, for the safe, ethical and effective management of data, "
            "including the classification of data by sensitivity and protocols for sharing information."
        ),
        "codebase_grounding": (
            "Cited for risk G3 in `docs/risk_register.md` (data on victims or restricted layers reaching people who "
            "should not have them) and in the notebook's section 11.6. Each layer is registered with its sensitivity "
            "and handling (`twin/layers.py`: open, restricted or simulated); `twin/data.py` drops the victims table's "
            "site names and municipal-seat coordinates on loading; victims appear only as counts by municipality; the "
            "restricted layers (access and security, consultation, teams) stay inside the twin (data rule 5 in "
            "`app/README.md`); no map has axes, a scale or coordinates."
        ),
        "modules": "twin/layers.py, twin/data.py, docs/risk_register.md, app/README.md",
        "apa_citation": (
            "Inter-Agency Standing Committee. (2023). IASC operational guidance on data responsibility in humanitarian "
            "action. https://interagencystandingcommittee.org/system/files/2023-04/IASC%20Operational%20Guidance%20on"
            "%20Data%20Responsibility%20in%20Humanitarian%20Action%2C%202023.pdf"
        ),
        "bibtex": (
            "@techreport{iasc2023data,\n"
            "  author      = {{Inter-Agency Standing Committee}},\n"
            "  title       = {{IASC} operational guidance on data responsibility in humanitarian action},\n"
            "  institution = {Inter-Agency Standing Committee},\n"
            "  year        = {2023},\n"
            "  url         = {https://interagencystandingcommittee.org/system/files/2023-04/IASC%20Operational%20Guidance%20on%20Data%20Responsibility%20in%20Humanitarian%20Action%2C%202023.pdf}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-11",
        "kind": "dataset",
        "short_cite": "Presidencia de la República de Colombia (n.d.): victims (`yhxn-eqqw`)",
        "year": "n.d.",
        "category": "Colombian open data and IDEAM",
        "title": "Situación Víctimas Minas Antipersonal en Colombia (data set yhxn-eqqw, 1990–2023)",
        "authors": "Presidencia de la República de Colombia",
        "venue": "Datos Abiertos de Colombia (datos.gov.co), data set yhxn-eqqw",
        "doi_or_url": (
            "https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-V-ctimas-Minas-Antipersonal-en-"
            "Colombia/yhxn-eqqw"
        ),
        "license_or_edition": "CC BY-SA 4.0 · the portal's export of 18 October 2026 · SHA-256: 6ac13df53d00...",
        "key_findings": (
            "12,390 rows × 18 columns, one row per victim, 1990 to 2023. P1 found that two column labels contradict "
            "their values (`estado` holds sex and `genero` killed or injured; `tipoevento` holds the place type), and "
            "3,858 rows (31%) that exactly repeat another row; since several victims of one accident in the same "
            "municipality and month look identical, the repeats are kept as separate victims "
            "(`docs/datasheets/01_victims.md`)."
        ),
        "codebase_grounding": (
            "Loaded by `twin/data.py` (`load_victims`), which drops the site names and the municipal-seat coordinates, "
            "and counted by municipality and month (`victims_by_month`) into the twin's victims layer (open, counts "
            "only; `twin/layers.py`), shown beside the season plan as the impact by municipality and not weighted in "
            "the plan's objective."
        ),
        "modules": "twin/data.py, twin/layers.py, docs/datasheets/01_victims.md",
        "apa_citation": (
            "Presidencia de la República de Colombia. (n.d.). Situación víctimas minas antipersonal en Colombia [Data "
            "set yhxn-eqqw]. Datos Abiertos de Colombia. Retrieved October 18, 2026, from "
            "https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-V-ctimas-Minas-Antipersonal-en-"
            "Colombia/yhxn-eqqw"
        ),
        "bibtex": (
            "@misc{colombia_victims_yhxneqqw,\n"
            "  author       = {{Presidencia de la Rep{\\'u}blica de Colombia}},\n"
            "  title        = {Situaci{\\'o}n V{\\'i}ctimas Minas Antipersonal en Colombia (data set yhxn-eqqw)},\n"
            "  howpublished = {Datos Abiertos de Colombia (datos.gov.co)},\n"
            "  note         = {CC BY-SA 4.0; the portal's export of 18 October 2026; SHA-256: 6ac13df53d0024bc9d96f8b48d7622934b5bd1a334d40cf4155bf4cea804a84d},\n"
            "  url          = {https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-V-ctimas-Minas-Antipersonal-en-Colombia/yhxn-eqqw}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-12",
        "kind": "dataset",
        "short_cite": "DAPRE (2024): demining (`jbqm-s5tn`)",
        "year": "2024",
        "category": "Colombian open data and IDEAM",
        "title": "Situación de Desminado Humanitario en Colombia (data set jbqm-s5tn, 2004–2023)",
        "authors": "Departamento Administrativo de la Presidencia de la República (DAPRE)",
        "venue": "Datos Abiertos de Colombia (datos.gov.co), data set jbqm-s5tn",
        "doi_or_url": (
            "https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-de-Desminado-Humanitario-en-"
            "Colombia/jbqm-s5tn"
        ),
        "license_or_edition": (
            "CC BY-SA 4.0 · rows last updated by the publisher on 19 January 2024 · SHA-256: eda568a71903..."
        ),
        "key_findings": (
            "5,721 rows × 18 columns, 2004 to 2023. Each row reports results or hazardous areas for one municipality, "
            "month, demining type and operator: land cleared (m²); items destroyed, by the register's categories "
            "(MAP, MUSE, AEI) and in total; the numbers of non-technical surveys, technical surveys and clearance "
            "operations; hazardous areas recorded and confirmed. P2 found that 1,183 combinations of municipality, "
            "month, demining type and operator appear twice, each pair splitting results from hazardous areas, so "
            "the rows are summed, not de-duplicated (`docs/datasheets/02_demining.md`)."
        ),
        "codebase_grounding": (
            "Loaded by `twin/data.py` (`load_demining`), which sums the rows by municipality and month, into the "
            "twin's demining layer (open; `twin/layers.py`): a check of the simulated clearance capacity against the "
            "pilot's past clearance output by year (notebook section 4.3)."
        ),
        "modules": "twin/data.py, twin/layers.py, docs/datasheets/02_demining.md",
        "apa_citation": (
            "Departamento Administrativo de la Presidencia de la República. (2024). Situación de desminado humanitario "
            "en Colombia [Data set jbqm-s5tn]. Datos Abiertos de Colombia. Retrieved October 18, 2026, from "
            "https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-de-Desminado-Humanitario-en-"
            "Colombia/jbqm-s5tn"
        ),
        "bibtex": (
            "@misc{colombia_demining_jbqms5tn,\n"
            "  author       = {{Departamento Administrativo de la Presidencia de la Rep{\\'u}blica (DAPRE)}},\n"
            "  title        = {Situaci{\\'o}n de Desminado Humanitario en Colombia (data set jbqm-s5tn)},\n"
            "  howpublished = {Datos Abiertos de Colombia (datos.gov.co)},\n"
            "  year         = {2024},\n"
            "  note         = {CC BY-SA 4.0; rows last updated by the publisher on 19 January 2024; SHA-256: eda568a71903f05d7468c582b3795dbd5a537a71c5413f84f72f0acbcffb6617},\n"
            "  url          = {https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-de-Desminado-Humanitario-en-Colombia/jbqm-s5tn}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-13",
        "kind": "reference",
        "short_cite": "IDEAM (2025)",
        "year": "2025",
        "category": "Colombian open data and IDEAM",
        "title": "Boletín agroclimático comunitario (No. 24)",
        "authors": "Instituto de Hidrología, Meteorología y Estudios Ambientales",
        "venue": "Instituto de Hidrología, Meteorología y Estudios Ambientales (IDEAM)",
        "doi_or_url": "https://ideam.gov.co/file-download/download/public/10929",
        "license_or_edition": "No. 24",
        "key_findings": (
            "Cited for the two rainy seasons of the Andean region, March to April and October to November (notebook "
            "section 5.2; `twin/layers.py`)."
        ),
        "codebase_grounding": (
            "The rainfall layer (simulated) and `Season.rainy_months = (3, 4, 10, 11)` in `twin/config.py`. How many "
            "working days heavy rain or high wind takes, about half in those months (`rainy_day_loss = 0.5`) and one "
            "in ten in the others (`other_day_loss = 0.1`), is the project's setting, not IDEAM's. The rates set the "
            "calendar's working days (`twin/core.py`), the planner's (`twin/planner.py`) and the forecaster's "
            "(`twin/forecast.py`), the days the simulated field loses (`twin/field.py`), and the report checker's "
            "expected loss and its rainy-month query (`twin/checker.py`)."
        ),
        "modules": "twin/config.py, twin/layers.py, twin/core.py, twin/planner.py, twin/forecast.py, twin/field.py, twin/checker.py",
        "apa_citation": (
            "Instituto de Hidrología, Meteorología y Estudios Ambientales. (2025). Boletín agroclimático comunitario "
            "(No. 24). https://ideam.gov.co/file-download/download/public/10929"
        ),
        "bibtex": (
            "@techreport{ideam2025boletin24,\n"
            "  author      = {{Instituto de Hidrolog{\\'i}a, Meteorolog{\\'i}a y Estudios Ambientales}},\n"
            "  title       = {Bolet{\\'i}n agroclim{\\'a}tico comunitario},\n"
            "  number      = {24},\n"
            "  institution = {IDEAM},\n"
            "  year        = {2025},\n"
            "  url         = {https://ideam.gov.co/file-download/download/public/10929}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-14",
        "kind": "reference",
        "short_cite": "Kritzinger et al. (2018)",
        "year": "2018",
        "category": "Peer-reviewed research",
        "title": "Digital twin in manufacturing: A categorical literature review and classification",
        "authors": "Kritzinger, W., Karner, M., Traar, G., Henjes, J., & Sihn, W.",
        "venue": "IFAC-PapersOnLine, 51(11), 1016–1022",
        "doi_or_url": "https://doi.org/10.1016/j.ifacol.2018.08.474",
        "license_or_edition": "Journal article",
        "key_findings": (
            "Distinguishes a digital model (no automatic data flow), a digital shadow (automatic flow from the "
            "physical object to the model) and a digital twin (automatic flow both ways) (as the notebook's section "
            "2.1 summarises it)."
        ),
        "codebase_grounding": (
            "The system is by design a digital shadow with approved write-back: reports flow into the twin as they "
            "arrive, but nothing flows back to the field, and no land changes status, without a person's decision "
            "(notebook sections 2.1 and 2.5). The twin's one log records every report and every decision "
            "(`twin/core.py`)."
        ),
        "modules": "integrated_system.ipynb (sections 2.1, 2.5), twin/core.py",
        "apa_citation": (
            "Kritzinger, W., Karner, M., Traar, G., Henjes, J., & Sihn, W. (2018). Digital twin in manufacturing: A "
            "categorical literature review and classification. IFAC-PapersOnLine, 51(11), 1016–1022. "
            "https://doi.org/10.1016/j.ifacol.2018.08.474"
        ),
        "bibtex": (
            "@article{kritzinger2018digitaltwin,\n"
            "  author  = {Kritzinger, W. and Karner, M. and Traar, G. and Henjes, J. and Sihn, W.},\n"
            "  title   = {Digital twin in manufacturing: A categorical literature review and classification},\n"
            "  journal = {IFAC-PapersOnLine},\n"
            "  volume  = {51},\n"
            "  number  = {11},\n"
            "  pages   = {1016--1022},\n"
            "  year    = {2018},\n"
            "  doi     = {10.1016/j.ifacol.2018.08.474}\n"
            "}"
        ),
    },
    {
        "ref_id": "LIT-15",
        "kind": "reference",
        "short_cite": "Smith & Winkler (2006)",
        "year": "2006",
        "category": "Peer-reviewed research",
        "title": "The optimizer's curse: Skepticism and postdecision surprise in decision analysis",
        "authors": "Smith, J. E., & Winkler, R. L.",
        "venue": "Management Science, 52(3), 311–322",
        "doi_or_url": "https://doi.org/10.1287/mnsc.1050.0451",
        "license_or_edition": "Journal article",
        "key_findings": (
            "A planner that picks the options that look best picks, on average, those whose value is most "
            "overestimated (as the notebook's section 6.1 summarises it)."
        ),
        "codebase_grounding": (
            "Why the risk overlay's baseline-first gate tests calibration as well as ranking: the calibration test was "
            "added after the first season replays showed the planner choosing the zones where a model's chances were "
            "furthest too high (`twin/prioritise.py`; notebook section 6.1; tab 4)."
        ),
        "modules": "twin/prioritise.py, models/gate.csv",
        "apa_citation": (
            "Smith, J. E., & Winkler, R. L. (2006). The optimizer's curse: Skepticism and postdecision surprise in "
            "decision analysis. Management Science, 52(3), 311–322. https://doi.org/10.1287/mnsc.1050.0451"
        ),
        "bibtex": (
            "@article{smith2006optimizerscurse,\n"
            "  author  = {Smith, J. E. and Winkler, R. L.},\n"
            "  title   = {The optimizer's curse: Skepticism and postdecision surprise in decision analysis},\n"
            "  journal = {Management Science},\n"
            "  volume  = {52},\n"
            "  number  = {3},\n"
            "  pages   = {311--322},\n"
            "  year    = {2006},\n"
            "  doi     = {10.1287/mnsc.1050.0451}\n"
            "}"
        ),
    },
]


# =====================================================================================================================
# 2. VISUAL LITERATURE-TO-ARCHITECTURE TRACEABILITY DIAGRAM
# =====================================================================================================================
def plot_literature_traceability_diagram() -> plt.Figure:
    """Render a 3-column diagram: the sources (left), the insights of the earlier projects (notebook section 2.4) and
    the NIST AI RMF (centre), and where they show in the twin's code, models and documents (right). The dashed arrows
    mark the two cross-links the code declares: the planner carries P3's insight, and the risk overlay's monthly
    learning P2's."""
    fig, ax = plt.subplots(figsize=(15.8, 6.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Column Headers
    headers = [
        (16.5, 95.5, "1. Sources: research, standards, data", "#1e3a8a"),
        (50.0, 95.5, "2. Insights (P1–P6) and the NIST AI RMF", "#0d6e52"),
        (83.5, 95.5, "3. Where they show in the twin", "#7c2d12"),
    ]
    for cx, cy, title, col in headers:
        ax.text(
            cx,
            cy,
            title,
            ha="center",
            va="center",
            fontsize=9.6,
            fontweight="bold",
            color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.42", facecolor=col, edgecolor=col, lw=1.0),
        )

    left_boxes = [
        (16.5, 81, "Dulce Rubio et al. (2024): RELand\n26,761 cells · 63 features\nSmith & Winkler (2006): the optimiser's curse", "#eff6ff", "#2563eb"),
        (16.5, 65, "datos.gov.co: victims (`yhxn-eqqw`, Presidencia)\nand demining (`jbqm-s5tn`, DAPRE)", "#f0fdf4", "#16a34a"),
        (16.5, 49, "IDEAM (2025): two rainy seasons\nMine Action Review (2025): Colombia", "#fefce8", "#ca8a04"),
        (16.5, 33, "UNMAS: IMAS 07.11 (2026), 08.10 and 08.20 (2019)\nLand release; source classes 1 to 7", "#fef2f2", "#dc2626"),
        (16.5, 17, "NIST (2023) · IASC (2023) · Parasuraman & Manzey (2010)\nGebru et al. (2021) · Mitchell et al. (2019)\nKritzinger et al. (2018)", "#f5f3ff", "#7c3aed"),
    ]

    mid_boxes = [
        (50.0, 81, "P3: a simple baseline first,\ntested on held-out municipalities\n(the gate: ranking and calibration)", "#eff6ff", "#2563eb"),
        (50.0, 65, "P1: check data as it arrives\n(labels, repeated rows, team reports)", "#f0fdf4", "#16a34a"),
        (50.0, 49, "P2: show uncertainty as intervals,\nand watch change over time", "#fefce8", "#ca8a04"),
        (50.0, 33, "P6: an evidence register, a coded veto\nand a person's sign-off", "#fef2f2", "#dc2626"),
        (50.0, 17, "P4: answer \"unknown\" when unsure\nP5: check generated output before use\nNIST AI RMF: Govern, Map, Measure, Manage", "#f5f3ff", "#7c3aed"),
    ]

    right_boxes = [
        (83.5, 81, "`twin/prioritise.py` & `models/gate.csv`\nRisk overlay; each month, a Bayesian update\nof one odds multiplier per municipality", "#eff6ff", "#1e40af"),
        (83.5, 65, "`twin/data.py` & `twin/checker.py`\nSHA-256 checks; coordinates set aside;\ncoded checks + 12-feature isolation forest", "#f0fdf4", "#15803d"),
        (83.5, 49, "`twin/forecast.py` & `twin/planner.py`\n200-draw completion forecaster (P10/P50/P90);\nmonthly integer programme vs rule of thumb", "#fefce8", "#b45309"),
        (83.5, 33, "`twin/agent.py`, `twin/review.py`, `twin/rules.py`\nDossier reviewer agent (5 tools);\nrule check R1–R6; QC reviewer's sign-off", "#fef2f2", "#b91c1c"),
        (83.5, 17, "`twin/overlays.py`, `twin/sitrep.py`, `docs/`\n\"insufficient evidence\"; the figure check;\n4 model cards, 3 datasheets, risk register", "#f5f3ff", "#6d28d9"),
    ]

    def _draw_column(boxes: list[tuple], w: float = 29.0, h: float = 11.5) -> None:
        for cx, cy, text, fc, ec in boxes:
            rect = mpatches.FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0.35,rounding_size=1.2",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.35,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                cx,
                cy,
                text,
                ha="center",
                va="center",
                fontsize=8.1,
                color=INK,
                fontweight="normal",
                zorder=4,
            )

    _draw_column(left_boxes)
    _draw_column(mid_boxes)
    _draw_column(right_boxes)

    # Connect arrows across columns
    for idx in range(5):
        y_val = left_boxes[idx][1]
        col = left_boxes[idx][4]
        ax.annotate(
            "",
            xy=(35.2, y_val),
            xytext=(31.2, y_val),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color=col, lw=1.8),
            zorder=5,
        )
        ax.annotate(
            "",
            xy=(68.7, y_val),
            xytext=(64.7, y_val),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color=col, lw=1.8),
            zorder=5,
        )

    # Cross-links the code declares: the planner carries P3's insight (`SeasonPlanner.insight`), and the risk overlay's
    # monthly learning P2's (`twin/prioritise.py`)
    cross_links = [
        (64.7, 81, 68.7, 49, "#64748b"),
        (64.7, 49, 68.7, 81, "#64748b"),
    ]
    for x1, y1, x2, y2, col in cross_links:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.35", color=col, lw=1.0, ls="--", alpha=0.65),
            zorder=2,
        )

    ax.text(
        50.0,
        3.5,
        "The four models (risk prioritisation, the report checker, the dossier reviewer agent, the situation report) have "
        "model cards, and the three datasets datasheets;\nevery review's recommendation is tested by the coded rule "
        "check (R1–R6) before a person signs it off.",
        ha="center",
        va="center",
        fontsize=8.4,
        style="italic",
        color=INK_2,
    )
    fig.tight_layout()
    return fig


# =====================================================================================================================
# 3. THE RISK REGISTER (`docs/risk_register.md`), AS WRITTEN
# =====================================================================================================================
RISK_REGISTER_FILE = "docs/risk_register.md"
REGISTER_EVIDENCE_COLUMN = "Evidence from the full run"
REGISTER_SHOWN_COLUMNS = ["ID", "Risk", "Control in the twin", "What remains", "Owner"]
# Passages of the register's other columns that report results of the season replay, the injected scenarios or the
# expert review. This screen is not the Evaluation screen, so they are not shown here: (risk ID, column) -> (the
# passage as the register writes it, what it reports).
REGISTER_RESULT_PASSAGES = {
    ("Me2", "Control in the twin"): (" (noticed the same week and cleared in 20 of 20 seasons)",
                                     "a result of the injected scenarios"),
    ("Me3", "What remains"): ("From 14 judged reviews, the error rate is not known", "from the expert review"),
    ("Me5", "What remains"): ("A P90 week reads as safer than it is", "a result of the season replay"),
    ("Ma2", "What remains"): ("A two-week delay in 4 of 20 seasons", "a result of the injected scenarios"),
}


def read_risk_register(root: Path) -> tuple[str, pd.DataFrame]:
    """The risk register as `docs/risk_register.md` writes it: its opening paragraph, and one row per risk with every
    column of its table and the NIST AI RMF function it is listed under."""
    path = root / RISK_REGISTER_FILE
    if not path.exists():
        return "", pd.DataFrame()
    intro, rows, header, function = [], [], None, ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            function, header = line[3:].strip(), None
            continue
        if not function:
            if line.strip() and not line.startswith("# "):
                intro.append(line.strip())
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if not "".join(cells).strip("-: "):
            continue   # the table's separator row
        rows.append({"NIST AI RMF function": function, **dict(zip(header, cells))})
    return " ".join(intro), pd.DataFrame(rows)


def build_risk_register_table(root: Path) -> pd.DataFrame:
    """The register's table as written, without its evidence column and without the passages of
    `REGISTER_RESULT_PASSAGES`, each marked where it is left out."""
    _, register = read_risk_register(root)
    if register.empty:
        return register
    table = register[["NIST AI RMF function", *REGISTER_SHOWN_COLUMNS]].copy()
    for (risk_id, column), (passage, what) in REGISTER_RESULT_PASSAGES.items():
        here = table["ID"] == risk_id
        table.loc[here, column] = table.loc[here, column].map(
            lambda text: text.replace(passage, f" [{what}: not shown on this screen]").strip()
        )
    return table


# =====================================================================================================================
# 4. THE REPORT CHECKER (`twin/checker.py`): ITS CHECKS, ITS FEATURES, AND SOME CHECKS RE-RUN ON SAMPLE REPORTS
# =====================================================================================================================
def render_report_checker_inspector(root: Path) -> None:
    """The report checker of `twin/checker.py` (Model Card 02): its coded checks as the notebook's section 7.2 lists
    them, the isolation forest's 12 features, and a re-run here of some of the coded checks on sample reports (not
    `ReportChecker.check`; the isolation forest is not run here).
    """
    season_cfg = config.Season()
    cal = core.Calendar(season_cfg)
    week_days = season_cfg.days_per_week
    clearance_days = season_cfg.task_days("CLEARANCE")
    ts_days = season_cfg.task_days("TS")
    rainy_loss = season_cfg.rainy_day_loss * week_days
    other_loss = season_cfg.other_day_loss * week_days

    st.markdown("#### 🔍 The report checker (`twin/checker.py`, Model Card 02)")
    st.caption(
        "P1's insight, *check data as it arrives* (`ReportChecker.insight`). Every report from the field's "
        f"{season_cfg.survey_teams} survey and {season_cfg.clearance_teams} clearance teams passes the report checker "
        "before it changes the twin: coded checks **hold** a report that is impossible or inconsistent, and the team "
        "is asked to send it again; a visit's report whose evidence items did not all arrive is accepted with a flag, "
        "and the twin holds the dossier; and an **isolation forest** on 12 features (200 trees, contamination 0.01), "
        "fitted on a season of reports with no fault injected, flags unusual reports for the operations officer to "
        "**query** (notebook section 7.2)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coded checks", "Hold", "impossible or inconsistent reports; the team resends")
    c2.metric("Isolation forest", f"{len(checker.FEATURES)} features", "200 trees, contamination 0.01; queries")
    c3.metric("Rainy-month run", "3 weeks in a row", "work and no day lost to rain or wind: queried (≈ 1 in 30,000)")
    c4.metric(
        "Shortest finish allowed",
        "A quarter of the planned days",
        f"clearance {0.25 * clearance_days:g} of {clearance_days:g} days; technical survey {0.25 * ts_days:g} of {ts_days:g}",
    )

    col_rules, col_feats = st.columns([1.25, 1.0])
    with col_rules:
        st.markdown("**The checks on every team report (notebook section 7.2; `twin/checker.py`)**")
        coded_rows = [
            {
                "Check": f"The fields are present, and the days are whole numbers from 0 to {week_days} that fit in a week of {week_days}",
                "Action": "🛑 Hold; the team is asked to send it again",
            },
            {
                "Check": "The task was ordered to that team, is still open, and is in that zone and area, in a week since the order",
                "Action": "🛑 Hold; the team is asked to send it again",
            },
            {
                "Check": "It is not a second copy of a report already received",
                "Action": "🛑 Hold; the team is asked to send it again",
            },
            {
                "Check": "No work is reported in a zone the access feed showed closed in the week the report covers",
                "Action": "🛑 Hold; the team is asked to send it again",
            },
            {
                "Check": (
                    "A finished task carries the result its kind needs: the zone survey's flagged areas, a visit's "
                    "evidence items, a technical survey's finding, a clearance's count of mine cells"
                ),
                "Action": "🛑 Hold; the team is asked to send it again",
            },
            {
                "Check": (
                    "A technical survey or clearance is not reported done in under a quarter of its planned days "
                    f"(technical survey {0.25 * ts_days:g} of {ts_days:g} days; clearance {0.25 * clearance_days:g} of "
                    f"{clearance_days:g})"
                ),
                "Action": "🛑 Hold; the team is asked to send it again",
            },
            {
                "Check": "A visit's report declares more evidence items than arrived",
                "Action": "⚠️ Accept with a flag; the twin holds the dossier until the rest arrive",
            },
            {
                "Check": (
                    "Three weeks in a row with work and no day lost to rain or wind, in a month when about half the "
                    "days are lost"
                ),
                "Action": "❓ Query (an honest team reports that about once in 30,000 such runs)",
            },
            {
                "Check": "The isolation forest finds the report unusual against the clean season's reports",
                "Action": "❓ Query: the operations officer queries the team",
            },
        ]
        st.dataframe(pd.DataFrame(coded_rows), width="stretch", hide_index=True)

    with col_feats:
        st.markdown("**The isolation forest's 12 features (`twin/checker.py` `FEATURES`)**")
        feat_meta = {
            "days_worked": f"Working days reported in the week (0 to {week_days})",
            "days_lost_weather": f"Days lost to heavy rain or high wind (0 to {week_days})",
            "days_lost_access": f"Days lost because the zone was closed (0 to {week_days})",
            "expected_loss": (
                f"Expected days lost to rain or wind in the report's month: {rainy_loss:g} in IDEAM's rainy months, "
                f"{other_loss:g} in the others (rates set in `twin/config.py`)"
            ),
            "complete": "1 if the task was finished this week, else 0",
            "progress": "(days done before + days worked) / planned days",
            "items": "Evidence items in a visit's report (0 for the other kinds)",
            "mine_cells": "Mine cells found and cleared, in a clearance report",
        }
        for kind in checker.KINDS:
            feat_meta[f"kind_{kind}"] = (
                f"1 for a {loaders.HUMAN_TASK_KIND.get(kind, kind).lower()} task "
                f"({season_cfg.task_days(kind):g} planned days), else 0"
            )
        feat_rows = [
            {"Feature #": idx + 1, "Feature Name (`checker.FEATURES`)": f, "Operational Meaning": feat_meta.get(f, "—")}
            for idx, f in enumerate(checker.FEATURES)
        ]
        st.dataframe(pd.DataFrame(feat_rows), width="stretch", hide_index=True)

    st.markdown("##### 🧪 Sample reports: their 12 features, and some coded checks re-run here")
    st.caption(
        "Pick a sample report. The table shows its 12 features as `twin.checker.features` computes them. The verdict "
        "comes from a re-run here of some of the coded checks (the days in a week, a closed zone, a quarter of the "
        "planned days, a visit's missing items, the rainy-month run): it does not call `ReportChecker.check`, and the "
        "isolation forest is not run here. The month of each week comes from the season's calendar."
    )
    presets = {
        "1. A clearance finished in its planned days (accepted)": {
            "week": 6,
            "kind": "CLEARANCE",
            "days_done_before": int(clearance_days) - 5,
            "days_worked": 5,
            "days_lost_weather": 0,
            "days_lost_access": 0,
            "zone_closed": False,
            "complete": True,
            "declared_items": 0,
            "arrived_items": 0,
            "mine_cells": 2,
            "rainy_streak_weeks": 0,
        },
        "2. Seven days reported in a five-day week: 4 worked and 3 lost to rain (held)": {
            "week": 10,
            "kind": "VISIT",
            "days_done_before": 0,
            "days_worked": 4,
            "days_lost_weather": 3,
            "days_lost_access": 0,
            "zone_closed": False,
            "complete": True,
            "declared_items": 4,
            "arrived_items": 4,
            "mine_cells": 0,
            "rainy_streak_weeks": 0,
        },
        "3. Work reported in a zone the access feed showed closed (held)": {
            "week": 14,
            "kind": "ZONE_SURVEY",
            "days_done_before": 5,
            "days_worked": 3,
            "days_lost_weather": 2,
            "days_lost_access": 0,
            "zone_closed": True,
            "complete": False,
            "declared_items": 0,
            "arrived_items": 0,
            "mine_cells": 0,
            "rainy_streak_weeks": 0,
        },
        "4. A clearance reported done after 3 working days, under a quarter of its planned days (held)": {
            "week": 18,
            "kind": "CLEARANCE",
            "days_done_before": 1,
            "days_worked": 2,
            "days_lost_weather": 0,
            "days_lost_access": 0,
            "zone_closed": False,
            "complete": True,
            "declared_items": 0,
            "arrived_items": 0,
            "mine_cells": 0,
            "rainy_streak_weeks": 0,
        },
        "5. A visit's report declares 5 evidence items; 3 arrived (accepted with a flag)": {
            "week": 22,
            "kind": "VISIT",
            "days_done_before": 0,
            "days_worked": 5,
            "days_lost_weather": 0,
            "days_lost_access": 0,
            "zone_closed": False,
            "complete": True,
            "declared_items": 5,
            "arrived_items": 3,
            "mine_cells": 0,
            "rainy_streak_weeks": 0,
        },
        "6. The third rainy week in a row with work and no day lost to rain or wind (queried)": {
            "week": 42,
            "kind": "CLEARANCE",
            "days_done_before": 10,
            "days_worked": 5,
            "days_lost_weather": 0,
            "days_lost_access": 0,
            "zone_closed": False,
            "complete": False,
            "declared_items": 0,
            "arrived_items": 0,
            "mine_cells": 0,
            "rainy_streak_weeks": 3,
        },
    }

    sel_preset = st.selectbox(
        "Select a sample report:",
        options=list(presets.keys()),
        index=0,
        key="lit_lib_checker_preset",
    )
    p = presets[sel_preset]
    month = cal.month(p["week"])
    rainy = season_cfg.day_loss(month) >= 0.5
    st.markdown(
        f"**Week {p['week']}:** {config.MONTHS[month - 1]}, "
        + ("one of IDEAM's rainy months" if rainy else "not a rainy month")
        + f" (expected loss {season_cfg.day_loss(month) * week_days:g} days a week) · "
        f"**Task:** {loaders.HUMAN_TASK_KIND.get(p['kind'], p['kind'])}, {season_cfg.task_days(p['kind']):g} planned days"
    )

    # A sample report and task, in the form `twin.checker.features` reads
    fake_result = {}
    if p["kind"] == "VISIT":
        fake_result["items"] = [f"E{i+1}" for i in range(p["arrived_items"])]
        fake_result["declared_items"] = p["declared_items"]
    elif p["kind"] == "CLEARANCE":
        fake_result["mine_cells"] = p["mine_cells"]

    fake_report = {
        "week": p["week"],
        "kind": "team report",
        "days_worked": p["days_worked"],
        "days_lost_weather": p["days_lost_weather"],
        "days_lost_access": p["days_lost_access"],
        "complete": p["complete"],
        "result": fake_result,
    }
    fake_task = {
        "kind": p["kind"],
        "days_done": p["days_done_before"],
    }
    feat_vec = checker.features(fake_report, fake_task, season_cfg, cal)

    # Some of the coded checks, re-run here as `twin/checker.py` words them (not `ReportChecker.check`)
    problems = []
    flags = []
    days_list = [p["days_worked"], p["days_lost_weather"], p["days_lost_access"]]
    if sum(days_list) > week_days:
        problems.append(f"{sum(days_list)} days reported in a week of {week_days}")
    if p["zone_closed"] and p["days_worked"] > 0:
        problems.append("work reported in a zone the access feed showed closed that week")
    if p["complete"] and p["kind"] in ("TS", "CLEARANCE"):
        done_total = p["days_done_before"] + p["days_worked"]
        planned_total = season_cfg.task_days(p["kind"])
        if done_total < 0.25 * planned_total:
            problems.append(f"done in {done_total} working days, under a quarter of the {planned_total:.0f} planned")
    if p["complete"] and p["kind"] == "VISIT" and p["declared_items"] != p["arrived_items"]:
        flags.append(f"the report declares {p['declared_items']} item(s); {p['arrived_items']} arrived")
    if rainy and p["days_worked"] and not p["days_lost_weather"] and p["rainy_streak_weeks"] >= 3:
        flags.append(f"{p['rainy_streak_weeks']} weeks in a row with no day lost to rain or wind in a rainy month")

    if problems:
        verdict = "🛑 HOLD: the report does not change the twin, and the team is asked to send it again"
        st.error(f"**{verdict}**\n\n- " + "\n- ".join(problems))
    elif any("weeks in a row" in f for f in flags):
        verdict = "❓ QUERY: the operations officer queries the team"
        st.warning(f"**{verdict}**\n\n- " + "\n- ".join(flags))
    elif flags:
        verdict = "⚠️ ACCEPT WITH A FLAG: what arrived is recorded; the twin holds the dossier until the rest arrive"
        st.info(f"**{verdict}**\n\n- " + "\n- ".join(flags))
    else:
        verdict = "✅ ACCEPT: the checks re-run here find no problem (the isolation forest is not run here)"
        st.success(f"**{verdict}**")

    feat_disp = pd.DataFrame([feat_vec]).T.reset_index()
    feat_disp.columns = ["Feature Name (`twin.checker.FEATURES`)", "Computed Value"]
    st.dataframe(feat_disp, width="stretch", hide_index=True)


# =====================================================================================================================
# 5. DOCUMENTS READER (`docs/datasheets/`, `rules/`, `prompts/`, `app/README.md`)
# =====================================================================================================================
DOC_CATALOGUE = [
    (
        "Datasheet 01 — Victims of anti-personnel mines in Colombia (`docs/datasheets/01_victims.md`)",
        "Datasheet (Gebru et al., 2021)",
        "docs/datasheets/01_victims.md",
    ),
    (
        "Datasheet 02 — Humanitarian demining results in Colombia (`docs/datasheets/02_demining.md`)",
        "Datasheet (Gebru et al., 2021)",
        "docs/datasheets/02_demining.md",
    ),
    (
        "Datasheet 03 — RELand's landmine-risk grid cells, Antioquia (`docs/datasheets/03_reland.md`)",
        "Datasheet (Gebru et al., 2021)",
        "docs/datasheets/03_reland.md",
    ),
    (
        "Land release rules for the area reviewer (`rules/land_release_rules.md`)",
        "IMAS paraphrase and project rules",
        "rules/land_release_rules.md",
    ),
    (
        "The dossier reviewer's prompt (`prompts/system_prompt.md`)",
        "Agent system prompt",
        "prompts/system_prompt.md",
    ),
    (
        "The weekly situation report's prompt (`prompts/sitrep_prompt.md`)",
        "Situation report prompt",
        "prompts/sitrep_prompt.md",
    ),
    (
        "The application's README (`app/README.md`)",
        "README",
        "app/README.md",
    ),
]
# Listed by path only: they include results of the full run's evaluation (the season replay against the hidden truth,
# the injected scenarios or the expert review), which stay on the Evaluation screen.
DOCS_NOT_SHOWN = [
    "docs/model_cards/01_risk_model.md",
    "docs/model_cards/02_report_checker.md",
    "docs/model_cards/03_dossier_reviewer.md",
    "docs/model_cards/04_situation_report.md",
    "docs/risk_register.md",
    "README.md",
]


def render_verbatim_docs_reader(root: Path) -> None:
    """Reader and full-text search for the datasheets, the rules file, the prompts and the application's README, as
    stored on disk; the model cards, the risk register and the repository's README are listed by path only.
    """
    st.markdown("#### 📄 Datasheets, rules, prompts and README (`docs/datasheets/`, `rules/`, `prompts/`, `app/`)")
    st.caption(
        "Read and search the datasheets, the rules file, the two prompts and the application's README as they are "
        "stored in the repository."
    )
    st.markdown(
        "**Listed, not shown on this screen:** "
        + ", ".join(f"`{p}`" for p in DOCS_NOT_SHOWN)
        + ". They include results of the full run's evaluation (the season replay, the injected scenarios or the "
        "expert review), so they are not shown on this screen; tab 2 shows the risk register's table without its "
        "evidence column."
    )

    col_sel, col_search = st.columns([1.6, 1.0])
    with col_sel:
        selected_label = st.selectbox(
            "Select a document to read:",
            options=[item[0] for item in DOC_CATALOGUE],
            index=0,
            key="lit_lib_doc_select",
        )
    with col_search:
        search_q = st.text_input(
            f"Search a keyword across the {len(DOC_CATALOGUE)} documents shown here:",
            value="",
            placeholder="e.g., Gebru, IMAS 08.10, reliability, SHA-256...",
            key="lit_lib_doc_search",
        ).strip()

    if search_q:
        hits = []
        q_low = search_q.lower()
        for label, kind, rel_path in DOC_CATALOGUE:
            fp = root / rel_path
            if not fp.exists():
                continue
            txt = fp.read_text(encoding="utf-8")
            matching_lines = [
                f"L{idx+1}: {line.strip()}"
                for idx, line in enumerate(txt.splitlines())
                if q_low in line.lower()
            ]
            if matching_lines:
                hits.append(
                    {
                        "Document": label,
                        "Category": kind,
                        "Matches": len(matching_lines),
                        "Sample Matching Line": matching_lines[0][:140],
                    }
                )
        if hits:
            st.markdown(f"**Found `{search_q}` in {len(hits)} document(s):**")
            st.dataframe(pd.DataFrame(hits), width="stretch", hide_index=True)
        else:
            st.info(f"No matches found for `{search_q}` across the {len(DOC_CATALOGUE)} documents shown here.")

    chosen_entry = next((item for item in DOC_CATALOGUE if item[0] == selected_label), DOC_CATALOGUE[0])
    doc_label, doc_kind, rel_path = chosen_entry
    full_path = root / rel_path

    if not full_path.exists():
        st.warning(f"Document `{rel_path}` was not found on disk.")
        return

    raw_md = full_path.read_text(encoding="utf-8")
    st.markdown(
        f"**Viewing:** `{rel_path}` · **Framework:** {doc_kind} · **Size:** `{len(raw_md.encode('utf-8')):,} bytes`"
    )

    with st.container(border=True):
        st.markdown(raw_md)

    st.download_button(
        label=f"⬇️ Download `{rel_path}` (Markdown)",
        data=raw_md.encode("utf-8"),
        file_name=Path(rel_path).name,
        mime="text/markdown",
        key=f"dl_doc_{Path(rel_path).stem}",
    )


# =====================================================================================================================
# 5B. RELAND'S 63 FEATURES BY MUNICIPALITY, AND THE BASELINE-FIRST GATE (DATASHEET 03 & MODEL CARD 01)
# =====================================================================================================================
def _reland_cells(root: Path) -> tuple[pd.DataFrame, list[str]]:
    """RELand's 26,761 cells as the twin holds them, from the shared bundle (`loaders.get_twin_bundle`): the 63
    features with the grid keys (municipality code, row, column, area and zone IDs). The bundle holds no coordinates
    and no labels; RELand's labels stay with the Evaluation screen."""
    cells = loaders.get_twin_bundle(root)["setup"].twin.cells
    return cells, data.features_of(cells)


@st.cache_data
def _load_calibration_and_scores(root_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load `models/gate.csv`, `models/calibration.csv`, and `models/area_scores.csv`."""
    root = Path(root_str)
    gate_df = loaders.load_gate_table(root_str)
    cal_path = root / "models" / "calibration.csv"
    scores_path = root / "models" / "area_scores.csv"
    cal_df = pd.read_csv(cal_path, dtype={"municipality_code": str}) if cal_path.exists() else pd.DataFrame()
    scores_df = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    return gate_df, cal_df, scores_df


FAMILY_LOG = "Distances, population and night-time lights"
FAMILY_LAND_USE = "Land use (one-hot)"
FAMILY_CLIMATE = "Climate zone (one-hot)"
FAMILY_RELIEF = "Relief (one-hot)"
FAMILY_OTHER = "Other continuous: terrain, climate, soil and municipal figures"
FEATURE_FAMILIES = [FAMILY_LOG, FAMILY_LAND_USE, FAMILY_CLIMATE, FAMILY_RELIEF, FAMILY_OTHER]
ALL_FEATURES_OPTION = "All of RELand's features"


def _classify_reland_feature(feat_name: str) -> tuple[str, str]:
    """Return `(family, logistic regression's preprocessing in twin/prioritise.py)` for one of RELand's 63 features.
    The gradient-boosted trees read the raw values."""
    if feat_name in prioritise.LOG_FEATURES:
        return FAMILY_LOG, "log(1 + x), then standardised (`prioritise.LOG_FEATURES`)"
    if feat_name.startswith("land_use_"):
        return FAMILY_LAND_USE, "as it is (`ONEHOT_PREFIXES`)"
    if feat_name.startswith("weather_"):
        return FAMILY_CLIMATE, "as it is (`ONEHOT_PREFIXES`)"
    if feat_name.startswith("relief_"):
        return FAMILY_RELIEF, "as it is (`ONEHOT_PREFIXES`)"
    return FAMILY_OTHER, "standardised (`StandardScaler`, on the training municipalities)"


def _gate_summary(gate_df: pd.DataFrame) -> dict:
    """Counts and names from `models/gate.csv`: what the gate chose in each municipality, and what ranking alone
    would have chosen."""
    chosen = gate_df["chosen"].astype(str)
    by_rank = gate_df["chosen by ranking alone"].astype(str)
    names = gate_df["municipality"].astype(str)
    return {
        "n": len(gate_df),
        "chosen": {k: list(names[chosen == k]) for k in ("baseline", "lr", "hgb")},
        "by_rank": {k: int((by_rank == k).sum()) for k in ("baseline", "lr", "hgb")},
    }


def plot_reland_covariate_and_gate_diagnostics(
    cells_df: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> plt.Figure:
    """Render a 2-panel figure:
    - Left: the 15 municipalities' mean values of 12 of RELand's features, as z-scores across the municipalities (how
      far apart the municipalities are; risk Me4: chances far off in some municipalities).
    - Right: the baseline-first gate (`models/gate.csv`): for each municipality, the model's gain over the baseline in
      ranking (mean PR-AUC) against its gain in calibration, both measured on the other 14 municipalities.
    No panel shows places: the heat map's rows are municipalities.
    """
    fig, (ax_heat, ax_gate) = plt.subplots(
        1, 2, figsize=(16.2, 6.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.18, 1.0]}
    )
    for ax in (ax_heat, ax_gate):
        ax.set_facecolor(SURFACE)

    # ---- Left Panel: municipal means of 12 RELand features, as z-scores ----
    key_covs = [
        ("dist_old_mine", "Dist. past accident"),
        ("coca_dist", "Dist. coca fields"),
        ("elevation", "Elevation"),
        ("rainfall", "Rainfall"),
        ("temperature", "Temperature"),
        ("settlement_dist", "Dist. settlement"),
        ("dist_roads_t1", "Dist. primary road"),
        ("dist_roads_t3", "Dist. tertiary road"),
        ("indrural", "Rurality index"),
        ("pib_percapita_cons", "GDP per capita"),
        ("population_2012", "Population (2012)"),
        ("nighttime_lights_2012", "Night-time lights (2012)"),
    ]
    present_covs = [(c, lbl) for c, lbl in key_covs if c in cells_df.columns]
    muni_codes = list(config.PILOT.keys())
    muni_names = [config.NAME_OF[c] for c in muni_codes]

    muni_means = (
        cells_df.groupby("municipality_code")[[c for c, _ in present_covs]]
        .mean()
        .reindex(muni_codes)
    )
    z_mat = (muni_means - muni_means.mean(axis=0)) / muni_means.std(axis=0).replace(0, 1.0)
    z_arr = z_mat.to_numpy(dtype=float)

    im = ax_heat.imshow(z_arr, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    for r_i in range(z_arr.shape[0]):
        for c_i in range(z_arr.shape[1]):
            val = z_arr[r_i, c_i]
            txt_col = "#ffffff" if abs(val) > 1.25 else INK
            ax_heat.text(
                c_i,
                r_i,
                f"{val:+.1f}",
                ha="center",
                va="center",
                fontsize=6.8,
                color=txt_col,
            )

    ax_heat.set_yticks(np.arange(len(muni_names)))
    ax_heat.set_yticklabels(muni_names, fontsize=8.0, color=INK, fontweight="bold")
    ax_heat.set_xticks(np.arange(len(present_covs)))
    ax_heat.set_xticklabels([lbl for _, lbl in present_covs], fontsize=7.6, color=INK, rotation=32, ha="right")
    ax_heat.set_title(
        f"How the {len(muni_codes)} Municipalities Differ on {len(present_covs)} of RELand's Features (risk Me4)\n"
        f"Municipal Means as Z-Scores Across the {len(muni_codes)} Municipalities ({len(cells_df):,} Grid Cells)",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.032, pad=0.02)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label(f"Municipal mean (SDs from the mean of the {len(muni_codes)})", fontsize=7.6, color=INK_2)

    # ---- Right Panel: the baseline-first gate on ranking and calibration (`models/gate.csv`) ----
    ax_gate.spines["top"].set_visible(False)
    ax_gate.spines["right"].set_visible(False)

    if not gate_df.empty:
        # For each municipality, the model the gate chose, or else the model with the larger ranking gain
        x_pr_gain = []
        y_cal_gain = []
        labels_m = []
        chosen_final = []

        for _, row in gate_df.iterrows():
            m_name = str(row["municipality"])
            lr_pr_gain = float(row["lr: model mean PR-AUC (others)"] - row["lr: baseline mean PR-AUC (others)"])
            hgb_pr_gain = float(row["hgb: model mean PR-AUC (others)"] - row["hgb: baseline mean PR-AUC (others)"])
            ch = str(row["chosen"])
            best_m = ch if ch in ("lr", "hgb") else ("hgb" if hgb_pr_gain > lr_pr_gain else "lr")
            pr_g = float(row[f"{best_m}: model mean PR-AUC (others)"] - row[f"{best_m}: baseline mean PR-AUC (others)"])
            # y > 0: the model's calibration error is below the baseline's (the gate's calibration test passes)
            cal_g = float(
                row[f"{best_m}: baseline calibration error (others)"]
                - row[f"{best_m}: model calibration error (others)"]
            )
            x_pr_gain.append(pr_g)
            y_cal_gain.append(cal_g)
            labels_m.append(f"{m_name} ({best_m.upper()})")
            chosen_final.append(ch)

        x_lo, x_hi = min(min(x_pr_gain), 0.0), max(max(x_pr_gain), 0.0)
        y_lo, y_hi = min(min(y_cal_gain), 0.0), max(max(y_cal_gain), 0.0)
        x_pad, y_pad = 0.12 * (x_hi - x_lo) + 1e-3, 0.12 * (y_hi - y_lo) + 1e-3
        x_lim = (x_lo - x_pad, x_hi + 2.5 * x_pad)    # room on the right for the labels
        y_lim = (y_lo - 3.0 * y_pad, y_hi + y_pad)    # room at the bottom for the legend, below every point

        # Shade the quadrant where both tests pass: ranking gain > 0 and calibration gain > 0
        ax_gate.axhline(0.0, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_gate.axvline(0.0, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.8)
        ax_gate.add_patch(
            mpatches.Rectangle((0.0, 0.0), x_lim[1], y_lim[1], color="#dcfce7", alpha=0.38, zorder=1, linewidth=0)
        )

        col_map = {"hgb": "#0d6e52", "lr": "#2563eb", "baseline": "#dc2626"}
        for x_v, y_v, lbl, ch in zip(x_pr_gain, y_cal_gain, labels_m, chosen_final):
            c_dot = col_map.get(ch, "#64748b")
            marker = "s" if ch in ("lr", "hgb") else "o"
            ax_gate.scatter(x_v, y_v, color=c_dot, s=68, marker=marker, edgecolor="#0f172a", linewidth=0.8, zorder=4)
            ax_gate.annotate(
                lbl,
                (x_v, y_v),
                xytext=(4, 3 if y_v >= 0 else -8),
                textcoords="offset points",
                fontsize=7.3,
                fontweight="bold" if ch in ("lr", "hgb") else "normal",
                color=c_dot,
                zorder=5,
            )

        ax_gate.set_xlim(*x_lim)
        ax_gate.set_ylim(*y_lim)
        ax_gate.set_xlabel(
            "Ranking: Gain over the Baseline on the Other 14 Municipalities (Δ Mean PR-AUC, > 0 passes)",
            fontsize=8.1,
            color=INK_2,
        )
        ax_gate.set_ylabel(
            "Calibration: Gain over the Baseline on the Other 14\n(Baseline Error minus Model Error, > 0 passes)",
            fontsize=8.1,
            color=INK_2,
        )
        summary = _gate_summary(gate_df)
        n_m = summary["n"]
        n_rank = n_m - summary["by_rank"]["baseline"]
        n_both = len(summary["chosen"]["lr"]) + len(summary["chosen"]["hgb"])
        ax_gate.set_title(
            "Why the Gate Tests Calibration: the Optimiser's Curse (`models/gate.csv`)\n"
            f"{n_rank} of {n_m} Beat the Baseline on Ranking Alone; {n_both} of {n_m} Also on Calibration (Shaded)",
            fontsize=9.4,
            fontweight="bold",
            color=INK,
        )
        ax_gate.grid(True, linestyle=":", alpha=0.45)

        def _legend_label(text: str, names: list[str]) -> str:
            return f"{text} ({len(names)}" + (f": {', '.join(names)})" if 0 < len(names) <= 4 else ")")

        leg_handles = [
            mpatches.Patch(color="#0d6e52", label=_legend_label("Gate chose gradient-boosted trees", summary["chosen"]["hgb"])),
            mpatches.Patch(color="#2563eb", label=_legend_label("Gate chose logistic regression", summary["chosen"]["lr"])),
            mpatches.Patch(color="#dc2626", label=_legend_label("Gate kept the baseline", summary["chosen"]["baseline"])),
        ]
        ax_gate.legend(handles=leg_handles, loc="lower left", fontsize=7.4, frameon=True)

    fig.tight_layout()
    return fig


def plot_municipality_calibration_and_scores(
    muni_code: str,
    cal_df: pd.DataFrame,
    scores_df: pd.DataFrame,
) -> plt.Figure:
    """Plot the saved 25-bin monotone calibration curves (`models/calibration.csv`) for a selected municipality, and
    the distribution of the models' area scores over all areas (`models/area_scores.csv`).
    """
    muni_name = config.NAME_OF.get(muni_code, muni_code)
    fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(15.4, 4.4), facecolor=SURFACE)
    for ax in (ax_cal, ax_hist):
        ax.set_facecolor(SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    sub_cal = cal_df[cal_df["municipality_code"] == muni_code] if not cal_df.empty else pd.DataFrame()
    rank_Palette = {
        "baseline": ("#64748b", "Baseline (nearness to past accidents)"),
        "lr": ("#2563eb", f"Logistic regression (`lr`, C = {prioritise.LR_C:g})"),
        "hgb": ("#0d6e52", f"Gradient-boosted trees (`hgb`, learning rate {prioritise.HGB_LEARNING_RATE:g})"),
        "random": ("#cbd5e1", "Random order (`random`)"),
    }

    if not sub_cal.empty:
        for r_key, (col, lbl) in rank_Palette.items():
            r_sub = sub_cal[sub_cal["ranking"] == r_key].sort_values("score")
            if r_sub.empty:
                continue
            bins_x = np.arange(1, len(r_sub) + 1)
            ax_cal.plot(
                bins_x,
                100.0 * r_sub["chance"].to_numpy(dtype=float),
                marker="o",
                markersize=3.8,
                linewidth=2.0 if r_key != "random" else 1.2,
                color=col,
                label=lbl,
            )

    ax_cal.set_xlabel(
        f"Calibration Bin (1 = Lowest Scores to {prioritise.CALIBRATION_BINS} = Highest Scores)", fontsize=8.2, color=INK_2
    )
    ax_cal.set_ylabel("Chance of Holding Mines (%)", fontsize=8.2, color=INK_2)
    ax_cal.set_title(
        f"{prioritise.CALIBRATION_BINS}-Bin Monotone Calibration (`models/calibration.csv`)\n"
        f"Fitted on the Other 14 Municipalities for Held-Out {muni_name} ({muni_code})",
        fontsize=9.4,
        fontweight="bold",
        color=INK,
    )
    ax_cal.legend(loc="upper left", fontsize=7.6, frameon=True)
    ax_cal.grid(True, linestyle=":", alpha=0.45)

    # Right panel: the models' area scores over all areas (each municipality scored by a model blind to it)
    n_areas = len(scores_df)
    if not scores_df.empty:
        for col_k, hex_c, lbl_s in [("lr", "#2563eb", "Logistic regression (`lr`)"), ("hgb", "#0d6e52", "Gradient-boosted trees (`hgb`)")]:
            if col_k in scores_df.columns:
                vals = scores_df[col_k].dropna().to_numpy(dtype=float)
                ax_hist.hist(vals, bins=32, alpha=0.45, color=hex_c, edgecolor=hex_c, label=f"All {n_areas:,} areas: {lbl_s}")

    ax_hist.set_xlabel("Area Score Before Calibration (`models/area_scores.csv`)", fontsize=8.2, color=INK_2)
    ax_hist.set_ylabel("Number of Areas", fontsize=8.2, color=INK_2)
    ax_hist.set_title(
        f"The Models' Area Scores Across All {n_areas:,} Areas\n"
        "(each municipality scored by a model trained on the other 14; scores before calibration)",
        fontsize=9.4,
        fontweight="bold",
        color=INK,
    )
    ax_hist.legend(loc="upper right", fontsize=7.8, frameon=True)
    ax_hist.grid(True, linestyle=":", alpha=0.45)

    fig.tight_layout()
    return fig


def render_reland_covariate_and_gate_explorer(root: Path, key_prefix: str = "reland_exp") -> None:
    """RELand's 63 features by municipality, and the baseline-first gate (Datasheet 03 + Model Card 01)."""
    st.markdown(
        "#### 🧬 RELand's features by municipality, and the baseline-first gate (Datasheet 03, Model Card 01)"
    )
    st.caption(
        "Explores RELand's **grid cells (500 m × 500 m)** and their **63 open features** on terrain, climate, land "
        "use, relief, population and distances (Dulce Rubio et al., 2024; `docs/datasheets/03_reland.md`), as the "
        "twin holds them: no coordinates and no labels. It also shows the **baseline-first gate** (`models/gate.csv`, "
        "`docs/model_cards/01_risk_model.md`): a model is used for a municipality only if it beats the baseline, "
        "nearness to past accidents, on both ranking (mean PR-AUC) and calibration, on the other 14 municipalities. "
        "The calibration test was added after the first season replays showed the planner choosing the zones where "
        "a model's chances were furthest too high, the **optimiser's curse** (Smith & Winkler, 2006; notebook "
        "section 6.1)."
    )

    cells_df, feat_list = _reland_cells(root)
    gate_df, cal_df, scores_df = _load_calibration_and_scores(str(root))

    families = {f: _classify_reland_feature(f)[0] for f in feat_list}
    n_log = sum(1 for fam in families.values() if fam == FAMILY_LOG)
    n_onehot = sum(1 for fam in families.values() if fam in (FAMILY_LAND_USE, FAMILY_CLIMATE, FAMILY_RELIEF))
    n_other = len(feat_list) - n_log - n_onehot

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RELand Grid Cells", f"{len(cells_df):,}", f"500 m × 500 m across {cells_df['municipality_code'].nunique()} municipalities")
    c2.metric("RELand Features", f"{len(feat_list)} Features", f"{n_log} log-scaled · {n_other} scaled · {n_onehot} one-hot (logistic regression)")
    if not gate_df.empty:
        summary = _gate_summary(gate_df)
        n_m = summary["n"]
        rank = summary["by_rank"]
        chosen = summary["chosen"]
        c3.metric(
            "Beat the Baseline on Ranking Alone",
            f"{n_m - rank['baseline']} / {n_m} Municipalities",
            f"ranking alone picks logistic regression in {rank['lr']}, the trees in {rank['hgb']}",
        )
        c4.metric(
            "Beat It on Ranking and Calibration",
            f"{len(chosen['lr']) + len(chosen['hgb'])} / {n_m} Municipalities",
            f"trees: {', '.join(chosen['hgb']) or 'none'}; logistic regression: {', '.join(chosen['lr']) or 'none'}",
        )
        c5.metric("Baseline Used", f"{len(chosen['baseline'])} / {n_m} Municipalities", "nearness to past accidents")

    fig_diag = plot_reland_covariate_and_gate_diagnostics(cells_df, gate_df)
    st.pyplot(fig_diag)
    plt.close(fig_diag)

    st.markdown("##### 📈 Calibration by municipality (`models/calibration.csv`) and RELand's features")
    col_m, col_f = st.columns([1.1, 1.3])
    muni_options = list(config.PILOT.keys())
    with col_m:
        sel_muni = st.selectbox(
            "Select a municipality to inspect its held-out 25-bin calibration:",
            options=muni_options,
            index=muni_options.index("05756") if "05756" in muni_options else 0,
            format_func=lambda c: f"{config.NAME_OF[c]} ({c})",
            key=f"{key_prefix}_muni_cal",
        )
    with col_f:
        fam_counts = {fam: sum(1 for v in families.values() if v == fam) for fam in FEATURE_FAMILIES}
        fam_options = [f"{ALL_FEATURES_OPTION} ({len(feat_list)})"] + [
            f"{fam} ({fam_counts[fam]})" for fam in FEATURE_FAMILIES
        ]
        sel_fam_label = st.selectbox(
            "Filter RELand's features by family:",
            options=fam_options,
            index=0,
            key=f"{key_prefix}_fam_filter",
        )
        sel_fam = next((fam for fam in FEATURE_FAMILIES if sel_fam_label.startswith(fam + " (")), ALL_FEATURES_OPTION)

    fig_cal = plot_municipality_calibration_and_scores(sel_muni, cal_df, scores_df)
    st.pyplot(fig_cal)
    plt.close(fig_cal)

    # The feature table: each feature's statistics in the selected municipality and across the 15 municipalities
    sub_m_cells = cells_df[cells_df["municipality_code"] == sel_muni]
    cov_rows = []
    for idx, f_name in enumerate(feat_list):
        fam, preproc = _classify_reland_feature(f_name)
        if sel_fam != ALL_FEATURES_OPTION and fam != sel_fam:
            continue
        all_s = cells_df[f_name]
        m_s = sub_m_cells[f_name]
        cov_rows.append(
            {
                "#": idx + 1,
                "RELand Feature Name": f_name,
                "Family": fam,
                "Logistic Regression's Preprocessing (`twin/prioritise.py`; the trees use raw values)": preproc,
                f"{config.NAME_OF[sel_muni]} Mean": round(float(m_s.mean()), 3),
                f"{config.NAME_OF[sel_muni]} Median": round(float(m_s.median()), 3),
                "15-Municipality Mean": round(float(all_s.mean()), 3),
                "15-Municipality Min–Max": f"{float(all_s.min()):.2f} – {float(all_s.max()):.2f}",
            }
        )
    st.markdown(
        f"**RELand's features ({len(cov_rows)} shown): {config.NAME_OF[sel_muni]} beside the 15 municipalities:**"
    )
    st.dataframe(pd.DataFrame(cov_rows), width="stretch", hide_index=True)


# =====================================================================================================================
# 6. MAIN STUDIO RENDERER: LITERATURE, STANDARDS, DATASHEETS & GOVERNANCE LIBRARY
# =====================================================================================================================
def render_literature_library_studio(root: Path, key_prefix: str = "lit_studio") -> None:
    """Render the library: references and datasets, the risk register, the report checker, RELand's features and the
    gate, the documents reader, and the earlier projects as insights."""
    st.caption("References, standards, Colombian open data, datasheets, model cards and the risk register")
    st.header("📚 Literature, Standards, Datasheets & Governance Library")
    st.markdown(
        "The twin draws on **peer-reviewed research** (Dulce Rubio et al., 2024; Kritzinger et al., 2018; Smith & "
        "Winkler, 2006; Parasuraman & Manzey, 2010), the **International Mine Action Standards (IMAS)** and Mine "
        "Action Review's report on Colombia, **Colombian open data** (`datos.gov.co`) and IDEAM's description of the "
        "rainy seasons, and **frameworks for documenting and governing AI and data** (Gebru et al., 2021; Mitchell et "
        "al., 2019; NIST, 2023; IASC, 2023). Its settings are the project's (`twin/config.py`), and the pilot's "
        "national criteria are invented for it; only \"no technical survey\" follows Colombia (Mine Action Review, 2025)."
    )

    n_refs = sum(1 for it in LITERATURE_CATALOGUE if it["kind"] == "reference")
    n_data = sum(1 for it in LITERATURE_CATALOGUE if it["kind"] == "dataset")
    n_datasheets = len(list((root / "docs" / "datasheets").glob("*.md")))
    n_model_cards = len(list((root / "docs" / "model_cards").glob("*.md")))
    register_df = build_risk_register_table(root)
    by_function = (
        register_df.groupby("NIST AI RMF function", sort=False).size().to_dict() if not register_df.empty else {}
    )
    cells_df, feat_list = _reland_cells(root)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("References & Datasets", f"{n_refs} References · {n_data} Datasets", "the notebook's references (section 12)")
    k2.metric("Dataset Datasheets", f"{n_datasheets} Datasheets", "Gebru et al. (2021) + SHA-256")
    k3.metric("Model Cards", f"{n_model_cards} Model Cards", "Mitchell et al. (2019), each with a section on replacing it")
    k4.metric(
        "Risk Register (NIST AI RMF)",
        f"{len(register_df)} Risks",
        " · ".join(f"{fn} ({n})" for fn, n in by_function.items()),
    )
    k5.metric("RELand Grid", f"{len(feat_list)} Features · {len(cells_df):,} Cells", "the baseline-first gate (`models/gate.csv`)")

    tab_cat, tab_nist, tab_checker, tab_reland, tab_docs, tab_more = st.tabs(
        [
            "📚 1. References & Datasets (+ BibTeX/APA)",
            "🛡️ 2. Risk Register (`docs/risk_register.md`)",
            "🔍 3. Report Checker (`twin/checker.py`)",
            "🧬 4. RELand's Features & the Gate (`Datasheet 03`)",
            "📄 5. Datasheets, Rules & Prompts Reader",
            "🎓 6. The Earlier Projects as Insights (P1–P7)",
        ]
    )

    with tab_cat:
        st.markdown("#### 🗺️ Sources, insights and where they show in the twin")
        st.caption(
            "Each row links sources (left), an insight of the earlier projects or the NIST AI RMF (centre), and where it "
            "shows in the twin's code, models and documents (right). The dashed arrows: the season planner also "
            "carries P3's insight, and the risk overlay's monthly learning P2's."
        )
        fig_trace = plot_literature_traceability_diagram()
        st.pyplot(fig_trace)
        plt.close(fig_trace)

        st.divider()
        st.markdown("#### 🔎 Searchable references and datasets")
        c_f1, c_f2 = st.columns([1.1, 1.4])
        categories = ["All Categories"] + sorted({row["category"] for row in LITERATURE_CATALOGUE})
        with c_f1:
            sel_cat = st.selectbox(
                "Filter by Domain / Category:",
                options=categories,
                index=0,
                key=f"{key_prefix}_cat_filter",
            )
        with c_f2:
            kw = st.text_input(
                "Search by author, title, DOI, IMAS number, or codebase module:",
                value="",
                placeholder="e.g., Dulce Rubio, IMAS 07.11, Parasuraman, datos.gov.co, planner.py...",
                key=f"{key_prefix}_kw_filter",
            ).strip()

        filtered = []
        for item in LITERATURE_CATALOGUE:
            if sel_cat != "All Categories" and item["category"] != sel_cat:
                continue
            if kw:
                blob = " ".join(str(v) for v in item.values()).lower()
                if kw.lower() not in blob:
                    continue
            filtered.append(item)

        summary_rows = [
            {
                "ID": it["ref_id"],
                "Reference": it["short_cite"],
                "Category": it["category"],
                "Title": it["title"],
                "Venue / Publisher": it["venue"],
                "Files It Grounds": it["modules"],
            }
            for it in filtered
        ]
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        if filtered:
            st.markdown("##### 📖 Reference card")
            sel_ref_id = st.selectbox(
                "Select a reference to inspect what it says, where it shows in the repository, and its APA and BibTeX entries:",
                options=[it["ref_id"] for it in filtered],
                format_func=lambda rid: next(
                    f"{x['ref_id']} — {x['short_cite']}: {x['title']}" for x in filtered if x["ref_id"] == rid
                ),
                key=f"{key_prefix}_ref_detail",
            )
            ref = next(x for x in filtered if x["ref_id"] == sel_ref_id)
            with st.container(border=True):
                st.markdown(f"### {ref['ref_id']}: {ref['title']}")
                st.markdown(
                    f"**Authors / Publisher:** {ref['authors']} ({ref['year']})  \n"
                    f"**Category:** `{ref['category']}` · **Venue / Series:** {ref['venue']}  \n"
                    f"**DOI / Official Source:** `{ref['doi_or_url']}` · **Edition / Licence:** `{ref['license_or_edition']}`"
                )
                st.markdown(f"**What it says, as this project uses it:**  \n{ref['key_findings']}")
                st.markdown(f"**Where it shows in the repository:**  \n{ref['codebase_grounding']}")
                st.markdown(f"**Code & Documentation Files:** `{ref['modules']}`")

                c_apa, c_bib = st.columns(2)
                with c_apa:
                    st.markdown(
                        "**APA Citation" + (" (as the notebook's section 12 gives it):**" if ref["kind"] == "reference" else ":**")
                    )
                    st.code(ref["apa_citation"], language="text")
                with c_bib:
                    st.markdown("**BibTeX Entry:**")
                    st.code(ref["bibtex"], language="bibtex")

        # Full Bibliography Export Buttons
        all_bib = "\n\n".join(it["bibtex"] for it in LITERATURE_CATALOGUE)
        all_apa = "\n\n".join(f"- {it['apa_citation']}" for it in LITERATURE_CATALOGUE)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download Complete Bibliography (`.bib` BibTeX)",
                data=all_bib.encode("utf-8"),
                file_name="antioquia_mine_action_twin_references.bib",
                mime="text/plain",
                key=f"{key_prefix}_dl_bib",
            )
        with dl2:
            st.download_button(
                "⬇️ Download Complete Reference List (`.md` APA 7th)",
                data=("# References — Antioquia Humanitarian Mine Action Digital Twin\n\n" + all_apa + "\n").encode("utf-8"),
                file_name="antioquia_mine_action_twin_references_apa.md",
                mime="text/markdown",
                key=f"{key_prefix}_dl_apa",
            )

    with tab_nist:
        st.markdown("#### 🛡️ The risk register, by the NIST AI RMF's four functions (`docs/risk_register.md`)")
        intro, _ = read_risk_register(root)
        if intro:
            st.markdown(intro)
        st.caption(
            f"The register's table as written, without its column \"{REGISTER_EVIDENCE_COLUMN}\" and without "
            f"{len(REGISTER_RESULT_PASSAGES)} passages of its other columns, each marked where it is left out: they "
            "include results of the full run's evaluation (the season replay, the injected scenarios or the expert "
            "review), which are not shown on this screen."
        )
        if register_df.empty:
            st.warning(f"`{RISK_REGISTER_FILE}` was not found on disk.")
        else:
            functions = list(by_function)
            fn_filter = st.radio(
                "Filter by NIST AI RMF function:",
                options=[f"All Functions ({len(register_df)})"] + [f"{fn} ({n})" for fn, n in by_function.items()],
                horizontal=True,
                key=f"{key_prefix}_nist_fn",
            )
            chosen_fn = next((fn for fn in functions if fn_filter.startswith(fn + " (")), None)
            nist_view = register_df[register_df["NIST AI RMF function"] == chosen_fn] if chosen_fn else register_df
            st.dataframe(nist_view, width="stretch", hide_index=True)

    with tab_checker:
        render_report_checker_inspector(root)

    with tab_reland:
        render_reland_covariate_and_gate_explorer(root, key_prefix=f"{key_prefix}_reland")

    with tab_docs:
        render_verbatim_docs_reader(root)

    with tab_more:
        render_p1_to_p7_synthesis_walkthrough(key_prefix=f"{key_prefix}_p1p7")


# The earlier projects as insights: each project's insight and where it shows, as the notebook's section 2.4 (cell c05)
# states them, with the `insight` the twin's code declares (twin/*.py). The repository names the repositories of P1 to
# P3 (data/README.md) and of P7 (README.md) only.
P1_TO_P7_WALKTHROUGH = [
    {
        "stage": "P1 — `ai-programming-foundations-project`",
        "insight": "Check data as it arrives: labels that do not match their values, values written several ways, repeated rows that are real",
        "in_code": (
            "`ReportChecker.insight = \"P1: check data as it arrives\"` (`twin/checker.py`); the loaders of "
            "`twin/data.py` apply it to the three datasets"
        ),
        "where": "the data layers' loaders (section 4); the report checker (section 7.2)",
        "detail": (
            "P1's dataset is reused as the twin's victims layer (datos.gov.co `yhxn-eqqw`). P1 found that three "
            "published labels do not match their values (`estado` holds sex, `genero` killed or injured, `tipoevento` "
            "the type of place), that \"no information\" is written three ways, and that repeated rows are real "
            "victims, not duplicates: 3,858 rows (31%) exactly repeat another row and are kept as separate victims "
            "(`twin/data.py`; `docs/datasheets/01_victims.md`). The report checker carries the same insight to the "
            "teams' reports: coded checks hold impossible or inconsistent reports, and an isolation forest flags "
            "unusual ones for a query."
        ),
        "in_app": "📚 this library, tab 3 (the report checker); 1. Weekly Operations (team reports; restricted)",
    },
    {
        "stage": "P2 — `statistical-analysis-project`",
        "insight": "Show uncertainty as intervals, and watch change over time",
        "in_code": (
            "`CompletionForecaster.insight = \"P2: uncertainty shown as intervals\"` (`twin/forecast.py`); "
            "`SeasonPlanner.insight` ends \"P2: plans compared on their outcomes, with ranges\" (`twin/planner.py`); "
            "the risk overlay's monthly learning, \"P2's insight: watch change over time\" (`twin/prioritise.py`)"
        ),
        "where": (
            "the completion forecaster (section 6.4); the season replay's ranges (section 9); the risk overlay's "
            "learning in the season (section 6.1)"
        ),
        "detail": (
            "P2's dataset is reused as the twin's demining layer (datos.gov.co `jbqm-s5tn`), a check of the simulated "
            "clearance capacity against the pilot's past clearance by year (notebook section 4.3). P2 found that 1,183 "
            "combinations of municipality, month, demining type and operator appear twice, each pair splitting "
            "results from hazardous areas, so the rows are summed, not de-duplicated "
            "(`docs/datasheets/02_demining.md`). Here the forecaster's 200 draws give each zone's P50 and P90 weeks "
            "and the season's totals as P10, P50 and P90; and each month the risk overlay updates one odds multiplier "
            "per municipality from the season's own results, with no label read."
        ),
        "in_app": "2. Plan & Completion Forecast (the completion forecast and the risk overlay's monthly learning; restricted)",
    },
    {
        "stage": "P3 — `applied-machine-learning-project`",
        "insight": "A simple baseline first, tested on held-out municipalities",
        "in_code": (
            "`RiskOverlay.insight = \"P3: a simple baseline first, tested on held-out municipalities\"` "
            "(`twin/prioritise.py`); `SeasonPlanner.insight` starts \"P3: a simple plan first\" (`twin/planner.py`)"
        ),
        "where": "the risk overlay's gate (section 6.1); the planner against a rule of thumb (section 9)",
        "detail": (
            "P3's dataset is reused as the twin's hazard and land-use layers (RELand's cells, Hugging Face "
            "`mdulcer10/landmine-detection`). In P3, nearness to past landmine accidents beat both of its models in 9 "
            "of its 15 held-out municipalities (`twin/prioritise.py`). Here each municipality's chances come from the "
            "other 14 municipalities' labels, and a model is used for a municipality only if it beats the baseline "
            "there on ranking and on calibration (the baseline-first gate, `models/gate.csv`)."
        ),
        "in_app": (
            "📚 this library, tab 4 (RELand's features and the gate); 8. About & Data Rules (the gate by "
            "municipality); 2. Plan & Completion Forecast (the integer programme beside the rule of thumb; restricted)"
        ),
    },
    {
        "stage": "P4",
        "insight": "A model should answer \"unknown\" when it is unsure",
        "in_code": (
            "the overlay contract (`twin/overlays.py`): \"insufficient evidence\", with the reason, instead of a weak "
            "answer (P4's insight)"
        ),
        "where": "every overlay can answer \"insufficient evidence\" (sections 5 to 8)",
        "detail": (
            "Every decision tool answers through one contract: a recommendation with the evidence it relied on, its "
            "uncertainty and its limits, or \"insufficient evidence\" and the reason. An overlay also answers "
            "\"insufficient evidence\" when a live layer it reads has not been updated for longer than the layer "
            "allows (notebook section 2.3)."
        ),
        "in_app": "1. Weekly Operations (\"insufficient evidence\" notices; restricted)",
    },
    {
        "stage": "P5",
        "insight": (
            "Synthetic data serve stress tests, never proof of field performance; generated output is checked before use"
        ),
        "in_code": (
            "`SituationReport.insight = \"P5: generated output is checked against its source before it is used\"` "
            "(`twin/sitrep.py`); the injected scenarios, \"P5's insight: synthetic data serve stress tests\" "
            "(`twin/scenarios.py`)"
        ),
        "where": (
            "the simulated field and the scenarios (sections 9 and 10); the situation report's figure check "
            "(section 7.3)"
        ),
        "detail": (
            "The season replay runs against a simulated field under stated settings, so its results show how the "
            "system behaves under those settings, not how it would do in the field (notebook section 1.5). Five "
            "scenarios inject things that go wrong in a season: a funding shock, a zone closure, new contamination in "
            "a zone declared free, a stale or corrupted feed, and a dossier with a missing item (`twin/scenarios.py`). "
            "The figure check reads every number in each weekly draft and removes each sentence with a number that is "
            "not among the week's facts (`twin/sitrep.py`)."
        ),
        "in_app": (
            "5. Weekly Situation Reports (each draft and its figure check; restricted); 7. Evaluation (the season "
            "replay and the scenarios)"
        ),
    },
    {
        "stage": "P6",
        "insight": "An evidence register, a coded veto and a person's sign-off",
        "in_code": (
            "`AgentReviewer.insight = \"P6\"` (`twin/review.py`); the agent follows P6's pattern (`twin/agent.py`); the "
            "rule check is adapted from P6's rules R1 to R7 (`twin/rules.py`)"
        ),
        "where": "the dossier reviewer (section 8); the one log as the only write-back (section 5)",
        "detail": (
            "After each visit, the dossier reviewer agent reads the area's dossier through five tools "
            "(`read_dossier`, `record_evidence`, `lookup_rules`, `check_decision`, `submit`), records each item in an "
            "evidence register, and tests its decision against the rule check (R1–R6), which can veto. A person in a "
            "stated role signs it off; in the simulated season, the QC reviewer (season sign-off policy: signs off "
            "what passes the rule check). The rule check tests the register, not the world: a reading error that fits "
            "the outcome passes it (`twin/rules.py`)."
        ),
        "in_app": "4. Dossier Sign-Off Reviews (each sign-off sheet and the rule check's results; restricted)",
    },
    {
        "stage": "P7 — `ai-systems-synthesis-project` (this project)",
        "insight": (
            "The six earlier projects appear as insights; none is reused as code, and P1's, P2's and P3's datasets "
            "are reused as data layers (section 2.4)"
        ),
        "in_code": (
            "no P7 insight is declared in `twin/`; `twin/overlays.py` (`overlay_table`) lists each decision tool with "
            "the insight it carries"
        ),
        "where": "the whole notebook; the twin is by design a digital shadow with approved write-back (section 2.1)",
        "detail": (
            "Kritzinger et al. (2018) distinguish a digital model, a digital shadow and a digital twin by how data flow "
            "between the physical object and the model. This system is a digital shadow with approved write-back, by "
            "choice: reports flow into the twin as they arrive, but nothing flows back to the field, and no land "
            "changes status, without a person's decision (notebook section 2.1). Six decision tools across five AI "
            "domains read the twin through one contract, and one log records every report and every decision."
        ),
        "in_app": "every screen of this application",
    },
]


def render_p1_to_p7_synthesis_walkthrough(key_prefix: str = "p1p7") -> None:
    """The earlier projects as insights (notebook section 2.4), with where the twin's code declares each."""
    st.markdown("#### 🎓 The earlier projects as insights (notebook section 2.4)")
    st.caption(
        "Shows the insight each earlier project gives the twin, as the notebook's section 2.4 states it, where the "
        "twin's code declares it, and where it shows. All six contribute; none is reused as code, and P1's, P2's "
        "and P3's datasets are reused as data layers."
    )

    overview_rows = [
        {
            "Project": item["stage"].split("—")[0].strip(),
            "Insight (Notebook Section 2.4)": item["insight"],
            "Declared in `twin/`": item["in_code"],
            "Where It Shows (Notebook)": item["where"],
        }
        for item in P1_TO_P7_WALKTHROUGH
    ]
    st.dataframe(pd.DataFrame(overview_rows), width="stretch", hide_index=True)

    sel_stage = st.selectbox(
        "Select a project (`P1` to `P7`) to inspect its insight, where the code declares it, and where it shows:",
        options=[it["stage"] for it in P1_TO_P7_WALKTHROUGH],
        index=6,
        key=f"{key_prefix}_stage_sel",
    )
    st_item = next(it for it in P1_TO_P7_WALKTHROUGH if it["stage"] == sel_stage)

    with st.container(border=True):
        st.markdown(f"### {st_item['stage']}")
        st.markdown(
            f"**Insight (notebook section 2.4):** {st_item['insight']}  \n"
            f"**Declared in the twin's code:** {st_item['in_code']}"
        )
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown("**1. What it means in this project:**")
            st.write(st_item["detail"])
        with c_p2:
            st.markdown("**2. Where it shows (notebook):**")
            st.write(st_item["where"])
            st.markdown("**3. Where to look in this application:**")
            st.write(st_item["in_app"])

    st.divider()
    st.markdown("#### 🧭 How the twin is grounded: five kinds of source")
    st.markdown(
        """
1. **Land use and the grid (Dulce Rubio et al., 2024: RELand):**
   - The twin's areas are blocks of 2 × 2 RELand cells, cut at municipal borders. Each carries attributes computed from RELand's 63 features: its main land use (`farming`, `livestock`, `agroforestry`, `forest`, `soil conservation`, `water`, `urban`), its distance to the nearest road (`road_km`) and settlement (`settlement_km`), and its population (`population`).
2. **Colombia's national open data (`datos.gov.co` `yhxn-eqqw` and `jbqm-s5tn`):**
   - 12,390 victim records (1990–2023), published by the **Presidencia de la República de Colombia**, which enter the twin only as counts by municipality and month; and 5,721 humanitarian demining records (2004–2023), published by the **Departamento Administrativo de la Presidencia de la República (DAPRE)**. `twin/data.py` checks each file against its SHA-256 checksum before loading it.
3. **Rainfall, zones and access (IDEAM, 2025; Mine Action Review, 2025):**
   - The rainfall layer is simulated from IDEAM's description of two rainy seasons in the Andean region (**March–April** and **October–November**), with about half the working days lost to heavy rain or high wind in those months and one in ten in the others (the project's settings, `twin/config.py`). Each municipality's areas are grouped into operational zones of about 25 areas by position and road access (292 zones), since Colombia's municipalities "are subdivided into operational zones based on accessibility" (Mine Action Review, 2025). A zone where communities must be consulted is not tasked until four weeks after the consultation is asked for, and each zone has a 1-in-50 chance a month of closing for one to three months (settings in `twin/config.py`).
4. **IMAS 07.11, 08.10 and 08.20, and the pilot's national criteria (`rules/land_release_rules.md`, `twin/config.py`):**
   - Every dossier review records each item's reliability class, 1 to 7, from **IMAS 08.10 (§8.2)** (class 1: direct physical evidence seen and recorded by the survey team; class 7: people told about the hazard by others who cannot be identified as first-hand sources), and every recommendation is tested by the rule check (`R1` to `R6` in `twin/rules.py`) before the QC reviewer signs it off. The national criteria are invented for this project, since IMAS leaves such numbers to each national authority; only "no technical survey" follows Colombia (Mine Action Review, 2025).
5. **AI governance, human factors and data responsibility (`NIST, 2023`, `Parasuraman & Manzey, 2010`, `IASC, 2023`):**
   - The risk register follows the NIST AI RMF's four functions (`G1` to `Ma5`). Against automation bias (`Ma1`), every recommendation comes with its evidence register and rule check, and an override needs a written reason and is logged. The situation report's figures are checked against the week's facts (`figure_check`, `twin/sitrep.py`); the report checker holds impossible, duplicate and wrong-task reports and queries unusual ones (`Me6`); and the restricted layers stay inside the twin (`G3`; data rule 5 in `app/README.md`).
"""
    )
