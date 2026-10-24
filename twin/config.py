"""Settings shared by the twin and its services.

Every number that shapes a result is here, with who set it: Travis's settings come from his answers in the
project interview; the others are marked as this project's settings, and section 10 of the notebook varies
the ones that matter.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
FIGURES_DIR = ROOT / "figures"
RULES_FILE = ROOT / "rules" / "land_release_rules.md"
PROMPT_FILE = ROOT / "prompts" / "system_prompt.md"

# Each data file, with the SHA-256 checksum published by the earlier project that first used it (data/README.md)
DATA_FILES = {
    "victims": ("p1_victims/victimas_minas_antipersonal.csv",
                "6ac13df53d0024bc9d96f8b48d7622934b5bd1a334d40cf4155bf4cea804a84d"),
    "demining": ("p2_demining/desminado_humanitario.csv",
                 "eda568a71903f05d7468c582b3795dbd5a537a71c5413f84f72f0acbcffb6617"),
    "reland": ("p3_reland/landmine_detection.csv",
               "f2ba3b981487ae3879175db4521bc1c6d02352d1053dc9889715eb5e703fe371"),
    "ctx_crops": ("p4_ctx_uxo/ctx_uxo_crops_64.zip",
                  "9f3cbc88609089864fba9b418c39ec8eb9a97a2451ffdddb2116b1a1695c87c1"),
    "ctx_manifest": ("p4_ctx_uxo/crops_manifest.csv",
                     "4a4322ff01b9390b14e5a7d416676cb1d6e457cf725e5e2ac10aba04f7e8c306"),
    "suland_crops": ("p5_suland/mine_crops_64.zip",
                     "2ddc12cd38dac2b2d621341171b3163d6e7b74d018854b09edabc7cceaa49adb"),
    "suland_manifest": ("p5_suland/crops_manifest.csv",
                        "cac0c5593c0b37df522bc7ac3762bf1801d0edf11f5148760c37dbecfe69c286"),
}

SEED = 0  # one seed for every random step

# The pilot: RELand's 15 municipalities, by DANE code, as RELand's file spells them, as they are written, and the
# three letters that start their area IDs. El Carmen de Viboral is "CARMEN DE VIBORAL" in the national tables.
PILOT = {
    "05002": ("ABEJORRAL", "Abejorral", "ABE"),
    "05021": ("ALEJANDRÍA", "Alejandría", "ALE"),
    "05148": ("EL CARMEN DE VIBORAL", "El Carmen de Viboral", "ECV"),
    "05172": ("CHIGORODÓ", "Chigorodó", "CHI"),
    "05197": ("COCORNÁ", "Cocorná", "COC"),
    "05313": ("GRANADA", "Granada", "GRA"),
    "05400": ("LA UNIÓN", "La Unión", "LUN"),
    "05483": ("NARIÑO", "Nariño", "NAR"),
    "05628": ("SABANALARGA", "Sabanalarga", "SAB"),
    "05649": ("SAN CARLOS", "San Carlos", "SCA"),
    "05652": ("SAN FRANCISCO", "San Francisco", "SFR"),
    "05660": ("SAN LUIS", "San Luis", "SLU"),
    "05667": ("SAN RAFAEL", "San Rafael", "SRA"),
    "05670": ("SAN ROQUE", "San Roque", "SRO"),
    "05756": ("SONSÓN", "Sonsón", "SON"),
}
CODE_OF = {reland_name: code for code, (reland_name, _, _) in PILOT.items()}
NAME_OF = {code: name for code, (_, name, _) in PILOT.items()}

# The grid. RELand's cells are 500 m x 500 m, with centres 0.004487 degrees apart in latitude and in longitude;
# the twin turns each centre into a row and a column once, and keeps no coordinate after that.
GRID_STEP_DEGREES = 0.004487
CELL_KM = 0.5
AREA_CELLS = 2            # an area is a block of 2 x 2 cells (1 km²), cut at municipal borders (Travis's setting)
EVENT_INSIDE_KM = 0.25    # a past accident closer than this to a cell's centre lies inside that cell


@dataclass(frozen=True)
class Assumptions:
    """What the simulated field teams return in the replay.

    Travis's settings (project interview, questions 13 to 18), except where marked as this project's setting.
    """
    # Non-technical survey of an area that holds mines: direct evidence, indirect evidence only, nothing
    mined: tuple = (0.50, 0.35, 0.15)
    # Non-technical survey of an area with no mines: safe use from reliable sources, rumours, nothing conclusive
    clean: tuple = (0.60, 0.15, 0.25)
    # Land use shapes the chances: signs of safe use likelier on farmed or grazed land near settlements, rarer in
    # forest or remote land. The size of the effect, in log-odds per standard deviation of the land-use score,
    # is this project's setting (0 means the same everywhere).
    land_use_effect: float = 0.5
    # Technical survey finds mines in this share of the mined areas it is sent to; in clean areas it finds nothing.
    # This project's addition: technical survey that starts from an item already seen always confirms it.
    ts_detection: float = 0.95
    # This project's setting, to be confirmed by Travis: on land with mines, a visit that finds nothing returns
    # what a visit to clean land returns (so it can look like safe use). False: it returns nothing conclusive.
    nothing_looks_clean: bool = True
    # Survey capacity: areas visited per round, and rounds
    visits_per_round: int = 20
    rounds: int = 8


@dataclass(frozen=True)
class Criteria:
    """The pilot's national criteria for cancellation, which the rule check enforces.

    Invented, like P6's criteria boxes, since IMAS sets no numbers (Travis's choices, interview questions 19 to 23).
    "No unexplained direct evidence" is read strictly (question 22): evidence of absence never explains direct
    evidence away, since direct evidence leads to technical survey or clearance (IMAS 07.11, §4.4).
    """
    min_years: int = 5        # years of safe use each source must report
    sources: int = 2          # reliable first-hand sources needed
    # Travis's setting (question 21): weaker indirect evidence (class 5 to 7: second-hand reports, rumours, the
    # national database's unchecked record) becomes a target for technical survey only if it is still unexplained
    # after this many non-technical survey visits (1: technical survey after the first visit).
    follow_up_visits: int = 2
    # Travis's setting (question 23): an area where the national database records a past landmine accident inside
    # is released only after technical survey, never by non-technical survey alone. Its record is then a target for
    # technical survey after the first visit, since a second visit could not release it.
    recorded_accident_needs_ts: bool = True

    def as_text(self):
        visits = f"{self.follow_up_visits} visit{'s' if self.follow_up_visits != 1 else ''}"
        return (f"Cancellation by non-technical survey is allowed when at least {self.sources} different reliable "
                f"first-hand sources (reliability class 1 to 4) each report at least {self.min_years} years of safe use "
                "of the area, there is no direct evidence in the area, and no indirect evidence is left unexplained. "
                "Clearance without technical survey is allowed where direct evidence of reliability class 1 to 4 "
                "defines the hazard. Weaker indirect evidence (reliability class 5 to 7) is followed up by "
                "non-technical survey before technical survey: it becomes a target for technical survey if it is still "
                f"unexplained after {visits}."
                + (" An area where the national database records a past landmine accident inside is released only "
                   "after technical survey, never by non-technical survey alone; its record is a target for technical "
                   "survey after the first visit." if self.recorded_accident_needs_ts else ""))
