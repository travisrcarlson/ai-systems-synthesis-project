"""Settings shared by the twin and its overlays.

Every number that shapes a result is here, with who set it: Travis's settings come from his answers in the
project interview (questions 13 to 23 for survey and the national criteria, 28 to 40 for the season); the
others are marked as this project's settings, and the notebook states them where they are used.
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
SITREP_PROMPT_FILE = ROOT / "prompts" / "sitrep_prompt.md"

# Each data file, with the SHA-256 checksum published by the earlier project that first used it (data/README.md)
DATA_FILES = {
    "victims": ("p1_victims/victimas_minas_antipersonal.csv",
                "6ac13df53d0024bc9d96f8b48d7622934b5bd1a334d40cf4155bf4cea804a84d"),
    "demining": ("p2_demining/desminado_humanitario.csv",
                 "eda568a71903f05d7468c582b3795dbd5a537a71c5413f84f72f0acbcffb6617"),
    "reland": ("p3_reland/landmine_detection.csv",
               "f2ba3b981487ae3879175db4521bc1c6d02352d1053dc9889715eb5e703fe371"),
}

SEED = 0  # one seed for every random step that is not part of the simulated field

# The pilot: RELand's 15 municipalities, by DANE code, as RELand's file spells them, as they are written, and the
# three letters that start their area and zone IDs. El Carmen de Viboral is "CARMEN DE VIBORAL" in the national tables.
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

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


@dataclass(frozen=True)
class Assumptions:
    """What a simulated non-technical survey visit to an area returns.

    Travis's settings (project interview, questions 13 to 18 and 20), except where marked as this project's setting.
    """
    # A visit to an area that holds mines: direct evidence, indirect evidence only, nothing
    mined: tuple = (0.50, 0.35, 0.15)
    # A visit to an area with no mines: safe use from reliable sources, rumours, nothing conclusive
    clean: tuple = (0.60, 0.15, 0.25)
    # Land use shapes the chances: signs of safe use likelier on farmed or grazed land near settlements, rarer in
    # forest or remote land. The size of the effect, in log-odds per standard deviation of the land-use score,
    # is this project's setting (0 means the same everywhere).
    land_use_effect: float = 0.5
    # Technical survey finds mines in this share of the mined areas it is sent to; in clean areas it finds nothing.
    # This project's addition: technical survey that starts from an item already seen always confirms it.
    ts_detection: float = 0.95
    # Travis kept this as built (question 20): on land with mines, a visit that finds nothing returns what a visit
    # to clean land returns (so it can look like safe use). False: it returns nothing conclusive.
    nothing_looks_clean: bool = True


@dataclass(frozen=True)
class Criteria:
    """The pilot's national criteria for land release, which the rule check enforces.

    Invented, like P6's criteria boxes, since IMAS sets no numbers (Travis's choices, interview questions 19 to 23).
    "No unexplained direct evidence" is read strictly (question 22): evidence of absence never explains direct
    evidence away, since direct evidence leads to technical survey or clearance (IMAS 07.11, §4.4).
    """
    min_years: int = 5        # years of safe use each source must report
    sources: int = 2          # reliable first-hand sources needed
    # Travis's setting (question 21): weaker indirect evidence (class 5 to 7: second-hand reports, rumours, the
    # national database's unchecked record) becomes a target for an intrusive check only if it is still unexplained
    # after this many non-technical survey visits (1: after the first visit).
    follow_up_visits: int = 2
    # Travis's setting (question 23): an area where the national database records a past landmine accident inside
    # is released only after an intrusive check (technical survey, or clearance where the criteria provide no
    # technical survey), never by non-technical survey alone. Its record is then a target after the first visit.
    recorded_accident_needs_ts: bool = True
    # Travis's setting (question 38): the main policy follows Colombia, where technical survey "is not foreseen" for
    # the baseline areas (Mine Action Review, 2025), so the hazardous areas that non-technical survey defines go to
    # clearance. True: the comparison, where technical survey is available and is the default intrusive check.
    technical_survey: bool = False

    def as_text(self):
        visits = f"{self.follow_up_visits} visit{'s' if self.follow_up_visits != 1 else ''}"
        check = "technical survey" if self.technical_survey else "clearance"
        text = (f"Cancellation by non-technical survey is allowed when at least {self.sources} different reliable "
                f"first-hand sources (reliability class 1 to 4) each report at least {self.min_years} years of safe "
                "use of the area, there is no direct evidence in the area and no indirect evidence that the survey "
                "team saw or a first-hand source reported (reliability class 1 to 4), and no weaker indirect "
                "evidence is left unexplained. ")
        if self.technical_survey:
            text += ("Technical survey is available, and is the default next step for evidence of explosive "
                     "ordnance. Clearance without technical survey is allowed where direct evidence of reliability "
                     "class 1 to 4 defines the hazard. ")
        else:
            text += ("Technical survey is not available in this programme: a hazardous area that non-technical "
                     "survey defines, on direct evidence or on indirect evidence that is a target, goes to "
                     "clearance. ")
        text += ("Weaker indirect evidence (reliability class 5 to 7) is followed up by non-technical survey first: "
                 f"it becomes a target for {check} if it is still unexplained after {visits}.")
        if self.recorded_accident_needs_ts:
            text += (" An area where the national database records a past landmine accident inside is released "
                     f"only after {check}, never by non-technical survey alone; its record is a target for "
                     f"{check} after the first visit.")
        return text


@dataclass(frozen=True)
class Season:
    """The season that the twin plans and runs.

    Travis's settings (project interview, questions 28 to 40), except where marked as this project's setting.
    """
    weeks: int = 52                    # a 12-month season, planned monthly and run weekly (question 28)
    days_per_week: int = 5
    # This project's setting: months of 5, 4 and 4 weeks in each quarter, so that 12 months make 52 weeks
    month_weeks: tuple = (5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4)
    zone_areas: int = 25               # about 25 areas per zone, grouped by access within each municipality (q. 29)
    # This project's setting: how strongly access groups the zones, as the weight of log(1 + km to the nearest
    # road) beside the areas' position in km when the zones are formed
    zone_access_weight: float = 6.0
    survey_teams: int = 4              # question 31
    clearance_teams: int = 4           # question 31; in the comparison, 2 of the 4 do technical survey (q. 30)
    ts_teams: int = 0
    zone_survey_weeks: float = 2       # a survey team surveys a zone with its communities (question 39)
    visit_weeks: float = 1             # then visits each area the zone survey flags (question 32)
    clearance_weeks: float = 8         # per hazardous area (question 33)
    ts_weeks: float = 2                # per area, in the comparison (question 38)
    # The zone survey flags every area with a recorded accident, and the areas people point to: 85% of those that
    # hold mines and 5% of those that do not (questions 39 and 40)
    flag_mined: float = 0.85
    flag_clean: float = 0.05
    # No work on days of heavy rain or high wind: about half the working days lost in the two rainy seasons IDEAM
    # describes for the Andean region (March to April, October to November), one in ten in other months (q. 34, 37)
    rainy_months: tuple = (3, 4, 10, 11)
    rainy_day_loss: float = 0.5
    other_day_loss: float = 0.1
    # Each zone has a 1-in-50 chance a month of closing for one to three months (question 36)
    closure_chance: float = 1 / 50
    closure_months: tuple = (1, 3)
    # This project's settings, for Travis to review:
    duration_spread: float = 0.3       # spread (lognormal sigma) of clearance and technical survey times around the mean
    consultation_share: float = 0.1    # share of zones where communities must be consulted before any tasking
    consultation_weeks: int = 4        # the consultation's lead time
    stale_after_weeks: int = 1         # a team with no report for longer than this is shown as stale
    zone_free_value: float = 0.01      # the planner's tie-break: a zone declared free counts as this many mine areas

    def __post_init__(self):
        if sum(self.month_weeks) != self.weeks or len(self.month_weeks) != 12:
            raise ValueError("month_weeks must give 12 months that add up to the season's weeks")

    def day_loss(self, month):
        """The chance that a working day of this month is lost to heavy rain or high wind."""
        return self.rainy_day_loss if month in self.rainy_months else self.other_day_loss

    def task_days(self, kind):
        """The planned working days of a task: its weeks in Travis's settings, times the working days in a week."""
        weeks = {"ZONE_SURVEY": self.zone_survey_weeks, "VISIT": self.visit_weeks, "TS": self.ts_weeks,
                 "CLEARANCE": self.clearance_weeks}[kind]
        return weeks * self.days_per_week

    def with_technical_survey(self):
        """The comparison of question 38: 2 of the 4 clearance teams do technical survey."""
        from dataclasses import replace
        return replace(self, clearance_teams=self.clearance_teams - 2, ts_teams=2)
