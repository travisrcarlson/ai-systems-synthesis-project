"""The data layers registered with the twin, and how the twin is built from them.

Each layer is registered with its source, licence and sensitivity, so that every overlay can say what it read and
every output can be checked against what it may show:
- open: public data, shown in public outputs by municipality (the hazard layer without coordinates; victims as
  counts only);
- restricted: kept inside the twin, never in public outputs (access and security, consultation, teams);
- simulated: a stated pattern, not data (the rainfall season).
The four simulated layers stand in for data an operator or the national authority would hold; their values come
from the season's settings and the simulated field. A new layer plugs in by adding an entry here and registering
it with the twin, without changing the core.
"""
from dataclasses import dataclass

from . import data
from .core import Twin


@dataclass(frozen=True)
class Layer:
    name: str
    title: str
    source: str
    licence: str
    sensitivity: str        # "open", "restricted" or "simulated"
    handling: str           # how the twin shows it
    unit: str               # what one record describes
    used_for: str
    as_of: str              # when the source was published or last updated, or how often it is fed
    stale_after_weeks: int | None = None   # for a live feed: a record older than this is stale


LAYERS = {layer.name: layer for layer in [
    Layer("hazard", "Hazard history: RELand's cell features, including the distance to past landmine accidents",
          "Dulce Rubio et al. (2024), RELand's file on Hugging Face (mdulcer10/landmine-detection), as P3 used it",
          "CC BY 4.0", "open", "shown without coordinates; maps have no axes or scale", "cell (500 m × 500 m)",
          "risk prioritisation; the desk study", "published with the RELand study (2024)"),
    Layer("land_use", "Land use and population", "RELand's cell features (Dulce Rubio et al., 2024)", "CC BY 4.0",
          "open", "shown by area and zone, without coordinates", "cell (500 m × 500 m)",
          "priority; access; what a survey finds", "published with the RELand study (2024)"),
    Layer("victims", "Victims of anti-personnel mines by municipality and month",
          "Presidencia de la República de Colombia, datos.gov.co yhxn-eqqw, as P1 cleaned it", "CC BY-SA 4.0",
          "open", "counts only; no names, sites or coordinates", "municipality and month",
          "impact, shown beside the plan", "the portal's export of 2026-10-18"),
    Layer("demining", "Humanitarian demining results by municipality and month",
          "Presidencia de la República de Colombia, datos.gov.co jbqm-s5tn, as P2 summed it", "CC BY-SA 4.0", "open",
          "counts and areas by municipality", "municipality and month",
          "a check of the simulated capacity against the pilot's past output",
          "rows last updated by the publisher on 2024-01-19"),
    Layer("access", "Access and security status by zone", "simulated (a security coordination feed)", "—",
          "restricted", "inside the twin only", "zone", "whether a zone can be tasked; pauses", "weekly feed",
          stale_after_weeks=2),
    Layer("rainfall", "Rainfall season: working days lost to heavy rain or high wind",
          "simulated from IDEAM's description of two rainy seasons in the Andean region (March to April, October "
          "to November), at Travis's rates", "—", "simulated", "by month", "month",
          "working days per month", "the season's pattern"),
    Layer("consultation", "Community consultation required before tasking", "simulated", "—", "restricted",
          "inside the twin only", "zone", "lead times before tasking", "set for the season"),
    Layer("teams", "Teams: type, capacity, productivity", "simulated (the operators' team reports)", "—",
          "restricted", "inside the twin only; teams by code, no names", "team",
          "the plan and the weekly loop", "team reports as tasks finish, and weekly", stale_after_weeks=1),
]}


def build_twin(criteria=None, season=None):
    """Check the data files, load the open layers, and build the twin with every layer registered.

    Returns (twin, truth, checks): truth is RELand's labels, which only the simulated field and the evaluation read;
    checks is the table of checksums.
    """
    checks = data.verify()
    cells, truth = data.load_cells()
    people = data.victims_by_month(data.load_victims())
    activity = data.load_demining()
    twin = Twin(cells, people, activity, criteria=criteria, season=season, layers=LAYERS)
    return twin, truth, checks


def layer_table():
    """The registry as a table, for the notebook."""
    import pandas as pd
    return pd.DataFrame([{"layer": l.title, "source": l.source, "licence": l.licence, "sensitivity": l.sensitivity,
                          "handling": l.handling, "used for": l.used_for, "as of": l.as_of}
                         for l in LAYERS.values()])
