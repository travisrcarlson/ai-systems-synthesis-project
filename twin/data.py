"""Loading and checking the three public datasets behind the twin's open data layers.

Each loader reads one file, checked against its SHA-256 checksum, and applies the handling that the earlier
project established for it: P1's cleaning of the victims table, P2's monthly totals of demining results, and P3's
grid of RELand cells. The functions are new code; the decisions they carry out are the earlier projects', and
each is named where it is applied (P1's insight: check data as it arrives).
"""
import hashlib
import unicodedata

import pandas as pd

from .config import CODE_OF, DATA_DIR, DATA_FILES, GRID_STEP_DEGREES


def sha256(path):
    """The SHA-256 checksum of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_of(name):
    return DATA_DIR / DATA_FILES[name][0]


def verify(names=None):
    """Check each data file against its published checksum. Returns a table; raises if any file differs."""
    rows = []
    for name in names or DATA_FILES:
        relative, expected = DATA_FILES[name]
        actual = sha256(DATA_DIR / relative)
        rows.append({"dataset": name, "file": relative, "bytes": (DATA_DIR / relative).stat().st_size,
                     "sha256 (first 12)": actual[:12], "matches": actual == expected})
    table = pd.DataFrame(rows).set_index("dataset")
    if not table["matches"].all():
        raise ValueError("A data file differs from its published checksum: "
                         + ", ".join(table.index[~table["matches"]]))
    return table


def normalise_text(value):
    """Lower case, no accents, single spaces: so 'Sin Información' and 'Sin informacion' compare equal."""
    if pd.isna(value):
        return value
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().split())


# ---- P1: victims of anti-personnel mines ----------------------------------------------------------------------
# P1 found that three published labels do not match their values (estado holds sex, genero holds killed or
# injured, tipoevento holds the type of place), that "no information" is written three ways, and that repeated
# rows are real victims, not duplicates. The twin keeps only what its people layer needs.
VICTIM_COLUMNS = {"codigodanedepartamento": "department_code", "departamento": "department",
                  "codigodanemunicipio": "municipality_code", "municipio": "municipality",
                  "ano": "year", "mes": "month", "condicion": "condition", "genero": "outcome",
                  "rangoedad": "age_group"}
VICTIM_CATEGORIES = {
    "condition": {"civil": "Civilian", "fuerza publica": "Security forces"},
    "outcome": {"herido": "Injured", "muerto": "Killed"},
    "age_group": {"mayor de 18 anos": "Adult", "menor de 18 anos": "Minor"},
}
# Dropped on loading, as in P1: the free-text site names and the coordinates of each municipality's main town,
# and the columns the people layer does not use (sex, ethnic group, activity, place type, area type).
VICTIM_DROPPED = ["sitio", "latitudcabecera", "longitudcabecera", "Ubicación", "estado", "grupoetnico",
                  "Actividad", "tipoevento", "tipoarea"]


def load_victims():
    """One row per recorded victim: where (DANE codes), when (year, month), condition, outcome and age group."""
    raw = pd.read_csv(path_of("victims"), dtype=str)
    assert set(VICTIM_COLUMNS) | set(VICTIM_DROPPED) == set(raw.columns), "unexpected columns in the victims table"
    assert set(raw["genero"].map(normalise_text).dropna()) <= {"herido", "muerto"}, "genero no longer holds the outcome"
    df = raw.drop(columns=VICTIM_DROPPED).rename(columns=VICTIM_COLUMNS)
    df["department_code"] = df["department_code"].astype(int).astype(str).str.zfill(2)
    df["municipality_code"] = df["municipality_code"].astype(int).astype(str).str.zfill(5)
    df["year"], df["month"] = df["year"].astype(int), df["month"].astype(int)
    for column, mapping in VICTIM_CATEGORIES.items():
        keys = df[column].map(normalise_text).mask(lambda s: s == "sin informacion")
        unmapped = set(keys.dropna()) - set(mapping)
        assert not unmapped, f"{column}: no translation for {unmapped}"
        df[column] = keys.map(mapping)
    return df


def victims_by_month(victims):
    """The people layer: victims counted by municipality, year, month, condition and outcome."""
    keys = ["department_code", "department", "municipality_code", "municipality", "year", "month",
            "condition", "outcome"]
    return victims.groupby(keys, dropna=False).size().rename("victims").reset_index()


# ---- P2: humanitarian demining results ------------------------------------------------------------------------
# P2 found that rows sharing a municipality, month, demining type and operator join two kinds of record (results
# and hazardous areas), so summing the rows gives correct totals and none should be deleted as duplicates.
DEMINING_COLUMNS = {
    "codigodanedepartamento": "department_code", "departamento": "department",
    "codigodanemunicipio": "municipality_code", "municipio": "municipality", "ano": "year", "mes": "month",
    "areadespejada": "area_cleared_m2", "map": "items_mines", "muse": "items_uxo", "aei": "items_ieds",
    "totalartefactosdestruidos": "items_total", "estudiosnotecnicos": "non_technical_surveys",
    "estudiostecnicos": "technical_surveys", "operacionesdespeje": "clearance_operations",
    "areaspeligrosas": "hazardous_areas", "areaspeligrosasconfirmadas": "confirmed_hazardous_areas",
}
DEMINING_MEASURES = ["area_cleared_m2", "items_mines", "items_uxo", "items_ieds", "items_total",
                     "non_technical_surveys", "technical_surveys", "clearance_operations", "hazardous_areas",
                     "confirmed_hazardous_areas"]


def load_demining():
    """The activity layer: demining results summed by municipality and month, over demining types and operators."""
    raw = pd.read_csv(path_of("demining"), dtype=str)
    df = raw.rename(columns=DEMINING_COLUMNS)
    df["department_code"] = df["department_code"].astype(int).astype(str).str.zfill(2)
    df["municipality_code"] = df["municipality_code"].astype(int).astype(str).str.zfill(5)
    for column in ["year", "month", *DEMINING_MEASURES]:
        df[column] = pd.to_numeric(df[column])
    assert (df[["items_mines", "items_uxo", "items_ieds"]].sum(axis=1) == df["items_total"]).all()
    keys = ["department_code", "municipality_code", "year", "month"]
    names = df.groupby(keys)[["department", "municipality"]].first()
    return pd.concat([names, df.groupby(keys)[DEMINING_MEASURES].sum()], axis=1).reset_index()


# ---- P3: RELand's cells ---------------------------------------------------------------------------------------
LABEL = "mines_outcome"
COORDINATES = ["LATITUD_Y", "LONGITUD_X"]


def load_cells():
    """RELand's 26,761 cells on a grid of rows and columns, and their labels kept apart.

    Returns (cells, truth). `cells` has one row per cell, indexed by cell number, with the municipality's DANE
    code, the cell's row and column, and the 63 features P3 used. The coordinates are read once here, to place
    each cell on the grid, and are not kept. `truth` holds RELand's labels (1 where clearance found mines); the
    twin never receives it; only the simulated field and the evaluation read it.
    """
    raw = pd.read_csv(path_of("reland"))
    rows = (raw["LATITUD_Y"] - raw["LATITUD_Y"].min()) / GRID_STEP_DEGREES
    cols = (raw["LONGITUD_X"] - raw["LONGITUD_X"].min()) / GRID_STEP_DEGREES
    assert max((rows - rows.round()).abs().max(), (cols - cols.round()).abs().max()) < 0.05, "cells are off the grid"
    cells = raw.drop(columns=COORDINATES + [LABEL]).rename(columns={"Municipio": "reland_name"})
    cells.insert(0, "municipality_code", cells["reland_name"].map(CODE_OF))
    cells.insert(1, "row", rows.round().astype(int))
    cells.insert(2, "col", cols.round().astype(int))
    cells = cells.drop(columns="reland_name")
    cells.index.name = "cell"
    assert cells["municipality_code"].notna().all() and not cells.duplicated(["row", "col"]).any()
    truth = raw[LABEL].astype(int).rename("mines")
    truth.index.name = "cell"
    return cells, truth


NOT_FEATURES = ("municipality_code", "row", "col", "area_row", "area_col", "area_id", "zone_id")


def features_of(cells):
    """The 63 RELand features, in the file's order: every column except the place keys the twin adds."""
    features = [c for c in cells.columns if c not in NOT_FEATURES]
    assert len(features) == 63, f"expected 63 features, found {len(features)}"
    return features
