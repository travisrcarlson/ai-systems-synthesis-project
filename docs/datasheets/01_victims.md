# Datasheet: victims of anti-personnel mines in Colombia

Sections follow Gebru et al. (2021), *Datasheets for datasets*, shortened.

**File:** `data/p1_victims/victimas_minas_antipersonal.csv` (first used in P1).

## Motivation
Published by the Presidencia de la República de Colombia on the national open data portal (dataset `yhxn-eqqw`) as the
national record of people killed or injured by anti-personnel mines and explosive remnants. It supports the national
mine action authority's planning, victim assistance and reporting.

## Composition
- 12,390 rows × 18 columns, one row per victim, 1990 to 2023.
- Columns: department and municipality with their official DANE codes; area type; site name; year and month; age group;
  ethnic group; civilian or security-force status; sex; killed or injured; the coordinates of the municipality's main
  town; place type; activity at the time.
- No victim or event identifier. P1 found 3,858 rows (31%) that exactly repeat another row; since several victims of one
  accident in the same municipality and month look identical, repeats are kept as separate victims.
- Two column labels contradict their values: `estado` holds sex and `genero` holds killed or injured; `tipoevento` holds
  the place type. P1 documented this, and this project applies the same correction.
- Sensitive: each row describes a person (age group, ethnic group, sex, outcome). It names no one.

## Collection
Published by the national authority; how the records are compiled is not described in this project's sources.
Inference: casualties in remote or insecure areas are under-recorded, as P1's README notes.

## Preprocessing in this project
The site names and municipal-seat coordinates are dropped on loading. Codes are zero-padded and used as keys. Victims
enter the twin only as counts by municipality and month (all victims, killed, injured, civilians, security forces).

## Uses
- In this project: the people layer of the twin and the strategic view, where harm and demining effort are compared.
- Not suitable for: identifying or contacting victims, or locating accidents.

## Distribution and maintenance
CC BY-SA 4.0. Snapshot downloaded on 2026-10-18; SHA-256 `6ac13df53d0024bc9d96f8b48d7622934b5bd1a334d40cf4155bf4cea804a84d`.
The publisher updates the portal table; this project uses the fixed snapshot.
