# Datasheet: RELand's landmine-risk grid cells, Antioquia

Sections follow Gebru et al. (2021), *Datasheets for datasets*, shortened.

**File:** `data/p3_reland/landmine_detection.csv` (first used in P3).

## Motivation
Released on Hugging Face (`mdulcer10/landmine-detection`) with the RELand study (Dulce Rubio et al., 2024), to support
research on predicting where landmines are, so that survey can be prioritised.

## Composition
- 26,761 grid cells of 500 × 500 m in 15 municipalities of Antioquia, Colombia, × 67 columns: latitude and longitude,
  the municipality, the label and 63 open features (27 continuous, 36 one-hot) on terrain, climate, land use, relief,
  population and distances to roads, settlements, services, waterways, coca fields and past events.
- Label `mines_outcome`: 1 for a cell that intersects "at least one confirmed hazardous area or a cleared area where
  landmines were found" (dataset card); 430 mine cells, 26,331 clear cells.
- The labels come from the records of Colombia's national mine action authority. All 15 municipalities have been
  declared mine-free (dataset card), so their outcomes are known.
- Sensitive: the coordinates locate former mine cells. The data are public under CC BY 4.0, and the areas are declared
  mine-free; this project still prints no coordinate.

## Collection
Assembled by the study's authors from open geographic sources and the national authority's records; see Dulce Rubio et
al. (2024).

## Preprocessing in this project
The coordinates place each cell on a grid of rows and columns and are then set aside; 2 × 2 blocks of cells within one
municipality form the areas that the replay tasks. The label is hidden from every decision service and read only by the
simulated field.

## Uses
- In this project: the base and hazard layers of the twin; the operational ranking; the replay's ground truth.
- The dataset card warns that predictions should not be "the sole basis for demining operations without expert human
  review and validation in the field". Labels mark where surveys and accidents happened, so models also learn reporting
  patterns. One region only.

## Distribution and maintenance
CC BY 4.0. Parquet file downloaded on 2026-10-18 at revision `30710e43dc447ee95bce30e9ed264f68e454e00c`; the CSV was made
from it by P3 with no change to any value. SHA-256 of the CSV:
`f2ba3b981487ae3879175db4521bc1c6d02352d1053dc9889715eb5e703fe371`.
