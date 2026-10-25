# Datasheet: humanitarian demining results in Colombia

Sections follow Gebru et al. (2021), *Datasheets for datasets*, shortened.

**File:** `data/p2_demining/desminado_humanitario.csv` (first used in P2).

## Motivation
Published by the Departamento Administrativo de la Presidencia de la República (DAPRE) on the national open data portal
(dataset `jbqm-s5tn`) to report the results of humanitarian demining operations.

## Composition
- 5,721 rows × 18 columns, 2004 to 2023; rows last updated by the publisher on 2024-01-19.
- Each row reports results or hazardous areas for one municipality, month, demining type and operator: land cleared
  (square metres); items found, by type and in total; the numbers of non-technical surveys, technical surveys and
  clearance operations; hazardous areas recorded and confirmed.
- P2 found that 1,183 combinations of municipality, month, demining type and operator appear twice, each pair splitting
  results from hazardous areas, never repeating a measure, so rows are summed, not de-duplicated. Records thin out at the
  end of 2023, probably because the last months were not yet fully reported.
- No personal data.

## Collection
Published by the national authority from operators' reports; how they are compiled is not described in this project's sources. The rows record outputs
(what was done and found), not effort or risk reduced.

## Preprocessing in this project
English column names, zero-padded codes, numbers as numbers, and rows summed by municipality and month.

## Uses
- In this project: the twin's demining layer (open), a check of the simulated teams' capacity against the pilot's past clearance output by year.
- Not suitable for: measuring how much risk an operation removed, or comparing operators' quality.

## Distribution and maintenance
CC BY-SA 4.0. Snapshot downloaded on 2026-10-18; SHA-256 `eda568a71903f05d7468c582b3795dbd5a537a71c5413f84f72f0acbcffb6617`.
