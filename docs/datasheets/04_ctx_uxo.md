# Datasheet: CTX-UXO crops

Sections follow Gebru et al. (2021), *Datasheets for datasets*, shortened.

**Files:** `data/p4_ctx_uxo/ctx_uxo_crops_64.zip` and `crops_manifest.csv` (made in P4).

## Motivation
*CTX-UXO* (Craioveanu & Stamatescu, 2025; Zenodo record 17052675) was published to support the detection and
identification of unexploded ordnance in photos of real scenes.

## Composition
- 3,520 photos, mostly from phones, split by the creators into training (2,461), validation (530) and test (529), with
  15,449 labelled objects in 12 categories. The record says some scenes use replicas.
- P4's crops: one 64 × 64 crop per labelled object, a square around its box with a margin, from the photo turned upright.
  Five categories hold 98% of them: projectile (6,121), grenade (4,269), mortar bomb (3,399), cartridge (987) and aviation
  bomb (333); the other seven categories hold 340.
- There is no category for battlefield debris, so a model trained on it can say which of its categories an object looks
  most like, but not whether the object is ordnance at all.

## Collection
Photographed by the dataset's creators; see the Zenodo record.

## Preprocessing in this project
None beyond P4's crops. The creators' splits are kept, so all crops from one photo stay in one split.

## Uses
- In this project: training and testing the photo screener on the five categories; the test crops stand in for field
  photos in the replay; the seven other categories test what the screener does with objects it never learned.
- Not from Colombia: the stand-in photos show the screener's behaviour, not field performance in Antioquia.

## Distribution and maintenance
CC BY 4.0 (adapted: cut out and resized by P4). SHA-256 of the zip:
`9f3cbc88609089864fba9b418c39ec8eb9a97a2451ffdddb2116b1a1695c87c1`; of the manifest:
`4a4322ff01b9390b14e5a7d416676cb1d6e457cf725e5e2ac10aba04f7e8c306`.
