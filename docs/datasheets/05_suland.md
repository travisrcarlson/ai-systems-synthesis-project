# Datasheet: SULAND surrogate-mine crops

Sections follow Gebru et al. (2021), *Datasheets for datasets*, shortened.

**Files:** `data/p5_suland/mine_crops_64.zip` and `crops_manifest.csv` (made in P5).

## Motivation
*SULAND* (Vivoli et al., 2024) was published for research on detecting surface landmines in optical images; the
SULAND_v2 labels (Lekhak et al., 2026, a preprint) refine its boxes. The record states that the dataset "is intended for
academic research".

## Composition
- Video frames of surrogates of two mine types, the PFM-1 and the PMA-2, filmed with a phone at a site in Italy
  (training, validation and test splits made by video) and at a second site in the United States.
- P5's crops: 10,397 crops of 64 × 64 around the boxes (Italy training 5,890; Italy test 933; United States 3,574), with
  boxes that touch the frame's edge or measure under 32 pixels dropped. The manifest also lists P5's background crops,
  which are not shipped here.

## Collection
Filmed by the dataset's creators; see Vivoli et al. (2024).

## Preprocessing in this project
None beyond P5's crops.

## Uses
- In this project: training the generator, whose images test whether the photo screener answers "unknown" for types it
  never learned; the real crops (Italy test and United States) give the same test on real photos.
- Synthetic images never train the screener or enter a decision. Surrogates at two sites only, so neither the real nor
  the generated images show what mines look like in Colombia.

## Distribution and maintenance
CC BY 4.0 (adapted: cut out and resized by P5). SHA-256 of the zip:
`2ddc12cd38dac2b2d621341171b3163d6e7b74d018854b09edabc7cceaa49adb`; of the manifest:
`cac0c5593c0b37df522bc7ac3762bf1801d0edf11f5148760c37dbecfe69c286`.
