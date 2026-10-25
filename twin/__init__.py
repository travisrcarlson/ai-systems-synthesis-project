"""A digital twin of land release in RELand's 15 Antioquia municipalities, which helps people plan a season and run
it week by week, with overlays (data layers and decision tools) that read it and recommend.

The modules, in the order the notebook uses them:
- config: file paths, checksums, the pilot municipalities, and every setting (survey, national criteria, season)
- data: loading and checking the three public datasets behind the open layers (P1's, P2's and P3's)
- layers: the data-layer registry (source, licence, sensitivity) and how the twin is built
- core: the twin itself: the state model, the calendar, the one log, the governed write-back, the views
- overlays: the overlay contract (a recommendation with evidence, uncertainty and limits, or "insufficient evidence")
- rules: the coded rule check and the reference reviewer
- field: the simulated field that sends the teams' reports
- agent and review: the dossier reviewer agent, and the reviewers of the season
"""
