"""A digital twin of land release in RELand's 15 Antioquia municipalities, with decision services above it.

The modules, in the order the notebook uses them:
- config: file paths, checksums, the pilot municipalities, the simulation's assumptions and the national criteria
- data: loading and checking the five datasets of P1 to P5
- core: the twin itself (its layers, areas, views and governed write-back)
- strategic: the national view of victims against demining effort (P1, P2)
- prioritise: the operational ranking of areas, baseline first (P3)
- field: the simulated field teams that return evidence in the replay
- rules: the coded rule check and the reference reviewer
- screener: the photo screener (P4) and its "unknown" cut-off
- generator: the image generator for stress tests (P5)
- agent: the tactical review agent (P6)
- replay: the closed loop, round after round, and its measures
"""
