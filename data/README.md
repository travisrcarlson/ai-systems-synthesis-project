# Data

Every file here is public data, copied unchanged from the earlier capstone project that first used it. Each one's
SHA-256 checksum matches the checksum that project published, and `twin/data.py` checks it before loading. A datasheet
for each dataset is in `docs/datasheets/`.

| Folder and file | Dataset | Earlier project | Licence | SHA-256 |
|---|---|---|---|---|
| `p1_victims/victimas_minas_antipersonal.csv` | *Situación Víctimas Minas Antipersonal en Colombia*, Presidencia de la República de Colombia, [datos.gov.co `yhxn-eqqw`](https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-V-ctimas-Minas-Antipersonal-en-Colombia/yhxn-eqqw); the portal's CSV export, downloaded on 2026-10-18 | P1, [`ai-programming-foundations-project`](https://github.com/travisrcarlson/ai-programming-foundations-project) at `d7f17f7` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | `6ac13df53d0024bc9d96f8b48d7622934b5bd1a334d40cf4155bf4cea804a84d` |
| `p2_demining/desminado_humanitario.csv` | *Situación de Desminado Humanitario en Colombia*, Presidencia de la República de Colombia, [datos.gov.co `jbqm-s5tn`](https://www.datos.gov.co/Inclusi-n-Social-y-Reconciliaci-n/Situaci-n-de-Desminado-Humanitario-en-Colombia/jbqm-s5tn); the portal's CSV export, downloaded on 2026-10-18 (rows last updated by the publisher on 2024-01-19) | P2, [`statistical-analysis-project`](https://github.com/travisrcarlson/statistical-analysis-project) at `6aad156` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | `eda568a71903f05d7468c582b3795dbd5a537a71c5413f84f72f0acbcffb6617` |
| `p3_reland/landmine_detection.csv` | *Landmine Detection in Antioquia, Colombia*, [Hugging Face `mdulcer10/landmine-detection`](https://huggingface.co/datasets/mdulcer10/landmine-detection) at revision `30710e43dc447ee95bce30e9ed264f68e454e00c`, published with the RELand study (Dulce Rubio et al., 2024); the CSV that P3 made from the Parquet file with no change to any value | P3, [`applied-machine-learning-project`](https://github.com/travisrcarlson/applied-machine-learning-project) at `10e9672` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `f2ba3b981487ae3879175db4521bc1c6d02352d1053dc9889715eb5e703fe371` |

The two Colombian tables are shared here under CC BY-SA 4.0, as their publisher licenses them. The RELand file is shared
under CC BY 4.0, with the attribution above.

## What this project does with the locations
RELand's file gives each cell's latitude and longitude, and the victims table gives the coordinates of each
municipality's main town. The twin uses the cells' coordinates only to place the cells on a grid of rows and columns;
it never prints a coordinate, and its maps have no axes or scale. It drops the municipal-seat coordinates and the
free-text site names when it loads the victims table, as P1 did, and keeps victims only as counts by municipality and
month.
