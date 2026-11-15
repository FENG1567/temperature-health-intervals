# Temperature-related health contrast intervals

Reproducible analysis code and aggregate data for **Interval Coverage and Sensitivity Bounds for Temperature–Health Contrasts With Applications to Cholecystectomy Records and All-Cause Mortality** (manuscript candidate EH-SR-20261114-C1).

The study separates estimation-center error, sampling-scale error and nonlinear coefficient propagation for a fixed-path expected-count contrast. Its response-sensitivity calculations are conditional on finite exposure support and a declared perturbation class. Real-data panels are model-conditional computational illustrations.

## Quick start

Python 3.12 is the tested runtime. From the repository root:

```sh
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py all
```

The last command checks repository checksums, rebuilds paper tables, regenerates figures and refits selected seeded outcomes. It uses the supplied formal results for output reconstruction; it does **not** launch the entire Monte Carlo experiment. See [reproduction levels](docs/REPRODUCTION_LEVELS.md).

```sh
python reproduce.py verify
python reproduce.py tables
python reproduce.py figures
python reproduce.py smoke
```

Recreated tables are under `outputs/tables/`. Plots and analysis reruns are under `outputs/run/`; archived inputs and published targets are kept separate. Set `CA_RUN_ROOT` to change the rerun destination. Choose a directory outside the repository (never its ancestor), or under outputs/.

## Full statistical reruns

```sh
python reproduce.py analysis --stage fixed --formal --workers 2
python reproduce.py analysis --stage heatday --formal --workers 2
python reproduce.py analysis --stage daily-state --formal --workers 2
python reproduce.py analysis --stage daily-city --formal --workers 2
python reproduce.py analysis --stage quasi-state --formal --workers 2
python reproduce.py analysis --stage quasi-city --formal --workers 2
python reproduce.py analysis --stage mechanism-state
python reproduce.py analysis --stage mechanism-city
python reproduce.py analysis --stage monthly
python reproduce.py analysis --stage sensitivity
python reproduce.py analysis --stage mvn --formal --workers 2
```

Later diagnostics replay the archived approved upstream outcomes where that is part of their comparison design. They are not additional independent test samples. Formal fixed families use 300 training, 300 calibration and 1000 test draws per scenario/design; daily tests contribute another 6000 outcomes. Total independent **test** outcomes are 24,000. See [analysis dependencies](docs/REPRODUCTION_LEVELS.md). Full reruns can take much longer than output reconstruction.

## Repository contents

| Directory | Purpose |
|---|---|
| `code/` | Fixed-temperature and heat-day experiments, daily inference, mechanism diagnostics, monthly bounds, plotting |
| `inputs/` | Approved configurations, pinned generating code, daily aggregate Parquet and four monthly aggregate CSV panels |
| `city_case/` | Municipality-specific daily configuration, aggregate input, engine and archived outcomes |
| `results/` | Archived base formal replication-level records, summaries and fixed designs |
| `additional_analyses/` | Matched coefficient-MVN and alternative lag-allocation analyses, input snapshots and archived outputs |
| `paper/tables/` | Current supplementary CSV targets S1–S14, with lettered subdivisions |
| `paper/figures/` | Six main and two supplementary PDF/TIFF reference figures, numbered as in C1 |
| `.data_chunks/` | Exact optional covariance-factor caches, split into <=16 MiB parts |
| `scripts/`, `tests/` | Output reconstruction, checksum verification, seeded execution checks and cache restoration |
| `environment/` | Original runtime record and independently installed/tested Linux versions |
| `outputs/` | Recreated results (ignored by Git) |

## Large files

The two optional coefficient-factor NPZ files exceed ordinary GitHub's 100 MiB file limit. Their exact bytes are stored as checked 16 MiB parts. This keeps every tracked file below the browser's 25 MiB limit without requiring Git LFS.

```sh
python reproduce.py restore-cache
# Or check the parts without reconstructing large NPZ files:
python scripts/restore_chunks.py --verify-only
```

Restored NPZ files are ignored by Git. Keep `.data_chunks/` and `.github/` when uploading or copying the repository. Do not upload the old `Code_and_Data.zip`.

## Data and attribution

The supplied health data are aggregate counts, not individual identifiable records. The reproducible starting point is those aggregate inputs, not a claim that every raw administrative extraction and unretained exposure-weighting step is reconstructable. Definitions and limits: [data provenance](docs/DATA_PROVENANCE.md), supplementary S7/S8. Public-source data retain their source terms. Code is MIT-licensed; authorship and software citation are in `CITATION.cff`.

No journal submission documents, internal reviewer replies, editorial assessments, machine environments or unrelated project backups are included. Funding: National Natural Science Foundation of China, grant 82270598. Correspondence: Xianzhi Meng, mengxianzhi@hrbmu.edu.cn.

## Software citation

Cite software version `0.1.1` (tag `v0.1.1`) from the [FENG1567/temperature-health-intervals repository](https://github.com/FENG1567/temperature-health-intervals), using the metadata in [`CITATION.cff`](CITATION.cff). No DOI is assigned to this release.

## Figure typography

Figures use bold text and uppercase 18-point Arial panel labels. Install Arial, or set `CA_FIGURE_FONT_DIR` to a directory containing `arial.ttf` and `arialbd.ttf`, to match the archived typography. Font files are not redistributed. Matplotlib uses an installed sans-serif fallback when Arial is unavailable.
