# What each command reproduces

`python reproduce.py verify` checks hashes and the presence of required scientific inputs. It does not fit a model.

`python reproduce.py tables` rebuilds interval summaries from the supplied formal replication-level records and assembles the four main-text tables. Its machine-readable report distinguishes recomputed numerical summaries from copied deterministic source tables and data dictionaries.

`python reproduce.py figures` regenerates the six main and two supplementary figures from the supplied formal outputs. PDF byte hashes may vary by fonts or plotting-library metadata; the scientific sources and figure numbering are retained.

`python reproduce.py smoke` refits selected seeded outcomes using the original numerical algorithms and checks them against the archived formal results. It is a small execution check, not the 24,000-outcome Monte Carlo experiment.

`python reproduce.py all` runs the preceding checks and paper-output generation. It does not silently launch the full Monte Carlo study.

`python reproduce.py analysis --stage ...` starts a statistical rerun from aggregate input data or fixed generating models. Pilot fixed-family runs use pilot streams and are not formal paper results. Pass `--formal` for the fixed formal experiment counts and formal streams. The quasi and MVN comparisons replay the original 6000 daily outcomes; they do not increase the independent test sample.

Every analysis rerun writes to `outputs/run/` by default (`CA_RUN_ROOT` can select another output root). Archived `results/`, `city_case/results/`, `additional_analyses/results/`, and `paper/` provide the comparison targets and must remain unchanged. A rerun of a later diagnostic may deliberately use archived upstream outputs; the stage documentation identifies this dependency rather than claiming a fully raw-to-paper replacement in one command.

Optional exact covariance-factor caches are stored as 16 MiB chunks under `.data_chunks/`. Restore them with `python reproduce.py restore-cache`. The two reconstructed NPZ files are ignored by Git. They are unnecessary for the default paper-output and smoke commands.
