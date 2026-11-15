# Data provenance and scope

All health inputs provided here are aggregate counts; no individual identifiable records are included. `paper/tables/Table_S7_data_dictionary.csv` gives event units, codes, dates, residence fields, denominators, sources, and exposure limitations. S8 gives the daily analytic samples. The scientific results correspond to manuscript candidate EH-SR-20261114-C1.

- SIM: Brazilian non-fetal mortality by death date and residence, from DATASUS SIM.
- SIH/SUS: AIH cholecystectomy authorisations (procedure 0407030026/0407030034) by administrative competence month and residence UF.
- SAEH-SSA: Mexican cholecystectomy discharge IDs linked to procedure codes 5121–5124; discharge date and usual residence.
- INEGI: Mexican all-cause registered deaths by occurrence month and residence state.
- NASA POWER: daily T2M point exposure at 46.6 W, 23.5 S for Sao Paulo.
- ERA5-Land: supplied monthly 2-m temperature series. Brazilian weight-surface vintage and the Mexican state-day spatial-weight builder were not retained; those are disclosed scientific limitations.
- IBGE municipal population estimates aggregated to Brazilian states; CONAPO 2023 pry23 mid-year state population for Mexico.

The reproducibility starting point is the supplied aggregate daily Parquet and monthly CSV inputs. The repository permits rerunning the statistical analyses from these inputs. It does not claim to reconstruct every raw administrative microdata extraction or the unretained spatial-weighting steps. Original data-system terms continue to apply; the software MIT licence does not relicense third-party source data.

Sources:
- https://datasus.saude.gov.br/transferencia-de-arquivos/
- https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- https://www.inegi.org.mx/programas/edr/
- https://www.gob.mx/conapo
- https://www.ibge.gov.br/
- https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- https://riisp.insp.mx/nada/index.php/catalog/18

The Gasparrini attributable-risk reference implementation is linked, not redistributed, because its pinned source repository contains no software licence. It is not executed by this Python workflow: https://github.com/gasparrini/2015_gasparrini_Lancet_Rcodedata/tree/7c3e03c425529d2f210b1659a01bbc582448b9c0

Historical provenance JSON retains the original retrieval records, including hashes of reference-only files omitted from this release; the current MANIFEST enumerates files actually supplied. Draft configurations are historical snapshots; the active configurations are inputs/fixed_config.json, inputs/fixed_heatday_config.json, inputs/real_config.json and city_case/inputs/real_config.json.
