# North Carolina, whole state

Statewide mobility against the American Community Survey, the unit-key audit and the county sets.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `acs_mover_rates.csv` | 13 | One row per ACS year: North Carolina one-year mover rate, movers and population (published Census aggregates; no 2020 release). | Section 6.3 | `currency.acs_movers from inputs/acs_nc_movers.tsv` |
| `county_sets.csv` | 100 | One row per county (county_id 1 to 100): development, held_out or confirmation. | Section 4.4 | `runs.py county split (stale_assignments/data)` |
| `mobility_vs_acs_by_age_band.csv` | 255 | One row per (year, ACS age band): adult registrants, recorded moves and exits over one year, and the ACS mover rate. Denominator: adults active on January 1. Columns in OUTPUTS.md. | Section 6.3 (by-band detail, not printed) | `calibration.py` |
| `recorded_moves_and_exits_by_year.csv` | 17 | One row per year: one-year rates of recorded moves, deaths, moves out, list maintenance and other exits among adult registrants, and the ACS rate. | Section 6.3 | `robustness/outputs.py` |
| `unit_key_audit.csv` | 24 | One row per (snapshot, county set): buildings and registrants, the share whose unit key agrees with North Carolina address points, and the disagreement, pooling and mixed-building shares. | Supplement E.3 | `unit_key.py` |
