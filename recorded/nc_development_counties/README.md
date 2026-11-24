# North Carolina, 20 development counties

The same files for the 20 development counties.

Statistics: kP (kappa_P) person-anchor exact share; mr / match_rate address-anchor match rate; kA address-anchor exact among matched; sigma stayer share; kS exact among stayers; mu exact among movers.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `currency_by_build_and_age.csv` | 20 | As in nc_confirmation_counties, for the development counties. | Supplement D | `pipeline.py` |
| `hypothesis_statistics.tsv` | 242 | As in nc_confirmation_counties. | Table S3 | `reproduce.py analyze phase` |
| `match_validity_by_policy.csv` | 48 | As in nc_confirmation_counties. | Table 4 | `currency.py` |
| `match_validity_statistics.tsv` | 674 | As in nc_confirmation_counties. | Supplement G.3 | `match_validity.py` |
| `matcher_same_population.csv` | 4 | As in nc_confirmation_counties. | Section 7.1 | `currency.link_same_population` |
| `reference_sensitivity.csv` | 4 | As in nc_confirmation_counties. | Supplement E.1 | `currency.py` |
| `right_wrong_unknown_bounds.csv` | 8 | As in nc_confirmation_counties. | Supplement F | `currency.py` |
