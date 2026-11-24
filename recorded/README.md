# Recorded results

Every result file a reproduction writes is listed with its SHA-256 in `OUTPUTS.sha256`. The files in the folders
below are shipped: aggregates that back the paper and supplement, under the same names the run writes. Every other
listed file stays in the work directory and is checked by hash only: county-level sums, cell counts with fewer
than 10 persons, intermediate tables, the unmerged fitted tables, and aggregates the paper does not print.

Common terms: a client is a trackable registrant aged 18 or over at the build date t (Election Day of an even year). k is
the evidence age in years and s = t + k the scoring date. Exact means the returned co-resident set equals the true
set at s. A share is a fraction between 0 and 1. Intervals are 99% county-cluster bootstrap percentiles unless a
column says 95%. Column definitions for the exit, reference-sensitivity and anchor diagnostic files are in
`../OUTPUTS.md`.

Rerun only: supplement D.1 prints one development-county value (0.105) from `results/development/mover_timing.csv`,
which holds cells under 10 persons and is checked by hash, not shipped. Likewise the held-out county pack
`robustness/stale_listing_county_pack_held_out.csv` is checked by hash only.

| Folder | Files | What it holds |
|---|---|---|
| [`nc_confirmation_counties/`](nc_confirmation_counties/README.md) | 16 | North Carolina, 60 confirmation counties. The main results. |
| [`nc_development_counties/`](nc_development_counties/README.md) | 7 | North Carolina, 20 development counties. The same files for the 20 development counties. |
| [`nc_november_2026_file/`](nc_november_2026_file/README.md) | 7 | North Carolina, November 2026 file. Scores against the 2026-11-03 snapshot (the later file). |
| [`nc_statewide/`](nc_statewide/README.md) | 5 | North Carolina, whole state. Statewide mobility against the American Community Survey, the unit-key audit and the county sets. |
| [`robustness_checks/`](robustness_checks/README.md) | 14 | Robustness and sensitivity checks. Stale-listing corrections, alternative county sets, the probabilistic matcher alone, household size and building type splits, reweighting and the unrecorded-move scenario. |
| [`fitted_tables/`](fitted_tables/README.md) | 11 | Fitted currency tables. Release copies of the four lookup tables (section 8.1) with no cell under 10 persons, the pooled share by evidence age, and the tables' scores on the held-out counties and the later file. |
| [`household_change_model/`](household_change_model/README.md) | 7 | Household-change model. The fitted model (section 8.2), its held-out test on the 20 held-out counties, and the stayer-definition checks. |
| [`ohio_washington/`](ohio_washington/README.md) | 7 | Ohio and Washington. The other registers (section 9): currency on the Ohio and Washington lists, the North Carolina rows they are compared with, the listed share in each register, and the transfer of the household-change model. |
| [`refresh_decision/`](refresh_decision/README.md) | 3 | Refresh decision. The refresh-decision test (section 8.3) and the currency curve over the first two years after a build. |
| [`paper_figures/`](paper_figures/README.md) | 2 | Figures 1 and 2. Every plotted value of the paper's two figures, copied from the run's results. |
| [`build_error_experiment/`](build_error_experiment/README.md) | 2 | Build-error experiment. Currency when the build itself is wrong (section 6.4): four build arms scored on the 2018 build. |

81 files shipped; `OUTPUTS.sha256` lists every output. Each folder's README gives one row per file.
