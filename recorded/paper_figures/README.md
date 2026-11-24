# Figures 1 and 2

Every plotted value of the paper's two figures, copied from the run's results. make_figures.py draws the figures from these files (it needs matplotlib; the images are not in OUTPUTS.sha256, since they depend on the installed fonts and renderer).

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `figure1_currency_by_age.csv` | 33 | Figure 1, 2016 build, confirmation counties: one row per (mark, series, k). Lines mr_addr, kA, kP, kS and mu at k = 2 to 10; band_lo and band_hi give the lowest and highest last-known exact share under the five stale-listing corrections at k = 2 to 8. | Figure 1; section 5.1 | `paper_figures.figure_data from nc_confirmation_counties/currency_by_build_and_age.csv, nc_november_2026_file/ten_year_statistics_2016_confirmation.csv, robustness_checks/stale_listing_summary.csv` |
| `figure2_validity_frontier.csv` | 44 | Figure 2, 2016 build, confirmation counties: per k, the probabilistic matcher's match rate and exact share among matched at each weight threshold (sweep), the annual rebuild, and stayer currency. | Figure 2; section 7.1 | `paper_figures.figure_data from nc_confirmation_counties/match_validity_by_policy.csv and currency_by_build_and_age.csv` |
