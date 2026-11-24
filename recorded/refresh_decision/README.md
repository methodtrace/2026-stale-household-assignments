# Refresh decision

The refresh-decision test (section 8.3) and the currency curve over the first two years after a build.

Statistics: kP (kappa_P) person-anchor exact share; mr / match_rate address-anchor match rate; kA address-anchor exact among matched; sigma stayer share; kS exact among stayers; mu exact among movers.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `first_two_years_curve.csv` | 76 | One row per (build, scoring date, stat): kP, sigma, kS and mu at up to four dates in the first two years, with 99% intervals. | Section 5.5; Table S18 | `refresh/statistics.curve_stats` |
| `predictor_diagnostics_posthoc.csv` | 7 | Post hoc, no decision: correlations of predicted with realized gain and of prediction with outcome, top ventiles of predicted gain, per predictor. | Supplement J.1 | `refresh/rows.diag_posthoc` |
| `refresh_results.csv` | 5617 | One row per (section, decision date, file, cost, county set, rule, budget, stat): exact share and gains under each refresh rule, with 99% intervals. | Table 8; Table S17; section 8.3 | `refresh/statistics.run` |
