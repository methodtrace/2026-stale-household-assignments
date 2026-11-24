# Household-change model

The fitted model (section 8.2), its held-out test on the 20 held-out counties, and the stayer-definition checks.

Statistics: kP (kappa_P) person-anchor exact share; mr / match_rate address-anchor match rate; kA address-anchor exact among matched; sigma stayer share; kS exact among stayers; mu exact among movers.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `fit_step_check.csv` | 2 | The largest total exit rate of any latent state and the step size times it (below 1: every step matrix is nonnegative). | Supplement K.3 | `household_change/held_out.step_check` |
| `held_out_by_build_and_age.csv` | 24 | One row per (build, k), 20 held-out counties: clients, observed and predicted kP, sigma, kS and mu. | Table S19 | `household_change/held_out.py` |
| `held_out_jackknife_vs_tables.csv` | 2 | The model's Brier score minus each table's on the held-out counties: difference, 99% interval, counties where the model is lower, sign test. | Section 8.2 | `household_change/held_out.py` |
| `held_out_summary.csv` | 16 | Held-out Brier scores of the model and the comparators, and their differences, with 99% and 95% intervals. | Table 7; section 8.2 | `household_change/held_out.py` |
| `model_fit.json` | JSON | The fitted parameters, negative log-likelihood per pair, pair count, builds and convergence (largest absolute gradient). | Section 8.2; supplement K.1 | `household_change fit (reproduce.py analyze phase)` |
| `stayer_crosstab_2016.csv` | 20 | 2016 build, held-out counties: clients and exact by the three stayer flags (unit key, readdressing, no re-match key) at each k. | Supplement K.5; Supplement M | `reproduce.py analyze phase` |
| `stayer_definitions.csv` | 29 | One row per (t, k): sigma, kS and mu under address re-match, the readdressing rule and the model, and the shifts between them. | Supplement K.5 | `household_change/held_out.stayer_definitions` |
