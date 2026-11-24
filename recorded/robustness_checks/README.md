# Robustness and sensitivity checks

Stale-listing corrections, alternative county sets, the probabilistic matcher alone, household size and building type splits, reweighting and the unrecorded-move scenario.

Statistics: kP (kappa_P) person-anchor exact share; mr / match_rate address-anchor match rate; kA address-anchor exact among matched; sigma stayer share; kS exact among stayers; mu exact among movers.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `age_reweighted_currency.csv` | 4 | One row per k: kP, kP reweighted to the ACS adult age mix, the difference, and the client and ACS shares aged 18 to 34. | Section 4.2; Table S9 | `robustness/outputs.py` |
| `client_universe.csv` | 10 | Registrants of the confirmation counties at the 2016 build by class: trackable clients and each exclusion, with shares of all registrants. | Section 4.2; Supplement E.3 | `robustness/outputs.py` |
| `confirmation_vs_held_out.csv` | 112 | One row per (t, k, stat): the confirmation value, the held-out value with its 99% interval, and the difference. | Section 5.6; Table S6 | `robustness/outputs.py` |
| `currency_all_county_sets.csv` | 2034 | One row per (county set, t, k, stat): every currency statistic with counties, n, 99% and 95% intervals, jackknife SE and leave-one-county-out range. County sets: confirmation, development and the later-file sets. | Section 5.2; Tables 2, S14 | `robustness/outputs.py` |
| `currency_by_building_type.csv` | 40 | One row per (county set, t, k): kP, kS and mu for single-unit and multi-unit buildings, and the single-unit share. | Section 5.3; Table S7; supplement D.1 | `reproduce.py analyze phase` |
| `currency_by_household_size.csv` | 40 | One row per (county set, k, household size at build): clients, exact, stayers, kP, sigma, kS, mu (2016 build). | Section 5.3 | `robustness/outputs.size_split_csv` |
| `forecast_bands.csv` | 28 | Later-file values against bands set from the earlier builds (last value and linear trend, plus or minus 0.05), for kP, kS and sigma. | Supplement B.1; Table S2 | `robustness/outputs.forecast_bands` |
| `matcher_by_weight_threshold.csv` | 24 | One row per (weight threshold, k, group): matched clients, match rate, exact among matched, kS, kP and the matcher's excess over kS. | Section 7.1; Table S12 | `robustness/outputs.py` |
| `matcher_paired_differences.csv` | 20 | Paired differences between the probabilistic matcher and the last-known household (by k, group alone or with others, weight threshold): point, intervals, jackknife SE. | Table S11 | `robustness/outputs.py` |
| `matcher_w20_by_reference.csv` | 48 | The matcher at W >= 20 with unmatched clients scored as recorded alone, by k and truth reference. | Section 7 | `robustness/outputs.w20_alone` |
| `stale_listing_intervals.csv` | 849 | One row per (t, k, policy, reference variant, stat): currency under each stale-listing correction with intervals, jackknife SE and leave-one-county-out range. | Table S8; Supplement E.1 | `robustness/outputs.py` |
| `stale_listing_summary.csv` | 376 | One row per (t, k, policy, reference variant): clients, exact share, sigma, kS, mu (confirmation counties). | Section 6.1; Table 3; Figure 1 (via paper_figures/) | `robustness/outputs.py` |
| `stale_listing_summary_held_out.csv` | 376 | As above, 20 held-out counties. | Not printed (held-out counterpart of stale_listing_summary.csv) | `robustness/outputs.py` |
| `unrecorded_move_scenario.csv` | 15 | The stress scenario: the unrecorded one-year move rate u compounded on stayers, sigma and kP before and after, by measure and k. | Section 6.3; Supplement E.2 | `robustness/outputs.py` |
