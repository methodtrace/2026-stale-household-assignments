# North Carolina, November 2026 file

Scores against the 2026-11-03 snapshot (the later file). Builds 2016 to 2024 at the evidence age that reaches November 2026.

Statistics: kP (kappa_P) person-anchor exact share; mr / match_rate address-anchor match rate; kA address-anchor exact among matched; sigma stayer share; kS exact among stayers; mu exact among movers.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `currency_by_build_confirmation.csv` | 14 | One row per (build, k): clients with truth on the November 2026 file, kP, match rate, kA, sigma, kS, mu, kappa_MA (60 confirmation counties). | Table S2 | `later_file.py` |
| `currency_by_build_development.csv` | 14 | As above, 20 development counties. | Section 8.1; Supplement H.3 | `later_file.py` |
| `exact_share_by_policy_confirmation.csv` | 24 | One row per (build, k, policy): exact share on the November 2026 file and its difference from kP, with 99% intervals. | Table S14; section 7.3 | `later_file.py` |
| `exact_share_by_policy_development.csv` | 24 | As above, development counties. | Supplement H.3 | `later_file.py` |
| `gap_statistics_2018_build_confirmation.csv` | 22 | The gap and widening statistics re-scored on the 2018 build: point, 99% interval, jackknife SE, threshold, decision. | Section 5.7; Supplement B.7 | `later_file.py` |
| `ten_year_statistics_2016_confirmation.csv` | 60 | The 2016 build at ten years (2026): G(k), widening and currency statistics with 99% intervals, confirmation counties. | Section 5.1; supplement B.4; Figure 1 (via paper_figures/) | `later_file.py` |
| `ten_year_statistics_2016_development.csv` | 60 | As above, development counties. | Supplement K.5 | `later_file.py` |
