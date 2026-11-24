# Ohio and Washington

The other registers (section 9): currency on the Ohio and Washington lists, the North Carolina rows they are compared with, the listed share in each register, and the transfer of the household-change model.

Statistics: kP (kappa_P) person-anchor exact share; mr / match_rate address-anchor match rate; kA address-anchor exact among matched; sigma stayer share; kS exact among stayers; mu exact among movers.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `evaluation_address_crosswalk.csv` | 27 | As above, linking by address crosswalk instead of voter identifier. | Section 9.1 | `other_states.evaluate` |
| `evaluation_voter_id.csv` | 63 | One row per (state, t, s, quantity): kP, sigma, kS, mu and table scores linked by voter identifier, with 99% interval, criterion and whether it holds. | Tables S20, S21; section 9.1 | `other_states.evaluate` |
| `listed_share_by_register.csv` | 26 | One row per (state, t, s): build-date clients, those listed at s, the listed share, kP and the listed-and-exact share, for North Carolina, Ohio and Washington. | Section 9.1; Table S20 | `other_states.listed_share` |
| `model_transfer_listing_fit.json` | JSON | The listing-model fits used to transfer the household-change model to each Ohio and Washington pair. | Section 9.2 | `household_change/transfer.py` |
| `model_transfer_summary.csv` | 185 | One row per (pair, kind, stat): observed and transferred-model levels, Brier scores and differences, with 99% intervals. | Tables S23, S24; section 9.2; supplement M | `household_change/transfer.py` |
| `nc_comparison_pair_sums.csv` | 44 | North Carolina statewide sums per (t, s, alone or with others): clients, listed, exact, stayers and movers, for comparison with the other registers. | Table S20; section 9.2 | `other_states.nc_pair_sums` |
| `nc_comparison_table_scores.csv` | 88 | Per North Carolina (t, s, predictor): shift, clients, mean outcome, mean prediction, raw and recalibrated Brier. | Section 9.2 | `other_states.nc_tables_summary` |
