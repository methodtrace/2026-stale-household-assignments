# Fitted currency tables

Release copies of the four lookup tables (section 8.1) with no cell under 10 persons, the pooled share by evidence age, and the tables' scores on the held-out counties and the later file. The paper scores the unmerged tables; they are checked by hash under tables/ in OUTPUTS.sha256.

| File | Rows | What it holds | Paper | Produced by |
|---|---|---|---|---|
| `coarse_table_by_build.csv` | 1399 | Coarse table, release copy: one row per (k, build, size6, bt, ab) with persons n, exact ne and p = (ne + pk) / (n + 1). Merged age bands read e.g. a65+a75. Every n >= 10. | Section 8.1 | `release_files.release_tables from tables/coarse_table.csv` |
| `coarse_table_by_parity.csv` | 559 | Coarse table with build parity (pres, mid) in place of build, release copy: sums of the by-build rows. | Section 8.1; later-file test | `release_files.release_tables from tables/coarse_table_parity.csv` |
| `comember_ages_table_by_build.csv` | 18160 | Table with co-member ages, release copy: one row per (k, build, size6, bt, ab, ny, mb, gb). Rows with ny, mb, gb = other pool a coarse cell's small cells. Every n >= 10. | Section 8.1 | `release_files.release_tables from tables/comember_age_table.csv` |
| `comember_ages_table_by_parity.csv` | 6947 | Table with co-member ages by build parity, release copy: sums of the by-build rows; a key is listed only where no build pools it. | Section 8.1; refresh predictors | `release_files.release_tables from tables/comember_age_table_parity.csv` |
| `held_out_cell_errors.csv` | 86 | One row per (table, reference, build, k): predicted and observed exact share on the held-out counties and their difference. | Table S15 | `table_scores.py` |
| `held_out_scores.csv` | 35 | One row per (table, reference, metric): Brier, log-loss, ECE and cell error on the held-out counties with bootstrap interval, the development value and the criterion. | Table S13 | `table_scores.py` |
| `pooled_share_by_age.csv` | 4 | The pooled exact share pk at each evidence age, the fallback for a key absent from a table and the age-only baseline. The same for all four tables. | Section 8.1 | `table_scores.fit_tables` |
| `stale_listing_probe_development.csv` | 40 | One row per (t, k, stayer or not), development counties: clients, exact under the registered reference and under each lookahead and inactive correction, and flagged members. | Supplement E.1 | `assignment_tables.stale_probe` |
| `table_scores.csv` | 42 | One row per (test, table, stat): persons, counties, Brier, log-loss, ECE and cell error with 95% and 99% intervals, held-out and later-file tests. | Tables S2, S13; section 8.1; supplement H.3 | `robustness/outputs.py` |
| `table_scores_by_build_and_age.csv` | 56 | One row per (test, table, build, k): persons, observed exact share, table prediction and age-only prediction. | Tables S15, S16; Supplement L.3 | `robustness/outputs.py` |
| `tables_vs_baselines.csv` | 154 | One row per (test, table, county set, predictor, stat): the tables against age-only and other baselines, with 95% and 99% intervals. | Tables 6, S16; section 8.1 | `robustness/outputs.py` |

## Release tables

  1. By build. A coarse cell under 10 persons merges with the adjacent age band at the same evidence age, build, size and
     building type, under a combined label such as a65+a75. The co-member-age table relabels the same rows and sums
     rows whose keys then agree. Within each coarse cell, co-member-age cells under 10 pool into one row whose ny, mb
     and gb read "other"; when that row is still under 10, the smallest listed cells join it until it reaches 10.
  2. By parity. Each parity row is the sum of by-build rows of the same parity, so the parity tables carry no count the
     by-build tables do not already give. A parity cell takes the union of its builds' band merges. A co-member-age key
     is listed in a parity cell only when every build listing a person in that key lists it; the rest of the cell is
     its "other" row.
  3. p = (ne + pk) / (n + 1) from the summed counts, pk the pooled exact share at the evidence age.
Lookup: the co-member-age key if listed, else the cell's "other" row, else the coarse value, else pk.
