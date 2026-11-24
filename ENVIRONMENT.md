# Environments

## What every run must reproduce

`reproduce_all.py` writes `<work>/results/OUTPUTS.sha256`, the SHA-256 of every text output (320 files): the statistics
and result files of every run under `results/`, the tables and their scores, the household-change model and its held-out
and transfer tests, the refresh test, the robustness checks, the build-error experiment, the Ohio and Washington results
the tables in `results/paper/` and the plotted values of Figures 1 and 2. Each run's `digests/` holds the row count and order-free row fingerprint of every
intermediate table, so a run that differs anywhere shows where. The run passes only when every line equals
`recorded/OUTPUTS.sha256`.

Byte equality across platforms rests on five rules, kept throughout the code:

1. No result depends on the platform's math library. Per-record scores are integer ratios; county cells sum them as exact
   fractions and round once. Statistics use `math.fsum`, IEEE addition, subtraction, multiplication and division, and
   `math.sqrt`, all correctly rounded. The base-2 logarithm in the probabilistic matcher's weights is computed in decimal arithmetic
   at 50 digits and rounded once. No Polars floating-point aggregation (sum, mean) is used anywhere; Polars sums only integers.
2. No result depends on row order. Every table is sorted on a total key before it is written, and every
   "first row per group" follows a sort on a key that is total or whose ties are identical rows. Polars' parallel hash
   joins and group-bys may return rows in any order; nothing downstream sees that order.
3. Randomness is a function of fixed strings: bootstrap weights come from FarmHash fingerprints of the seed, county and
   replicate number; the client sample and the probabilistic matcher's tie-break are fingerprints too. The one use of
   Python's generator (`paper_tables.py`, the bands behind Figures 1 and 2) is seeded with a fixed string, which CPython hashes the same way on
   every platform.
4. Text is written the same way everywhere: UTF-8, LF line ends, floats as the shortest text that reads back to the same
   double (`repr`), JSON with sorted keys.
5. Model fits use no platform math. The household-change model and the table scores take exp and log from
   `exact_math.py` (IEEE add, subtract, multiply and divide only) and sum in a fixed order, and the optimizer
   (`household_change/optimize.py`) is plain Python, so a fit gives the same bits on every platform. Work split across
   processes (`parallel.py`) is merged in a fixed order, so the number of cores does not change any output.

Seeds are fixed purpose labels in `constants.py`: a string salt is the label itself, and an integer seed is the first 8
bytes of the SHA-256 of its label.

## Dependencies

`requirements.txt` pins Polars 2.1.0 (with its 32-bit-index runtime) and cityhash 0.4.10, whose `farmhash` module
provides FarmHash Fingerprint64. Both publish binary wheels for Linux x86_64 and aarch64, macOS x86_64 and arm64 and
Windows; cityhash 0.4.10 has wheels up to CPython 3.13, so the release supports CPython 3.12 and 3.13.
`requirements-lock-linux-cp313.txt` and `requirements-lock-macos-arm64-cp313.txt` additionally pin the exact wheel of each
package, by SHA-256, for CPython 3.13 on Linux x86_64 (glibc) and on macOS arm64. `bootstrap.py` uses a wheel lock there and
the version pins elsewhere. Everything else is the Python standard library.

## Tested executions

Two full reproductions on 2026-11-07 and 2026-11-08, from the same 41 snapshot files, the same copies of the two moving
inputs (`inputs/voter_history.tsv`, `inputs/address_points.tsv`) and the same Ohio and Washington lists
(`inputs/other_states.tsv`):

| | Machine A | Machine B |
|---|---|---|
| Hardware | Apple M5 Pro, 15 cores, 24 GB | AMD EPYC 7543, 32 slots of a Grid Engine node |
| Operating system | macOS 27.0.1, arm64 | RHEL 9, Linux 5.14.0-687, x86_64, glibc 2.34 |
| Interpreter, Polars | CPython 3.13.18, Polars 2.1.0 | CPython 3.13.18, Polars 2.1.0 |
| Lock file | `requirements-lock-macos-arm64-cp313.txt` | `requirements-lock-linux-cp313.txt` |
| Unit tests | 38, OK | 38, OK |
| Wall time | not measured (the run was resumed after a restart) | 2 h 52 min from an empty work directory, plus 16 min for Ohio and Washington |
| Outputs | 320 | 320 |

`compare_outputs.py` on the two `OUTPUTS.sha256` lists: `SAME: 320 of 320 outputs identical across 2 lists`. Machine A's
list is `recorded/OUTPUTS.sha256`. Peak memory and disk use were not measured.

## Relation to the BigQuery reference

The analysis was first specified and run as SQL in a cloud warehouse (BigQuery). This version was
compared with those tables on 2026-11-01 and 2026-11-02, by row counts and order-free row fingerprints (XOR of FarmHash
Fingerprint64 of each row's text, which BigQuery computes with `FARM_FINGERPRINT`).

Exact (every row, every column):

| Stage | Tables |
|---|---|
| 1 | all 40 typed address tables of the registered snapshots (556,508,593 rows kept) |
| 2 | household tags and graph records for all 40 dates; relabel edges (109,051); canonical address map (204,140); age flags except the ties below |
| 3, development counties | person events (6,803,756); household events (3,571,492); rematch (73,052,356); records (190,153,084); stored IDs (20,411,268); first-seen best (6,777,058) and records (6,803,756); age cells; integer columns of cells, descriptive cells and validity cells; keys of descriptive and validity cells |

To a tolerance (floating-point columns):

| Output | Agreement |
|---|---|
| cells (f1, f1c, f1c_u, loss and its source splits; 4,640 rows) | within 5.6e-7 absolute, 2.2e-12 relative; 99 column-and-policy groups checked, 77 identical at six decimals, the rest compared value by value |
| descriptive and validity cells (num, den) | identical at six decimals |
| stats.csv, desc_stats.csv, valid_stats.csv | identical at nine decimals (point, intervals, jackknife SE, LOCO range, threshold); decisions identical |
| paper tables (results/paper) | identical text, four decimals |

Why the floats differ: this version sums each county's per-record scores as exact fractions and rounds once; BigQuery sums
doubles in an order it does not fix, so its sums carry rounding error that grows with county size (largest in county
60, Mecklenburg) and can change between runs. The Python value is the correctly rounded exact sum.

Documented differences in what is computed:

- Age-flag ties. A registration whose age is coded 0 or above 110 takes an imputed age from the nearest snapshot with a
  plausible age. When two snapshots are equally near, BigQuery took an arbitrary one; this version takes the earlier
  snapshot, then the smaller age. 950 of 226,146 flagged rows are such ties. They change the minor flag of one person in
  one snapshot (confirmation county 34, 2016-11-08) and nothing in the development counties.
- Quoting. The warehouse loaded the five Election Day files of 2016 to 2024 without quote handling and the other 35 with
  it. This version reproduces that per file (`snapshots.UNQUOTED`, INPUT_DATA.md); the rule changes one used-column line in
  the 40 registered files.
- NCID re-issue ties. The crosswalk takes, for each NCID, the name and birth-year keys of one row: at its first snapshot
  the active-or-inactive row with the latest registration date, and at its last active-or-inactive snapshot the row with
  the latest registration date. When an NCID has two rows on that snapshot with the same registration date and different
  keys, BigQuery took an arbitrary one; this version takes the smallest keys. The rebuilt crosswalk has 342,611 pairs
  against the warehouse's 342,613: two pairs only the warehouse has, and in each of four years one pair differs. All four
  involve an NCID with such a tie (checked by fingerprint against the warehouse rows). No development-county client is
  affected: the re-linked counts and exact counts in ref_sensitivity.csv equal the warehouse query at every k.
- Anchor-currency and reference outputs, development counties: n_mixed, n_relinked and the re-linked exact counts equal
  the warehouse queries exactly; link_same_population equals them at every printed decimal (four); anchor_currency at
  four decimals in all 20 cells; three_outcome, ghost and slag at three decimals (the precision the warehouse results
  were logged at); tlag at three decimals with a different weekly copy of the voter history.
- The confirmation run (60 counties) was not compared row by row here; its integer tables are expected to differ only by
  the age-flag tie above.
