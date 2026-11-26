# 2026-stale-household-assignments

Reproducibility artifact for the paper "How Fast Stored Household Assignments Go Stale: Evidence from North Carolina Voter Rolls" by Prashant Chaudhary.

For the current version of the paper and its publication status, see this repository:
https://github.com/methodtrace/2026-stale-household-assignments

## Identifiers

- Cite this version: https://doi.org/10.5281/zenodo.24026471
- All versions: https://doi.org/10.5281/zenodo.24026470
- Preprint: arXiv:2611.32091, https://arxiv.org/abs/2611.32091
- Published version: Not yet accepted

Artifact version: **1.0.0**.

## What this is

The study measures how fast stored household assignments go stale. It simulates a vendor that builds an address-based
household graph from North Carolina voter registration snapshots, re-matches client records against that graph years
later, and scores each returned household against the registered co-residents at the scoring date. The code in this
repository rebuilds every derived table from the 41 public snapshot files and recomputes every result in the paper: the
statistics with county-cluster bootstrap intervals for the 20 development and 60 confirmation counties, the held-out and
later-file checks, the currency tables, the household-change model, the refresh test, the robustness checks, the
build-error experiment and the Ohio and Washington comparison.

The repository holds code, the expected input checksums and the recorded aggregate results. It holds no voter records and
no record-level file. Anyone can download the same public North Carolina files from the State Board of Elections; the Ohio
and Washington lists come from those states under their published terms (INPUT_DATA.md).

## Layout

- `stale_assignments/`: the pipeline.
  - `snapshots.py` stage 1: snapshot ZIP to a typed address table (names kept only as fingerprints).
  - `prepare.py` stage 2: household tags, building relabel edges and their components, graph records, age flags.
  - `events.py`, `rematch.py`, `stored_assignment.py`, `score.py` pipeline steps 1 to 5: person and household events,
    re-matching against simulated vendor graphs, stored assignments carried across rebuilds, per-record scores and county
    cells.
  - `statistics.py`, `bootstrap.py` steps 6, 10 and 13: statistics with county-cluster intervals.
  - `descriptives.py` steps 8 and 9: build age against calendar, descriptive cells.
  - `probabilistic_matcher.py`, `match_validity.py` steps 11 and 12: the Fellegi-Sunter baseline with fixed parameters,
    validity of reported matches, link survival and support ceilings.
  - `currency.py`: anchor currency, exits by removal reason with the three-outcome bounds, reference sensitivities, the
    validity frontier, small-cluster intervals, tipping points, envelopes and level bands.
  - `assignment_tables.py`, `table_scores.py`, `later_file.py`: the currency tables, their scores on held-out counties and
    on the later file.
  - `household_change/`: the household-change model, its fit, the held-out test and the transfer to Ohio and Washington.
  - `refresh/`: the refresh decision test. `robustness/`: the robustness checks.
  - `other_registers.py`, `other_states.py`: the Ohio and Washington lists and the build-error experiment.
  - `calibration.py`, `unit_key.py`, `voter_history.py`, `paper_tables.py`: the ACS mobility calibration, the unit-key
    audit against North Carolina address points, the 2016 voters, and the tables behind the figures and Table 1.
  - `exact_math.py`, `student_t.py`, `record_functions.py`, `row_digest.py`: platform-independent exp and log, Student t
    quantiles, the warehouse record functions the analysis was first specified with (FarmHash fingerprints, Soundex, date
    differences) and order-free table fingerprints.
  - `reproduce.py`, `runs.py`, `pipeline.py`, `parallel.py`, `layout.py`, `store.py`: the phases, the stage 3 runs,
    process pools with fixed-order merges, and the result and work-directory layouts.
  - `data/`: the fixed probabilistic-matcher parameters, the county split and the pair lists.
- `inputs/`: sizes and SHA-256 of every input file (`snapshots.tsv`, `later_file.tsv`, `other_states.tsv`,
  `voter_history.tsv`, `address_points.tsv`) and the published Census aggregates (`acs_nc_movers.tsv`,
  `acs_nc_mobility_by_age.tsv`).
- `OUTPUTS.md`: column definitions (unit, denominator) of the exit, reference-sensitivity and anchor diagnostic files.
- `recorded/`: the shipped results and `OUTPUTS.sha256`, the list every reproduction is compared with (320 files).
  `recorded/README.md` is the index: eleven folders of descriptively named aggregates (81 files, among them the plotted
  values of Figures 1 and 2), each with a README that gives one row per file (what it holds, which part of the paper it
  backs, how the run produces it). The other 239 listed files are checked by SHA-256 only: county-level sums, cell counts
  under 10 registrants, intermediate tables, the unmerged fitted tables, and aggregates the paper does not print.
- `make_figures.py`: draws Figures 1 and 2 from `recorded/paper_figures/` or a run's results (needs matplotlib).
- `tests/`: known-answer tests for the record functions, the parser and the exact arithmetic.
- `reproduce_all.py`: runs every phase and compares the outputs with `recorded/OUTPUTS.sha256`.
- `fetch_address_points.py`: downloads the address points once (INPUT_DATA.md).
- `compare_outputs.py`: compares the `OUTPUTS.sha256` lists of two or more runs; `RUNBOOK.md`: how to run on two machines
  and compare.
- `bootstrap.py`, `requirements.txt`, `requirements-lock-*.txt`: private environment setup, version pins and exact wheel
  locks.
- `INPUT_DATA.md`, `ENVIRONMENT.md`, `CHANGELOG.md`, `LICENSE`, `LICENSE-DATA`, `CITATION.cff`, `.zenodo.json`,
  `MANIFEST.sha256`.

## Reproducing the results

1. Download the 41 snapshot ZIP files listed in `inputs/snapshots.tsv` and `inputs/later_file.tsv` into one directory
   (INPUT_DATA.md), or let the run fetch them with `--fetch`. Put the Ohio and Washington lists at the paths in
   `inputs/other_states.tsv`.
2. Run, with CPython 3.12 or 3.13:

   ```bash
   python3 bootstrap.py --workspace /path/outside/repo --snapshots /path/to/zips --allow-download \
       --voter-history /path/to/ncvhis_Statewide.zip --address-points /path/to/address_points \
       --other-states /path/to/other_states [--fetch --discard-inputs]
   ```

   or, in an environment that already has the pinned packages:

   ```bash
   python3 reproduce_all.py --snapshots /path/to/zips --work /path/to/work --voter-history /path/to/ncvhis_Statewide.zip \
       --address-points /path/to/address_points --other-states /path/to/other_states
   ```

   The run uses every core by default (cores minus one, or `NSLOTS` under Grid Engine); no output depends on the number.
3. The run ends with `PASS: N of N recorded outputs identical` or lists every output that differs. Results are in
   `<work>/results/` with `OUTPUTS.sha256`: the shipped files under the same folder names as `recorded/`, the others in
   the working folders (`results/` per run, `tables/`, `table_cells/`, `household_change/`, `refresh/`, `robustness/`,
   `build_error/`, `other_states/`). The last step of the run writes the shipped copies (`release_files.py`).

A rerun resumes from the tables already in the work directory. Delete the work directory to start over.

`recorded/` comes from the copies of the voter history file and the address points named in `inputs/voter_history.tsv`
and `inputs/address_points.tsv` (both downloaded on 2026-11-02). Neither source keeps old versions, and the input data are
not redistributed here. With other copies, run with `--allow-changed-inputs`: the four outputs that read them differ
(`nc_development_counties/reference_sensitivity.csv`, `nc_confirmation_counties/reference_sensitivity.csv`,
`results/unit_key_county.csv`, `nc_statewide/unit_key_audit.csv`) and the other 316 are expected to be identical.

### Platform support

| Platform | Status |
|---|---|
| Linux x86_64, glibc, CPython 3.13 | Tested: full reproduction on RHEL 9 (AMD EPYC 7543, 32 cores, Grid Engine), all outputs identical to `recorded/OUTPUTS.sha256`. Dependencies from the exact wheel lock. |
| macOS (Apple silicon), CPython 3.13 | Tested: full reproduction on macOS 27.0.1 (Apple M5 Pro, 15 cores, 24 GB), all outputs identical to `recorded/OUTPUTS.sha256`. Dependencies from the exact wheel lock. |
| Other platforms with binary wheels for the pins | Not tested; expected to work. |

### What is exact, and why it does not depend on the platform

Every output file is compared byte for byte. That works across platforms because no result depends on the platform's
floating-point library:

- Per-record scores are rationals (counts over counts). County cells sum them exactly as fractions and round once to the
  nearest double.
- Statistics use only correctly rounded operations: `math.fsum` for sums, IEEE addition, subtraction, multiplication and
  division, and `math.sqrt`.
- The base-2 logarithm in the probabilistic matcher's weights is computed in decimal arithmetic at 50 digits and rounded
  once. The Student t quantiles of the small-cluster intervals are likewise computed in decimal arithmetic at 60 digits
  and rounded once (`student_t.py`).
- Model fits (the household-change model, the table scores) take exp and log from `exact_math.py`, built from IEEE
  addition, subtraction, multiplication and division only, and sum in a fixed order.
- Work split across processes is merged in a fixed order, so the number of cores does not change any output.
- Level bands are least-squares fits in exact rational arithmetic.
- Bootstrap weights come from FarmHash fingerprints of fixed strings, not from a platform random generator.
- Every table is sorted before it is written; floats are written as the shortest text that reads back to the same double.

### Relation to the warehouse version

The analysis was first run in BigQuery SQL. This Python version reproduces it: the stage 1 to 3 tables match the
warehouse tables row for row on every column that holds integers, strings, dates or booleans. Floating-point columns match
to a tolerance: county cells within 1e-6 absolute (2.2e-12 relative), every reported statistic to nine decimals. This
version sums exact fractions and rounds once; the warehouse sums doubles in an unspecified order, so its last digits are
not reproducible even run to run. ENVIRONMENT.md records the comparison and the few documented places where the warehouse
chose an arbitrary row and this version makes a fixed choice.

## Data, ethics and licenses

The input is public-record voter registration data. The code reads names and addresses only to build household keys and
to simulate matching, inside the working directory you choose; names are kept only as fingerprints after stage 1.
No registrant under 18 at the build date is ever a client, in any analysis or state (North Carolina: the minor flag;
Ohio and Washington: the full birth date); minors stay in the files only as co-members of adult clients.
Record-level tables stay in the work directory (as Parquet tables) and are not part of the release. `results/` holds
aggregates only; the result files whose county or fine cells count fewer than 10 registrants are checked by SHA-256 and
not shipped in `recorded/`. The fitted tables ship as release copies with no cell under 10 persons
(`recorded/fitted_tables/README.md`).

Code: Apache-2.0 (`LICENSE`). Recorded aggregate results under `recorded/`: CC BY 4.0 (`LICENSE-DATA`). The snapshot files
are published by the North Carolina State Board of Elections and are not redistributed here.
