# Runbook: reproduce the results on two machines and compare

This runbook rebuilds every result from the public inputs on two machines and checks that the output files are
byte-identical. It is written for a macOS arm64 desktop and a Linux cluster with a Grid Engine scheduler, but any two
machines with CPython 3.13 work the same way.

## 1. What each machine needs

- CPython 3.13 (3.12 also works). Not 3.14: the FarmHash package (cityhash 0.4.10) publishes wheels only up to 3.13.
- About 30 GB of free disk for the work directory, plus 1.5 GB for one snapshot file at a time when `--discard-inputs`
  is used (or 40 GB to keep all 41 files).
- Network access to `s3.amazonaws.com` (the NCSBE snapshot files) and to PyPI for the first setup; `services.nconemap.gov`
  on the machine that downloads the address points. Without PyPI, build a
  wheelhouse on a connected machine and pass `--wheelhouse DIR`.
- The voter history file `ncvhis_Statewide.zip`, the address-point directory and the Ohio and Washington lists, the same
  copies on both machines (section 2).
- Memory: the full run completed on a 24 GB machine; give a cluster job at least that. The run sizes its process pools
  from the memory it finds, so more memory lets more steps run at once.
- Time for the full run from an empty work directory, downloads included: 2 hours 52 minutes on 32 cores of an AMD EPYC
  7543 node, plus 16 minutes for Ohio and Washington. The run uses every core: by default the number of cores minus one,
  or `NSLOTS` under Grid Engine. No output depends on the number of cores.

## 2. Inputs that change over time

The 41 snapshot files are fixed: each one is checked against `inputs/snapshots.tsv` or `inputs/later_file.tsv` (size and
SHA-256), and the run stops if one differs. So are the Ohio and Washington lists, checked against
`inputs/other_states.tsv`; put them in one directory at the relative paths listed there and pass it with `--other-states`
(INPUT_DATA.md). Without it the other-state results are not written and those outputs will not match.

NCSBE replaces `ncvhis_Statewide.zip` every week, and two weekly copies disagree on a few hundred of the about 4.77
million NCIDs recorded as voting on 2016-11-08. Download it once, copy that file to the second machine, and keep it. The
run records its SHA-256 in `results/prep_manifest.json`. When `inputs/voter_history.tsv` names a copy, the run stops if
the file given differs from it. Without `--voter-history`, the `tlag` column of `ref_sensitivity.csv` is left blank and
those files will not match the recorded ones.

```bash
curl -o ncvhis_Statewide.zip https://s3.amazonaws.com/dl.ncsbe.gov/data/ncvhis_Statewide.zip
shasum -a 256 ncvhis_Statewide.zip      # macOS; sha256sum on Linux
```

The NC OneMap address points are a live service with no versions. Download them once (about 6.3 million points, a few
minutes) and copy the whole directory, with its `SHA256SUMS`, to the second machine:

```bash
python3 fetch_address_points.py ~/stale-assignments/address_points
```

The run checks every file against `SHA256SUMS` and records the digest of `SHA256SUMS`. The option must be given on the
first run in a work directory, because the audit reads address tables that stage 2 later slims.

## 3. Machine A: macOS arm64

```bash
mkdir -p ~/stale-assignments/zips ~/stale-assignments/run
cd /path/to/release                      # the unpacked release
python3.13 bootstrap.py --workspace ~/stale-assignments/run --snapshots ~/stale-assignments/zips --allow-download \
    --fetch --discard-inputs --voter-history ~/stale-assignments/ncvhis_Statewide.zip \
    --address-points ~/stale-assignments/address_points --other-states ~/stale-assignments/other_states
```

`bootstrap.py` creates `~/stale-assignments/run/venv` from the exact wheel lock for macOS arm64 and CPython 3.13 (hashes checked), runs
the unit tests, then runs `reproduce_all.py`. Keep the machine awake for the whole run (`caffeinate -i` in front of the
command). A run that stops resumes from the tables already built when the same command is run again.

The last line the command prints is

- `PASS: 320 of 320 recorded outputs identical` (exit status 0), or
- `FAIL: K of 320 recorded outputs differ` (exit status 1), after one `DIFFERS` line per output that differs.

The progress log is `~/stale-assignments/run/work/run.log`, and the list of outputs is
`~/stale-assignments/run/work/results/OUTPUTS.sha256`.

## 4. Machine B: Linux cluster (Grid Engine)

Find the Python 3.13 module (`module avail python`), then submit a job like this one. Change the queue, the parallel
environment and the paths to the cluster's names.

```bash
#!/bin/bash
#$ -N stale-assignments
#$ -q long
#$ -pe smp 32
#$ -l h_vmem=4G            # per slot: 32 x 4 GB = 128 GB
#$ -j y
#$ -o stale-assignments.log
module load python/3.13
cd /path/to/release
python3 bootstrap.py --workspace "$HOME/stale-assignments/run" --snapshots "$HOME/stale-assignments/zips" --allow-download \
    --fetch --discard-inputs --voter-history "$HOME/stale-assignments/ncvhis_Statewide.zip" \
    --address-points "$HOME/stale-assignments/address_points" --other-states "$HOME/stale-assignments/other_states"
```

On Linux x86_64 with CPython 3.13, `bootstrap.py` uses `requirements-lock-linux-cp313.txt`. On another interpreter or
architecture it installs the version pins of `requirements.txt` without hashes, and says so. Use scratch space for the
workspace if the home quota is under 45 GB.

When compute nodes cannot reach PyPI, build the environment on the login node first (a minute, one core, and the unit
tests take under a second), then submit the job; the job finds the environment ready and installs nothing:

```bash
module load python/3.13
python3 bootstrap.py --workspace "$HOME/stale-assignments/run" --allow-download --setup-only
```

The last line reads `SETUP OK:` with the interpreter, platform, Polars version and lock file used.

## 5. Compare

Copy the cluster's `OUTPUTS.sha256` to the desktop and run:

```bash
python3 compare_outputs.py ~/stale-assignments/run/work/results/OUTPUTS.sha256 cluster-OUTPUTS.sha256
```

`SAME: N of N outputs identical across 2 lists` means every result file and every table digest is byte-identical across
the two machines. The script uses only the standard library.

To check repeatability on one machine without rebuilding stages 1 and 2, move `work/results` aside, run the same command
with `--phases runs,build,analyze,other_states` and compare the new `OUTPUTS.sha256` with the old one.

## 6. If something differs

- `DIFFERS` on a file under a run's `results/<run>/` directory: look at its `digests/` files to find the first
  intermediate table whose digest differs. If the earliest ones differ too, compare the `addr`, `tags` and `graph`
  sections of the two `results/prep_manifest.json` files date by date. A difference in `stage1_input` means an input file
  differs; the run would have stopped unless `--allow-changed-inputs` was given.
- Only `ref_sensitivity.csv` differs: check that both runs read the same voter history copy (`voter_history` in
  `prep_manifest.json`). Only the `unit_key_*.csv` files differ: check `address_points` there. Only files under
  `other_states/` or `household_change/transfer/` differ: check `other_states_input` there.

## 7. What to send back

From one machine, the text outputs (aggregates only; the Parquet tables in the work directory hold record-level rows and
stay on the machine):

```bash
cd ~/stale-assignments/run/work && zip -r -X results-text.zip results -i '*.csv' '*.tsv' '*.json' '*/OUTPUTS.sha256'
```

From the other machine, its `OUTPUTS.sha256`. From both, the first and last lines of `run.log` (they name the platform,
interpreter and Polars version).
