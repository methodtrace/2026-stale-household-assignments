#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Rebuild every derived table and result from the published files, then compare with the recorded output list.

Usage:
    python reproduce_all.py --work DIR --snapshots DIR [--voter-history ZIP] [--address-points DIR]
                            [--other-states DIR]
                            [--phases tables,runs,build,analyze,other_states] [--jobs N] [--fetch] [--discard-inputs]
                            [--allow-changed-inputs]

  --work       working directory (peak about 25 GB of tables plus the stage 3 runs); a rerun resumes where the last stopped
  --snapshots  directory holding the 41 VR_Snapshot_YYYYMMDD.zip files (INPUT_DATA.md)
  --fetch      download any missing snapshot file from the NCSBE address in INPUT_DATA.md (checked like any other file)
  --discard-inputs  delete each snapshot file after its address table is built
  --voter-history   ncvhis_Statewide.zip, the copy named in inputs/voter_history.tsv; without it tlag is blank
  --address-points  directory written by fetch_address_points.py; without it the unit-key audit is not written
  --other-states    directory holding the Ohio and Washington lists named in inputs/other_states.tsv; without it the
                    other-state results are not written
  --phases     tables (stages 1 and 2), runs (stage 3), build (cell tables), analyze (statistics and result files),
               other_states (Ohio and Washington: section 9 and the household-change transfer)
  --jobs       stage 3 runs at the same time (default: memory / 12 GB, at most the workers)

Results go to <work>/results/ (layout in OUTPUTS.md). The analyze phase writes <work>/results/OUTPUTS.sha256; this
script then compares it, line by line, with recorded/OUTPUTS.sha256 and ends with "PASS: N of N recorded outputs
identical" (exit status 0) or "FAIL: K of N recorded outputs differ" (exit status 1).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def work_dir(argv):
    for i, a in enumerate(argv):
        if a == "--work" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--work="):
            return a.split("=", 1)[1]
    return None


def main(argv):
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return 0
    r = subprocess.run([sys.executable, "-B", "-m", "stale_assignments.reproduce", *argv], cwd=HERE,
                       env={**os.environ, "PYTHONPATH": HERE})
    if r.returncode:
        return r.returncode
    w = work_dir(argv)
    new = os.path.join(os.path.abspath(w), "results", "OUTPUTS.sha256") if w else None
    rec = os.path.join(HERE, "recorded", "OUTPUTS.sha256")
    if not new or not os.path.exists(new):
        print("no OUTPUTS.sha256 written (the analyze phase did not run); nothing compared")
        return 0
    if not os.path.exists(rec):
        print(f"no recorded/OUTPUTS.sha256 in this checkout; outputs listed in {new}")
        return 0
    return compare(rec, new)


def compare(rec, new):
    """Print every output that differs from the recorded list, then PASS: N of N recorded outputs identical, or
    FAIL: K of N recorded outputs differ (exit status 1). An output missing from either list counts as differing."""
    from compare_outputs import read
    want, got = read(rec), read(new)
    names = sorted(set(want) | set(got))
    bad = [x for x in names if want.get(x) != got.get(x)]
    for x in bad:
        print(f"DIFFERS {x}: recorded={want.get(x, 'missing')[:12]} this run={got.get(x, 'missing')[:12]}")
    if bad:
        print(f"FAIL: {len(bad)} of {len(names)} recorded outputs differ")
        return 1
    print(f"PASS: {len(names)} of {len(names)} recorded outputs identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
