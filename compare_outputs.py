#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Compare OUTPUTS.sha256 lists from two or more runs (other machines, or repeated runs on one machine).

Usage:
    python3 compare_outputs.py A/OUTPUTS.sha256 B/OUTPUTS.sha256 [C/OUTPUTS.sha256 ...]

Prints every output path whose SHA-256 differs between the lists or that is missing from one of them, then one summary
line. Exit status 0 means every list holds the same paths with the same digests. Standard library only, so it runs
without the pinned environment.
"""
import sys


def read(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, ln in enumerate(f, 1):
            ln = ln.rstrip("\n")
            if not ln:
                continue
            digest, sep, name = ln.partition("  ")
            if not sep or len(digest) != 64:
                sys.exit(f"{path}:{n}: not a sha256sum line")
            out[name] = digest
    return out


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    lists = [read(p) for p in argv]
    names = sorted(set().union(*lists))
    bad = 0
    for name in names:
        ds = [x.get(name, "missing") for x in lists]
        if len(set(ds)) > 1:
            bad += 1
            print(f"DIFFERS {name}: " + ", ".join(f"{p}={d[:12]}" for p, d in zip(argv, ds)))
    print(f"{'SAME' if not bad else 'DIFFERENT'}: {len(names) - bad} of {len(names)} outputs identical across {len(argv)} lists")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
