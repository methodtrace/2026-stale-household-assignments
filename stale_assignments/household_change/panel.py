# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The data the household-change fit reads: for each build, the count of anchors by composition cell, building type,
snapshot interval and observed transition (y0 at the earlier snapshot, y1 at the later one), and the snapshot dates as
years since the build.

Inputs (aggregates only, no person-level records):
  transitions  csv.gz with t_dt, cell, bt, j, code (y0-y1 over EB EO XB XO U), n
  snapshots    csv with t_dt, n (snapshot number), u (years since build); also s_dt (snapshot date) for prediction
"""
import csv
import gzip
from collections import defaultdict

import numpy as np

from .model import Cells, H, OBS5


class BuildPanel:
    """The pair counts of one build: cells sorted by (cell, building type), N (n, J, 5, 5), grid indexes gi."""

    def __init__(self, rows, snaps, t):
        keys = sorted({(r[0], r[1]) for r in rows})
        ix = {k: i for i, k in enumerate(keys)}
        u = [0.0] + [x[1] for x in sorted(snaps)]
        self.gi = [int(round(x / H)) for x in u]
        assert all(b > a for a, b in zip(self.gi, self.gi[1:]))
        J = len(u) - 1
        N = np.zeros((len(keys), J, 5, 5))
        o = {c: i for i, c in enumerate(OBS5)}
        for cell, bt, j, code, n in rows:
            y0, y1 = code.split("-")
            N[ix[(cell, bt)], j - 1, o[y0], o[y1]] += n
        self.keys, self.N, self.t = keys, N, t
        self.C = Cells([k[0] for k in keys], [k[1] for k in keys], t)


def load(transitions_path, snapshots_path, builds):
    sn = {}
    with open(snapshots_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sn.setdefault(r["t_dt"], []).append((int(r["n"]), float(r["u"])))
    rows = {}
    with gzip.open(transitions_path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["t_dt"], []).append((r["cell"], r["bt"], int(r["j"]), r["code"], int(r["n"])))
    return [BuildPanel(rows[t], sn[t], t) for t in builds]


def snapshot_grid(snapshots_path):
    """{build date: [(snapshot number, snapshot date, years since build)]} in snapshot order."""
    g = defaultdict(list)
    for r in csv.DictReader(open(snapshots_path, encoding="utf-8")):
        g[r["t_dt"]].append((int(r["n"]), r["s_dt"], float(r["u"])))
    return {t: sorted(v) for t, v in g.items()}
