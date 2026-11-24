# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The currency tables (paper section 8.3): fit on the pooled development cells, score on the held-out and later-file
cells (Brier, log-loss, ECE over 10 equal-count bins, per-(build, age) cell MAE) with county-cluster bootstrap percentile
intervals (2,000 replicates, counties resampled with replacement).

coarse_table (buyer side): k, build, size6, bt, ab. comember_age_table (vendor side): k, build, size6, bt, ab, ny, mb, gb.
Each by build date (the held-out test) and by build parity (pres = presidential year, mid = midterm; the later-file
test). Prediction for a cell with training counts (n, ne) at age k: p = (ne + pk) / (n + 1), pk the pooled exact share
at age k; a cell absent from the table takes pk; p is clamped to [1e-6, 1 - 1e-6].

Logarithms are computed in decimal arithmetic (decimal_log), not by the platform's math library, so the log-loss is the
same on every platform.
"""
import csv
import os
import random
import re
from collections import defaultdict
from decimal import Context, Decimal

from . import parallel
from .constants import TABLE_BOOTSTRAP_SEED, BOOTSTRAP_REPS

_CTX = Context(prec=34)
_LOG = {}
EPS = 1e-6


def decimal_log(x: float) -> float:
    """Natural logarithm, correctly rounded to 34 significant digits in decimal, then to the nearest double."""
    v = _LOG.get(x)
    if v is None:
        v = _LOG[x] = float(_CTX.ln(Decimal(x)))
    return v


TABLES = {"coarse_table": ["k", "t_dt", "size6", "bt", "ab"],
          "comember_age_table": ["k", "t_dt", "size6", "bt", "ab", "ny", "mb", "gb"],
          "coarse_table_parity": ["k", "parity", "size6", "bt", "ab"],
          "comember_age_table_parity": ["k", "parity", "size6", "bt", "ab", "ny", "mb", "gb"]}


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["n"] = int(r["n"]); r["ne"] = int(r["ne"]); r["k"] = int(r["k"])
    return rows


def parity(t_dt):
    return "pres" if int(t_dt[:4]) % 4 == 0 else "mid"


def fit(rows, cols):
    tot = defaultdict(lambda: [0, 0]); byk = defaultdict(lambda: [0, 0])
    for r in rows:
        key = tuple(r[c] for c in cols); tot[key][0] += r["n"]; tot[key][1] += r["ne"]
        byk[r["k"]][0] += r["n"]; byk[r["k"]][1] += r["ne"]
    return tot, {k: v[1] / v[0] for k, v in byk.items()}


def fit_tables(cells_path, out):
    """<table>.csv and <table>_pk.csv for each table, from the development cells."""
    rows = load(cells_path)
    os.makedirs(out, exist_ok=True)
    for r in rows:
        r["parity"] = parity(r["t_dt"])
    for name, cols in TABLES.items():
        tot, pk = fit(rows, cols)
        path = os.path.join(out, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["build" if c == "t_dt" else c for c in cols] + ["n", "ne", "p"])
            for key in sorted(tot, key=lambda t: tuple(str(x) for x in t)):
                n, ne = tot[key]; k = key[cols.index("k")]
                w.writerow(list(key) + [n, ne, f"{(ne + pk[k]) / (n + 1):.10f}"])
        with open(os.path.join(out, f"{name}_pk.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n"); w.writerow(["k", "pk"])
            for k in sorted(pk):
                w.writerow([k, f"{pk[k]:.10f}"])


def read_table(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = [c for c in rows[0] if c not in ("n", "ne", "p")]
    tab = {tuple(r[c] for c in cols): float(r["p"]) for r in rows}
    pk = {int(r["k"]): float(r["pk"]) for r in csv.DictReader(open(path[:-4] + "_pk.csv", encoding="utf-8"))}
    return cols, tab, pk


def predict(cols, tab, pk, r):
    key = tuple(str(r["t_dt"] if c == "build" else parity(r["t_dt"]) if c == "parity" else r[c]) for c in cols)
    return min(max(tab.get(key, pk[r["k"]]), EPS), 1 - EPS)


def metrics(scored, counties=None):
    """scored: list of (county, t_dt, k, n, ne, p). counties: county weights (bootstrap) or None."""
    w = counties
    N = brier = ll = 0.0
    cell = defaultdict(lambda: [0, 0.0, 0.0]); preds = []
    for c, t, k, n, ne, p in scored:
        m = 1 if w is None else w.get(c, 0)
        if m == 0:
            continue
        n *= m; ne *= m
        N += n; brier += ne * (1 - p) ** 2 + (n - ne) * p ** 2
        ll -= ne * decimal_log(p) + (n - ne) * decimal_log(1 - p)
        cp = cell[(t, k)]; cp[0] += n; cp[1] += p * n; cp[2] += ne
        preds.append((p, n, ne))
    preds.sort(); target = N / 10; acc = 0; b = 0; bins = defaultdict(lambda: [0, 0.0, 0.0])
    for p, n, ne in preds:
        if acc >= target * (b + 1) and b < 9:
            b += 1
        bins[b][0] += n; bins[b][1] += p * n; bins[b][2] += ne; acc += n
    ece = sum(abs(sp - sy) for bn, sp, sy in bins.values()) / N
    mae = sum(abs(sp / n - ne / n) for n, sp, ne in cell.values()) / len(cell)
    return dict(brier=brier / N, logloss=ll / N, ece=ece, cell_mae=mae, n=int(N),
                cells={tk: (v[1] / v[0], v[2] / v[0], v[0]) for tk, v in sorted(cell.items())})


def county_draws(counties, reps, seed):
    """The bootstrap replicates: for each, how many times each county is drawn (len(counties) draws with replacement
    from the sorted county list), in the generator's order."""
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        d = defaultdict(int)
        for _ in counties:
            d[rng.choice(counties)] += 1
        out.append(dict(d))
    return out


def _boot_block(task):
    scored, draws = task
    return [{k: v for k, v in metrics(scored, d).items() if k != "cells"} for d in draws]


def score(table_path, cells_path, label, reps=BOOTSTRAP_REPS, seed=TABLE_BOOTSTRAP_SEED):
    """The score report text (label names the table in the first line)."""
    cols, tab, pk = read_table(table_path)
    rows = load(cells_path)
    scored = [(r["county0"], r["t_dt"], r["k"], r["n"], r["ne"], predict(cols, tab, pk, r)) for r in rows]
    m = metrics(scored)
    out = [f"{label}: n={m['n']} brier={m['brier']:.5f} logloss={m['logloss']:.5f} ece={m['ece']:.4f} cell_mae={m['cell_mae']:.4f}"]
    for (t, k), (p, e, n) in m["cells"].items():
        out.append(f"  {t} k={k} n={int(n)} predicted={p:.4f} exact={e:.4f} diff={p - e:+.4f}")
    draws = county_draws(sorted({s[0] for s in scored}), reps, seed)
    boot = defaultdict(list)
    for part in parallel.pmap(_boot_block, [(scored, blk) for blk in parallel.chunks(draws, parallel.workers())]):
        for bm in part:
            for key in ("brier", "logloss", "ece", "cell_mae"):
                boot[key].append(bm[key])
    for key, v in boot.items():
        v.sort(); lo = v[int(0.025 * reps)]; hi = v[int(0.975 * reps) - 1]
        out.append(f"  {key} 95% county bootstrap [{lo:.5f}, {hi:.5f}] ({reps} reps)")
    return "\n".join(out) + "\n"


# Development values (leave-county-out, build-date form) and criteria stated before the held-out test
DEVELOPMENT = {("coarse_table", "brier"): 0.19720, ("coarse_table", "cell_mae"): 0.0043, ("coarse_table", "ece"): 0.0045,
               ("comember_age_table", "brier"): 0.19138, ("comember_age_table", "cell_mae"): 0.0045,
               ("comember_age_table", "ece"): 0.0048,
               ("coarse_table_parity", "brier"): 0.19801, ("comember_age_table_parity", "brier"): 0.19219}
CRITERIA = {("coarse_table", "brier"): 0.2022, ("comember_age_table", "brier"): 0.1964, ("coarse_table", "cell_mae"): 0.010,
            ("comember_age_table", "cell_mae"): 0.010, ("coarse_table", "ece"): 0.010, ("comember_age_table", "ece"): 0.010,
            ("coarse_table_parity", "cell_mae"): 0.030, ("comember_age_table_parity", "cell_mae"): 0.030}
BASELINES = {"age_only_brier": 0.23431, "tree_brier_4fold": 0.18923, "comember_age_table_brier_4fold": 0.19136,
             "coarse_table_parity_brier": 0.19801, "comember_age_table_parity_brier": 0.19219}
REPORTS = (("registered", "held_out_score_{}.txt", ("coarse_table", "comember_age_table")),
           ("lookahead_one_election", "held_out_lookahead_one_election_score_{}.txt", ("coarse_table", "comember_age_table")),
           ("later_file", "later_file_score_{}.txt", ("coarse_table_parity", "comember_age_table_parity")))


def table_results(src, out):
    """Collect the score reports under src into out/held_out_metrics.csv and out/held_out_cells.csv. The development
    column holds the leave-county-out values and the criterion column the thresholds stated before the test."""
    os.makedirs(out, exist_ok=True)
    mets, cells = [], []
    for ref, pat, tabs in REPORTS:
        for tab in tabs:
            lines = open(os.path.join(src, pat.format(tab)), encoding="utf-8").read().split("\n")
            head = re.search(r"n=(\d+) brier=([\d.]+) logloss=([\d.]+) ece=([\d.]+) cell_mae=([\d.]+)", lines[0])
            vals = dict(zip(("brier", "logloss", "ece", "cell_mae"), map(float, head.groups()[1:])))
            ci = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in
                  (re.match(r"\s+(\w+) 95% county bootstrap \[([\d.]+), ([\d.]+)\]", ln) for ln in lines) if m}
            for met, v in vals.items():
                lo, hi = ci[met]
                dev = DEVELOPMENT.get((tab, met), "") if ref in ("registered", "later_file") else ""
                crit = CRITERIA.get((tab, met), "") if ref in ("registered", "later_file") else ""
                mets.append((tab, ref, met, v, lo, hi, dev, crit))
            mets.append((tab, ref, "n", int(head.group(1)), "", "", "", ""))
            for ln in lines:
                m = re.match(r"\s+(\d{4}-\d{2}-\d{2}) k=(\d) n=(\d+) predicted=([\d.]+) exact=([\d.]+) diff=([-+][\d.]+)", ln)
                if m:
                    cells.append((tab, ref, m.group(1)[:4], m.group(2), m.group(3), m.group(4), m.group(5), float(m.group(6))))
    for k, v in BASELINES.items():
        mets.append(("baseline", "development", k, v, "", "", "", ""))
    with open(os.path.join(out, "held_out_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(("table", "reference", "metric", "value", "lo", "hi", "development", "criterion"))
        w.writerows(mets)
    with open(os.path.join(out, "held_out_cells.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(("table", "reference", "build", "k", "n", "predicted", "exact", "diff")); w.writerows(cells)
