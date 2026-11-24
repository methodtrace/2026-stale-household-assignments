# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Aggregate tables behind the paper's figures and Table 1 (development counties), written to results/paper/.

development_policies_2016.tsv     every policy and client-file lag at the 2016 build: match rates, co-member and self-inclusive F1, gap
development_never_by_build.tsv    the never-refreshed graph for every Election Day build 2008 to 2016
development_loss_sources_2016.tsv share of lost accuracy by error source
development_age_curve_pooled.tsv  the pooled November age curve with 99% county-cluster intervals (from
                                  descriptive_stats.csv)
development_bands.tsv             99% bands for Figures 1 and 2 by a county resampling bootstrap (seed below)
development_table1_matched.tsv    Table 1: co-member F1 and exact household among matched records, alone among unmatched

Ratios are of exact sums and printed with four decimals. Every other number in the paper's tables is a row of
stats.csv, descriptive_stats.csv or validity_stats.csv of the development or confirmation run (named in each table's caption or notes).
"""
import csv
import math
import os
import random
from fractions import Fraction

import polars as pl

from .constants import FIGURE_BANDS_SEED as BANDS_SEED
BANDS_REPS = 2000


def _w(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _r4(x):
    return "" if x is None else f"{x:.4f}"


def _sum(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) if all(isinstance(x, int) for x in xs) else math.fsum(xs)


def _div(a, b):
    return None if a is None or not b else a / b


def _group(cells, keys, cols):
    g = {}
    for r in cells.sort(keys).iter_rows(named=True):
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return {k: {c: _sum([r[c] for r in rs]) for c in cols} for k, rs in g.items()}


def write_tables(res: str):
    dev = os.path.join(res, "development")
    if not os.path.exists(os.path.join(dev, "cells.parquet")):
        return
    out = os.path.join(res, "paper")
    os.makedirs(out, exist_ok=True)
    cells = pl.read_parquet(os.path.join(dev, "cells.parquet"))
    cols = ["n_all", "n", "n_m", "n_m_all", "f1c", "f1", "n_alone", "n_mov", "loss", "loss_mover_together", "loss_mover_split",
            "n_mover_split"]

    g = _group(cells.filter(pl.col("t") == 2016), ["policy", "lag", "k"], cols)
    rows = []
    for (pol, lag, k), s in sorted(g.items()):
        mr, f1 = _div(s["n_m"], s["n"]), _div(s["f1c"], s["n"])
        rows.append([pol, lag, k, _r4(mr), _r4(_div(s["n_m_all"], s["n_all"])), _r4(f1), _r4(_div(s["f1"], s["n"])),
                     _r4(None if mr is None or f1 is None else mr - f1)])
    _w(os.path.join(out, "development_policies_2016.tsv"),
       ["policy", "crm_lag", "k", "match_rate_truth", "match_rate_all", "f1_comember", "f1_selfinclusive", "gap"], rows)

    g = _group(cells.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0)), ["t", "k"], cols)
    rows = []
    for (t, k), s in sorted(g.items()):
        mr, f1 = _div(s["n_m"], s["n"]), _div(s["f1c"], s["n"])
        rows.append([t, k, s["n_all"], s["n"], _r4(mr), _r4(_div(s["n_m_all"], s["n_all"])), _r4(f1), _r4(mr - f1),
                     _r4(_div(s["n_alone"], s["n"]))])
    _w(os.path.join(out, "development_never_by_build.tsv"),
       ["t", "k", "n_clients", "n_truth", "match_rate_truth", "match_rate_all", "f1_comember", "gap", "share_alone"], rows)

    pols = ("never_rebuilt", "member_tracker", "lagged_rebuild", "last_known", "never_rebuilt_name_check")
    g = _group(cells.filter((pl.col("t") == 2016) & (pl.col("lag") == 0) & pl.col("policy").is_in(pols)), ["policy", "k"], cols)
    rows = []
    for (pol, k), s in sorted(g.items()):
        s1, s2 = _div(s["loss_mover_together"], s["loss"]), _div(s["loss_mover_split"], s["loss"])
        rows.append([pol, k, _r4(s1), _r4(s2), _r4(1 - (s["loss_mover_together"] + s["loss_mover_split"]) / s["loss"]), _r4(_div(s["loss"], s["n"])),
                     _r4(_div(s["n_mover_split"], s["n"]))])
    _w(os.path.join(out, "development_loss_sources_2016.tsv"),
       ["policy", "k", "share_movers_together", "share_movers_split", "share_stayers_and_noise", "mean_loss",
        "split_mover_share_of_persons"], rows)

    ds = {r["stat"]: r for r in pl.read_csv(os.path.join(dev, "descriptive_stats.csv"), infer_schema=False).iter_rows(named=True)}
    f = lambda v: None if v in (None, "") else float(v)
    rows = []
    for a in range(1, 9):
        fc, mr, gp = ds.get(f"build_age|age|{a}|f1c"), ds.get(f"build_age|age|{a}|mr"), ds.get(f"pooled November build-age G, age {a}")
        rows.append([a, _r4(f(fc["point"])), _r4(f(fc["lo99"])), _r4(f(fc["hi99"])), _r4(f(mr["point"])), _r4(f(gp["point"])),
                     _r4(f(gp["lo99"])), _r4(f(gp["hi99"]))])
    _w(os.path.join(out, "development_age_curve_pooled.tsv"),
       ["age_years", "f1_comember", "f1_lo99", "f1_hi99", "match_rate", "gap", "gap_lo99", "gap_hi99"], rows)

    _bands(cells, os.path.join(out, "development_bands.tsv"))
    _table1(pl.read_parquet(os.path.join(dev, "keep.parquet")), os.path.join(out, "development_table1_matched.tsv"))


def _pct(xs, q):
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _bands(cells, path):
    """Resample the counties with replacement (Python's Mersenne Twister, string seed), percentile 99% interval."""
    sel = cells.filter((pl.col("lag") == 0) & ((pl.col("policy") == "never_rebuilt") | ((pl.col("policy") == "never_rebuilt_name_check") & (pl.col("t") == 2016))))
    g = _group(sel, ["t", "policy", "county0", "k"], ["n", "n_m", "f1c"])
    by = {}
    for (t, pol, c, k), s in g.items():
        by.setdefault((t, pol), {}).setdefault(c, {})[k] = (s["n"], s["n_m"], s["f1c"])
    rng = random.Random(BANDS_SEED)
    out = []
    for (t, pol), byc in sorted(by.items()):
        cs = sorted(byc)

        def st(sample):
            res = {}
            for k in (2, 4, 6, 8):
                n = math.fsum(byc[c][k][0] for c in sample)
                mr = math.fsum(byc[c][k][1] for c in sample) / n
                f1 = math.fsum(byc[c][k][2] for c in sample) / n
                res[k] = (mr, f1, mr - f1)
            return res

        point = st(cs)
        boots = [st([rng.choice(cs) for _ in cs]) for _ in range(BANDS_REPS)]
        for k in (2, 4, 6, 8):
            for j, name in enumerate(("match_rate", "f1_comember", "gap")):
                vals = [b[k][j] for b in boots]
                out.append([t, pol, k, name, _r4(point[k][j]), _r4(_pct(vals, 0.005)), _r4(_pct(vals, 0.995))])
        widen = [b[8][2] - b[2][2] for b in boots]
        out.append([t, pol, "8-2", "gap_widening", _r4(point[8][2] - point[2][2]), _r4(_pct(widen, 0.005)), _r4(_pct(widen, 0.995))])
    _w(path, ["t", "policy", "k", "stat", "point", "lo99", "hi99"], out)


def _table1(keep, path):
    r = keep.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
    rows = []
    for k in (2, 4, 6, 8):
        x = r.filter(pl.col("k") == k)
        m = x.filter(pl.col("matched"))
        f1 = sum((Fraction(a, b) for a, b in m.select("f1c_n", "f1c_d").iter_rows()), Fraction(0))
        exact = m.filter(pl.col("f1c_n") == pl.col("f1c_d")).height
        u = x.filter(~pl.col("matched"))
        rows.append([k, _r4(float(f1 / m.height)), _r4(exact / m.height), _r4(u.filter(pl.col("t_size") == 1).height / u.height)])
    _w(path, ["k", "f1_given_match", "exact_household", "alone_among_unmatched"], rows)
