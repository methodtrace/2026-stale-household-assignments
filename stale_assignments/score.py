# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline steps 4 and 5: per-record scores and county cells.

Per record, for persons with truth (has_truth), with ac = |A| - self, ic = |A and T| - self, tc = |T| - 1 (0 when unmatched):
  f1c (co-member F1, primary) = 1 if tc = 0 and ac = 0; 0 if tc = 0 or ac = 0 or ic = 0; else 2 ic / (ac + tc);
  f1 (self-inclusive) = 2 |A and T| / (|A| + |T|) when matched, else 0;
  fm = ac - ic, fs = tc - ic (false merges and splits); f1c_u = co-member F1 against the readdressed truth.
Policies: the step 2 rows; never_rebuilt_name_check (matched = name_hit), last_known (own cluster at t) and everyone_alone
from the never_rebuilt lag 0 rows; member_tracker, address_held and member_tracker_keep from step 3.
src (t = 2016 rows only): mover_together (mover in moved_together), mover_split (mover in split), stayer_changed (stayer in
grow, shrink or shrink_grow), stayer_split (stayer in split), format_change, stayer_stable (stayer in stable), else other.
bal: the client has truth at every k in {2, 4, 6, 8} for its t.

Scores are rationals. Every score is kept as an integer numerator and denominator and each county cell is summed exactly,
then rounded once to the nearest double, so cell values do not depend on row order, thread count or platform.
"""
from fractions import Fraction

import polars as pl

from .common import T0

KEYS = ["t", "policy", "lag", "k", "county0", "bal"]
COLS = ["t_dt", "ncid", "county0", "policy", "lag", "k", "has_truth", "matched", "self_in_a", "t_size", "a_size", "a_int",
        "t_size_u", "a_int_u"]
KEEP_POLICIES = ("never_rebuilt", "last_known", "annual_rebuild")  # t = 2016, lag 0 rows kept whole for steps 9, 11 and 12


def derived_rows(rm: pl.DataFrame) -> list[pl.DataFrame]:
    """The step 4 union for one step 2 pair (or the step 3 rows); each frame has COLS."""
    out = [rm.select(COLS)]
    if rm["policy"][0] == "never_rebuilt" and rm["lag"][0] == 0 and "f_size" in rm.columns:
        out.append(rm.select(
            "t_dt", "ncid", "county0", pl.lit("never_rebuilt_name_check").alias("policy"), "lag", "k", "has_truth",
            pl.col("name_hit").alias("matched"), (pl.col("self_in_a") & pl.col("name_hit")).alias("self_in_a"), "t_size",
            pl.when(pl.col("name_hit")).then(pl.col("a_size")).otherwise(None).alias("a_size"),
            pl.when(pl.col("name_hit")).then(pl.col("a_int")).otherwise(0).alias("a_int"), "t_size_u",
            pl.when(pl.col("name_hit")).then(pl.col("a_int_u")).otherwise(0).alias("a_int_u")))
        out.append(rm.select(
            "t_dt", "ncid", "county0", pl.lit("last_known").alias("policy"), "lag", "k", "has_truth", pl.lit(True).alias("matched"),
            pl.lit(True).alias("self_in_a"), "t_size", pl.col("f_size").alias("a_size"), pl.col("f_int").alias("a_int"),
            "t_size_u", pl.col("f_int_u").alias("a_int_u")))
        one = pl.when(pl.col("has_truth")).then(1).otherwise(0).cast(pl.Int64)
        out.append(rm.select(
            "t_dt", "ncid", "county0", pl.lit("everyone_alone").alias("policy"), "lag", "k", "has_truth", pl.lit(True).alias("matched"),
            pl.lit(True).alias("self_in_a"), "t_size", pl.lit(1, dtype=pl.Int64).alias("a_size"), one.alias("a_int"), "t_size_u",
            one.alias("a_int_u")))
    return out


def _f1c(tc, ac, ic):
    """(numerator, denominator) of co-member F1 with SQL CASE semantics (a null condition falls through)."""
    c1 = (tc == 0) & (ac == 0)
    c2 = (tc == 0) | (ac == 0) | (ic == 0)
    num = pl.when(c1).then(1).when(c2).then(0).otherwise(2 * ic)
    den = pl.when(c1).then(1).when(c2).then(1).otherwise(ac + tc)
    return num, den


def score(r: pl.DataFrame, bal: pl.DataFrame, ev: pl.DataFrame | None) -> pl.DataFrame:
    """Step 4 for a frame of union rows. bal: (ncid, bal) for this t; ev: (ncid, k, event_list, people_event, src) at t = 2016."""
    m, s = pl.col("matched"), pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    r = r.with_columns(
        pl.when(m).then(pl.col("a_size") - s).otherwise(0).alias("ac"),
        pl.when(m).then(pl.col("a_int") - s).otherwise(0).alias("ic"),
        pl.when(m).then(pl.col("a_int_u") - s).otherwise(0).alias("ic_u"),
        (pl.col("t_size") - 1).alias("tc"), (pl.col("t_size_u") - 1).alias("tc_u"))
    ht = pl.col("has_truth")
    tc, ac, ic, tcu, icu = (pl.col(c) for c in ("tc", "ac", "ic", "tc_u", "ic_u"))
    n1, d1 = _f1c(tc, ac, ic)
    nu, du = _f1c(tcu, ac, icu)
    f1ok = m & (pl.col("a_int") > 0)
    hu = ht & tcu.is_not_null()
    r = r.with_columns(
        pl.when(ht).then(n1).alias("f1c_n"), pl.when(ht).then(d1).alias("f1c_d"),
        pl.when(ht).then(pl.when(f1ok).then(2 * pl.col("a_int")).otherwise(0)).alias("f1_n"),
        pl.when(ht).then(pl.when(f1ok).then(pl.col("a_size") + pl.col("t_size")).otherwise(1)).alias("f1_d"),
        pl.when(ht).then(ac - ic).alias("fm"), pl.when(ht).then(tc - ic).alias("fs"),
        pl.when(hu).then(nu).alias("f1cu_n"), pl.when(hu).then(du).alias("f1cu_d"))
    r = r.drop("ac", "ic", "ic_u", "tc", "tc_u").join(bal, on="ncid", how="left").with_columns(pl.col("bal").fill_null(False))
    if ev is not None and r["t_dt"][0] == T0:
        r = r.join(ev, on=["ncid", "k"], how="left")
    else:
        r = r.with_columns(*(pl.lit(None, dtype=pl.String).alias(c) for c in ("event_list", "people_event", "src")))
    return r


def events_src(pe: pl.DataFrame, he: pl.DataFrame) -> pl.DataFrame:
    p = pe.filter(pl.col("client")).select("ncid", "k", "event_list", "dp0")
    j = p.join(he.select("dp0", "k", "people_event"), on=["dp0", "k"], how="inner")
    el, hp = pl.col("event_list"), pl.col("people_event")
    mv = el.str.starts_with("move")
    src = (pl.when(mv & (hp == "moved_together")).then(pl.lit("mover_together"))
           .when(mv & (hp == "split")).then(pl.lit("mover_split"))
           .when((el == "stay") & (hp == "split")).then(pl.lit("stayer_split"))
           .when((el == "stay") & hp.is_in(["grow", "shrink", "shrink_grow"])).then(pl.lit("stayer_changed"))
           .when(el == "format_change").then(pl.lit("format_change"))
           .when((el == "stay") & (hp == "stable")).then(pl.lit("stayer_stable")).otherwise(pl.lit("other")))
    return j.select("ncid", "k", "event_list", "people_event", src.alias("src"))


# ---- step 5: county cells, summed exactly --------------------------------------------------------------------------------

RATIONAL = {  # cell column: (row filter or None, numerator, denominator) on scored rows
    "f1c": (None, "f1c_n", "f1c_d"),
    "f1": (None, "f1_n", "f1_d"),
    "f1c_u": (None, "f1cu_n", "f1cu_d"),
    "loss": (None, "1-f1c", "f1c_d"),
    "loss1": (None, "1-f1", "f1_d"),
    "loss_mover_together": ("mover_together", "1-f1c", "f1c_d"),
    "loss_mover_split": ("mover_split", "1-f1c", "f1c_d"),
    "loss1_mover_split": ("mover_split", "1-f1", "f1_d"),
    "loss1_mov": ("MOV", "1-f1", "f1_d"),
}
CELL_COLS = ["t", "policy", "lag", "k", "county0", "bal", "n_all", "n", "n_m_all", "n_m", "f1c", "f1", "f1c_u", "n_u", "fm9", "fs9",
             "loss", "loss1", "loss_mover_together", "loss_mover_split", "loss1_mover_split", "loss1_mov", "n_mov", "n_mover_split",
             "n_alone"]


def rsum(df: pl.DataFrame, keys, num: str, den: str, name: str) -> pl.DataFrame:
    """Exact SQL SUM of the rationals num / den per group, rounded once to the nearest double; rows with a null numerator
    are skipped and a group with none is absent (callers left-join, which gives SQL's null)."""
    g = (df.filter(pl.col(num).is_not_null()).group_by(*keys, den).agg(pl.col(num).sum().alias(num)))
    acc = {}
    for row in g.iter_rows():
        key, d, n = row[:-2], row[-2], row[-1]
        acc[key] = acc.get(key, Fraction(0)) + Fraction(n, d)
    schema = {**{k: df.schema[k] for k in keys}, name: pl.Float64}
    return pl.DataFrame([list(k) + [float(v)] for k, v in acc.items()], schema=schema, orient="row")


def _isum(e: pl.Expr) -> pl.Expr:
    """SQL SUM over integers: null when every input is null."""
    return pl.when(e.count() == 0).then(None).otherwise(e.sum())


def cells(sc: pl.DataFrame) -> pl.DataFrame:
    """Step 5 for scored rows that hold every row of their cells."""
    sc = sc.with_columns(pl.col("t_dt").dt.year().cast(pl.Int64).alias("t"),
                         (pl.col("f1c_d") - pl.col("f1c_n")).alias("1-f1c"), (pl.col("f1_d") - pl.col("f1_n")).alias("1-f1"))
    ht, src = pl.col("has_truth"), pl.col("src")
    cnt = lambda e: e.fill_null(False).sum().cast(pl.Int64)
    base = sc.group_by(KEYS).agg(
        pl.len().cast(pl.Int64).alias("n_all"), cnt(ht).alias("n"), cnt(pl.col("matched")).alias("n_m_all"),
        cnt(ht & pl.col("matched")).alias("n_m"), cnt(ht & pl.col("f1cu_n").is_not_null()).alias("n_u"),
        _isum(pl.min_horizontal(pl.col("fm"), pl.lit(9)).filter(pl.col("fm").is_not_null())).alias("fm9"),
        _isum(pl.min_horizontal(pl.col("fs"), pl.lit(9)).filter(pl.col("fs").is_not_null())).alias("fs9"),
        cnt(ht & src.is_in(["mover_together", "mover_split"])).alias("n_mov"), cnt(ht & (src == "mover_split")).alias("n_mover_split"),
        cnt(ht & (pl.col("t_size") == 1)).alias("n_alone"))
    for name, (flt, num, den) in RATIONAL.items():
        if flt is None:
            sel = pl.lit(True)
        elif flt == "MOV":
            sel = src.is_in(["mover_together", "mover_split"]).fill_null(False)
        else:
            sel = (src == flt).fill_null(False)
        # IF(sel, value, 0): rows outside the filter add an exact 0 and still count as non-null
        part = sc.select(*KEYS, pl.when(sel).then(pl.col(num)).otherwise(0).alias("_n"),
                         pl.when(sel).then(pl.col(den)).otherwise(1).alias("_d"))
        vals = rsum(part, KEYS, "_n", "_d", name)
        base = base.join(vals, on=KEYS, how="left", nulls_equal=True)
    return base.select(CELL_COLS)


def sort_cells(c: pl.DataFrame) -> pl.DataFrame:
    return c.sort(KEYS, nulls_last=False)
