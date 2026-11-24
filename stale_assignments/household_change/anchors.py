# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Household-change model data (paper section 8.2): anchors and their build households, the anchor state at later
snapshots, and the scoring cells.

Anchors: trackable persons (voter unit key) at each Election Day build t in 2008 to 2016 whose county at t is in the run's
list, minors at t excluded. Household at t = the anchor's own cluster in the voter-key graph (smallest key). State at s:
present = anchor has a reference key at s; stay = its reference key at s equals the one at t; n_keep = co-members at t with
the anchor's key at s; exact = every co-member kept and nobody gained (alone at t and alone at s included). Composition
cell = the anchor's age band, ':', the co-members' age bands in order; building type S (SFH_LIKELY) or M. Only the 40
registered snapshots are read.
"""
import csv
import datetime as dt
import os

import polars as pl

from .. import store as S

I = pl.Int64
BUILDS = [d for d in S.SNAPSHOTS if S.series_of(d) == "eday" and d.year <= 2016]


def band(a):
    e = pl.when(a.is_null()).then(pl.lit("Z"))
    for c, b in ((22, "A"), (26, "B"), (30, "C"), (35, "D"), (45, "E"), (55, "F"), (65, "G"), (75, "H")):
        e = e.when(a < c).then(pl.lit(b))
    return e.otherwise(pl.lit("I"))


def members(st: S.Store, counties, t) -> pl.DataFrame:
    """One row per anchor and co-member at t (mem null for anchors recorded alone)."""
    minor = st.read("age_flags").filter((pl.col("snapshot_dt") == t) & pl.col("minor")).select("ncid").unique()
    tags = st.read("tags", t, columns=["ncid", "unit_key", "trackable_voter", "age"])
    a = (st.read("addr", t, columns=["ncid", "unit_key", "county_id", "is_ai"]).filter(pl.col("is_ai").fill_null(False))
         .filter(pl.col("county_id").is_in(list(counties))))
    univ = (tags.filter(pl.col("trackable_voter")).join(a, on=["ncid", "unit_key"], how="inner").join(minor, on="ncid", how="anti")
            .group_by("ncid").agg(pl.col("county_id").min().alias("county0")))
    g = st.read("graph", t, columns=["ncid", "akey", "tk"])
    g1 = g.group_by("ncid").agg(pl.col("akey").min().alias("akey0"), pl.col("tk").min().alias("tk0"))
    gm = g.select(pl.col("akey").alias("akey0"), pl.col("ncid").alias("mem")).unique()
    ag = tags.group_by("ncid").agg(pl.col("age").min())
    u = univ.join(g1, on="ncid", how="inner").join(ag.rename({"age": "age_i"}), on="ncid", how="left")
    m = (u.join(gm, on="akey0", how="left").with_columns(
        pl.when(pl.col("mem") == pl.col("ncid")).then(None).otherwise(pl.col("mem")).alias("mem")))
    # anchors whose only member is themselves keep one row with mem null; others drop the self row
    has_other = m.group_by("ncid").agg(pl.col("mem").is_not_null().any().alias("ho"))
    m = m.join(has_other, on="ncid").filter(pl.col("mem").is_not_null() | ~pl.col("ho")).drop("ho").unique()
    m = m.join(ag.rename({"ncid": "mem", "age": "age_m"}), on="mem", how="left")
    return m.select(pl.lit(t).alias("t_dt"), "ncid", "county0", "age_i", "akey0", "tk0", "mem", "age_m")


def comp(st: S.Store, mem: pl.DataFrame, t) -> pl.DataFrame:
    """(ncid, county0, cell, bt) per anchor at t."""
    b = band(pl.col("age_m"))
    c = (mem.with_columns(pl.when(pl.col("mem").is_null()).then(None).otherwise(b).alias("bm"))
         .sort(["ncid", "bm"], nulls_last=True)
         .group_by("ncid", maintain_order=True).agg(pl.col("county0").first(), pl.col("age_i").first(),
                                                    pl.col("bm").drop_nulls().str.join("").alias("bms")))
    c = c.select("ncid", "county0", pl.concat_str([band(pl.col("age_i")), pl.lit(":"), pl.col("bms")]).alias("cell"))
    bt = (st.read("tags", t, columns=["ncid", "voter_tag"]).group_by("ncid")
          .agg(pl.when(pl.col("voter_tag") == "SFH_LIKELY").then(pl.lit("S")).otherwise(pl.lit("M")).min().alias("bt")))
    return c.join(bt, on="ncid", how="left").with_columns(pl.col("bt").fill_null("M"))


def state(st: S.Store, mem: pl.DataFrame, s, ts=None) -> pl.DataFrame:
    """Anchor state at s: (ncid, m, present, stay, n_keep, n_gone, n_gain, exact)."""
    if ts is None:
        ts = st.read("graph", s, columns=["ncid", "tk"]).group_by("ncid").agg(pl.col("tk").min())
    ts = ts.filter(pl.col("tk").is_not_null())
    tsize = ts.group_by("tk").agg(pl.len().cast(I).alias("nz"))
    anc = mem.group_by("ncid").agg(pl.col("tk0").first(), pl.col("mem").is_not_null().sum().cast(I).alias("m"))
    a = anc.join(ts.rename({"tk": "tk_s"}), on="ncid", how="left")
    mm = (mem.filter(pl.col("mem").is_not_null()).select("ncid", "mem")
          .join(a.select("ncid", "tk_s"), on="ncid", how="inner")
          .join(ts.rename({"ncid": "mem", "tk": "tk_m"}), on="mem", how="left")
          .group_by("ncid").agg((pl.col("tk_m") == pl.col("tk_s")).fill_null(False).sum().cast(I).alias("n_keep"),
                                pl.col("tk_m").is_null().sum().cast(I).alias("n_gone")))
    x = a.join(mm, on="ncid", how="left").join(tsize.rename({"tk": "tk_s"}), on="tk_s", how="left").with_columns(
        pl.col("n_keep").fill_null(0), pl.col("n_gone").fill_null(0))
    pres = pl.col("tk_s").is_not_null()
    gain = pl.col("nz") - 1 - pl.col("n_keep")
    return x.select("ncid", "m", pres.alias("present"), (pl.col("tk_s") == pl.col("tk0")).fill_null(False).alias("stay"),
                    "n_keep", "n_gone", pl.when(pres).then(gain).otherwise(None).alias("n_gain"),
                    pl.when(pres).then((pl.col("n_keep") == pl.col("m")) & (gain == 0)).otherwise(None).alias("exact"))


def score_cells(st: S.Store, counties) -> pl.DataFrame:
    """Scoring cells: per (county0, t_dt, k, cell, bt) counts of listed anchors by observed state at the Election Days
    700 to 3,700 days after the build."""
    ed = [d for d in S.SNAPSHOTS if S.series_of(d) == "eday"]
    out = []
    for t in BUILDS:
        mem = members(st, counties, t)
        c = comp(st, mem, t)
        for s in ed:
            days = (s - t).days
            if not 700 <= days <= 3700:
                continue
            x = state(st, mem, s).join(c, on="ncid", how="inner").filter(pl.col("present"))
            ex, sy = pl.col("exact").fill_null(False), pl.col("stay")
            cnt = lambda e: e.sum().cast(I)
            out.append(x.group_by("county0", "cell", "bt").agg(
                pl.len().cast(I).alias("n"), cnt(ex & sy).alias("nEB"), cnt(ex & ~sy).alias("nEO"), cnt(~ex & sy).alias("nXB"),
                cnt(~ex & ~sy).alias("nXO")).with_columns(pl.lit(t).alias("t_dt"), pl.lit(int(days / 365.25 + 0.5), dtype=I).alias("k")))
    r = pl.concat(out).filter(pl.col("n") > 0)
    return r.select("county0", "t_dt", "k", "cell", "bt", "n", "nEB", "nEO", "nXB", "nXO")


def stayer_xtab(st: S.Store, rm0: pl.LazyFrame, counties, t=None) -> pl.DataFrame:
    """stayer_xtab_2016.csv: the address re-match stayer flag against the readdressing-rule stayer flag (state at s), main
    build, policy never_rebuilt, lag 0, persons with truth; exact = exact under the readdressing rule at s."""
    t = t or BUILDS[-1]
    r = (rm0.filter((pl.col("t_dt") == t) & (pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
         .select("k", "ncid", "s_dt", "mkey", "last_known_key").collect())
    mem = members(st, counties, t)
    out = []
    for s in sorted(r["s_dt"].unique().to_list()):
        x = r.filter(pl.col("s_dt") == s).join(state(st, mem, s), on="ncid", how="inner")
        out.append(x.group_by("k", (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay_unitkey"),
                              pl.col("stay").alias("stay_readdress"), pl.col("mkey").is_null().alias("no_rematch_key"))
                   .agg(pl.len().cast(I).alias("n"), pl.col("exact").fill_null(False).sum().cast(I).alias("exact")))
    return pl.concat(out).sort(["k", "stay_unitkey", "stay_readdress", "no_rematch_key"])


def _within(t, months=122):
    """The registered snapshots s with t < s <= t + months (calendar months, the day clamped to the month's end)."""
    y, m = divmod(t.month - 1 + months, 12)
    y, m = t.year + y, m + 1
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    end = dt.date(y, m, min(t.day, (nxt - dt.timedelta(days=1)).day))
    return [s for s in S.SNAPSHOTS if t < s <= end]


def snapshot_rows(builds=None):
    """(t_dt, n, s_dt, u): the snapshots after each build that the fit reads, numbered from 1; u = days / 365.25."""
    out = []
    for t in builds or BUILDS:
        for i, s in enumerate(_within(t), 1):
            out.append((t.isoformat(), i, s.isoformat(), (s - t).days / 365.25))
    return out


def write_snapshots(path, builds=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t_dt", "n", "s_dt", "u"])
        for r in snapshot_rows(builds):
            w.writerow([r[0], r[1], r[2], repr(r[3])])


def transitions(st: S.Store, counties) -> pl.DataFrame:
    """The fitting panel: per (t_dt, cell, bt, j, code) the anchors whose state moved from c0 at snapshot j - 1 (the build
    for j = 1, state EB) to c1 at snapshot j, code 'c0-c1'; states EB, EO, XB, XO (listed: exact or not, same reference key
    as at t or not) and U (not listed). Snapshots: every registered one up to 122 months after the build."""
    out = []
    for t in BUILDS:
        mem = members(st, counties, t)
        c = comp(st, mem, t)
        prev = None
        for j, s in enumerate(_within(t), 1):
            x = state(st, mem, s)
            code = (pl.when(pl.col("present"))
                    .then(pl.concat_str([pl.when(pl.col("exact")).then(pl.lit("E")).otherwise(pl.lit("X")),
                                         pl.when(pl.col("stay")).then(pl.lit("B")).otherwise(pl.lit("O"))]))
                    .otherwise(pl.lit("U")))
            cur = x.select("ncid", code.alias("c1"))
            y = cur.join(prev, on="ncid", how="left") if prev is not None else cur.with_columns(pl.lit(None, dtype=pl.String).alias("c0"))
            y = y.with_columns(pl.col("c0").fill_null("EB"))
            out.append(y.join(c.select("ncid", "cell", "bt"), on="ncid", how="inner")
                       .group_by("cell", "bt", pl.concat_str(["c0", pl.lit("-"), "c1"]).alias("code"))
                       .agg(pl.len().cast(I).alias("n"))
                       .with_columns(pl.lit(t).alias("t_dt"), pl.lit(j, dtype=I).alias("j")))
            prev = cur.rename({"c1": "c0"})
    return pl.concat(out).select("t_dt", "cell", "bt", "j", "code", "n").sort(["t_dt", "cell", "bt", "j", "code"])
