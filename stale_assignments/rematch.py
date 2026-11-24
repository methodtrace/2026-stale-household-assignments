# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline step 2: simulated vendor graphs and the re-match matcher (address-only builder).

Universes: t = 2016-11-08 (step 1 clients) for every policy; t in {2008, 2010, 2012, 2014} Election Days for the graph-age
replication (policy never_rebuilt only), each built like step 1's universe at its own t. Pairs (t, policy, lag, b, s), s an Election
Day 2 to 8 years after t: never_rebuilt = graph at t; lagged_rebuild = graph at the Election Day before s;
same_date_rebuild = graph at s; four_year_rebuild = latest of t, t + 4, t + 8 not after s; annual_rebuild = latest January 1
snapshot on or before s; lag 1 (client file one Election Day earlier) for never_rebuilt and same_date_rebuild at t = 2016. CRM record: the person's latest Election Day graph record from t to s minus
lag (smallest key on a tie). The record matches iff its address key exists in the graph at b; name_hit: the same first-name
and surname fingerprints at that key. Reference at s: the smallest readdressed key of the person's records at s.
"""
import gc

import polars as pl

from . import store as S
from . import config
from .common import T0, client_flag

TDATES = sorted(set(config.AUX) | {T0})  # 2008 to 2016 in the main design
POLICIES = ["never_rebuilt", "lagged_rebuild", "same_date_rebuild", "four_year_rebuild", "annual_rebuild"]


def pairs():
    """Step 2's (t, policy, lag, s, crm, b) pairs. The main build pairs with the next MAX_ED Election Days, the others with
    the next four (STALE_MAX_ED). With SCORE_PAIRS (STALE_SCORE_PAIRS) exactly the listed pairs, policy never_rebuilt,
    lag 0, CRM record and graph as for never."""
    if config.SCORE_PAIRS:
        assert all(t in TDATES and s > t for t, s in config.SCORE_PAIRS), "score pairs: t must be a build of the run, s after t"
        return [dict(t_dt=t, policy="never_rebuilt", lag=0, s_dt=s, crm_dt=s, b_dt=t) for t, s in config.SCORE_PAIRS]
    ed = S.EDAYS
    out = []
    for t in TDATES:
        ti = ed.index(t)
        w = config.MAX_ED if t == T0 else 4
        for si in range(ti + 1, min(ti + w, len(ed) - 1) + 1):
            for pol in POLICIES:
                for lag in (0, 1):
                    if t == T0:
                        if not (lag == 0 or pol in ("never_rebuilt", "same_date_rebuild")):
                            continue
                    elif not (pol == "never_rebuilt" and lag == 0):
                        continue
                    s = ed[si]
                    b = {"never_rebuilt": t, "lagged_rebuild": ed[si - 1], "same_date_rebuild": s, "four_year_rebuild": ed[ti + ((si - ti) // 2) * 2]}.get(pol)
                    if pol == "annual_rebuild":
                        b = max(j for j in S.JAN1 if j <= s)
                    out.append(dict(t_dt=t, policy=pol, lag=lag, s_dt=s, crm_dt=ed[si - lag], b_dt=b))
    return out


def universe(st: S.Store, t, counties, sample, pe=None) -> pl.DataFrame:
    if t == T0:
        return pe.filter(pl.col("client")).group_by("ncid").agg(pl.col("county0").first()).sort("ncid")
    minor = st.read("age_flags").filter((pl.col("snapshot_dt") == t) & pl.col("minor")).select("ncid").unique()
    a = st.read("addr", t).filter(pl.col("is_ai").fill_null(False)).select("ncid", "unit_key", "county_id")
    h = st.read("tags", t).filter(pl.col("trackable_voter")).select("ncid", "unit_key")
    u = (h.join(a, on=["ncid", "unit_key"], how="inner").filter(pl.col("county_id").is_in(list(counties)))
         .join(minor, on="ncid", how="anti"))
    u = u.filter(client_flag(u["ncid"], sample))
    return u.group_by("ncid").agg(pl.col("county_id").min().alias("county0")).sort("ncid")


class Graph:
    """Per-snapshot lookups from the graph records, computed once per snapshot."""

    def __init__(self, st: S.Store):
        self.st = st
        self._cache = {}

    def _get(self, d):
        if d in self._cache:
            self._cache[d] = self._cache.pop(d)  # least recently used goes first
        else:
            if len(self._cache) >= 3:
                self._cache.pop(next(iter(self._cache)))
            g = self.st.read("graph", d, columns=["ncid", "first_fp", "sur_fp", "akey", "tk"])
            truth = g.group_by("ncid").agg(pl.col("tk").min().alias("tkey"), pl.col("akey").min().alias("ckey"))
            tsize = truth.group_by("tkey").agg(pl.len().cast(pl.Int64).alias("t_size"))
            mem = g.select("akey", "ncid").unique()
            csize = mem.group_by("akey").agg(pl.len().cast(pl.Int64).alias("n"))
            names = g.select("akey", "first_fp", "sur_fp").unique()
            self._cache[d] = dict(truth=truth, tsize=tsize, mem=mem, csize=csize, names=names)
            del g
        return self._cache[d]

    def __getitem__(self, d):
        return self._get(d)


def crm_records(st: S.Store, ncids: pl.DataFrame, dates) -> pl.DataFrame:
    """Each person's record at each Election Day in dates: smallest key, then first-name and surname fingerprints."""
    out = []
    for d in dates:
        g = (st.read("graph", d, columns=["ncid", "first_fp", "sur_fp", "akey"]).join(ncids, on="ncid", how="semi")
             .sort(["ncid", "akey", "first_fp", "sur_fp"], nulls_last=False)
             .group_by("ncid", maintain_order=True).first())
        out.append(g.with_columns(pl.lit(d).alias("rec_dt")))
    return pl.concat(out)


def cluster_overlap(G: Graph, b, s, keys: pl.DataFrame) -> pl.DataFrame:
    """For each needed key at b: its members grouped by their reference key at s, counted (members without one dropped)."""
    mem = G[b]["mem"].join(keys, on="akey", how="semi")
    return (mem.join(G[s]["truth"].select("ncid", "tkey"), on="ncid", how="inner")
            .group_by("akey", "tkey").agg(pl.len().cast(pl.Int64).alias("cnt")))


def balanced(st: S.Store, U: pl.DataFrame, sdates) -> pl.DataFrame:
    """(ncid, bal): the person has truth (a reference key) at every scoring date."""
    b = U.select("ncid").with_columns(pl.lit(0, dtype=pl.Int64).alias("n"))
    for s in sdates:
        has = st.read("graph", s, columns=["ncid", "tk"]).filter(pl.col("tk").is_not_null()).select("ncid").unique()
        b = b.join(has.with_columns(pl.lit(1, dtype=pl.Int64).alias("h")), on="ncid", how="left").with_columns(
            (pl.col("n") + pl.col("h").fill_null(0)).alias("n")).drop("h")
    return b.select("ncid", (pl.col("n") == len(sdates)).alias("bal"))


def rematch(st: S.Store, counties, sample, pe: pl.DataFrame, emit):
    """Compute step 2 for every pair. emit(t_dt, None, bal) is called once per universe with its balanced-panel flags,
    then emit(t_dt, pair, df) once per pair with that pair's rows."""
    G = Graph(st)
    P = pairs()
    U = rec = r = m = cm = cf = None
    for t in TDATES:
        # memory: release the previous build's universe, records and cached graphs before loading this build's
        U = rec = r = m = cm = cf = None
        G._cache.clear()
        gc.collect()
        U = universe(st, t, counties, sample, pe)
        pt = [p for p in P if p["t_dt"] == t]
        if not pt:
            continue
        last = max(p["crm_dt"] for p in pt)
        extra = {p["s_dt"] for p in pt} if config.SCORE_PAIRS else set()  # records at the listed s (STALE_SCORE_PAIRS)
        rec = crm_records(st, U.select("ncid"), sorted({d for d in S.EDAYS if t <= d <= last} | extra))
        emit(t, None, balanced(st, U, sorted({p["s_dt"] for p in pt})))
        for p in sorted(pt, key=lambda p: (p["b_dt"], p["s_dt"], p["policy"], p["lag"])):
            # the CRM record: the latest Election Day record from t to crm_dt (with score pairs also the record at s itself)
            ok = (pl.col("rec_dt").is_in(S.EDAYS) | (pl.col("rec_dt") == p["crm_dt"])) if config.SCORE_PAIRS else pl.lit(True)
            r = (rec.filter((pl.col("rec_dt") >= t) & (pl.col("rec_dt") <= p["crm_dt"]) & ok)
                 .sort(["ncid", "rec_dt"], descending=[False, True]).group_by("ncid", maintain_order=True).first())
            m = U.join(r, on="ncid", how="inner")
            b, s = p["b_dt"], p["s_dt"]
            gb, gs = G[b], G[s]
            m = (m.join(gs["truth"].select("ncid", "tkey"), on="ncid", how="left")
                 .join(gb["csize"].rename({"n": "a_size"}), on="akey", how="left")
                 .join(gb["names"].with_columns(pl.lit(True).alias("name_hit")), on=["akey", "first_fp", "sur_fp"], how="left")
                 .join(gb["truth"].select("ncid", pl.col("ckey").alias("fkey")), on="ncid", how="left"))
            m = m.with_columns(
                pl.col("tkey").is_not_null().alias("has_truth"),
                pl.when(pl.col("a_size").is_not_null()).then(pl.col("akey")).otherwise(None).alias("mkey"),
                pl.col("name_hit").fill_null(False))
            if p["policy"] == "never_rebuilt" and p["lag"] == 0:
                m = m.join(G[t]["truth"].select("ncid", pl.col("ckey").alias("last_known_key")), on="ncid", how="left")
            else:
                m = m.with_columns(pl.lit(None, dtype=pl.String).alias("last_known_key"))
            need = m.filter(pl.col("has_truth") & pl.col("mkey").is_not_null()).select(pl.col("mkey").alias("akey")).unique()
            cm = cluster_overlap(G, b, s, need)
            m = m.join(cm.rename({"akey": "mkey", "cnt": "a_int"}), on=["mkey", "tkey"], how="left")
            needf = m.filter(pl.col("has_truth") & pl.col("last_known_key").is_not_null()).select(pl.col("last_known_key").alias("akey")).unique()
            if needf.height:  # only the never_rebuilt, lag 0 rows carry a last-known key; other pairs skip the graph at t
                cf = cluster_overlap(G, t, s, needf)
                m = (m.join(cf.rename({"akey": "last_known_key", "cnt": "f_int"}), on=["last_known_key", "tkey"], how="left")
                     .join(G[t]["csize"].rename({"akey": "last_known_key", "n": "f_size"}), on="last_known_key", how="left"))
            else:
                m = m.with_columns(pl.lit(None, dtype=pl.Int64).alias("f_int"), pl.lit(None, dtype=pl.Int64).alias("f_size"))
            m = m.join(gs["tsize"], on="tkey", how="left")
            # a_int and f_int count only rows with truth (the specification joins them for has_truth rows)
            m = m.with_columns(
                pl.when(pl.col("has_truth")).then(pl.col("a_int")).otherwise(None).fill_null(0).alias("a_int"),
                pl.when(pl.col("has_truth")).then(pl.col("f_int")).otherwise(None).fill_null(0).alias("f_int"))
            emit(t, p, m.select(
                pl.lit(t).alias("t_dt"), "ncid", "county0", pl.lit(p["policy"]).alias("policy"), pl.lit(p["lag"], dtype=pl.Int64).alias("lag"),
                pl.lit(b).alias("b_dt"), pl.lit(s).alias("s_dt"), pl.lit(s.year - t.year, dtype=pl.Int64).alias("k"),
                pl.col("rec_dt").alias("crm_rec_dt"), "has_truth", pl.col("mkey").is_not_null().alias("matched"), "name_hit",
                (pl.col("fkey") == pl.col("mkey")).fill_null(False).alias("self_in_a"),
                "t_size", pl.col("t_size").alias("t_size_u"), "a_size", "a_int", pl.col("a_int").alias("a_int_u"),
                "f_size", "f_int", pl.col("f_int").alias("f_int_u"), "mkey", "last_known_key"))
