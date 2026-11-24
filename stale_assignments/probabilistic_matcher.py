# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline step 11: the probabilistic matcher, a Fellegi-Sunter baseline, prediction only with fixed parameters.

The m and u probabilities per k, comparison and level were estimated once on the development counties (EM in three
sessions) and are shipped in data/probabilistic_matcher_parameters.tsv; they are never re-estimated. load_params() checks
the file against the fingerprint recorded when they were fixed. The prior (1 / graph records at b) and the term-frequency shares come from the
statewide graph at b.

Model: name and address only. Levels fn 3 exact / 2 first three letters / 1 Soundex / 0, sn 2 exact / 1 Soundex / 0, mi
1 / 0, sx (suffix) 1 / 0, ad 4 unit / 3 building / 2 street and ZIP / 1 ZIP / 0; -1 = missing (no weight). Blocking: same
unit key, or exact first name and surname statewide, or surname Soundex, first-name Soundex and ZIP. Weight = log2 of the
prior odds plus log2(m / u) per comparison, with u replaced by the value's share of the graph for exact fn, sn and unit
(term frequency). The best candidate per client is the highest weight (ties: smallest fingerprint of
MATCHER_TIE_SALT + '|' + NCID (constants.py)); the match threshold is applied in step 12. NCID equality is used only to score.

log2 is computed in decimal arithmetic at 50 digits and rounded once to a double, so weights are the same on every
platform. Scope: build t = b = 2016-11-08, policy never_rebuilt, lag 0, k = 2 to 8; clients and CRM records are step 2's.
"""
import os
from decimal import Decimal, localcontext

import polars as pl

from . import record_functions as rf
from . import store as S
from .common import T0
from .constants import MATCHER_TIE_SALT

PARAMS_FP = 7503141542285507977
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOG2 = {}


def log2(x: float) -> float:
    """Correctly rounded (to within a 1e-50 relative error before the final rounding) base-2 logarithm of a double."""
    v = _LOG2.get(x)
    if v is None:
        if not x > 0:
            raise ValueError(f"log2 of {x!r}")
        with localcontext() as c:
            c.prec = 50
            v = float(Decimal(x).ln() / Decimal(2).ln())
        _LOG2[x] = v
    return v


def load_params(path=None) -> pl.DataFrame:
    """(k, comp, lvl, m, u, w) with w = log2(max(m, 1e-12) / u)."""
    path = path or os.path.join(_HERE, "data", "probabilistic_matcher_parameters.tsv")
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0].split("\t") == ["k", "comp", "lvl", "m", "u"], "probabilistic matcher parameter file header"
    rows, x = [], 0
    for ln in lines[1:]:
        k, comp, lvl, m, u = ln.split("\t")
        x ^= rf.fingerprint("|".join((k, comp, lvl, m, u)))
        rows.append((int(k), comp, int(lvl), float(m), float(u)))
    if len(rows) != 64 or x != PARAMS_FP:
        raise SystemExit("probabilistic matcher parameters have changed (fingerprint mismatch); they must not be re-estimated")
    p = pl.DataFrame(rows, schema={"k": pl.Int64, "comp": pl.String, "lvl": pl.Int64, "m": pl.Float64, "u": pl.Float64}, orient="row")
    return p.with_columns(pl.Series("w", [log2(max(m, 1e-12) / u) for m, u in zip(p["m"], p["u"])], dtype=pl.Float64))


def _nz(c):
    """NULLIF(UPPER(TRIM(x)), '')."""
    v = pl.col(c).str.strip_chars(" \t\n\r\x0b\x0c").str.to_uppercase()
    return pl.when(v == "").then(None).otherwise(v)


def _fields(a: pl.DataFrame) -> pl.DataFrame:
    return a.select(
        "ncid", "unit_key", pl.col("first_fp").alias("fn"), pl.col("first3_fp").alias("f3"), pl.col("first_sdx_fp").alias("fsx"),
        pl.col("surname_norm_fp").alias("sn"), pl.col("surname_sdx_fp").alias("ssx"), pl.col("mid_init_fp").alias("mi"),
        _nz("name_sufx").alias("sx"), "bldg_key",
        pl.concat_str([pl.col("street_name").fill_null("").str.to_uppercase(), pl.lit("|"), pl.col("zip5").fill_null("")]).alias("stz"),
        "zip5")


ADDR_COLS = ["ncid", "unit_key", "is_ai", "first_fp", "first3_fp", "first_sdx_fp", "surname_norm_fp", "surname_sdx_fp",
             "mid_init_fp", "name_sufx", "bldg_key", "street_name", "zip5"]


def _graph_fields(st: S.Store, d) -> pl.DataFrame:
    """Graph records at d joined to their active address rows (one row per graph row and matching address row)."""
    g = st.read("graph", d, columns=["ncid", "akey"]).with_columns(pl.col("akey").str.slice(1).cast(pl.Int64).alias("unit_key"))
    a = st.read("addr", d, columns=ADDR_COLS).filter(pl.col("is_ai").fill_null(False)).drop("is_ai")
    return g.join(_fields(a), on=["ncid", "unit_key"], how="inner")


def matcher_graph(st: S.Store) -> pl.DataFrame:
    return _graph_fields(st, T0)


def matcher_clients(st: S.Store, rm: pl.DataFrame, fg: pl.DataFrame) -> pl.DataFrame:
    """rm: step 2 rows for t = 2016, policy never_rebuilt, lag 0. One row per (ncid, k): the CRM record's fields at crm_rec_dt."""
    out = []
    own = fg.group_by("ncid").agg(pl.col("akey").min().alias("fkey"))
    for d in sorted(set(rm["crm_rec_dt"].drop_nulls().to_list())):
        r = rm.filter(pl.col("crm_rec_dt") == d).select("ncid", "county0", "s_dt", "k", "has_truth", "t_size", pl.col("matched").alias("det_matched"))
        g = _graph_fields(st, d).join(r.select("ncid").unique(), on="ncid", how="semi")
        x = r.join(g, on="ncid", how="inner")
        cols = ["akey", "fn", "f3", "fsx", "sn", "ssx", "mi", "sx", "unit_key", "bldg_key", "stz", "zip5"]
        x = x.sort(["ncid", "k"] + cols, nulls_last=False).group_by(["ncid", "k"], maintain_order=True).first()
        out.append(x.drop("akey"))
    c = pl.concat(out).join(own, on="ncid", how="left")
    return c.sort(["k", "ncid"])


def _levels(p: pl.DataFrame) -> pl.DataFrame:
    c, g = lambda x: pl.col(x), lambda x: pl.col(x + "_g")
    lfn = (pl.when(c("fn").is_null() | g("fn").is_null()).then(-1).when(c("fn") == g("fn")).then(3)
           .when(c("f3") == g("f3")).then(2).when(c("fsx") == g("fsx")).then(1).otherwise(0))
    lsn = (pl.when(c("sn").is_null() | g("sn").is_null()).then(-1).when(c("sn") == g("sn")).then(2)
           .when(c("ssx") == g("ssx")).then(1).otherwise(0))
    lmi = pl.when(c("mi").is_null() | g("mi").is_null()).then(-1).when(c("mi") == g("mi")).then(1).otherwise(0)
    lsx = pl.when(c("sx").is_null() & g("sx").is_null()).then(-1).when(c("sx") == g("sx")).then(1).otherwise(0)
    lad = (pl.when(c("unit_key") == g("unit_key")).then(4).when(c("bldg_key") == g("bldg_key")).then(3)
           .when(c("stz") == g("stz")).then(2).when(c("zip5") == g("zip5")).then(1).otherwise(0))
    return p.with_columns(lfn.cast(pl.Int64).alias("lfn"), lsn.cast(pl.Int64).alias("lsn"), lmi.cast(pl.Int64).alias("lmi"),
                          lsx.cast(pl.Int64).alias("lsx"), lad.cast(pl.Int64).alias("lad"))


class Weights:
    def __init__(self, fg: pl.DataFrame, params: pl.DataFrame):
        self.ng = fg.height
        self.ng_fn = fg.filter(pl.col("fn").is_not_null()).height
        prior = 1 / self.ng
        self.w0 = log2(prior / (1 - prior))
        self.tfn = fg.filter(pl.col("fn").is_not_null()).group_by("fn").agg(pl.len().cast(pl.Int64).alias("n_fn"))
        self.tsn = fg.group_by("sn").agg(pl.len().cast(pl.Int64).alias("n_sn"))
        self.tad = fg.group_by("unit_key").agg(pl.len().cast(pl.Int64).alias("n_ad"))
        self.p = {(k, comp, lvl): (m, w) for k, comp, lvl, m, u, w in params.iter_rows()}

    def _term(self, comp, k, lvl, n, exact_lvl, total):
        if (k, comp, lvl) not in self.p:
            return None
        m, w = self.p[(k, comp, lvl)]
        if lvl == exact_lvl:
            return None if n is None else log2(max(m, 1e-12) / (n / total))
        return w

    def add(self, p: pl.DataFrame) -> pl.DataFrame:
        p = (p.join(self.tfn, left_on="fn", right_on="fn", how="left").join(self.tsn, left_on="sn", right_on="sn", how="left")
             .join(self.tad, left_on="unit_key", right_on="unit_key", how="left"))
        p = p.with_columns(pl.when(pl.col("lfn") == 3).then(pl.col("n_fn")).alias("n_fn"),
                           pl.when(pl.col("lsn") == 2).then(pl.col("n_sn")).alias("n_sn"),
                           pl.when(pl.col("lad") == 4).then(pl.col("n_ad")).alias("n_ad"))
        terms = []
        for comp, lc, nc, ex, tot in (("fn", "lfn", "n_fn", 3, self.ng_fn), ("sn", "lsn", "n_sn", 2, self.ng),
                                      ("mi", "lmi", None, None, None), ("sx", "lsx", None, None, None), ("ad", "lad", "n_ad", 4, self.ng)):
            keys = [lc] + ([nc] if nc else [])
            u = p.select("k", *keys).unique()
            vals = [self._term(comp, r[0], r[1], r[2] if nc else None, ex, tot) for r in u.iter_rows()]
            u = u.with_columns(pl.Series(f"t_{comp}", vals, dtype=pl.Float64))
            p = p.join(u, on=["k"] + keys, how="left", nulls_equal=True)
            terms.append(pl.col(f"t_{comp}").fill_null(0.0))
        w = pl.lit(self.w0, dtype=pl.Float64)
        for t in terms:  # summed left to right, as written in the specification
            w = w + t
        return p.with_columns(w.alias("w")).drop("n_fn", "n_sn", "n_ad", *(f"t_{c}" for c in ("fn", "sn", "mi", "sx", "ad")))


def best(c: pl.DataFrame, fg: pl.DataFrame, W: Weights, chunks: int = 16) -> pl.DataFrame:
    """Highest-weight candidate per (k, client), with candidate counts and the client's own weight."""
    gcols = ["ncid", "akey", "fn", "f3", "fsx", "sn", "ssx", "mi", "sx", "unit_key", "bldg_key", "stz", "zip5"]
    g = fg.select(gcols)
    ids = c.select("ncid").unique().sort("ncid")["ncid"]
    edges = [ids[int(len(ids) * i / chunks)] for i in range(1, chunks)] if len(ids) else []
    out = []
    for i in range(chunks):
        lo = edges[i - 1] if i > 0 else None
        hi = edges[i] if i < chunks - 1 else None
        cc = c
        if lo is not None:
            cc = cc.filter(pl.col("ncid") >= lo)
        if hi is not None:
            cc = cc.filter(pl.col("ncid") < hi)
        if cc.height == 0:
            continue
        cand = pl.concat([
            cc.select("k", pl.col("ncid").alias("cid"), "unit_key").join(g.select(pl.col("ncid").alias("gid"), "unit_key"), on="unit_key")
              .select("k", "cid", "gid", pl.lit(1).alias("r")),
            cc.select("k", pl.col("ncid").alias("cid"), "fn", "sn").join(g.select(pl.col("ncid").alias("gid"), "fn", "sn"), on=["fn", "sn"])
              .select("k", "cid", "gid", pl.lit(2).alias("r")),
            cc.select("k", pl.col("ncid").alias("cid"), "ssx", "fsx", "zip5")
              .join(g.select(pl.col("ncid").alias("gid"), "ssx", "fsx", "zip5"), on=["ssx", "fsx", "zip5"])
              .select("k", "cid", "gid", pl.lit(3).alias("r"))])
        p = cand.group_by("k", "cid", "gid").agg((pl.col("r") == 1).any().alias("b1"), (pl.col("r") == 2).any().alias("b2"),
                                                  (pl.col("r") == 3).any().alias("b3"))
        # one row per pair and graph row (a person with two graph rows is two candidates, as the join gives)
        p = (p.join(cc.rename({"ncid": "cid"}).drop("county0", "s_dt", "has_truth", "t_size", "det_matched", "fkey"), on=["k", "cid"])
             .join(g.rename({x: x + "_g" for x in gcols if x != "ncid"}).rename({"ncid": "gid"}), on="gid"))
        p = _levels(p).with_columns((pl.col("cid") == pl.col("gid")).alias("same"), pl.col("akey_g").alias("gakey"))
        p = p.select("k", "cid", "gid", "gakey", "same", "lfn", "lsn", "lmi", "lsx", "lad", "b1", "b2", "b3", "fn", "sn", "unit_key")
        p = W.add(p).drop("fn", "sn", "unit_key")
        tie = rf.map_unique(p.select(pl.concat_str([pl.lit(MATCHER_TIE_SALT + "|"), pl.col("gid")]))[:, 0], rf.fingerprint, pl.Int64)
        p = p.with_columns(tie.alias("tie"))
        agg = p.group_by("k", "cid").agg(pl.len().cast(pl.Int64).alias("n_cand"), pl.col("same").any().alias("own_in_cand"),
                                         pl.col("w").filter(pl.col("same")).max().alias("w_own"))
        order = ["k", "cid", "w", "tie", "gakey", "lfn", "lsn", "lmi", "lsx", "lad"]
        b = (p.sort(order, descending=[False, False, True] + [False] * 7, nulls_last=False)
             .group_by("k", "cid", maintain_order=True).first().drop("tie"))
        out.append(b.join(agg, on=["k", "cid"]))
    return pl.concat(out).sort(["k", "cid"])


def matcher_records(st: S.Store, c: pl.DataFrame, bx: pl.DataFrame, fg: pl.DataFrame, src: pl.DataFrame) -> pl.DataFrame:
    """Per client and k: the best candidate's weight and its cluster's overlap with the client's truth at s.
    src: (ncid, k, src) from step 4."""
    csize = fg.group_by("akey").agg(pl.col("ncid").n_unique().cast(pl.Int64).alias("a_size"))
    out = []
    for s in sorted(set(c["s_dt"].to_list())):
        cs = c.filter(pl.col("s_dt") == s)
        ts = (st.read("graph", s, columns=["ncid", "tk"]).group_by("ncid").agg(pl.col("tk").min().alias("tkey")))
        x = cs.join(bx.rename({"cid": "ncid"}), on=["k", "ncid"], how="left")
        need = x.select("gakey").drop_nulls().unique()
        cm = (fg.select("akey", "ncid").join(need.rename({"gakey": "akey"}), on="akey", how="semi")
              .join(ts, on="ncid", how="inner").group_by("akey", "tkey").agg(pl.len().cast(pl.Int64).alias("a_int"))
              .rename({"akey": "gakey"}))
        x = (x.join(csize.rename({"akey": "gakey"}), on="gakey", how="left")
             .join(ts.rename({"tkey": "ctkey"}), on="ncid", how="left")
             .join(cm.rename({"tkey": "ctkey"}), on=["gakey", "ctkey"], how="left")
             .with_columns(pl.col("a_int").fill_null(0), (pl.col("fkey") == pl.col("gakey")).fill_null(False).alias("self_in_a")))
        out.append(x)
    x = pl.concat(out).join(src, on=["ncid", "k"], how="left")
    return x.select("k", "ncid", "county0", "s_dt", "has_truth", "t_size", "det_matched", "src", "w", "lfn", "lsn", "lmi", "lsx",
                    "lad", "same", "n_cand", "own_in_cand", "w_own", "b1", "b2", "b3", "a_size", "self_in_a", "a_int",
                    "gakey").sort(["k", "ncid"])
