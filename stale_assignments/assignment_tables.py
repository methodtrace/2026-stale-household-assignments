# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Row tables and cells of the assignment-currency tables (paper section 8.3): one row per scored client, the stale-
listing probe, and the fine cells the coarse table and the table with co-member ages are fitted on.

Rows: one per client-row of the lag 0, policy never_rebuilt step 2 rows of persons with truth, with the build-time features of
the client's stored unit (last_known_key) at t: building type bt, client age band ab, co-members (other registrants of the unit
at t, one row per person, smallest age): count, ny (aged 18 to 25: n0 none, y0, y1, y2 two or more), mean age band mb,
age gap band gb. e = exact household for the person anchor. Held-out rows add the stale-listing reference one Election
Day ahead (has_lookahead1, e_registered, e_lookahead1): members of T at s who at the next Election Day are removed with reason RM, or are listed in
another county with a registration date on or before s, are dropped from T.
"""
import polars as pl

from . import store as S

I = pl.Int64


def exact_e():
    fp, tp, fi = pl.col("f_size") - 1, pl.col("t_size") - 1, pl.col("f_int") - 1
    return pl.when(((fp == 0) & (tp == 0)) | ((fp > 0) & (fi == fp) & (fi == tp))).then(1).otherwise(0).cast(I)


def _features(st: S.Store, t, clients: pl.DataFrame | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """h: (ncid, unit_key) -> the first tags row (bldg_key, age); cma: (unit_key, client) -> co-member aggregates.
    clients (ncid): restrict h and cma to these clients (memory; the rows of other clients are never joined)."""
    hv = st.read("tags", t, columns=["ncid", "unit_key", "has_unit", "bldg_n", "unit_n", "age", "bldg_key"])
    h = hv if clients is None else hv.join(clients.select("ncid").unique(), on="ncid", how="semi")
    h = (h.sort(["ncid", "unit_key", "bldg_key", "age", "has_unit", "bldg_n", "unit_n"], nulls_last=True)
         .group_by(["ncid", "unit_key"], maintain_order=True).first())
    # co-member aggregates per (unit, client) over the unit's other members (min age per member), computed from unit
    # totals less the client's own term, which equals the client x member self-join without forming the pairs
    # (group-quarters units hold thousands of members). Ages are integers, so the sums and the mean are exact.
    mm = hv.group_by("unit_key", "ncid").agg(pl.col("age").min().alias("mage"))
    a = pl.col("mage")
    y = a.is_between(18, 25).fill_null(False).cast(I)
    u = mm.group_by("unit_key").agg(
        pl.len().cast(I).alias("N"), y.sum().alias("NY"), a.cast(pl.Float64).sum().alias("SA"), a.count().cast(I).alias("NA"),
        a.max().alias("mx1"), (a == a.max()).sum().cast(I).alias("cmx"), a.filter(a < a.max()).max().alias("mx2"),
        a.min().alias("mn1"), (a == a.min()).sum().cast(I).alias("cmn"), a.filter(a > a.min()).min().alias("mn2"))
    c = (h.select("unit_key", pl.col("ncid").alias("client")).unique()
         .join(mm.rename({"ncid": "client"}), on=["unit_key", "client"], how="inner").join(u, on="unit_key", how="inner")
         .filter(pl.col("N") > 1))
    own = a.is_not_null()
    na = pl.col("NA") - own.cast(I)
    cma = c.select(
        "unit_key", "client", (pl.col("N") - 1).alias("nco"), (pl.col("NY") - y).alias("ny"),
        pl.when(na > 0).then((pl.col("SA") - a.cast(pl.Float64).fill_null(0.0)) / na).otherwise(None).alias("mean_age"),
        pl.when(own & (a == pl.col("mx1")) & (pl.col("cmx") == 1)).then(pl.col("mx2")).otherwise(pl.col("mx1")).alias("max_age"),
        pl.when(own & (a == pl.col("mn1")) & (pl.col("cmn") == 1)).then(pl.col("mn2")).otherwise(pl.col("mn1")).alias("min_age"))
    return h, cma


def _band(x, cuts, labels, null_label):
    e = pl.when(x.is_null()).then(pl.lit(null_label))
    for c, lab in zip(cuts, labels):
        e = e.when(x < c).then(pl.lit(lab))
    return e.otherwise(pl.lit(labels[-1]))


def rows_by_build(st: S.Store, rm0: pl.LazyFrame, where=None):
    """The row table plus ncid, s_dt and last_known_key for the reference variants, one build at
    a time (memory: the 60-county row table does not fit in 24 GB at once)."""
    f = (pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth")
    if where is not None:
        f = f & where
    builds = sorted(rm0.filter(f).select(pl.col("t_dt").unique()).collect()["t_dt"].to_list())
    hn, cn = pl.col("_h").is_null(), pl.col("_c").is_null()
    age = pl.col("age")
    d1, d2 = (age - pl.col("max_age")).abs(), (age - pl.col("min_age")).abs()
    cols = [
        "t_dt", "k", "county0", pl.col("t_dt").cast(pl.String).alias("build"), pl.min_horizontal(pl.col("f_size"), pl.lit(4)).alias("size4"),
        "f_size",
        pl.when(hn).then(pl.lit("na")).when(pl.col("has_unit") | (pl.col("bldg_n") > pl.col("unit_n"))).then(pl.lit("multi"))
        .otherwise(pl.lit("single")).alias("bt"),
        _band(age, [25, 35, 50, 65], ["a18", "a25", "a35", "a50", "a65"], "na").alias("ab"),
        "age", "bldg_n", "unit_n", "has_unit",
        pl.when(cn).then(pl.lit("n0")).when(pl.col("ny") == 0).then(pl.lit("y0")).when(pl.col("ny") == 1).then(pl.lit("y1"))
        .otherwise(pl.lit("y2")).alias("ny"),
        pl.when(cn).then(pl.lit("none")).when(pl.col("mean_age") < 30).then(pl.lit("m18")).when(pl.col("mean_age") < 45)
        .then(pl.lit("m30")).when(pl.col("mean_age") < 65).then(pl.lit("m45")).otherwise(pl.lit("m65")).alias("mb"),
        pl.when(cn).then(pl.lit("none")).when((d1 < 10) & (d2 < 10)).then(pl.lit("g0"))
        .when(pl.when(d1 > d2).then(d1).otherwise(d2) < 25).then(pl.lit("g10")).otherwise(pl.lit("g25")).alias("gb"),
        "nco", pl.col("ny").alias("ny_n"), pl.col("mean_age").alias("co_mean_age"), pl.col("max_age").alias("co_max_age"),
        pl.col("min_age").alias("co_min_age"), "e", "ncid", "s_dt", "last_known_key", "t_size", "f_int"]
    for t in builds:
        rt = (rm0.filter(f & (pl.col("t_dt") == t)).select("t_dt", "k", "ncid", "county0", "last_known_key", "s_dt", "f_size", "t_size", "f_int")
              .collect().with_columns(pl.col("last_known_key").str.slice(1).cast(I).alias("ukey"), exact_e().alias("e")))
        h, cma = _features(st, t, rt)
        yield (rt.join(h.rename({"unit_key": "ukey"}).with_columns(pl.lit(True).alias("_h")), on=["ncid", "ukey"], how="left")
               .join(cma.rename({"unit_key": "ukey", "client": "ncid"}).with_columns(pl.lit(True).alias("_c")),
                     on=["ukey", "ncid"], how="left")
               .select(cols))
        del h, cma, rt


def rows(st: S.Store, rm0: pl.LazyFrame, where=None) -> pl.DataFrame:
    return pl.concat(list(rows_by_build(st, rm0, where)))


def _member_flags(st: S.Store, s, ed):
    """Per ncid listed at s: inact_s, st2, st4 (next and second-next Election Day in ed; False past the end)."""
    a = st.read("addr", s, columns=["ncid", "county_id", "status_cd"])
    ms = a.group_by("ncid").agg((pl.col("status_cd") == "I").fill_null(False).any().alias("inact_s"),
                                pl.col("county_id").min().alias("county_s"))
    i = ed.index(s)
    for name, j in (("st2", i + 1), ("st4", i + 2)):
        if j < len(ed):
            n = st.read("addr", ed[j], columns=["ncid", "county_id", "is_ai", "reason_cd", "registr_dt"]).join(
                ms.select("ncid", "county_s"), on="ncid", how="inner")
            c = ((~pl.col("is_ai") & (pl.col("reason_cd") == "RM"))
                 | (pl.col("is_ai") & (pl.col("county_id") != pl.col("county_s")) & (pl.col("registr_dt") <= s))).fill_null(False)
            ms = ms.join(n.group_by("ncid").agg(c.any().alias(name)), on="ncid", how="left").with_columns(pl.col(name).fill_null(False))
        else:
            ms = ms.with_columns(pl.lit(False).alias(name))
    return ms.select("ncid", "inact_s", "st2", "st4")


def stale_counts(st: S.Store, r: pl.DataFrame, ed=None) -> pl.DataFrame:
    """Per client-row (t_dt, k, ncid): nf, nt, nft and their variants without flagged members;
    has_lookahead1, has_lookahead2 from the Election Day index of s in ed (default the 40-file Election Days)."""
    ed = ed or [d for d in S.SNAPSHOTS if S.series_of(d) == "eday"]
    out = []
    g = {}

    def graph(d):
        if d not in g:
            if len(g) >= 2:
                g.pop(next(iter(g)))
            g[d] = st.read("graph", d, columns=["ncid", "akey", "tk"])
        return g[d]

    for s in sorted(r["s_dt"].unique().to_list()):
        fl = _member_flags(st, s, ed)
        ctk_all = graph(s).group_by("ncid").agg(pl.col("tk").min().alias("tk"))
        rs = r.filter(pl.col("s_dt") == s).select("t_dt", "k", "ncid", "last_known_key")
        ctk = rs.join(ctk_all, on="ncid", how="inner")
        tmem = ctk_all.filter(pl.col("tk").is_not_null()).rename({"ncid": "m"})
        tset = ctk.join(tmem, on="tk", how="inner").filter(pl.col("m") != pl.col("ncid")).select("t_dt", "k", "ncid", "m").unique()
        fparts = []
        for t in sorted(rs["t_dt"].unique().to_list()):
            gt = graph(t).select(pl.col("akey").alias("last_known_key"), pl.col("ncid").alias("m")).unique()
            fparts.append(rs.filter(pl.col("t_dt") == t).join(gt, on="last_known_key", how="inner").filter(pl.col("m") != pl.col("ncid"))
                          .select("t_dt", "k", "ncid", "m").unique())
        fset = pl.concat(fparts)
        u = (fset.with_columns(pl.lit(True).alias("inf"))
             .join(tset.with_columns(pl.lit(True).alias("int_")), on=["t_dt", "k", "ncid", "m"], how="full", coalesce=True)
             .with_columns(pl.col("inf").fill_null(False), pl.col("int_").fill_null(False))
             .join(fl.rename({"ncid": "m"}), on="m", how="left")
             .with_columns([pl.col(c).fill_null(False) for c in ("inact_s", "st2", "st4")]))
        # flags count only for members of T (tset)
        cnt = lambda e: e.sum().cast(I)
        inf, it = pl.col("inf"), pl.col("int_")
        out.append(u.group_by("t_dt", "k", "ncid").agg(
            cnt(inf).alias("nf"), cnt(it).alias("nt"), cnt(inf & it).alias("nft"),
            cnt(it & ~pl.col("st2")).alias("nt2"), cnt(inf & it & ~pl.col("st2")).alias("nft2"),
            cnt(it & ~pl.col("st4")).alias("nt4"), cnt(inf & it & ~pl.col("st4")).alias("nft4"),
            cnt(it & ~pl.col("inact_s")).alias("nti"), cnt(inf & it & ~pl.col("inact_s")).alias("nfti"),
            cnt(it & pl.col("st2")).alias("fl2"), cnt(it & pl.col("st4")).alias("fl4"), cnt(it & pl.col("inact_s")).alias("fli")))
    pc = pl.concat(out)
    idx = pl.DataFrame({"s_dt": ed, "i": list(range(len(ed)))})
    last = len(ed) - 1
    z = [c for c in pc.columns if c not in ("t_dt", "k", "ncid")]
    return (r.join(idx, on="s_dt", how="inner").join(pc, on=["t_dt", "k", "ncid"], how="left")
            .with_columns([pl.col(c).fill_null(0) for c in z])
            .with_columns((pl.col("i") + 1 <= last).alias("has_lookahead1"), (pl.col("i") + 2 <= last).alias("has_lookahead2")))


def held_out_rows(st: S.Store, rm0: pl.LazyFrame) -> pl.DataFrame:
    r = rows(st, rm0)
    x = stale_counts(st, r)
    eq = lambda a, b: pl.when((pl.col("nf") == pl.col(a)) & (pl.col("nf") == pl.col(b))).then(1).otherwise(0).cast(I)
    return x.with_columns(eq("nt", "nft").alias("e_registered"), eq("nt2", "nft2").alias("e_lookahead1"))


def stale_probe(st: S.Store, rm0: pl.LazyFrame) -> pl.DataFrame:
    """stale_probe_development.csv: one row per (t_dt, k, stay)."""
    r = (rm0.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
         .select("t_dt", "k", "ncid", "last_known_key", "s_dt", (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay"),
                 "f_size", "t_size", "f_int").collect().with_columns(exact_e().alias("e_rem")))
    x = stale_counts(st, r)
    nf = pl.col("nf")
    ok = lambda a, b: (nf == pl.col(a)) & (nf == pl.col(b))
    s = lambda e: e.cast(I).sum()
    av2, av4 = pl.col("has_lookahead1"), pl.col("has_lookahead2")
    return x.group_by("t_dt", "k", "stay").agg(
        pl.len().cast(I).alias("n"), pl.col("e_rem").sum().alias("e_rem"), s(ok("nt", "nft")).alias("e_registered"),
        s(av2).alias("n_has_lookahead1"), s(av2 & ok("nt", "nft")).alias("e_registered_has_lookahead1"),
        s(av2 & ok("nt2", "nft2")).alias("e_lookahead1"),
        s(av4).alias("n_has_lookahead2"), s(av4 & ok("nt", "nft")).alias("e_registered_has_lookahead2"),
        s(av4 & ok("nt4", "nft4")).alias("e_lookahead2"),
        s(ok("nti", "nfti")).alias("e_inactive"), pl.col("nt").sum().alias("members"),
        pl.when(av2).then(pl.col("fl2")).otherwise(0).sum().alias("flagged_lookahead1"),
        pl.when(av4).then(pl.col("fl4")).otherwise(0).sum().alias("flagged_lookahead2"), pl.col("fli").sum().alias("flagged_inactive"),
        pl.when(av2).then(pl.col("nt")).otherwise(0).sum().alias("members_has_lookahead1"),
        pl.when(av4).then(pl.col("nt")).otherwise(0).sum().alias("members_has_lookahead2")).sort(["t_dt", "k", "stay"])


def fine_cells(rows: pl.DataFrame, label="e", where=None) -> pl.DataFrame:
    """The fine cells of the coarse table and the table with co-member ages (t_dt, k, county0, size6, bt, ab, ny, mb,
    gb, n, ne), sorted by their keys."""
    x = rows if where is None else rows.filter(where)
    age, mean = pl.col("age"), pl.col("co_mean_age")
    x = x.select(
        "t_dt", "k", "county0", pl.min_horizontal(pl.col("f_size"), pl.lit(6)).alias("size6"), "bt",
        _band(age, [25, 35, 45, 55, 65, 75], ["a18", "a25", "a35", "a45", "a55", "a65", "a75"], "na").alias("ab"), "ny",
        pl.when(pl.col("nco").is_null()).then(pl.lit("none")).when(mean < 25).then(pl.lit("m18")).when(mean < 35).then(pl.lit("m25"))
        .when(mean < 45).then(pl.lit("m35")).when(mean < 55).then(pl.lit("m45")).when(mean < 65).then(pl.lit("m55"))
        .otherwise(pl.lit("m65")).alias("mb"), "gb", pl.col(label).alias("e"))
    keys = ["t_dt", "k", "county0", "size6", "bt", "ab", "ny", "mb", "gb"]
    c = x.group_by(keys).agg(pl.len().cast(I).alias("n"), pl.col("e").sum().cast(I).alias("ne")).sort(keys)
    return c.with_columns(pl.col("t_dt").cast(pl.String))
