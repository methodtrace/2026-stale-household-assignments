# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Robustness checks of the anchor accounting (paper section 9 and the supplement's robustness tables).

stale_cells: stale-listing and inactive-ghost reference variants of the exact-household outcome, county cells by build,
age, policy, county, matched, stay, ghost flags and lookahead availability. Returned set A = graph members of the returned cluster at b; true set T = registrants sharing the
client's reference key at s; both without the client. Exact = (A = T). Variants of T (member flags):
  lookahead_one_election / lookahead_two_elections: drop members who at the next (two-year) or second-next (four-year) Election Day are removed with reason RM, or
           are listed active or inactive in another county with a registration date on or before s;
  inactive_dropped: drop members inactive (status I) at s.
A client itself flagged (client_flag_lookahead1: the one-election rule applied to the client; client_flag_inactive:
inactive at s) counts as exact only if stored alone. Returned sets: last_known (the client's own cluster at t),
never_rebuilt and the rebuild policies (lag 0 step 2 rows), and the probabilistic matcher at W >= 0 and W >= 20 (main
build).
"""
import datetime as dt

import polars as pl

from .. import store as S
from ..common import T0

I = pl.Int64
POLS = ["never_rebuilt", "lagged_rebuild", "same_date_rebuild", "four_year_rebuild", "annual_rebuild"]


def _ed():
    # the ten Election Days, the later file's included
    return sorted(set(S.EDAYS) | {S.LATER_FILE_DATE})


def _flags(st: S.Store, s, ed):
    """Per ncid in the address file at s: inact, st2, st4 (False when absent)."""
    a = st.read("addr", s, columns=["ncid", "county_id", "status_cd"])
    ms = a.group_by("ncid").agg((pl.col("status_cd") == "I").fill_null(False).any().alias("inact"),
                                pl.col("county_id").min().alias("county_s"))
    i = ed.index(s)
    for name, j in (("st2", i + 1), ("st4", i + 2)):
        if j < len(ed):
            n = st.read("addr", ed[j], columns=["ncid", "county_id", "is_ai", "reason_cd", "registr_dt"])
            n = n.join(ms.select("ncid", "county_s"), on="ncid", how="inner")
            c = ((~pl.col("is_ai") & (pl.col("reason_cd") == "RM"))
                 | (pl.col("is_ai") & (pl.col("county_id") != pl.col("county_s")) & (pl.col("registr_dt") <= s))).fill_null(False)
            f = n.group_by("ncid").agg(c.any().alias(name))
            ms = ms.join(f, on="ncid", how="left").with_columns(pl.col(name).fill_null(False))
        else:
            ms = ms.with_columns(pl.lit(False).alias(name))
    return ms.select("ncid", "inact", "st2", "st4")


def stale_rows(rm0: pl.LazyFrame, fsr: pl.DataFrame) -> pl.DataFrame:
    """The returned-set rows: (t_dt, k, ncid, county0, s_dt, policy, b_dt, key, stay)."""
    base = (rm0.filter((pl.col("lag") == 0) & pl.col("has_truth") & pl.col("policy").is_in(POLS))
            .select("t_dt", "k", "ncid", "county0", "s_dt", "policy", "b_dt", "mkey", "last_known_key").collect())
    stay = base.filter(pl.col("policy") == "never_rebuilt").select(
        "t_dt", "k", "ncid", (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay"))
    cols = ["t_dt", "k", "ncid", "county0", "s_dt", "policy", "b_dt", "key"]
    parts = [base.select(*cols[:7], pl.col("mkey").alias("key")),
             base.filter(pl.col("policy") == "never_rebuilt").select(
                 "t_dt", "k", "ncid", "county0", "s_dt", pl.lit("last_known").alias("policy"), pl.col("t_dt").alias("b_dt"),
                 pl.col("last_known_key").alias("key"))]
    f = fsr.filter(pl.col("has_truth"))
    for th in (0, 20):
        parts.append(f.select(pl.lit(T0).alias("t_dt"), "k", "ncid", "county0", "s_dt", pl.lit(f"probabilistic_matcher_w{th}").alias("policy"),
                              pl.lit(T0).alias("b_dt"),
                              pl.when(pl.col("w") >= th).then(pl.col("gakey")).otherwise(None).alias("key")))
    r = pl.concat([p.select(cols) for p in parts])
    return r.join(stay, on=["t_dt", "k", "ncid"], how="inner")


def stale_cells(st: S.Store, rws: pl.DataFrame) -> pl.DataFrame:
    ed = _ed()
    out = []
    gcache = {}

    def graph(d):
        if d not in gcache:
            if len(gcache) >= 2:
                gcache.pop(next(iter(gcache)))
            gcache[d] = st.read("graph", d, columns=["ncid", "akey", "tk"])
        return gcache[d]

    for s in sorted(rws["s_dt"].unique().to_list()):
        rs = rws.filter(pl.col("s_dt") == s)
        fl = _flags(st, s, ed)
        ctk = graph(s).group_by("ncid").agg(pl.col("tk").min().alias("tk")).filter(pl.col("tk").is_not_null())
        cf = ctk.join(fl, on="ncid", how="left").with_columns([pl.col(c).fill_null(False) for c in ("inact", "st2", "st4")])
        tsz = cf.group_by("tk").agg(pl.len().cast(I).alias("nt"), (~pl.col("st2")).sum().cast(I).alias("nt2"),
                                    (~pl.col("st4")).sum().cast(I).alias("nt4"), (~pl.col("inact")).sum().cast(I).alias("nti"))
        si = ed.index(s)
        for b in sorted(rs["b_dt"].unique().to_list()):
            r = rs.filter(pl.col("b_dt") == b)
            mem = graph(b).select("akey", "ncid").unique()
            need = r.select(pl.col("key").alias("akey")).drop_nulls().unique()
            mb = mem.join(need, on="akey", how="semi")
            asz = mb.group_by("akey").agg(pl.len().cast(I).alias("na"))
            cm = (mb.join(cf, on="ncid", how="inner").group_by("akey", "tk")
                  .agg(pl.len().cast(I).alias("c"), (~pl.col("st2")).sum().cast(I).alias("c2"),
                       (~pl.col("st4")).sum().cast(I).alias("c4"), (~pl.col("inact")).sum().cast(I).alias("ci")))
            selfa = mb.select(pl.col("akey").alias("key"), "ncid", pl.lit(True).alias("sf"))
            x = (r.join(cf.rename({"inact": "g_i", "st2": "g_2", "st4": "g_4"}), on="ncid", how="inner")
                 .join(tsz, on="tk", how="inner")
                 .join(asz.rename({"akey": "key"}), on="key", how="left")
                 .join(selfa, on=["key", "ncid"], how="left")
                 .join(cm.rename({"akey": "key"}), on=["key", "tk"], how="left"))
            sf = pl.col("sf").fill_null(False).cast(I)
            z = lambda c: pl.col(c).fill_null(0)
            x = x.select(
                pl.col("t_dt").dt.year().cast(I).alias("t"), "k", "policy", "county0", pl.col("key").is_not_null().alias("matched"),
                "stay", pl.lit(si + 1 <= 9).alias("has_lookahead1"), pl.lit(si + 2 <= 9).alias("has_lookahead2"),
                pl.col("g_2").alias("client_flag_lookahead1"), pl.col("g_i").alias("client_flag_inactive"),
                (z("na") - sf).alias("nA"), (pl.col("nt") - 1).alias("nT"),
                (pl.col("nt2") - (~pl.col("g_2")).cast(I)).alias("nT2"), (pl.col("nt4") - (~pl.col("g_4")).cast(I)).alias("nT4"),
                (pl.col("nti") - (~pl.col("g_i")).cast(I)).alias("nTi"),
                (z("c") - sf).alias("nAT"), (z("c2") - (pl.col("sf").fill_null(False) & ~pl.col("g_2")).cast(I)).alias("nAT2"),
                (z("c4") - (pl.col("sf").fill_null(False) & ~pl.col("g_4")).cast(I)).alias("nAT4"),
                (z("ci") - (pl.col("sf").fill_null(False) & ~pl.col("g_i")).cast(I)).alias("nATi"))
            e = lambda a, b: ((pl.col("nA") == pl.col(a)) & (pl.col("nA") == pl.col(b))).sum().cast(I)
            out.append(x.group_by("t", "k", "policy", "county0", "matched", "stay", "has_lookahead1", "has_lookahead2",
                                  "client_flag_lookahead1", "client_flag_inactive").agg(
                pl.len().cast(I).alias("n"), e("nT", "nAT").alias("e_registered"), e("nT2", "nAT2").alias("e_lookahead1"),
                e("nT4", "nAT4").alias("e_lookahead2"), e("nTi", "nATi").alias("e_inactive"), (pl.col("nA") == 0).sum().cast(I).alias("e_alone"),
                pl.col("nT").sum().cast(I).alias("mem"), (pl.col("nT") - pl.col("nT2")).sum().cast(I).alias("flag_lookahead1"),
                (pl.col("nT") - pl.col("nT4")).sum().cast(I).alias("flag_lookahead2"),
                (pl.col("nT") - pl.col("nTi")).sum().cast(I).alias("flag_inactive")))
    keys = ["t", "k", "policy", "county0", "matched", "stay", "has_lookahead1", "has_lookahead2", "client_flag_lookahead1",
            "client_flag_inactive"]
    c = pl.concat(out)
    return c.group_by(keys).agg([pl.col(x).sum().cast(I) for x in c.columns if x not in keys]).sort(keys)


def county_pack(cells: pl.DataFrame) -> pl.DataFrame:
    """Per (t, k, policy, variant) the county list 'county:n:exact' of matched clients."""
    c = cells.filter(pl.col("matched"))
    av2 = pl.col("has_lookahead1")
    spec = [("registered", av2, "e_registered"), ("lookahead_one_election", av2, "e_lookahead1"),
            ("lookahead_two_elections", pl.col("has_lookahead2"), "e_lookahead2"), ("inactive_dropped", av2, "e_inactive"),
            ("client_lookahead", av2,
             pl.when(pl.col("client_flag_lookahead1")).then(pl.col("e_alone")).otherwise(pl.col("e_lookahead1"))),
            ("client_inactive", av2, pl.when(pl.col("client_flag_inactive")).then(pl.col("e_alone")).otherwise(pl.col("e_inactive")))]
    parts = []
    for v, av, e in spec:
        e = pl.col(e) if isinstance(e, str) else e
        parts.append(c.group_by("t", "k", "policy", "county0").agg(
            pl.when(av).then(pl.col("n")).otherwise(0).sum().cast(I).alias("n"),
            pl.when(av).then(e).otherwise(0).sum().cast(I).alias("e")).with_columns(pl.lit(v).alias("variant")))
    v = pl.concat(parts).filter(pl.col("n") > 0).sort(["t", "k", "policy", "variant", "county0"])
    return (v.group_by("t", "k", "policy", "variant", maintain_order=True)
            .agg(pl.concat_str([pl.col("county0").cast(pl.String), pl.col("n").cast(pl.String), pl.col("e").cast(pl.String)],
                               separator=":").str.join(" ").alias("cells")))


# ---- universe classes and kappa_P by age band ------------------------------------------------------------------

UNIVERSE_CLASSES = ["trackable", "size_6_plus_single_family", "size_6_plus_unit", "group_quarters_no_unit",
               "group_quarters_with_unit", "mobile_home_park_no_unit", "mobile_home_park_with_unit",
               "multiunit_building_no_unit", "large_building_9_to_20"]


def universe_counts(st, counties, t=dt.date(2016, 11, 8)) -> dict:
    """Active and inactive registrants at t in the counties by universe class (minors at t dropped)."""
    a = (st.read("addr", t, columns=["ncid", "unit_key", "county_id", "is_ai"]).filter(pl.col("is_ai").fill_null(False))
         .filter(pl.col("county_id").is_in(list(counties))).select("ncid", "unit_key"))
    minor = st.read("age_flags").filter((pl.col("snapshot_dt") == t) & pl.col("minor")).select("ncid").unique()
    h = (st.read("tags", t, columns=["ncid", "unit_key", "voter_tag", "trackable_voter", "has_unit"])
         .join(a, on=["ncid", "unit_key"], how="inner").join(minor, on="ncid", how="anti"))
    tag, tr, hu = pl.col("voter_tag"), pl.col("trackable_voter"), pl.col("has_unit")
    cls = (pl.when(tr).then(pl.lit("trackable"))
           .when(tag == "SFH_LIKELY").then(pl.lit("size_6_plus_single_family"))
           .when(tag == "UNIT_IN_MULTIUNIT").then(pl.lit("size_6_plus_unit"))
           .when((tag == "GROUP_QUARTERS_OR_OTHER") & ~hu).then(pl.lit("group_quarters_no_unit"))
           .when(tag == "GROUP_QUARTERS_OR_OTHER").then(pl.lit("group_quarters_with_unit"))
           .when((tag == "MOBILE_HOME_PARK") & ~hu).then(pl.lit("mobile_home_park_no_unit"))
           .when(tag == "MOBILE_HOME_PARK").then(pl.lit("mobile_home_park_with_unit"))
           .when(tag == "MULTIUNIT_NO_UNIT").then(pl.lit("multiunit_building_no_unit"))
           .when(tag.is_in(["NOUNIT_LARGE_9_20", "UNIT_LARGE_9_20"])).then(pl.lit("large_building_9_to_20"))
           .otherwise(pl.lit("other")))
    g = h.group_by(cls.alias("c")).agg(pl.len().alias("n"))
    d = dict(g.iter_rows())
    assert set(d) <= set(UNIVERSE_CLASSES), f"unclassified tags: {set(d) - set(UNIVERSE_CLASSES)}"
    return {c: int(d.get(c, 0)) for c in UNIVERSE_CLASSES}


def age_band(rm0: pl.LazyFrame, pe: pl.DataFrame, t=dt.date(2016, 11, 8)) -> pl.DataFrame:
    """Person-anchor exact membership (policy never_rebuilt, lag 0, persons with truth) at build t by age
    at build in ACS bands: robustness/age_band.csv (k, band, n, exact)."""
    r = (rm0.filter((pl.col("t_dt") == t) & (pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
         .select("k", "ncid", (((pl.col("f_size") == 1) & (pl.col("t_size") == 1))
                                | ((pl.col("f_size") > 1) & (pl.col("f_int") == pl.col("f_size")) & (pl.col("f_int") == pl.col("t_size"))))
                 .cast(pl.Int64).alias("e")).collect())
    p = pe.select("ncid", "k", "age0").unique().with_columns(pl.lit(True).alias("j"))
    a = pl.col("age0")
    b5 = (a // 5) * 5
    band = (pl.when(a.is_null()).then(pl.lit("na")).when(a < 18).then(pl.lit("<18")).when(a < 20).then(pl.lit("18-19"))
            .when(a < 25).then(pl.lit("20-24")).when(a < 75).then(pl.concat_str([b5.cast(pl.String), pl.lit("-"), (b5 + 4).cast(pl.String)]))
            .otherwise(pl.lit("75+")))
    x = r.join(p, on=["ncid", "k"], how="left")
    return (x.group_by("k", band.alias("band")).agg(pl.len().cast(pl.Int64).alias("n"), pl.col("e").sum().cast(pl.Int64).alias("exact"))
            .sort(["k", "band"]))


# ---- graded classes and the probabilistic matcher by recorded-alone stratum ------------------------------------------------------------

def graded(rm0: pl.LazyFrame) -> pl.DataFrame:
    """Graded membership classes of the last-known stored set against the true set (both without the client),
    policy never_rebuilt, lag 0, persons with truth, per (build year, k)."""
    fi, fz, tz = pl.col("f_int"), pl.col("f_size"), pl.col("t_size")
    c = lambda e: e.fill_null(False).sum().cast(pl.Int64)
    return (rm0.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
            .group_by(pl.col("t_dt").dt.year().cast(pl.Int64).alias("t"), "k")
            .agg(pl.len().cast(pl.Int64).alias("n"), c((fi == fz) & (fi == tz)).alias("exact"), c((fi == fz) & (fz < tz)).alias("subset"),
                 c((fi == tz) & (tz < fz)).alias("superset"), c((fi > 1) & (fi < fz) & (fi < tz)).alias("turnover"),
                 c((fi == 1) & (fz > 1) & (tz > 1)).alias("disjoint"), c((fz == 1) & (tz == 1)).alias("exact_alone"))
            .sort(["t", "k"]).collect())


def matcher_alone_cells(fsr: pl.LazyFrame, rm0: pl.LazyFrame, t=dt.date(2016, 11, 8)) -> pl.DataFrame:
    """robustness/probabilistic_matcher_alone_cells.csv (w_min, k, alone_t, stay, n, matched, exact_matched,
    exact_last_known)."""
    m = pl.col("gakey").is_not_null()
    sf = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    f = (fsr.filter(pl.col("has_truth"))
         .select("k", "ncid", "w", (pl.col("t_size") - 1).alias("tp"), pl.when(m).then(pl.col("a_size") - sf).alias("nap"),
                 pl.when(m).then(pl.col("a_int") - sf).alias("nai"), m.alias("m")))
    r = (rm0.filter((pl.col("t_dt") == t) & (pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
         .select("k", "ncid", (pl.col("f_size") == 1).alias("alone_t"), (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay"),
                 (((pl.col("f_size") == 1) & (pl.col("t_size") == 1))
                  | ((pl.col("f_size") > 1) & (pl.col("f_int") == pl.col("f_size")) & (pl.col("f_int") == pl.col("t_size")))).cast(pl.Int64).alias("eP")))
    x = f.join(r, on=["k", "ncid"], how="inner").collect()
    ex = ((pl.col("nap") == 0) & (pl.col("tp") == 0)) | ((pl.col("nap") > 0) & (pl.col("nai") == pl.col("nap")) & (pl.col("nai") == pl.col("tp")))
    out = []
    for th in (0, 20):
        mt = pl.col("m") & (pl.col("w") >= th)
        out.append(x.group_by("k", "alone_t", "stay").agg(
            pl.len().cast(pl.Int64).alias("n"), mt.fill_null(False).sum().cast(pl.Int64).alias("matched"),
            (mt & ex).fill_null(False).sum().cast(pl.Int64).alias("exact_matched"), pl.col("eP").sum().cast(pl.Int64).alias("exact_last_known"))
            .with_columns(pl.lit(th, dtype=pl.Int64).alias("w_min")))
    return (pl.concat(out).select("w_min", "k", "alone_t", "stay", "n", "matched", "exact_matched", "exact_last_known")
            .sort(["w_min", "k", "alone_t", "stay"]))


def _exact_last_known():
    return (((pl.col("f_size") == 1) & (pl.col("t_size") == 1))
            | ((pl.col("f_size") > 1) & (pl.col("f_int") == pl.col("f_size")) & (pl.col("f_int") == pl.col("t_size")))).cast(pl.Int64)


def matcher_paired_county_pack(fsr: pl.LazyFrame, rm0: pl.LazyFrame, t=dt.date(2016, 11, 8)) -> pl.DataFrame:
    """robustness/matcher_paired_county_pack.csv: per (w_min th, k, county0, alone_t) over the never_rebuilt lag 0 clients with
    truth at the build t: n, n_stay, e_stay, n_ans (matcher answers at W >= th), e10_ans (matcher exact among them), eP_ans
    (last-known exact among them), eP_all."""
    m = pl.col("gakey").is_not_null()
    sf = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    f = (fsr.filter(pl.col("has_truth"))
         .select("k", "ncid", "w", (pl.col("t_size") - 1).alias("tp"), pl.when(m).then(pl.col("a_size") - sf).alias("nap"),
                 pl.when(m).then(pl.col("a_int") - sf).alias("nai"), m.alias("m")))
    r = (rm0.filter((pl.col("t_dt") == t) & (pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
         .select("k", "ncid", "county0", (pl.col("f_size") == 1).alias("alone_t"),
                 (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay"), _exact_last_known().alias("eP")))
    ex = ((pl.col("nap") == 0) & (pl.col("tp") == 0)) | ((pl.col("nap") > 0) & (pl.col("nai") == pl.col("nap")) & (pl.col("nai") == pl.col("tp")))
    x = (r.join(f, on=["k", "ncid"], how="left")
         .with_columns(pl.when(pl.col("m").fill_null(False) & ex.fill_null(False)).then(1).otherwise(0).cast(pl.Int64).alias("e10"))
         .collect())
    out = []
    for th in (0, 20):
        a = (pl.col("m") & (pl.col("w") >= th)).fill_null(False)
        out.append(x.group_by("k", "county0", "alone_t").agg(
            pl.len().cast(pl.Int64).alias("n"), pl.col("stay").sum().cast(pl.Int64).alias("n_stay"),
            pl.col("eP").filter(pl.col("stay")).sum().cast(pl.Int64).alias("e_stay"), a.sum().cast(pl.Int64).alias("n_ans"),
            pl.col("e10").filter(a).sum().cast(pl.Int64).alias("e10_ans"), pl.col("eP").filter(a).sum().cast(pl.Int64).alias("eP_ans"),
            pl.col("eP").sum().cast(pl.Int64).alias("eP_all")).with_columns(pl.lit(th, dtype=pl.Int64).alias("th")))
    return (pl.concat(out).select("th", "k", "county0", "alone_t", "n", "n_stay", "e_stay", "n_ans", "e10_ans", "eP_ans", "eP_all")
            .sort(["th", "k", "county0", "alone_t"]))


def anchor_currency_pack(sources) -> pl.DataFrame:
    """robustness/anchor_currency_pack.csv: per (src, t, k, single) the counties packed as
    county:n:eP:m:eA:ns:eS:mm:eMA (never_rebuilt, lag 0, persons with truth; single = the client lived alone at t).
    sources: [(src label, rematch0 LazyFrame, build years to take or None for all, k values to take or None for all)]. A
    later-file run with further builds also holds those builds' rows; each build is taken from its own run."""
    parts = []
    for src, rm0, years, ks in sources:
        r = rm0.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
        if years is not None:
            r = r.filter(pl.col("t_dt").dt.year().is_in(list(years)))
        if ks is not None:
            r = r.filter(pl.col("k").is_in(list(ks)))
        sf = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
        nap, nai, tp = pl.col("a_size") - sf, pl.col("a_int") - sf, pl.col("t_size") - 1
        eA = (pl.col("matched") & (((nap == 0) & (tp == 0)) | ((nap > 0) & (nai == nap) & (nai == tp)))).fill_null(False)
        stay = (pl.col("mkey") == pl.col("last_known_key")).fill_null(False)
        mm = pl.col("matched") & ~stay
        y = r.select(pl.lit(src).alias("src"), pl.col("t_dt").dt.year().cast(pl.Int64).alias("t"), "k", "county0",
                     (pl.col("f_size") == 1).alias("single"), _exact_last_known().alias("eP"), pl.col("matched"),
                     eA.cast(pl.Int64).alias("eA"), stay.alias("stay"), mm.alias("mm"))
        c = (y.group_by("src", "t", "k", "single", "county0")
             .agg(pl.len().cast(pl.Int64).alias("n"), pl.col("eP").sum().alias("eP"), pl.col("matched").sum().cast(pl.Int64).alias("m"),
                  pl.col("eA").sum().alias("eA"), pl.col("stay").sum().cast(pl.Int64).alias("ns"),
                  pl.col("eP").filter(pl.col("stay")).sum().alias("eS"), pl.col("mm").sum().cast(pl.Int64).alias("mm"),
                  pl.col("eA").filter(pl.col("mm")).sum().alias("eMA"))
             .collect())
        parts.append(c)
    c = pl.concat(parts).sort(["src", "t", "k", "single", "county0"])
    f = pl.concat_str([pl.col(x).cast(pl.String) for x in ("county0", "n", "eP", "m", "eA", "ns", "eS", "mm", "eMA")], separator=":")
    return (c.group_by("src", "t", "k", "single", maintain_order=True).agg(f.str.join(" ").alias("cells"))
            .sort(["src", "t", "k", "single"]))


def size_split(rm0: pl.LazyFrame, t=dt.date(2016, 11, 8)) -> pl.DataFrame:
    """One split: person-anchor exact membership by household size at build with the stayer split:
    (k, size, n, exact, stayers, stayers_exact)."""
    e = (((pl.col("f_size") == 1) & (pl.col("t_size") == 1))
         | ((pl.col("f_size") > 1) & (pl.col("f_int") == pl.col("f_size")) & (pl.col("f_int") == pl.col("t_size"))))
    sy = (pl.col("mkey") == pl.col("last_known_key")).fill_null(False)
    return (rm0.filter((pl.col("t_dt") == t) & (pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
            .group_by("k", pl.col("f_size").alias("size"))
            .agg(pl.len().cast(pl.Int64).alias("n"), e.sum().cast(pl.Int64).alias("exact"), sy.sum().cast(pl.Int64).alias("stayers"),
                 (sy & e).sum().cast(pl.Int64).alias("stayers_exact"))
            .sort(["k", "size"]).collect())


# ---- register one-year mobility with exits split by removal reason ------------------------------------------------

def mobility_reasons(st, pairs, client_ages: pl.DataFrame) -> pl.DataFrame:
    """Per (year of d0, age band) adults active at d0 with a reference key, moved, exited, and the
    exits by the removal reason on the most recent non-active row at d1 (by registration date; RD died, RL/RS moved out,
    RM/R2 list maintenance, other, none); n_clients = confirmation clients at the 2016 build by band (client_ages: age).
    Needs registr_dt in the January 1 tables."""
    from ..calibration import band_expr
    I = pl.Int64
    parts = []
    for d0, d1 in pairs:
        a0 = st.read("addr", d0, columns=["ncid", "is_ai", "status_cd", "age"])
        act = (a0.filter(pl.col("is_ai").fill_null(False) & (pl.col("status_cd") == "A") & pl.col("age").is_not_null())
               .group_by("ncid").agg(pl.col("age").min()).filter(pl.col("age") >= 18))
        g0 = st.read("graph", d0, columns=["ncid", "tk"]).filter(pl.col("tk").is_not_null()).unique()
        base = act.join(g0.select("ncid").unique(), on="ncid", how="semi")
        a1 = st.read("addr", d1, columns=["ncid", "is_ai", "reason_cd", "registr_dt"])
        pres = a1.filter(pl.col("is_ai").fill_null(False)).select("ncid").unique().with_columns(pl.lit(True).alias("present"))
        rsn = (a1.filter(~pl.col("is_ai").fill_null(True))
               .sort(["ncid", "registr_dt", "reason_cd"], descending=[False, True, False], nulls_last=[False, True, False])
               .group_by("ncid", maintain_order=True).agg(pl.col("reason_cd").first().alias("reason")))
        g1 = st.read("graph", d1, columns=["ncid", "tk"]).filter(pl.col("tk").is_not_null()).unique()
        same = g0.join(g1, on=["ncid", "tk"], how="inner").select("ncid").unique().with_columns(pl.lit(True).alias("same"))
        x = (base.join(pres, on="ncid", how="left").join(same, on="ncid", how="left").join(rsn, on="ncid", how="left")
             .with_columns(pl.col("present").fill_null(False), pl.col("same").fill_null(False), band_expr(pl.col("age")).alias("band"))
             .filter(pl.col("band").is_not_null()))
        r = pl.col("reason")
        ex = (pl.when(pl.col("present")).then(None).when(r == "RD").then(pl.lit("died")).when(r.is_in(["RL", "RS"])).then(pl.lit("moved_out"))
              .when(r.is_in(["RM", "R2"])).then(pl.lit("list_maint")).when(r.is_null()).then(pl.lit("none")).otherwise(pl.lit("other")))
        x = x.with_columns(ex.alias("ex"))
        cnt = lambda e: e.fill_null(False).cast(I).sum()
        parts.append(x.group_by("band").agg(
            pl.len().cast(I).alias("n"), cnt(pl.col("present") & ~pl.col("same")).alias("moved"), cnt(~pl.col("present")).alias("exited"),
            *[cnt(pl.col("ex") == v).alias(f"ex_{v}") for v in ("died", "moved_out", "list_maint", "other", "none")])
            .with_columns(pl.lit(d0.year, dtype=I).alias("year")))
    cl = (client_ages.with_columns(band_expr(pl.col("age")).alias("band")).filter(pl.col("band").is_not_null())
          .group_by("band").agg(pl.len().cast(I).alias("n_clients")))
    return (pl.concat(parts).join(cl, on="band", how="left").with_columns(pl.col("n_clients").fill_null(0))
            .select("year", "band", "n", "moved", "exited", "ex_died", "ex_moved_out", "ex_list_maint", "ex_other", "ex_none", "n_clients")
            .sort(["year", "band"]))
