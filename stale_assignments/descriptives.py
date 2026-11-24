# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline steps 8 and 9: build age against calendar (build_age) and the descriptive county cells.

Step 8. A never-refreshed address-only graph built at b is scored at later s. Pairs (b, s), b < s: every November snapshot
against every later one; every snapshot from 2018-11-06 to 2020-01-01 against every later one in that set; each January 1
build against the next November. Client file at b: trackable persons whose county at b is in the run's list, minors at b
excluded. The CRM record is current (the person's smallest key at s), so only persons with truth at s are scored.
core = every row of the person's household at b is active. Output: sums per (b, s, county at b, core).

Step 9. Long-form cells (county0, key, num, den), t = 2016, policy never_rebuilt, lag 0, persons with truth:
  household_attribute|<dim>|<val>|<metric>|<k>  dim size (5 = 5 or more), tag (voter tag), core;
  address_tenure|tenure|<bucket>|<metric>|<k>  years at the 2016 key in an unbroken run of snapshots ending at t (8+ = every snapshot);
    metrics mr, f1c, mov (share with a move event), pop (share of persons with truth);
  household_change|chg|<k>, household_change|loss|<k>; build_age keys from step 8.
"""
import datetime as dt
import math

import polars as pl

from . import store as S
from .common import T0, client_flag, logical_and, months
from .rematch import Graph, cluster_overlap
from .score import _f1c, rsum

NOV = [d for d in S.SNAPSHOTS if S.series_of(d) in ("eday", "nov_odd")]
DENSE_B, DENSE_END = dt.date(2018, 11, 6), dt.date(2020, 1, 1)


def age_pairs():
    d19 = [d for d in S.SNAPSHOTS if DENSE_B <= d <= DENSE_END]
    p = {(b, s) for b in NOV for s in NOV if s > b} | {(b, s) for b in d19 for s in d19 if s > b}
    for j in S.JAN1:
        later = [n for n in NOV if n > j]
        if later:
            p.add((j, min(later)))
    return sorted(p)


def age_universe(st: S.Store, b, counties, sample) -> pl.DataFrame:
    minor = st.read("age_flags").filter((pl.col("snapshot_dt") == b) & pl.col("minor")).select("ncid").unique()
    a = st.read("addr", b, columns=["ncid", "unit_key", "county_id", "is_ai"]).filter(pl.col("is_ai").fill_null(False))
    h = st.read("tags", b, columns=["ncid", "unit_key", "trackable_voter", "unit_all_active"]).filter(pl.col("trackable_voter"))
    u = (h.join(a, on=["ncid", "unit_key"], how="inner").filter(pl.col("county_id").is_in(list(counties)))
         .join(minor, on="ncid", how="anti"))
    u = u.filter(client_flag(u["ncid"], sample))
    return u.group_by("ncid").agg(pl.col("county_id").min().alias("county_b"), logical_and(pl.col("unit_all_active")).alias("core"))


def age_cells(st: S.Store, counties, sample) -> pl.DataFrame:
    G = Graph(st)
    out = []
    pairs = age_pairs()
    for b in sorted({b for b, _ in pairs}):
        U = age_universe(st, b, counties, sample)
        own = G[b]["truth"].select("ncid", pl.col("ckey").alias("fkey"))
        for _, s in [p for p in pairs if p[0] == b]:
            gb, gs = G[b], G[s]
            m = (U.join(gs["truth"].select("ncid", "tkey", "ckey"), on="ncid", how="inner")
                 .join(gb["csize"].rename({"akey": "ckey", "n": "a_size"}), on="ckey", how="left")
                 .join(own, on="ncid", how="inner")
                 .with_columns(pl.when(pl.col("a_size").is_not_null()).then(pl.col("ckey")).alias("mkey")))
            need = m.filter(pl.col("mkey").is_not_null()).select(pl.col("mkey").alias("akey")).unique()
            cm = cluster_overlap(G, b, s, need).rename({"akey": "mkey", "cnt": "a_int"})
            m = (m.join(gs["tsize"], on="tkey", how="left").join(cm, on=["mkey", "tkey"], how="left")
                 .with_columns(pl.col("a_int").fill_null(0), pl.col("mkey").is_not_null().alias("matched"),
                               (pl.col("fkey") == pl.col("mkey")).fill_null(False).alias("self_in_a")))
            sf = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
            m = m.with_columns(pl.when(pl.col("matched")).then(pl.col("a_size") - sf).otherwise(0).alias("ac"),
                               pl.when(pl.col("matched")).then(pl.col("a_int") - sf).otherwise(0).alias("ic"),
                               (pl.col("t_size") - 1).alias("tc"))
            n1, d1 = _f1c(pl.col("tc"), pl.col("ac"), pl.col("ic"))
            ok = pl.col("matched") & (pl.col("a_int") > 0)
            m = m.with_columns(n1.alias("f1c_n"), d1.alias("f1c_d"),
                               pl.when(ok).then(2 * pl.col("a_int")).otherwise(0).alias("f1_n"),
                               pl.when(ok).then(pl.col("a_size") + pl.col("t_size")).otherwise(1).alias("f1_d"))
            keys = ["county0", "core"]
            m = m.rename({"county_b": "county0"})
            c = m.group_by(keys).agg(pl.len().cast(pl.Int64).alias("n"), pl.col("matched").sum().cast(pl.Int64).alias("n_m"))
            c = (c.join(rsum(m, keys, "f1c_n", "f1c_d", "f1c"), on=keys, how="left", nulls_equal=True)
                 .join(rsum(m, keys, "f1_n", "f1_d", "f1"), on=keys, how="left", nulls_equal=True))
            out.append(c.with_columns(pl.lit(b).alias("b_dt"), pl.lit(s).alias("s_dt"),
                                      pl.lit((s.year - b.year) * 12 + s.month - b.month, dtype=pl.Int64).alias("age_m"),
                                      pl.lit(S.series_of(b)).alias("b_series"), pl.lit(S.series_of(s)).alias("s_series")))
    cols = ["b_dt", "s_dt", "age_m", "b_series", "s_series", "county0", "core", "n", "n_m", "f1c", "f1"]
    return pl.concat([o.select(cols) for o in out]).sort(["b_dt", "s_dt", "county0", "core"], nulls_last=False)


ATTRIBUTE_DIMS = ("size", "tag", "core")


def tenure(st: S.Store, ncids: pl.DataFrame) -> pl.DataFrame:
    """(ncid, tenure) for clients with a graph record at t."""
    snaps = [d for d in S.SNAPSHOTS if d <= T0]
    own = (st.read("graph", T0, columns=["ncid", "akey"]).join(ncids, on="ncid", how="semi")
           .group_by("ncid").agg(pl.col("akey").min().alias("fkey")))
    last_off = own.select("ncid").with_columns(pl.lit(None, dtype=pl.Date).alias("last_off"))
    for d in snaps:
        g = st.read("graph", d, columns=["ncid", "akey"]).join(own, on="ncid", how="inner")
        at = g.group_by("ncid").agg((pl.col("akey") == pl.col("fkey")).any().alias("at"))
        x = own.select("ncid").join(at, on="ncid", how="left").with_columns(pl.col("at").fill_null(False))
        off = x.filter(~pl.col("at")).select("ncid", pl.lit(d).alias("d"))
        last_off = (last_off.join(off, on="ncid", how="left")
                    .with_columns(pl.coalesce(pl.col("d"), pl.col("last_off")).alias("last_off")).drop("d"))
    nxt = {d: e for d, e in zip(snaps, snaps[1:])}
    lo = last_off.with_columns(pl.col("last_off").replace_strict(nxt, default=None, return_dtype=pl.Date).alias("first_on"))
    mo = months(pl.lit(T0), pl.col("first_on"))
    return lo.select("ncid", pl.when(pl.col("last_off").is_null()).then(pl.lit("8+"))
                     .when(mo < 24).then(pl.lit("<2")).when(mo < 48).then(pl.lit("2-4")).otherwise(pl.lit("4-8")).alias("tenure"))


def _cell(df, county, key, num_n, num_d, den):
    """Rows (county0, key, num_n, num_d, den_n, den_d) for one key family."""
    return df.select(pl.col(county).alias("county0"), key.alias("key"), num_n.cast(pl.Int64).alias("nn"),
                     num_d.cast(pl.Int64).alias("nd"), den.cast(pl.Int64).alias("dn"), pl.lit(1, dtype=pl.Int64).alias("dd"))


def long_sum(rows: pl.DataFrame) -> pl.DataFrame:
    """SUM(num), SUM(den) per (county0, key), exactly."""
    k = ["county0", "key"]
    a = rsum(rows, k, "nn", "nd", "num")
    b = rsum(rows, k, "dn", "dd", "den")
    return a.join(b, on=k, how="full", coalesce=True, nulls_equal=True).select("county0", "key", "num", "den")


def descriptive_cells(st: S.Store, pe: pl.DataFrame, keep: pl.DataFrame, age: pl.DataFrame) -> pl.DataFrame:
    """keep: the scored t = 2016, lag 0 rows (step 4 columns); age: step 8 output."""
    attr = (pe.filter(pl.col("client")).group_by("ncid")
            .agg(pl.min_horizontal(pl.col("size0"), pl.lit(5)).first().alias("size0"), pl.col("voter_tag0").first().alias("tag0"),
                 pl.col("core").first()))
    ten = tenure(st, attr.select("ncid"))
    r = (keep.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
         .select("ncid", "county0", "k", "matched", "f1c_n", "f1c_d", "event_list", "people_event")
         .join(attr, on="ncid", how="inner").join(ten, on="ncid", how="left")
         .with_columns(pl.col("tenure").fill_null("none")))
    dims = [pl.concat_str([pl.lit("household_attribute|size|"), pl.col("size0").cast(pl.String)]),
            pl.concat_str([pl.lit("household_attribute|tag|"), pl.col("tag0").fill_null("none")]),
            pl.concat_str([pl.lit("household_attribute|core|"), pl.col("core").cast(pl.String).str.to_lowercase()]),
            pl.concat_str([pl.lit("address_tenure|tenure|"), pl.col("tenure")])]
    ks = pl.col("k").cast(pl.String)
    one = pl.lit(1)
    mv = pl.col("event_list").str.starts_with("move").fill_null(False)
    # Every key family is summed on its own (the families' keys are disjoint), so no stacked copy of the rows is built.
    outs = [long_sum(_cell(r, "county0", pl.concat_str([pl.lit("household_change|chg|"), ks]), (pl.col("people_event") != "stable").fill_null(False), one, one)),
            long_sum(_cell(r, "county0", pl.concat_str([pl.lit("household_change|loss|"), ks]), pl.col("f1c_d") - pl.col("f1c_n"), pl.col("f1c_d"), one))]
    # pop: every county and k in the denominator, every group seen anywhere in the numerator (0 where absent)
    tot = r.group_by("county0", "k").agg(pl.len().cast(pl.Int64).alias("n"))
    pops = []
    for g in dims:
        grp = r.with_columns(g.alias("g"))
        outs += [long_sum(_cell(grp, "county0", pl.concat_str([pl.col("g"), pl.lit("|mr|"), ks]), pl.col("matched").fill_null(False), one, one)),
                 long_sum(_cell(grp, "county0", pl.concat_str([pl.col("g"), pl.lit("|f1c|"), ks]), pl.col("f1c_n"), pl.col("f1c_d"), one)),
                 long_sum(_cell(grp, "county0", pl.concat_str([pl.col("g"), pl.lit("|mov|"), ks]), mv, one, one))]
        cnt = grp.group_by("county0", "k", "g").agg(pl.len().cast(pl.Int64).alias("cnt"))
        pops.append(tot.join(grp.select("g").unique(), how="cross").join(cnt, on=["county0", "k", "g"], how="left")
                    .with_columns(pl.col("cnt").fill_null(0)))
        del grp
    out = pl.concat(outs)
    pop = pl.concat(pops).select("county0", pl.concat_str([pl.col("g"), pl.lit("|pop|"), pl.col("k").cast(pl.String)]).alias("key"),
                                 pl.col("cnt").cast(pl.Float64).alias("num"), pl.col("n").cast(pl.Float64).alias("den"))
    # build_age from step 8: sums of its (already summed) county cells
    nov = pl.col("b_series").is_in(["eday", "nov_odd"]) & pl.col("s_series").is_in(["eday", "nov_odd"])
    ys = pl.col("s_dt").dt.year().cast(pl.String)
    age_specs = [
        (age.filter((pl.col("b_series") == "jan1") & pl.col("s_series").is_in(["eday", "nov_odd"]) & pl.col("age_m").is_between(9, 11)),
         pl.concat_str([pl.lit("build_age|jan|"), ys]), "f1c", "n"),
        (age.filter(nov & (pl.col("age_m") == 12)), pl.concat_str([pl.lit("build_age|nov|"), ys]), "f1c", "n"),
        (age.filter(nov), pl.concat_str([pl.lit("build_age|age|"), ((pl.col("age_m") + 6) // 12).cast(pl.String), pl.lit("|f1c")]), "f1c", "n"),
        (age.filter(nov), pl.concat_str([pl.lit("build_age|age|"), ((pl.col("age_m") + 6) // 12).cast(pl.String), pl.lit("|mr")]), "n_m", "n"),
        (age.filter((pl.col("b_dt") == DENSE_B) & (pl.col("s_dt") <= DENSE_END)),
         pl.concat_str([pl.lit("build_age|dense|"), pl.col("age_m").cast(pl.String), pl.lit("|all")]), "f1c", "n"),
        (age.filter((pl.col("b_dt") == DENSE_B) & (pl.col("s_dt") <= DENSE_END) & pl.col("core").fill_null(False)),
         pl.concat_str([pl.lit("build_age|dense|"), pl.col("age_m").cast(pl.String), pl.lit("|core")]), "f1c", "n")]
    age_rows = []
    for df, key, num, den in age_specs:
        x = df.select("county0", key.alias("key"), pl.col(num).cast(pl.Float64).alias("num"), pl.col(den).cast(pl.Float64).alias("den"))
        acc = {}
        for c, k, a, b in x.sort(["county0", "key"]).iter_rows():
            acc.setdefault((c, k), ([], []))
            if a is not None:
                acc[(c, k)][0].append(a)
            if b is not None:
                acc[(c, k)][1].append(b)
        age_rows += [(c, k, math.fsum(a) if a else None, math.fsum(b) if b else None) for (c, k), (a, b) in acc.items()]
    age_df = pl.DataFrame(age_rows, schema={"county0": out.schema["county0"], "key": pl.String, "num": pl.Float64, "den": pl.Float64},
                        orient="row")
    return pl.concat([out, pop.cast({"county0": out.schema["county0"]}), age_df]).sort(["key", "county0"])
