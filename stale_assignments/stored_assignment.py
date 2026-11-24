# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline step 3: stored household ID carried by the vendor across rebuilds (member_tracker, address_held,
member_tracker_keep).

The client stores its own 2016 address-only cluster c0 (smallest key). The vendor rebuilds at each Election Day 2018 to 2024
and carries the ID: member_tracker = the new cluster holding the most members of the previous carrier (record pairs counted; ties:
smallest key); address_held = the ID stays on c0 while anyone is registered there; member_tracker_keep = keep the previous carrier key if any of
its previous members is still there, else member_tracker. At s the client gets the carrier's members.
"""
import polars as pl

from . import store as S
from . import config
from .common import T0

ED = [d for d in S.EDAYS if T0 <= d <= config.LAST_S]  # idx 0 = 2016 ... 4 = 2024 in the main design


def _step(prev: pl.DataFrame, g0: pl.DataFrame, g1: pl.DataFrame, hybrid: bool) -> pl.DataFrame:
    """prev: (c0, carrier) at idx-1; returns (c0, carrier) at idx."""
    o = prev.join(g0.rename({"akey": "carrier"}), on="carrier", how="inner")          # members of the carrier (rows)
    n = o.join(g1, on="ncid", how="inner")                                             # their records now (rows)
    cnt = n.group_by("c0", "carrier", "akey").agg(pl.len().alias("cnt"))
    best = (cnt.sort(["c0", "cnt", "akey"], descending=[False, True, False])
            .group_by("c0", maintain_order=True).agg(pl.col("akey").first().alias("best"), pl.col("carrier").first().alias("prev"),
                                                      (pl.col("akey") == pl.col("carrier")).any().alias("kept")))
    if hybrid:
        return best.select("c0", pl.when(pl.col("kept")).then(pl.col("prev")).otherwise(pl.col("best")).alias("carrier"))
    return best.select("c0", pl.col("best").alias("carrier"))


def stored_assignments(st: S.Store, pe: pl.DataFrame) -> pl.DataFrame:
    g = {i: st.read("graph", d, columns=["ncid", "akey", "tk"]) for i, d in enumerate(ED)}
    client = pe.filter(pl.col("client")).group_by("ncid").agg(pl.col("county0").first())
    id0 = client.join(g[0].select("ncid", "akey"), on="ncid", how="inner").group_by("ncid", "county0").agg(pl.col("akey").min().alias("c0"))
    s0 = id0.select("c0").unique().with_columns(pl.col("c0").alias("carrier"))
    carriers = []
    for pol, hyb in (("member_tracker", False), ("member_tracker_keep", True)):
        prev = s0
        for i in range(1, len(ED)):
            prev = _step(prev, g[i - 1].select("ncid", "akey"), g[i].select("ncid", "akey"), hyb)
            carriers.append(prev.with_columns(pl.lit(pol).alias("policy"), pl.lit(i).alias("idx")))
    for i in range(1, len(ED)):
        keep = s0.select("c0").join(g[i].select(pl.col("akey").alias("c0")).unique(), on="c0", how="semi")
        carriers.append(keep.with_columns(pl.col("c0").alias("carrier"), pl.lit("address_held").alias("policy"), pl.lit(i).alias("idx")))
    carriers = pl.concat([c.select("c0", "carrier", "policy", pl.col("idx").cast(pl.Int32)) for c in carriers])
    out = []
    for i in range(1, len(ED)):
        truth = g[i].group_by("ncid").agg(pl.col("tk").min().alias("tkey"))
        tsize = truth.group_by("tkey").agg(pl.len().cast(pl.Int64).alias("t_size"))
        base = (id0.join(pl.DataFrame({"policy": ["member_tracker", "address_held", "member_tracker_keep"]}), how="cross")
                .join(carriers.filter(pl.col("idx") == i).drop("idx"), on=["c0", "policy"], how="left")
                .join(truth, on="ncid", how="left").join(tsize, on="tkey", how="left"))
        mem = (base.filter(pl.col("carrier").is_not_null()).select("ncid", "policy", "carrier", "tkey")
               .join(g[i].select(pl.col("akey").alias("carrier"), pl.col("ncid").alias("mid")), on="carrier", how="inner")
               .join(truth.rename({"ncid": "mid", "tkey": "mtkey"}), on="mid", how="left"))
        ov = mem.group_by("ncid", "policy").agg(
            pl.col("mid").n_unique().cast(pl.Int64).alias("a_size"),
            pl.col("mid").filter(pl.col("mtkey") == pl.col("tkey")).n_unique().cast(pl.Int64).alias("a_int"),
            (pl.col("mid") == pl.col("ncid")).any().alias("self_in_a"))
        s = ED[i]
        r = base.join(ov, on=["ncid", "policy"], how="left")
        out.append(r.select(
            pl.lit(T0).alias("t_dt"), "ncid", "county0", "policy", pl.lit(0, dtype=pl.Int64).alias("lag"), pl.lit(s).alias("s_dt"),
            pl.lit(s.year - T0.year, dtype=pl.Int64).alias("k"), pl.col("tkey").is_not_null().alias("has_truth"),
            pl.col("carrier").is_not_null().alias("matched"), pl.col("self_in_a").fill_null(False), "t_size", "a_size",
            pl.col("a_int").fill_null(0)))
    return pl.concat(out)
