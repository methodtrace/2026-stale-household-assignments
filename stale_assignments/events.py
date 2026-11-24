# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline step 1: universe at t = 2016-11-08, person events at each later Election Day, household events.

Universe: trackable persons at t (household key = voter unit key) whose county at t is in the run's county list, minors at t
excluded. At each later even-year Election Day s the person's representative row is the first under (active or inactive
first, latest registration date, reason, status, unit key; then county, ZIP, building, normalized unit, fuzzy key so the
choice is total). Address change rule (readdressing): a change is a format change when the fuzzy key is unchanged, or when
both buildings are in one relabel component and the normalized unit is the same.
"""
import polars as pl

from . import store as S
from . import config
from .common import T0, client_flag, fuzzy, logical_and, logical_or, nunit

LATER = [d for d in S.EDAYS if T0 < d <= config.LAST_S]  # 2018, 2020, 2022, 2024 in the main design


def _addr_keys(a: pl.DataFrame) -> pl.DataFrame:
    return a.with_columns(nunit("unit").alias("nu"), fuzzy(a).alias("fz"))


def person_events(st: S.Store, counties, sample: int) -> pl.DataFrame:
    canon = st.read("canon")
    minor0 = st.read("age_flags").filter((pl.col("snapshot_dt") == T0) & pl.col("minor")).select("ncid").unique()
    a0 = st.read("addr", T0).filter(pl.col("is_ai").fill_null(False))
    a0 = _addr_keys(a0).select("ncid", "county_id", "zip5", "bldg_key", "unit_key", "nu", "fz")
    h = st.read("tags", T0).filter(pl.col("trackable_voter"))
    base = (h.join(a0, on=["ncid", "unit_key"], how="inner")
            .filter(pl.col("county_id").is_in(list(counties)))
            .join(canon.rename({"bldg_key": "bldg_key_c", "canon_bldg": "cb0"}), left_on="bldg_key", right_on="bldg_key_c", how="left")
            .join(minor0, on="ncid", how="anti"))
    base = base.select(
        "ncid", pl.col("unit_key").alias("dp0"), pl.col("unit_key").alias("uk0"), pl.col("bldg_key").alias("bk0"),
        pl.col("unit_all_active").alias("core"), pl.col("size_voter").alias("size0"), pl.col("voter_tag").alias("dp_tag0"),
        pl.col("voter_tag").alias("voter_tag0"), pl.col("age").alias("age0"), pl.col("county_id").alias("county0"),
        pl.col("zip5").alias("zip0"), pl.col("fz").alias("fz0"), pl.col("nu").alias("nu0"), "cb0")
    base = base.with_columns(client_flag(base["ncid"], sample).alias("client"))
    ids = base.select("ncid")
    out = []
    for s in LATER:
        a = st.read("addr", s).join(ids, on="ncid", how="semi")
        a = _addr_keys(a)
        n_ai = a.group_by("ncid").agg((pl.col("is_ai") == True).sum().alias("n_ai"))  # noqa: E712
        a = a.with_columns(pl.when(pl.col("is_ai") == True).then(2).when(pl.col("is_ai") == False).then(1)  # noqa: E712
                           .otherwise(0).alias("_ai"))
        r = (a.sort(["ncid", "_ai", "registr_dt", "reason_cd", "status_cd", "unit_key", "county_id", "zip5", "bldg_key", "nu", "fz"],
                    descending=[False, True, True, False, False, False, False, False, False, False, False],
                    nulls_last=[False, True, True, False, False, False, False, False, False, False, False])
             .group_by("ncid", maintain_order=True).first()
             .select("ncid", pl.col("is_ai").alias("r_is_ai"), pl.col("status_cd").alias("status_s"), pl.col("reason_cd").alias("reason_s"),
                     pl.col("county_id").alias("r_county"), pl.col("zip5").alias("r_zip5"), pl.col("bldg_key").alias("r_bldg"),
                     pl.col("unit_key").alias("uk_s"), pl.col("nu").alias("r_nu"), pl.col("fz").alias("r_fz")))
        later = n_ai.join(r, on="ncid", how="inner")
        tg = (st.read("tags", s).filter(pl.col("n_ai_rows") == 1)
              .select("ncid", pl.col("unit_key").alias("uk_s"), pl.col("unit_key").alias("dp_s"),
                      pl.col("voter_tag").alias("dp_tag_s"), pl.col("trackable_voter").alias("trackable_s")))
        j = (base.join(later, on="ncid", how="left")
             .join(tg, on=["ncid", "uk_s"], how="left")
             .join(canon.rename({"bldg_key": "r_bldg", "canon_bldg": "cb_s"}), on="r_bldg", how="left"))
        j = j.with_columns(pl.lit(s).alias("snapshot_dt"), pl.lit(s.year - T0.year, dtype=pl.Int64).alias("k"))
        out.append(j)
    j = pl.concat(out, how="vertical")
    absent = pl.col("n_ai").is_null()
    relabel = (pl.col("cb0").is_not_null() & pl.col("cb_s").is_not_null() & (pl.col("cb0") == pl.col("cb_s"))
               & (pl.col("r_nu") == pl.col("nu0")) & (pl.col("r_bldg") != pl.col("bk0"))).fill_null(False)
    j = j.with_columns(relabel.alias("relabel"))

    def ev(stay):
        return (pl.when(absent).then(pl.lit("exit_absent"))
                .when((pl.col("n_ai") == 0) & (pl.col("reason_s") == "RD")).then(pl.lit("exit_died"))
                .when((pl.col("n_ai") == 0) & (pl.col("reason_s") == "RS")).then(pl.lit("exit_left_state"))
                .when(pl.col("n_ai") == 0).then(pl.lit("exit_maintenance"))
                .when(pl.col("n_ai") > 1).then(pl.lit("dup_rows"))
                .when(stay).then(pl.lit("stay"))
                .when((pl.col("r_fz") == pl.col("fz0")) | pl.col("relabel")).then(pl.lit("format_change"))
                .when(pl.col("r_bldg") == pl.col("bk0")).then(pl.lit("move_building"))
                .when(pl.col("r_zip5") == pl.col("zip0")).then(pl.lit("move_local"))
                .otherwise(pl.lit("move_far")))
    return j.select(
        "ncid", "client", "dp0", "uk0", "core", "size0", "dp_tag0", "voter_tag0", "age0", "county0", "snapshot_dt", "k",
        "dp_s", "uk_s", "dp_tag_s", "trackable_s", pl.col("n_ai").cast(pl.Int64), "status_s", "reason_s",
        ev(pl.col("dp_s") == pl.col("dp0")).alias("event_list"), ev(pl.col("uk_s") == pl.col("uk0")).alias("event_voter"),
        (pl.col("relabel") & ~(pl.col("r_fz") == pl.col("fz0")).fill_null(False)).alias("relabel_only"),
        ((pl.col("n_ai") == 1) & (pl.col("r_county") != pl.col("county0"))).alias("county_change"))


def household_events(st: S.Store, pe: pl.DataFrame) -> pl.DataFrame:
    p = pe.with_columns(
        (~pl.col("event_list").str.starts_with("exit") & (pl.col("event_list") != "dup_rows")).alias("present"),
        pl.when(pl.col("dp_s").is_not_null()).then(pl.concat_str([pl.lit("d"), pl.col("dp_s").cast(pl.String)]))
        .otherwise(pl.concat_str([pl.lit("u"), pl.col("uk_s").cast(pl.String)])).alias("dest"))
    dest_mix = (p.filter(pl.col("present") & pl.col("dest").is_not_null()).group_by("snapshot_dt", "dest")
                .agg(pl.col("dp0").n_unique().alias("n_hh0")))
    hh0 = p.select("dp0").unique()
    mem0 = p.select("dp0", "ncid").unique()
    ent = []
    for s in LATER:
        t = (st.read("tags", s, columns=["ncid", "unit_key"]).rename({"unit_key": "dp0"})
             .join(hh0, on="dp0", how="semi").join(mem0, on=["dp0", "ncid"], how="anti"))
        ent.append(t.group_by("dp0").agg(pl.len().cast(pl.Int64).alias("n_entrants")).with_columns(pl.lit(s).alias("snapshot_dt")))
    ent = pl.concat(ent)
    q = p.join(dest_mix, on=["snapshot_dt", "dest"], how="left")
    moved = pl.col("present") & (pl.col("event_list") != "stay")
    h = q.group_by("dp0", "snapshot_dt").agg(
        pl.col("k").first(), pl.col("size0").first(), logical_and(pl.col("core")).alias("core"), pl.col("dp_tag0").first(),
        pl.len().cast(pl.Int64).alias("m0"), (pl.col("event_list") == "stay").sum().cast(pl.Int64).alias("n_stay"),
        (~pl.col("present")).sum().cast(pl.Int64).alias("n_exit"), moved.sum().cast(pl.Int64).alias("n_moved"),
        pl.col("dest").filter(pl.col("present")).drop_nulls().n_unique().cast(pl.Int64).alias("n_dest"),
        logical_or(moved & (pl.col("n_hh0") > 1)).alias("merged"))
    h = h.join(ent, on=["dp0", "snapshot_dt"], how="left").with_columns(pl.col("n_entrants").fill_null(0))
    ne = pl.col("n_entrants")
    return h.with_columns(
        pl.when(pl.col("n_exit") == pl.col("m0")).then(pl.lit("dissolved"))
        .when(pl.col("n_dest") >= 2).then(pl.lit("split"))
        .when(pl.col("n_stay") == 0).then(pl.lit("moved_together"))
        .when((pl.col("n_exit") == 0) & (ne == 0)).then(pl.lit("stable"))
        .when(pl.col("n_exit") == 0).then(pl.lit("grow"))
        .when(ne == 0).then(pl.lit("shrink"))
        .otherwise(pl.lit("shrink_grow")).alias("people_event"),
        pl.when(pl.col("n_stay") > 0).then(pl.lit("held")).when(ne > 0).then(pl.lit("replaced")).otherwise(pl.lit("emptied")).alias("address_event"))
