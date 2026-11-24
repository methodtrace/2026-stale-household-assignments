# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Stage 2: tables derived from the typed address snapshots, statewide.

household_tags   per snapshot: active-or-inactive registrants with a building key, with building and unit counts, the
                 voter-file housing tag and the trackable flag (trackable: one active-or-inactive row in the snapshot,
                 tag SFH_LIKELY or UNIT_IN_MULTIUNIT, 1 to 5 registrants at the unit).
graph_records    per snapshot: the vendor-visible records (name fingerprints and unit key) of every tagged registrant;
                 akey = ukey = 'u' + unit key; tk = the reference household key with the readdressing rule (for a building
                 in a relabel component, the canonical building plus the normalized unit text).
age_flags        records with age 0 or above 110 with the age implied by the nearest plausible age of the same NCID, and the
                 minor flag (implied age under 18, or age 0 with no plausible age anywhere).
relabel_edges    building relabel edges between consecutive snapshots; canon: their connected components, each labelled
                 by its smallest building key.
"""
import polars as pl

from . import record_functions as rf

TAG_GROUP = "GROUP_QUARTERS_OR_OTHER"
MHP_RE = r"(^|[\t\n\f\r ])(MOBILE|MHP|TRAILER|TRLR|MH PARK|MOBILE HOME|CAMPGROUND|RV PARK|MANUFACTURED)([\t\n\f\r ]|$)"


def household_tags(addr: pl.DataFrame) -> pl.DataFrame:
    p = addr.filter(pl.col("is_ai").fill_null(False) & pl.col("bldg_key").is_not_null()).select(
        "snapshot_dt", "ncid", "status_cd", "age", "bldg_key", "unit_key", "unit", "unit_prefix", "street_name")
    pc = p.filter(pl.col("ncid").is_not_null()).group_by("ncid").agg(pl.len().alias("n_ai_rows"))
    b = p.group_by("bldg_key").agg(
        pl.len().alias("b_n"), pl.col("unit").is_not_null().sum().alias("b_nu"),
        ((pl.col("age") >= 18) & (pl.col("age") <= 24)).fill_null(False).sum().alias("b_y"),
        (pl.col("age") >= 80).fill_null(False).sum().alias("b_o"))
    u = p.group_by("unit_key").agg(
        pl.len().alias("u_n"), pl.col("unit").is_not_null().any().alias("u_hu"),
        (pl.col("status_cd") == "A").fill_null(False).sum().alias("u_na"),
        (pl.col("unit_prefix").is_in(["LOT", "TRLR", "SPC", "SPACE"]).fill_null(False)
         | pl.col("street_name").str.contains(MHP_RE).fill_null(False)).any().alias("u_mhp"),
        pl.col("unit_prefix").is_in(["STE", "SUITE", "OFC", "RM", "ROOM"]).fill_null(False).any().alias("u_inst"))
    t = p.join(pc, on="ncid", how="inner").join(b, on="bldg_key", how="inner").join(u, on="unit_key", how="inner")
    tag = (pl.when(pl.col("u_n") > 20).then(pl.lit(TAG_GROUP))
           .when((pl.col("b_n") >= 21) & (pl.col("b_y") / pl.col("b_n") >= 0.6)).then(pl.lit(TAG_GROUP))
           .when((pl.col("b_n") >= 6) & (pl.col("b_o") / pl.col("b_n") >= 0.5)).then(pl.lit(TAG_GROUP))
           .when(pl.col("u_inst")).then(pl.lit(TAG_GROUP))
           .when(pl.col("u_mhp")).then(pl.lit("MOBILE_HOME_PARK"))
           .when(pl.col("u_hu") & (pl.col("u_n") <= 8)).then(pl.lit("UNIT_IN_MULTIUNIT"))
           .when(pl.col("u_hu")).then(pl.lit("UNIT_LARGE_9_20"))
           .when(pl.col("b_nu") > 0).then(pl.lit("MULTIUNIT_NO_UNIT"))
           .when(pl.col("u_n") >= 9).then(pl.lit("NOUNIT_LARGE_9_20"))
           .otherwise(pl.lit("SFH_LIKELY")))
    t = t.select(
        "snapshot_dt", "ncid", "bldg_key", "unit_key", "status_cd", "age", pl.col("unit").is_not_null().alias("has_unit"),
        pl.col("n_ai_rows").cast(pl.Int64), tag.alias("voter_tag"), pl.col("b_n").cast(pl.Int64).alias("bldg_n"),
        pl.col("u_n").cast(pl.Int64).alias("unit_n"), (pl.col("u_na") == pl.col("u_n")).alias("unit_all_active"))
    return t.with_columns(
        ((pl.col("n_ai_rows") == 1) & pl.col("voter_tag").is_in(["SFH_LIKELY", "UNIT_IN_MULTIUNIT"])
         & (pl.col("unit_n") >= 1) & (pl.col("unit_n") <= 5)).alias("trackable_voter"),
        pl.col("unit_n").alias("size_voter"))


def ukey_expr(c="unit_key"):
    return pl.concat_str([pl.lit("u"), pl.col(c).cast(pl.String)])


def graph_records(tags: pl.DataFrame, addr: pl.DataFrame, canon: pl.DataFrame) -> pl.DataFrame:
    """One row per (tag row, matching active-or-inactive address row), as the specification's join produces."""
    a = addr.filter(pl.col("is_ai").fill_null(False) & pl.col("unit_key").is_not_null())
    g = tags.select("snapshot_dt", "ncid", "unit_key", "n_ai_rows").join(
        a.select("ncid", "unit_key", "first_fp", pl.col("surname_norm_fp").alias("sur_fp")),
        on=["ncid", "unit_key"], how="inner")
    urx = (a.join(canon, on="bldg_key", how="inner")
           .select(ukey_expr().alias("ukey"),
                   pl.concat_str([pl.col("canon_bldg").cast(pl.String), pl.lit("|"),
                                  pl.col("unit").fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "")]).alias("rx_s")))
    urx = urx.with_columns(pl.concat_str([pl.lit("r"), rf.fp_col(urx["rx_s"]).cast(pl.String)]).alias("ukey_rx"))
    urx = urx.group_by("ukey").agg(pl.col("ukey_rx").min(), pl.col("ukey_rx").n_unique().alias("_n"))
    # one relabel key per unit key, so the reference key computed here equals the one the scoring step derives
    assert urx["_n"].max() is None or urx["_n"].max() == 1, "a unit key maps to two relabel keys"
    urx = urx.drop("_n")
    g = g.with_columns(ukey_expr().alias("ukey")).join(urx, on="ukey", how="left")
    # stored per snapshot file, so the date and series columns are implied; akey equals ukey for the address-only builder
    return g.select("ncid", "first_fp", "sur_fp", pl.col("ukey").alias("akey"), pl.coalesce("ukey_rx", "ukey").alias("tk"),
                    "n_ai_rows")


def relabel_pair(a0: pl.DataFrame, a1: pl.DataFrame, s0, s1) -> pl.DataFrame:
    """Relabel edges between consecutive snapshots s0 < s1 (specification: building relabel crosswalk)."""
    def x_of(a):
        x = a.filter(pl.col("is_ai").fill_null(False) & pl.col("bldg_key").is_not_null() & pl.col("ncid").is_not_null()
                     & (pl.col("ncid") != "")).select(
            "ncid", "bldg_key", "county_id", pl.col("house_num").str.replace_all(r"[^0-9]", "").alias("hn"),
            pl.col("street_name").alias("st"),
            pl.concat_str(["half_code", "street_dir", "street_type_cd", "street_sufx_cd"], separator="|").alias("aux"), "zip5")
        return x.with_columns(pl.len().over("ncid").alias("n_ai"))
    x0, x1 = x_of(a0), x_of(a1)
    p0 = x0.filter(pl.col("n_ai") == 1).drop("n_ai")
    p1 = x1.filter(pl.col("n_ai") == 1).drop("n_ai")
    bl0 = x0.select("bldg_key").unique()
    bl1 = x1.select("bldg_key").unique()
    alive1 = x1.select("ncid").unique()
    b0pop = p0.join(alive1, on="ncid", how="inner").group_by("bldg_key").agg(pl.len().alias("n_persist")).rename({"bldg_key": "b0"})
    mv = p0.join(p1, on="ncid", how="inner", suffix="_1").filter(pl.col("bldg_key") != pl.col("bldg_key_1"))
    sx = rf.soundex_col(pl.concat([mv["st"], mv["st_1"]]))
    mv = mv.with_columns(sx[: mv.height].alias("sx0"), sx[mv.height:].alias("sx1"))
    same_hn = pl.col("hn") == pl.col("hn_1")
    same_st = pl.col("st") == pl.col("st_1")
    same_aux = pl.col("aux") == pl.col("aux_1")
    same_zip = pl.col("zip5") == pl.col("zip5_1")
    pat = (pl.when(same_hn & same_st & same_aux & ~same_zip).then(pl.lit("zip_only"))
           .when(same_hn & ~same_st & (pl.col("sx0") == pl.col("sx1")) & same_zip).then(pl.lit("st_spell"))
           .when(same_hn & (pl.col("sx0") != pl.col("sx1")) & same_zip).then(pl.lit("st_rename"))
           .when(same_hn & same_st & ~same_aux & same_zip).then(pl.lit("aux_only"))
           .when(~same_hn & same_st & same_aux & same_zip).then(pl.lit("renumber")))
    mv = mv.select(pl.col("bldg_key").alias("b0"), pl.col("bldg_key_1").alias("b1"), "county_id", pat.alias("pat"),
                   pl.concat_str(["zip5", "st", "aux"], separator="|").alias("street0"))
    mv = mv.filter(pl.col("pat").is_not_null())
    mv = mv.join(bl1.rename({"bldg_key": "b0"}), on="b0", how="anti").join(bl0.rename({"bldg_key": "b1"}), on="b1", how="anti")
    # county_id, pat and street0: pat and street0 are functions of (b0, b1); county_id is informational only
    e = mv.group_by("b0", "b1").agg(pl.col("county_id").min(), pl.col("pat").min(), pl.col("street0").min(),
                                    pl.len().alias("n_movers"))
    e2 = e.join(b0pop, on="b0", how="inner").filter(pl.col("n_movers") >= 0.8 * pl.col("n_persist"))
    e2 = e2.with_columns(pl.len().over("pat", "street0").alias("n_on_street"))
    out = e2.filter((pl.col("pat") != "renumber") | (pl.col("n_on_street") >= 3))
    return out.select(pl.lit(s0).alias("s0"), pl.lit(s1).alias("s1"), "b0", "b1", "county_id", "pat",
                      pl.col("n_movers").cast(pl.Int64), pl.col("n_persist").cast(pl.Int64))


def canon_from_edges(edges: pl.DataFrame) -> pl.DataFrame:
    """Connected components of the undirected edges, each key mapped to the smallest key in its component."""
    parent = {}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for a, b in zip(edges["b0"].to_list(), edges["b1"].to_list()):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    keys = sorted(parent)
    return pl.DataFrame({"bldg_key": keys, "canon_bldg": [find(k) for k in keys]}, schema={"bldg_key": pl.Int64, "canon_bldg": pl.Int64})


def age_flags(addr_by_date, flag_dates=None) -> pl.DataFrame:
    """addr_by_date: list of (date, callable returning that date's address table). Two passes over the snapshots.
    flag_dates: flag only the records of these dates (default every date); every date is an age source."""
    f_parts = []
    for d, load in addr_by_date:
        if flag_dates is not None and d not in flag_dates:
            continue
        a = load().select("ncid", "is_ai", "age")
        f_parts.append(a.filter(pl.col("is_ai").fill_null(False) & ((pl.col("age") == 0) | (pl.col("age") > 110))
                                & pl.col("ncid").is_not_null() & (pl.col("ncid") != ""))
                       .select(pl.lit(d).alias("snapshot_dt"), "ncid", "age").unique())
    f = pl.concat(f_parts)
    ids = f.select("ncid").unique()
    ok = []
    for d, load in addr_by_date:
        a = load().select("ncid", "age").join(ids, on="ncid", how="semi")
        ok.append(a.filter((pl.col("age") >= 18) & (pl.col("age") <= 110)).unique()
                  .select(pl.lit(d).alias("ok_dt"), "ncid", pl.col("age").alias("ok_age")))
    ok = pl.concat(ok)
    j = f.join(ok, on="ncid", how="left")
    j = j.with_columns(
        (pl.col("snapshot_dt") - pl.col("ok_dt")).dt.total_days().abs().alias("dist"),
        (pl.col("ok_age") + pl.col("snapshot_dt").dt.year().cast(pl.Int64) - pl.col("ok_dt").dt.year().cast(pl.Int64)).alias("imp"))
    # nearest snapshot with a plausible age; ties (equal distance) resolved by the earlier snapshot, then the smaller age
    near = (j.sort(["snapshot_dt", "ncid", "age", "dist", "ok_dt", "ok_age"], nulls_last=True)
            .group_by(["snapshot_dt", "ncid", "age"], maintain_order=True).agg(pl.col("imp").first().alias("age_imp")))
    return near.select(
        "snapshot_dt", "ncid", "age", "age_imp",
        pl.when(pl.col("age") == 0).then(pl.lit("age0")).otherwise(pl.lit("age_gt110")).alias("flag"),
        pl.coalesce(pl.col("age_imp") < 18, pl.col("age") == 0).alias("minor")).sort(["snapshot_dt", "ncid", "age"])


# ---- NCID re-issue crosswalk -------------------------------------------------------------------------------------------
# One person, two NCIDs in sequence: an old NCID ends (its last active-or-inactive snapshot L is before 2026-01-01) and a new
# NCID first appears (any row) at F, after the old NCID's first appearance and no later than L + 3 years. Pairs share first
# name, normalized surname, middle-initial and suffix fingerprints and implied birth year (snapshot year minus age) within
# one year; only pairs unique in both directions are kept. The state is built one snapshot at a time, in date order, before
# stage 2 drops the name columns: per NCID the first snapshot and its record (active-or-inactive rows first, then the latest
# registration date), and the last active-or-inactive snapshot and its record (latest registration date). Remaining ties
# are broken on the record's own fields, so the choice is fixed.

REISSUE_FIELDS = ["first_fp", "surname_norm_fp", "mi", "sx", "byr", "county_id", "unit_key"]


def _reissue_rows(addr: pl.DataFrame, d) -> pl.DataFrame:
    return (addr.filter(pl.col("ncid").is_not_null() & (pl.col("ncid") != ""))
            .select("ncid", pl.col("is_ai").fill_null(False).alias("ai"), "registr_dt", "first_fp", "surname_norm_fp",
                    pl.col("mid_init_fp").fill_null(0).alias("mi"), pl.col("name_sufx").fill_null("").alias("sx"),
                    (pl.lit(d.year) - pl.col("age")).cast(pl.Int64).alias("byr"), "county_id", "unit_key"))


def _pick(x: pl.DataFrame, by_ai: bool) -> pl.DataFrame:
    cols = (["ai"] if by_ai else []) + ["registr_dt"] + REISSUE_FIELDS
    desc = ([True] if by_ai else []) + [True] + [False] * len(REISSUE_FIELDS)
    return x.sort(["ncid"] + cols, descending=[False] + desc, nulls_last=True).group_by("ncid", maintain_order=True).first()


def reissue_update(state: pl.DataFrame | None, addr: pl.DataFrame, d) -> pl.DataFrame:
    """Apply snapshot d (dates must arrive in increasing order) to the crosswalk state."""
    x = _reissue_rows(addr, d)
    f = _pick(x, True).select("ncid", pl.lit(d).alias("first_dt"), *[pl.col(c).alias("f_" + c) for c in REISSUE_FIELDS])
    l = _pick(x.filter(pl.col("ai")), False).select("ncid", pl.lit(d).alias("last_ai_dt"),
                                                      *[pl.col(c).alias("l_" + c) for c in REISSUE_FIELDS])
    if state is None:
        return f.join(l, on="ncid", how="left")
    new = f.join(state.select("ncid"), on="ncid", how="anti")
    st = pl.concat([state, new.join(l.head(0), on="ncid", how="left")], how="vertical_relaxed")
    keep = st.join(l.select("ncid"), on="ncid", how="anti")
    upd = st.join(l.select("ncid"), on="ncid", how="semi").drop(["last_ai_dt"] + ["l_" + c for c in REISSUE_FIELDS]).join(l, on="ncid")
    return pl.concat([keep, upd.select(keep.columns)]).sort("ncid")


def reissue_crosswalk(state: pl.DataFrame) -> pl.DataFrame:
    import datetime as dt
    o = state.filter(pl.col("last_ai_dt").is_not_null() & (pl.col("last_ai_dt") < dt.date(2026, 1, 1))
                     & pl.col("l_first_fp").is_not_null() & pl.col("l_surname_norm_fp").is_not_null())
    n = state.filter(pl.col("first_dt") > dt.date(2008, 11, 4))
    key = ["first_fp", "surname_norm_fp", "mi", "sx"]
    c = (o.select(pl.col("ncid").alias("old_ncid"), pl.col("first_dt").alias("old_first_dt"), "last_ai_dt",
                  *[pl.col("l_" + k).alias(k) for k in key], pl.col("l_byr").alias("obyr"))
         .join(n.select(pl.col("ncid").alias("new_ncid"), pl.col("first_dt").alias("new_first_dt"),
                        *[pl.col("f_" + k).alias(k) for k in key], pl.col("f_byr").alias("nbyr")), on=key, how="inner")
         .filter(((pl.col("nbyr") - pl.col("obyr")).abs() <= 1) & (pl.col("new_first_dt") > pl.col("old_first_dt"))
                 & (pl.col("new_first_dt") <= pl.col("last_ai_dt").dt.offset_by("3y")) & (pl.col("new_ncid") != pl.col("old_ncid"))))
    c = c.with_columns(pl.len().over("old_ncid").alias("n_o"), pl.len().over("new_ncid").alias("n_n"))
    return (c.filter((pl.col("n_o") == 1) & (pl.col("n_n") == 1))
            .select("old_ncid", "new_ncid", "last_ai_dt", "new_first_dt").sort("old_ncid"))
