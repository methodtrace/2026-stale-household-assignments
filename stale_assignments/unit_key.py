# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Unit-key audit: voter building and unit keys against North Carolina address points (NC OneMap AddressNC).

Both sides are reduced to a building match key built from uppercased text only:
  hn    house number digits (a letter or fraction suffix becomes a unit discriminator)
  full  sorted street tokens after USPS-style abbreviation (directions, street types, highway words)
  core  the same without direction, street-type and highway-class tokens (name tokens only)
Matching tiers: T1 zip5 + hn + full, T2 zip5 + hn + core, T3 county + hn + core, each accepted only when the key names
exactly one address-point building. Per voter building (county, zip5, hn, full, core; active or inactive rows with a
building key):
  k_v       distinct unit discriminators among the registrants, a registrant without one counting as one more
  d         max(1, distinct unit discriminators among the address points of the matched building)
  disagree  (d = 1 and k_v > 1) or (d >= 2 and some registrant has no unit) or (k_v > d)
  pooled    d >= 2, at least two registrants, none with a unit (the voter key merges the building into one household)
Only per-county integer counts are kept (unit_counts); the shares are computed from them.
"""
import glob
import os
import re

import polars as pl

AUDIT_DATES = ["2016-11-08", "2018-11-06", "2020-11-03", "2022-11-08", "2024-11-05", "2026-01-01"]
AP_STREET = ["st_predir", "st_premod", "st_pretyp", "st_name", "st_postyp", "st_posdir", "st_posmod"]
V_STREET = ["street_dir", "street_name", "street_type_cd", "street_sufx_cd"]
CLASSES = ["single_agree", "single_split_units", "single_split_mixed", "multi_no_unit", "multi_partial", "multi_excess",
           "multi_agree"]
I = pl.Int64

DIRS = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "NORTHEAST": "NE", "NORTHWEST": "NW",
        "SOUTHEAST": "SE", "SOUTHWEST": "SW", "N": "N", "S": "S", "E": "E", "W": "W", "NE": "NE", "NW": "NW",
        "SE": "SE", "SW": "SW"}
# USPS Publication 28 C1 primary suffix abbreviations (common subset seen in NC data), plus highway words.
TYPES = {
    "ALLEY": "ALY", "ALY": "ALY", "AVENUE": "AVE", "AVE": "AVE", "AV": "AVE", "BEND": "BND", "BND": "BND",
    "BOULEVARD": "BLVD", "BLVD": "BLVD", "BRANCH": "BR", "BR": "BR", "BROOK": "BRK", "BRK": "BRK", "BYPASS": "BYP",
    "BYP": "BYP", "CIRCLE": "CIR", "CIR": "CIR", "CIRCLES": "CIRS", "COMMON": "CMN", "CMN": "CMN", "CORNER": "COR",
    "COURT": "CT", "CT": "CT", "COURTS": "CTS", "COVE": "CV", "CV": "CV", "CREEK": "CRK", "CRK": "CRK",
    "CRESCENT": "CRES", "CRES": "CRES", "CROSSING": "XING", "XING": "XING", "CROSSROADS": "XRDS", "XRDS": "XRDS",
    "DRIVE": "DR", "DR": "DR", "ESTATE": "EST", "EST": "EST", "ESTATES": "ESTS", "ESTS": "ESTS", "EXPRESSWAY": "EXPY",
    "EXTENSION": "EXT", "EXTENDED": "EXT", "EXT": "EXT", "FREEWAY": "FWY", "GLEN": "GLN", "GLN": "GLN", "GREEN": "GRN",
    "GRN": "GRN", "GROVE": "GRV", "GRV": "GRV", "HARBOR": "HBR", "HBR": "HBR", "HEIGHTS": "HTS", "HTS": "HTS",
    "HIGHWAY": "HWY", "HWY": "HWY", "HILL": "HL", "HL": "HL", "HILLS": "HLS", "HLS": "HLS", "HOLLOW": "HOLW",
    "HOLW": "HOLW", "JUNCTION": "JCT", "KNOLL": "KNL", "KNL": "KNL", "KNOLLS": "KNLS", "LANDING": "LNDG",
    "LNDG": "LNDG", "LANE": "LN", "LN": "LN", "LOOP": "LOOP", "MEADOW": "MDW", "MEADOWS": "MDWS", "MDWS": "MDWS",
    "MEWS": "MEWS", "MOUNTAIN": "MTN", "MTN": "MTN", "OVERLOOK": "OVLK", "OVLK": "OVLK", "PARK": "PARK",
    "PARKWAY": "PKWY", "PKWY": "PKWY", "PASS": "PASS", "PATH": "PATH", "PIKE": "PIKE", "PLACE": "PL", "PL": "PL",
    "PLAZA": "PLZ", "PLZ": "PLZ", "POINT": "PT", "PT": "PT", "POINTE": "PT", "RIDGE": "RDG", "RDG": "RDG",
    "ROAD": "RD", "RD": "RD", "ROW": "ROW", "RUN": "RUN", "SPUR": "SPUR", "SQUARE": "SQ", "SQ": "SQ",
    "STATION": "STA", "STA": "STA", "STREET": "ST", "ST": "ST", "TERRACE": "TER", "TER": "TER", "TRACE": "TRCE",
    "TRCE": "TRCE", "TRAIL": "TRL", "TRL": "TRL", "VALLEY": "VLY", "VLY": "VLY", "VIEW": "VW", "VW": "VW",
    "VILLAGE": "VLG", "VLG": "VLG", "VISTA": "VIS", "VIS": "VIS", "WALK": "WALK", "WAY": "WAY", "WOODS": "WDS",
    "BUSINESS": "BUS", "BUS": "BUS", "INTERSTATE": "I", "OLD": "OLD"}
# tokens dropped from the core key: directions, street types (abbreviated), highway class words
_TYPE_ABBR = set(TYPES.values()) - {"OLD"}
CORE_DROP = set(DIRS.values()) | _TYPE_ABBR | {"US", "NC", "SR", "I", "HWY", "STATE", "ROUTE", "RT", "RTE"}
PHRASES = [(r"\bNORTH CAROLINA\b", "NC"), (r"\bUNITED STATES\b", "US"), (r"\bSTATE ROAD\b", "SR"),
           (r"\bSTATE ROUTE\b", "SR"), (r"\bSAINT\b", "ST"), (r"\bMOUNT\b", "MT")]


def norm_tokens(parts):
    s = " ".join(p for p in parts if p).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    for a, b in PHRASES:
        s = re.sub(a, b, s)
    out = []
    for t in s.split():
        t = DIRS.get(t, TYPES.get(t, t))
        out.append(t)
    return out


def keys_from_parts(parts):
    toks = norm_tokens(parts)
    full = " ".join(sorted(toks))
    core = " ".join(sorted(t for t in toks if t not in CORE_DROP))
    return full, (core if core else full)


def street_keys(df: pl.DataFrame, cols) -> pl.DataFrame:
    """Add full and core street keys from the listed text columns (computed once per distinct street)."""
    st = df.select(cols).unique().sort(cols, nulls_last=True)
    fk, ck = [], []
    for r in st.rows():
        f, c = keys_from_parts(r)
        fk.append(f)
        ck.append(c)
    st = st.with_columns(pl.Series("full", fk, dtype=pl.String), pl.Series("core", ck, dtype=pl.String))
    return df.join(st, on=cols, how="left", nulls_equal=True)


def house_parts(col: str):
    """House number digits and the non-digit remainder (suffix), from a house-number string."""
    c = pl.col(col).fill_null("").str.to_uppercase().str.strip_chars()
    return c.str.extract(r"^0*(\d+)", 1), c.str.replace(r"^\d+", "").str.replace_all(r"[\s\-#\.]", "")


def unit_norm(expr: pl.Expr) -> pl.Expr:
    """Same normalization as the voter unit: strip a leading designator, then spaces, #, - and dots."""
    e = expr.fill_null("").str.to_uppercase().str.strip_chars()
    e = e.str.replace(r"^(APT|APARTMENT|UNIT|STE|SUITE|LOT|RM|ROOM|BLDG|TRLR|SPC|#)\.?\s*", "")
    e = e.str.replace_all(r"[\s#\-\.]", "")
    return pl.when(e == "").then(None).otherwise(e)


def load_address_points(ap_dir: str) -> pl.DataFrame:
    """Address points from ap_*.csv.gz (fetch_address_points.py), reduced to county, zip5, hn, street keys and unit."""
    fs = sorted(glob.glob(os.path.join(ap_dir, "ap_*.csv.gz")))
    if not fs:
        raise FileNotFoundError(f"no ap_*.csv.gz in {ap_dir}")
    ap = pl.concat([pl.read_csv(f, infer_schema=False) for f in fs], how="vertical")
    hn, rest = house_parts("add_number")
    ap = ap.with_columns(
        hn.alias("hn"),
        pl.col("post_code").fill_null("").str.replace_all(r"[^0-9]", "").str.slice(0, 5).alias("zip5"),
        ((pl.col("countyfips").cast(I, strict=False) + 1) // 2).alias("county_id"),
        pl.concat_str([rest, unit_norm(pl.col("addnum_suf")).fill_null("")]).alias("suf"),
        unit_norm(pl.col("building")).alias("bld"),
        unit_norm(pl.col("unit")).alias("u"))
    ap = ap.with_columns(
        pl.when((pl.col("suf") == "") & pl.col("bld").is_null() & pl.col("u").is_null()).then(None)
        .otherwise(pl.concat_str([pl.col("suf"), pl.col("bld").fill_null(""), pl.col("u").fill_null("")], separator="|"))
        .alias("ud"))
    ap = ap.filter(pl.col("hn").is_not_null() & (pl.col("zip5").str.len_chars() == 5) & pl.col("st_name").is_not_null())
    return street_keys(ap, AP_STREET).select("county_id", "zip5", "hn", "full", "core", "ud")


def _groups(ap, keys):
    return ap.group_by(keys).agg(pl.col("ud").drop_nulls().n_unique().cast(I).alias("ap_ud"),
                                 pl.struct("zip5", "full").n_unique().alias("ap_nb"))


def unit_counts(addr: pl.DataFrame, ap: pl.DataFrame, d) -> pl.DataFrame:
    """Per-county integer counts for one snapshot's address table against the address points."""
    v = addr.filter(pl.col("is_ai").fill_null(False) & pl.col("bldg_key").is_not_null())
    hn, rest = house_parts("house_num")
    v = v.with_columns(hn.alias("hn"),
                       pl.concat_str([rest, pl.col("half_code").fill_null("").str.replace_all(r"[\s\-#\.]", "")]).alias("suf"))
    v = v.with_columns(pl.when((pl.col("suf") == "") & pl.col("unit").is_null()).then(None)
                       .otherwise(pl.concat_str([pl.col("suf"), pl.col("unit").fill_null("")], separator="|")).alias("ud"))
    v = street_keys(v.filter(pl.col("hn").is_not_null()), V_STREET)
    vb = v.group_by("county_id", "zip5", "hn", "full", "core").agg(
        pl.len().cast(I).alias("n_reg"), pl.col("ud").drop_nulls().n_unique().cast(I).alias("v_ud"),
        pl.col("ud").is_null().sum().cast(I).alias("v_nounit"))
    g1 = _groups(ap, ["zip5", "hn", "full"]).select("zip5", "hn", "full", "ap_ud")
    g2 = _groups(ap, ["zip5", "hn", "core"]).filter(pl.col("ap_nb") == 1).select("zip5", "hn", "core", "ap_ud")
    g3 = _groups(ap, ["county_id", "hn", "core"]).filter(pl.col("ap_nb") == 1).select("county_id", "hn", "core", "ap_ud")
    m1 = vb.join(g1, on=["zip5", "hn", "full"], how="left")
    hit1, un = m1.filter(pl.col("ap_ud").is_not_null()), m1.filter(pl.col("ap_ud").is_null()).drop("ap_ud")
    m2 = un.join(g2, on=["zip5", "hn", "core"], how="left")
    hit2, un2 = m2.filter(pl.col("ap_ud").is_not_null()), m2.filter(pl.col("ap_ud").is_null()).drop("ap_ud")
    m3 = un2.join(g3, on=["county_id", "hn", "core"], how="left")
    m = pl.concat([hit1, hit2, m3], how="vertical")
    assert m.height == vb.height, (m.height, vb.height)
    m = m.with_columns((pl.col("v_ud") + (pl.col("v_nounit") > 0).cast(I)).alias("k_v"),
                       pl.max_horizontal(pl.lit(1, dtype=I), pl.col("ap_ud")).alias("d"), (pl.col("n_reg") >= 2).alias("multi"),
                       pl.col("ap_ud").is_not_null().alias("matched"))
    m = m.with_columns(
        (((pl.col("d") == 1) & (pl.col("k_v") > 1)) | ((pl.col("d") >= 2) & (pl.col("v_nounit") > 0))
         | (pl.col("k_v") > pl.col("d"))).alias("disagree"),
        ((pl.col("d") >= 2) & (pl.col("v_ud") == 0) & pl.col("multi")).alias("pooled"),
        (pl.col("multi") & (pl.col("v_ud") == 0)).alias("multi_nounit"),
        pl.when(~pl.col("matched")).then(pl.lit("unmatched"))
        .when((pl.col("d") == 1) & (pl.col("k_v") == 1)).then(pl.lit("single_agree"))
        .when((pl.col("d") == 1) & (pl.col("v_nounit") == 0)).then(pl.lit("single_split_units"))
        .when(pl.col("d") == 1).then(pl.lit("single_split_mixed"))
        .when(pl.col("v_ud") == 0).then(pl.lit("multi_no_unit"))
        .when(pl.col("v_nounit") > 0).then(pl.lit("multi_partial"))
        .when(pl.col("k_v") > pl.col("d")).then(pl.lit("multi_excess"))
        .otherwise(pl.lit("multi_agree")).alias("cls"))
    b = lambda c: pl.col(c).cast(I)
    nr = pl.col("n_reg")
    mm = m.filter("matched")
    per = mm.group_by("county_id").agg(
        pl.len().cast(I).alias("bld"), nr.sum().alias("reg"),
        b("disagree").sum().alias("bld_disagree"), (b("disagree") * nr).sum().alias("reg_disagree"),
        b("multi").sum().alias("bld_multi"), (b("multi") & b("disagree")).sum().alias("bld_multi_disagree"),
        (b("multi") * b("disagree") * nr).sum().alias("reg_multi_disagree"), (b("multi") * nr).sum().alias("reg_multi"),
        b("multi_nounit").sum().alias("bld_multi_nounit"), (b("multi_nounit") * nr).sum().alias("reg_multi_nounit"),
        pl.when((pl.col("d") >= 2) & (pl.col("v_nounit") >= 2)).then(pl.col("v_nounit")).otherwise(0).sum()
        .alias("reg_nounit_pool_apmulti"),
        b("pooled").sum().alias("bld_pooled"), (b("pooled") * nr).sum().alias("reg_pooled"),
        *[(pl.col("cls") == c).cast(I).sum().alias(f"bld_{c}") for c in CLASSES],
        *[((pl.col("cls") == c).cast(I) * nr).sum().alias(f"reg_{c}") for c in CLASSES])
    allb = m.group_by("county_id").agg(pl.len().cast(I).alias("bld_all"), nr.sum().alias("reg_all"),
                                       b("multi").sum().alias("bld_multi_all"), b("multi_nounit").sum().alias("bld_multi_nounit_all"))
    out = allb.join(per, on="county_id", how="left").with_columns(pl.lit(d).alias("snapshot_dt"))
    cnt = [c for c in out.columns if c not in ("county_id", "snapshot_dt")]
    return out.with_columns(pl.col(cnt).fill_null(0).cast(I)).select("snapshot_dt", "county_id", *cnt).sort("county_id")


SHARES = [("match_rate_bld", ["bld"], ["bld_all"]), ("disagree_bld", ["bld_disagree"], ["bld"]),
          ("disagree_reg", ["reg_disagree"], ["reg"]), ("disagree_multi_bld", ["bld_multi_disagree"], ["bld_multi"]),
          ("disagree_multi_reg", ["reg_multi_disagree"], ["reg_multi"]), ("multi_nounit_bld", ["bld_multi_nounit"], ["bld_multi"]),
          ("pooled_of_multi_bld", ["bld_pooled"], ["bld_multi"]), ("pooled_of_multi_nounit_bld", ["bld_pooled"], ["bld_multi_nounit"]),
          ("pooled_reg", ["reg_pooled"], ["reg"]), ("nounit_pool_apmulti_reg", ["reg_nounit_pool_apmulti"], ["reg"]),
          ("mixed_bld", ["bld_single_split_mixed", "bld_multi_partial"], ["bld"]),
          ("mixed_reg", ["reg_single_split_mixed", "reg_multi_partial"], ["reg"]),
          ("ap_fewer_bld", ["bld_single_split_units"], ["bld"]), ("apv_bld", ["bld_multi_no_unit", "bld_multi_excess"], ["bld"])]


def _shares(r: dict) -> dict:
    out = {}
    for name, num, den in SHARES:
        n, dd = sum(r[c] for c in num), sum(r[c] for c in den)
        out[name] = n / dd if dd else None  # integer over integer: the correctly rounded double
    return out


def statewide(counts: pl.DataFrame, splits: dict) -> pl.DataFrame:
    """unit_key_statewide.csv: per snapshot and split (all, development, confirmation, held_out), building and registrant totals
    and the shares above. splits maps county_id to its split name."""
    cnt = [c for c in counts.columns if c not in ("snapshot_dt", "county_id")]
    x = counts.with_columns(pl.col("county_id").replace_strict(splits, return_dtype=pl.String).alias("split"))
    g = pl.concat([x.group_by("snapshot_dt", "split").agg(pl.col(cnt).sum()),
                   x.group_by("snapshot_dt").agg(pl.col(cnt).sum()).with_columns(pl.lit("all").alias("split"))], how="diagonal")
    rows = [{"snapshot_dt": r["snapshot_dt"], "split": r["split"], "bld_all": r["bld_all"], "reg_all": r["reg_all"], **_shares(r)}
            for r in g.sort("snapshot_dt", "split").iter_rows(named=True)]
    sch = {"snapshot_dt": pl.Date, "split": pl.String, "bld_all": I, "reg_all": I, **{n: pl.Float64 for n, _, _ in SHARES}}
    return pl.DataFrame(rows, schema=sch)
