# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Ohio and Washington voter lists (paper section 9): reading, normalization, household frames and scored cells.

Inputs (inputs/other_states.tsv): the statewide lists of Ohio (2017-07-18, 2026-10-23) and Washington (2017-06-30,
2018-06-30, 2026-10-01). Reading follows the CSV rules the results were first computed under. Ohio is comma separated
and double-quoted, with no newlines inside fields. Washington is pipe or tab separated with no quoting. The first line of
each file is a header. A line that does not parse, or has more fields than the layout, is skipped and counted. Empty
fields are missing.

Normalization, per listed row:
- the voter ID is its digits as an integer;
- names and address parts are uppercased and trimmed;
- Ohio's one street line is parsed into house number, street and a trailing unit;
- Washington's parts are joined;
- the unit loses its designator and punctuation;
- building key = state, house number, street, ZIP5; unit key adds the unit.
Age is the snapshot year minus the birth year. minor: the full birth date is after the snapshot date less 18 years (the
registrant is under 18 on the snapshot date); missing when the file has no full birth date (Washington 2026, a scoring
date only, never a build).

Household frame: the North Carolina voter-only household rule applied unchanged (tag by unit and building counts,
trackable = one listed row, single-family or unit in a small multi-unit building, unit count 1 to 5).

Crosswalk frame: the person key is replaced by first name, surname letters, middle initial and birth year within the
state. A key held by more than one ID on any date, or missing a part, never links.

Transfer cells: clients by household-change composition cell and building type, with the observed state at s under the
raw unit-key stayer rule and under the fuzzy address rule. Aggregate outputs only.
"""
import csv
import datetime as dt
import gzip
import os

import polars as pl

from .record_functions import soundex

I = pl.Int64
WS = r"[ \t\n\f\r]"  # the whitespace class of the original expressions (ASCII only)

# column positions in each file layout (0-based) and the layout width; the remaining columns are not read
OH_COLS = dict(vid=0, county=1, ln=3, fn=4, mn=5, sfx=6, dob=7, status=9, line=11, unit=12, zip=15)
WA17_COLS = dict(vid=0, fn=3, mn=4, ln=5, sfx=6, birthdate=7, stnum=9, stfrac=10, sname=11, stype=12, utype=13, predir=14,
                 postdir=15, unit=16, zip=19, county=20, status=36)
WA26_COLS = dict(vid=0, fn=1, mn=2, ln=3, sfx=4, birthyear=5, stnum=7, stfrac=8, sname=9, stype=10, utype=11, predir=12,
                 postdir=13, unit=14, zip=17, county=18, status=32)
LAYOUTS = {  # (state, date) -> (delimiter, quoted, width, columns)
    ("OH", "2017-07-18"): (",", True, 104, OH_COLS),
    ("OH", "2026-10-23"): (",", True, 140, OH_COLS),
    ("WA", "2017-06-30"): ("|", False, 41, WA17_COLS),
    ("WA", "2018-06-30"): ("\t", False, 40, WA17_COLS),
    ("WA", "2026-10-01"): ("|", False, 36, WA26_COLS),
}
ACTIVE = {"OH": ("ACTIVE", "CONFIRMATION"), "WA": ("A", "I", "Active", "Inactive")}
PAIRS = [("OH", "2017-07-18", "2026-10-23"), ("WA", "2017-06-30", "2026-10-01"), ("WA", "2018-06-30", "2026-10-01"),
         ("WA", "2017-06-30", "2018-06-30")]
CHUNK = 500_000
UNIT_WORDS = "APT|APARTMENT|UNIT|STE|SUITE|SPC|SPACE|LOT|TRLR|BLDG|RM|ROOM"
UNIT_TAIL = rf" ((?:(?:{UNIT_WORDS}) ?#?|#) ?[A-Z0-9\-]+)$"
PREFIX = "APT|APARTMENT|UNIT|STE|SUITE|LOT|RM|ROOM|BLDG|TRLR|SPC|SPACE|#"
HOUSE = r"^([0-9]+(?: [0-9]/[0-9])?[A-Z]?)(?: |$)"


# ---- reading -------------------------------------------------------------------------------------------------------------

def _fields_quoted(line, width):
    try:
        f = next(csv.reader([line], strict=True))
    except (csv.Error, StopIteration):
        return None
    return f if len(f) <= width else None


def _fields_plain(line, delim, width):
    f = line.split(delim)
    return f if len(f) <= width else None


def read_file(task):
    """One input file -> a Parquet file of the needed raw columns (strings, missing for empty). Returns counts."""
    path, out, state, date = task
    delim, quoted, width, cols = LAYOUTS[(state, date)]
    names, idx = list(cols), list(cols.values())
    parts, buf = [], {k: [] for k in names}
    lines = bad = header = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        next(f, None)  # header line
        for ln in f:
            ln = ln.rstrip("\r\n")
            lines += 1
            if not ln:
                lines -= 1
                continue
            fl = _fields_quoted(ln, width) if quoted else _fields_plain(ln, delim, width)
            if fl is None:
                bad += 1
                continue
            n = len(fl)
            for k, i in zip(names, idx):
                v = fl[i] if i < n else ""
                buf[k].append(v if v != "" else None)
            if len(buf[names[0]]) >= CHUNK:
                parts.append(pl.DataFrame(buf, schema={k: pl.String for k in names}))
                buf = {k: [] for k in names}
    parts.append(pl.DataFrame(buf, schema={k: pl.String for k in names}))
    df = pl.concat(parts)
    if quoted:  # header lines repeated inside the file
        header = df.filter(pl.col("vid") == "SOS_VOTERID").height
        df = df.filter((pl.col("vid") != "SOS_VOTERID").fill_null(True))
    df.write_parquet(out)
    return dict(file=os.path.basename(path), lines=lines, bad=bad, header_rows=header, rows=df.height)


# ---- normalization -------------------------------------------------------------------------------------------------------

def _up(c):
    return pl.col(c).str.strip_chars().str.to_uppercase()


def _int(expr):
    """SAFE_CAST(string AS INT64): optional sign and digits (surrounding spaces allowed), else missing; missing beyond 64 bits."""
    t = expr.str.strip_chars()
    return pl.when(t.str.contains(r"^[+-]?[0-9]+$")).then(t.str.replace(r"^\+", "")).otherwise(None).cast(I, strict=False)


def _mdy_date(expr):
    """SAFE.PARSE_DATE('%m/%d/%Y', x): missing when x is not a valid calendar date."""
    m = expr.str.extract_groups(r"^([0-9]{1,2})/([0-9]{1,2})/([0-9]{1,4})$")
    iso = pl.concat_str([m.struct.field("3").str.zfill(4), m.struct.field("1").str.zfill(2), m.struct.field("2").str.zfill(2)],
                        separator="-")
    d = iso.str.to_date("%Y-%m-%d", strict=False)
    return pl.when(d.is_not_null() & (d.dt.year() >= 1) & (d.dt.strftime("%Y-%m-%d") == iso)).then(d).otherwise(None)


def _mdy_year(expr):
    """Year of SAFE.PARSE_DATE('%m/%d/%Y', x), missing when x is not a valid calendar date."""
    return _mdy_date(expr).dt.year().cast(I)


def _ymd_date(expr):
    """SAFE.PARSE_DATE('%Y-%m-%d', x) on the first 10 characters: missing when not a valid calendar date."""
    t = expr.str.strip_chars().str.slice(0, 10)
    d = t.str.to_date("%Y-%m-%d", strict=False)
    return pl.when(d.is_not_null() & (d.dt.strftime("%Y-%m-%d") == t)).then(d).otherwise(None)


def normalize(raw: pl.DataFrame, state: str, date: str) -> pl.DataFrame:
    """Raw columns of one list -> normalized rows: vid, is_ai, age, county, hn, street, unit, unit_prefix, zip5, bldg_key,
    unit_key and the crosswalk name parts (fn, surname letters, middle initial, birth year)."""
    year = int(date[:4])
    snap = dt.date.fromisoformat(date)
    cut = snap.replace(year=snap.year - 18, day=28) if (snap.month, snap.day) == (2, 29) else snap.replace(year=snap.year - 18)
    x = raw
    if state == "OH":
        by = _int(pl.col("dob").str.slice(0, 4))
        x = x.with_columns(by.alias("birth_year"), _ymd_date(pl.col("dob")).alias("bdate"), pl.lit(None, pl.String).alias("hn0"),
                           pl.lit("").alias("street_c"), pl.lit(None, pl.String).alias("utype"))
    else:
        by = _int(pl.col("birthyear")) if "birthyear" in x.columns else _mdy_year(pl.col("birthdate"))
        bd = pl.lit(None, pl.Date) if "birthyear" in x.columns else _mdy_date(pl.col("birthdate"))
        x = x.with_columns(bd.alias("bdate"))
        parts = [_up("predir"), _up("sname").str.replace_all(WS + "+", " "), _up("stype"), _up("postdir")]
        street_c = pl.concat_list([p.fill_null("") for p in parts]).list.eval(pl.element().filter(pl.element() != "")).list.join(" ")
        x = x.with_columns(by.alias("birth_year"), (pl.col("stnum").fill_null("") + pl.col("stfrac").fill_null("")).alias("hn0"),
                           street_c.alias("street_c"), pl.lit(None, pl.String).alias("line"))
    digits = pl.col("vid").str.replace_all(r"[^0-9]", "")
    x = x.with_columns(
        _int(digits).cast(pl.String).alias("vid_n"),
        pl.col("status").is_in(list(ACTIVE[state])).fill_null(False).alias("is_ai"),
        (pl.lit(year) - pl.col("birth_year")).alias("age"),
        (pl.col("bdate") > pl.lit(cut)).alias("minor"),
        pl.col("ln").str.strip_chars().str.to_uppercase().alias("lnu"),
        pl.col("fn").fill_null("").str.replace_all(r"[^A-Za-z]", "").str.to_uppercase().alias("fnu"),
        pl.col("mn").fill_null("").str.replace_all(r"[^A-Za-z]", "").str.slice(0, 1).str.to_uppercase().alias("mi"),
        pl.col("line").fill_null("").str.to_uppercase().str.replace_all(r"[\.,]", " ").str.strip_chars()
          .str.replace_all(WS + "+", " ").alias("line"),
        pl.col("hn0").fill_null("").str.strip_chars().str.to_uppercase().str.replace(r"^0+([0-9])", "${1}").alias("hn0"),
        pl.col("unit").fill_null("").str.strip_chars().str.to_uppercase().str.replace_all(WS + "+", " ").alias("unit_raw0"),
        pl.when(pl.col("utype").str.strip_chars().str.to_uppercase() == "").then(None)
          .otherwise(pl.col("utype").str.strip_chars().str.to_uppercase()).alias("utype"),
        pl.col("zip").fill_null("").str.replace_all(r"[^0-9]", "").str.slice(0, 5).alias("zip5"))
    line = pl.col("line")
    one = line != ""
    tail = line.str.extract(UNIT_TAIL, 1)
    unit_raw = pl.when(pl.col("unit_raw0") != "").then(pl.col("unit_raw0")).otherwise(pl.when(one).then(tail).otherwise(None))
    x = x.with_columns(
        pl.when(one).then(line.str.extract(HOUSE, 1).fill_null("")).otherwise(pl.col("hn0")).alias("hn"),
        pl.when(one).then(line.str.replace(r"^[0-9]+(?: [0-9]/[0-9])?[A-Z]?(?: |$)", "").str.replace(UNIT_TAIL, ""))
          .otherwise(pl.col("street_c")).alias("street"),
        unit_raw.alias("unit_raw"))
    x = x.with_columns(pl.coalesce(pl.col("utype"), pl.col("unit_raw").str.extract(rf"^({PREFIX})", 1)).alias("unit_prefix"),
                       pl.col("unit_raw").fill_null("").str.replace(rf"^({PREFIX})\.?{WS}*", "")
                       .str.replace_all(r"[ \t\n\f\r#\-\.]", "").alias("unit0"))
    x = x.with_columns(pl.when(pl.col("unit0") == "").then(None).otherwise(pl.col("unit0")).alias("unit"))
    ok = (pl.col("hn") != "") & (pl.col("street") != "") & (pl.col("zip5").str.len_chars() >= 5)
    b = pl.concat_str([pl.lit(state), pl.col("hn"), pl.col("street"), pl.col("zip5")], separator="|")
    x = x.with_columns(pl.when(ok).then(b).otherwise(None).alias("bldg_key"),
                       pl.when(ok).then(pl.concat_str([b, pl.col("unit").fill_null("")], separator="|")).otherwise(None).alias("unit_key"))
    sn = pl.col("lnu").fill_null("").str.replace_all(r"[^A-Z]", "")
    return (x.filter(pl.col("vid_n").is_not_null())
            .select(pl.col("vid_n").alias("vid"), "is_ai", "age", "minor", pl.col("county"), "hn", "street", "unit", "unit_prefix", "zip5",
                    "bldg_key", "unit_key",
                    pl.when(pl.col("fnu") == "").then(None).otherwise(pl.col("fnu")).alias("first"),
                    pl.when(sn == "").then(None).otherwise(sn).alias("surname"),
                    pl.when(pl.col("mi") == "").then(None).otherwise(pl.col("mi")).alias("mi"), "birth_year"))


# ---- household frames ----------------------------------------------------------------------------------------------------

MHP = r"(^|[ \t\n\f\r])(MOBILE|MHP|TRAILER|TRLR|MH PARK|MOBILE HOME|CAMPGROUND|RV PARK|MANUFACTURED)([ \t\n\f\r]|$)"


def households(p: pl.DataFrame) -> pl.DataFrame:
    """North Carolina's voter-only household rule on normalized rows: vid, county, unit_key, bldg_key, age, minor, has_unit,
    n_ai_rows, voter_tag, bldg_n, unit_n, trackable (one row per listed row with a building key)."""
    p = p.filter(pl.col("is_ai") & pl.col("bldg_key").is_not_null())
    pc = p.group_by("vid").agg(pl.len().cast(I).alias("n_ai_rows"))
    b = p.group_by("bldg_key").agg(pl.len().cast(I).alias("b_n"), pl.col("unit").is_not_null().sum().cast(I).alias("b_nu"),
                                    pl.col("age").is_between(18, 24).fill_null(False).sum().cast(I).alias("b_y"),
                                    (pl.col("age") >= 80).fill_null(False).sum().cast(I).alias("b_o"))
    u = p.group_by("unit_key").agg(
        pl.len().cast(I).alias("u_n"), pl.col("unit").is_not_null().any().alias("u_hu"),
        (pl.col("unit_prefix").is_in(["LOT", "TRLR", "SPC", "SPACE"]).fill_null(False)
         | pl.col("street").str.contains(MHP).fill_null(False)).any().alias("u_mhp"),
        pl.col("unit_prefix").is_in(["STE", "SUITE", "OFC", "RM", "ROOM"]).fill_null(False).any().alias("u_inst"))
    h = p.join(pc, on="vid").join(b, on="bldg_key").join(u, on="unit_key")
    tag = (pl.when(pl.col("u_n") > 20).then(pl.lit("GROUP_QUARTERS_OR_OTHER"))
           .when((pl.col("b_n") >= 21) & (pl.col("b_y") / pl.col("b_n") >= 0.6)).then(pl.lit("GROUP_QUARTERS_OR_OTHER"))
           .when((pl.col("b_n") >= 6) & (pl.col("b_o") / pl.col("b_n") >= 0.5)).then(pl.lit("GROUP_QUARTERS_OR_OTHER"))
           .when(pl.col("u_inst")).then(pl.lit("GROUP_QUARTERS_OR_OTHER"))
           .when(pl.col("u_mhp")).then(pl.lit("MOBILE_HOME_PARK"))
           .when(pl.col("u_hu") & (pl.col("u_n") <= 8)).then(pl.lit("UNIT_IN_MULTIUNIT"))
           .when(pl.col("u_hu")).then(pl.lit("UNIT_LARGE_9_20"))
           .when(pl.col("b_nu") > 0).then(pl.lit("MULTIUNIT_NO_UNIT"))
           .when(pl.col("u_n") >= 9).then(pl.lit("NOUNIT_LARGE_9_20"))
           .otherwise(pl.lit("SFH_LIKELY")))
    h = h.with_columns(tag.alias("voter_tag"))
    return h.select("vid", "county", "unit_key", "bldg_key", "age", "minor", pl.col("unit").is_not_null().alias("has_unit"), "n_ai_rows",
                    "voter_tag", pl.col("b_n").alias("bldg_n"), pl.col("u_n").alias("unit_n"),
                    ((pl.col("n_ai_rows") == 1) & pl.col("voter_tag").is_in(["SFH_LIKELY", "UNIT_IN_MULTIUNIT"])
                     & pl.col("u_n").is_between(1, 5)).alias("trackable"))


def crosswalk_keys(persons: dict) -> dict:
    """{date: frame of (vid, pk)} for one state: pk = 'K' + name and birth-year key, or a date-specific key that never links
    when a part is missing or the key is held by more than one ID on any date of the state."""
    k0 = {}
    for d, p in persons.items():
        g = (p.filter(pl.col("is_ai")).group_by("vid")
             .agg(pl.col("first").min(), pl.col("surname").min(), pl.col("mi").min(), pl.col("birth_year").min()))
        xk = pl.when(pl.col("first").is_null() | pl.col("surname").is_null() | pl.col("birth_year").is_null()).then(None).otherwise(
            pl.concat_str([pl.col("first"), pl.col("surname"), pl.col("mi").fill_null(""), pl.col("birth_year").cast(pl.String)],
                          separator="|"))
        k0[d] = g.select("vid", xk.alias("xk"))
    bad = pl.concat([f.filter(pl.col("xk").is_not_null()).group_by("xk").agg(pl.col("vid").n_unique().alias("nv"))
                     .filter(pl.col("nv") > 1).select("xk") for f in k0.values()]).unique()
    out = {}
    for d, f in k0.items():
        f = f.join(bad.with_columns(pl.lit(True).alias("bad")), on="xk", how="left")
        pk = (pl.when(pl.col("xk").is_null() | pl.col("bad").fill_null(False)).then(pl.lit(f"X{d}|") + pl.col("vid"))
              .otherwise(pl.lit("K") + pl.col("xk")))
        out[d] = f.select("vid", pk.alias("pk"))
    return out


def with_crosswalk(h: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    return h.join(keys, on="vid", how="inner").with_columns(pl.col("pk").alias("vid")).drop("pk")


# ---- transfer cells ------------------------------------------------------------------------------------------------------

BANDS = ((22, "A"), (26, "B"), (30, "C"), (35, "D"), (45, "E"), (55, "F"), (65, "G"), (75, "H"))


def _band(c):
    e = pl.when(c.is_null()).then(pl.lit("Z"))
    for cut, lab in BANDS:
        e = e.when(c < cut).then(pl.lit(lab))
    return e.otherwise(pl.lit("I"))


def fuzzy_keys(p: pl.DataFrame) -> pl.DataFrame:
    """Per unit key: the smallest fuzzy address key (house-number digits | ZIP5 | Soundex of street | unit letters and digits)."""
    q = p.filter(pl.col("is_ai") & pl.col("unit_key").is_not_null())
    streets = q.select(pl.col("street").fill_null("").str.to_uppercase().alias("s")).unique()
    sx = streets.with_columns(pl.col("s").map_elements(soundex, return_dtype=pl.String).alias("sx"))
    q = q.with_columns(pl.col("street").fill_null("").str.to_uppercase().alias("s")).join(sx, on="s", how="left")
    fz = pl.concat_str([pl.col("hn").fill_null("").str.to_uppercase().str.replace_all(r"[^0-9]", ""), pl.col("zip5").fill_null(""),
                        pl.col("sx").fill_null(""), pl.col("unit").fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "")],
                       separator="|")
    return q.group_by("unit_key").agg(fz.min().alias("fz"))


def transfer_cells(state, t, s, ht, hs, fz_t, fz_s, exclude) -> pl.DataFrame:
    """Clients at t by (county, composition cell, building type hbt, table features): n_t, n listed at s and, among the
    listed, exact or not by stayer rule (raw unit key: EB EO XB XO; fuzzy address too: EB2 XB2). exclude (vid): registrants
    who are not clients (under 18 at t); they stay co-members."""
    one_t = ht.filter(pl.col("n_ai_rows") == 1).select("vid", "unit_key", "age")
    one_s = hs.filter(pl.col("n_ai_rows") == 1).select("vid", "unit_key")
    cl = (ht.filter(pl.col("trackable")).join(exclude, on="vid", how="anti")
          .select("vid", "county", pl.col("unit_key").alias("ut"), "age", "has_unit", "bldg_n", "unit_n",
                  pl.when(pl.col("voter_tag") == "SFH_LIKELY").then(pl.lit("S")).otherwise(pl.lit("M")).alias("hbt")))
    at_s = cl.join(one_s.rename({"unit_key": "us"}), on="vid", how="left")
    co = (cl.select("vid", "ut").join(one_t.rename({"vid": "jv", "unit_key": "ut", "age": "jage"}), on="ut", how="inner")
          .filter(pl.col("jv") != pl.col("vid"))
          .join(one_s.rename({"vid": "jv", "unit_key": "jus"}), on="jv", how="left"))
    g = (co.join(at_s.select("vid", "us"), on="vid", how="left").with_columns(_band(pl.col("jage")).alias("jb"))
         .group_by("vid").agg(pl.len().cast(I).alias("nco"), (pl.col("jus") == pl.col("us")).fill_null(False).sum().cast(I).alias("inter"),
                              pl.col("jage").is_between(18, 25).fill_null(False).sum().cast(I).alias("ny"),
                              pl.col("jb").sort().str.join("").alias("co_bands"),
                              pl.col("jage").mean().alias("co_mean_age"), pl.col("jage").max().alias("max_age"),
                              pl.col("jage").min().alias("min_age")))
    sz = one_s.group_by("unit_key").agg(pl.len().cast(I).alias("ssn")).rename({"unit_key": "us"})
    any_t = ht.select(pl.col("unit_key").alias("us")).unique().with_columns(pl.lit(True).alias("anyt"))
    ad = (at_s.filter(pl.col("us").is_not_null()).select("vid", "us")
          .join(one_t.select(pl.col("vid").alias("ov"), pl.col("unit_key").alias("us")), on="us", how="inner")
          .filter(pl.col("ov") != pl.col("vid"))
          .join(one_s.rename({"vid": "ov", "unit_key": "ous"}), on="ov", how="left")
          .group_by("vid").agg(pl.len().cast(I).alias("na"), (pl.col("ous") == pl.col("us")).fill_null(False).sum().cast(I).alias("inter_a")))
    r = (at_s.join(g, on="vid", how="left").join(sz, on="us", how="left").join(any_t, on="us", how="left").join(ad, on="vid", how="left")
         .join(fz_t.rename({"unit_key": "ut", "fz": "f0"}), on="ut", how="left")
         .join(fz_s.rename({"unit_key": "us", "fz": "f1"}), on="us", how="left")
         .with_columns(pl.col("nco").fill_null(0), pl.col("inter").fill_null(0), pl.col("ny").fill_null(0),
                       pl.col("co_bands").fill_null("")))
    pres = pl.col("us").is_not_null()
    nts = pl.col("ssn").fill_null(1) - 1
    nco, inter = pl.col("nco"), pl.col("inter")
    e = (pres & (((nco == 0) & (nts == 0)) | ((nco > 0) & (inter == nco) & (inter == pl.col("ssn") - 1)))).fill_null(False)
    stay = (pres & (pl.col("us") == pl.col("ut"))).fill_null(False)
    stay2 = (pres & ((pl.col("us") == pl.col("ut")).fill_null(False) | (pl.col("f0") == pl.col("f1")).fill_null(False))).fill_null(False)
    age, mean = pl.col("age"), pl.col("co_mean_age")
    d1, d2 = (age - pl.col("max_age")).abs(), (age - pl.col("min_age")).abs()
    ab = pl.when(age.is_null()).then(pl.lit("na"))
    for c, lab in ((25, "a18"), (35, "a25"), (45, "a35"), (55, "a45"), (65, "a55"), (75, "a65")):
        ab = ab.when(age < c).then(pl.lit(lab))
    ab = ab.otherwise(pl.lit("a75"))
    mb = pl.when(nco == 0).then(pl.lit("none"))
    for c, lab in ((25, "m18"), (35, "m25"), (45, "m35"), (55, "m45"), (65, "m55")):
        mb = mb.when(mean < c).then(pl.lit(lab))
    mb = mb.otherwise(pl.lit("m65"))
    x = r.select(
        "county", (_band(age) + pl.lit(":") + pl.col("co_bands")).alias("cell"), "hbt",
        pl.min_horizontal(nco + 1, pl.lit(6)).cast(I).alias("size6"),
        pl.when(pl.col("has_unit") | (pl.col("bldg_n") > pl.col("unit_n"))).then(pl.lit("multi")).otherwise(pl.lit("single")).alias("bt"),
        ab.alias("ab"),
        pl.when(nco == 0).then(pl.lit("n0")).when(pl.col("ny") == 0).then(pl.lit("y0")).when(pl.col("ny") == 1).then(pl.lit("y1"))
        .otherwise(pl.lit("y2")).alias("ny"),
        mb.alias("mb"),
        pl.when(nco == 0).then(pl.lit("none")).when((d1 < 10) & (d2 < 10)).then(pl.lit("g0"))
        .when(pl.max_horizontal(d1, d2) < 25).then(pl.lit("g10")).otherwise(pl.lit("g25")).alias("gb"),
        pres.alias("p"), e.alias("e"), stay.alias("st"), stay2.alias("st2"))
    keys = ["county", "cell", "hbt", "size6", "bt", "ab", "ny", "mb", "gb"]
    p, e, st, st2 = pl.col("p"), pl.col("e"), pl.col("st"), pl.col("st2")
    return (x.group_by(keys).agg(pl.len().cast(I).alias("n_t"), p.sum().cast(I).alias("n"),
                                 (p & e & st).sum().cast(I).alias("nEB"), (p & e & ~st).sum().cast(I).alias("nEO"),
                                 (p & ~e & st).sum().cast(I).alias("nXB"), (p & ~e & ~st).sum().cast(I).alias("nXO"),
                                 (p & e & st2).sum().cast(I).alias("nEB2"), (p & ~e & st2).sum().cast(I).alias("nXB2"))
            .with_columns(pl.lit(state).alias("state"), pl.lit(t).alias("t"), pl.lit(s).alias("s"))
            .select(["state", "t", "s"] + keys + ["n_t", "n", "nEB", "nEO", "nXB", "nXO", "nEB2", "nXB2"]).sort(["state", "t", "s"] + keys))


def retention(tc: pl.DataFrame) -> pl.DataFrame:
    return (tc.group_by("state", "t", "s", "cell", pl.col("hbt").alias("bt")).agg(pl.col("n_t").sum().cast(I), pl.col("n").sum().cast(I).alias("n_listed"))
            .sort("state", "t", "s", "cell", "bt"))


def transfer_outcomes(tc: pl.DataFrame) -> pl.DataFrame:
    c = ["n", "nEB", "nEO", "nXB", "nXO", "nEB2", "nXB2"]
    return (tc.filter(pl.col("n") > 0).group_by("state", "t", "s", "county", "cell", pl.col("hbt").alias("bt"))
            .agg([pl.col(k).sum().cast(I) for k in c]).sort("state", "t", "s", "county", "cell", "bt"))


def transfer_table_cells(tc: pl.DataFrame) -> pl.DataFrame:
    k = ["state", "t", "s", "county", "size6", "bt", "ab", "ny", "mb", "gb"]
    return (tc.filter(pl.col("n") > 0).group_by(k).agg(pl.col("n").sum().cast(I), (pl.col("nEB") + pl.col("nEO")).sum().cast(I).alias("ne"))
            .sort(k))
