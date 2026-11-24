# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Stage 1: one North Carolina voter registration snapshot file to the typed address table.

Input: the snapshot ZIP as published by the North Carolina State Board of Elections (one tab-separated UTF-16 text file
with a header line and 90 columns, see INPUT_DATA.md). Columns are read by position. A line with up to two extra fields is
kept with them in extra_1 and extra_2 and later dropped as misaligned; a line with more is rejected as a bad record. A field
that is exactly empty is null. A field written as "..." is unquoted; any other quote character is literal.

Output: one row per snapshot row with a parseable snapshot date and no extra field. Names are kept only as FarmHash
fingerprints (every name is uppercased first); address parts are uppercased and normalized.
"""
import io
import os
import zipfile

import polars as pl

from . import record_functions as rf

RAW_COLUMNS = (
    "snapshot_dt county_id county_desc voter_reg_num ncid status_cd voter_status_desc reason_cd voter_status_reason_desc "
    "absent_ind name_prefx_cd last_name first_name midl_name name_sufx_cd house_num half_code street_dir street_name "
    "street_type_cd street_sufx_cd unit_designator unit_num res_city_desc state_cd zip_code mail_addr1 mail_addr2 mail_addr3 "
    "mail_addr4 mail_city mail_state mail_zipcode area_cd phone_num race_code race_desc ethnic_code ethnic_desc party_cd "
    "party_desc sex_code sex age birth_place registr_dt precinct_abbrv precinct_desc municipality_abbrv municipality_desc "
    "ward_abbrv ward_desc cong_dist_abbrv cong_dist_desc super_court_abbrv super_court_desc judic_dist_abbrv judic_dist_desc "
    "NC_senate_abbrv NC_senate_desc NC_house_abbrv NC_house_desc county_commiss_abbrv county_commiss_desc township_abbrv "
    "township_desc school_dist_abbrv school_dist_desc fire_dist_abbrv fire_dist_desc water_dist_abbrv water_dist_desc "
    "sewer_dist_abbrv sewer_dist_desc sanit_dist_abbrv sanit_dist_desc rescue_dist_abbrv rescue_dist_desc munic_dist_abbrv "
    "munic_dist_desc dist_1_abbrv dist_1_desc dist_2_abbrv dist_2_desc confidential_ind cancellation_dt vtd_abbrv vtd_desc "
    "load_dt age_group extra_1 extra_2").split()
assert len(RAW_COLUMNS) == 92
USED = ["snapshot_dt", "county_id", "ncid", "status_cd", "reason_cd", "last_name", "first_name", "midl_name", "name_sufx_cd",
        "house_num", "half_code", "street_dir", "street_name", "street_type_cd", "street_sufx_cd", "unit_num", "zip_code",
        "age", "registr_dt", "extra_1"]
_USED_IDX = [RAW_COLUMNS.index(c) for c in USED]

ADDR_SCHEMA = {
    "snapshot_dt": pl.Date, "series": pl.String, "ncid": pl.String, "county_id": pl.Int64, "status_cd": pl.String,
    "reason_cd": pl.String, "is_ai": pl.Boolean, "age": pl.Int64, "registr_dt": pl.Date, "house_num": pl.String,
    "half_code": pl.String, "street_dir": pl.String, "street_name": pl.String, "street_type_cd": pl.String,
    "street_sufx_cd": pl.String, "unit": pl.String, "unit_prefix": pl.String, "zip5": pl.String, "bldg_key": pl.Int64,
    "unit_key": pl.Int64, "surname_fp": pl.Int64, "surname_norm_fp": pl.Int64, "surname_sdx_fp": pl.Int64,
    "first_fp": pl.Int64, "first3_fp": pl.Int64, "first_sdx_fp": pl.Int64, "mid_init_fp": pl.Int64, "name_sufx": pl.String}


class BadInput(Exception):
    pass


def _split_quoted(ln: str):
    """Split a line that contains a quote character as the warehouse CSV reader does (quote '"', no quoted newlines).

    A field that starts with a quote runs to the next lone quote (two quotes stand for one) and may hold tabs; the closing
    quote may be followed by spaces (fixed-width padding, dropped), then must be followed by a tab or the end of the line.
    Any other quote is literal. Returns None for a bad record."""
    out = []
    i, n = 0, len(ln)
    while True:
        if i < n and ln[i] == '"':
            j = i + 1
            buf = []
            while True:
                k = ln.find('"', j)
                if k < 0:
                    return None
                buf.append(ln[j:k])
                if k + 1 < n and ln[k + 1] == '"':
                    buf.append('"')
                    j = k + 2
                    continue
                break
            out.append("".join(buf))
            i = k + 1
            while i < n and ln[i] == " ":
                i += 1
            if i == n:
                return out
            if ln[i] != "\t":
                return None
            i += 1
        else:
            k = ln.find("\t", i)
            if k < 0:
                out.append(ln[i:])
                return out
            out.append(ln[i:k])
            i = k + 1


PART_LINES = 1_000_000


# The warehouse loaded these five Election Day files in a first batch without quote processing (every tab separates fields
# and a double quote is an ordinary character) and every other file with standard quoting. The used columns differ only on
# lines where a field opens with a quote and text follows its closing quote: one line, in the 2024-11-05 file.
UNQUOTED = {"20161108", "20181106", "20201103", "20221108", "20241105"}


def decode(zip_path: str, tsv_prefix: str, quoting: bool = True) -> tuple:
    """Decode the snapshot to UTF-8 tab-separated parts of PART_LINES lines with exactly the used columns. With
    quoting=False every tab separates fields and quotes are ordinary characters.

    Returns (line counts, list of part paths)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise BadInput(f"expected one member in {zip_path}, found {len(names)}")
        rep = {"lines": 0, "kept": 0, "bad_too_many_fields": 0, "bad_quote": 0, "extra_fields": 0, "quote_lines": 0}
        parts = []
        r = io.TextIOWrapper(zf.open(names[0]), encoding="utf-16", newline="")
        header = True
        buf = ""
        out = []

        def flush():
            path = f"{tsv_prefix}.{len(parts):04d}.tsv"
            with open(path, "w", encoding="utf-8", newline="\n") as w:
                w.write("\n".join(out))
                w.write("\n")
            parts.append(path)
            out.clear()

        for chunk in iter(lambda: r.read(1 << 22), ""):
            buf += chunk
            lines = buf.split("\n")
            buf = lines.pop()
            for ln in lines:
                if header:
                    header = False
                    continue
                if ln.endswith("\r"):
                    ln = ln[:-1]
                if "\r" in ln:
                    raise BadInput("carriage return inside a line")
                rep["lines"] += 1
                if '"' in ln:
                    rep["quote_lines"] += 1
                if '"' in ln and quoting:
                    f = _split_quoted(ln)
                    if f is None:
                        rep["bad_quote"] += 1
                        continue
                    if any("\t" in x for x in f):
                        # a tab inside a quoted field; every used column strips control characters or keeps digits only
                        rep["tab_in_quotes"] = rep.get("tab_in_quotes", 0) + 1
                        f = [x.replace("\t", "\x00") for x in f]
                else:
                    f = ln.split("\t")
                nt = len(f) - 1
                if nt > 91:
                    rep["bad_too_many_fields"] += 1
                    continue
                if nt > 89:
                    rep["extra_fields"] += 1
                if nt < 91:
                    f += [""] * (91 - nt)
                out.append("\t".join([f[i] for i in _USED_IDX]))
                rep["kept"] += 1
                if len(out) >= PART_LINES:
                    flush()
        if buf.strip("\r") != "":
            raise BadInput("file does not end with a line break")
        if out:
            flush()
    return rep, parts


CTRL = r'[\x00-\x1f"]'
WS = r"[\t\n\f\r ]"  # whitespace class of the specification's regular expressions
UNIT_PREFIX = r"^(APT|APARTMENT|UNIT|STE|SUITE|LOT|RM|ROOM|BLDG|TRLR|SPC|#)"


def _clean(c):
    return pl.col(c).str.replace_all(CTRL, "")


def addr_table(tsv_path: str) -> pl.DataFrame:
    raw = pl.read_csv(tsv_path, separator="\t", has_header=False, quote_char=None, new_columns=USED,
                      schema={c: pl.String for c in USED}, empty_string_is_null=True, encoding="utf8")
    s = raw.select(
        pl.col("snapshot_dt").str.strip_chars().str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False).alias("snapshot_dt"),
        _clean("ncid").str.strip_chars().alias("ncid"),
        pl.col("county_id").str.replace_all(r"[^0-9]", "").str.strip_chars().cast(pl.Int64, strict=False).alias("county_id"),
        _clean("status_cd").str.strip_chars().str.to_uppercase().alias("status_cd"),
        _clean("reason_cd").str.strip_chars().str.to_uppercase().alias("reason_cd"),
        pl.col("age").str.replace_all(r"[^0-9]", "").str.strip_chars().cast(pl.Int64, strict=False).alias("age"),
        pl.coalesce(pl.col("registr_dt").str.strip_chars().str.to_date("%m/%d/%Y", strict=False),
                    pl.col("registr_dt").str.strip_chars().str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)).alias("registr_dt"),
        _clean("last_name").str.strip_chars().str.to_uppercase().alias("ln"),
        _clean("first_name").str.strip_chars().str.replace_all(r"[^A-Za-z]", "").str.to_uppercase().alias("fn"),
        _clean("midl_name").str.strip_chars().str.replace_all(r"[^A-Za-z]", "").str.slice(0, 1).str.to_uppercase().alias("mi"),
        _clean("name_sufx_cd").str.strip_chars().str.to_uppercase().alias("name_sufx"),
        *[_clean(c).str.strip_chars().str.to_uppercase().alias(c) for c in
          ("house_num", "half_code", "street_dir", "street_type_cd", "street_sufx_cd")],
        _clean("street_name").str.strip_chars().str.replace_all(WS + "+", " ").str.to_uppercase().alias("street_name"),
        _clean("unit_num").str.strip_chars().str.to_uppercase().alias("unit_raw"),
        pl.col("zip_code").str.replace_all(r"[^0-9]", "").alias("zip_digits"),
        pl.col("extra_1").str.strip_chars().alias("extra_1"),
    )
    s = s.filter(pl.col("snapshot_dt").is_not_null() & ~(pl.col("extra_1").is_not_null() & (pl.col("extra_1") != "")))
    s = s.with_columns(
        pl.when(pl.col("name_sufx") == "").then(None).otherwise(pl.col("name_sufx")).alias("name_sufx"),
        pl.col("unit_raw").str.replace(UNIT_PREFIX + r"\.?" + WS + "*", "").str.replace_all(r"[\t\n\f\r #\-\.]", "").alias("unit"),
        pl.col("unit_raw").str.extract(UNIT_PREFIX, 1).alias("unit_prefix"),
        pl.col("zip_digits").str.slice(0, 5).alias("zip5"),
        pl.col("ln").str.replace_all(r"[^A-Z]", "").alias("ln_norm"),
    ).with_columns(pl.when(pl.col("unit") == "").then(None).otherwise(pl.col("unit")).alias("unit"))
    d = s["snapshot_dt"]
    mo, dy, yr = d.dt.month(), d.dt.day(), d.dt.year()
    series = (pl.DataFrame({"m": mo, "d": dy, "y": yr}).select(
        pl.when((pl.col("m") == 1) & (pl.col("d") == 1)).then(pl.lit("jan1"))
        .when((pl.col("m") == 11) & (pl.col("y") % 2 == 0)).then(pl.lit("eday"))
        .when(pl.col("m") == 11).then(pl.lit("nov_odd")).otherwise(pl.lit("other")))[:, 0])
    nonempty = lambda c: pl.when(pl.col(c).is_null() | (pl.col(c) == "")).then(None).otherwise(pl.col(c))
    keyok = (pl.col("house_num") != "") & (pl.col("street_name") != "") & pl.col("zip5").is_not_null() & (pl.col("zip5").str.len_chars() >= 5)
    parts = [pl.col(c) for c in ("house_num", "half_code", "street_dir", "street_name", "street_type_cd", "street_sufx_cd", "zip5")]
    k = s.select(
        nonempty("ln").alias("ln_k"), nonempty("ln_norm").alias("lnn_k"), nonempty("fn").alias("fn_k"),
        nonempty("fn").str.slice(0, 3).alias("f3_k"), nonempty("mi").alias("mi_k"),
        pl.when(keyok).then(pl.concat_str(parts, separator="|", ignore_nulls=False)).otherwise(None).alias("bldg_s"),
        pl.when(keyok).then(pl.concat_str(parts + [pl.col("unit").fill_null("")], separator="|", ignore_nulls=False)).otherwise(None).alias("unit_s"))
    ln_sdx = rf.soundex_col(k["ln_k"])
    fn_sdx = rf.soundex_col(k["fn_k"])
    out = s.select(
        "snapshot_dt", "ncid", "county_id", "status_cd", "reason_cd", "age", "registr_dt", "house_num", "half_code",
        "street_dir", "street_name", "street_type_cd", "street_sufx_cd", "unit", "unit_prefix", "zip5", "name_sufx",
    ).with_columns(
        series.alias("series"),
        pl.col("status_cd").is_in(["A", "I"]).alias("is_ai"),
        rf.fp_col(k["bldg_s"]).alias("bldg_key"),
        rf.fp_col(k["unit_s"]).alias("unit_key"),
        rf.fp_col(k["ln_k"]).alias("surname_fp"),
        rf.fp_col(k["lnn_k"]).alias("surname_norm_fp"),
        rf.fp_col(ln_sdx).alias("surname_sdx_fp"),
        rf.fp_col(k["fn_k"]).alias("first_fp"),
        rf.fp_col(k["f3_k"]).alias("first3_fp"),
        rf.fp_col(fn_sdx).alias("first_sdx_fp"),
        rf.fp_col(k["mi_k"]).alias("mid_init_fp"),
    )
    # is_in on a null status gives null in the specification (status_cd IN (...)); keep that
    out = out.with_columns(pl.when(pl.col("status_cd").is_null()).then(None).otherwise(pl.col("is_ai")).alias("is_ai"))
    return out.select(list(ADDR_SCHEMA)).cast(ADDR_SCHEMA)


def build_snapshot(zip_path: str, out_parquet: str, tmp_dir: str) -> dict:
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_parquet)), exist_ok=True)
    prefix = os.path.join(tmp_dir, os.path.basename(zip_path))
    stamp = os.path.basename(zip_path)[len("VR_Snapshot_"):][:8]
    parts = []
    try:
        rep, parts = decode(zip_path, prefix, quoting=stamp not in UNQUOTED)
        frames = []
        for p in parts:
            frames.append(addr_table(p))
            os.remove(p)
        df = pl.concat(frames, how="vertical")
    finally:
        for p in parts:
            if os.path.exists(p):
                os.remove(p)
    df.write_parquet(out_parquet, compression="zstd", compression_level=9, statistics=False)
    rep["rows"] = df.height
    return rep
