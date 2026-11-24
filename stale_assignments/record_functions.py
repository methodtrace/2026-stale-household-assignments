# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Record functions with fixed, documented semantics, the same as BigQuery's FARM_FINGERPRINT, SOUNDEX and DATE_DIFF, so
any result can be cross-checked in SQL.

fingerprint: FarmHash Fingerprint64 of the UTF-8 bytes, as a signed 64-bit integer.
soundex: the first ASCII letter is kept as written; every other character that is not an ASCII letter is skipped;
A E I O U Y H W are dropped before adjacent equal digits are collapsed, so they do not separate repeated codes; the
result is padded with zeros to four characters. No letter gives the empty string.
Date differences count calendar boundaries.
"""
import datetime as _dt

import polars as pl

try:  # the cityhash distribution's farmhash module (binary wheels for Linux, macOS and Windows)
    from farmhash import Fingerprint64 as _fp64
except ImportError:  # the pyfarmhash distribution, same algorithm
    from farmhash import fingerprint64 as _fp64

_U64 = 1 << 64
_I63 = 1 << 63


def fingerprint(s: str) -> int:
    """FarmHash Fingerprint64 of the UTF-8 bytes of s, as a signed 64-bit integer."""
    v = _fp64(s)
    return v - _U64 if v >= _I63 else v


_SDX = {}
for _letters, _d in (("BFPV", "1"), ("CGJKQSXZ", "2"), ("DT", "3"), ("L", "4"), ("MN", "5"), ("R", "6")):
    for _c in _letters:
        _SDX[_c] = _d
        _SDX[_c.lower()] = _d


def _is_ascii_letter(c: str) -> bool:
    return ("A" <= c <= "Z") or ("a" <= c <= "z")


def soundex(s: str) -> str:
    out = ""
    last = None
    for c in s:
        if not _is_ascii_letter(c):
            continue
        if not out:
            out = c
            last = _SDX.get(c)
            continue
        d = _SDX.get(c)
        if d is None or d == last:
            continue
        out += d
        last = d
        if len(out) == 4:
            break
    if not out:
        return ""
    return (out + "000")[:4]


def map_unique(expr_series: pl.Series, fn, dtype) -> pl.Series:
    """Apply a Python function to the distinct non-null values of a series and broadcast the result (nulls stay null)."""
    u = expr_series.drop_nulls().unique()
    m = pl.DataFrame({"k": u, "v": pl.Series([fn(x) for x in u.to_list()], dtype=dtype)})
    return (pl.DataFrame({"k": expr_series}).join(m, on="k", how="left", maintain_order="left", nulls_equal=False)["v"])


def fp_col(series: pl.Series) -> pl.Series:
    return map_unique(series, fingerprint, pl.Int64)


def soundex_col(series: pl.Series) -> pl.Series:
    return map_unique(series, soundex, pl.String)


def years_between(a: _dt.date, b: _dt.date) -> int:
    """DATE_DIFF(a, b, YEAR)."""
    return a.year - b.year


def months_between(a: _dt.date, b: _dt.date) -> int:
    """DATE_DIFF(a, b, MONTH)."""
    return (a.year - b.year) * 12 + (a.month - b.month)
