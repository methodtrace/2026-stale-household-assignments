# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Helpers shared by the pipeline steps."""
import datetime as dt

import polars as pl

from . import config
from . import record_functions as rf

T0 = config.BUILD  # the main build (config.py; 2016-11-08 in the main design)
# salt of the 10% client sample (fingerprint input; changing it changes the sample)
from .constants import CLIENT_SAMPLE_SALT as CLIENT_SEED


def logical_or(e: pl.Expr) -> pl.Expr:
    """LOGICAL_OR: true if any input is true, null if every input is null, else false."""
    return pl.when(e.count() == 0).then(None).otherwise(e.any())


def logical_and(e: pl.Expr) -> pl.Expr:
    return pl.when(e.count() == 0).then(None).otherwise(e.all())


def nunit(c: str) -> pl.Expr:
    return pl.col(c).fill_null("").str.to_uppercase().str.replace_all(r"[^A-Z0-9]", "")


def fuzzy(df: pl.DataFrame, house="house_num", zip5="zip5", street="street_name", unit="unit") -> pl.Series:
    """Fuzzy address key: house-number digits | ZIP | Soundex of the street | normalized unit."""
    sx = rf.soundex_col(df[street].fill_null("").str.to_uppercase())
    return df.select(pl.concat_str([
        pl.col(house).fill_null("").str.to_uppercase().str.replace_all(r"[^0-9]", ""), pl.lit("|"),
        pl.col(zip5).fill_null(""), pl.lit("|"), pl.lit(sx), pl.lit("|"), nunit(unit)]).alias("fz"))["fz"]


def client_flag(ncids: pl.Series, sample: int) -> pl.Series:
    """Client file: everyone (sample 0) or the seeded 10% (FARM_FINGERPRINT(seed | ncid) mod 10 = 0)."""
    if sample == 0:
        return pl.Series([True] * len(ncids), dtype=pl.Boolean)
    assert sample == 10
    f = rf.map_unique(ncids, lambda x: abs(rf.fingerprint(f"{CLIENT_SEED}|{x}")) % 10 == 0, pl.Boolean)
    return f.fill_null(False)


def years(a: pl.Expr, b) -> pl.Expr:
    return (a.dt.year().cast(pl.Int64) - (b.year if isinstance(b, dt.date) else b.dt.year().cast(pl.Int64)))


def months(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return ((a.dt.year().cast(pl.Int64) - b.dt.year().cast(pl.Int64)) * 12
            + a.dt.month().cast(pl.Int64) - b.dt.month().cast(pl.Int64))
