# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Order-free table fingerprints: row count, XOR and integer sum of the FarmHash fingerprint of each row's text.

A row's text joins its columns with '|', each written as the warehouse writes CAST(x AS STRING) (integers in decimal,
dates as YYYY-MM-DD, booleans as true or false) and nulls as \\N. Two tables with the same three numbers over the same
columns hold the same multiset of rows (used to compare runs, not to produce results).
"""
import polars as pl

from .record_functions import fingerprint


def row_text(df: pl.DataFrame, cols) -> pl.Series:
    ex = []
    for c in cols:
        dt = df.schema[c]
        e = pl.col(c)
        if dt == pl.Boolean:
            e = pl.when(e).then(pl.lit("true")).when(~e).then(pl.lit("false")).otherwise(None)
        elif dt == pl.Date:
            e = e.dt.strftime("%Y-%m-%d")
        elif dt == pl.String:
            pass
        elif dt.is_integer():
            e = e.cast(pl.String)
        else:
            raise TypeError(f"column {c} of type {dt} is not fingerprinted (floats are compared numerically)")
        ex.append(e.fill_null(r"\N"))
    return df.select(pl.concat_str(ex, separator="|").alias("t"))["t"]


def table_fp(df: pl.DataFrame, cols, by=None, chunk: int = 1_000_000) -> pl.DataFrame:
    parts = []
    for off in range(0, max(df.height, 1), chunk):
        sl = df.slice(off, chunk)
        t = row_text(sl, cols)
        fps = pl.Series("fp", [fingerprint(x) for x in t.to_list()], dtype=pl.Int64)
        parts.append(pl.DataFrame({"fp": fps}) if by is None else sl.select(by).with_columns(fps))
    d = pl.concat(parts)
    agg = [pl.len().alias("n"), pl.col("fp").bitwise_xor().alias("x"), pl.col("fp").cast(pl.Int128).sum().alias("sm")]
    return d.select(agg) if by is None else d.group_by(by).agg(agg).sort(by)

