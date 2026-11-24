# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Working-directory layout. Every table is a Parquet file; per-snapshot tables are one file per snapshot date."""
import datetime as dt
import os

import polars as pl

SNAPSHOTS = [dt.date.fromisoformat(s) for s in (
    "2008-11-04 2009-01-01 2009-11-03 2010-01-01 2010-11-02 2011-01-01 2012-01-01 2012-11-06 2013-01-01 2013-11-05 "
    "2014-01-01 2014-11-04 2015-01-01 2015-11-03 2016-01-01 2016-11-08 2017-01-01 2017-11-07 2018-01-01 2018-11-06 "
    "2019-01-01 2019-04-30 2019-05-14 2019-07-09 2019-09-10 2019-10-08 2019-11-05 2020-01-01 2020-11-03 2021-01-01 "
    "2021-11-02 2022-01-01 2022-11-08 2023-01-01 2023-11-07 2024-01-01 2024-11-05 2025-01-01 2025-11-04 2026-01-01").split()]


LATER_FILE_DATE = dt.date(2026, 11, 3)  # the later file (later_file runs only)


def series_of(d: dt.date) -> str:
    if d.month == 1 and d.day == 1:
        return "jan1"
    if d.month == 11 and d.year % 2 == 0:
        return "eday"
    if d.month == 11:
        return "nov_odd"
    return "other"


from . import config  # noqa: E402

# Snapshots a stage 3 run sees: the 40 registered files, plus the 2026-11-03 file in later_file runs.
RUN_SNAPSHOTS = SNAPSHOTS + ([LATER_FILE_DATE] if config.FILES == "later_file" else [])
EDAYS = [d for d in RUN_SNAPSHOTS if series_of(d) == "eday"]
JAN1 = [d for d in RUN_SNAPSHOTS if series_of(d) == "jan1"]
# Address tables read with every column after stage 2: the Election Days from 2016 on (re-matching, events, the probabilistic matcher).
FULL_ADDR = {d for d in SNAPSHOTS + [LATER_FILE_DATE] if series_of(d) == "eday" and d >= dt.date(2016, 11, 8)}
# Columns no later step reads from the other address tables; stage 2 drops them once a date's tags exist.
# Columns stage 3 reads from the other address tables; stage 2 keeps only these once its own tables are built.
ADDR_STAGE3 = ["snapshot_dt", "ncid", "county_id", "status_cd", "reason_cd", "is_ai", "age", "registr_dt", "bldg_key", "unit_key"]
SLIM_DROP = ["series", "surname_fp", "surname_sdx_fp", "first3_fp", "first_sdx_fp", "mid_init_fp", "name_sufx"]


class Store:
    def __init__(self, root: str):
        self.root = root

    # tables later_file runs replace: the relabel components and age flags rebuilt with the 2026-11-03 file
    LATER_FILE_TABLES = {"canon": "canon_later_file", "age_flags": "age_flags_later_file"}

    def path(self, table: str, d: dt.date | None = None) -> str:
        if config.FILES == "later_file":
            table = self.LATER_FILE_TABLES.get(table, table)
        if d is None:
            return os.path.join(self.root, f"{table}.parquet")
        return os.path.join(self.root, table, f"{d.isoformat()}.parquet")

    def exists(self, table, d=None):
        return os.path.exists(self.path(table, d))

    def read(self, table, d=None, columns=None) -> pl.DataFrame:
        return pl.read_parquet(self.path(table, d), columns=columns)

    def write(self, df: pl.DataFrame, table, d=None):
        p = self.path(table, d)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.write_parquet(p + ".part", compression="zstd", statistics=False)
        os.replace(p + ".part", p)
