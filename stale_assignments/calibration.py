# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Mobility calibration of the reference: the register's one-year move and exit rates by age against the ACS.

The reference counts a client as a stayer when the client's registered address at s is the address recorded at t. A move
the register never records (the person moved and did not re-register) keeps the person a stayer. This module measures
how far the register's one-year mobility falls short of the American Community Survey's.

Register side, statewide, for each pair of consecutive January 1 snapshots (d0, d1): persons with an active row and an
address at d0, aged 18 or over at d0, by ACS age band. moved = present at d1 (active or inactive row) with no address at d1
equal to any of their addresses at d0 (reference household keys tk, so a county readdressing is not a move); exited = no active or
inactive row at d1. Recorded ages above 200 are left out. Register moves cluster in election years (registrants update
their address when they vote), so single years swing; compare multi-year means. ACS side (inputs/acs_nc_mobility_by_age.tsv, table B07001, ACS 1-year, North Carolina): persons
aged 18 or over who lived in a different house one year earlier, by the same bands, for the survey year of d1's previous
year (the ACS year that covers most of the interval; 2020 has no standard 1-year release and uses the mean of 2019 and
2021).

mobility_calibration.csv: per (year, band) and per year for all adults: n, moved, exited, the register rates, the ACS rate
and the gap. Age-standardized rows (band "std2016") weight each band's rates by the 2016 client file's age mix at the
build (the clients of the run, aged at t = 2016-11-08), so the comparison is for the clients' ages, not the state's.
"""
import polars as pl

I = pl.Int64
BANDS = [(18, 19, "18-19"), (20, 24, "20-24"), (25, 29, "25-29"), (30, 34, "30-34"), (35, 39, "35-39"), (40, 44, "40-44"),
         (45, 49, "45-49"), (50, 54, "50-54"), (55, 59, "55-59"), (60, 64, "60-64"), (65, 69, "65-69"), (70, 74, "70-74"),
         (75, 200, "75+")]


def band_expr(age: pl.Expr) -> pl.Expr:
    e = pl.lit(None, dtype=pl.String)
    for lo, hi, name in reversed(BANDS):
        e = pl.when((age >= lo) & (age <= hi)).then(pl.lit(name)).otherwise(e)
    return e


def register_pair(st, d0, d1) -> pl.DataFrame:
    """One pair of snapshots: (band, n, moved, exited) for adults active at d0."""
    a0 = st.read("addr", d0, columns=["ncid", "is_ai", "status_cd", "age"])
    act = (a0.filter(pl.col("is_ai").fill_null(False) & (pl.col("status_cd") == "A") & pl.col("age").is_not_null())
           .group_by("ncid").agg(pl.col("age").min()).filter(pl.col("age") >= 18))
    g0 = st.read("graph", d0, columns=["ncid", "tk"]).filter(pl.col("tk").is_not_null()).unique()
    base = act.join(g0.select("ncid").unique(), on="ncid", how="semi")
    a1 = st.read("addr", d1, columns=["ncid", "is_ai"]).filter(pl.col("is_ai").fill_null(False)).select("ncid").unique()
    g1 = st.read("graph", d1, columns=["ncid", "tk"]).filter(pl.col("tk").is_not_null()).unique()
    same = g0.join(g1, on=["ncid", "tk"], how="inner").select("ncid").unique().with_columns(pl.lit(True).alias("same"))
    x = (base.join(a1.with_columns(pl.lit(True).alias("present")), on="ncid", how="left")
         .join(same, on="ncid", how="left")
         .with_columns(pl.col("present").fill_null(False), pl.col("same").fill_null(False), band_expr(pl.col("age")).alias("band"))
         .filter(pl.col("band").is_not_null()))  # ages above 200 are recording errors
    return (x.group_by("band").agg(pl.len().cast(I).alias("n"),
                                   (pl.col("present") & ~pl.col("same")).cast(I).sum().alias("moved"),
                                   (~pl.col("present")).cast(I).sum().alias("exited"))
            .sort("band"))


def acs_by_age(path: str) -> pl.DataFrame:
    a = pl.read_csv(path, separator="\t", schema_overrides={"year": I, "population": I, "same_house_1yr_ago": I})
    yrs = set(a["year"].to_list())
    if 2020 not in yrs and {2019, 2021} <= yrs:  # no standard 2020 1-year release: mean of the neighbours
        m = (a.filter(pl.col("year").is_in([2019, 2021])).group_by("age_band")
             .agg(pl.col("population").sum(), pl.col("same_house_1yr_ago").sum()).with_columns(pl.lit(2020).cast(I).alias("year")))
        a = pl.concat([a, m.select(a.columns)])
    return a.rename({"age_band": "band"})


def mobility_calibration(st, pairs, acs: pl.DataFrame, client_ages: pl.DataFrame) -> pl.DataFrame:
    """pairs: [(d0, d1)] consecutive January 1 snapshots. client_ages: frame of age (clients at the 2016 build)."""
    parts = []
    for d0, d1 in pairs:
        r = register_pair(st, d0, d1).with_columns(pl.lit(d0.year).cast(I).alias("year"))
        parts.append(r)
    reg = pl.concat(parts)
    w = (client_ages.filter(pl.col("age") >= 18).with_columns(band_expr(pl.col("age")).alias("band"))
         .group_by("band").agg(pl.len().cast(I).alias("n_clients")))
    acs = acs.with_columns((pl.col("population") - pl.col("same_house_1yr_ago")).alias("acs_movers"))
    j = reg.join(acs.select("year", "band", "population", "acs_movers"), on=["year", "band"], how="left").join(w, on="band", how="left")
    rows = []
    for (yr,), g in sorted(j.group_by("year"), key=lambda x: x[0]):
        g = g.sort("band")
        have_acs = g["population"].null_count() == 0
        for r in g.iter_rows(named=True):
            rows.append(_row(yr, r["band"], r["n"], r["moved"], r["exited"],
                             r["acs_movers"] / r["population"] if r["population"] else None))
        n, mv, ex = g["n"].sum(), g["moved"].sum(), g["exited"].sum()
        acs_all = g["acs_movers"].sum() / g["population"].sum() if have_acs else None
        rows.append(_row(yr, "18+", n, mv, ex, acs_all))
        W = g["n_clients"].fill_null(0).sum()
        f = lambda col: (g[col] * g["n_clients"].fill_null(0)).sum() / W
        rm = sum(c * m / nn for c, m, nn in zip(g["n_clients"].fill_null(0), g["moved"], g["n"]) if nn) / W
        re = sum(c * e / nn for c, e, nn in zip(g["n_clients"].fill_null(0), g["exited"], g["n"]) if nn) / W
        ra = (sum(c * a / p for c, a, p in zip(g["n_clients"].fill_null(0), g["acs_movers"], g["population"])) / W
              if have_acs else None)
        rows.append({"year": yr, "band": "std2016", "n": None, "moved": None, "exited": None, "reg_moved": rm,
                     "reg_exited": re, "reg_moved_or_exited": rm + re, "acs_moved": ra,
                     "gap": (ra - rm - re) if ra is not None else None})
    sch = {"year": I, "band": pl.String, "n": I, "moved": I, "exited": I, "reg_moved": pl.Float64, "reg_exited": pl.Float64,
           "reg_moved_or_exited": pl.Float64, "acs_moved": pl.Float64, "gap": pl.Float64}
    return pl.DataFrame(rows, schema=sch)


def _row(yr, band, n, mv, ex, acs):
    rm, re = mv / n, ex / n
    return {"year": yr, "band": band, "n": n, "moved": mv, "exited": ex, "reg_moved": rm, "reg_exited": re,
            "reg_moved_or_exited": rm + re, "acs_moved": acs, "gap": (acs - rm - re) if acs is not None else None}
