# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The refresh-decision test (paper section 8.4): which stored households to rebuild first under a budget.

rows():      U_T per decision date td (2024-11-05: stored builds 2018, 2020, 2022, fresh row k = 2; 2022-11-08: stored
             builds 2018, 2020, fresh row k = 4), from the policy never_rebuilt, lag 0, s = 2026-11-03 rows of the
             2018-build later-file run: the person's cohort (one of its stored builds, chosen by hash), county, features of the cohort
             row's stored unit at the cohort build, e_stored and e_fresh (persons with truth), the complete-case flag and
             the two tie-break hashes.
predict():   p_stored, p_fresh and the predicted gain g = p_fresh - p_stored for the age x size, coarse table and
             co-member age table parity tables (table_scores predict rules: absent cell takes the table's pooled share at its k, clamped).
aggregate(): seven orderings, person and record cost, full and complete-case files, county sets primary80 and all100,
             budgets 0, 10, 25, 50 and 100 percent: per-county sums (n_u, s_e, n_ref, rec_ref) for the bootstrap.
Floating sums use math.fsum; record keys are rounded to 12 decimals before ordering.
"""
import collections
import csv
import datetime as dt
import math
import os

import polars as pl

from .. import store as S
from ..constants import REFRESH_COHORT_SALT, REFRESH_SAMPLE_SALT
from ..assignment_tables import _band, _features, exact_e
from ..record_functions import fingerprint
from ..table_scores import read_table

I = pl.Int64
S_DT = dt.date(2026, 11, 3)
SRC_PAIRS = [(dt.date(2018, 11, 6), 8), (dt.date(2020, 11, 3), 6), (dt.date(2022, 11, 8), 4), (dt.date(2024, 11, 5), 2)]
DEC = [(dt.date(2024, 11, 5), 2, 3), (dt.date(2022, 11, 8), 4, 2)]
POLICIES = ["random", "oldest", "age_x_size", "coarse_table", "comember_age_table", "lowest_p_comember_age_table", "oracle"]
BPS = [0, 1000, 2500, 5000, 10000]


def rows(st: S.Store, rm0: pl.LazyFrame) -> pl.DataFrame:
    """One row per (td, ncid) of U_T."""
    pf = pl.lit(False)
    for t, k in SRC_PAIRS:
        pf = pf | ((pl.col("t_dt") == t) & (pl.col("k") == k))
    src = (rm0.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & (pl.col("s_dt") == S_DT) & pf)
           .select("t_dt", "k", "ncid", "county0", "last_known_key", "has_truth", "f_size", exact_e().alias("e")).collect())
    assert src.select("t_dt", "ncid").is_duplicated().sum() == 0, "more than one source row per (t_dt, ncid)"
    ht = src.group_by("ncid").agg(pl.col("has_truth").n_unique().alias("u"), pl.col("has_truth").null_count().alias("z"))
    assert ht.filter((pl.col("u") > 1) | (pl.col("z") > 0)).height == 0, "has_truth differs across a person's rows"
    st_rows = src.filter(pl.col("t_dt") < dt.date(2024, 11, 5))
    assert st_rows.filter(pl.col("last_known_key").is_null() | pl.col("ncid").is_null() | pl.col("county0").is_null()).height == 0
    out = []
    for td, kf, n_stored in DEC:
        fresh = src.filter(pl.col("t_dt") == td).select("ncid", "has_truth", pl.col("e").alias("e_fresh"))
        lst = (src.filter(pl.col("t_dt") < td).join(fresh.select("ncid"), on="ncid", how="semi")
               .sort(["ncid", "t_dt"]).group_by("ncid", maintain_order=True).agg(pl.col("t_dt").alias("listed")))
        c = fresh.join(lst, on="ncid", how="inner").with_columns(pl.col("listed").list.len().cast(I).alias("n_listed"))
        h = [fingerprint(f"{x}|{REFRESH_COHORT_SALT}") for x in c["ncid"].to_list()]
        n = c["n_listed"].to_list()
        j = [((hh % nn) + nn) % nn for hh, nn in zip(h, n)]  # Python % is already non-negative; kept as the SQL form
        c = c.with_columns(pl.Series("j", j, dtype=I)).with_columns(pl.col("listed").list.get(pl.col("j")).alias("cohort_t"))
        r = (c.join(src.rename({"t_dt": "cohort_t", "e": "e_cohort"}), on=["ncid", "cohort_t"], how="inner", suffix="_src")
             .with_columns(pl.lit(td).alias("td"), pl.lit(kf, dtype=I).alias("kf"), (pl.col("n_listed") == n_stored).alias("complete_case"),
                           pl.col("last_known_key").str.slice(1).cast(I).alias("ukey")))
        out.append(r.select("td", "ncid", "has_truth", "n_listed", "complete_case", "cohort_t", "k", "kf", "county0", "last_known_key",
                            "f_size", "ukey", "e_cohort", "e_fresh"))
    r = pl.concat(out)
    parts = []
    for t in sorted(r["cohort_t"].unique().to_list()):
        h, cma = _features(st, t)
        parts.append(r.filter(pl.col("cohort_t") == t)
                     .join(h.rename({"unit_key": "ukey"}).with_columns(pl.lit(True).alias("_h")), on=["ncid", "ukey"], how="left")
                     .join(cma.rename({"unit_key": "ukey", "client": "ncid"}).with_columns(pl.lit(True).alias("_c")),
                           on=["ukey", "ncid"], how="left"))
        del h, cma
    x = pl.concat(parts)
    hn, cn = pl.col("_h").is_null(), pl.col("_c").is_null()
    age, mean = pl.col("age"), pl.col("mean_age")
    d1, d2 = (age - pl.col("max_age")).abs(), (age - pl.col("min_age")).abs()
    x = x.select(
        "td", "ncid", "has_truth", "n_listed", "complete_case", "cohort_t", "k", "kf", "county0", "last_known_key", "f_size",
        pl.min_horizontal(pl.col("f_size"), pl.lit(6)).alias("size6"),
        pl.when(hn).then(pl.lit("na")).when(pl.col("has_unit") | (pl.col("bldg_n") > pl.col("unit_n"))).then(pl.lit("multi"))
        .otherwise(pl.lit("single")).alias("bt"),
        _band(age, [25, 35, 45, 55, 65, 75], ["a18", "a25", "a35", "a45", "a55", "a65", "a75"], "na").alias("ab"),
        pl.when(cn).then(pl.lit("n0")).when(pl.col("ny") == 0).then(pl.lit("y0")).when(pl.col("ny") == 1).then(pl.lit("y1"))
        .otherwise(pl.lit("y2")).alias("ny"),
        pl.when(cn).then(pl.lit("none")).when(mean < 25).then(pl.lit("m18")).when(mean < 35).then(pl.lit("m25"))
        .when(mean < 45).then(pl.lit("m35")).when(mean < 55).then(pl.lit("m45")).when(mean < 65).then(pl.lit("m55"))
        .otherwise(pl.lit("m65")).alias("mb"),
        pl.when(cn).then(pl.lit("none")).when((d1 < 10) & (d2 < 10)).then(pl.lit("g0"))
        .when(pl.max_horizontal(d1, d2) < 25).then(pl.lit("g10")).otherwise(pl.lit("g25")).alias("gb"),
        pl.when(pl.col("has_truth")).then(pl.col("e_cohort")).alias("e_stored"),
        pl.when(pl.col("has_truth")).then(pl.col("e_fresh")).alias("e_fresh"))
    hp = [fingerprint(f"{n}|{REFRESH_SAMPLE_SALT}") for n in x["ncid"].to_list()]
    hr = [fingerprint(f"{t.isoformat()}|{fk}|{c}|{REFRESH_SAMPLE_SALT}")
          for t, fk, c in zip(x["cohort_t"].to_list(), x["last_known_key"].to_list(), x["county0"].to_list())]
    x = x.with_columns(pl.Series("h_person", hp, dtype=I), pl.Series("h_record", hr, dtype=I))
    assert x.select("td", "ncid").is_duplicated().sum() == 0
    return x.sort(["td", "ncid"])


def _age_size(dev_cells_path):
    g = collections.defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(dev_cells_path, encoding="utf-8")):
        a = g[(r["k"], r["size6"])]; a[0] += int(r["n"]); a[1] += int(r["ne"])
    return {k: v[1] / v[0] for k, v in g.items()}


def predict(rw: pl.DataFrame, tables_dir, dev_cells_path) -> pl.DataFrame:
    """(td, ncid, p_stored_*, p_fresh_*, g_*) for as, coarse_table, comember_age_table."""
    clamp = lambda p: min(max(p, 1e-6), 1 - 1e-6)
    par = lambda y: "pres" if y % 4 == 0 else "mid"
    _, t7, pk7 = read_table(os.path.join(tables_dir, "coarse_table_parity.csv"))
    _, t78, pk78 = read_table(os.path.join(tables_dir, "comember_age_table_parity.csv"))
    AS = _age_size(dev_cells_path)
    cols = {c: rw[c].to_list() for c in ("k", "kf", "cohort_t", "td", "size6", "bt", "ab", "ny", "mb", "gb")}
    out = collections.defaultdict(list)
    for i in range(rw.height):
        s6, bt, ab, ny, mb, gb = (str(cols[c][i]) for c in ("size6", "bt", "ab", "ny", "mb", "gb"))
        for side, k, y in (("stored", cols["k"][i], cols["cohort_t"][i].year), ("fresh", cols["kf"][i], cols["td"][i].year)):
            ks, p = str(k), par(y)
            out[f"p_{side}_as"].append(clamp(AS.get((ks, s6), pk7[k])))
            out[f"p_{side}_coarse_table"].append(clamp(t7.get((ks, p, s6, bt, ab), pk7[k])))
            out[f"p_{side}_comember_age_table"].append(clamp(t78.get((ks, p, s6, bt, ab, ny, mb, gb), pk78[k])))
    d = rw.select("td", "ncid").with_columns(*[pl.Series(c, v, dtype=pl.Float64) for c, v in out.items()])
    return d.with_columns(*[(pl.col(f"p_fresh_{m}") - pl.col(f"p_stored_{m}")).alias(f"g_{m}") for m in ("as", "coarse_table", "comember_age_table")])


def _rank(df, keys, desc):
    """1-based row number in the order of keys (desc flags; nulls last), as ROW_NUMBER() OVER (ORDER BY ...)."""
    return df.sort(keys, descending=desc, nulls_last=True).with_row_index("rk", offset=1)


def aggregate(rw: pl.DataFrame, pr: pl.DataFrame, split: dict) -> pl.DataFrame:
    """refresh_agg (cost, td, file, cset, policy, bp, county0, n_u, s_e, n_ref, rec_ref, n_persons,
    n_records, n_units, cutoff), dense over the counties of each set."""
    sp = pl.DataFrame({"county0": list(split), "split": list(split.values())}, schema={"county0": I, "split": pl.String})
    base = rw.join(pr, on=["td", "ncid"], how="inner").join(sp, on="county0", how="inner")
    out = []
    for td in sorted(base["td"].unique().to_list()):
        for file in ("full", "complete_case"):
            for cset in ("primary80", "all100"):
                pop = base.filter(pl.col("td") == td)
                if file == "complete_case":
                    pop = pop.filter(pl.col("complete_case"))
                if cset == "primary80":
                    pop = pop.filter(pl.col("split") != "development")
                counties = sorted(c for c, s in split.items() if cset == "all100" or s != "development")
                out += _stratum(pop, td, file, cset, counties)
    sch = {"cost": pl.String, "td": pl.Date, "file": pl.String, "cset": pl.String, "policy": pl.String, "bp": I, "county0": I,
           "n_u": I, "s_e": I, "n_ref": I, "rec_ref": I, "n_persons": I, "n_records": I, "n_units": I, "cutoff": I}
    return pl.DataFrame(out, schema=sch, orient="row").sort(["cost", "td", "file", "cset", "policy", "bp", "county0"])


def _stratum(pop, td, file, cset, counties):
    pop = pop.with_columns(pl.when(pl.col("has_truth")).then(pl.col("e_fresh") - pl.col("e_stored")).alias("gain"))
    n_p = pop.height
    rec_keys = ["cohort_t", "last_known_key", "county0"]
    n_r = pop.select(rec_keys).unique().height
    rows = []
    # person cost
    pords = {"random": (["h_person", "ncid"], [False, False]), "oldest": (["cohort_t", "h_person", "ncid"], [False] * 3),
             "age_x_size": (["g_as", "h_person", "ncid"], [True, False, False]), "coarse_table": (["g_coarse_table", "h_person", "ncid"], [True, False, False]),
             "comember_age_table": (["g_comember_age_table", "h_person", "ncid"], [True, False, False]),
             "lowest_p_comember_age_table": (["p_stored_comember_age_table", "h_person", "ncid"], [False] * 3),
             "oracle": (["gain", "h_person", "ncid"], [True, False, False])}
    for pol in POLICIES:
        keys, desc = pords[pol]
        x = _rank(pop, keys, desc)
        for bp in BPS:
            c = n_p * bp // 10000
            ref = x.with_columns((pl.col("rk") <= c).alias("ref"))
            g = ref.group_by("county0").agg(
                pl.col("has_truth").sum().cast(I).alias("n_u"),
                (pl.when(pl.col("has_truth")).then(pl.when(pl.col("ref")).then(pl.col("e_fresh")).otherwise(pl.col("e_stored")))
                 .otherwise(0)).sum().cast(I).alias("s_e"),
                pl.col("ref").sum().cast(I).alias("n_ref"))
            rr = ref.group_by(rec_keys).agg(pl.col("ref").any()).group_by("county0").agg(pl.col("ref").sum().cast(I).alias("rec_ref"))
            rows += _dense(g.join(rr, on="county0", how="left"), "person", td, file, cset, pol, bp, counties, n_p, n_r, n_p, c)
    # record cost
    rec = pop.group_by(rec_keys).agg(
        pl.col("h_record").first(), pl.len().cast(I).alias("n_p"), pl.col("has_truth").sum().cast(I).alias("n_u"),
        pl.when(pl.col("has_truth")).then(pl.col("e_stored")).otherwise(0).sum().cast(I).alias("s_st"),
        pl.col("gain").fill_null(0).sum().cast(I).alias("s_gain"),
        *[pl.col(f"g_{m}").alias(f"_g_{m}") for m in ("as", "coarse_table", "comember_age_table")],
        pl.col("p_stored_comember_age_table").alias("_p"))
    rnd = lambda v: round(math.fsum(v), 12)
    rec = rec.with_columns(
        *[pl.Series(f"sg_{m}", [rnd(v) for v in rec[f"_g_{m}"].to_list()], dtype=pl.Float64)
          for m in ("as", "coarse_table", "comember_age_table")],
        pl.Series("mp_comember_age_table", [round(math.fsum(v) / len(v), 12) for v in rec["_p"].to_list()], dtype=pl.Float64),
        pl.when(pl.col("n_u") > 0).then(pl.col("s_gain")).alias("og"))
    tie = ["h_record", "cohort_t", "last_known_key", "county0"]
    rords = {"random": (tie, [False] * 4), "oldest": (["cohort_t", "h_record", "last_known_key", "county0"], [False] * 4),
             "age_x_size": (["sg_as"] + tie, [True] + [False] * 4),
             "coarse_table": (["sg_coarse_table"] + tie, [True] + [False] * 4),
             "comember_age_table": (["sg_comember_age_table"] + tie, [True] + [False] * 4),
             "lowest_p_comember_age_table": (["mp_comember_age_table"] + tie, [False] * 5),
             "oracle": (["og"] + tie, [True] + [False] * 4)}
    for pol in POLICIES:
        keys, desc = rords[pol]
        x = _rank(rec, keys, desc)
        for bp in BPS:
            c = n_r * bp // 10000
            ref = x.with_columns((pl.col("rk") <= c).alias("ref"))
            g = ref.group_by("county0").agg(
                pl.col("n_u").sum().cast(I).alias("n_u"),
                (pl.col("s_st") + pl.when(pl.col("ref")).then(pl.col("s_gain")).otherwise(0)).sum().cast(I).alias("s_e"),
                pl.when(pl.col("ref")).then(pl.col("n_p")).otherwise(0).sum().cast(I).alias("n_ref"),
                pl.col("ref").sum().cast(I).alias("rec_ref"))
            rows += _dense(g, "record", td, file, cset, pol, bp, counties, n_p, n_r, n_r, c)
    return rows


def _dense(g, cost, td, file, cset, pol, bp, counties, n_p, n_r, n_units, c):
    d = {r[0]: r[1:] for r in g.select("county0", "n_u", "s_e", "n_ref", "rec_ref").iter_rows()}
    out = []
    for cty in counties:
        n_u, s_e, n_ref, rec_ref = d.get(cty, (0, 0, 0, 0))
        out.append((cost, td, file, cset, pol, bp, cty, n_u, s_e, n_ref, rec_ref or 0, n_p, n_r, n_units, c))
    return out


def write_agg(agg: pl.DataFrame, path):
    """refresh_agg packed one row per stratum, counties as 'county|n_u|s_e|n_ref|rec_ref;...'."""
    keys = ["cost", "td", "file", "cset", "policy", "bp"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(keys + ["n_persons", "n_records", "n_units", "cutoff", "n_counties", "cells"])
        for k, g in sorted(agg.group_by(keys), key=lambda x: tuple(str(v) for v in x[0])):
            g = g.sort("county0"); r0 = g.row(0, named=True)
            cells = ";".join(f"{c}|{a}|{b}|{d}|{e}" for c, a, b, d, e in g.select("county0", "n_u", "s_e", "n_ref", "rec_ref").iter_rows())
            w.writerow([k[0], k[1].isoformat(), k[2], k[3], k[4], k[5], r0["n_persons"], r0["n_records"], r0["n_units"], r0["cutoff"],
                        g.height, cells])


def curve_cells(rm0: pl.LazyFrame) -> pl.DataFrame:
    """Per (t_dt, s_dt, county0) counts n, n_exact, n_stay, n_exact_stay of the score-pair run's policy never_rebuilt,
    lag 0 rows of persons with truth."""
    sy = (pl.col("mkey") == pl.col("last_known_key")).fill_null(False)
    return (rm0.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
            .with_columns(exact_e().alias("e"), sy.alias("stay"))
            .group_by("t_dt", "s_dt", "county0")
            .agg(pl.len().cast(I).alias("n"), pl.col("e").sum().cast(I).alias("n_exact"), pl.col("stay").sum().cast(I).alias("n_stay"),
                 pl.when(pl.col("stay")).then(pl.col("e")).otherwise(0).sum().cast(I).alias("n_exact_stay"))
            .with_columns((pl.col("s_dt") - pl.col("t_dt")).dt.total_days().cast(I).alias("days"))
            .select("t_dt", "s_dt", "days", "county0", "n", "n_exact", "n_stay", "n_exact_stay")
            .sort(["t_dt", "s_dt", "county0"]).collect())


def _corr(x, y):
    """Pearson correlation over the pairs where both are non-null (fsum)."""
    p = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    n = len(p); mx = math.fsum(a for a, _ in p) / n; my = math.fsum(b for _, b in p) / n
    sxy = math.fsum((a - mx) * (b - my) for a, b in p)
    sxx = math.fsum((a - mx) ** 2 for a, _ in p); syy = math.fsum((b - my) ** 2 for _, b in p)
    return sxy / math.sqrt(sxx * syy)


def _ntile(n, q=20):
    """NTILE(q) bucket (1-based) of each of n ordered rows: the first n mod q buckets hold one more row."""
    base, extra = divmod(n, q)
    out = []
    for b in range(q):
        out += [b + 1] * (base + (1 if b < extra else 0))
    return out


def diag_posthoc(rw: pl.DataFrame, pr: pl.DataFrame, split: dict, path):
    """Post hoc (no decision): primary stratum correlations of predicted with realized gain and of
    p with e, and the top two ventiles of predicted gain. Writes refresh_diag_posthoc.csv."""
    dev = [c for c, s in split.items() if s == "development"]
    b = (rw.filter((pl.col("td") == dt.date(2024, 11, 5)) & ~pl.col("county0").is_in(dev)).join(pr, on=["td", "ncid"], how="inner")
         .with_columns(pl.when(pl.col("has_truth")).then(pl.col("e_fresh") - pl.col("e_stored")).alias("gain"),
                       pl.when(pl.col("has_truth")).then(pl.col("e_stored")).alias("es"),
                       pl.when(pl.col("has_truth")).then(pl.col("e_fresh")).alias("ef"),
                       pl.col("cohort_t").dt.year().alias("cy")))
    L = lambda c: b[c].to_list()
    gain = L("gain")
    corr_g = {m: _corr(L(f"g_{m}"), gain) for m in ("as", "coarse_table", "comember_age_table")}
    corr_ps = {m: _corr(L(f"p_stored_{m}"), L("es")) for m in ("as", "comember_age_table")}
    corr_pf = {m: _corr(L(f"p_fresh_{m}"), L("ef")) for m in ("as", "comember_age_table")}
    ven = {}
    for m in ("as", "comember_age_table"):
        d = b.sort([f"g_{m}", "h_person"], descending=[True, False]).with_columns(pl.Series("q", _ntile(b.height)))
        v = {}
        for q in (1, 2):
            x = d.filter(pl.col("q") == q)
            g = [y for y in x["gain"].to_list() if y is not None]
            v[q] = dict(g=math.fsum(x[f"g_{m}"].to_list()) / x.height, real=math.fsum(g) / len(g),
                        sh18=x.filter(pl.col("cy") == 2018).height / x.height)
        ven[m] = v
    f4 = lambda v: f"{v:.4f}"
    rows = [("corr_g_with_realized_gain", f4(corr_g["as"]), f4(corr_g["coarse_table"]), f4(corr_g["comember_age_table"])),
            ("corr_p_stored_with_e_stored", f4(corr_ps["as"]), "", f4(corr_ps["comember_age_table"])),
            ("corr_p_fresh_with_e_fresh", f4(corr_pf["as"]), "", f4(corr_pf["comember_age_table"])),
            ("top_ventile_mean_g", f4(ven["as"][1]["g"]), "", f4(ven["comember_age_table"][1]["g"])),
            ("top_ventile_realized_gain", f4(ven["as"][1]["real"]), "", f4(ven["comember_age_table"][1]["real"])),
            ("top_ventile_share_2018_cohort", f4(ven["as"][1]["sh18"]), "", f4(ven["comember_age_table"][1]["sh18"])),
            ("second_ventile_realized_gain", f4(ven["as"][2]["real"]), "", f4(ven["comember_age_table"][2]["real"]))]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["quantity", "age_x_size", "coarse_table_parity", "comember_age_table_parity"])
        w.writerows(rows)
