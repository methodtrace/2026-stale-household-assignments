# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Other registers (paper section 9) and the build-error experiment, one definition for every state.

Household frame (one per state and snapshot): vid, county (string), unit_key, bldg_key, age, has_unit, n_ai_rows, voter_tag,
bldg_n, unit_n, trackable. For North Carolina it is the voter household tags with the county of the listed row; other
states build it from their normalized registrant rows by the same rule.

score_cells(): client trackable at t and not under 18 at t (minors(): North Carolina's minor flag at t, the other states'
full birth date); stored household = co-members at t on its unit key (one listed row); registrant at s =
one listed row at s; truth = co-members at s on the client's unit key; exact with the client excluded; stayer = same unit key;
address anchor = the client's unit at s held a listed registrant at t. Cells per (state, t, s, county, size6, bt, ab, ny, mb,
gb, present, stayer, am): n, ne, nea.
tables(): the North Carolina currency tables carried to the scored cells at k = 8 (by build parity; a build that is not an
even-year November takes the mean of the midterm and presidential cells), recalibrated in the large by one logit shift
found on a two-stage grid; per-county Brier sums.
evaluate(): the accounting quantities and criteria per pair, and the ranking of the predictors after recalibration, with
99% county-cluster intervals. Each pair draws its replicates from its own stream, so a pair's intervals do not depend on
which other pairs are evaluated.
build_error(): the build-error experiment (stale and noisy builds of North Carolina at 2018-11-06).
Exponentials use exact_math.exp, logarithms table_scores.decimal_log, and sums math.fsum or integer arithmetic.
"""
import collections
import csv
import datetime as dt
import hashlib
import math
import os

import numpy as np
import polars as pl

from .constants import BOOTSTRAP_REPS, BUILD_ERROR_BOOTSTRAP_SEED, OTHER_STATES_BOOTSTRAP_SEED
from .exact_math import exp
from .record_functions import fingerprint
from .table_scores import county_draws, decimal_log

I = pl.Int64
PREDICTORS = ("constant", "age_by_size", "coarse_table", "comember_age_table")


def nc_households(st, d) -> pl.DataFrame:
    """The household frame for North Carolina at d: the voter household tags with the listed row's county."""
    t = st.read("tags", d, columns=["ncid", "unit_key", "bldg_key", "age", "has_unit", "n_ai_rows", "voter_tag", "bldg_n", "unit_n",
                                     "trackable_voter"])
    a = (st.read("addr", d, columns=["ncid", "unit_key", "county_id", "is_ai"]).filter(pl.col("is_ai").fill_null(False))
         .select("ncid", "unit_key", "county_id").unique())
    return (t.join(a, on=["ncid", "unit_key"], how="left")
            .select(pl.col("ncid").alias("vid"), pl.col("county_id").cast(pl.String).alias("county"), "unit_key", "bldg_key", "age",
                    "has_unit", "n_ai_rows", "voter_tag", "bldg_n", "unit_n", pl.col("trackable_voter").alias("trackable")))


def minors(ht: pl.DataFrame) -> pl.DataFrame:
    """vid of the registrants under 18 on the snapshot date t in another state's household frame (full birth date)."""
    return ht.filter(pl.col("minor").fill_null(False)).select("vid").unique()


def nc_minors(st, d) -> pl.DataFrame:
    """vid of the North Carolina registrants flagged as minors at d (age_flags), the main scorer's client rule."""
    return st.read("age_flags").filter((pl.col("snapshot_dt") == d) & pl.col("minor")).select(pl.col("ncid").alias("vid")).unique()


def score_cells(state, t, s, ht: pl.DataFrame, hs: pl.DataFrame, exclude: pl.DataFrame) -> pl.DataFrame:
    """Scored cells for one (state, t, s) from the household frames at t and at s. exclude (vid): registrants who are not
    clients (under 18 at t); they stay co-members."""
    one_t = ht.filter(pl.col("n_ai_rows") == 1).select("vid", "unit_key", "age")
    one_s = hs.filter(pl.col("n_ai_rows") == 1).select("vid", "unit_key")
    cl = (ht.filter(pl.col("trackable")).join(exclude, on="vid", how="anti")
          .select("vid", "county", pl.col("unit_key").alias("ut"), "age", "has_unit", "bldg_n", "unit_n"))
    at_s = cl.join(one_s.rename({"unit_key": "us"}), on="vid", how="left")
    co = (cl.select("vid", "ut").join(one_t.rename({"vid": "jv", "unit_key": "ut", "age": "jage"}), on="ut", how="inner")
          .filter(pl.col("jv") != pl.col("vid"))
          .join(one_s.rename({"vid": "jv", "unit_key": "jus"}), on="jv", how="left"))
    g = (co.join(at_s.select("vid", "us"), on="vid", how="left")
         .group_by("vid").agg(pl.len().cast(I).alias("nco"), (pl.col("jus") == pl.col("us")).fill_null(False).sum().cast(I).alias("inter"),
                              pl.col("jage").is_between(18, 25).fill_null(False).sum().cast(I).alias("ny"),
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
         .with_columns(pl.col("nco").fill_null(0), pl.col("inter").fill_null(0), pl.col("ny").fill_null(0)))
    pres = pl.col("us").is_not_null()
    nts = pl.col("ssn").fill_null(1) - 1
    nco, inter = pl.col("nco"), pl.col("inter")
    e = pres & (((nco == 0) & (nts == 0)) | ((nco > 0) & (inter == nco) & (inter == pl.col("ssn") - 1)))
    am = pres & pl.col("anyt").fill_null(False)
    na = pl.col("na").fill_null(0)
    ea = am & (((na == 0) & (nts == 0)) | ((na > 0) & (pl.col("inter_a") == na) & (pl.col("inter_a") == pl.col("ssn") - 1)))
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
        "county", pres.alias("present"), (pres & (pl.col("us") == pl.col("ut"))).fill_null(False).alias("stayer"),
        e.fill_null(False).alias("e"), am.alias("am"), ea.fill_null(False).alias("ea"),
        pl.min_horizontal(nco + 1, pl.lit(6)).cast(I).alias("size6"),
        pl.when(pl.col("has_unit") | (pl.col("bldg_n") > pl.col("unit_n"))).then(pl.lit("multi")).otherwise(pl.lit("single")).alias("bt"),
        ab.alias("ab"),
        pl.when(nco == 0).then(pl.lit("n0")).when(pl.col("ny") == 0).then(pl.lit("y0")).when(pl.col("ny") == 1).then(pl.lit("y1"))
        .otherwise(pl.lit("y2")).alias("ny"),
        mb.alias("mb"),
        pl.when(nco == 0).then(pl.lit("none")).when((d1 < 10) & (d2 < 10)).then(pl.lit("g0"))
        .when(pl.max_horizontal(d1, d2) < 25).then(pl.lit("g10")).otherwise(pl.lit("g25")).alias("gb"))
    keys = ["county", "size6", "bt", "ab", "ny", "mb", "gb", "present", "stayer", "am"]
    return (x.group_by(keys).agg(pl.len().cast(I).alias("n"), pl.col("e").sum().cast(I).alias("ne"), pl.col("ea").sum().cast(I).alias("nea"))
            .with_columns(pl.lit(state).alias("state"), pl.lit(t).alias("t"), pl.lit(s).alias("s"))
            .select(["state", "t", "s"] + keys + ["n", "ne", "nea"]).sort(["state", "t", "s"] + keys))


def _sums(cells, g_alone, g_with):
    p, sy, am = pl.col("present"), pl.col("present") & pl.col("stayer"), pl.col("present") & pl.col("am")
    z = lambda c, e: pl.when(c).then(e).otherwise(0).sum().cast(I)
    return (cells.with_columns(pl.when(pl.col("size6") == 1).then(pl.lit(g_alone)).otherwise(pl.lit(g_with)).alias("g")),
            [z(p, pl.col("n")).alias("n_p"), z(p, pl.col("ne")).alias("ne"), z(sy, pl.col("n")).alias("n_st"),
             z(sy, pl.col("ne")).alias("ne_st"), z(am, pl.col("n")).alias("n_am"), z(am, pl.col("nea")).alias("nea")])


def county_sums(cells: pl.DataFrame) -> pl.DataFrame:
    """Per (state, t, s, county, g) with g = a (alone) or w (with others): n_p, ne, n_st, ne_st, n_am, nea."""
    c, aggs = _sums(cells, "a", "w")
    return c.group_by("state", "t", "s", "county", "g").agg(*aggs).sort(["state", "t", "s", "county", "g"])


def nc_pair_sums(cells: pl.DataFrame, path):
    """North Carolina statewide sums per (t, s, alone or with)."""
    c, aggs = _sums(cells, "alone", "with")
    x = (c.group_by("t", "s", "g").agg(pl.col("n").sum().cast(I).alias("n_t"), *aggs)
         .with_columns((pl.col("n_p") - pl.col("n_st")).alias("n_mv"), (pl.col("ne") - pl.col("ne_st")).alias("ne_mv"))
         .sort(["t", "s", "g"]))
    cols = ["t", "s", "g", "n_t", "n_p", "ne", "n_st", "ne_st", "n_mv", "ne_mv", "n_am", "nea"]
    _write_rows(path, cols, [[str(v) for v in r] for r in x.select(cols).iter_rows()])


def listed_share(cells: pl.DataFrame, path):
    """Per pair: the share of build-date clients listed at s, and the listed-and-exact share."""
    x = (cells.group_by("state", "t", "s").agg(pl.col("n").sum().cast(I).alias("n_all"),
                                               pl.when(pl.col("present")).then(pl.col("n")).otherwise(0).sum().cast(I).alias("n_present"),
                                               pl.when(pl.col("present")).then(pl.col("ne")).otherwise(0).sum().cast(I).alias("ne_present"))
         .sort(["state", "t", "s"]))
    rows = [[st_, str(t), str(s), na, npr, nep, f"{npr / na:.6f}", f"{nep / npr:.6f}", f"{nep / na:.6f}"]
            for st_, t, s, na, npr, nep in x.iter_rows()]
    _write_rows(path, ["state", "t", "s", "n_all", "n_present", "ne_present", "present_share", "kP", "uncond_exact"], rows)


def _write_rows(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(header); w.writerows(rows)


# ---- the North Carolina tables carried to other registers ---------------------------------------------------------------

def _k8(tables_dir, dev_cells_path):
    def par_avg(name, cols):
        tab = collections.defaultdict(dict)
        for r in csv.DictReader(open(os.path.join(tables_dir, f"{name}_parity.csv"), encoding="utf-8")):
            if r["k"] == "8":
                tab[tuple(r[c] for c in cols)][r["parity"]] = float(r["p"])
        return tab
    coarse = par_avg("coarse_table", ["size6", "bt", "ab"])
    comember = par_avg("comember_age_table", ["size6", "bt", "ab", "ny", "mb", "gb"])
    pk0 = {r["k"]: float(r["pk"]) for r in csv.DictReader(open(os.path.join(tables_dir, "coarse_table_parity_pk.csv"),
                                                               encoding="utf-8"))}["8"]
    g = collections.defaultdict(lambda: [0, 0])
    for r in csv.DictReader(open(dev_cells_path, encoding="utf-8")):
        if r["k"] == "8":
            g[r["size6"]][0] += int(r["n"]); g[r["size6"]][1] += int(r["ne"])
    by_size = {k: v[1] / v[0] for k, v in g.items()}
    return coarse, comember, pk0, by_size


def _parity(t: dt.date):
    if t.month == 11 and t.year % 2 == 0:
        return "pres" if t.year % 4 == 0 else "mid"
    return "both"


def _pick(v, par, pk0):
    if not v:
        return pk0
    pm, pp = v.get("mid"), v.get("pres")
    if par == "mid":
        return pm if pm is not None else pk0
    if par == "pres":
        return pp if pp is not None else pk0
    a = pm if pm is not None else pp; b = pp if pp is not None else pm
    return (a + b) / 2


def _expit_sum(rows, g):
    return math.fsum(n / (1 + float(exp(-(lp + g)))) for n, ne, lp in rows)


def _shift(rows):
    """Two-stage grid: step 0.05 on [-3, 3], then 0.0005 within +-0.05 of the first; the shift whose expected count is
    closest to the observed count (first on a tie)."""
    tgt = math.fsum(ne for _, ne, _ in rows)
    best = lambda grid: min(grid, key=lambda g: abs(_expit_sum(rows, g) - tgt))
    d1 = best([-3.0 + i * 0.05 for i in range(121)])
    return best([d1 + (-0.05 + i * 0.0005) for i in range(201)])


def _pair_tables(task):
    """tables() rows of one (state, t, s) pair."""
    (state, t, s), R, coarse, comember, pk0, by_size = task
    clamp = lambda p: min(max(p, 1e-6), 1 - 1e-6)
    out = []
    par = _parity(t)
    m = sum(r["ne"] for r in R) / sum(r["n"] for r in R)
    cp = collections.defaultdict(list)
    for r in R:
        s6 = str(r["size6"])
        p = {"constant": m, "age_by_size": by_size.get(s6, pk0),
             "coarse_table": _pick(coarse.get((s6, r["bt"], r["ab"])), par, pk0),
             "comember_age_table": _pick(comember.get((s6, r["bt"], r["ab"], r["ny"], r["mb"], r["gb"])), par, pk0)}
        for pred, v in p.items():
            cp[pred].append((r["county"], r["n"], r["ne"], clamp(v)))
    for pred in PREDICTORS:
        rows = [(n, ne, decimal_log(p / (1 - p))) for _, n, ne, p in cp[pred]]
        d = _shift(rows)
        agg = collections.defaultdict(lambda: [0, 0, [], [], [], []])
        for (cty, n, ne, p), (_, _, lp) in zip(cp[pred], rows):
            q = 1 / (1 + float(exp(-(lp + d))))
            a = agg[cty]; a[0] += n; a[1] += ne
            a[2].append(n * p); a[3].append(n * p * p - 2 * p * ne + ne); a[4].append(n * q * q - 2 * q * ne + ne); a[5].append(n * q)
        for cty in sorted(agg):
            a = agg[cty]
            out.append(dict(state=state, t=t, s=s, county=cty, pred=pred, delta=d, n=a[0], ne=a[1], sp=math.fsum(a[2]),
                            b_raw=math.fsum(a[3]), b_cal=math.fsum(a[4]), sq=math.fsum(a[5])))
    return out


def tables(cells: pl.DataFrame, tables_dir, dev_cells_path):
    """Per (state, t, s, county, predictor): delta, n, ne, sp, b_raw, b_cal, sq. Pairs run in parallel, merged in pair order."""
    from .parallel import pmap
    coarse, comember, pk0, by_size = _k8(tables_dir, dev_cells_path)
    c = (cells.filter(pl.col("present")).group_by("state", "t", "s", "county", "size6", "bt", "ab", "ny", "mb", "gb")
         .agg(pl.col("n").sum(), pl.col("ne").sum()).sort("state", "t", "s", "county", "size6", "bt", "ab", "ny", "mb", "gb"))
    tasks = [(key, g.rows(named=True), coarse, comember, pk0, by_size)
             for key, g in c.group_by(["state", "t", "s"], maintain_order=True)]
    return [r for part in pmap(_pair_tables, tasks) for r in part]


def nc_tables_summary(tabs, path):
    """Per North Carolina (t, s, predictor): shift, n, mean outcome, mean prediction, raw and recalibrated Brier."""
    g = collections.defaultdict(list)
    for r in tabs:
        g[(r["t"], r["s"], r["pred"])].append(r)
    rows = []
    for (t, s, pred) in sorted(g, key=lambda k: (k[0], k[1], PREDICTORS.index(k[2]))):
        x = g[(t, s, pred)]; n = sum(r["n"] for r in x); ne = sum(r["ne"] for r in x)
        f = lambda c: math.fsum(r[c] for r in x)
        rows.append([str(t), str(s), pred, repr(round(x[0]["delta"], 6) + 0.0), n, repr(round(ne / n, 4)), repr(round(f("sp") / n, 4)),
                     repr(round(f("b_raw") / n, 5)), repr(round(f("b_cal") / n, 5))])
    _write_rows(path, ["t", "s", "pred", "delta", "n", "ybar", "pbar", "brier_raw", "brier_cal"], rows)


def county_briers(tabs):
    """Per (state, t, s, predictor, county): n and the raw and recalibrated Brier sums, rounded to 3 places."""
    return [dict(state=r["state"], t=str(r["t"]), s=str(r["s"]), pred=r["pred"], county=r["county"], n=r["n"],
                 b_raw=round(r["b_raw"], 3), b_cal=round(r["b_cal"], 3)) for r in tabs]


# ---- evaluation --------------------------------------------------------------------------------------------------------

# North Carolina statewide lines against the gap in years (intercept, slope), from five build pairs scored in November 2026
NC_FIT = dict(kP=(0.4112, -0.0113), sig=(0.7884, -0.0240), kS=(0.5090, -0.0119), mu=(0.1702, 0.0018))
QUANTITIES = ("kP", "sig", "kS", "mu", "PA", "kA", "kP_a", "kP_w")
# (first, second, threshold, name): D = Brier(first) - Brier(second) per client after recalibration
RANKING = (("comember_age_table", "age_by_size", -0.010, "comember_vs_age"), ("coarse_table", "age_by_size", 0.0, "coarse_vs_age"),
           ("comember_age_table", "coarse_table", 0.0, "comember_vs_coarse"), ("age_by_size", "constant", -0.015, "age_vs_constant"))


def _nc_fit(q, k):
    return NC_FIT[q][0] + NC_FIT[q][1] * k


def _stats(v):
    a, w = v[:6], v[6:]
    n_p, ne, n_st, ne_st, n_am, nea = [a[i] + w[i] for i in range(6)]
    return dict(kP=ne / n_p, sig=n_st / n_p, kS=ne_st / n_st, mu=(ne - ne_st) / (n_p - n_st), PA=n_am / n_p,
                kA=nea / n_am, kP_a=a[1] / a[0], kP_w=w[1] / w[0])


def _seed(state, t, s, part):
    """The integer seed of one pair's replicate stream (part 0: accounting, part 1: ranking)."""
    h = hashlib.sha256(f"{OTHER_STATES_BOOTSTRAP_SEED}|{state}|{t}|{s}|{part}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _replicate_sums(X, draws):
    """X: county rows in sorted county order (lists of ints or floats); draws: {county index: count} per replicate."""
    out = []
    for d in draws:
        cols = list(zip(*[[x * c for x in X[i]] for i, c in sorted(d.items())]))
        out.append([math.fsum(col) if isinstance(col[0], float) else sum(col) for col in cols])
    return out


def _interval(vals):
    lo, hi = np.percentile(np.array(vals, dtype=np.float64), [0.5, 99.5])
    return float(lo), float(hi)


def evaluate(cells_path, tabs_path=None, bias=None, reps=BOOTSTRAP_REPS):
    """Rows (state, t, s, gap, quantity, value, lo99, hi99, criterion, holds), values written in full precision (shortest
    repr) so a printed value is rounded once. cells_path: county_sums output; tabs_path:
    county_briers output for the same pairs; bias: {state: crosswalk bias subtracted from kP}."""
    bias = bias or {}
    cells = collections.defaultdict(lambda: collections.defaultdict(lambda: [0] * 12))
    for r in csv.DictReader(open(cells_path, encoding="utf-8")):
        off = 0 if r["g"] == "a" else 6
        v = cells[(r["state"], r["t"], r["s"])][r["county"]]
        for i, k in enumerate(("n_p", "ne", "n_st", "ne_st", "n_am", "nea")):
            v[off + i] += int(r[k])
    out = []
    for (state, t, s), cty in sorted(cells.items()):
        k = (dt.date.fromisoformat(s) - dt.date.fromisoformat(t)).days / 365.25
        names = sorted(cty); X = [cty[c] for c in names]
        draws = county_draws(list(range(len(names))), reps, _seed(state, t, s, 0))
        reps_v = [_stats(v) for v in _replicate_sums(X, draws)]
        pt = _stats([sum(col) for col in zip(*X)])
        b = bias.get(state, 0.0); pt["kP"] -= b
        for q in QUANTITIES:
            lo, hi = _interval([r[q] - (b if q == "kP" else 0) for r in reps_v])
            out.append([state, t, s, f"{k:.2f}", q, repr(pt[q]), repr(lo), repr(hi), "descriptive", ""])
        row = lambda q, val, crit, holds: out.append([state, t, s, f"{k:.2f}", q, repr(float(val)), "", "", crit, holds])
        if k >= 7:
            nc = _nc_fit("kP", k)
            row("level_vs_nc_fit", pt["kP"] - nc, f"|dev|<=0.05 and kP in [0.25,0.40] (NC fit {nc:.4f})",
                abs(pt["kP"] - nc) <= 0.05 and 0.25 <= pt["kP"] <= 0.40)
            star = pt["sig"] * _nc_fit("kS", k) + (1 - pt["sig"]) * _nc_fit("mu", k)
            row("level_vs_decomposition", pt["kP"] - star, f"|kP-kP*|<=0.025 (kP* {star:.4f})", abs(pt["kP"] - star) <= 0.025)
            if abs(pt["kP"] - nc) > 0.02:
                row("decomposition_explains_half", abs(pt["kP"] - star), f"< |dev|/2 = {abs(pt['kP'] - nc) / 2:.4f}",
                    abs(pt["kP"] - star) < abs(pt["kP"] - nc) / 2)
            row("orderings", 0, "kS>kP>mu and kP_a>kP_w", pt["kS"] > pt["kP"] > pt["mu"] and pt["kP_a"] > pt["kP_w"])
        if k >= 0.99:   # 'a year or more' includes the 365-day pairs
            row("mover_currency_band", pt["mu"], "in [0.15,0.25]", 0.15 <= pt["mu"] <= 0.25)
        if 0.9 <= k <= 1.2:
            row("one_year_level", pt["kP"], "in [0.70,0.88]", 0.70 <= pt["kP"] <= 0.88)
        if k < 0.2:
            row("one_month_level", pt["kP"], "in [0.95,0.995]", 0.95 <= pt["kP"] <= 0.995)
        row("anchor_agreement", abs(pt["kA"] - pt["kP"]), "|kA-kP|<=0.04", abs(pt["kA"] - pt["kP"]) <= 0.04)
    if tabs_path:
        T = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in csv.DictReader(open(tabs_path, encoding="utf-8")):
            T[(r["state"], r["t"], r["s"])][r["county"]][r["pred"]] = (int(r["n"]), float(r["b_cal"]))
        for (state, t, s), cty in sorted(T.items()):
            names = sorted(cty)
            X = [[cty[c][p][1] for p in PREDICTORS] + [float(cty[c]["constant"][0])] for c in names]
            draws = county_draws(list(range(len(names))), reps, _seed(state, t, s, 1))
            reps_v = _replicate_sums(X, draws)
            tot = [math.fsum(col) for col in zip(*X)]
            for x, y, thr, nm in RANKING:
                D = lambda v: (v[PREDICTORS.index(x)] - v[PREDICTORS.index(y)]) / v[-1]
                v = D(tot); lo, hi = _interval([D(r) for r in reps_v])
                ok = (v <= thr if thr < 0 else v < 0) and hi < 0
                out.append([state, t, s, "", f"{nm} D({x},{y})", repr(v), repr(lo), repr(hi),
                            f"D<={thr} and hi<0" if thr < 0 else "D<0 and hi<0", ok])
    return out


def write_evaluation(rows, path):
    _write_rows(path, ["state", "t", "s", "gap", "quantity", "value", "lo99", "hi99", "criterion", "holds"], rows)


# ---- build-error experiment -----------------------------------------------------------------------------------------

# (arm, lagged source, building-drop rate, standalone rate), in the order the bootstrap stream visits them
ARMS = [("stale", True, 0.0, 0.0), ("noisy", False, 0.10, 0.02), ("stale_and_noisy", True, 0.10, 0.02), ("exact", False, 0.0, 0.0)]
BUILD_SOURCES = [dt.date(2016, 11, 8), dt.date(2017, 1, 1), dt.date(2017, 11, 7), dt.date(2018, 1, 1), dt.date(2018, 11, 6)]
BUILD_DATE = dt.date(2018, 11, 6)
SCORE_DATES = [dt.date(2020, 1, 1), dt.date(2026, 11, 3)]


def _u(s, m):
    """|fingerprint(s)| mod m, divided by m."""
    return (abs(fingerprint(s)) % m) / m


def _set_fp(vids):
    return fingerprint(",".join(sorted(vids)))


def build_error(st) -> pl.DataFrame:
    """Cells (arm, s, county, alone, stayer, lag_b, built, ev, ep, n), on the main scorer's clients and keys: clients are
    rematch.universe() at BUILD_DATE over all counties (trackable, not a minor at the build); records are the graph records
    (every row, each person on every key of its rows; the client's own key is its smallest key); truth at s is the members
    of the client's reference key (smallest tk) at s; stayer = same smallest key at s as at the build. A vendor build puts
    each person's rows, from its source date, on S + ncid (standalone), B + building key (building dropped) or the unit key."""
    from . import rematch as RM, runs
    U = RM.universe(st, BUILD_DATE, runs.counties("all"), 0)
    cl = [(v, str(c)) for v, c in U.select("ncid", "county0").iter_rows()]
    del U
    pos, sets, univ = {}, {}, set()
    for d in BUILD_SOURCES:
        g = st.read("graph", d, columns=["ncid", "akey"]).unique()
        bk = (st.read("tags", d, columns=["unit_key", "bldg_key"]).filter(pl.col("bldg_key").is_not_null())
              .group_by("unit_key").agg(pl.col("bldg_key").min())
              .select(pl.concat_str([pl.lit("u"), pl.col("unit_key").cast(pl.String)]).alias("akey"), "bldg_key"))
        g = g.join(bk, on="akey", how="left").sort(["ncid", "akey"])
        p = collections.defaultdict(list)
        for v, a, b in g.iter_rows():
            p[v].append((a, b))
        pos[d] = {v: tuple(r) for v, r in p.items()}
        univ.update(pos[d])
        if d == BUILD_DATE:
            m = collections.defaultdict(list)
            for v, a, _ in g.iter_rows():
                m[a].append(v)
            sets[d] = {a: (_set_fp(vs), len(vs)) for a, vs in m.items()}
        del g, bk, p
    truth, ckey = {}, {}
    for s in SCORE_DATES:
        g = st.read("graph", s, columns=["ncid", "akey", "tk"]).group_by("ncid").agg(pl.col("tk").min(), pl.col("akey").min())
        m = collections.defaultdict(list)
        for v, k in g.select("ncid", "tk").iter_rows():
            m[k].append(v)
        ts = {k: (_set_fp(vs), len(vs)) for k, vs in m.items()}
        truth[s] = {v: ts[k] for v, k in g.select("ncid", "tk").iter_rows()}
        ckey[s] = dict(g.select("ncid", "akey").iter_rows())
        del g, m, ts
    univ = sorted(univ)
    hl = {v: _u(f"{v}|lag", 1000) for v in univ}
    hf = {v: _u(f"{v}|fresh", 1000) for v in univ}
    hs = {v: _u(f"{v}|std", 100000) for v in univ}
    drop = {}

    def hdrop(b):
        if b not in drop:
            drop[b] = _u(f"{b}|drop", 100000)
        return drop[b]
    out = collections.Counter()
    for arm, lag, p_drop, p_std in ARMS:
        vk, sd, g = {}, {}, collections.defaultdict(list)
        for v in univ:
            if not lag:
                d = BUILD_DATE
            else:
                x = hl[v]
                d = (BUILD_DATE if x < 0.40 else BUILD_SOURCES[3] if x < 0.65 else BUILD_SOURCES[2] if x < 0.80
                     else BUILD_SOURCES[1] if x < 0.90 else BUILD_SOURCES[0])
            r = pos[d].get(v)
            if r is None and d != BUILD_DATE and hf[v] < 0.50:
                r = pos[BUILD_DATE].get(v)
            if r is None:
                continue
            if hs[v] < p_std:
                ks = [f"S{v}"]
            else:
                ks = [f"B{b}" if b is not None and hdrop(b) < p_drop else f"U{a}" for a, b in r]
            for k in dict.fromkeys(ks):
                g[k].append(v)
            vk[v] = ks[0]; sd[v] = d
        vset = {k: (_set_fp(vs), len(vs)) for k, vs in g.items()}
        del g
        for v, cty in cl:
            t1 = sets[BUILD_DATE][pos[BUILD_DATE][v][0][0]]
            k = vk.get(v)
            lag_b = "omitted" if k is None else ("L0" if sd[v] == BUILD_DATE else "L+")
            vfp = vset[k] if k is not None else None
            for s in SCORE_DATES:
                t2 = truth[s].get(v)
                if t2 is None:
                    continue
                out[(arm, s, cty, t1[1] == 1, ckey[s][v] == pos[BUILD_DATE][v][0][0], lag_b, vfp == t1, vfp == t2, t1 == t2)] += 1
        del vk, sd, vset
    rows = [k + (n,) for k, n in out.items()]
    sch = {"arm": pl.String, "s": pl.Date, "county": pl.String, "alone": pl.Boolean, "stayer": pl.Boolean, "lag_b": pl.String,
           "built": pl.Boolean, "ev": pl.Boolean, "ep": pl.Boolean, "n": I}
    return pl.DataFrame(rows, schema=sch, orient="row").sort(["arm", "s", "county", "alone", "stayer", "lag_b", "built", "ev", "ep"])


def build_error_counties(cells: pl.DataFrame, path):
    """Per (arm, s, county): n, n_built, n_ev, n_ep, n_ev_built."""
    z = lambda c: pl.when(c).then(pl.col("n")).otherwise(0).sum().cast(I)
    x = (cells.group_by("arm", "s", "county").agg(pl.col("n").sum().cast(I).alias("n"), z(pl.col("built")).alias("n_built"),
                                                  z(pl.col("ev")).alias("n_ev"), z(pl.col("ep")).alias("n_ep"),
                                                  z(pl.col("ev") & pl.col("built")).alias("n_ev_built"))
         .sort(["arm", "s", "county"]))
    _write_rows(path, ["arm", "s", "county", "n", "n_built", "n_ev", "n_ep", "n_ev_built"], [[str(v) for v in r] for r in x.iter_rows()])


def build_error_summary(county_path, path, reps=BOOTSTRAP_REPS):
    """Per (arm, s): build accuracy a, kV, kP, kVc, kVw, kVc - kP and kV / (a kP) with 99% county bootstrap intervals
    (numpy default_rng(BUILD_ERROR_BOOTSTRAP_SEED); arms in ARMS order, scoring dates in order), and kV / kP. The exact arm
    has no build error, so its intervals are its point values."""
    cells = {}
    for r in csv.DictReader(open(county_path, encoding="utf-8")):
        cells.setdefault((r["arm"], r["s"]), []).append([int(r[k]) for k in ("n", "n_built", "n_ev", "n_ep", "n_ev_built")])

    def stats(m):
        n, b, ev, ep, evb = m
        a, kv, kp = b / n, ev / n, ep / n
        kvc, kvw = evb / b, (ev - evb) / (n - b) if n > b else float("nan")
        return dict(a=a, kV=kv, kP=kp, kVc=kvc, kVw=kvw, d_c=kvc - kp, ratio=kv / (a * kp), keep=kv / kp)
    rng = np.random.default_rng(BUILD_ERROR_BOOTSTRAP_SEED)
    lines = ["arm,s,a,kV,kP,kVc,kVw,kVc_minus_kP,lo_c,hi_c,ratio,lo_r,hi_r,kV_over_kP"]
    order = [a[0] for a in ARMS]
    for (arm, s), x in sorted(cells.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1])):
        X = np.array(x, dtype=np.int64); pt = stats([int(v) for v in X.sum(0)])
        idx = rng.integers(0, len(X), size=(reps, len(X)))
        bs = [stats([int(v) for v in X[i].sum(0)]) for i in idx]
        q = lambda k: np.percentile([b[k] for b in bs], [0.5, 99.5]) if arm != "exact" else (pt[k], pt[k])
        dlo, dhi = q("d_c"); rlo, rhi = q("ratio")
        lines.append(f"{arm},{s},{pt['a']:.4f},{pt['kV']:.4f},{pt['kP']:.4f},{pt['kVc']:.4f},{pt['kVw']:.4f},"
                     f"{pt['d_c']:+.4f},{dlo:+.4f},{dhi:+.4f},{pt['ratio']:.3f},{rlo:.3f},{rhi:.3f},{pt['keep']:.3f}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def exact_build_2018(county_path, table_skill_cells_path, path):
    """The exact build scored two ways at the 2018 build, 2026-11-03: the build-error experiment's exact arm (n and n_ev / n)
    and the later-file row of the coarse table at 2018-11-06, k = 8 (n, exact as written there). They must be equal."""
    n = ev = 0
    for r in csv.DictReader(open(county_path, encoding="utf-8")):
        if r["arm"] == "exact" and r["s"] == "2026-11-03":
            n += int(r["n"]); ev += int(r["n_ev"])
    lf = [r for r in csv.DictReader(open(table_skill_cells_path, encoding="utf-8"))
          if (r["test"], r["table"], r["t_dt"], r["k"]) == ("later_file_all", "coarse_table", "2018-11-06", "8")]
    assert len(lf) == 1
    _write_rows(path, ["source", "n", "exact"], [["build_error", str(n), repr(ev / n)], ["later_file", lf[0]["n"], lf[0]["exact"]]])
    # one client universe and one exact rule: the two must agree (to the precision the table cell is written at)
    x = lf[0]["exact"]
    places = len(x.split(".")[1]) if "." in x else 0
    assert n == int(lf[0]["n"]) and round(ev / n, places) == round(float(x), places), f"exact build differs: {n} {ev / n} vs {lf[0]}"
