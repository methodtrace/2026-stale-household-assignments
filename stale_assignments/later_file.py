# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Later-file result files from the runs on the 41 files (paper sections 6.4 and 8.1): the anchor-currency rows of every
build at the Election Days to 2026-11-03, written from each run's anchor_currency.csv, the gap rows of the 2018 build,
the exact shares of the validity rules and the ten-year statistics of the 2016 build.
"""
import csv
import os

from . import store as S

EDAY = {d.year: d for d in S.SNAPSHOTS + [S.LATER_FILE_DATE] if S.series_of(d) == "eday"}


def anchor_rows(sources, path, last_year=2026):
    """later_file/anchor_currency_<split>.csv: build, k, s_dt, n, kappa_P, match_rate, kappa_A, sigma, kappa_S, mu, kappa_MA (4 places).
    sources: [(anchor_currency.csv path, builds to take from it)], in output order."""
    cols = [("kappa_P", "kP"), ("match_rate", "mr_addr"), ("kappa_A", "kA"), ("sigma", "sigma"), ("kappa_S", "kS"), ("mu", "mu"),
            ("kappa_MA", "kMA")]
    out = []
    for p, builds in sources:
        rows = {(int(r["t"]), int(r["k"])): r for r in csv.DictReader(open(p, encoding="utf-8"))}
        for b in builds:
            for k in (2, 4, 6, 8):
                if b + k > last_year or (b, k) not in rows:
                    continue
                r = rows[(b, k)]
                out.append([b, k, EDAY[b + k].isoformat(), r["n"]] + [f"{float(r[c]):.4f}" if r[c] else "" for _, c in cols])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["build", "k", "s_dt", "n"] + [c for c, _ in cols])
        w.writerows(out)


GAP_ROWS = [  # (ord in stats.csv, stat name)
    (0, 'gap G(2), never_rebuilt'),
    (1, 'widening G(8) - G(2), never_rebuilt'),
    (2, 'F1 margin, never_rebuilt'),
    (5, 'gap G(2), never_rebuilt | balanced panel'),
    (6, 'widening G(8) - G(2), never_rebuilt | balanced panel'),
    (7, 'F1 margin, never_rebuilt | balanced panel'),
    (10, 'gap G(2), never_rebuilt | county-weighted'),
    (11, 'widening G(8) - G(2), never_rebuilt | county-weighted'),
    (71, 'F1 margin, never_rebuilt | county-weighted'),
    (12, 'gap G(2), never_rebuilt | self-inclusive'),
    (13, 'widening G(8) - G(2), never_rebuilt | self-inclusive'),
    (14, 'F1 margin, never_rebuilt | self-inclusive'),
    (40, 'gap G(2), never_rebuilt | exits scored 0'),
    (41, 'widening G(8) - G(2), never_rebuilt | exits scored 0'),
    (42, 'F1 margin, never_rebuilt | exits scored 0'),
    (50, 'gap G(2), never_rebuilt | exits scored 1'),
    (51, 'widening G(8) - G(2), never_rebuilt | exits scored 1'),
    (52, 'F1 margin, never_rebuilt | exits scored 1'),
    (101, 'match rate among persons with truth, never_rebuilt, k = 2'),
    (102, 'co-member F1, never_rebuilt, k = 2'),
    (164, 'match rate among persons with truth, never_rebuilt, k = 8'),
    (165, 'co-member F1, never_rebuilt, k = 8'),
]


def gap_rows(stats_tsv, path):
    """later_file/gap_2018_confirmation.csv: the gap and widening rows of the 2018-build run's stats.tsv (point, 99% interval, jackknife SE at 4 places;
    threshold, decision and the leave-one-county-out flag of the decision rows)."""
    st = {r["stat"]: r for r in csv.DictReader(open(stats_tsv, encoding="utf-8"), delimiter="\t")}
    thr = lambda v: "" if v == "" else ("0" if float(v) == 0 else repr(float(v)))
    out = []
    for o, name in GAP_ROWS:
        r = st[name]
        dec = r["decision"]
        out.append([o, name, f"{float(r['point']):.4f}", f"{float(r['lo99']):.4f}", f"{float(r['hi99']):.4f}", f"{float(r['jk_se']):.4f}",
                    thr(r["thr"]) if dec else "", dec, r["loco_fragile"] if dec else ""])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ord", "stat", "point", "lo99", "hi99", "jk_se", "threshold", "decision", "loco_fragile"])
        w.writerows(out)


VALIDITY_RULES = ["last_known", "never_rebuilt", "probabilistic_matcher_w0", "probabilistic_matcher_w20",
                  "probabilistic_matcher_w25", "annual_rebuild"]
NO_INTERVAL = {"probabilistic_matcher_w25"}  # reported without intervals (point only)


def validity_rows(validity_cells, build, k, seed, reps):
    """Rows of later_file/validity_exact_<split>.csv for one run's main build at age k: the exact share among reported
    matches (exit treatment a) of each rule with its 99% county interval, and its difference from the last_known share
    (kappa_P) with the interval of the difference."""
    from .statistics import Key, ratio_stats
    ex = lambda r: f"match_validity|{r}|a|exact|{k}"
    derived = [(f"diff|{r}", Key(ex(r)) - Key(ex("last_known"))) for r in VALIDITY_RULES if r != "last_known"]
    cells = validity_cells.filter(validity_cells["key"].is_in([ex(r) for r in VALIDITY_RULES]))
    st = {r["stat"]: r for r in ratio_stats(cells, derived, seed, reps).iter_rows(named=True)}
    f4 = lambda v: "" if v is None else f"{v:.4f}"
    out = []
    for r in VALIDITY_RULES:
        e = st.get(ex(r))
        if e is None:  # no reported matches (a share without a denominator)
            out.append([build, k, r, "", "", "", "", "", ""])
        elif r == "last_known":
            out.append([build, k, r, f4(e["point"]), f4(e["lo99"]), f4(e["hi99"]), "0", "0", "0"])
        elif r in NO_INTERVAL:
            out.append([build, k, r, f4(e["point"]), "", "", "", "", ""])
        else:
            d = st[f"diff|{r}"]
            out.append([build, k, r, f4(e["point"]), f4(e["lo99"]), f4(e["hi99"]), f4(d["point"]), f4(d["lo99"]), f4(d["hi99"])])
    return out


def validity_exact(parts, path):
    """later_file/validity_exact_<split>.csv from [validity_rows(...)] in output order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["build", "k", "policy", "exact_share", "lo99", "hi99", "minus_kappa_P", "minus_kappa_P_lo99", "minus_kappa_P_hi99"])
        for p in parts:
            w.writerows(p)


TEN_YEAR_KS = (2, 4, 6, 8, 10)
TEN_YEAR_POLICIES = ("never_rebuilt", "lagged_rebuild", "same_date_rebuild", "four_year_rebuild", "annual_rebuild")
TEN_YEAR_ANCHOR = {"kP": "kP", "kA": "kA", "sig": "sigma", "kS": "kS", "mu": "mu", "kMA": "kMA"}  # name -> anchor_cells quantity


def ten_year_defs():
    """The statistics of the ten-year file, in output order: (name, expression over keys "<metric>|<policy>|<k>")."""
    from .statistics import Key
    q = lambda m, k, pol="never_rebuilt": Key(f"{m}|{pol}|{k}")
    g = lambda k, mr="mr", f="f1c": q(mr, k) - q(f, k)
    s = []
    for k in TEN_YEAR_KS:
        s += [(f"G({k})", g(k)), (f"mr({k})", q("mr", k)), (f"f1c({k})", q("f1c", k)), (f"kappa_P({k})", q("kP", k)),
              (f"kappa_A({k})", q("kA", k)), (f"sigma({k})", q("sig", k)), (f"kappa_S({k})", q("kS", k)), (f"mu({k})", q("mu", k)),
              (f"kappa_MA({k})", q("kMA", k))]
    s += [("G(10) - G(2)", g(10) - g(2)), ("G(10) - G(8)", g(10) - g(8)), ("G(8) - G(2)", g(8) - g(2)),
          ("G(10) - G(2), exits scored 1", g(10, "mr_all", "hi") - g(2, "mr_all", "hi")),
          ("G(10) - G(2), exits scored 0", g(10, "mr_all", "lo") - g(2, "mr_all", "lo")),
          ("kappa_P(8) - kappa_P(10)", q("kP", 8) - q("kP", 10)), ("mu(10) - mu(8)", q("mu", 10) - q("mu", 8))]
    for pol in TEN_YEAR_POLICIES[1:]:
        for k in (8, 10):
            s.append((f"f1c({k}), {pol}", q("f1c", k, pol)))
    return s


def ten_year_cells(cells, anchor, t_year=2016):
    """(county0, key, num, den) for the ten-year statistics from a run's step 3 cells (lag 0, build t_year, every client:
    the balanced flag is summed over) and its anchor_cells. Sums as statistics.METRIC_SUMS; hi = (f1c + n_all - n) / n_all."""
    import polars as pl
    c = (cells.filter((pl.col("t") == t_year) & (pl.col("lag") == 0) & pl.col("policy").is_in(list(TEN_YEAR_POLICIES))
         & pl.col("k").is_in(list(TEN_YEAR_KS)))
         .group_by("county0", "policy", "k").agg([pl.col(x).sum() for x in ("n_all", "n", "n_m_all", "n_m", "f1c")]))
    key = lambda m: pl.concat_str([pl.lit(m + "|"), pl.col("policy"), pl.lit("|"), pl.col("k").cast(pl.String)]).alias("key")
    F = pl.Float64
    spec = {"mr": ("n_m", "n"), "f1c": ("f1c", "n"), "mr_all": ("n_m_all", "n_all"), "lo": ("f1c", "n_all"),
            "hi": (pl.col("f1c") + pl.col("n_all") - pl.col("n"), "n_all")}
    out = [c.select("county0", key(m), (pl.col(a) if isinstance(a, str) else a).cast(F).alias("num"), pl.col(b).cast(F).alias("den"))
           for m, (a, b) in spec.items()]
    a = anchor.with_columns(pl.col("key").str.split("|").alias("_p")).with_columns(
        pl.col("_p").list.get(1).alias("q"), pl.col("_p").list.get(2).cast(pl.Int64).alias("t"), pl.col("_p").list.get(3).cast(pl.Int64).alias("k"))
    a = a.filter((pl.col("t") == t_year) & pl.col("k").is_in(list(TEN_YEAR_KS)))
    for m, qn in TEN_YEAR_ANCHOR.items():
        out.append(a.filter(pl.col("q") == qn).select("county0", pl.concat_str([pl.lit(m + "|never_rebuilt|"), pl.col("k").cast(pl.String)]).alias("key"),
                                                     pl.col("num").cast(F), pl.col("den").cast(F)))
    n = {k: v for k, v in a.filter(pl.col("q") == "n").group_by("k").agg(pl.col("num").sum()).iter_rows()}
    return pl.concat(out).sort(["key", "county0"]), n


def ten_year(cells, anchor, seed, reps, path, n_path, t_year=2016):
    """later_file/ten_year_2016_<split>.csv (ord, stat, point, lo99, hi99, jk_se at 6 places) and
    later_file/ten_year_2016_<split>_n.csv (k, n)."""
    from .statistics import ratio_stats
    kc, n = ten_year_cells(cells, anchor, t_year)
    defs = ten_year_defs()
    st = {r["stat"]: r for r in ratio_stats(kc, defs, seed, reps).iter_rows(named=True)}
    f6 = lambda v: "" if v is None else f"{v:.6f}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["ord", "stat", "point", "lo99", "hi99", "jk_se"])
        for i, (name, _) in enumerate(defs):
            r = st[name]
            w.writerow([i, name, f6(r["point"]), f6(r["lo99"]), f6(r["hi99"]), f6(r["jk_se"])])
    with open(n_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["k", "n"])
        w.writerows([k, n[k]] for k in sorted(n))
