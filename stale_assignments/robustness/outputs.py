# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Robustness checks: statistics and intervals from the county packs (paper section 9 and the supplement's robustness
tables). Every function reads and writes files under a results root R (layout.py). County-bootstrap intervals use
bootstrap.weights and bootstrap.summarize (Poisson(1) county weights, 2,000 replicates); the table checks use the table
bootstrap (counties resampled with replacement). Sums of floats use math.fsum, so values do not depend on the platform.
"""
import collections
import csv
import math
import os
import random

from .. import layout as L, parallel
from ..bootstrap import summarize, weights
from ..constants import BOOTSTRAP_REPS, POISSON_BOOTSTRAP_SALT, TABLE_BOOTSTRAP_SEED

REPS, SEED = BOOTSTRAP_REPS, POISSON_BOOTSTRAP_SALT

_SKILL_TESTS = [("held_out", "", "held_out_cells.csv", None), ("later_file_all", "_parity", "later_file_cells.csv", None),
                ("later_file_development", "_parity", "later_file_cells.csv", "development")]
_BASELINE_TESTS = _SKILL_TESTS + [("later_file_held_out", "_parity", "later_file_cells.csv", "held_out_only")]


def _write(path, cols, rows, fmt=lambda v: f"{v:.6f}" if isinstance(v, float) else v):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, "")) for c in cols})


def _sub(a, b):
    return None if a is None or b is None else a - b


# ---- stale-listing references ---------------------------------------------------------------------------------

def intervals(pack, outp):
    """Exact share per (t, k, policy, reference), minus the registered reference, and the 2016 policy gaps."""
    D = {}
    for r in csv.DictReader(open(pack, encoding="utf-8")):
        D[(int(r["t"]), int(r["k"]), r["policy"], r["variant"])] = {
            int(c.split(":")[0]): (int(c.split(":")[1]), int(c.split(":")[2])) for c in r["cells"].split()}
    counties = sorted({c for v in D.values() for c in v})
    W = weights(counties, SEED, REPS)

    def share(cells, w):
        n = math.fsum(w.get(c, 0) * x[0] for c, x in cells.items())
        e = math.fsum(w.get(c, 0) * x[1] for c, x in cells.items())
        return e / n if n else None

    rep = lambda f: summarize({b: f(w) for b, w in W.items()}, REPS)
    out = []
    emit = lambda t, k, pol, var, stat, s: out.append(dict(t=t, k=k, policy=pol, variant=var, stat=stat, **s))
    for (t, k, pol, var), cells in sorted(D.items()):
        emit(t, k, pol, var, "exact", rep(lambda w: share(cells, w)))
        if var != "registered" and (t, k, pol, "registered") in D:
            b = D[(t, k, pol, "registered")]
            emit(t, k, pol, var, "minus_base", rep(lambda w: _sub(share(cells, w), share(b, w))))
    for k in sorted({k for (t, k, _, _) in D if t == 2016}):
        pols = [p for (t, kk, p, v) in D if t == 2016 and kk == k and v == "registered"]
        order = sorted(pols, key=lambda p: -share(D[(2016, k, p, "registered")], W[0]))
        for var in L.REFERENCES:
            for hi, lo in zip(order, order[1:]):
                if (2016, k, hi, var) in D and (2016, k, lo, var) in D:
                    a, b = D[(2016, k, hi, var)], D[(2016, k, lo, var)]
                    emit(2016, k, f"{hi}-{lo}", var, "policy_gap", rep(lambda w: _sub(share(a, w), share(b, w))))
    _write(outp, ["t", "k", "policy", "variant", "stat", "point", "lo99", "hi99", "lo95", "hi95", "jk_se", "loco_min",
                  "loco_max", "n_null"], out)


def summary(inp, outp):
    """Exact shares by build, age, policy and reference; sigma, kappa_S, mu of the person anchor (last_known policy)."""
    R = list(csv.DictReader(open(inp, encoding="utf-8")))
    for r in R:
        for c in ("matched", "stay", "has_lookahead1", "has_lookahead2", "client_flag_lookahead1", "client_flag_inactive"):
            r[c] = r[c] == "true"
        for c in ("n", "e_registered", "e_lookahead1", "e_lookahead2", "e_inactive", "e_alone", "mem", "flag_lookahead1", "flag_lookahead2",
                  "flag_inactive"):
            r[c] = int(r[c])

    def ex(r, v):
        if v == "registered": return r["e_registered"], r["stay"]
        if v == "lookahead_one_election": return r["e_lookahead1"], r["stay"]
        if v == "lookahead_two_elections": return r["e_lookahead2"], r["stay"]
        if v == "inactive_dropped": return r["e_inactive"], r["stay"]
        if v == "client_lookahead":
            return (r["e_alone"], False) if r["client_flag_lookahead1"] else (r["e_lookahead1"], r["stay"])
        if v == "client_inactive":
            return (r["e_alone"], False) if r["client_flag_inactive"] else (r["e_inactive"], r["stay"])

    out = []
    for t, k in sorted({(int(r["t"]), int(r["k"])) for r in R}):
        for pol in L.POLICIES:
            rows = [r for r in R if int(r["t"]) == t and int(r["k"]) == k and r["policy"] == pol and r["matched"]]
            if not rows:
                continue
            for v in L.REFERENCES:
                sub = [r for r in rows if (r["has_lookahead2"] if v == "lookahead_two_elections" else r["has_lookahead1"])]
                if not sub:
                    continue
                n = sum(r["n"] for r in sub)
                e = ns = es = 0
                for r in sub:
                    x, st = ex(r, v)
                    e += x
                    if st:
                        ns += r["n"]
                        es += x
                row = dict(t=t, k=k, policy=pol, variant=v, n=n, exact=e / n)
                if pol == "last_known":
                    row.update(sigma=ns / n, kappa_S=es / ns if ns else None, mu=(e - es) / (n - ns) if n > ns else None)
                out.append(row)
    _write(outp, ["t", "k", "policy", "variant", "n", "exact", "sigma", "kappa_S", "mu"], out,
           lambda v: f"{v:.6f}" if isinstance(v, float) else ("" if v is None else v))


def compare(R):
    """Confirmation against held-out counties on the anchor accounting, with the held-out 99% intervals."""
    rd = lambda f: {(r["t"], r["k"], r["policy"], r["variant"]): r for r in csv.DictReader(open(L.robustness(R, f), encoding="utf-8"))}
    C, Rs = rd("stale_listing_summary.csv"), rd("stale_listing_summary_held_out.csv")
    I = {(r["t"], r["k"], r["policy"], r["variant"]): r
         for r in csv.DictReader(open(L.robustness(R, "stale_listing_intervals_held_out.csv"), encoding="utf-8")) if r["stat"] == "exact"}
    out = []
    for t in ("2008", "2010", "2012", "2014", "2016"):
        for k in ("2", "4", "6", "8"):
            for stat, pol, col in (("kappa_P", "last_known", "exact"), ("sigma", "last_known", "sigma"),
                                   ("kappa_S", "last_known", "kappa_S"), ("mu", "last_known", "mu"),
                                   ("kappa_A", "never_rebuilt", "exact"),
                                   ("probabilistic_matcher_w0", "probabilistic_matcher_w0", "exact"),
                                   ("probabilistic_matcher_w20", "probabilistic_matcher_w20", "exact"),
                                   ("annual_rebuild", "annual_rebuild", "exact")):
                key = (t, k, pol, "registered")
                if key not in C or key not in Rs:
                    continue
                iv = I.get(key) if col == "exact" else None
                c, r = float(C[key][col]), float(Rs[key][col])
                out.append(dict(t=t, k=k, stat=stat, confirmation=c, held_out=r, diff=r - c,
                                held_out_lo99=float(iv["lo99"]) if iv and iv["lo99"] else "",
                                held_out_hi99=float(iv["hi99"]) if iv and iv["hi99"] else ""))
    _write(L.robustness(R, "confirmation_vs_held_out.csv"), list(out[0]), out)


def w20_alone(R):
    """The probabilistic matcher at W >= 20 with unmatched clients scored as recorded alone."""
    rows_all = list(csv.DictReader(open(L.robustness(R, "stale_listing_rows.csv"), encoding="utf-8")))
    S = {(r["t"], r["k"], r["policy"], r["variant"]): r
         for r in csv.DictReader(open(L.robustness(R, "stale_listing_summary.csv"), encoding="utf-8"))}
    out = []
    for k in ("2", "4", "6", "8"):
        for pol in ("probabilistic_matcher_w20", "probabilistic_matcher_w0", "never_rebuilt", "annual_rebuild"):
            rows = [r for r in rows_all if r["t"] == "2016" and r["k"] == k and r["policy"] == pol]
            n = sum(int(r["n"]) for r in rows)
            m = sum(int(r["n"]) for r in rows if r["matched"] == "true")
            for v, col in (("registered", "e_registered"), ("lookahead_one_election", "e_lookahead1"),
                           ("inactive_dropped", "e_inactive")):
                em = sum(int(r[col]) for r in rows if r["matched"] == "true")
                eu = sum(int(r[col]) for r in rows if r["matched"] == "false")
                fz = S[("2016", k, "last_known", v)]
                out.append(dict(k=k, policy=pol, variant=v, n=n, match_rate=m / n, exact_matched=em / m,
                                exact_all_alone=(em + eu) / n, kappa_P=float(fz["exact"]), kappa_S=float(fz["kappa_S"])))
    _write(L.robustness(R, "probabilistic_matcher_w20_alone.csv"), list(out[0]), out)


# ---- unrecorded moves ------------------------------------------------------------------------------------------

def bracket(R):
    """Register move rates by measure, the unrecorded move rate u, and the bracket on sigma and kappa_P."""
    rows = list(csv.DictReader(open(L.robustness(R, "mobility_reasons.csv"), encoding="utf-8")))
    acs = {int(r["year"]): float(r["acs_moved"]) for r in
           csv.DictReader(open(L.results(R, "confirmation", "mobility_calibration.csv"), encoding="utf-8"))
           if r["band"] == "std2016" and r["acs_moved"]}
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["year"])].append(r)
    rates = []
    for y, g in sorted(by.items()):
        W = sum(int(r["n_clients"]) for r in g)
        f = lambda col: sum(int(r["n_clients"]) * int(r[col]) / int(r["n"]) for r in g) / W
        mv, died, out_, lm, oth, none = (f(c) for c in ("moved", "ex_died", "ex_moved_out", "ex_list_maint", "ex_other", "ex_none"))
        rates.append(dict(year=y, moved=mv, died=died, moved_out=out_, list_maint=lm, other=oth, none=none,
                          moved_only=mv, moved_or_out=mv + out_, all_exits=mv + out_ + lm + oth + none,
                          acs=acs.get(y)))
    _write(L.robustness(R, "mobility_rates.csv"), list(rates[0]), rates, lambda v: f"{v:.6f}" if isinstance(v, float) else ("" if v is None else v))
    yrs = [r for r in rates if r["acs"] is not None and 2010 <= r["year"] <= 2024]
    mean = lambda c: sum(r[c] for r in yrs) / len(yrs)
    U = {m: mean("acs") - mean(m) for m in ("moved_only", "moved_or_out", "all_exits")}
    K = {r["stat"]: float(r["point"]) for r in csv.DictReader(open(L.results(R, "later_file", "ten_year_2016_confirmation.csv"), encoding="utf-8"))}
    out = []
    for m, u in U.items():
        for k in (2, 4, 6, 8, 10):
            q = 1 - (1 - u) ** k
            s, kp, ks, mu = K[f"sigma({k})"], K[f"kappa_P({k})"], K[f"kappa_S({k})"], K[f"mu({k})"]
            out.append(dict(measure=m, u=u, k=k, q=q, sigma=s, sigma_adj=s * (1 - q), kappa_P=kp, kappa_P_A=kp,
                            kappa_P_B=kp - s * q * (ks - mu), dkappa_P_B=-s * q * (ks - mu)))
    _write(L.robustness(R, "unrecorded_move_bracket.csv"), list(out[0]), out)


# ---- paired differences of the probabilistic matcher -----------------------------------------------------------------------------------------

def paired(R):
    """W >= 20 minus kappa_S within recorded-alone strata; the probabilistic matcher minus the perfect person link."""
    C = {}
    for r in csv.DictReader(open(L.robustness(R, "matcher_paired_county_pack.csv"), encoding="utf-8")):
        v = {x: int(r[x]) for x in ("n", "n_stay", "e_stay", "n_ans", "e10_ans", "eP_ans")}
        g = "alone" if r["alone_t"] == "true" else "others"
        for grp in (g, "all"):
            d = C.setdefault((int(r["th"]), int(r["k"]), grp), {}).setdefault(int(r["county0"]), dict.fromkeys(v, 0))
            for x in v:
                d[x] += v[x]
    counties = sorted({c for d in C.values() for c in d})
    W = weights(counties, SEED, REPS)

    def ratio(cells, num, den, w):
        a = math.fsum(w.get(c, 0) * x[num] for c, x in cells.items())
        b = math.fsum(w.get(c, 0) * x[den] for c, x in cells.items())
        return a / b if b else None

    sel = lambda cells: lambda w: _sub(ratio(cells, "e10_ans", "n_ans", w), ratio(cells, "e_stay", "n_stay", w))
    link = lambda cells: lambda w: _sub(ratio(cells, "e10_ans", "n_ans", w), ratio(cells, "eP_ans", "n_ans", w))
    out = []
    for k in (2, 4, 6, 8):
        for grp in ("alone", "others", "all"):
            out.append(dict(diff="w20_minus_kappaS", k=k, group=grp, th=20, **summarize({b: sel(C[(20, k, grp)])(w) for b, w in W.items()}, REPS)))
        for th in (0, 20):
            out.append(dict(diff="matcher_minus_perfect_link", k=k, group="all", th=th,
                            **summarize({b: link(C[(th, k, "all")])(w) for b, w in W.items()}, REPS)))
    _write(L.robustness(R, "matcher_paired.csv"), ["diff", "k", "group", "th", "point", "lo99", "hi99", "lo95", "hi95", "jk_se",
                                               "loco_min", "loco_max", "n_null"], out)


# ---- anchor currency, full precision ----------------------------------------------------------------------------

_F16 = ["n", "eP", "m", "eA", "ns", "eS", "mm", "eMA"]
_FAM = {"confirmation": "confirmation", "development": "development", "confirmation_k10": "confirmation",
        "development_k10": "development", "later_file_confirmation": "later_file_confirmation",
        "later_file_development": "later_file_development",
        "later_file_confirmation_b2016": "later_file_confirmation_b2016"}


def _div(a, b):
    return a / b if b else None


def _stats16(S):
    """S: {'single': fields, 'multi': fields} of weighted sums -> the statistics (None where undefined)."""
    A = {f: S["single"][f] + S["multi"][f] for f in _F16}
    out = dict(kappa_P=_div(A["eP"], A["n"]), match_rate=_div(A["m"], A["n"]), kappa_A=_div(A["eA"], A["m"]),
               sigma=_div(A["ns"], A["n"]), kappa_S=_div(A["eS"], A["ns"]), mu=_div(A["eP"] - A["eS"], A["n"] - A["ns"]),
               kappa_MA=_div(A["eMA"], A["mm"]), sigma_kappa_S=_div(A["eS"], A["n"]),
               mover_share_of_exact=_div(A["eP"] - A["eS"], A["eP"]), single_sh=_div(S["single"]["n"], A["n"]))
    for sl in ("single", "multi"):
        X = S[sl]
        out[f"kappa_P_{sl}"] = _div(X["eP"], X["n"])
        out[f"kappa_S_{sl}"] = _div(X["eS"], X["ns"])
        out[f"mu_{sl}"] = _div(X["eP"] - X["eS"], X["n"] - X["ns"])
    out["mover_single_sh"] = _div(S["single"]["n"] - S["single"]["ns"], A["n"] - A["ns"])
    out["mu_from_single"] = _div(S["single"]["eP"] - S["single"]["eS"], A["eP"] - A["eS"])
    return out


def anchor_full(R):
    """Unrounded anchor-currency values with county-bootstrap intervals, the size split, and differences between adjacent
    ages within each build."""
    D = {}
    for r in csv.DictReader(open(L.robustness(R, "anchor_currency_pack.csv"), encoding="utf-8")):
        if r["src"].endswith("_k10") and r["k"] != "10":
            continue
        key = (_FAM[r["src"]], int(r["t"]), int(r["k"]))
        sl = "single" if r["single"] == "true" else "multi"
        for c in r["cells"].split():
            v = list(map(int, c.split(":")))
            d = D.setdefault(key, {}).setdefault(sl, {})
            assert v[0] not in d, (key, sl, v[0])
            d[v[0]] = v[1:]
    WC = {}
    zero = [0] * len(_F16)

    def series(cell):
        cs = tuple(sorted(set(cell.get("single", {})) | set(cell.get("multi", {}))))
        if cs not in WC:
            WC[cs] = weights(list(cs), SEED, REPS)
        W = WC[cs]
        vals = collections.defaultdict(dict)
        for b, w in W.items():
            S = {sl: {f: math.fsum(w[c] * cell.get(sl, {}).get(c, zero)[i] for c in cs) for i, f in enumerate(_F16)}
                 for sl in ("single", "multi")}
            for s, v in _stats16(S).items():
                vals[s][b] = v
        return vals, len(cs)

    def summ(v):
        s = summarize(v, REPS)
        return {c: s[c] for c in ("point", "lo99", "hi99", "lo95", "hi95", "jk_se", "loco_min", "loco_max")}

    rows, allst = [], {}
    for key in sorted(D):
        st, nc = series(D[key])
        allst[key] = st
        n = sum(x[0] for sl in D[key].values() for x in sl.values())
        for s, v in st.items():
            rows.append(dict(family=key[0], t=key[1], k=key[2], stat=s, counties=nc, n=n, **summ(v)))
    for (f, t, k) in sorted(D):
        nxt = [kk for (ff, tt, kk) in D if ff == f and tt == t and kk > k]
        if not nxt:
            continue
        k2 = min(nxt)
        a, b = allst[(f, t, k2)], allst[(f, t, k)]
        if set(next(iter(a.values()))) != set(next(iter(b.values()))):
            continue  # different county sets
        for s in a:
            rows.append(dict(family=f, t=t, k=f"{k2}-{k}", stat=s, counties="", n="",
                             **summ({r: _sub(a[s][r], b[s][r]) for r in a[s]})))
    _write(L.robustness(R, "anchor_currency_full.csv"), ["family", "t", "k", "stat", "counties", "n", "point", "lo99", "hi99", "lo95",
                                                    "hi95", "jk_se", "loco_min", "loco_max"], rows,
           lambda v: repr(float(v)) if isinstance(v, float) else ("" if v is None else v))


# ---- age-only baseline and Brier skill of the tables ---------------------------------------------------------------------------

def _vec(scored, counties):
    import numpy as np
    ci = {c: i for i, c in enumerate(counties)}
    S = sorted(scored, key=lambda s: (s[5], s[3], s[4]))
    cells = sorted({(s[1], s[2]) for s in S}); kc = {x: i for i, x in enumerate(cells)}
    a = lambda j: np.array([s[j] for s in S], dtype=float)
    return dict(c=np.array([ci[s[0]] for s in S]), cell=np.array([kc[(s[1], s[2])] for s in S]), n=a(3), ne=a(4), p=a(5),
                nc=len(cells))


def _vmetrics(v, wv):
    """Replicate metrics: rows sorted once by prediction; equal-mass deciles from the weighted running total. Sums by fsum,
    running totals by cumsum and group sums by bincount (fixed order), so values are the same on every platform."""
    import numpy as np
    m = wv[v["c"]]; n = v["n"] * m; ne = v["ne"] * m; p = v["p"]; N = math.fsum(n)
    brier = math.fsum(ne * (1 - p) ** 2 + (n - ne) * p ** 2) / N
    acc = np.cumsum(n) - n; b = np.minimum(9, np.floor(acc / (N / 10))).astype(int)
    ece = math.fsum(np.abs(np.bincount(b, p * n, 10) - np.bincount(b, ne, 10))) / N
    cn = np.bincount(v["cell"], n, v["nc"]); cp = np.bincount(v["cell"], p * n, v["nc"]); ce = np.bincount(v["cell"], ne, v["nc"])
    ok = cn > 0
    return dict(brier=brier, ece=ece, cell_mae=math.fsum(np.abs(cp[ok] - ce[ok]) / cn[ok]) / int(ok.sum()))


def _skill_one(task):
    """One test (worker of skill): its table_skill rows and table_skill_cells rows."""
    import numpy as np
    from ..table_scores import load, metrics, predict, read_table
    R, split, reps, seed, (test, tab, tname, cname, drop) = task
    out, cellrows = [], []
    cols, T, pk = read_table(L.tables(R, tname)); rows = load(L.table_cells(R, cname))
    if drop:
        rows = [r for r in rows if split[str(r["county0"])] != drop]
    st = [(r["county0"], r["t_dt"], r["k"], r["n"], r["ne"], predict(cols, T, pk, r)) for r in rows]
    sa = [(r["county0"], r["t_dt"], r["k"], r["n"], r["ne"], min(max(pk[r["k"]], 1e-6), 1 - 1e-6)) for r in rows]
    mt, ma = metrics(st), metrics(sa)
    for (t, k), (p, e, n) in mt["cells"].items():
        cellrows.append(dict(test=test, table=tab, t_dt=t, k=k, n=int(n), exact=e, pred_table=p, pred_age=ma["cells"][(t, k)][0]))
    counties = sorted({s[0] for s in st})
    rng = random.Random(seed)
    boot = collections.defaultdict(list)
    vt, va = _vec(st, counties), _vec(sa, counties)
    for _ in range(reps):
        draw = collections.defaultdict(int)
        for _ in counties:
            draw[rng.choice(counties)] += 1
        wv = np.array([draw.get(c, 0) for c in counties], dtype=float)
        bt, ba = _vmetrics(vt, wv), _vmetrics(va, wv)
        boot["brier"].append(bt["brier"]); boot["brier_age"].append(ba["brier"])
        boot["skill"].append(1 - bt["brier"] / ba["brier"])
        boot["cell_mae"].append(bt["cell_mae"]); boot["cell_mae_age"].append(ba["cell_mae"])
        boot["ece"].append(bt["ece"]); boot["ece_age"].append(ba["ece"])
    pt = dict(brier=mt["brier"], brier_age=ma["brier"], skill=1 - mt["brier"] / ma["brier"], cell_mae=mt["cell_mae"],
              cell_mae_age=ma["cell_mae"], ece=mt["ece"], ece_age=ma["ece"])
    for key, v in boot.items():
        v.sort()
        out.append(dict(test=test, table=tab, n=mt["n"], counties=len(counties), stat=key, point=pt[key],
                        lo95=v[50], hi95=v[1949], lo99=v[10], hi99=v[1989]))
    return out, cellrows


def skill(R, reps=BOOTSTRAP_REPS, seed=TABLE_BOOTSTRAP_SEED):
    """table_skill.csv and table_skill_cells.csv. The six tests run in parallel, results kept in test order."""
    split = {r["county_id"]: r["split"] for r in csv.DictReader(open(L.robustness(R, "county_split.csv"), encoding="utf-8"))}
    tests = [(t, tab, f"{tab}{suf}.csv", cells, drop) for t, suf, cells, drop in _SKILL_TESTS for tab in ("coarse_table", "comember_age_table")]
    out, cellrows = [], []
    for o, c in parallel.pmap(_skill_one, [(R, split, reps, seed, x) for x in tests]):
        out += o; cellrows += c
    _write(L.robustness(R, "table_skill.csv"), list(out[0]), out)
    _write(L.robustness(R, "table_skill_cells.csv"), list(cellrows[0]), cellrows)


# ---- stronger baselines, Murphy decomposition, level-set ECE ----------------------------------------------------

class _Pred:
    def __init__(self, c, n, ne, p, cell, nc):
        import numpy as np
        self.c, self.n, self.ne, self.p, self.cell, self.nc = c, n, ne, p, cell, nc
        self.lv, self.inv = np.unique(p, return_inverse=True); self.L = len(self.lv)

    def stats(self, wv):
        import numpy as np
        fs = math.fsum
        m = wv[self.c]; n = self.n * m; ne = self.ne * m; N = fs(n); yb = fs(ne) / N
        nl = np.bincount(self.inv, n, self.L); el = np.bincount(self.inv, ne, self.L); p = self.lv
        brier = fs(el * (1 - p) ** 2 + (nl - el) * p ** 2) / N
        ok = nl > 0; yl = np.where(ok, el / np.where(ok, nl, 1), 0)
        rel = fs(nl[ok] * (p[ok] - yl[ok]) ** 2) / N; res = fs(nl[ok] * (yl[ok] - yb) ** 2) / N
        b = np.minimum(9, np.floor((np.cumsum(nl) - nl) / (N / 10))).astype(int)
        ece = fs(np.abs(np.bincount(b, p * nl, 10) - np.bincount(b, el, 10))) / N
        cn = np.bincount(self.cell, n, self.nc); cp = np.bincount(self.cell, self.p * n, self.nc)
        ce = np.bincount(self.cell, ne, self.nc)
        okc = cn > 0
        return dict(brier=brier, rel=rel, res=res, cell_mae=fs(np.abs(cp[okc] - ce[okc]) / cn[okc]) / int(okc.sum()), ece_lv=ece)


def _baseline_one(task):
    """One test (worker of baselines): its table_baselines rows."""
    import numpy as np
    from ..table_scores import load, predict, read_table
    R, split, AS, AB, reps, seed, (test, tab, tname, cname, drop) = task
    clamp = lambda p: min(max(p, 1e-6), 1 - 1e-6)
    out = []
    cols, T, pk = read_table(L.tables(R, tname)); rows = load(L.table_cells(R, cname))
    if drop == "development":
        rows = [r for r in rows if split[str(r["county0"])] != "development"]
    if drop == "held_out_only":
        rows = [r for r in rows if split[str(r["county0"])] == "held_out"]
    counties = sorted({r["county0"] for r in rows}); ci = {c: i for i, c in enumerate(counties)}
    c = np.array([ci[r["county0"]] for r in rows]); n = np.array([r["n"] for r in rows], float)
    ne = np.array([r["ne"] for r in rows], float)
    cells = sorted({(r["t_dt"], r["k"]) for r in rows}); kc = {x: i for i, x in enumerate(cells)}
    cell = np.array([kc[(r["t_dt"], r["k"])] for r in rows])
    P = {"table": [predict(cols, T, pk, r) for r in rows], "age": [clamp(pk[r["k"]]) for r in rows],
         "age_x_size": [clamp(AS[(r["k"], r["size6"])]) for r in rows]}
    if test == "held_out":
        P["age_x_build"] = [clamp(AB[(r["k"], r["t_dt"])]) for r in rows]
    PR = {k: _Pred(c, n, ne, np.array(v), cell, len(cells)) for k, v in P.items()}

    def all_stats(wv):
        s = {f"{k}.{x}": v for k, pr in PR.items() for x, v in pr.stats(wv).items()}
        for base in ("age", "age_x_size"):
            s[f"table.skill_over_{base}"] = 1 - s["table.brier"] / s[f"{base}.brier"]
        s["age_x_size.skill_over_age"] = 1 - s["age_x_size.brier"] / s["age.brier"]
        return s
    pt = all_stats(np.ones(len(counties)))
    rng = random.Random(seed)
    boot = collections.defaultdict(list)
    for _ in range(reps):
        draw = collections.defaultdict(int)
        for _ in counties:
            draw[rng.choice(counties)] += 1
        for k, v in all_stats(np.array([draw.get(x, 0) for x in counties], float)).items():
            boot[k].append(v)
    for key in pt:
        v = sorted(boot[key]); pred, stat = key.split(".")
        out.append(dict(test=test, table=tab, counties=len(counties), predictor=pred, stat=stat, point=pt[key],
                        lo95=v[50], hi95=v[1949], lo99=v[10], hi99=v[1989]))
    return out


def baselines(R, reps=BOOTSTRAP_REPS, seed=TABLE_BOOTSTRAP_SEED):
    """table_baselines.csv. The eight tests run in parallel, results kept in test order."""
    from ..table_scores import load
    split = {r["county_id"]: r["split"] for r in csv.DictReader(open(L.robustness(R, "county_split.csv"), encoding="utf-8"))}
    tests = [(t, tab, f"{tab}{suf}.csv", cells, drop) for t, suf, cells, drop in _BASELINE_TESTS
             for tab in ("coarse_table", "comember_age_table")]
    dev = load(L.table_cells(R, "development_fine_cells.csv"))
    AS, AB = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0])
    for r in dev:
        for g, key in ((AS, (r["k"], r["size6"])), (AB, (r["k"], r["t_dt"]))):
            g[key][0] += r["n"]; g[key][1] += r["ne"]
    AS = {k: v[1] / v[0] for k, v in AS.items()}; AB = {k: v[1] / v[0] for k, v in AB.items()}
    out = []
    for o in parallel.pmap(_baseline_one, [(R, split, AS, AB, reps, seed, x) for x in tests]):
        out += o
    _write(L.robustness(R, "table_baselines.csv"), list(out[0]), out)


# ---- forecast bands for the later file -------------------------------------------------------------------------------------

def _linfit(x, y):
    """Least-squares line through (x, y): slope and intercept in closed form (fsum), in place of a library fit."""
    n = len(x); mx = math.fsum(x) / n; my = math.fsum(y) / n
    sl = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / math.fsum((a - mx) ** 2 for a in x)
    return sl, my - sl * mx


def forecast_bands(R):
    """forecast_bands.csv: bands for the later-file values from the earlier builds (last value and linear trend, +/- 0.05),
    from results/{confirmation,development}/anchor_currency.csv and results/later_file/anchor_currency_*.csv."""
    tests = [(2018, 8, [2010, 2014]), (2020, 6, [2008, 2012, 2016]), (2022, 4, [2010, 2014]), (2024, 2, [2008, 2012, 2016])]
    stats_ = ("kP", "kS", "sigma"); v2col = {"kP": "kappa_P", "kS": "kappa_S", "sigma": "sigma"}
    out = []
    for ds in ("confirmation", "development"):
        A = {(int(r["t"]), int(r["k"])): r for r in csv.DictReader(open(L.results(R, ds, "anchor_currency.csv"), encoding="utf-8"))}
        V = {(int(r["build"]), int(r["k"])): r
             for r in csv.DictReader(open(L.results(R, "later_file", f"anchor_currency_{ds}.csv"), encoding="utf-8"))}
        for T, k, an in tests:
            for c in stats_:
                ys = [float(A[(t, k)][c]) for t in an]; sl, ic = _linfit([float(t) for t in an], ys)
                tr, last = sl * T + ic, ys[-1]; lo, hi = min(tr, last) - 0.05, max(tr, last) + 0.05
                v = float(V[(T, k)][v2col[c]])
                out.append(dict(split=ds, kind="band", T=T, k=k, stat=c, last=last, trend=tr, lo=lo, hi=hi, value=v,
                                margin=min(v - lo, hi - v)))
        for kind, f in (("backtest_last", lambda y8, y12: y12), ("backtest_trend", lambda y8, y12: 2 * y12 - y8)):
            m = max(abs(float(A[(2016, k)][c]) - f(float(A[(2008, k)][c]), float(A[(2012, k)][c])))
                    for k in (2, 4, 6, 8) for c in stats_)
            out.append(dict(split=ds, kind=kind, T=2016, k="2-8", stat="kP,kS,sigma", margin=m))
    cols = ["split", "kind", "T", "k", "stat", "last", "trend", "lo", "hi", "value", "margin"]
    p = L.robustness(R, "forecast_bands.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols); w.writeheader()
        for r in out:
            w.writerow(r)


# ---- client universe and age reweighting ------------------------------------------------------------------------

def universe(R, ex, acs_path):
    """client_universe.csv from the class counts ex (robustness.data.universe_counts) and age_reweight.csv from
    age_band.csv and the ACS 2016 adult age mix (acs_path)."""
    tot = sum(ex.values())
    p = L.robustness(R, "client_universe.csv")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["class", "registrants", "share"])
        for c, v in ex.items():
            w.writerow([c, v, f"{v / tot:.6f}"])
        w.writerow(["all", tot, "1.000000"])
    acs = {r["age_band"]: int(r["population"]) for r in csv.DictReader(open(acs_path, encoding="utf-8"), delimiter="\t")
           if r["year"] == "2016"}
    A = collections.defaultdict(dict)
    for r in csv.DictReader(open(L.robustness(R, "age_band.csv"), encoding="utf-8")):
        if r["band"] in acs:
            A[int(r["k"])][r["band"]] = (int(r["n"]), int(r["exact"]))
    P = sum(acs.values()); out = []
    young = ("18-19", "20-24", "25-29", "30-34")
    for k, b in sorted(A.items()):
        n = sum(x[0] for x in b.values()); e = sum(x[1] for x in b.values())
        rw = sum(acs[band] / P * x[1] / x[0] for band, x in b.items())
        out.append(dict(k=k, n=n, kappa_P=e / n, kappa_P_acs=rw, diff=rw - e / n,
                        client_share_18_34=sum(b[x][0] for x in young) / n, acs_share_18_34=sum(acs[x] for x in young) / P))
    _write(L.robustness(R, "age_reweight.csv"), list(out[0]), out)


# ---- graded classes (shares) and the probabilistic matcher by recorded-alone stratum ---------------------------------------------------

GRADED = ["exact", "subset", "superset", "turnover", "disjoint"]


def graded_csv(R, g):
    """graded_matches.csv from the counts of robustness.data.graded."""
    rows = [dict(r, **{f"share_{c}": r[c] / r["n"] for c in GRADED}) for r in g.iter_rows(named=True)]
    _write(L.robustness(R, "graded_matches.csv"), list(g.columns) + [f"share_{c}" for c in GRADED], rows)


def matcher_alone(R):
    """probabilistic_matcher_alone.csv from probabilistic_matcher_alone_cells.csv: per (w_min, k) and group (all, recorded
    alone, recorded with others) the match rate, exact share among matched, kappa_S (last-known exact among stayers),
    kappa_P, the excess of the matched exact share over kappa_S, the stayers' share of matched and the exact share among
    matched stayers."""
    cells = list(csv.DictReader(open(L.robustness(R, "probabilistic_matcher_alone_cells.csv"), encoding="utf-8")))
    g = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in cells:
        for grp in ("all", "alone" if r["alone_t"] == "true" else "with_others"):
            a = g[(int(r["w_min"]), int(r["k"]), grp)]
            st = r["stay"] == "true"
            for c in ("n", "matched", "exact_matched", "exact_last_known"):
                a[c] += int(r[c])
                if st:
                    a["s_" + c] += int(r[c])
    order = {"all": 0, "alone": 1, "with_others": 2}
    out = []
    for (w, k, grp) in sorted(g, key=lambda x: (x[0], x[1], order[x[2]])):
        a = g[(w, k, grp)]
        em = a["exact_matched"] / a["matched"]; kS = a["s_exact_last_known"] / a["s_n"]
        out.append(dict(w_min=w, k=k, group=grp, n=a["n"], match_rate=a["matched"] / a["n"], exact_matched=em, kappa_S=kS,
                        kappa_P=a["exact_last_known"] / a["n"], excess_over_kappa_S=em - kS,
                        stayer_share_matched=a["s_matched"] / a["matched"],
                        exact_matched_stayers=a["s_exact_matched"] / a["s_matched"]))
    _write(L.robustness(R, "probabilistic_matcher_alone.csv"), list(out[0]), out)


def write_frame(df, path):
    """A count frame as CSV: header, then rows; booleans as true/false, nulls empty."""
    fmt = lambda v: "" if v is None else ("true" if v else "false") if isinstance(v, bool) else v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(df.columns)
        for r in df.iter_rows():
            w.writerow([fmt(v) for v in r])


def size_split_csv(R, parts):
    """size_split.csv from [(split, robustness.data.size_split frame)] with kappa_P, sigma, kappa_S and mu."""
    rows = []
    for split, df in parts:
        for r in df.iter_rows(named=True):
            n, e, ns, es = r["n"], r["exact"], r["stayers"], r["stayers_exact"]
            rows.append(dict(split=split, **r, kappa_P=e / n, sigma=ns / n, kappa_S=es / ns if ns else None,
                             mu=(e - es) / (n - ns) if n > ns else None))
    cols = ["split", "k", "size", "n", "exact", "stayers", "stayers_exact", "kappa_P", "sigma", "kappa_S", "mu"]
    _write(L.robustness(R, "size_split.csv"), cols, sorted(rows, key=lambda r: (r["split"], r["k"], r["size"])),
           fmt=lambda v: "" if v is None else f"{v:.6f}" if isinstance(v, float) else v)
