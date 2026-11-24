# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The household-change model on the held-out counties (paper section 8.2 and Table S9).

Predictions among listed anchors at the Election Day snapshots 2 to 10 years after each build; Brier against the two
currency tables and an age x size comparator on the same universe (the held-out cells of the table test); county-cluster
bootstrap of the Brier differences (2,000 replicates, counties resampled with replacement); a delete-one-county
jackknife of the same differences; and the build x age accounting cells (kappa_P, sigma, kappa_S, mu, model against
observed).

Inputs (aggregates only):
  outcomes      csv.gz: county0, t_dt, k, cell, bt, n, nEB, nEO, nXB, nXO (held-out anchors listed at the scoring date,
                by observed outcome)
  table_cells   csv: the held-out cells of the table test (county0, t_dt, k, table features, n, ne)
  fine_cells    csv: the development cells with size6 and ab (for the age x size comparator)
  tables        directory with coarse_table.csv and comember_age_table.csv (table_scores.fit_tables)
Writes under out: predictions.csv.gz, county_sse.csv, build_age_cells.csv, summary.csv, model_jackknife.csv, result.txt.
"""
import csv
import datetime as dt
import gzip
import math
import os
from collections import defaultdict

import numpy as np

from . import model as M
from ..files import gz_text_writer
from .panel import snapshot_grid
from ..constants import MODEL_BOOTSTRAP_SEED, BOOTSTRAP_REPS
from ..table_scores import load, predict, read_table, county_draws

ELECTION_DAYS = ["2010-11-02", "2012-11-06", "2014-11-04", "2016-11-08", "2018-11-06", "2020-11-03", "2022-11-08",
                 "2024-11-05", "2026-11-03"]
BUILDS = ["2008-11-04", "2010-11-02", "2012-11-06", "2014-11-04", "2016-11-08"]
EPS = 1e-6
MODEL = "household_change_model"
COMPARATORS = ("coarse_table", "comember_age_table", "age_size")
T99_19DF = 2.860935                     # 0.995 quantile of Student t with 19 degrees of freedom
fs = math.fsum


def _age(t, s):
    return int((dt.date.fromisoformat(s) - dt.date.fromisoformat(t)).days / 365.25 + 0.5)


def predict_cells(P, keys, t, grid):
    """keys: sorted (cell, bt); returns {(k, cell, bt): (kP, sig, kS, mu)} at the Election Days 2 to 10 years on."""
    s = grid[t]
    u = np.r_[0.0, [x[2] for x in s]]; sdt = [t] + [x[1] for x in s]
    gi = np.rint(u / M.H).astype(int)
    assert (np.diff(gi) > 0).all()
    C = M.Cells([c for c, _ in keys], [b for _, b in keys], t)
    mg = M.marginals(P, C, gi)
    out = {}
    for sd in ELECTION_DAYS:
        if sd not in sdt:
            continue
        k = _age(t, sd)
        if not 2 <= k <= 10:
            continue
        q = M.derived(mg[sdt.index(sd)])
        for i, (c, b) in enumerate(keys):
            out[(k, c, b)] = (float(q["kP"][i]), float(q["sig"][i]), float(q["kS"][i]), float(q["mu"][i]))
    return out


def _read_outcomes(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(county0=r["county0"], t_dt=r["t_dt"], k=int(r["k"]), cell=r["cell"], bt=r["bt"],
                             **{c: int(r[c]) for c in ("n", "nEB", "nEO", "nXB", "nXO")}))
    return rows


def _fcol(vals):
    """One float column as a data-frame printout shows it after rounding to 4 places: six decimals with the trailing
    zeros that every value shares removed (one decimal kept)."""
    v = [f"{float(np.round(x, 4)):.6f}" for x in vals]
    while all(x.endswith("0") for x in v) and all(len(x.split(".")[1]) > 1 for x in v):
        v = [x[:-1] for x in v]
    return v


def _frame(bk, cols):
    """The (t_dt, k)-indexed table as text (index levels left-aligned, the first level blank where it repeats, columns
    right-aligned, two spaces between columns)."""
    keys = list(bk)
    i0 = [t for t, _ in keys]; i1 = [str(k) for _, k in keys]
    w0 = max(len("t_dt"), *map(len, i0)); w1 = max(len("k"), *map(len, i1))
    body = {c: _fcol([bk[key][c] for key in keys]) for c in cols}
    wc = {c: max(len(c), *map(len, body[c])) for c in cols}
    lines = [" " * (w0 + 1 + w1) + "".join("  " + c.rjust(wc[c]) for c in cols),
             "t_dt".ljust(w0) + " " + "k".ljust(w1) + "".join("  " + " " * wc[c] for c in cols)]
    prev = None
    for r, (t, k) in enumerate(keys):
        a = (t if t != prev else "").ljust(w0); prev = t
        lines.append(a + " " + i1[r].ljust(w1) + "".join("  " + body[c][r].rjust(wc[c]) for c in cols))
    return "\n".join(lines)


def score(fit_path, snapshots_path, outcomes_path, table_cells_path, fine_cells_path, tables_dir, out,
          reps=BOOTSTRAP_REPS, seed=MODEL_BOOTSTRAP_SEED, expect_sha256=None):
    os.makedirs(out, exist_ok=True)
    log = []
    say = log.append
    rows = _read_outcomes(outcomes_path)
    P = M.load_fit(fit_path, expect_sha256)
    grid = snapshot_grid(snapshots_path)

    # predictions
    pred = {}; pred_rows = []
    for t in BUILDS:
        keys = sorted({(r["cell"], r["bt"]) for r in rows if r["t_dt"] == t})
        p = predict_cells(P, keys, t, grid)
        for (k, c, b), v in p.items():
            pred[(t, k, c, b)] = v
        for sd in ELECTION_DAYS:
            k = _age(t, sd)
            if 2 <= k <= 10 and (k, keys[0][0], keys[0][1]) in p:
                pred_rows += [(t, k, c, b) + p[(k, c, b)] for c, b in keys]
        say(f"predicted build {t}: {len(keys)} (cell, bt)")
    with gz_text_writer(os.path.join(out, "predictions.csv.gz")) as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["t_dt", "k", "cell", "bt", "kP", "sig", "kS", "mu"])
        for r in pred_rows:
            w.writerow(list(r[:4]) + [f"{v:.10f}" for v in r[4:]])
    for r in rows:
        q = pred[(r["t_dt"], r["k"], r["cell"], r["bt"])]
        r["kP"], r["sig"], r["kS"], r["mu"] = q
        r["ne"] = r["nEB"] + r["nEO"]
        r["p"] = min(max(q[0], EPS), 1 - EPS)

    # the universe of the table test, and the comparators on it
    tc = load(table_cells_path)
    a = defaultdict(lambda: [0, 0])
    for r in tc:
        g = a[(int(r["county0"]), r["t_dt"], r["k"])]; g[0] += r["n"]; g[1] += r["ne"]
    h = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["k"] <= 8:
            g = h[(int(r["county0"]), r["t_dt"], r["k"])]; g[0] += r["n"]; g[1] += r["ne"]
    dn = sum(abs(h[c][0] - v[0]) for c, v in a.items()); de = sum(abs(h[c][1] - v[1]) for c, v in a.items())
    say(f"universe check: {len(a)} groups, |dn| sum {dn}, |d ne| sum {de} (n total {sum(v[0] for v in a.values())})")
    scored = {}
    for nm in ("coarse_table", "comember_age_table"):
        cols, tab, pk = read_table(os.path.join(tables_dir, f"{nm}.csv"))
        scored[nm] = [(r["county0"], r["n"], r["ne"], predict(cols, tab, pk, r)) for r in tc]
    T = defaultdict(lambda: [0, 0]); K = defaultdict(lambda: [0, 0])
    for r in load(fine_cells_path):
        g = T[(r["k"], r["t_dt"], r["size6"], r["ab"])]; g[0] += r["n"]; g[1] += r["ne"]
        K[r["k"]][0] += r["n"]; K[r["k"]][1] += r["ne"]

    def p_age_size(r):
        g = T.get((r["k"], r["t_dt"], r["size6"], r["ab"])); g = g if g and g[0] > 0 else K[r["k"]]
        return min(max(g[1] / g[0], EPS), 1 - EPS)
    scored["age_size"] = [(r["county0"], r["n"], r["ne"], p_age_size(r)) for r in tc]

    # county sums of squared error
    acc = defaultdict(list); NN = defaultdict(int)
    for r in rows:
        if r["k"] <= 8:
            p = r["p"]
            acc[r["county0"]].append(r["ne"] * (1 - p) ** 2 + (r["n"] - r["ne"]) * p ** 2); NN[r["county0"]] += r["n"]
    sse = {MODEL: {c: fs(v) for c, v in acc.items()}}
    for nm, sc in scored.items():
        acc = defaultdict(list); n_ = defaultdict(int)
        for c, n, ne, p in sc:
            acc[c].append(ne * (1 - p) ** 2 + (n - ne) * p ** 2); n_[c] += n
        sse[nm] = {c: fs(v) for c, v in acc.items()}
        assert all(n_[c] == NN[c] for c in NN)
    counties = sorted(NN)
    with open(os.path.join(out, "county_sse.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["county0", "N"] + list(sse))
        for c in counties:
            w.writerow([c, NN[c]] + [f"{sse[m][c]:.6f}" for m in sse])

    def brier(nm, wt):
        return fs(wt[c] * sse[nm][c] for c in wt) / sum(wt[c] * NN[c] for c in wt)
    B = {nm: brier(nm, {c: 1 for c in counties}) for nm in sse}
    for nm in sse:
        say(f"Brier {nm}: {B[nm]:.5f}")
    boot = defaultdict(list)
    for d in county_draws(counties, reps, seed):
        for b in COMPARATORS:
            boot[b].append(brier(MODEL, d) - brier(b, d))

    def ci(v, lev):
        v = sorted(v); return v[int((1 - lev) / 2 * reps)], v[int((1 + lev) / 2 * reps) - 1]
    intervals = {}
    for b in COMPARATORS:
        c99, c95 = ci(boot[b], 0.99), ci(boot[b], 0.95); intervals[b] = (c99, c95)
        say(f"D({MODEL}, {b}) = {B[MODEL] - B[b]:+.5f}  99% [{c99[0]:+.5f}, {c99[1]:+.5f}]  "
            f"95% [{c95[0]:+.5f}, {c95[1]:+.5f}]  ({reps} reps)")

    # build x age accounting cells
    g = defaultdict(list)
    for r in rows:
        g[(r["t_dt"], r["k"])].append(r)
    bk = {}
    for key in sorted(g):
        x = g[key]
        n = sum(r["n"] for r in x); nB = sum(r["nEB"] + r["nXB"] for r in x); nO = n - nB
        pB = fs(r["n"] * r["sig"] for r in x)
        bk[key] = dict(n=n, kP_obs=sum(r["ne"] for r in x) / n, kP=fs(r["n"] * r["kP"] for r in x) / n, sig_obs=nB / n,
                       sig=pB / n, kS_obs=sum(r["nEB"] for r in x) / max(nB, 1),
                       kS=fs(r["n"] * r["sig"] * r["kS"] for r in x) / pB, mu_obs=sum(r["nEO"] for r in x) / max(nO, 1),
                       mu=fs(r["n"] * (1 - r["sig"]) * r["mu"] for r in x) / fs(r["n"] * (1 - r["sig"]) for r in x))
    cols = ["n", "kP_obs", "kP", "sig_obs", "sig", "kS_obs", "kS", "mu_obs", "mu"]
    with open(os.path.join(out, "build_age_cells.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(["t_dt", "k"] + cols)
        for (t, k), v in bk.items():
            w.writerow([t, k] + [f"{float(v[c]):.5f}" for c in cols])
    say("== build x k (held out, among listed)"); say(_frame(bk, cols))
    b20 = {key: v for key, v in bk.items() if key[1] <= 8}
    assert len(b20) == 20
    mae = {q: fs(abs(v[q] - v[q + "_obs"]) for v in b20.values()) / 20 for q in ("kP", "sig", "kS", "mu")}
    for q in ("sig", "kS", "mu"):
        say(f"{q} cell MAE over 20 cells: {mae[q]:.4f}")
    say(f'kappa_P cell MAE over 20 cells: {mae["kP"]:.4f}')
    ct = defaultdict(lambda: [[], []])
    for r, (c, n, ne, p) in zip(tc, scored["coarse_table"]):
        ct[(r["t_dt"], r["k"])][0].append(n * p); ct[(r["t_dt"], r["k"])][1].append(n)
    mae_ct = fs(abs(fs(ct[i][0]) / sum(ct[i][1]) - b20[i]["kP_obs"]) for i in b20) / len(b20)
    say(f"coarse_table kappa_P cell MAE over 20 cells: {mae_ct:.4f}")
    say("k = 10:"); say(_frame({key: v for key, v in bk.items() if key[1] == 10}, cols))
    with open(os.path.join(out, "result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    _summary(out, intervals, mae, mae_ct)
    _jackknife(out)


def _county_rows(out):
    return list(csv.DictReader(open(os.path.join(out, "county_sse.csv"), encoding="utf-8")))


def _summary(out, intervals, mae, mae_ct):
    """summary.csv: Brier and Brier differences from the county sums as written, the bootstrap intervals, the cell
    MAEs and the county-level differences against the coarse table."""
    rows = _county_rows(out)
    N = sum(float(r["N"]) for r in rows)
    res = []
    for m in (MODEL,) + COMPARATORS:
        res.append({"stat": f"brier_{m}", "point": f"{fs(float(r[m]) for r in rows) / N:.8f}"})
    for m in COMPARATORS:
        (lo99, hi99), (lo95, hi95) = intervals[m]
        d = fs(float(r[MODEL]) - float(r[m]) for r in rows) / N
        res.append({"stat": f"D_{m}", "point": f"{d:.8f}", "lo99": f"{lo99:+.5f}", "hi99": f"{hi99:+.5f}",
                    "lo95": f"{lo95:+.5f}", "hi95": f"{hi95:+.5f}"})
    for q in ("sig", "kS", "mu"):
        res.append({"stat": f"cell_mae_{q}", "point": f"{mae[q]:.4f}"})
    res.append({"stat": f"cell_mae_kP_{MODEL}", "point": f"{mae['kP']:.4f}"})
    res.append({"stat": "cell_mae_kP_coarse_table", "point": f"{mae_ct:.4f}"})
    d = [(float(r[MODEL]) - float(r["coarse_table"])) / float(r["N"]) for r in rows]
    res += [{"stat": "county_D_coarse_table_min", "point": f"{min(d):.8f}"},
            {"stat": "county_D_coarse_table_max", "point": f"{max(d):.8f}"},
            {"stat": "county_D_coarse_table_nneg", "point": str(sum(1 for x in d if x < 0))},
            {"stat": "N", "point": str(int(N))}]
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["stat", "point", "lo99", "hi99", "lo95", "hi95"]); w.writeheader(); w.writerows(res)


def _jackknife(out):
    """model_jackknife.csv: delete-one-county jackknife-t 99% intervals of the Brier differences (19 df)."""
    rows = _county_rows(out)
    diff = lambda rs, a, b: (fs(float(r[a]) for r in rs) - fs(float(r[b]) for r in rs)) / sum(float(r["N"]) for r in rs)
    res = []
    for b in ("coarse_table", "comember_age_table"):
        G = len(rows); full = diff(rows, MODEL, b)
        loo = [diff(rows[:i] + rows[i + 1:], MODEL, b) for i in range(G)]
        m = fs(loo) / G; se = math.sqrt((G - 1) / G * fs((x - m) ** 2 for x in loo))
        lower = sum(1 for r in rows if float(r[MODEL]) < float(r[b]))
        res.append({"comparator": b, "diff": f"{full:.6f}", "lo99": f"{full - T99_19DF * se:.6f}",
                    "hi99": f"{full + T99_19DF * se:.6f}", "counties": G, "model_lower": lower,
                    "sign_p_one_sided": f"{0.5 ** G:.3e}" if lower == G else ""})
    with open(os.path.join(out, "model_jackknife.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)


def stayer_definitions(out, rematch_summary_path):
    """stayer_def.csv: the accounting cells under the two stayer definitions: address re-match (from the robustness
    summary: last_known policy, registered reference) and the readdressing rule (from build_age_cells.csv)."""
    Rr = {(r["t"], r["k"]): r for r in csv.DictReader(open(rematch_summary_path, encoding="utf-8"))
          if r["policy"] == "last_known" and r["variant"] == "registered"}
    B = {(r["t_dt"][:4], r["k"]): r for r in csv.DictReader(open(os.path.join(out, "build_age_cells.csv"), encoding="utf-8"))
         if r["k"] in ("2", "4", "6", "8")}
    rows, E = [], {}
    for key in sorted(B):
        b, r = B[key], Rr[key]
        row = dict(t=key[0], k=key[1])
        for s, c in (("sig", "sigma"), ("kS", "kappa_S"), ("mu", "mu")):
            row[f"{s}_rematch"] = float(r[c]); row[f"{s}_readdr"] = float(b[f"{s}_obs"]); row[f"{s}_model"] = float(b[s])
            row[f"{s}_shift"] = row[f"{s}_readdr"] - row[f"{s}_rematch"]
            E.setdefault((s, "rematch"), []).append(abs(row[f"{s}_model"] - row[f"{s}_rematch"]))
            E.setdefault((s, "readdr"), []).append(abs(row[f"{s}_model"] - row[f"{s}_readdr"]))
        rows.append(row)
    cols = list(rows[0])
    summ = []
    for s in ("sig", "kS", "mu"):
        for d in ("rematch", "readdr"):
            e = E[(s, d)]
            summ.append(dict(t="all", k=f"mae_{s}_{d}", **{c: "" for c in cols[2:]}, value=sum(e) / len(e)))
        sh = [abs(r[f"{s}_shift"]) for r in rows]
        summ.append(dict(t="all", k=f"maxshift_{s}", **{c: "" for c in cols[2:]}, value=max(sh)))
    with open(os.path.join(out, "stayer_def.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols + ["value"], lineterminator="\n"); w.writeheader()
        for r in rows + summ:
            w.writerow({c: (f"{v:.6f}" if isinstance(v, float) else v) for c, v in r.items()})


def step_check(fit_path, panels, out):
    """The largest total exit rate of any latent state over every fitted (build, cell, building type) and grid step, and
    h times it (below 1: every second-order step matrix is nonnegative). panels: BuildPanel list. Writes step_check.csv."""
    P = M.load_fit(fit_path)
    mx = 0.0
    for B in panels:
        R = M.rates(P, B.C)
        er = None
        tot = [None] * 16
        for k in range(R["brk"].shape[1]):
            er = M.edge_rates(R, k)
            tot = [None] * 16
            for (_, i, _j), r in zip(M.EDGES, er):
                tot[i] = r if tot[i] is None else tot[i] + r
            mx = max(mx, max(float(np.max(np.broadcast_to(t, (B.C.n,)))) for t in tot))
        del er, tot
    with open(os.path.join(out, "step_check.csv"), "w", encoding="utf-8") as f:
        f.write(f"item,value\nmax_exit_rate,{mx!r}\nh_times_max,{mx * M.H!r}\n")
    return mx
