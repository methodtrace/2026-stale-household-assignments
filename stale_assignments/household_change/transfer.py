# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The household-change model carried to Ohio and Washington with each register's own listing process (paper section 9).

For each target pair (build t, later file s) the North Carolina household-change coefficients stay fixed. Only the
removal hazard of the listing sub-chain is re-estimated, from build-time retention: the number of clients at t still
listed at s, by composition cell and building type. No household outcome at s is read. Variants, with the removal rate
rm x exp(adjustment):
  nc_listing           North Carolina listing, unchanged.
  listing_offset       one log offset per pair.
  listing_age_profile  a piecewise-linear adjustment of the removal age profile in current age on knots 22, 35, 50, 65, 80 (constant beyond
                       the end knots), plus a multi-unit building shift, with a ridge penalty 1,000 x sum(theta^2).
The retention fit uses the same deterministic arithmetic, step, gradient and optimizer as the household-change fit.
Writes fit_listing.json and pred_cells.csv.gz.
"""
import csv
import datetime as dt
import gzip
import json
import math

import numpy as np

from ..exact_math import exp, log
from . import model as M
from ..files import gz_text_writer
from .model import KN, H, EDGES, REMOVAL_EDGES, edge_rates, flow, flow_t, removal_factor, propagate
from .fit import _bc
from .optimize import LBFGS

KC = np.array([22., 35, 50, 65, 80])
RIDGE = 1000.0
CH = 2500
PAIRS = [("OH", "2017-07-18", "2026-10-23"), ("WA", "2017-06-30", "2026-10-01"), ("WA", "2018-06-30", "2026-10-01"),
         ("WA", "2017-06-30", "2018-06-30")]
VARIANTS = ("nc_listing", "listing_offset", "listing_age_profile")
NPAR = {"nc_listing": 0, "listing_offset": 1, "listing_age_profile": 6}


def _knots(C, n_steps):
    """Index and weight of current age on the KC knots, and the multi-unit indicator."""
    age = KN[C.ia[:, :n_steps]] + C.fa[:, :n_steps] * (KN[C.ia[:, :n_steps] + 1] - KN[C.ia[:, :n_steps]])
    a = np.clip(age, KC[0], KC[-1] - 1e-9)
    i = np.searchsorted(KC, a, side="right") - 1
    return i, (a - KC[i]) / (KC[i + 1] - KC[i])


def _adj(C, var, th, n_steps, kn):
    if var == "nc_listing":
        return None
    if var == "listing_offset":
        return np.full((C.n, n_steps), th[0])
    i, fr = kn
    return th[:5][i] * (1 - fr) + th[:5][i + 1] * fr + th[5] * C.bt[:, None]


def chunk_loss(P, C, n_steps, var, th, nt, nl, grad):
    """Binomial retention loss of cells C (and its gradient in theta)."""
    R = M.rates(P, C)
    rm0 = R["rm"][:, :n_steps]
    kn = _knots(C, n_steps) if var == "listing_age_profile" else None
    adj = _adj(C, var, th, n_steps, kn)
    R = dict(R)
    R["rm"] = rm0 * exp(adj) if adj is not None else rm0
    s, hist = propagate(R, C.n, n_steps, keep=grad)
    p = ((((((s[0] + s[1]) + s[2]) + s[3]) + s[4]) + s[5]) + s[6]) + s[7]
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    loss = -math.fsum((nl * log(pc) + (nt - nl) * log(1 - pc)).tolist())
    if not grad or var == "nc_listing":
        return loss, None
    gp = np.where((p > 1e-12) & (p < 1 - 1e-12), -(nl / pc - (nt - nl) / (1 - pc)), 0.0)
    lam = [gp if i < 8 else np.zeros(C.n) for i in range(16)]
    grm = np.zeros((C.n, n_steps))
    for k in range(n_steps - 1, -1, -1):
        er = edge_rates(R, k); x0 = hist[k]
        a = flow(x0, er)
        mu = flow_t(lam, er); mu2 = flow_t(mu, er)
        g = None
        for e in range(8):
            _, i, j = EDGES[REMOVAL_EDGES[e]]
            ge = (H * x0[i] + 0.5 * H * H * a[i]) * (lam[j] - lam[i]) + (0.5 * H * H * x0[i]) * (mu[j] - mu[i])
            t = ge * removal_factor(R, e)
            g = t if g is None else g + t
        grm[:, k] = g
        lam = [lam[i] + H * mu[i] + 0.5 * H * H * mu2[i] for i in range(16)]
    gadj = grm * R["rm"]                                   # d loss / d adjustment
    if var == "listing_offset":
        return loss, np.array([math.fsum(gadj.ravel().tolist())])
    i, fr = kn
    gt = np.zeros(6)
    gt[:5] = _bc(i, gadj * (1 - fr), 5) + _bc(i + 1, gadj * fr, 5)
    gt[5] = math.fsum((gadj * C.bt[:, None]).ravel().tolist())
    return loss, gt


def pair_loss(P, C, n_steps, var, th, nt, nl, grad=True):
    L, G = [], []
    for a in range(0, C.n, CH):
        ix = np.arange(a, min(a + CH, C.n))
        l, g = chunk_loss(P, C.subset(ix), n_steps, var, th, nt[ix], nl[ix], grad)
        L.append(l); G.append(g)
    loss = math.fsum(L)
    if var == "listing_age_profile":
        loss = loss + RIDGE * math.fsum((th * th).tolist())
    if not grad or var == "nc_listing":
        return loss, None
    g = np.array([math.fsum(x[p] for x in G) for p in range(NPAR[var])])
    if var == "listing_age_profile":
        g = g + 2 * RIDGE * th
    return loss, g


def marginals(P, C, n_steps, var, th):
    out = []
    for a in range(0, C.n, CH):
        ix = np.arange(a, min(a + CH, C.n)); c = C.subset(ix)
        R = dict(M.rates(P, c)); rm0 = R["rm"][:, :n_steps]
        adj = _adj(c, var, th, n_steps, _knots(c, n_steps) if var == "listing_age_profile" else None)
        R["rm"] = rm0 * exp(adj) if adj is not None else rm0
        out.append(propagate(R, c.n, n_steps)[0])
    return [np.concatenate([o[i] for o in out]) for i in range(16)]


def _run_pair(task):
    """One pair's three listing variants: (key, fits, prediction rows, report lines). Pairs are independent."""
    fit_path, retention_path, (st, t, s) = task
    P = M.load_fit(fit_path)
    g = sorted((r for r in csv.DictReader(open(retention_path, encoding="utf-8"))
                if r["state"] == st and r["t"] == t and r["s"] == s), key=lambda r: (r["cell"], r["bt"]))
    gap = (dt.date.fromisoformat(s) - dt.date.fromisoformat(t)).days / 365.25
    n_steps = int(np.rint(gap / H))
    C = M.Cells([r["cell"] for r in g], [r["bt"] for r in g], t)
    nt = np.array([float(r["n_t"]) for r in g]); nl = np.array([float(r["n_listed"]) for r in g])
    key = f"{st} {t} {s}"; fits, preds = {}, []
    say = [f"== {key}: gap {gap:.3f} y, {n_steps} steps, {len(g)} cells, n_t {int(nt.sum())}, "
           f"retention {math.fsum(nl.tolist()) / math.fsum(nt.tolist()):.4f}"]
    for var in VARIANTS:
        th = None
        if NPAR[var]:
            fg = lambda x, var=var: pair_loss(P, C, n_steps, var, x, nt, nl)
            opt = LBFGS(fg, np.zeros(NPAR[var]), max_iter=60, tolerance_grad=1e-7, tolerance_change=1e-10,
                        history_size=20)
            opt.step(); th = opt.x
        nll = pair_loss(P, C, n_steps, var, th, nt, nl, grad=False)[0]
        q = M.derived(marginals(P, C, n_steps, var, th))
        r_model = math.fsum((nt * q["listed"]).tolist()) / math.fsum(nt.tolist())
        fits[var] = dict(nll=nll, params=None if th is None else th.tolist(), retention_model=r_model)
        say.append(f"{var}: nll {nll:.1f}, model retention {r_model:.4f}" + ("" if th is None else f", theta {np.round(th, 4).tolist()}"))
        for r, i in zip(g, range(len(g))):
            preds.append([st, t, s, var, r["cell"], r["bt"], r["n_t"], r["n_listed"]]
                         + [f"{q[c][i]:.10f}" for c in ("listed", "kP", "sig", "kS", "mu")])
    return key, fits, preds, say


def run(fit_path, retention_path, out_dir, log=print):
    """The listing variants for every pair, the pairs side by side, merged in pair order."""
    from ..parallel import pmap
    fits, preds = {}, []
    for key, f, pr, say in pmap(_run_pair, [(fit_path, retention_path, p) for p in PAIRS], share=True):
        fits[key] = f; preds += pr
        for line in say:
            log(line)
    with gz_text_writer(f"{out_dir}/pred_cells.csv.gz") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["state", "t", "s", "var", "cell", "bt", "n_t", "n_listed", "listed", "kP", "sig", "kS", "mu"])
        w.writerows(preds)
    with open(f"{out_dir}/fit_listing.json", "w", encoding="utf-8") as f:
        json.dump(fits, f, indent=1); f.write("\n")
    return fits


# ---- scoring at the later file ------------------------------------------------------------------------------------------
# Table predictions at k = 8 (parity 'both' = mean of the mid and pres values; a missing cell takes the coarse table's
# pooled k = 8 share); the age x size comparator's k = 8 shares by household size (development cells). Probabilities are
# clipped to [1e-6, 1 - 1e-6]. Brier raw (primary) and after one logit shift per pair and predictor that matches the
# observed mean (reported). County-cluster bootstrap, 2,000 replicates, counties resampled within each pair. All sums are
# math.fsum or exact integer sums.
LONG = [("OH", "2017-07-18", "2026-10-23"), ("WA", "2017-06-30", "2026-10-01"), ("WA", "2018-06-30", "2026-10-01")]
SHORT = [("WA", "2017-06-30", "2018-06-30")]
PAIR_LABELS = {"OH 2017-07-18 2026-10-23": "OH17", "WA 2017-06-30 2026-10-01": "WA17", "WA 2018-06-30 2026-10-01": "WA18",
               "WA 2017-06-30 2018-06-30": "WA1y"}
AGE_SIZE_K8 = {"1": 0.48095755305976756, "2": 0.423426738305604, "3": 0.13904610453987881, "4": 0.0750304426215581,
               "5": 0.037402273932796756}
PRIMARY = "model_listing_age_profile"
LEVELS = ("kP", "sig", "kS", "mu", "sig2", "kS2", "mu2")
EPS = 1e-6
fs = math.fsum


def _clip(p):
    return min(max(p, EPS), 1 - EPS)


def _expit(x):
    return float(1.0 / (1.0 + exp(-x)))


def _logit(p):
    return float(log(p / (1 - p)))


def _parity_table(path, cols):
    tab = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["k"] == "8":
            tab.setdefault(tuple(r[c] for c in cols), {})[r["parity"]] = float(r["p"])
    return {k: (v.get("mid", v.get("pres")) + v.get("pres", v.get("mid"))) / 2 for k, v in tab.items()}


def _shift(n, ne, p):
    """Logit shift d with sum n expit(logit p + d) = sum ne (bisection, 100 halvings of [-5, 5])."""
    lp = [_logit(x) for x in p]; lo, hi = -5.0, 5.0; tgt = fs(ne)
    lp = np.array(lp); n = np.asarray(n, dtype=float)
    for _ in range(100):
        mid = (lo + hi) / 2
        if fs((n / (1 + exp(-(lp + mid)))).tolist()) < tgt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _county_sse(rows, p, recal):
    """{county: sum of squared error} (and the logit shift) for predictions p aligned with rows."""
    n = [r["n"] for r in rows]; ne = [r["ne"] for r in rows]
    d = _shift(n, ne, p) if recal else 0.0
    if recal:
        lp = np.array([_logit(x) for x in p])
        p = [_clip(x) for x in (1 / (1 + exp(-(lp + d)))).tolist()]
    acc = {}
    for r, q in zip(rows, p):
        acc.setdefault(r["county"], []).append(r["ne"] * (1 - q) ** 2 + (r["n"] - r["ne"]) * q ** 2)
    return {c: fs(v) for c, v in acc.items()}, d


def _observed(h):
    """Observed split under the raw unit-key rule (sig, kS, mu) and the fuzzy-key rule (sig2, kS2, mu2)."""
    S = lambda c: sum(r[c] for r in h)
    n = S("n"); ne = S("ne"); nEB = S("nEB"); nB = nEB + S("nXB"); nEB2 = S("nEB2"); nB2 = nEB2 + S("nXB2")
    return dict(n=n, kP=ne / n, sig=nB / n, kS=nEB / nB, mu=(ne - nEB) / (n - nB),
                sig2=nB2 / n, kS2=nEB2 / nB2, mu2=(ne - nEB2) / (n - nB2))


def _table_text(rows):
    cols = ["n"] + list(LEVELS)
    w0 = max(len(r["model"]) for r in rows)
    out = ["model".ljust(w0) + "".join(f"{c:>9}" for c in cols)]
    for r in rows:
        out.append(r["model"].ljust(w0) + "".join(
            f"{r['n']:>9d}" if c == "n" else (f"{r[c]:>9.4f}" if c in r else f"{'':>9}") for c in cols))
    return "\n".join(out)


def _read_gz(path, ints):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for c in ints:
            r[c] = int(r[c])
    return rows


def score(fit_dir, outcomes_path, table_cells_path, tables_dir, retention_path, out_dir, seed, reps=2000):
    """Score the transfer predictions (fit_dir/pred_cells.csv.gz, fit_listing.json) at the later file. outcomes: listed
    clients per (pair, county, cell, bt) by observed state; table_cells: the same clients per (pair, county, size6, bt,
    ab, ny, mb, gb). Writes result.txt and summary_long.csv (pair, kind, stat, point, lo99, hi99) under out_dir."""
    import random
    lines = []
    say = lines.append
    P = {}
    with gzip.open(f"{fit_dir}/pred_cells.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            P[(r["state"], r["t"], r["s"], r["var"], r["cell"], r["bt"])] = r
    H = _read_gz(outcomes_path, ("n", "nEB", "nEO", "nXB", "nXO", "nEB2", "nXB2"))
    for r in H:
        r["ne"] = r["nEB"] + r["nEO"]
    Bt = _read_gz(table_cells_path, ("n", "ne"))
    pk8 = {r["k"]: float(r["pk"]) for r in csv.DictReader(open(f"{tables_dir}/coarse_table_parity_pk.csv", encoding="utf-8"))}["8"]
    CT = _parity_table(f"{tables_dir}/coarse_table_parity.csv", ["size6", "bt", "ab"])
    CA = _parity_table(f"{tables_dir}/comember_age_table_parity.csv", ["size6", "bt", "ab", "ny", "mb", "gb"])
    for r in Bt:
        r["p_coarse_table"] = _clip(CT.get((r["size6"], r["bt"], r["ab"]), pk8))
        r["p_comember_age_table"] = _clip(CA.get((r["size6"], r["bt"], r["ab"], r["ny"], r["mb"], r["gb"]), pk8))
        r["p_age_size"] = _clip(AGE_SIZE_K8.get(r["size6"], pk8))
    rng = random.Random(seed)
    summary = {}
    for st, t, s in LONG + SHORT:
        key = f"{st} {t} {s}"; say(f"\n== {key}")
        h = [r for r in H if (r["state"], r["t"], r["s"]) == (st, t, s)]
        b = [r for r in Bt if (r["state"], r["t"], r["s"]) == (st, t, s)]
        nh = sum(r["n"] for r in h)
        assert nh == sum(r["n"] for r in b) and sum(r["ne"] for r in h) == sum(r["ne"] for r in b), "universes differ"
        o = _observed(h); rows = [dict(model="observed", **o)]
        pred = {}
        for v in VARIANTS:
            x = [P[(st, t, s, v, r["cell"], r["bt"])] for r in h]
            q = {c: [float(z[c]) for z in x] for c in ("kP", "sig", "kS", "mu")}
            n = [r["n"] for r in h]
            nB = fs(a * g for a, g in zip(n, q["sig"]))
            sg = nB / nh; ks = fs(a * g * k for a, g, k in zip(n, q["sig"], q["kS"])) / nB
            mu = fs(a * (1 - g) * m for a, g, m in zip(n, q["sig"], q["mu"])) / fs(a * (1 - g) for a, g in zip(n, q["sig"]))
            rows.append(dict(model=f"model_{v}", n=nh, kP=fs(a * k for a, k in zip(n, q["kP"])) / nh, sig=sg, kS=ks, mu=mu,
                             sig2=sg, kS2=ks, mu2=mu))      # the model has one split; compared with both observed rules
            pred[f"model_{v}"] = [_clip(k) for k in q["kP"]]
        # selection arithmetic: the unchanged listing's unconditional exact share over the cell's observed retention
        x = [P[(st, t, s, "nc_listing", r["cell"], r["bt"])] for r in h]
        pred["retention_ratio"] = [_clip(min(1.0, float(z["kP"]) * float(z["listed"]) / (int(z["n_listed"]) / int(z["n_t"]))))
                                   for z in x]
        rows.append(dict(model="retention_ratio", n=nh, kP=fs(r["n"] * p for r, p in zip(h, pred["retention_ratio"])) / nh))
        if (st, t, s) not in SHORT:
            nb = sum(r["n"] for r in b)
            for c in ("coarse_table", "comember_age_table", "age_size"):
                rows.append(dict(model=c, n=nb, kP=fs(r["n"] * r["p_" + c] for r in b) / nb))
        tb = {r["model"]: r for r in rows}
        say(_table_text(rows))
        for v in [f"model_{x}" for x in VARIANTS] + ["retention_ratio"]:
            say(f"  error {v}: " + ", ".join(f"{a} {tb[v][a] - o[a]:+.4f}" for a in LEVELS if a in tb[v]))
        summary[key] = dict(table=tb)
        if (st, t, s) in SHORT:
            continue
        preds = {m: (h, pred[m]) for m in [f"model_{x}" for x in VARIANTS] + ["retention_ratio"]}
        preds.update({c: (b, [r["p_" + c] for r in b]) for c in ("coarse_table", "comember_age_table", "age_size")})
        N = {}
        for r in h:
            N[r["county"]] = N.get(r["county"], 0) + r["n"]
        cs = sorted(N)
        comps = ["coarse_table", "comember_age_table", "age_size", "model_nc_listing", "retention_ratio", "model_listing_offset"]
        for mode in ("raw", "recal"):
            sse, sh = {}, {}
            for nm, (df, p) in preds.items():
                sse[nm], sh[nm] = _county_sse(df, p, mode == "recal")
            br = {nm: fs(sse[nm].values()) / nh for nm in sse}
            say(f"  Brier {mode}: " + ", ".join(f"{nm} {br[nm]:.5f}" for nm in br)
                + ("" if mode == "raw" else "; shifts " + ", ".join(f"{nm} {sh[nm]:+.3f}" for nm in sh)))
            boot = {c: [] for c in comps}
            for _ in range(reps):
                w = {}
                for _ in cs:
                    c = rng.choice(cs); w[c] = w.get(c, 0) + 1
                nn = sum(N[c] * m for c, m in w.items())
                bp = {nm: fs(sse[nm][c] * m for c, m in w.items()) / nn for nm in [PRIMARY] + comps}
                for c in comps:
                    boot[c].append(bp[PRIMARY] - bp[c])
            for c in comps:
                v = sorted(boot[c]); d = br[PRIMARY] - br[c]
                lo99, hi99 = v[int(0.005 * reps)], v[int(0.995 * reps) - 1]
                lo95, hi95 = v[int(0.025 * reps)], v[int(0.975 * reps) - 1]
                say(f"  D({PRIMARY}, {c}) {mode} = {d:+.5f}  99% [{lo99:+.5f}, {hi99:+.5f}]  95% [{lo95:+.5f}, {hi95:+.5f}]")
                summary[key][(mode, c)] = (d, lo99, hi99)
    _comparisons(summary, say)
    with open(f"{out_dir}/result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines).lstrip("\n") + "\n")
    _summary_long(summary, f"{fit_dir}/fit_listing.json", retention_path, f"{out_dir}/summary_long.csv")
    return summary


def _comparisons(summary, say):
    """The level and Brier comparisons the paper reports for the transfer (section 9)."""
    OH, WA9, WA8 = [f"{a} {b} {c}" for a, b, c in LONG]
    err = lambda k, m, a: summary[k]["table"][m][a] - summary[k]["table"]["observed"][a]
    say("\n== comparisons")
    say(f"Ohio kappa_P error: listing_age_profile {err(OH, PRIMARY, 'kP'):+.4f}, "
        f"nc_listing {err(OH, 'model_nc_listing', 'kP'):+.4f}")
    say("Washington kappa_P error, listing_age_profile: "
        + ", ".join(f"{k} {err(k, PRIMARY, 'kP'):+.4f}" for k in (WA9, WA8)))
    for rule, cols in (("raw unit-key rule", ("sig", "kS", "mu")), ("fuzzy-key rule", ("sig2", "kS2", "mu2"))):
        say(f"decomposition errors, {rule}: " + "; ".join(
            f"{k}: " + ", ".join(f"{a} {err(k, PRIMARY, a):+.4f}" for a in cols) for k in (OH, WA9, WA8)))


def _summary_long(summary, fit_path, retention_path, out_path):
    fit = json.load(open(fit_path, encoding="utf-8"))
    obs = {}
    for r in csv.DictReader(open(retention_path, newline="", encoding="utf-8")):
        k = f"{r['state']} {r['t']} {r['s']}"; a = obs.setdefault(k, [0, 0]); a[0] += int(r["n_t"]); a[1] += int(r["n_listed"])
    rows = []
    for key, p in PAIR_LABELS.items():
        v = summary[key]
        for model, r in v["table"].items():
            for col in LEVELS:
                if col in r:
                    rows.append((p, "level", f"{model}.{col}", f"{float(r[col]):.6f}", "", ""))
        for k, val in v.items():
            if k == "table":
                continue
            kind, other = k
            pt, lo, hi = (float(z) for z in val)
            rows.append((p, kind, f"D.{other}", f"{pt:.6f}", f"{lo:.6f}", f"{hi:.6f}"))
    for key, p in PAIR_LABELS.items():
        rows.append((p, "listing", "listing_offset.offset", f"{fit[key]['listing_offset']['params'][0]:.6f}", "", ""))
        for v_ in VARIANTS:
            rows.append((p, "listing", f"{v_}.retention", f"{fit[key][v_]['retention_model']:.6f}", "", ""))
        rows.append((p, "listing", "observed.retention", f"{obs[key][1] / obs[key][0]:.6f}", "", ""))
        rows.append((p, "listing", "n_listed", str(obs[key][1]), "", ""))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["pair", "kind", "stat", "point", "lo99", "hi99"]); w.writerows(rows)
