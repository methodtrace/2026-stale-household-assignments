# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Fit the household-change model from IEEE arithmetic only, so a fit gives the same bits on every platform.

The likelihood is the composite likelihood of every consecutive snapshot pair: for each composition cell, P(y1 | y0) with
the latent state at the earlier snapshot taken from the model marginal given y0. Everything here uses elementwise add,
subtract, multiply and divide, the exp and log of exact_math, and sequential sums (bincount, math.fsum): no platform math
library, no BLAS, no vectorized reductions. The gradient is the exact reverse-mode derivative of the same computation,
written out by hand, through the edge list and step of model.

Command line:
  python -m stale_assignments.household_change.fit --transitions T.csv.gz --snapshots S.csv --out fit.json
"""
import math

import numpy as np

from .. import exact_math
from . import model as M
from .model import (SPEC, NPAR, ND, NY, K, H, EDGES, OBSOF, REMOVAL_EDGES, REENTRY_EDGES, MOVED, removal_factor, rates,
                    edge_rates, flow, flow_t, step, unpack, pack, fixed_mask, cold_start, coefficient_sha256)
from .panel import load

# ---- rates: reverse pass --------------------------------------------------------------------------------------------------

def _bc(idx, w, n):
    return np.bincount(np.ravel(idx), weights=np.ravel(w), minlength=n)[:n]


def _pl_back(g, i, fr, n=K):
    """Gradient of pl(theta, i, fr) with respect to theta, for upstream g."""
    return _bc(i, g * (1 - fr), n) + _bc(np.asarray(i) + 1, g * fr, n)


def rates_back(P, C, R, G):
    """Parameter gradient from G: gradients of the loss with respect to brk, dep, im, xm, rm (n, steps), kr, rA, rB (n,)
    and the scalars ab, ob, uO, uX, re, eO. Rates past the last step used carry zero gradient."""
    g = {nm: np.zeros(max(n, 1)) for nm, n, _ in SPEC}
    nG = G["brk"].shape[1]
    sl = lambda a: a[..., :nG]
    dbin, cyi = C.dbin[:nG], C.cyi[:nG]
    ia, fa = C.ia[:, :nG], C.fa[:, :nG]
    # brk = sc (los + gain)
    gb = G["brk"]
    sc, los, gain, lo = sl(R["sc"]), sl(R["los"]), sl(R["gain"]), sl(R["lo"])
    gsc = _bc(np.broadcast_to(np.arange(nG)[None, :], gb.shape), gb * (los + gain), nG)       # sum over cells, per step
    e = gsc * sc[0]
    g["dl"] += _bc(dbin, e, ND); g["cy"] += _bc(cyi, e, NY)
    gin = gb * sc
    elo = gin[:, None, :] * lo                                                                    # (n, 4, G)
    im_, fm_ = C.im[:, :, :nG], C.fm[:, :, :nG]
    g["fL"] += _pl_back(elo, im_, fm_)
    egl = ((elo[:, 0] + elo[:, 1]) + elo[:, 2]) + elo[:, 3]
    g["gL"] += _pl_back(egl, ia, fa)
    cell_lo = _bc(np.broadcast_to(np.arange(C.n)[:, None], egl.shape), egl, C.n)
    g["hL"] += _bc(C.mi, cell_lo, 5); g["bL"] += _bc(C.bt, cell_lo, 2)
    cj = _bc(np.broadcast_to((np.arange(C.n)[:, None] * 4 + np.arange(4)[None, :])[:, :, None], elo.shape), elo, C.n * 4)
    g["gp"] += _bc(C.gapb, cj, 4)
    eg = gin * gain
    g["fG"] += _pl_back(eg, ia, fa)
    cg = _bc(np.broadcast_to(np.arange(C.n)[:, None], eg.shape), eg, C.n)
    g["hG"] += _bc(C.mi, cg, 5); g["bG"] += _bc(C.bt, cg, 2)
    # dep, im, xm, rm: exp of a linear predictor
    for rk, (dlk, cyk, fk, hk, bk) in (("dep", ("dlD", "cy", "fD", "hD", "bD")), ("im", ("dlM", "cyM", "fM", "hM", "bM")),
                                       ("xm", (None, "cyM", "fX", "hX", "bX")), ("rm", ("dU", "cU", "fU", None, "bU"))):
        e = G[rk] * sl(R[rk])
        col = _bc(np.broadcast_to(np.arange(nG)[None, :], e.shape), e, nG)
        if dlk: g[dlk] += _bc(dbin, col, ND)
        g[cyk] += _bc(cyi, col, NY)
        g[fk] += _pl_back(e, ia, fa)
        ce = _bc(np.broadcast_to(np.arange(C.n)[:, None], e.shape), e, C.n)
        if hk: g[hk] += _bc(C.mi, ce, 5)
        g[bk] += _bc(C.bt, ce, 2)
    g["kR"] += _bc(C.g, G["kr"] * R["kr"], 3)
    g["rA"] += _bc(C.mi, G["rA"] * R["rA"], 5); g["rB"] += _bc(C.mi, G["rB"] * R["rB"], 5)
    for a, b in (("ab", "phi"), ("ob", "w0"), ("uO", "uO"), ("uX", "uX"), ("re", "e0"), ("eO", "eO")):
        g[b][0] += G[a] * R[a]
    return pack({nm: (g[nm][0] if n == 0 else g[nm]) for nm, n, _ in SPEC})



# ---- the likelihood -------------------------------------------------------------------------------------------------------

def _rowsum(x):
    """Sum over the leading axis (the start vectors) in a fixed order."""
    t = x[0]
    for r in x[1:]:
        t = t + r
    return t


def _starts(v):
    """Start vectors for one interval: v itself, then v restricted to each observed category and normalized."""
    X = [np.empty((6,) + v[0].shape) for _ in range(16)]
    S = [None] * 5
    for c in range(5):
        st = [i for i in range(16) if OBSOF[i] == c]
        t = v[st[0]]
        for i in st[1:]:
            t = t + v[i]
        S[c] = t
    for i in range(16):
        X[i][0] = v[i]
        for c in range(5):
            X[i][1 + c] = v[i] / np.maximum(S[c], 1e-300) if OBSOF[i] == c else 0.0
    return X, S


def _starts_back(v, S, lamX):
    """Gradient with respect to v of the start vectors, given their adjoints lamX (16 arrays (6, n))."""
    dv = [lamX[i][0].copy() for i in range(16)]
    for c in range(5):
        st = [i for i in range(16) if OBSOF[i] == c]
        ok = S[c] > 1e-300
        Sc = np.where(ok, S[c], 1.0)
        dot = lamX[st[0]][1 + c] * v[st[0]]
        for i in st[1:]:
            dot = dot + lamX[i][1 + c] * v[i]
        for i in st:
            dv[i] = dv[i] + np.where(ok, (lamX[i][1 + c] - dot / Sc) / Sc, 0.0)
    return dv


_OBS_STATES = [[i for i in range(16) if OBSOF[i] == c] for c in range(5)]


def _probs(X):
    """Pr[a][b] = P(y1 = b | y0 = a): mass of category b in start vector 1 + a."""
    out = []
    for b in range(5):
        st = _OBS_STATES[b]
        t = X[st[0]][1:]
        for i in st[1:]:
            t = t + X[i][1:]
        out.append(t)                    # (5 a, n)
    return out


def block_loss(P, C, N, gi, grad=True):
    """Negative composite log likelihood of cells C with pair counts N (n, J, 5 y0, 5 y1); grid indexes gi of the
    snapshots (gi[0] = 0). Returns (loss, gradient vector or None)."""
    R = rates(P, C)
    n = C.n; J = len(gi) - 1
    v = [np.zeros(n) for _ in range(16)]; v[0] = np.ones(n)
    terms = []
    ints = []                                     # per interval: v at its start, start sums, and the end state
    for j in range(1, J + 1):
        X, S = _starts(v)
        X0 = X
        for k in range(gi[j - 1], gi[j]):
            X, _ = step(X, edge_rates(R, k))
        Pr = _probs(X)
        for a in range(5):
            for b in range(5):
                w = N[:, j - 1, a, b]
                if w.any():
                    terms.append(w * exact_math.log(np.maximum(Pr[b][a], 1e-12)))
        ints.append((v, S, X0))
        v = [X[i][0] for i in range(16)]
    loss = -math.fsum(np.concatenate(terms)) if terms else 0.0
    if not grad:
        return loss, None
    nG = gi[-1]
    G = dict(brk=np.zeros((n, nG)), dep=np.zeros((n, nG)), im=np.zeros((n, nG)), xm=np.zeros((n, nG)),
             rm=np.zeros((n, nG)), kr=np.zeros(n), rA=np.zeros(n), rB=np.zeros(n))
    sc = dict(ab=[], ob=[], uO=[], uX=[], re=[], eO=[])
    dv_next = [np.zeros(n) for _ in range(16)]    # gradient with respect to v at the end of the current interval
    for j in range(J, 0, -1):
        v0, S0, X0 = ints[j - 1]
        # recompute the interval's states
        Xs = [X0]; ers = []
        for k in range(gi[j - 1], gi[j]):
            er = edge_rates(R, k); ers.append(er)
            Xs.append(step(Xs[-1], er)[0])
        Xe = Xs[-1]
        Pr = _probs(Xe)
        lam = [np.zeros((6, n)) for _ in range(16)]
        for i in range(16):
            lam[i][0] = dv_next[i]
        for b in range(5):
            pb = Pr[b]
            gpb = np.zeros((5, n))
            for a in range(5):
                w = N[:, j - 1, a, b]
                if w.any():
                    gpb[a] = np.where(pb[a] > 1e-12, -w / np.maximum(pb[a], 1e-12), 0.0)
            for i in _OBS_STATES[b]:
                lam[i][1:] = lam[i][1:] + gpb
        for kk in range(len(ers) - 1, -1, -1):
            k = gi[j - 1] + kk
            er = ers[kk]; s = Xs[kk]
            a = flow(s, er)
            mu = flow_t(lam, er)
            mu2 = flow_t(mu, er)
            x = [H * s[i] + 0.5 * H * H * a[i] for i in range(16)]
            y = [0.5 * H * H * s[i] for i in range(16)]
            ge = []
            for (_, i, jj) in EDGES:
                ge.append(_rowsum(x[i] * (lam[jj] - lam[i]) + y[i] * (mu[jj] - mu[i])))
            _edge_grads(R, k, ge, G, sc)
            lam = [lam[i] + H * mu[i] + 0.5 * H * H * mu2[i] for i in range(16)]
        dv_next = _starts_back(v0, S0, lam)
    Gs = {a: math.fsum(np.concatenate([np.atleast_1d(t) for t in sc[a]])) if sc[a] else 0.0 for a in sc}
    G.update(Gs)
    return loss, rates_back(P, C, R, G)


def _edge_grads(R, k, ge, G, sc):
    """Distribute edge-rate gradients ge (one (n,) array per edge) to the model rates at step k."""
    b = R["brk"][:, k]
    gb = None; gbr = None; gd = None; gim = None; gxm = None; gra = None; grb = None; gab = []; gob = []
    for (r, i, j), g in zip(EDGES, ge):
        if r == "brk": gb = g if gb is None else gb + g
        elif r == "br": gbr = g if gbr is None else gbr + g
        elif r == "dep": gd = g if gd is None else gd + g
        elif r == "im": gim = g if gim is None else gim + g
        elif r == "xm": gxm = g if gxm is None else gxm + g
        elif r == "rA": gra = g if gra is None else gra + g
        elif r == "rB": grb = g if grb is None else grb + g
        elif r == "ab": gab.append(g)
        elif r == "ob": gob.append(g)
    G["brk"][:, k] += gb + gbr * R["kr"]
    G["kr"] += gbr * b
    G["dep"][:, k] += gd; G["im"][:, k] += gim; G["xm"][:, k] += gxm
    G["rA"] += gra; G["rB"] += grb
    sc["ab"].append(_cat(gab)); sc["ob"].append(_cat(gob))
    rm = R["rm"][:, k]
    grm = None; guO = []; guX = []; gre = []; geO = []
    for i in range(8):
        g = ge[REMOVAL_EDGES[i]]
        f = removal_factor(R, i)
        t = g * f
        grm = t if grm is None else grm + t
        rr = g * (rm * f)                                     # gradient times the edge rate: d / d log factor
        if MOVED[i]: guO.append(rr)
        if M.BROKEN[i]: guX.append(rr)
        g2 = ge[REENTRY_EDGES[i]]
        e = R["re"] * (R["eO"] if MOVED[i] else 1.0)
        gre.append(g2 * e)                                    # d / d log re, divided back below
        if MOVED[i]: geO.append(g2 * e)
    G["rm"][:, k] += grm
    # store gradients with respect to the scalar rates themselves (rates_back multiplies by the rate)
    sc["uO"].append(_cat(guO) / R["uO"]); sc["uX"].append(_cat(guX) / R["uX"])
    sc["re"].append(_cat(gre) / R["re"]); sc["eO"].append(_cat(geO) / R["eO"])


def _cat(xs):
    return np.concatenate([np.atleast_1d(x) for x in xs]) if xs else np.zeros(0)


# ---- parallel evaluation ------------------------------------------------------------------------------------------------
_W = {}


def _init(transitions_path, snapshots_path, builds, chunk):
    _W["B"] = load(transitions_path, snapshots_path, builds)
    _W["chunk"] = chunk


def blocks(Bs, chunk):
    """Evaluation blocks (build index, first cell) in their fixed summation order."""
    return [(bi, a) for bi, B in enumerate(Bs) for a in range(0, B.C.n, chunk)]


def _eval_blocks(task):
    x, blk, grad = task
    P = unpack(x); out = []
    for bi, a in blk:
        B = _W["B"][bi]
        ix = np.arange(a, min(a + _W["chunk"], B.C.n))
        out.append(block_loss(P, B.C.subset(ix), B.N[ix], B.gi, grad))
    return out


class Objective:
    """Loss per pair and its gradient over every block, computed by a pool of worker processes. Block results are added
    in the fixed block order, so the value does not depend on the number of workers."""

    def __init__(self, transitions_path, snapshots_path, builds, chunk=500, workers=None):
        from .. import parallel
        _init(transitions_path, snapshots_path, builds, chunk)
        self.blocks = blocks(_W["B"], chunk)
        self.pairs = math.fsum(math.fsum(B.N.ravel()) for B in _W["B"])
        self.fixed = fixed_mask()
        self.nw = workers or parallel.workers()
        self.pool = parallel.pool(self.nw, _init, (transitions_path, snapshots_path, builds, chunk))
        per = max(1, len(self.blocks) // (4 * self.nw))
        self.groups = [self.blocks[i:i + per] for i in range(0, len(self.blocks), per)]
        self.evals = 0

    def __call__(self, x, grad=True):
        self.evals += 1
        x = np.where(self.fixed, 0.0, x)
        res = [r for part in self.pool.map(_eval_blocks, [(x, g, grad) for g in self.groups]) for r in part]
        L = math.fsum(r[0] for r in res) / self.pairs
        if not grad:
            return L, None
        G = np.array([math.fsum(r[1][p] for r in res) for p in range(NPAR)]) / self.pairs
        G[self.fixed] = 0.0
        return L, G

    def close(self):
        self.pool.close(); self.pool.join()


# ---- the fit ------------------------------------------------------------------------------------------------------------
BUILDS = ["2008-11-04", "2010-11-02", "2012-11-06", "2014-11-04", "2016-11-08"]


def fit(transitions_path, snapshots_path, out_path, builds=BUILDS, chunk=500, workers=None, max_iter=150, rounds=2, log=print):
    """Cold-start fit: LBFGS (strong Wolfe, history 50, tolerances 1e-9 on the gradient and 1e-12 on change), rounds
    calls of at most max_iter iterations each. Writes the coefficients, the loss per pair at them, the largest gradient
    component and a SHA-256 of the coefficient bytes to out_path (JSON)."""
    import json, time
    from .optimize import LBFGS
    obj = Objective(transitions_path, snapshots_path, builds, chunk, workers)
    log(f"builds {builds} pairs {obj.pairs:.0f} blocks {len(obj.blocks)} workers {obj.nw}")
    t0 = [time.time()]

    def lg(it, ev, loss):
        log(f"  iter {it} evals {ev}: {loss:.10f} ({time.time() - t0[0]:.0f}s)")
    opt = LBFGS(obj, cold_start(), max_iter=max_iter, log=lg)
    for r in range(rounds):
        opt.step()
        log(f"round {r}: evals {opt.func_evals}")
    x = np.where(obj.fixed, 0.0, opt.x)
    L, G = obj(x)
    obj.close()
    assert np.isfinite(L) and np.isfinite(G).all(), "the fit ended at a non-finite loss or gradient"
    P = unpack(x)
    res = dict(nll_per_pair=L, max_abs_gradient=float(np.max(np.abs(G))), pairs=obj.pairs, builds=list(builds),
               evals=opt.func_evals, iterations=opt.n_iter, chunk=chunk,
               sha256=coefficient_sha256(x),
               params={k: (v if isinstance(v, float) else v.tolist()) for k, v in P.items()})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    log(f"nll/pair {L!r} max|grad| {res['max_abs_gradient']:.3e} sha256 {res['sha256']}")
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fit the household-change model (identical bits on every platform).")
    ap.add_argument("--transitions", required=True); ap.add_argument("--snapshots", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int); ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--chunk", type=int, default=500)
    a = ap.parse_args()
    fit(a.transitions, a.snapshots, a.out, chunk=a.chunk, workers=a.workers, max_iter=a.max_iter, rounds=a.rounds,
        log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
