# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The household-change model (paper section 8.2): its states, rates and forward step.

Latent continuous-time chain per anchor, 16 states: the 8 household states (0 IB intact at the build unit, 1 IO intact
elsewhere, 2 XaB / 3 XaO broken recently, 4 XbB / 5 XbO broken long, 6 RB / 7 RO re-formed; B = build unit, O =
elsewhere) while listed (0-7) and the same 8 while unlisted (8-15). Rates per year at current ages (members age with time
since build u), duration since build and calendar year, piecewise linear in age on the knots KN:
  brk = exp(dl[dur] + cy[year]) (sum_j exp(fL(a_j) + gL(a_i) + hL[m] + bL[bt] + gp[gap_j]) + exp(fG(a_i) + hG[m] + bG[bt]))
  dep = exp(dlD[dur] + cy[year] + fD(a_i) + hD[m] + bD[bt]);  im = exp(dlM[dur] + cyM[year] + fM(a_i) + hM[m] + bM[bt])
  xm = exp(cyM[year] + fX(a_i) + hX[m] + bX[bt]);  re-formed break rate brk x exp(kR[size group]); return rates exp(rA[m]),
  exp(rB[m]); ageing of a break exp(phi); return to the build unit exp(w0);
  removal from the list exp(cU[year] + dU[dur] + fU(a_i) + bU[bt]) x exp(uO) if moved x exp(uX) if broken;
  re-entry exp(e0) x exp(eO) if moved.

The chain is an edge list: each edge moves mass from state i to state j at one of these rates. One grid step of h = 1/24
year is s' = s + h F(s) + h^2 / 2 F(F(s)), F the flow, which is linear in s and in each rate. The same step is used to
fit the model and to predict from it. Derived among listed anchors: kappa_P = P(EB or EO), sigma = P(EB or XB),
kappa_S = P(EB) / sigma, mu = P(EO) / (1 - sigma).

Only IEEE add, subtract, multiply, divide and the exp of exact_math are used, so a prediction is the same bits on every
platform.
"""
import datetime as dt
import hashlib
import json

import numpy as np

from ..exact_math import exp

KN = np.array([16., 22, 26, 30, 35, 45, 55, 65, 75, 85, 100])         # age knots
MID = {"A": 19.5, "B": 23.5, "C": 27.5, "D": 32.0, "E": 39.5, "F": 49.5, "G": 59.5, "H": 69.5, "I": 80.0, "Z": 45.0}
DUR = np.array([0, 0.2, 1, 2, 3, 4, 6, 8, 99.])                       # duration-since-build bins, years
H = 1 / 24.0                                                          # grid step, years
UMAX = 10.25
GRID = np.arange(0, UMAX + 1e-9, H)
MIDG = GRID[:-1] + H / 2
CY0, CY1 = 2008, 2026                                                 # calendar-year effects
GAPB = np.array([0, 5, 15, 30, 999.])                                 # anchor to co-member age gap bins

# ---- parameters ---------------------------------------------------------------------------------------------------------
K = len(KN); NY = CY1 - CY0 + 1; ND = len(DUR) - 1
SPEC = [("fL", K, -2.0), ("gL", K, 0.0), ("fG", K, -2.5), ("fD", K, -2.5), ("fM", K, -2.5), ("fX", K, -2.0),
        ("hL", 5, 0.0), ("hG", 5, 0.0), ("hD", 5, 0.0), ("hM", 5, 0.0), ("hX", 5, 0.0),
        ("dl", ND, 0.0), ("dlD", ND, 0.0), ("dlM", ND, 0.0), ("cy", NY, 0.0), ("cyM", NY, 0.0), ("gp", 4, 0.0),
        ("bL", 2, 0.0), ("bG", 2, 0.0), ("bD", 2, 0.0), ("bM", 2, 0.0), ("bX", 2, 0.0), ("kR", 3, 0.0),
        ("rA", 5, -0.5), ("rB", 5, -2.0), ("phi", 0, 0.0), ("w0", 0, -4.0),
        ("cU", NY, -3.0), ("dU", ND, 0.0), ("fU", K, 0.0), ("bU", 2, 0.0),
        ("uO", 0, 1.0), ("uX", 0, 0.0), ("e0", 0, -1.5), ("eO", 0, 0.0)]          # (name, size, cold start); size 0: scalar
_REF_YEAR = 2017 - CY0
FIXED = {"hL": [0, 1], "hG": [0], "hD": [0], "hM": [0], "hX": [0], "gL": [4], "dl": [0], "dlD": [0], "dlM": [0],
         "cy": [_REF_YEAR], "cyM": [_REF_YEAR], "gp": [0], "bL": [0], "bG": [0], "bD": [0], "bM": [0], "bX": [0],
         "dU": [0], "bU": [0], "fU": [4]}                                     # reference levels, held at zero


def _layout():
    off, out = 0, {}
    for nm, n, _ in SPEC:
        out[nm] = (off, n); off += max(n, 1)
    return out, off


LAYOUT, NPAR = _layout()


def cold_start():
    """The documented cold start of the fit."""
    x = np.zeros(NPAR)
    for nm, n, v in SPEC:
        o, _ = LAYOUT[nm]; x[o:o + max(n, 1)] = v
    return x


def fixed_mask():
    f = np.zeros(NPAR, dtype=bool)
    for nm, ix in FIXED.items():
        o, _ = LAYOUT[nm]
        for i in ix:
            f[o + i] = True
    return f


def unpack(x):
    """Parameter vector to {name: float or array}."""
    return {nm: (float(x[o]) if n == 0 else np.array(x[o:o + n])) for nm, (o, n) in LAYOUT.items()}


def pack(P):
    x = np.zeros(NPAR)
    for nm, (o, n) in LAYOUT.items():
        x[o:o + max(n, 1)] = P[nm]
    return x


def coefficient_sha256(x):
    """SHA-256 of the coefficient vector as little-endian float64 bytes: the fingerprint a fit file records."""
    return hashlib.sha256(np.asarray(x, dtype=np.float64).astype("<f8").tobytes()).hexdigest()


def load_fit(path, expect_sha256=None):
    """Coefficients of a fit file. The fingerprint recorded in the file is checked against the coefficients, and against
    expect_sha256 when given."""
    fit = json.load(open(path, encoding="utf-8"))
    P = unpack(pack({k: (np.asarray(v, dtype=np.float64) if np.ndim(v) else float(v)) for k, v in fit["params"].items()}))
    h = coefficient_sha256(pack(P))
    assert "sha256" not in fit or fit["sha256"] == h, f"{path}: coefficients do not match their recorded fingerprint"
    assert expect_sha256 is None or h == expect_sha256, f"{path}: not the expected fit ({h})"
    return P


# ---- composition cells --------------------------------------------------------------------------------------------------

def interp_idx(ages):
    """Knot interval and weight of each age for the piecewise-linear age profiles (constant beyond the end knots)."""
    a = np.clip(ages, KN[0], KN[-1] - 1e-9)
    i = np.searchsorted(KN, a, side="right") - 1
    return i, (a - KN[i]) / (KN[i + 1] - KN[i])


def pl(theta, i, fr):
    theta = np.asarray(theta)
    return theta[i] * (1 - fr) + theta[i + 1] * fr


class Cells:
    """Composition cells 'a:bcd' (anchor age band, co-member age bands) with building type at build date t."""

    def __init__(self, cells, bts, t):
        self.n = len(cells)
        ai = np.array([MID[c.split(":")[0]] for c in cells])
        mem = [c.split(":")[1] for c in cells]
        self.m = np.array([len(x) for x in mem])
        am = np.full((self.n, 4), 45.0); mask = np.zeros((self.n, 4))
        for r, x in enumerate(mem):
            for j, ch in enumerate(x[:4]):
                am[r, j] = MID[ch]; mask[r, j] = 1
        self.mask = mask
        self.gapb = np.clip(np.searchsorted(GAPB, np.abs(am - ai[:, None]), side="right") - 1, 0, 3)
        g = MIDG[None, :]
        self.ia, self.fa = interp_idx(ai[:, None] + g)
        self.im, self.fm = interp_idx(am[:, :, None] + g[None])
        self.mi = np.minimum(self.m, 4); self.g = np.minimum(self.m, 2)
        self.bt = (np.asarray(bts) == "M").astype(int)
        self.dbin = np.searchsorted(DUR, MIDG, side="right") - 1
        d = dt.date.fromisoformat(str(t))
        ty = d.year + (d.timetuple().tm_yday - 1) / 365.25
        self.cyi = np.clip(np.floor(ty + MIDG).astype(int), CY0, CY1) - CY0

    def subset(self, ix):
        """The cells at positions ix (same build date)."""
        import copy
        c = copy.copy(self)
        for a in ("mask", "gapb", "ia", "fa", "im", "fm", "mi", "g", "bt"):
            setattr(c, a, getattr(self, a)[ix])
        c.n = len(ix)
        return c


# ---- rates ----------------------------------------------------------------------------------------------------------------

def rates(P, C):
    """Every model rate of cells C on the grid: (n, steps) arrays, (n,) arrays for the size-group rates, and floats."""
    R = {}
    R["sc"] = exp(P["dl"][C.dbin][None, :] + P["cy"][C.cyi][None, :])
    gl = pl(P["gL"], C.ia, C.fa)
    R["lo"] = exp(pl(P["fL"], C.im, C.fm) + gl[:, None, :] + (P["hL"][C.mi] + P["bL"][C.bt])[:, None, None]
                  + P["gp"][C.gapb][:, :, None]) * C.mask[:, :, None]
    R["los"] = ((R["lo"][:, 0] + R["lo"][:, 1]) + R["lo"][:, 2]) + R["lo"][:, 3]
    R["gain"] = exp(pl(P["fG"], C.ia, C.fa) + (P["hG"][C.mi] + P["bG"][C.bt])[:, None])
    R["brk"] = R["sc"] * (R["los"] + R["gain"])
    R["dep"] = exp(P["dlD"][C.dbin][None, :] + P["cy"][C.cyi][None, :] + pl(P["fD"], C.ia, C.fa)
                   + (P["hD"][C.mi] + P["bD"][C.bt])[:, None])
    R["im"] = exp(P["dlM"][C.dbin][None, :] + P["cyM"][C.cyi][None, :] + pl(P["fM"], C.ia, C.fa)
                  + (P["hM"][C.mi] + P["bM"][C.bt])[:, None])
    R["xm"] = exp(P["cyM"][C.cyi][None, :] + pl(P["fX"], C.ia, C.fa) + (P["hX"][C.mi] + P["bX"][C.bt])[:, None])
    R["kr"] = exp(P["kR"][C.g])
    R["rA"] = exp(P["rA"][C.mi]); R["rB"] = exp(P["rB"][C.mi])
    R["rm"] = exp(P["cU"][C.cyi][None, :] + P["dU"][C.dbin][None, :] + pl(P["fU"], C.ia, C.fa) + P["bU"][C.bt][:, None])
    for a, b in (("ab", "phi"), ("ob", "w0"), ("uO", "uO"), ("uX", "uX"), ("re", "e0"), ("eO", "eO")):
        R[a] = float(exp(np.array(P[b])))
    return R


# ---- the chain ----------------------------------------------------------------------------------------------------------
# household edges (rate, from, to); 'br' = brk x kr, the break rate of a re-formed set
_HH = [("brk", 0, 2), ("dep", 0, 3), ("im", 0, 1), ("ob", 1, 0), ("brk", 1, 3), ("dep", 1, 3),
       ("xm", 2, 3), ("ab", 2, 4), ("rA", 2, 6), ("ob", 3, 2), ("ab", 3, 5), ("rA", 3, 7),
       ("xm", 4, 5), ("rB", 4, 6), ("ob", 5, 4), ("rB", 5, 7),
       ("br", 6, 2), ("dep", 6, 3), ("im", 6, 7), ("ob", 7, 6), ("br", 7, 3), ("dep", 7, 3)]
MOVED = [0, 1, 0, 1, 0, 1, 0, 1]          # household state is away from the build unit
BROKEN = [0, 0, 1, 1, 1, 1, 0, 0]         # household state is broken
EDGES = ([(r, i, j) for r, i, j in _HH] + [(r, i + 8, j + 8) for r, i, j in _HH]
         + [(f"rm{i}", i, i + 8) for i in range(8)] + [(f"re{i}", i + 8, i) for i in range(8)])
REMOVAL_EDGES = range(44, 52)
REENTRY_EDGES = range(52, 60)
OBSOF = [0, 1, 2, 3, 2, 3, 0, 1] + [4] * 8           # observed category of each latent state: EB EO XB XO U
OBS5 = ["EB", "EO", "XB", "XO", "U"]


def removal_factor(R, i):
    """Removal rate multiplier of household state i: exp(uO) if moved, times exp(uX) if broken."""
    f = 1.0
    if MOVED[i]: f = f * R["uO"]
    if BROKEN[i]: f = f * R["uX"]
    return f


def edge_rates(R, k):
    """Rate of every edge at grid step k: arrays (n,) or floats, in EDGES order."""
    b = R["brk"][:, k]
    base = dict(brk=b, dep=R["dep"][:, k], im=R["im"][:, k], xm=R["xm"][:, k], br=b * R["kr"], rA=R["rA"], rB=R["rB"],
                ab=R["ab"], ob=R["ob"])
    rm = R["rm"][:, k]
    for i in range(8):
        base[f"rm{i}"] = rm * removal_factor(R, i)
        base[f"re{i}"] = R["re"] * (R["eO"] if MOVED[i] else 1.0)
    return [base[r] for r, _, _ in EDGES]


def flow(s, er):
    """F(s): the net flow into each state, edge by edge in EDGES order."""
    a = [None] * 16
    for (_, i, j), r in zip(EDGES, er):
        f = s[i] * r
        a[i] = -f if a[i] is None else a[i] - f
        a[j] = f if a[j] is None else a[j] + f
    return a


def flow_t(lam, er):
    """Adjoint of flow: m_i = sum over edges from i of r (lam_j - lam_i)."""
    m = [None] * 16
    for (_, i, j), r in zip(EDGES, er):
        t = (lam[j] - lam[i]) * r
        m[i] = t if m[i] is None else m[i] + t
    return [x if x is not None else np.zeros_like(lam[0]) for x in m]


def step(s, er):
    """One grid step s + h F(s) + h^2 / 2 F(F(s)); returns the new state and F(s)."""
    a = flow(s, er); a2 = flow(a, er)
    return [s[i] + H * a[i] + 0.5 * H * H * a2[i] for i in range(16)], a


def build_state(n):
    """Every anchor intact at the build unit and listed."""
    s = [np.zeros(n) for _ in range(16)]
    s[0] = np.ones(n)
    return s


def propagate(R, n, n_steps, keep=False):
    """The marginal after n_steps grid steps from the build (and every intermediate state when keep)."""
    s = build_state(n)
    hist = [s] if keep else None
    for k in range(n_steps):
        s = step(s, edge_rates(R, k))[0]
        if keep:
            hist.append(s)
    return s, hist


def marginals(P, C, gi):
    """Latent marginals at grid indexes gi (gi[0] = 0 is the build), as a list of 16 arrays per snapshot."""
    R = rates(P, C)
    s = build_state(C.n)
    out = [list(s)]
    k = 0
    for j in range(1, len(gi)):
        while k < gi[j]:
            s = step(s, edge_rates(R, k))[0]; k += 1
        out.append(list(s))
    return out


def derived(s):
    """Among listed: kP, sig, kS, mu, and the listed share, from the 16 marginals of one snapshot."""
    L = s[:8]
    listed = ((((((L[0] + L[1]) + L[2]) + L[3]) + L[4]) + L[5]) + L[6]) + L[7]
    den = np.maximum(listed, 1e-12)
    o = [(L[0] + L[6]) / den, (L[1] + L[7]) / den, (L[2] + L[4]) / den, (L[3] + L[5]) / den]
    E = o[0] + o[1]; B = o[0] + o[2]
    return dict(kP=E, sig=B, kS=o[0] / np.maximum(B, 1e-12), mu=o[1] / np.maximum(1 - B, 1e-12), listed=listed)
