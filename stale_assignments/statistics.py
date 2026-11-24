# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline steps 6, 10 and 13: statistics with county-cluster intervals.

Replicates: rep 0 is the point estimate; reps 1 to R use Poisson(1) county weights drawn from a fingerprint of
(seed, county, rep); reps 10000 + county leave that county out (jackknife SE and the leave-one-county-out check). The 99%
interval takes offsets R * 0.005 - 1 and R * 0.995 - 1 of the sorted replicates, the 95% interval R * 0.025 - 1 and
R * 0.975 - 1.

Decision: supported if the 99% interval lies wholly on the predicted side of the threshold, contradicted if wholly on the
other side, else not supported. loco_fragile: the leave-one-county-out estimates fall on both sides of the threshold.

Arithmetic is limited to correctly rounded operations (sums by math.fsum, IEEE + - * /, math.sqrt), so every value is the
same on every platform. Null follows SQL: any null operand gives null; safe_div gives null on a zero denominator.
"""
import math
from collections import defaultdict

import polars as pl

from . import config
from . import parallel as par
from .bootstrap import summarize, weights
from .common import T0

# ---- expressions over named values, with SQL null semantics -------------------------------------------------------------


class E:
    def __add__(self, o): return Op("+", self, o)
    def __radd__(self, o): return Op("+", o, self)
    def __sub__(self, o): return Op("-", self, o)
    def __rsub__(self, o): return Op("-", o, self)
    def __mul__(self, o): return Op("*", self, o)
    def __rmul__(self, o): return Op("*", o, self)
    def __truediv__(self, o): return Op("/", self, o)
    def __rtruediv__(self, o): return Op("/", o, self)

    def keys(self):
        return set()


def ev(x, v):
    return x.eval(v) if isinstance(x, E) else x


def ks(x):
    return x.keys() if isinstance(x, E) else set()


class Key(E):
    def __init__(self, key): self.key = key
    def eval(self, v): return v.get(self.key)
    def keys(self): return {self.key}


class Op(E):
    def __init__(self, op, a, b): self.op, self.a, self.b = op, a, b

    def eval(self, v):
        a, b = ev(self.a, v), ev(self.b, v)
        if a is None or b is None:
            return None
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if b == 0:
            raise ZeroDivisionError("division by zero in a statistic (SQL '/' is not safe)")
        return a / b

    def keys(self): return ks(self.a) | ks(self.b)


class SafeDiv(E):
    def __init__(self, a, b): self.a, self.b = a, b

    def eval(self, v):
        a, b = ev(self.a, v), ev(self.b, v)
        return None if a is None or b is None or b == 0 else a / b

    def keys(self): return ks(self.a) | ks(self.b)


class Case(E):
    """CASE WHEN a <= c THEN x ... ELSE d END; a null comparison falls through."""
    def __init__(self, whens, default): self.whens, self.default = whens, default

    def eval(self, v):
        for a, c, x in self.whens:
            av = ev(a, v)
            if av is not None and av <= c:
                return ev(x, v)
        return ev(self.default, v)

    def keys(self):
        return set().union(*(ks(a) | ks(x) for a, _, x in self.whens)) | ks(self.default)


class AllBelow(E):
    """IF(GREATEST(a) < LEAST(b), 1, 0): GREATEST and LEAST are null if any input is null, and a null test gives 0."""
    def __init__(self, a, b): self.a, self.b = a, b

    def eval(self, v):
        a, b = [ev(x, v) for x in self.a], [ev(x, v) for x in self.b]
        if any(x is None for x in a + b):
            return 0.0
        return 1.0 if max(a) < min(b) else 0.0

    def keys(self): return set().union(*(ks(x) for x in self.a + self.b))


def mean(xs):
    """(x1 + ... + xn) / n, summed left to right as written."""
    s = xs[0]
    for x in xs[1:]:
        s = s + x
    return s / len(xs)


# ---- replicate weights and interval summaries ----------------------------------------------------------------------------

def _wsum(rows, w):
    """SUM(w.wt * c.x) over cell rows (county, x): null when every x is null."""
    xs = [w[c] * x for c, x in rows if x is not None]
    if not xs:
        return None
    if all(isinstance(x, int) for x in xs):
        return sum(xs)
    return math.fsum(xs)


# ---- step 6 ----------------------------------------------------------------------------------------------------------------

SUMS = ["n_all", "n", "n_m_all", "n_m", "f1c", "f1", "f1c_u", "n_u", "fm9", "fs9", "loss", "loss1", "loss_mover_together",
        "loss_mover_split", "loss1_mover_split", "loss1_mov", "n_mov", "n_mover_split", "n_alone"]
METRICS = {  # metric: function of the group's weighted sums (numerator, denominator)
    "mr": lambda s: (s["n_m"], s["n"]), "mr_all": lambda s: (s["n_m_all"], s["n_all"]), "f1c": lambda s: (s["f1c"], s["n"]),
    "f1": lambda s: (s["f1"], s["n"]), "f1c_u": lambda s: (s["f1c_u"], s["n_u"]), "lo": lambda s: (s["f1c"], s["n_all"]),
    "hi": lambda s: (_add(_add(s["f1c"], s["n_all"]), _neg(s["n"])), s["n_all"]), "floor": lambda s: (s["n_alone"], s["n"]),
    "split_share": lambda s: (s["loss_mover_split"], s["loss"]),
    "split_share_lo": lambda s: (s["loss_mover_split"], _add(_add(s["loss"], s["n_all"]), _neg(s["n"]))),
    "split_share_self": lambda s: (s["loss1_mover_split"], s["loss1"]),
    "f1c_mover_split": lambda s: (_add(s["n_mover_split"], _neg(s["loss_mover_split"])), s["n_mover_split"]),
    "mov_pop": lambda s: (s["n_mov"], s["n"]), "mov_loss": lambda s: (_add(s["loss_mover_together"], s["loss_mover_split"]), s["loss"]),
    "fm9": lambda s: (s["fm9"], s["n"]), "fs9": lambda s: (s["fs9"], s["n"])}
METRIC_SUMS = {"mr": ("n_m", "n"), "mr_all": ("n_m_all", "n_all"), "f1c": ("f1c", "n"), "f1": ("f1", "n"), "f1c_u": ("f1c_u", "n_u"),
               "lo": ("f1c", "n_all"), "hi": ("f1c", "n_all", "n"), "floor": ("n_alone", "n"),
               "split_share": ("loss_mover_split", "loss"), "split_share_lo": ("loss_mover_split", "loss", "n_all", "n"),
               "split_share_self": ("loss1_mover_split", "loss1"), "f1c_mover_split": ("n_mover_split", "loss_mover_split"),
               "mov_pop": ("n_mov", "n"), "mov_loss": ("loss_mover_together", "loss_mover_split", "loss"), "fm9": ("fm9", "n"), "fs9": ("fs9", "n")}
CW = ["mr", "f1c", "split_share"]


def _add(a, b):
    return None if a is None or b is None else a + b


def _neg(a):
    return None if a is None else -a


def _safe(nd):
    a, b = nd
    return None if a is None or b is None or b == 0 else a / b


_T0Y = T0.year  # the main build's year (config.py)
_BUILDS_Y = [d.year for d in sorted(set(config.AUX) | {T0})]  # builds of the widening by build: 2008 to 2016 in the main design


def K(metric, policy, k, t=_T0Y, lag=0, panel="all"):
    return Key(f"{metric}|{policy}|{lag}|{t}|{k}|{panel}")


def G(policy, k, t=_T0Y, lag=0, panel="all", score="f1c"):
    return K("mr", policy, k, t, lag, panel) - K(score, policy, k, t, lag, panel)


def halflife(policy):
    e = {k: (K("f1c", policy, k) - K("floor", policy, k)) / (1 - K("floor", policy, k)) for k in (2, 4, 6, 8)}
    pts = [(0, 1.0)] + [(k, e[k]) for k in (2, 4, 6, 8)]
    whens = [(e1, 0.5, k0 + (e0 - 0.5) / (e0 - e1) * (k1 - k0)) for (k0, e0), (k1, e1) in zip(pts, pts[1:])]
    return Case(whens, 99)


VARIANTS = [
    ("balanced panel", dict(mr="mr", f1c="f1c", split="split_share", p="bal")),
    ("county-weighted", dict(mr="cw_mr", f1c="cw_f1c", split="cw_split_share", p="all")),
    ("self-inclusive", dict(mr="mr", f1c="f1", split="split_share_self", p="all")),
    ("exits scored 0", dict(mr="mr_all", f1c="lo", split="split_share_lo", p="all")),
    ("exits scored 1", dict(mr="mr_all", f1c="hi", split="split_share", p="all"))]


def part_rows(part, v):
    k_ = lambda m, pol, k, t=_T0Y: K(v[m], pol, k, t, panel=v["p"])
    g = lambda pol, k, t=_T0Y: k_("mr", pol, k, t) - k_("f1c", pol, k, t)
    if part == "f1_margin":
        return [("F1 margin, never_rebuilt", (k_("f1c", "never_rebuilt", 2) - k_("f1c", "never_rebuilt", 8)) - 2 * (k_("mr", "never_rebuilt", 2) - k_("mr", "never_rebuilt", 8)), 0, ">")]
    if part == "widening_by_build":
        return [(f"widening by build G(8) - G(2), never_rebuilt, build {b}", g("never_rebuilt", 8, b) - g("never_rebuilt", 2, b), 0.10, ">") for b in _BUILDS_Y]
    if part == "split_share":
        return [(f"split-household share of loss, member_tracker, k = {k}", k_("split", "member_tracker", k), 0.50, ">") for k in (4, 8)]
    if part == "lagged_gap":
        return [(f"lagged-rebuild gap G({k})", g("lagged_rebuild", k), 0.05, ">") for k in (2, 4, 6, 8)]
    if part == "loss_cut":
        return [(f"loss cut vs never_rebuilt, k = {k}", 1 - (1 - k_("f1c", "lagged_rebuild", k)) / (1 - k_("f1c", "never_rebuilt", k)), 0.25, ">")
                for k in (4, 6, 8)]
    raise ValueError(part)


def stat_defs():
    """(name, expression, threshold or None, predicted side or None), in output order."""
    s = []
    for panel in ("all", "bal"):
        sfx = "" if panel == "all" else " | balanced panel"
        s.append((f"gap G(2), never_rebuilt{sfx}", G("never_rebuilt", 2, panel=panel), 0, ">"))
        s.append((f"widening G(8) - G(2), never_rebuilt{sfx}", G("never_rebuilt", 8, panel=panel) - G("never_rebuilt", 2, panel=panel), 0.10, ">"))
        s.append((f"F1 margin, never_rebuilt{sfx}",
                  (K("f1c", "never_rebuilt", 2, panel=panel) - K("f1c", "never_rebuilt", 8, panel=panel))
                  - 2 * (K("mr", "never_rebuilt", 2, panel=panel) - K("mr", "never_rebuilt", 8, panel=panel)), 0, ">"))
        for k in (4, 8):
            s.append((f"split-household share of loss, member_tracker, k = {k}{sfx}", K("split_share", "member_tracker", k, panel=panel), 0.50, ">"))
    s.append(("gap G(2), never_rebuilt | county-weighted", K("cw_mr", "never_rebuilt", 2) - K("cw_f1c", "never_rebuilt", 2), 0, ">"))
    s.append(("widening G(8) - G(2), never_rebuilt | county-weighted",
              (K("cw_mr", "never_rebuilt", 8) - K("cw_f1c", "never_rebuilt", 8)) - (K("cw_mr", "never_rebuilt", 2) - K("cw_f1c", "never_rebuilt", 2)), 0.10, ">"))
    s.append(("gap G(2), never_rebuilt | self-inclusive", G("never_rebuilt", 2, score="f1"), 0, ">"))
    s.append(("widening G(8) - G(2), never_rebuilt | self-inclusive", G("never_rebuilt", 8, score="f1") - G("never_rebuilt", 2, score="f1"), 0.10, ">"))
    s.append(("F1 margin, never_rebuilt | self-inclusive", (K("f1", "never_rebuilt", 2) - K("f1", "never_rebuilt", 8)) - 2 * (K("mr", "never_rebuilt", 2) - K("mr", "never_rebuilt", 8)), 0, ">"))
    by_build = []
    for b in _BUILDS_Y:
        x = G("never_rebuilt", 8, t=b) - G("never_rebuilt", 2, t=b)
        by_build.append(x)
        s.append((f"widening by build G(8) - G(2), never_rebuilt, build {b}", x, 0.10, ">"))
    s.append(("widening by build, pooled mean of the five builds", mean(by_build), None, None))
    for k in (4, 8):
        s.append((f"split-household share of loss, member_tracker, k = {k} | self-inclusive", K("split_share_self", "member_tracker", k), 0.50, ">"))
    for k in (2, 4, 6, 8):
        s.append((f"lagged-rebuild gap G({k})", G("lagged_rebuild", k), 0.05, ">"))
        s.append((f"lagged-rebuild gap G({k}) | self-inclusive", G("lagged_rebuild", k, score="f1"), 0.05, ">"))
    for k in (4, 6, 8):
        s.append((f"loss cut vs never_rebuilt, k = {k}", 1 - (1 - K("f1c", "lagged_rebuild", k)) / (1 - K("f1c", "never_rebuilt", k)), 0.25, ">"))
        s.append((f"loss cut vs never_rebuilt, k = {k} | self-inclusive", 1 - (1 - K("f1", "lagged_rebuild", k)) / (1 - K("f1", "never_rebuilt", k)), 0.25, ">"))
    for k in (2, 4, 6, 8):
        s.append((f"same-date rebuild F1, client-file lag 1, k = {k}", K("f1c", "same_date_rebuild", k, lag=1), 0.95, "<"))
    for b, lab in (("lo", "exits scored 0"), ("hi", "exits scored 1")):
        gb = lambda pol, k: K("mr_all", pol, k) - K(b, pol, k)
        s.append((f"gap G(2), never_rebuilt | {lab}", gb("never_rebuilt", 2), 0, ">"))
        s.append((f"widening G(8) - G(2), never_rebuilt | {lab}", gb("never_rebuilt", 8) - gb("never_rebuilt", 2), 0.10, ">"))
        s.append((f"F1 margin, never_rebuilt | {lab}", (K(b, "never_rebuilt", 2) - K(b, "never_rebuilt", 8)) - 2 * (K("mr_all", "never_rebuilt", 2) - K("mr_all", "never_rebuilt", 8)), 0, ">"))
        for k in (2, 4, 6, 8):
            s.append((f"lagged-rebuild gap G({k}) | {lab}", gb("lagged_rebuild", k), 0.05, ">"))
        for k in (4, 6, 8):
            s.append((f"loss cut vs never_rebuilt, k = {k} | {lab}", 1 - (1 - K(b, "lagged_rebuild", k)) / (1 - K(b, "never_rebuilt", k)), 0.25, ">"))
    for lab, v in VARIANTS:
        need = {"balanced panel": ("widening_by_build", "lagged_gap", "loss_cut"),
                "county-weighted": ("f1_margin", "widening_by_build", "split_share", "lagged_gap", "loss_cut"),
                "self-inclusive": ("widening_by_build",), "exits scored 0": ("widening_by_build", "split_share"),
                "exits scored 1": ("widening_by_build", "split_share")}[lab]
        s.extend((f"{n} | {lab}", e, t, d) for part in need for n, e, t, d in part_rows(part, v))
    for k in (2, 4, 6, 8):
        s.append((f"match rate among persons with truth, never_rebuilt, k = {k}", K("mr", "never_rebuilt", k), None, None))
        s.append((f"co-member F1, never_rebuilt, k = {k}", K("f1c", "never_rebuilt", k), None, None))
        s.append((f"co-member F1 exits scored 0, never_rebuilt, k = {k}", K("lo", "never_rebuilt", k), None, None))
        s.append((f"co-member F1 exits scored 1, never_rebuilt, k = {k}", K("hi", "never_rebuilt", k), None, None))
        s.append((f"co-member F1 against voter-key truth, never_rebuilt, k = {k}", K("f1c_u", "never_rebuilt", k), None, None))
        s.append((f"living-alone floor, k = {k}", K("floor", "never_rebuilt", k), None, None))
        s.append((f"mover share of persons with truth, never_rebuilt, k = {k}", K("mov_pop", "never_rebuilt", k), None, None))
        s.append((f"mover share of loss, never_rebuilt, k = {k}", K("mov_loss", "never_rebuilt", k), None, None))
        s.append((f"mover loss lift, never_rebuilt, k = {k}", K("mov_loss", "never_rebuilt", k) / K("mov_pop", "never_rebuilt", k), None, None))
        s.append((f"last_known minus never_rebuilt, co-member F1, k = {k}", K("f1c", "last_known", k) - K("f1c", "never_rebuilt", k), None, None))
        s.append((f"last_known minus never_rebuilt, self-inclusive F1, k = {k}", K("f1", "last_known", k) - K("f1", "never_rebuilt", k), None, None))
        for a, b in (("never_rebuilt", "everyone_alone"), ("last_known", "everyone_alone"), ("never_rebuilt", "last_known"),
                     ("member_tracker", "everyone_alone")):
            s.append((f"crossover r, {a} vs {b}, k = {k}",
                      SafeDiv(K("fs9", b, k) - K("fs9", a, k), K("fm9", a, k) - K("fm9", b, k)), None, None))
        s.append((f"match rate, never_rebuilt_name_check, k = {k}", K("mr", "never_rebuilt_name_check", k), None, None))
        s.append((f"co-member F1, never_rebuilt_name_check, k = {k}", K("f1c", "never_rebuilt_name_check", k), None, None))
        s.append((f"G({k}), never_rebuilt_name_check", G("never_rebuilt_name_check", k), None, None))
        s.append((f"mover share of loss, never_rebuilt_name_check, k = {k}", K("mov_loss", "never_rebuilt_name_check", k), None, None))
        s.append((f"match rate overall (exits included), never_rebuilt_name_check, k = {k}", K("mr_all", "never_rebuilt_name_check", k), None, None))
        s.append((f"match rate overall (exits included), never_rebuilt, k = {k}", K("mr_all", "never_rebuilt", k), None, None))
    for k in (2, 4, 6, 8):
        for pol in ("member_tracker", "address_held", "member_tracker_keep"):
            s.append((f"co-member F1, carrier {pol}, k = {k}", K("f1c", pol, k), None, None))
            s.append((f"match rate, carrier {pol}, k = {k}", K("mr", pol, k), None, None))
        for pol in ("member_tracker", "address_held", "member_tracker_keep", "last_known", "never_rebuilt", "everyone_alone"):
            s.append((f"co-member F1 of movers in splits, {pol}, k = {k}", K("f1c_mover_split", pol, k), None, None))
    s.append(("G(8) - G(2), never_rebuilt_name_check", G("never_rebuilt_name_check", 8) - G("never_rebuilt_name_check", 2), None, None))
    s.append(("half-life, never_rebuilt, co-member F1 (99 = not reached)", halflife("never_rebuilt"), None, None))
    s.append(("half-life, last_known, co-member F1 (99 = not reached)", halflife("last_known"), None, None))
    return s


def _float(x):
    return None if x is None else float(x)


def _decide(thr, side, lo99, hi99):
    if thr is None:
        return None
    if lo99 is None or hi99 is None:
        return "not supported"
    if (side == ">" and lo99 > thr) or (side == "<" and hi99 < thr):
        return "supported"
    if (side == ">" and hi99 < thr) or (side == "<" and lo99 > thr):
        return "contradicted"
    return "not supported"


def _fragile(thr, a, b):
    if thr is None or a is None or b is None:
        return None
    return (a > thr) != (b > thr)


def main_stats(cells: pl.DataFrame, seed: str, reps: int) -> pl.DataFrame:
    """Step 6 from the county cells."""
    defs = stat_defs()
    keys = set().union(*(e.keys() for _, e, _, _ in defs))
    need = defaultdict(set)  # (t, policy, lag, k, panel) -> metrics
    for key in keys:
        m, pol, lag, t, k, panel = key.split("|")
        need[(int(t), pol, int(lag), int(k), panel)].add(m)
    counties = sorted(set(cells["county0"].to_list()))
    W = weights(counties, seed, reps)
    rows = defaultdict(list)  # (group, sum column) -> [(county, value)] in cell order
    cw_rows = defaultdict(lambda: defaultdict(list))
    for r in cells.sort(["t", "policy", "lag", "k", "county0", "bal"]).iter_rows(named=True):
        base = (r["t"], r["policy"], r["lag"], r["k"])
        for panel in ("all", "bal"):
            if panel == "bal" and not r["bal"]:
                continue
            g = base + (panel,)
            if g not in need:
                continue
            for x in set().union(*(METRIC_SUMS[m] for m in need[g] if not m.startswith("cw_"))):
                rows[(g, x)].append((r["county0"], r[x]))
            if any(m.startswith("cw_") for m in need[g]):
                for x in SUMS:
                    cw_rows[base][(r["county0"], x)].append(r[x])
    # county-weighted: each county's own metric (sums over its cells), averaged with the replicate weights
    cw_vals = {}
    for base, d in cw_rows.items():
        per = {}
        for c in counties:
            s = {}
            for x in SUMS:
                xs = [v for v in d.get((c, x), []) if v is not None]
                s[x] = None if not xs else (sum(xs) if all(isinstance(v, int) for v in xs) else math.fsum(xs))
            if any(d.get((c, x)) for x in SUMS):
                per[c] = {m: _safe(METRICS[m](s)) for m in CW}
        cw_vals[base] = per
    out = {}
    for rep, w in W.items():
        v = {}
        for g, ms in need.items():
            sums = {}
            for m in ms:
                if m.startswith("cw_") and g[4] != "all":
                    val = None  # county-weighted metrics exist for the full panel only
                elif m.startswith("cw_"):
                    mm = m[3:]
                    per = cw_vals.get(g[:4], {})
                    num = [w[c] * x[mm] for c, x in per.items() if x[mm] is not None]
                    den = sum(w[c] for c, x in per.items() if x[mm] is not None)
                    val = None if not num or den == 0 else math.fsum(num) / den
                else:
                    for x in METRIC_SUMS[m]:
                        if x not in sums:
                            sums[x] = _wsum(rows.get((g, x), []), w)
                    val = _safe(METRICS[m](sums)) if rows.get((g, METRIC_SUMS[m][0])) is not None else None
                v[f"{m}|{g[1]}|{g[2]}|{g[0]}|{g[3]}|{g[4]}"] = _float(val)
        for i, (name, e, thr, side) in enumerate(defs):
            out.setdefault(i, {})[rep] = _float(e.eval(v))
    res = []
    for i, (name, e, thr, side) in enumerate(defs):
        sm = summarize(out[i], reps)
        res.append(dict(stat=name, ord=i, **{k: sm[k] for k in ("point", "lo99", "hi99", "lo95", "hi95", "jk_se", "loco_min", "loco_max")},
                        thr=_float(thr), side=side, n_null=sm["n_null"], decision=_decide(_float(thr), side, sm["lo99"], sm["hi99"]),
                        loco_fragile=_fragile(_float(thr), sm["loco_min"], sm["loco_max"])))
    return pl.DataFrame(res, schema={"stat": pl.String, "ord": pl.Int64, "point": pl.Float64, "lo99": pl.Float64, "hi99": pl.Float64,
                                     "lo95": pl.Float64, "hi95": pl.Float64, "jk_se": pl.Float64, "loco_min": pl.Float64,
                                     "loco_max": pl.Float64, "thr": pl.Float64, "side": pl.String, "n_null": pl.Int64,
                                     "decision": pl.String, "loco_fragile": pl.Boolean})


# ---- steps 10 and 13: every key of a (county0, key, num, den) cell table as a ratio, plus derived statistics -------------

def _ratio_reps(task):
    """Worker of ratio_stats: {name: {rep: value}} for one block of replicates."""
    by, derived, W = task
    out = defaultdict(dict)
    for rep, w in W:
        v = {}
        for key, rs in by.items():
            num = [w[c] * n for c, n, _ in rs if n is not None]
            den = [w[c] * d for c, _, d in rs if d is not None]
            sn, sd = (math.fsum(num) if num else None), (math.fsum(den) if den else None)
            v[key] = None if sn is None or sd is None or sd == 0 else sn / sd
            out[key][rep] = v[key]
        for name, e in derived:
            out[name][rep] = _float(e.eval(v))
    return out


def ratio_stats(cells: pl.DataFrame, derived, seed: str, reps: int) -> pl.DataFrame:
    counties = sorted(set(cells["county0"].to_list()))
    W = weights(counties, seed, reps)
    by = defaultdict(list)
    for c, key, num, den in cells.sort(["key", "county0"]).select("county0", "key", "num", "den").iter_rows():
        by[key].append((c, num, den))
    by = dict(by)
    vals = defaultdict(dict)
    for part in par.pmap(_ratio_reps, [(by, derived, blk) for blk in par.chunks(W.items(), par.workers())]):
        for name, d in part.items():
            vals[name].update(d)
    res = []
    for stat in sorted(vals):
        sm = summarize(vals[stat], reps)
        res.append(dict(stat=stat, point=sm["point"], lo99=sm["lo99"], hi99=sm["hi99"], lo95=sm["lo95"], hi95=sm["hi95"],
                        jk_se=sm["jk_se"], n_null=sm["n_null"]))
    return pl.DataFrame(res, schema={"stat": pl.String, "point": pl.Float64, "lo99": pl.Float64, "hi99": pl.Float64,
                                     "lo95": pl.Float64, "hi95": pl.Float64, "jk_se": pl.Float64, "n_null": pl.Int64})


BUILD_AGE_JAN = [2009, 2010] + list(range(2012, 2026))
BUILD_AGE_NOV = [2009, 2010] + list(range(2013, 2026))
PRES = (2012, 2016, 2020, 2024)
ATTRIBUTE_GROUPS = [("size", str(v)) for v in range(1, 6)] + [("tag", "SFH_LIKELY"), ("tag", "UNIT_IN_MULTIUNIT"), ("core", "true"), ("core", "false")]
TENURE_GROUPS = [("tenure", b) for b in ("<2", "2-4", "4-8", "8+")]


def desc_derived():
    q = Key
    s = []
    for m, groups in (("household_attribute", ATTRIBUTE_GROUPS), ("address_tenure", TENURE_GROUPS)):
        for dim, val in groups:
            for k in (2, 4, 6, 8):
                s.append((f"G({k}), {dim} = {val}", q(f"{m}|{dim}|{val}|mr|{k}") - q(f"{m}|{dim}|{val}|f1c|{k}")))
    for k in (2, 4, 6, 8):
        s.append((f"loss over household change, k = {k}", q(f"household_change|loss|{k}") / q(f"household_change|chg|{k}")))
    for a in range(1, 18):
        s.append((f"pooled November build-age G, age {a}", q(f"build_age|age|{a}|mr") - q(f"build_age|age|{a}|f1c")))
    jp = [q(f"build_age|jan|{y}") for y in BUILD_AGE_JAN if y in PRES]
    jo = [q(f"build_age|jan|{y}") for y in BUILD_AGE_JAN if y not in PRES]
    s.append(("phase gap, January 1 to November: mean of other years minus mean of presidential years", mean(jo) - mean(jp)))
    s.append(("phase rank: all four presidential January windows below every other (1 = yes)", AllBelow(jp, jo)))
    npres = [q(f"build_age|nov|{y}") for y in BUILD_AGE_NOV if y in PRES]
    noth = [q(f"build_age|nov|{y}") for y in BUILD_AGE_NOV if y not in PRES]
    s.append(("phase gap, November to November: mean of other years minus mean of presidential years", mean(noth) - mean(npres)))
    return s


MATCHER_THRESHOLDS = ["none", "-10", "-5", "0", "5", "10", "15", "20", "25"]


def validity_derived():
    q = Key
    s = []
    for th in MATCHER_THRESHOLDS:
        p = f"probabilistic_matcher_w{th}"
        lab = "no threshold" if th == "none" else f"W >= {th}"
        g = lambda k, p=p: q(f"match_validity|{p}|mr|{k}") - q(f"match_validity|{p}|f1c|{k}")
        for k in (2, 4, 6, 8):
            s.append((f"probabilistic matcher G({k}), {lab}", g(k)))
        s.append((f"probabilistic matcher G(8) - G(2), {lab}", g(8) - g(2)))
        s.append((f"probabilistic matcher F1 margin, {lab}",
                  (q(f"match_validity|{p}|f1c|2") - q(f"match_validity|{p}|f1c|8")) - 2 * (q(f"match_validity|{p}|mr|2") - q(f"match_validity|{p}|mr|8"))))
    for k in (2, 4, 6, 8):
        s.append((f"support ceiling loss, co-residents absent from the 2016 graph, k = {k}", 1 - q(f"support_ceiling|ceil_any|{k}")))
        s.append((f"support ceiling loss, in the 2016 graph at neither address, k = {k}",
                  q(f"support_ceiling|ceil_any|{k}") - q(f"support_ceiling|ceil_reach|{k}")))
        s.append((f"support ceiling loss, choice among reachable candidates, deterministic, k = {k}",
                  q(f"support_ceiling|ceil_reach|{k}") - q(f"match_validity|never_rebuilt|f1c|{k}")))
        s.append((f"support ceiling loss, choice among reachable candidates, probabilistic matcher W >= 0, k = {k}",
                  q(f"support_ceiling|ceil_reach|{k}") - q(f"match_validity|probabilistic_matcher_w0|f1c|{k}")))
        s.append((f"support ceiling total loss, deterministic, k = {k}", 1 - q(f"match_validity|never_rebuilt|f1c|{k}")))
    return s
