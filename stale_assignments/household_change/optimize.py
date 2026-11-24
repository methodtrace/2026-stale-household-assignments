# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Limited-memory BFGS with a strong Wolfe line search, in plain Python and elementwise numpy.

The algorithm and its defaults follow the widely used PyTorch implementation (torch.optim.LBFGS, line_search_fn =
'strong_wolfe'), so the two make the same decisions. Every inner product is math.fsum of elementwise products, which is
correctly rounded, so the iterates do not depend on the platform's BLAS or summation order. Used by the household-change fit and the
listing refit of the transfer test.
"""
import math

import numpy as np


def dot(a, b):
    return math.fsum((a * b).tolist())


def _cubic(x1, f1, g1, x2, f2, g2, bounds=None):
    if bounds is not None:
        lo, hi = bounds
    else:
        lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
    if not all(math.isfinite(v) for v in (x1, f1, g1, x2, f2, g2)):
        return (lo + hi) / 2.0  # a trial point overflowed: bisect
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2s = d1 * d1 - g1 * g2
    if d2s >= 0:
        d2 = math.sqrt(d2s)
        if x1 <= x2:
            m = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            m = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(m, lo), hi)
    return (lo + hi) / 2.0


def _finite(fg):
    """A trial point whose loss or gradient overflows counts as an infinite loss, so the search backs off from it."""
    def h(x):
        f, g = fg(x)
        if not (math.isfinite(f) and np.isfinite(g).all()):
            return math.inf, np.zeros_like(g)
        return f, g
    return h


def _strong_wolfe(fg, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25):
    fg = _finite(fg)
    d_norm = float(np.max(np.abs(d)))
    f_new, g_new = fg(x + t * d)
    evals = 1
    gtd_new = dot(g_new, d)
    t_prev, f_prev, g_prev, gtd_prev = 0.0, f, g, gtd
    done = False
    it = 0
    while it < max_ls:
        if f_new > (f + c1 * t * gtd) or (it > 1 and f_new >= f_prev):
            br, bf, bg, bgtd = [t_prev, t], [f_prev, f_new], [g_prev, g_new], [gtd_prev, gtd_new]
            break
        if abs(gtd_new) <= -c2 * gtd:
            br, bf, bg, bgtd = [t], [f_new], [g_new], [gtd_new]
            done = True
            break
        if gtd_new >= 0:
            br, bf, bg, bgtd = [t_prev, t], [f_prev, f_new], [g_prev, g_new], [gtd_prev, gtd_new]
            break
        lo_s = t + 0.01 * (t - t_prev); hi_s = t * 10
        tmp = t
        t = _cubic(t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(lo_s, hi_s))
        t_prev, f_prev, g_prev, gtd_prev = tmp, f_new, g_new, gtd_new
        f_new, g_new = fg(x + t * d)
        evals += 1
        gtd_new = dot(g_new, d)
        it += 1
    if it == max_ls:
        br, bf, bg, bgtd = [0.0, t], [f, f_new], [g, g_new], [gtd, gtd_new]
    insuf = False
    lp, hp = (0, 1) if bf[0] <= bf[-1] else (1, 0)
    while not done and it < max_ls:
        if abs(br[1] - br[0]) * d_norm < tolerance_change:
            break
        t = _cubic(br[0], bf[0], bgtd[0], br[1], bf[1], bgtd[1])
        eps = 0.1 * (max(br) - min(br))
        if min(max(br) - t, t - min(br)) < eps:
            if insuf or t >= max(br) or t <= min(br):
                t = max(br) - eps if abs(t - max(br)) < abs(t - min(br)) else min(br) + eps
                insuf = False
            else:
                insuf = True
        else:
            insuf = False
        f_new, g_new = fg(x + t * d)
        evals += 1
        gtd_new = dot(g_new, d)
        it += 1
        if f_new > (f + c1 * t * gtd) or f_new >= bf[lp]:
            br[hp], bf[hp], bg[hp], bgtd[hp] = t, f_new, g_new, gtd_new
            lp, hp = (0, 1) if bf[0] <= bf[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (br[hp] - br[lp]) >= 0:
                br[hp], bf[hp], bg[hp], bgtd[hp] = br[lp], bf[lp], bg[lp], bgtd[lp]
            br[lp], bf[lp], bg[lp], bgtd[lp] = t, f_new, g_new, gtd_new
    return bf[lp], bg[lp], br[lp], evals


class LBFGS:
    """fg(x) -> (loss, gradient). step() runs one call of at most max_iter iterations, keeping its history for the next."""

    def __init__(self, fg, x, lr=1.0, max_iter=150, max_eval=None, tolerance_grad=1e-9, tolerance_change=1e-12,
                 history_size=50, log=None):
        self.fg, self.x = fg, np.array(x, dtype=np.float64)
        self.lr, self.max_iter = lr, max_iter
        self.max_eval = max_eval if max_eval is not None else max_iter * 5 // 4
        self.tg, self.tc, self.hist = tolerance_grad, tolerance_change, history_size
        self.n_iter = 0; self.func_evals = 0
        self.d = self.t = self.prev_g = self.prev_loss = None
        self.dirs, self.stps, self.ro, self.H = [], [], [], 1.0
        self.log = log

    def step(self):
        loss, g = self.fg(self.x)
        orig = loss
        evals = 1; self.func_evals += 1
        if float(np.max(np.abs(g))) <= self.tg:
            return orig
        n = 0
        d, t = self.d, self.t
        while n < self.max_iter:
            n += 1; self.n_iter += 1
            if self.n_iter == 1:
                d = -g; self.dirs, self.stps, self.ro, self.H = [], [], [], 1.0
            else:
                y = g - self.prev_g; s = d * t; ys = dot(y, s)
                if ys > 1e-10:
                    if len(self.dirs) == self.hist:
                        self.dirs.pop(0); self.stps.pop(0); self.ro.pop(0)
                    self.dirs.append(y); self.stps.append(s); self.ro.append(1.0 / ys)
                    self.H = ys / dot(y, y)
                al = [0.0] * len(self.dirs)
                q = -g
                for i in range(len(self.dirs) - 1, -1, -1):
                    al[i] = dot(self.stps[i], q) * self.ro[i]
                    q = q - al[i] * self.dirs[i]
                r = q * self.H
                for i in range(len(self.dirs)):
                    be = dot(self.dirs[i], r) * self.ro[i]
                    r = r + (al[i] - be) * self.stps[i]
                d = r
            self.prev_g = g.copy(); self.prev_loss = loss
            t = min(1.0, 1.0 / math.fsum(np.abs(g).tolist())) * self.lr if self.n_iter == 1 else self.lr
            gtd = dot(g, d)
            if gtd > -self.tc:
                break
            loss, g, t, ls = _strong_wolfe(self.fg, self.x, t, d, loss, g, gtd)
            self.x = self.x + t * d
            opt = float(np.max(np.abs(g))) <= self.tg
            evals += ls; self.func_evals += ls
            if self.log:
                self.log(self.n_iter, self.func_evals, loss)
            if n == self.max_iter or evals >= self.max_eval or opt:
                break
            if float(np.max(np.abs(d * t))) <= self.tc:
                break
            if abs(loss - self.prev_loss) < self.tc:
                break
        self.d, self.t = d, t
        return orig
