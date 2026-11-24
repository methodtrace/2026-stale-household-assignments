# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""County-cluster bootstrap with Poisson(1) county weights, and the delete-one-county jackknife.

Replicate b gives county c the weight k where u = (|fingerprint(seed|county|c|b)| mod 10^6) / 10^6 falls in the k-th
interval of the Poisson(1) distribution function (truncated at 6). The weights depend only on the seed string, the
county and the replicate number, so any subset of statistics can be recomputed alone and in any order. Replicates
10000 + d drop county d (jackknife).
"""
import math

from .record_functions import fingerprint

POISSON = (0.367879, 0.735759, 0.919699, 0.981012, 0.996340, 0.999406)


def weights(counties, seed: str, reps: int) -> dict:
    """rep -> {county: weight}."""
    cs = sorted(counties)
    w = {0: {c: 1 for c in cs}}
    for b in range(1, reps + 1):
        row = {}
        for c in cs:
            u = (abs(fingerprint(f"{seed}|county|{c}|{b}")) % 1000000) / 1000000
            row[c] = next((i for i, p in enumerate(POISSON) if u < p), 6)
        w[b] = row
    for d in cs:
        w[10000 + d] = {c: 0 if c == d else 1 for c in cs}
    return w


def offsets(reps):
    return int(reps * 0.005) - 1, int(reps * 0.995) - 1, int(reps * 0.025) - 1, int(reps * 0.975) - 1


def _at(xs, i):
    return xs[i] if 0 <= i < len(xs) else None


def summarize(vals: dict, reps: int) -> dict:
    """vals: rep -> value. Point, percentile intervals, jackknife SE, leave-one-county-out range, nulls among reps 1 to R."""
    bs = sorted(v for r, v in vals.items() if 1 <= r <= reps and v is not None)
    jk = [v for r, v in sorted(vals.items()) if r >= 10000 and v is not None]
    lo99, hi99, lo95, hi95 = offsets(reps)
    n = len(jk)
    if n:
        avg = math.fsum(jk) / n
        jk_se = math.sqrt((n - 1) / n * math.fsum((j - avg) * (j - avg) for j in jk))
    else:
        jk_se = None
    return dict(point=vals.get(0), lo99=_at(bs, lo99), hi99=_at(bs, hi99), lo95=_at(bs, lo95), hi95=_at(bs, hi95), jk_se=jk_se,
                loco_min=min(jk) if jk else None, loco_max=max(jk) if jk else None,
                n_null=sum(1 for r, v in vals.items() if 1 <= r <= reps and v is None))


