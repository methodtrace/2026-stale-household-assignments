# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Student t quantiles in decimal arithmetic, rounded once to a double, so the value does not depend on the platform's math
library. Integer degrees of freedom only, through the closed-form t distribution function (even df: algebraic; odd df:
one arctangent, also computed in decimal)."""
from decimal import Decimal, localcontext

PREC = 60


def _atan(x: Decimal) -> Decimal:
    """arctan by argument halving (atan x = 2 atan(x / (1 + sqrt(1 + x^2)))) and the Taylor series."""
    halvings = 0
    while abs(x) > Decimal("0.1"):
        x = x / (1 + (1 + x * x).sqrt())
        halvings += 1
    s, term, n, x2 = Decimal(0), x, 1, x * x
    eps = Decimal(10) ** (-(PREC + 5))
    while abs(term) > eps:
        s += term / n
        term *= -x2
        n += 2
    return s * (2 ** halvings)


def _pi() -> Decimal:
    return 16 * _atan(Decimal(1) / 5) - 4 * _atan(Decimal(1) / 239)


def cdf(t: Decimal, df: int) -> Decimal:
    nu = Decimal(df)
    r = (nu + t * t).sqrt()
    s, c = t / r, nu.sqrt() / r  # sin and cos of theta = atan(t / sqrt(df))
    c2 = c * c
    if df % 2 == 0:
        acc, term = Decimal(1), Decimal(1)
        for j in range(1, df // 2):
            term *= c2 * (2 * j - 1) / (2 * j)
            acc += term
        return Decimal("0.5") + s * acc / 2
    theta = _atan(t / nu.sqrt())
    if df == 1:
        return Decimal("0.5") + theta / _pi()
    acc, term = Decimal(1), Decimal(1)
    for j in range(1, (df - 1) // 2):
        term *= c2 * (2 * j) / (2 * j + 1)
        acc += term
    return Decimal("0.5") + (theta + s * c * acc) / _pi()


def quantile(p: str, df: int) -> float:
    """t such that P(T <= t) = p, for 0.5 < p < 1, by bisection in decimal arithmetic; rounded once to a double."""
    with localcontext() as ctx:
        ctx.prec = PREC
        target = Decimal(p)
        lo, hi = Decimal(0), Decimal(1)
        while cdf(hi, df) < target:
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            if cdf(mid, df) < target:
                lo = mid
            else:
                hi = mid
        return float((lo + hi) / 2)
