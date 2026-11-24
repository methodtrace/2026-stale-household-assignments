# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""exp and log from IEEE add, subtract, multiply and divide only, so results are the same bits on every platform.

The platform math library (libm) may round exp and log differently on different machines. These versions use a fixed
sequence of correctly rounded basic operations, which every IEEE 754 machine evaluates identically. They agree with a
correctly rounded exp and log to within about one unit in the last place.
"""
import numpy as np

_INV_LN2 = 1.4426950408889634
_LN2_HI = 6.93147180369123816490e-01   # high part with trailing zero bits: k * _LN2_HI is exact for |k| < 2^20
_LN2_LO = 1.90821492927058770002e-10
_SQRT1_2 = 0.7071067811865476
_EXP_C = [1.0 / f for f in (1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800,
                            87178291200)]


def exp(x):
    """exp(x) elementwise: x = k ln 2 + r, |r| <= ln 2 / 2, Taylor polynomial of degree 14 in r (Horner), times 2^k."""
    # beyond +-800 the result is inf or 0 either way; the clip keeps 2^k representable (no value inside changes)
    x = np.clip(np.asarray(x, dtype=np.float64), -800.0, 800.0)
    k = np.rint(x * _INV_LN2)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    p = np.full_like(r, _EXP_C[-1])
    for c in reversed(_EXP_C[:-1]):
        p = p * r + c
    return np.ldexp(p, k.astype(np.int64))


def log(x):
    """log(x) elementwise for x > 0: x = m 2^e, m in [sqrt(1/2), sqrt(2)), log m = 2 atanh(s), s = (m - 1) / (m + 1)."""
    x = np.asarray(x, dtype=np.float64)
    m, e = np.frexp(x)                       # m in [0.5, 1)
    lo = m < _SQRT1_2
    m = np.where(lo, m * 2.0, m); e = np.where(lo, e - 1, e).astype(np.float64)
    s = (m - 1.0) / (m + 1.0); z = s * s
    p = np.full_like(z, 1.0 / 25)
    for k in range(11, -1, -1):
        p = p * z + 1.0 / (2 * k + 1)
    return (e * _LN2_HI + (2.0 * s * p + e * _LN2_LO))
