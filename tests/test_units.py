# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Known-answer and property tests for the parts that must agree with the warehouse functions the analysis was specified
with, and for the exact arithmetic. Run from the repository root: python -m unittest discover -s tests"""
import datetime as dt
import math
import os
import random
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl  # noqa: E402

from stale_assignments import bootstrap, probabilistic_matcher, record_functions, score, snapshots  # noqa: E402

# reference values computed by the warehouse (BigQuery FARM_FINGERPRINT and SOUNDEX)
SOUNDEX = {"TYMCZAK": "T520", "HONEYMAN": "H500", "BHAWB": "B000", "ROBERT": "R163", "RUPERT": "R163", "ASHCRAFT": "A261",
           "PFISTER": "P236", "TUTTLE": "T400", "LEE": "L000", "O'BRIEN": "O165", "  smith": "s530", "123": "", "": "",
           "MÜLLER": "M460", "van der Berg": "v536", "JACKSON": "J500"}
FP = {"TYMCZAK": 4091093839204682910, "HONEYMAN": 2994481352921252988, "BHAWB": 2966896190519734540,
      "ROBERT": -4650792308109301990, "RUPERT": -4424835594784345172, "ASHCRAFT": -1632174157414038185,
      "PFISTER": -6805599336888411207, "TUTTLE": 1913487044441010363, "LEE": -3257462043470508325,
      "O'BRIEN": -3099838876544056097, "  smith": 3741734093499533175, "123": -3222588021317909685,
      "": -7286425919675154353, "MÜLLER": -2922060130175513788, "van der Berg": 2563796720209703404,
      "JACKSON": -9006328901320912031, "2|ad|0|0.20911025169757486|0.9968422022706132": 6558231184543150476}


class Warehouse(unittest.TestCase):
    def test_fingerprint(self):
        for s, v in FP.items():
            self.assertEqual(record_functions.fingerprint(s), v, s)

    def test_soundex(self):
        for s, v in SOUNDEX.items():
            self.assertEqual(record_functions.soundex(s), v, s)

    def test_months(self):
        self.assertEqual(record_functions.months_between(dt.date(2018, 11, 6), dt.date(2016, 11, 8)), 24)  # DATE_DIFF(a, b, MONTH)
        self.assertEqual(record_functions.years_between(dt.date(2017, 1, 1), dt.date(2016, 12, 31)), 1)


class Tokenizer(unittest.TestCase):
    def test_quotes(self):
        f = snapshots._split_quoted
        self.assertEqual(f('a\t"b"\tc'), ["a", "b", "c"])
        self.assertEqual(f('a\t"b"   \tc'), ["a", "b", "c"])          # padding after the closing quote
        self.assertEqual(f('a\t"b\tx"\tc'), ["a", "b\tx", "c"])        # tab inside quotes
        self.assertEqual(f('a\t"b""q"\tc'), ["a", 'b"q', "c"])         # doubled quote
        self.assertEqual(f('a\tb"q\tc'), ["a", 'b"q', "c"])            # quote not at field start is literal
        self.assertIsNone(f('a\t"b\tc'))                               # unclosed
        self.assertIsNone(f('a\t"b" x\tc'))                            # text after the closing quote


class Log2(unittest.TestCase):
    def test_exact_powers(self):
        for e in range(-1074, 1024, 7):
            self.assertEqual(probabilistic_matcher.log2(math.ldexp(1.0, e)), float(e))

    def test_close_to_libm(self):
        rnd = random.Random(7)
        for _ in range(2000):
            x = math.ldexp(rnd.random() + 0.5, rnd.randint(-60, 60))
            a, b = probabilistic_matcher.log2(x), math.log2(x)
            self.assertLessEqual(abs(a - b), 2 * math.ulp(b) if b else 0.0)

    def test_params(self):
        p = probabilistic_matcher.load_params()
        self.assertEqual(p.height, 64)


class Exact(unittest.TestCase):
    def test_rsum(self):
        df = pl.DataFrame({"g": [1, 1, 1, 2], "n": [1, 1, 2, None], "d": [3, 3, 3, 5]})
        r = score.rsum(df, ["g"], "n", "d", "v").sort("g")
        self.assertEqual(r["v"].to_list(), [float(Fraction(4, 3))])

    def test_f1c(self):
        df = pl.DataFrame({"tc": [0, 0, 2, 3, None], "ac": [0, 2, 0, 1, 1], "ic": [0, 0, 0, 1, 0]})
        n, d = score._f1c(pl.col("tc"), pl.col("ac"), pl.col("ic"))
        r = df.select(n.alias("n"), d.alias("d"))
        self.assertEqual(r.rows(), [(1, 1), (0, 1), (0, 1), (2, 4), (0, 1)])


class Bootstrap(unittest.TestCase):
    def test_weights_deterministic(self):
        w = bootstrap.weights([1, 9, 11], "county-weights", 50)
        self.assertEqual(w, bootstrap.weights([11, 1, 9], "county-weights", 50))
        self.assertTrue(all(0 <= v <= 6 for r in w.values() for v in r.values()))
        self.assertEqual(w[10009], {1: 1, 9: 0, 11: 1})

    def test_summary(self):
        vals = {0: 0.5, **{b: b / 2000 for b in range(1, 2001)}, 10001: 1.0, 10002: 3.0}
        s = bootstrap.summarize(vals, 2000)
        self.assertEqual((s["lo99"], s["hi99"], s["lo95"], s["hi95"]), (10 / 2000, 1990 / 2000, 50 / 2000, 1950 / 2000))
        self.assertEqual(s["jk_se"], 1.0)


if __name__ == "__main__":
    unittest.main()
