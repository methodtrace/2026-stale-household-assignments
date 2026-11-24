# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Property tests for the release copies of the fitted tables (release_files.py) on a small synthetic table set.
Run from the repository root: python -m unittest discover -s tests"""
import csv
import os
import random
import sys
import tempfile
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stale_assignments import release_files as RF, table_scores as TS  # noqa: E402

BUILDS = ["2008-11-04", "2010-11-02", "2012-11-06", "2014-11-04"]


def synthetic_cells(seed=7):
    """Development-style cells (t_dt, k, size6, bt, ab, ny, mb, gb, n, ne) with many small fine cells and one coarse
    cell under 10 persons (k 8, 2010, size 5, multi, a75)."""
    rnd = random.Random(seed)
    rows = []
    for t in BUILDS:
        for k in (2, 8):
            for size6 in ("1", "5"):
                for ab in RF.BANDS:
                    fines = [("n0", "none", "none")] if size6 == "1" else [
                        (f"y{y}", f"m{m}", f"g{g}") for y in (0, 1) for m in (18, 45) for g in (0, 10)]
                    for f in fines:
                        n = rnd.choice([1, 2, 3, 5, 8, 12, 30, 200])
                        if (k, t, size6, ab) == (8, "2010-11-02", "5", "a75"):
                            n = 1
                        rows.append({"t_dt": t, "k": k, "size6": size6, "bt": "multi", "ab": ab, "ny": f[0], "mb": f[1],
                                     "gb": f[2], "n": n, "ne": rnd.randint(0, n)})
    return rows


class ReleaseTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cells = os.path.join(cls.dir, "cells.csv")
        rows = synthetic_cells()
        with open(cells, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        TS.fit_tables(cells, cls.dir)
        cls.out = RF.release_tables(cls.dir)
        cls.raw = {name: RF._read(os.path.join(cls.dir, f"{name}.csv")) for name in TS.TABLES}
        cls.pk = RF._pooled(cls.raw["coarse_table"])

    def table(self, path):
        header, rows = self.out[path]
        return [dict(zip(header, r)) for r in rows]

    def test_no_cell_under_ten(self):
        for path, (header, rows) in self.out.items():
            self.assertTrue(rows, path)
            self.assertGreaterEqual(min(r[header.index("n")] for r in rows), RF.MIN_PERSONS, path)

    def test_small_coarse_cell_merges_with_adjacent_band(self):
        c = self.table("fitted_tables/coarse_table_by_build.csv")
        labels = {r["ab"] for r in c if (r["k"], r["build"], r["size6"]) == ("8", "2010-11-02", "5")}
        self.assertIn("a65+a75", labels)
        self.assertNotIn("a75", labels)
        p = self.table("fitted_tables/coarse_table_by_parity.csv")
        self.assertIn("a65+a75", {r["ab"] for r in p if (r["k"], r["parity"], r["size6"]) == ("8", "mid", "5")})
        fine = self.table("fitted_tables/comember_ages_table_by_build.csv")
        self.assertIn("a65+a75", {r["ab"] for r in fine if (r["k"], r["build"], r["size6"]) == ("8", "2010-11-02", "5")})

    def test_counts_are_preserved(self):
        total = sum(r["n"] for r in self.raw["coarse_table"])
        for path, (header, rows) in self.out.items():
            self.assertEqual(sum(r[header.index("n")] for r in rows), total, path)

    def test_shares_recomputed_from_counts(self):
        for path, (header, rows) in self.out.items():
            for r in rows:
                d = dict(zip(header, r))
                self.assertEqual(d["p"], f"{(d['ne'] + self.pk[d['k']]) / (d['n'] + 1):.10f}")

    def test_unmerged_rows_keep_their_values(self):
        with open(os.path.join(self.dir, "coarse_table.csv"), encoding="utf-8") as f:
            orig = {tuple(r[:-1]): r[-1] for r in csv.reader(f)}
        for r in self.out["fitted_tables/coarse_table_by_build.csv"][1]:
            key = tuple(str(x) for x in r[:-1])
            if key in orig:
                self.assertEqual(orig[key], r[-1])

    def test_other_row_pools_at_least_two_cells(self):
        """Each "other" row is the coarse value minus the listed rows, and holds two or more cells, so no single
        small cell can be recovered from one table."""
        fine = self.table("fitted_tables/comember_ages_table_by_build.csv")
        coarse = {(r["k"], r["build"], r["size6"], r["bt"], r["ab"]): r["n"]
                  for r in self.table("fitted_tables/coarse_table_by_build.csv")}
        by = defaultdict(list)
        for r in fine:
            by[(r["k"], r["build"], r["size6"], r["bt"], r["ab"])].append(r)
        n_other = 0
        for c, rs in by.items():
            self.assertEqual(sum(r["n"] for r in rs), coarse[c])
            other = [r for r in rs if r["ny"] == RF.OTHER]
            if other:
                n_other += 1
                listed = {(r["ny"], r["mb"], r["gb"]) for r in rs if r["ny"] != RF.OTHER}
                raw_keys = {(r["ny"], r["mb"], r["gb"]) for r in self.raw["comember_age_table"]
                            if (str(r["k"]), r["build"], r["size6"], r["bt"]) == c[:4] and r["ab"] in c[4].split("+")}
                self.assertGreaterEqual(len(raw_keys - listed), 2, c)
        self.assertGreater(n_other, 0)

    def test_parity_tables_follow_from_the_by_build_tables(self):
        """Every parity row equals a sum of published by-build rows, so the parity tables add no count."""
        bc = self.table("fitted_tables/coarse_table_by_build.csv")
        bf = self.table("fitted_tables/comember_ages_table_by_build.csv")
        pc = self.table("fitted_tables/coarse_table_by_parity.csv")
        pf = self.table("fitted_tables/comember_ages_table_by_parity.csv")

        def inside(build_row, parity_row):
            return (build_row["k"] == parity_row["k"] and TS.parity(build_row["build"]) == parity_row["parity"]
                    and build_row["size6"] == parity_row["size6"] and build_row["bt"] == parity_row["bt"]
                    and set(build_row["ab"].split("+")) <= set(parity_row["ab"].split("+")))
        for p in pc:
            self.assertEqual(p["n"], sum(b["n"] for b in bc if inside(b, p)))
        for p in pf:
            if p["ny"] == RF.OTHER:
                continue
            f = (p["ny"], p["mb"], p["gb"])
            self.assertEqual(p["n"], sum(b["n"] for b in bf if inside(b, p) and (b["ny"], b["mb"], b["gb"]) == f))

    def test_lookup_order(self):
        coarse = {tuple(r[:5]): float(r[-1]) for r in self.out["fitted_tables/coarse_table_by_build.csv"][1]}
        fine = {tuple(r[:8]): float(r[-1]) for r in self.out["fitted_tables/comember_ages_table_by_build.csv"][1]}
        listed = next(k for k in fine if k[5] != RF.OTHER)
        self.assertEqual(RF.lookup(coarse, fine, listed[:5], listed[5:], self.pk), fine[listed])
        other = next(k for k in fine if k[5] == RF.OTHER)
        self.assertEqual(RF.lookup(coarse, fine, other[:5], ("y9", "m99", "g99"), self.pk), fine[other])
        merged = ("8", "2010-11-02", "5", "multi", "a75")
        self.assertIn(RF.lookup(coarse, fine, merged, ("y9", "m99", "g99"), self.pk),
                      {coarse[merged[:4] + ("a65+a75",)], fine.get(merged[:4] + ("a65+a75",) + (RF.OTHER,) * 3)})
        self.assertEqual(RF.lookup(coarse, fine, ("4", "2008-11-04", "1", "multi", "a18"), ("n0", "none", "none"), {"4": 0.5}), 0.5)


class PublishedNames(unittest.TestCase):
    def test_names_are_distinct_and_outside_the_working_folders(self):
        work = {"tables", "table_cells", "robustness", "results", "household_change", "other_states", "build_error", "refresh"}
        published = RF.published()
        self.assertEqual(len(published), len(set(published)))
        self.assertEqual(len(published), 81)
        for p in published:
            self.assertNotIn(p.split("/")[0], work, p)
        self.assertFalse(set(RF.RELEASE_TABLES.values()) & set(RF.SHIPPED))


if __name__ == "__main__":
    unittest.main()
