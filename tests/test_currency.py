# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Tests for the anchor-currency layer: the exact-household rule, the stayer and mover identity, the level bands and the
NCID re-issue crosswalk. Run from the repository root: python -m unittest discover -s tests"""
import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl  # noqa: E402

from stale_assignments import currency, prepare  # noqa: E402

D = dt.date


class AnchorTest(unittest.TestCase):
    def frame(self):
        # t_size counts the client; f_* the client's own cluster at t; a_* the re-matched cluster (self_in_a: client in it)
        rows = [  # stayer, exact both ways
            dict(t_size=2, f_size=2, f_int=2, matched=True, a_size=2, a_int=2, self_in_a=True, mkey="u1", last_known_key="u1"),
            # stayer, household changed
            dict(t_size=3, f_size=2, f_int=2, matched=True, a_size=2, a_int=2, self_in_a=True, mkey="u2", last_known_key="u2"),
            # mover, own cluster exact (moved together), address anchor wrong
            dict(t_size=2, f_size=2, f_int=2, matched=True, a_size=3, a_int=1, self_in_a=False, mkey="u3", last_known_key="u4"),
            # unmatched, lives alone, cluster alone: person anchor exact
            dict(t_size=1, f_size=1, f_int=1, matched=False, a_size=None, a_int=0, self_in_a=False, mkey=None, last_known_key="u5"),
            # missing last-known cluster: never exact
            dict(t_size=2, f_size=None, f_int=None, matched=False, a_size=None, a_int=0, self_in_a=False, mkey=None, last_known_key=None)]
        df = pl.DataFrame(rows).with_columns(pl.lit(D(2016, 11, 8)).alias("t_dt"), pl.lit(8).alias("k"), pl.lit(1).alias("county0"),
                                              pl.lit("never_rebuilt").alias("policy"), pl.lit(0).alias("lag"), pl.lit(True).alias("has_truth"))
        return df

    def test_counts_and_identity(self):
        c = currency.anchor_counts(self.frame()).row(0, named=True)
        self.assertEqual((c["n"], c["eP"], c["m"], c["eA"], c["st"], c["eS"], c["eM"], c["mm"], c["eMA"]), (5, 3, 3, 1, 2, 1, 2, 1, 0))
        # kP = sigma kS + (1 - sigma) mu, exactly, in counts
        self.assertEqual(c["eP"], c["eS"] + c["eM"])

    def test_cells_layout(self):
        cells = currency.anchor_cells([currency.anchor_counts(self.frame())])
        d = {r["key"]: (r["num"], r["den"]) for r in cells.iter_rows(named=True)}
        self.assertEqual(d["anchor|kP|2016|8"], (3, 5))
        self.assertEqual(d["anchor|mu|2016|8"], (2, 3))
        self.assertEqual(d["anchor|n|2016|8"], (5, None))

    def test_other_policies_ignored(self):
        f = self.frame().with_columns(pl.lit("last_known").alias("policy"))
        self.assertEqual(currency.anchor_counts(f).height, 0)


class BandsTest(unittest.TestCase):
    # confirmation-county point values (four decimals) and the bands they give
    POINTS = {(2008, 2): (0.7305, 0.7786, 0.9157), (2012, 2): (0.7099, 0.7613, 0.9077), (2016, 2): (0.671, 0.7321, 0.8844),
              (2010, 4): (0.5206, 0.6034, 0.7984), (2014, 4): (0.478, 0.5719, 0.7561),
              (2008, 6): (0.4515, 0.5399, 0.7439), (2012, 6): (0.4226, 0.5196, 0.7001), (2016, 6): (0.3788, 0.4688, 0.6651),
              (2010, 8): (0.3677, 0.4677, 0.6407), (2014, 8): (0.3172, 0.4176, 0.577)}
    BANDS = {(2018, "kP"): (0.22, 0.37), (2018, "kS"): (0.32, 0.47), (2018, "sigma"): (0.46, 0.63),
            (2020, "kP"): (0.29, 0.43), (2020, "kS"): (0.39, 0.52), (2020, "sigma"): (0.57, 0.72),
            (2022, "kP"): (0.34, 0.53), (2022, "kS"): (0.46, 0.62), (2022, "sigma"): (0.62, 0.81),
            (2024, "kP"): (0.56, 0.72), (2024, "kS"): (0.64, 0.78), (2024, "sigma"): (0.81, 0.93)}

    def test_bands(self):
        acur = pl.DataFrame([dict(t=t, k=k, kP=v[0], kS=v[1], sigma=v[2]) for (t, k), v in self.POINTS.items()])
        for r in currency.currency_bands(acur).iter_rows(named=True):
            self.assertEqual((round(r["lo"], 2), round(r["hi"], 2)), self.BANDS[(r["target_t"], r["quantity"])])


class ReissueTest(unittest.TestCase):
    def addr(self, rows, d):
        base = dict(ncid=None, is_ai=True, registr_dt=D(2000, 1, 1), first_fp=1, surname_norm_fp=2, mid_init_fp=None, name_sufx=None,
                    age=40, county_id=1, unit_key="u")
        return pl.DataFrame([{**base, **r} for r in rows], schema_overrides={"age": pl.Int64, "mid_init_fp": pl.Int64,
                                                                             "name_sufx": pl.String})

    def test_pair(self):
        st = None
        st = prepare.reissue_update(st, self.addr([dict(ncid="A", age=40), dict(ncid="X", first_fp=9, age=50)], D(2012, 1, 1)), D(2012, 1, 1))
        st = prepare.reissue_update(st, self.addr([dict(ncid="A", age=41), dict(ncid="X", first_fp=9, age=51)], D(2013, 1, 1)), D(2013, 1, 1))
        # A disappears; B appears two years later with the same name and birth year within one
        st = prepare.reissue_update(st, self.addr([dict(ncid="B", age=44), dict(ncid="X", first_fp=9, age=53)], D(2015, 1, 1)), D(2015, 1, 1))
        x = prepare.reissue_crosswalk(st)
        self.assertEqual(x.select("old_ncid", "new_ncid").rows(), [("A", "B")])
        self.assertEqual(x["new_first_dt"][0], D(2015, 1, 1))

    def test_ambiguous_dropped(self):
        st = prepare.reissue_update(None, self.addr([dict(ncid="A", age=40)], D(2012, 1, 1)), D(2012, 1, 1))
        st = prepare.reissue_update(st, self.addr([dict(ncid="B", age=43), dict(ncid="C", age=43)], D(2015, 1, 1)), D(2015, 1, 1))
        self.assertEqual(prepare.reissue_crosswalk(st).height, 0)

    def test_too_late(self):
        st = prepare.reissue_update(None, self.addr([dict(ncid="A", age=40)], D(2010, 1, 1)), D(2010, 1, 1))
        st = prepare.reissue_update(st, self.addr([dict(ncid="B", age=44)], D(2014, 1, 1)), D(2014, 1, 1))
        self.assertEqual(prepare.reissue_crosswalk(st).height, 0)


class DerivedTest(unittest.TestCase):
    def stats(self):
        rows = [("gap G(2), never_rebuilt", 0.2, 0.01), ("F1 margin, never_rebuilt", 0.1, 0.02),
                ("split-household share of loss, member_tracker, k = 4", 0.6, 0.01), ("same-date rebuild F1, client-file lag 1, k = 2", 0.9, 0.01),
                ("widening by build, pooled mean of the five builds", 0.2, None), ("living-alone floor, k = 2", 0.2, 0.01),
                ("gap G(2), never_rebuilt | exits scored 0", 0.3, 0.01), ("gap G(2), never_rebuilt | exits scored 1", -0.1, 0.01),
                ("F1 margin, never_rebuilt | exits scored 0", 0.25, 0.01), ("F1 margin, never_rebuilt | exits scored 1", 0.05, 0.01),
                ("split-household share of loss, member_tracker, k = 4 | exits scored 0", 0.25, 0.01),
                ("split-household share of loss, member_tracker, k = 4 | exits scored 1", 0.6, 0.01),
                ("widening G(8) - G(2), never_rebuilt | exits scored 0", 0.34, 0.01)]
        return pl.DataFrame([dict(stat=s, ord=i, point=p, jk_se=j) for i, (s, p, j) in enumerate(rows)],
                            schema={"stat": pl.String, "ord": pl.Int64, "point": pl.Float64, "jk_se": pl.Float64})

    def test_small_cluster(self):
        sc = currency.small_cluster(self.stats(), 20)
        self.assertEqual(sc["stat"].to_list()[:3], ["gap G(2), never_rebuilt", "F1 margin, never_rebuilt",
                                                     "split-household share of loss, member_tracker, k = 4"])
        self.assertNotIn("widening by build, pooled mean of the five builds", sc["stat"].to_list())  # no jackknife SE
        self.assertNotIn("living-alone floor, k = 2", sc["stat"].to_list())
        r = sc.row(0, named=True)
        self.assertAlmostEqual(r["hi"] - r["point"], 2.860935 * 0.01, places=7)
        self.assertEqual(r["method"], "jackknife t19")

    def test_tipping(self):
        tp = {r["stat"]: r for r in currency.tipping_points(self.stats()).iter_rows(named=True)}
        self.assertNotIn("same-date rebuild F1, client-file lag 1, k = 2", tp)
        self.assertAlmostEqual(tp["gap G(2), never_rebuilt"]["e_star"], 0.75)  # 0.3 - 0.4 e = 0
        self.assertIsNone(tp["F1 margin, never_rebuilt"]["e_star"])  # 1.25 > 1
        self.assertEqual(tp["F1 margin, never_rebuilt"]["exits0"], 0.25)
        # reciprocal: 1/0.25 + e (1/0.6 - 1/0.25) = 2
        self.assertAlmostEqual(tp["split-household share of loss, member_tracker, k = 4"]["e_star"], (2 - 4) / (1 / 0.6 - 4))
        self.assertIsNone(tp["widening by build, pooled mean of the five builds"]["exits0"])

    def test_two_score_and_envelopes(self):
        ibk = pl.DataFrame([dict(k=k, exit_share=q, f1_never=f, f1_refresh=g, mr_whole_never=m, mr_whole_refresh=m)
                            for k, q, f, g, m in ((2, 0.1, 0.8, 0.8, 0.95), (4, 0.2, 0.6, 0.7, 0.9), (6, 0.3, 0.5, 0.6, 0.88),
                                                  (8, 0.4, 0.4, 0.55, 0.85))])
        st = self.stats()
        two = currency.tipping_two_score(st, ibk).row(0, named=True)
        self.assertAlmostEqual(two["e8_star_given_e2_0"], (0.34 - 0.10) / 0.4)
        self.assertAlmostEqual(two["e8_star_given_e2_1"], (0.34 + 0.1 - 0.10) / 0.4)
        env = {r["part"]: (r["lo"], r["hi"]) for r in currency.envelopes(ibk).iter_rows(named=True)}
        # widening: Gw(8,1) - Gw(2,0) and Gw(8,0) - Gw(2,1)
        gw = lambda m, q, f, e: m - ((1 - q) * f + q * e)
        self.assertAlmostEqual(env["widening"][0], gw(0.85, 0.4, 0.4, 1) - gw(0.95, 0.1, 0.8, 0))
        self.assertAlmostEqual(env["widening"][1], gw(0.85, 0.4, 0.4, 0) - gw(0.95, 0.1, 0.8, 1))
        self.assertEqual(sorted(env), ["f1_margin", "loss_cut k=4", "loss_cut k=6", "loss_cut k=8", "widening"])
        self.assertLessEqual(env["f1_margin"][0], env["f1_margin"][1])


class LinkSameTest(unittest.TestCase):
    def test_counts(self):
        b = dict(k=2, t_size=2)
        fz = pl.DataFrame([{**b, "ncid": "a", "a_size": 2, "a_int": 2, "self_in_a": True},   # exact
                           {**b, "ncid": "b", "a_size": 3, "a_int": 2, "self_in_a": True},   # not exact
                           {**b, "ncid": "c", "a_size": 2, "a_int": 2, "self_in_a": True}])  # exact, matcher unmatched
        fs = pl.DataFrame([{**b, "ncid": "a", "has_truth": True, "w": 21.0, "a_size": 2, "a_int": 2, "self_in_a": True},
                           {**b, "ncid": "b", "has_truth": True, "w": 3.0, "a_size": 2, "a_int": 2, "self_in_a": True},
                           {**b, "ncid": "c", "has_truth": True, "w": -4.0, "a_size": 2, "a_int": 2, "self_in_a": True}])
        r = currency.link_same(fz, fs).row(0, named=True)
        self.assertEqual((r["n"], r["eP"], r["n_w0"], r["e10_w0"], r["eP_w0"], r["n_w20"], r["e10_w20"], r["eP_w20"]),
                         (3, 2, 2, 2, 1, 1, 1, 1))
        out = currency.link_same_population(currency.link_same(fz, fs)).row(0, named=True)
        self.assertEqual((out["kP_all"], out["matcher_w0"], out["kP_on_w0"]), (2 / 3, 1.0, 0.5))


class HistoryTest(unittest.TestCase):
    def test_voted(self):
        import tempfile
        import zipfile
        from stale_assignments import voter_history
        head = '"county_id"\t"election_lbl"\t"election_desc"\t"ncid"\r\n'
        rows = ['"1"\t"11/08/2016"\t"11/08/2016 GENERAL"\t"AA1"\r\n', '"1"\t"11/08/2016"\t"11/08/2016 GENERAL"\t"AA1"\r\n',
                '"2"\t"11/06/2018"\t"11/08/2016 IN NAME ONLY"\t"AA2"\r\n', '"3"\t"11/08/2016"\t"x"\t""\r\n',
                '"4"\t"11/08/2016"\t"x"\t"BB3"\r\n']
        with tempfile.TemporaryDirectory() as d:
            z = os.path.join(d, "h.zip")
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("ncvhis_Statewide.txt", head + "".join(rows))
            self.assertEqual(voter_history.voted(z)["ncid"].to_list(), ["AA1", "BB3"])


class MixedTest(unittest.TestCase):
    def test_mixed(self):
        b = dict(is_ai=True, bldg_key=1, county_id=1, zip5="27601", street_dir="", street_name="MAIN", street_type_cd="ST",
                 street_sufx_cd="", unit=None, half_code=None)
        rows = [{**b, "ncid": "A", "house_num": "012"}, {**b, "ncid": "B", "house_num": "12", "unit": "2"},  # mixed building 12
                {**b, "ncid": "C", "house_num": "14"}, {**b, "ncid": "D", "house_num": "14"},                 # clean
                {**b, "ncid": "E", "house_num": "16B"}, {**b, "ncid": "F", "house_num": "16-B"},              # all suffixed
                {**b, "ncid": "G", "house_num": "18", "bldg_key": None}]                                      # no key

        class St:
            def read(self, t, d, columns):
                return pl.DataFrame(rows, schema_overrides={"unit": pl.String, "half_code": pl.String, "bldg_key": pl.Int64}).select(columns)
        self.assertEqual(currency.mixed_buildings(St(), D(2016, 11, 8))["ncid"].to_list(), ["A", "B"])


class UnitKeyTest(unittest.TestCase):
    def test_keys(self):
        from stale_assignments import unit_key
        self.assertEqual(unit_key.keys_from_parts(["NORTH", "", "", "MAIN", "STREET", "", ""]), ("MAIN N ST", "MAIN"))
        self.assertEqual(unit_key.keys_from_parts(["", "", "", "NORTH CAROLINA HIGHWAY 54", "", "", ""]), ("54 HWY NC", "54"))

    def test_counts(self):
        from stale_assignments import unit_key
        v = dict(ncid="x", county_id=1, is_ai=True, half_code=None, street_dir="", street_name="MAIN", street_type_cd="ST",
                 street_sufx_cd="", zip5="27601", bldg_key=1)
        addr = pl.DataFrame([{**v, "house_num": "10", "unit": None}, {**v, "house_num": "10", "unit": None},   # pooled
                             {**v, "house_num": "12", "unit": "1"}, {**v, "house_num": "12", "unit": None}],  # partial
                            schema_overrides={"unit": pl.String, "half_code": pl.String})
        ap = pl.DataFrame({"county_id": [1] * 4, "zip5": ["27601"] * 4, "hn": ["10", "10", "12", "12"], "full": ["MAIN ST"] * 4,
                           "core": ["MAIN"] * 4, "ud": ["||A", "||B", "||1", "||2"]})
        r = unit_key.unit_counts(addr, ap, D(2016, 11, 8)).row(0, named=True)
        self.assertEqual((r["bld_all"], r["bld"], r["bld_pooled"], r["bld_multi_partial"], r["bld_disagree"]), (2, 2, 1, 1, 2))


class OutcomeTest(unittest.TestCase):
    def test_three_outcome_rd_and_active(self):
        # whole file of 10: 6 with truth (4 right, 1 wrong, 1 none), 4 exits; an unmatched RD exit is "none" 
        rows = [("TRUTH", "right", "right", 4), ("TRUTH", "wrong", "wrong", 1), ("TRUTH", "none", "none", 1),
                ("RD", "none", "none", 1), ("RD", "wrong", "wrong", 1), ("RL", "wrong", "unknown", 1), ("RM", "unknown", "unknown", 1)]
        ec = pl.DataFrame([dict(policy="never_rebuilt", k=8, **{"class": c}, outcome=o, outcome_active=oa, n=n, n_rm_mover_wrong=0)
                           for c, o, oa, n in rows])
        r = currency.three_outcome(ec).row(0, named=True)
        self.assertEqual(r["n_file"], 10)
        self.assertAlmostEqual(r["right_lo"], 0.4)
        self.assertAlmostEqual(r["none"], 0.2)
        self.assertAlmostEqual(r["right_lo_rd_excluded"], 4 / 8)
        self.assertAlmostEqual(r["wrong_lo"], 0.3)
        self.assertAlmostEqual(r["wrong_lo_active"], 0.2)
        self.assertAlmostEqual(r["unknown_active"], 0.2)
        self.assertAlmostEqual(r["rd_share"], 0.2)

    def test_matcher_answers_and_build_alone(self):
        keep = AnchorTest().frame().with_columns(pl.col("matched").alias("m")).drop("m")
        fsr = pl.DataFrame({"ncid": ["a", "b", "c", "d", "e"], "k": [8] * 5, "has_truth": [True] * 5, "w": [25.0, 25.0, 0.0, None, None],
                            "same": [True, True, False, None, None], "a_size": [2, 2, 3, None, None], "a_int": [2, 2, 1, 0, 0],
                            "self_in_a": [True, True, False, False, False], "t_size": [2, 3, 2, 1, 2]})
        keep = keep.with_columns(pl.Series("ncid", ["a", "b", "c", "d", "e"]))
        d = currency.matcher_answer_diagnostics(keep, fsr)
        r = d.filter((pl.col("rule") == "probabilistic_matcher_w20") & (pl.col("link") == "all")).row(0, named=True)
        self.assertEqual((r["n"], r["n_ans"], r["n_ans_stay"], r["eV"], r["eP_ans_stay"]), (5, 2, 2, 1, 1))
        self.assertAlmostEqual(r["s_theta"], 1.0)
        b = d.filter(pl.col("rule") == "build_alone").row(0, named=True)
        self.assertEqual((b["n_ans"], b["eV"]), (1, 1))

    def test_age_bands(self):
        from stale_assignments import calibration
        x = pl.DataFrame({"age": [17, 18, 19, 20, 74, 75, 120, 250]}).with_columns(calibration.band_expr(pl.col("age")).alias("b"))
        self.assertEqual(x["b"].to_list(), [None, "18-19", "18-19", "20-24", "70-74", "75+", "75+", None])


if __name__ == "__main__":
    unittest.main()
