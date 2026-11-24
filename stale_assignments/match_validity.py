# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Pipeline step 12: county cells (county0, key, num, den) for match validity, link survival and support ceilings. Descriptive only.

Build t = 2016-11-08, lag 0, persons with truth. Per record: ap = returned people other than the client, tp = true
co-residents at s other than the client, ai = returned people who are true co-residents.
match_validity, validity of reported matches, by policy: never_rebuilt, last_known, annual_rebuild, and the
probabilistic matcher at W >= none, -10, -5, 0, 5, 10, 15, 20, 25.
Among reported matches: exact household (returned set equals the true one; returned alone and living alone counts as exact,
and its share is a separate key), only non-residents, mean per-record member precision (ai / ap over matches with ap > 0).
Also match rate, co-member F1, mover share of loss, and the matcher's person-link precision (NCID used only to score).
Exit treatments for returned members absent at s: a = every absent member counts as not co-resident; b = soft exits (no
inactive row at s with reason RD death or RS moved out of state) are dropped from the returned set as unknown.
link_survival, over the last_known rows: ai / ap with a = every exit broken, b = only death and out-of-state broken, c = exits
excluded. support_ceiling, information ceilings (NCID used only as an oracle): per client, the best co-member F1 achievable from the 2016
graph (any subset: ceil_any), from people at the client's 2016 or current unit in the 2016 graph (ceil_reach), from the old
household only (ceil_old), from the 2016 occupants of the current unit only (ceil_new); the share of true co-residents
present in the 2016 graph. Graph rows of persons with one active row only.
Every num and den is a rational, summed exactly per county and key.
"""
import polars as pl

from . import store as S
from .common import T0
from .descriptives import long_sum

THRESHOLDS = [("none", -1e9), ("-10", -10.0), ("-5", -5.0), ("0", 0.0), ("5", 5.0), ("10", 10.0), ("15", 15.0), ("20", 20.0),
              ("25", 25.0)]
I = pl.Int64


def _absent(st: S.Store, need: pl.DataFrame) -> pl.DataFrame:
    """(b_dt, key, s_dt, n_soft, n_hard): returned members absent from the graph at s, by exit class."""
    out = []
    for b in sorted(set(need["b_dt"].to_list())):
        gb = st.read("graph", b, columns=["akey", "ncid"]).unique()
        nb = need.filter(pl.col("b_dt") == b)
        for s in sorted(set(nb["s_dt"].to_list())):
            mem = nb.filter(pl.col("s_dt") == s).join(gb.rename({"akey": "key"}), on="key", how="inner").unique()
            pres = st.read("graph", s, columns=["ncid"]).unique().with_columns(pl.lit(True).alias("p"))
            hx = (st.read("addr", s, columns=["ncid", "is_ai", "reason_cd"]).filter(~pl.col("is_ai"))
                  .group_by("ncid").agg(pl.when(pl.col("reason_cd").count() == 0).then(None)
                                        .otherwise(pl.col("reason_cd").is_in(["RD", "RS"]).any()).alias("hard")))
            m = mem.join(pres, on="ncid", how="left").join(hx, on="ncid", how="left")
            ab = pl.col("p").is_null()
            hard = pl.col("hard").fill_null(False)
            out.append(m.group_by("b_dt", "key", "s_dt").agg((ab & ~hard).sum().cast(I).alias("n_soft"),
                                                             (ab & hard).sum().cast(I).alias("n_hard")))
    return pl.concat(out) if out else pl.DataFrame(schema={"b_dt": pl.Date, "key": pl.String, "s_dt": pl.Date, "n_soft": I, "n_hard": I})


def _rows(df, key, nn, nd, dn, dd, keep=None):
    """One key family's cells, summed at once (families' keys are disjoint, so no stacked copy of the rows is built)."""
    x = df.select("county0", key.alias("key"), nn.cast(I).alias("nn"), nd.cast(I).alias("nd"), dn.cast(I).alias("dn"), dd.cast(I).alias("dd"))
    return long_sum(x if keep is None else x.filter(keep))


def validity_cells(st: S.Store, keep: pl.DataFrame, fsr: pl.DataFrame) -> pl.DataFrame:
    """keep: scored t = 2016, lag 0 rows of policies never_rebuilt, last_known and annual_rebuild with step 2's b_dt, s_dt,
    mkey, last_known_key;
    fsr: step 11 per-client records."""
    sf = lambda: pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    rec = keep.filter((pl.col("t_dt") == T0) & (pl.col("lag") == 0) & pl.col("has_truth")
                      & pl.col("policy").is_in(["never_rebuilt", "last_known", "annual_rebuild"])).select(
        "county0", "ncid", "k", "policy", "matched", (pl.col("t_size") - 1).alias("tp"),
        (pl.col("a_size") - sf()).alias("ap"), (pl.col("a_int") - sf()).alias("ai"), "f1c_n", "f1c_d", "src", "s_dt",
        pl.when(pl.col("policy") == "last_known").then(pl.col("t_dt")).otherwise(pl.col("b_dt")).alias("b_dt"),
        pl.when(pl.col("policy") == "last_known").then(pl.col("last_known_key")).otherwise(pl.col("mkey")).alias("key"),
        pl.lit(None, dtype=pl.Boolean).alias("same"))
    fs = fsr.filter(pl.col("has_truth")).select(
        "county0", "ncid", "k", (pl.col("t_size") - 1).alias("tp"), (pl.col("a_size") - sf()).alias("ap"),
        (pl.col("a_int") - sf()).alias("ai"), "src", "s_dt", pl.lit(T0).alias("b_dt"), pl.col("gakey").alias("key"), "same", "w")
    need = pl.concat([rec.select("b_dt", "key", "s_dt"), fs.select("b_dt", "key", "s_dt")]).drop_nulls("key").unique()
    ab = _absent(st, need)
    rec = rec.join(ab, on=["b_dt", "key", "s_dt"], how="left", nulls_equal=False)
    fs = fs.join(ab, on=["b_dt", "key", "s_dt"], how="left", nulls_equal=False)
    common = ["county0", "k", "policy", "matched", "tp", "ap", "ai", "f1c_n", "f1c_d", "mov", "same", "n_soft", "n_hard"]
    xs = [rec.with_columns(pl.col("ap").fill_null(0), pl.col("ai").fill_null(0), pl.col("src").is_in(["mover_together", "mover_split"]).alias("mov"))
          .select(common)]
    for th, tv in THRESHOLDS:
        mt = (pl.col("w") >= tv).fill_null(False)
        tp, ap, ai = pl.col("tp"), pl.col("ap"), pl.col("ai")
        c1, c2, c3 = ~mt, (tp == 0) & (ap == 0), (tp == 0) | (ap == 0) | (ai == 0)
        fn = pl.when(c1).then(pl.when(tp == 0).then(1).otherwise(0)).when(c2).then(1).when(c3).then(0).otherwise(2 * ai)
        fd = pl.when(c1).then(1).when(c2).then(1).when(c3).then(1).otherwise(ap + tp)
        xs.append(fs.with_columns(pl.lit(f"probabilistic_matcher_w{th}").alias("policy"), mt.alias("matched"), fn.alias("f1c_n"), fd.alias("f1c_d"),
                                  pl.col("src").is_in(["mover_together", "mover_split"]).alias("mov"))
                  .with_columns(pl.col("ap").fill_null(0), pl.col("ai").fill_null(0)).select(common))
    x = pl.concat(xs).with_columns(pl.col("n_soft").fill_null(0), pl.col("n_hard").fill_null(0))
    k = pl.col("k").cast(pl.String)
    pk = lambda m: pl.concat_str([pl.lit("match_validity|"), pl.col("policy"), pl.lit(f"|{m}|"), k])
    one, zero = pl.lit(1), pl.lit(0)
    m, mov, same = pl.col("matched"), pl.col("mov"), pl.col("same")
    lossn = pl.col("f1c_d") - pl.col("f1c_n")
    parts = [
        _rows(x, pk("mr"), pl.when(m).then(1).otherwise(0), one, one, one),
        _rows(x, pk("f1c"), pl.col("f1c_n"), pl.col("f1c_d"), one, one),
        _rows(x, pk("mov_loss"), pl.when(mov).then(lossn).otherwise(0), pl.col("f1c_d"), lossn, pl.col("f1c_d"), keep=pl.col("dn") != 0),
        _rows(x, pk("plink"), pl.when(m & same).then(1).otherwise(0), one,
              pl.when(m & same.is_not_null()).then(1).otherwise(0), one, keep=pl.col("dn") != 0)]
    xm = x.filter(m)
    for ex in ("a", "b"):
        y = xm.with_columns((pl.col("ap") - (pl.col("n_soft") if ex == "b" else 0)).alias("ape"))
        ape, tp, ai = pl.col("ape"), pl.col("tp"), pl.col("ai")
        ek = lambda mm: pl.concat_str([pl.lit("match_validity|"), pl.col("policy"), pl.lit(f"|{ex}|{mm}|"), k])
        b01 = lambda e: pl.when(e).then(1).otherwise(0)
        parts += [
            _rows(y, ek("exact"), b01(((ape == 0) & (tp == 0)) | ((ape > 0) & (ai == ape) & (ai == tp))), one, one, one),
            _rows(y, ek("alone_exact"), b01((ape == 0) & (tp == 0)), one, one, one),
            _rows(y, ek("nonres_only"), b01((ape > 0) & (ai == 0)), one, one, one),
            _rows(y, ek("mprec"), pl.when(ape > 0).then(ai).otherwise(0), pl.when(ape > 0).then(ape).otherwise(1),
                  b01(ape > 0), one)]
    fz = x.filter(pl.col("policy") == "last_known")
    for lab, den in (("a", pl.col("ap")), ("b", pl.col("ap") - pl.col("n_soft")), ("c", pl.col("ap") - pl.col("n_soft") - pl.col("n_hard"))):
        parts.append(_rows(fz, pl.concat_str([pl.lit(f"link_survival|{lab}|"), k]), pl.col("ai"), one, den, one))
    parts += _support_ceiling(st, rec)
    out = pl.concat(parts).sort(["key", "county0"])
    assert out.select("key", "county0").is_duplicated().sum() == 0, "two key families share a key"
    return out


def _support_ceiling(st: S.Store, rec: pl.DataFrame) -> list:
    g0 = (st.read("graph", T0, columns=["ncid", "akey", "n_ai_rows"]).filter(pl.col("n_ai_rows") == 1).select("ncid", "akey"))
    out = []
    nev = rec.filter(pl.col("policy") == "never_rebuilt").select("county0", "k", "ncid", "s_dt")
    cs = []
    for s in sorted(set(nev["s_dt"].to_list())):
        sg = (st.read("graph", s, columns=["ncid", "tk", "akey", "n_ai_rows"]).filter(pl.col("n_ai_rows") == 1).select("ncid", "tk", "akey"))
        c = (nev.filter(pl.col("s_dt") == s).join(sg.rename({"akey": "akey_s"}), on="ncid", how="inner")
             .join(g0.rename({"akey": "akey_t"}), on="ncid", how="left"))
        co = (c.select("k", "ncid", "tk", "akey_t", "akey_s").join(sg.rename({"ncid": "oid", "akey": "o_akey"}), on="tk", how="inner")
              .filter(pl.col("oid") != pl.col("ncid"))
              .join(g0.rename({"ncid": "oid", "akey": "g2"}), on="oid", how="left")
              .select("k", "ncid", "oid", pl.col("g2").is_not_null().alias("in_g"), (pl.col("g2") == pl.col("akey_t")).alias("is_old"),
                      (pl.col("g2") == pl.col("akey_s")).alias("is_new")))
        j = c.select("county0", "k", "ncid").join(co, on=["k", "ncid"], how="left")
        old, new = pl.col("is_old").fill_null(False), pl.col("is_new").fill_null(False)
        cs.append(j.group_by("county0", "k", "ncid").agg(
            pl.col("oid").count().cast(I).alias("tp"), pl.col("in_g").fill_null(False).sum().cast(I).alias("c_any"),
            old.sum().cast(I).alias("c_old"), new.sum().cast(I).alias("c_new"), (old | new).sum().cast(I).alias("c_reach")))
    if not cs:
        return out
    ca = pl.concat(cs)
    k = pl.col("k").cast(pl.String)
    one, tp = pl.lit(1), pl.col("tp")
    for name, c in (("ceil_any", "c_any"), ("ceil_reach", "c_reach"), ("ceil_old", "c_old"), ("ceil_new", "c_new")):
        cc = pl.col(c)
        out.append(_rows(ca, pl.concat_str([pl.lit(f"support_ceiling|{name}|"), k]), pl.when(tp == 0).then(1).otherwise(2 * cc),
                         pl.when(tp == 0).then(1).otherwise(cc + tp), one, one))
    out.append(_rows(ca, pl.concat_str([pl.lit("support_ceiling|alone_share|"), k]), pl.when(tp == 0).then(1).otherwise(0), one, one, one))
    out.append(_rows(ca, pl.concat_str([pl.lit("support_ceiling|coresidents_in_graph|"), k]), pl.col("c_any"), one, tp, one))
    return out
