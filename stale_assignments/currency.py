# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Anchor currency: how current the household recorded around an anchor still is, by build t and age k.

Rows: step 2 re-match rows of policy never_rebuilt, lag 0, persons with truth at s, every build t. Exact household means the
returned set equals the true co-resident set at s, the client excluded; returned members absent at s count as not
co-resident.

  kP     person anchor: the client's own cluster at t (the last-known key) is exact
  mr     address anchor: share re-matched by the address-only builder (never_rebuilt)
  kA     address anchor, among matches: the cluster at the client's last address is exact
  sigma  stayer share: the client's address at s maps to the client's own cluster at t (mkey = last-known key)
  kS     person anchor among stayers
  mu     person anchor among the rest (movers and unmatched addresses)
  kMA    address anchor among matched movers

kP = sigma * kS + (1 - sigma) * mu holds exactly (total probability over stayers and the rest).

anchor_cells holds integer (num, den) per (county0, key) with key "anchor|<quantity>|<t year>|<k>", so the
county-cluster bootstrap of statistics.ratio_stats applies unchanged.
"""
import itertools

import polars as pl

from . import student_t

I = pl.Int64
QUANTITIES = ("kP", "mr", "kA", "sigma", "kS", "mu", "kMA")
OUT_NAME = {"kP": "kP", "mr": "mr_addr", "kA": "kA", "sigma": "sigma", "kS": "kS", "mu": "mu", "kMA": "kMA"}
_COUNTS = ["n", "eP", "m", "eA", "st", "eS", "eM", "mm", "eMA"]


def _exact(ap, ai, tp):
    """SQL three-valued semantics: (ap = 0 AND tp = 0) OR (ap > 0 AND ai = ap AND ai = tp), null counted as false."""
    return ((ap == 0) & (tp == 0) | (ap > 0) & (ai == ap) & (ai == tp)).fill_null(False)


def anchor_counts(df: pl.DataFrame) -> pl.DataFrame:
    """Integer counts per (t_dt, k, county0) for one step 2 frame; empty unless it holds never_rebuilt, lag 0 rows."""
    x = df.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
    if x.height == 0:
        return pl.DataFrame(schema={"t_dt": pl.Date, "k": I, "county0": I, **{c: I for c in _COUNTS}})
    tp = pl.col("t_size") - 1
    s = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    m = pl.col("matched").fill_null(False)
    eP = _exact(pl.col("f_size") - 1, pl.col("f_int") - 1, tp)
    eA = m & _exact(pl.col("a_size") - s, pl.col("a_int") - s, tp)
    st = (pl.col("mkey") == pl.col("last_known_key")).fill_null(False)
    one = lambda e: e.cast(I).sum()
    return (x.group_by("t_dt", "k", "county0")
            .agg(pl.len().cast(I).alias("n"), one(eP).alias("eP"), one(m).alias("m"), one(eA).alias("eA"), one(st).alias("st"),
                 one(st & eP).alias("eS"), one(~st & eP).alias("eM"), one(m & ~st).alias("mm"), one(m & ~st & eA).alias("eMA")))


def anchor_cells(parts: list[pl.DataFrame]) -> pl.DataFrame:
    """Sum the per-frame counts and lay them out as (county0, key, num, den), sorted."""
    c = (pl.concat(parts).group_by("t_dt", "k", "county0").agg([pl.col(x).sum().cast(I) for x in _COUNTS]))
    tk = pl.concat_str([pl.col("t_dt").dt.year().cast(pl.String), pl.lit("|"), pl.col("k").cast(pl.String)])
    spec = {"kP": ("eP", "n"), "mr": ("m", "n"), "kA": ("eA", "m"), "sigma": ("st", "n"), "kS": ("eS", "st"),
            "mu": ("eM", pl.col("n") - pl.col("st")), "kMA": ("eMA", "mm"), "n": ("n", None)}
    out = []
    for q, (num, den) in spec.items():
        d = pl.lit(None, dtype=I) if den is None else (pl.col(den) if isinstance(den, str) else den)
        out.append(c.select("county0", pl.concat_str([pl.lit(f"anchor|{q}|"), tk]).alias("key"), pl.col(num).cast(I).alias("num"),
                            d.cast(I).alias("den")))
    return pl.concat(out).sort(["key", "county0"])


def anchor_currency(cells: pl.DataFrame, rstats: pl.DataFrame) -> pl.DataFrame:
    """anchor_currency.csv: one row per (t, k); n, then each quantity's point and nominal county-cluster intervals.
    rstats: statistics.ratio_stats over cells (the stat column is the cell key)."""
    st = {r["stat"]: r for r in rstats.iter_rows(named=True)}
    n = {r["key"]: r["num"] for r in cells.filter(pl.col("key").str.starts_with("anchor|n|")).group_by("key")
         .agg(pl.col("num").sum()).iter_rows(named=True)}
    rows = []
    for key in sorted(n):
        _, _, t, k = key.split("|")
        row = {"t": int(t), "k": int(k), "n": n[key]}
        for q in QUANTITIES:
            r = st.get(f"anchor|{q}|{t}|{k}", {})
            for c in ("point", "lo99", "hi99", "lo95", "hi95"):
                row[OUT_NAME[q] if c == "point" else f"{OUT_NAME[q]}_{c}"] = r.get(c)
        rows.append(row)
    cols = ["t", "k", "n"] + [c for q in QUANTITIES for c in (OUT_NAME[q], f"{OUT_NAME[q]}_lo99", f"{OUT_NAME[q]}_hi99",
                                                               f"{OUT_NAME[q]}_lo95", f"{OUT_NAME[q]}_hi95")]
    return pl.DataFrame(rows, schema={c: (I if c in ("t", "k", "n") else pl.Float64) for c in cols}).sort(["t", "k"])


def inputs_by_k(cells: pl.DataFrame, t0_year: int = 2016) -> pl.DataFrame:
    """inputs_by_k.csv: the whole-file inputs at the 2016 build, lag 0, for the never-refreshed graph (never_rebuilt) and
    the refreshed graph (lagged_rebuild). Exits are clients without truth at s; exits_matched is the share of exits the
    policy still answers, (n_m_all - n_m) / (n_all - n), with the count beside it in n_exits_matched. f1 is co-member F1
    among persons with truth (unmatched scored as in the cells)."""
    import math
    c = cells.filter((pl.col("t") == t0_year) & (pl.col("lag") == 0) & pl.col("policy").is_in(["never_rebuilt", "lagged_rebuild"]))
    g = {}
    for r in c.sort(["policy", "k", "county0", "bal"]).iter_rows(named=True):
        a = g.setdefault((r["policy"], r["k"]), {"n_all": 0, "n": 0, "n_m": 0, "n_m_all": 0, "f1c": []})
        for x in ("n_all", "n", "n_m", "n_m_all"):
            a[x] += r[x] or 0
        if r["f1c"] is not None:
            a["f1c"].append(r["f1c"])
    rows = []
    for k in sorted({k for _, k in g}):
        nv, rf = g[("never_rebuilt", k)], g[("lagged_rebuild", k)]
        assert nv["n_all"] == rf["n_all"] and nv["n"] == rf["n"], "never_rebuilt and refresh rows cover different clients"
        rows.append({"k": k, "n_file": nv["n_all"], "n_truth": nv["n"], "exit_share": (nv["n_all"] - nv["n"]) / nv["n_all"],
                     "mr_whole_never": nv["n_m_all"] / nv["n_all"], "mr_truth_never": nv["n_m"] / nv["n"],
                     "mr_whole_refresh": rf["n_m_all"] / rf["n_all"], "mr_truth_refresh": rf["n_m"] / rf["n"],
                     "f1_never": math.fsum(nv["f1c"]) / nv["n"], "f1_refresh": math.fsum(rf["f1c"]) / rf["n"],
                     "exits_matched_never": (nv["n_m_all"] - nv["n_m"]) / (nv["n_all"] - nv["n"]),
                     "exits_matched_refresh": (rf["n_m_all"] - rf["n_m"]) / (rf["n_all"] - rf["n"]),
                     "n_exits_matched_never": nv["n_m_all"] - nv["n_m"], "n_exits_matched_refresh": rf["n_m_all"] - rf["n_m"]})
    sch = {"k": I, "n_file": I, "n_truth": I, "exit_share": pl.Float64, "mr_whole_never": pl.Float64, "mr_truth_never": pl.Float64,
           "mr_whole_refresh": pl.Float64, "mr_truth_refresh": pl.Float64, "f1_never": pl.Float64, "f1_refresh": pl.Float64,
           "exits_matched_never": pl.Float64, "exits_matched_refresh": pl.Float64, "n_exits_matched_never": I,
           "n_exits_matched_refresh": I}
    return pl.DataFrame(rows, schema=sch)


FRONTIER_RULES = ["never_rebuilt", "last_known", "annual_rebuild"] + [f"probabilistic_matcher_w{t}"
                                                               for t in ("none", "-10", "-5", "0", "5", "10", "15", "20", "25")]


def validity_frontier(vc: pl.DataFrame, vstats: pl.DataFrame) -> pl.DataFrame:
    """validity_frontier.csv: per (k, rule), at the 2016 build: persons with truth, match rate, exact household among
    reported matches (exit treatment a; exact_share_soft_unknown uses treatment b, soft exits dropped as unknown), co-member F1, and the
    probabilistic matcher's person-link precision, with 99% county-cluster intervals. Read from the match_validity cells
    and validity_stats.csv."""
    st = {r["stat"]: r for r in vstats.iter_rows(named=True)}
    den = {r["key"]: r["den"] for r in vc.group_by("key").agg(pl.col("den").sum()).iter_rows(named=True)}
    rows = []
    for k in (2, 4, 6, 8):
        for rule in FRONTIER_RULES:
            row = {"k": k, "rule": rule}
            nkey = f"match_validity|{rule}|mr|{k}"
            row["n"] = None if den.get(nkey) is None else int(den[nkey])
            for col, key in (("match_rate", f"match_validity|{rule}|mr|{k}"), ("exact_share", f"match_validity|{rule}|a|exact|{k}"),
                             ("exact_share_soft_unknown", f"match_validity|{rule}|b|exact|{k}"),
                             ("f1", f"match_validity|{rule}|f1c|{k}"), ("plink", f"match_validity|{rule}|plink|{k}")):
                r = st.get(key, {})
                row[col], row[col + "_lo99"], row[col + "_hi99"] = r.get("point"), r.get("lo99"), r.get("hi99")
            rows.append(row)
    cols = {"k": I, "rule": pl.String, "n": I}
    for c in ("match_rate", "exact_share", "exact_share_soft_unknown", "f1", "plink"):
        cols.update({c: pl.Float64, c + "_lo99": pl.Float64, c + "_hi99": pl.Float64})
    return pl.DataFrame(rows, schema=cols).sort(["k", "rule"])


# ---- whole client file, exits identified by removal reason ---------------------------------------------------------------

EXIT_CLASS = {"RD": "RD", "RS": "RS", "RL": "RL", "RH": "RL", "RM": "RM", "R2": "DUP", "RG": "DUP", "DU": "DUP"}


def _never_rows(keep: pl.DataFrame) -> pl.DataFrame:
    return keep.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0)).select(
        "ncid", "k", "s_dt", "has_truth", "matched", "self_in_a", "t_size", "a_size", "a_int", "f_size", "f_int", "mkey", "last_known_key")


def _graph(st, d, cols, single=False):
    g = st.read("graph", d, columns=cols + (["n_ai_rows"] if single else []))
    if single:
        g = g.filter(pl.col("n_ai_rows") == 1).drop("n_ai_rows")
    return g.unique()


def exit_counts(st, keep: pl.DataFrame, t0) -> pl.DataFrame:
    """exit_counts.csv: (policy, k, class, outcome) -> n, n_rm_mover_wrong at build t0, lag 0, every client in the file.

    Policies: last_known (person anchor, the client's own cluster at t0) and never_rebuilt (address anchor). Persons with truth (class
    TRUTH): right = matched and exact, wrong = matched and not exact, none = unmatched. Exits (no truth at s) take the class of
    the client's own last row on or before s (active or inactive rows count as AI): RD deceased, RS moved from state, RL moved
    within state (RL, RH), RM removed after inactivity, DUP duplicates (R2, RG, DU), OTHER. Outcomes for exits: unmatched =
    none (every class, the deceased included: failing to find the address earns no credit); RD matched = wrong; RS wrong if
    a returned member other than the client is a registrant at s; RL wrong if a returned member is still at the returned
    address at s; any class wrong if a returned member is recorded deceased at s or the returned members sit at two or more
    addresses at s; everything else unknown (unknown_alone when the returned cluster holds no one but the client).
    outcome_active applies the RS and RL checks to returned members with an active row at s only (inactive registrants are
    often movers who did not update). n_rm_mover_wrong counts the unknown RM rows that the RL rule would make wrong
    (scenario: RM treated as movers)."""
    from . import store as S
    c = _never_rows(keep)
    ex = c.filter(~pl.col("has_truth")).select("ncid", "k", "s_dt")
    ids = ex.select("ncid").unique()
    last_s = max(ex["s_dt"].to_list())
    picks = []
    for d in S.SNAPSHOTS:
        if d < t0 or d > last_s:
            continue
        a = st.read("addr", d, columns=["ncid", "is_ai", "reason_cd"]).join(ids, on="ncid", how="semi")
        rc = pl.when(pl.col("is_ai").fill_null(False)).then(pl.lit("AI")).otherwise(pl.col("reason_cd"))
        a = (a.with_columns(rc.alias("rc"), pl.col("is_ai").fill_null(False).alias("ai"))
             .sort(["ncid", "ai", "rc"], descending=[False, True, False], nulls_last=False)
             .group_by("ncid", maintain_order=True).agg(pl.col("rc").first()))
        picks.append(a.with_columns(pl.lit(d).alias("d")))
    p = pl.concat(picks)
    last = (ex.join(p, on="ncid", how="inner").filter(pl.col("d") <= pl.col("s_dt"))
            .sort(["ncid", "k", "d"], descending=[False, False, True]).group_by("ncid", "k", maintain_order=True)
            .agg(pl.col("rc").first()))
    cls = pl.col("rc").replace_strict(EXIT_CLASS, default="OTHER", return_dtype=pl.String)
    exc = c.filter(~pl.col("has_truth")).join(last.with_columns(cls.alias("cls")), on=["ncid", "k"], how="inner")
    pol = pl.concat([
        exc.select("ncid", "k", "s_dt", "cls", pl.lit("last_known").alias("policy"), pl.lit(True).alias("m"), pl.col("last_known_key").alias("key")),
        exc.select("ncid", "k", "s_dt", "cls", pl.lit("never_rebuilt").alias("policy"), pl.col("matched").fill_null(False).alias("m"),
                   pl.col("mkey").alias("key"))])
    g0 = _graph(st, t0, ["akey", "ncid"])
    mem_parts = []
    pm = pol.filter(pl.col("m") & pl.col("key").is_not_null())
    for s in sorted(set(pm["s_dt"].to_list())):
        gs = _graph(st, s, ["ncid", "akey"]).rename({"akey": "akey_s"})
        ad = st.read("addr", s, columns=["ncid", "is_ai", "reason_cd", "status_cd"])
        dead = (ad.filter(~pl.col("is_ai").fill_null(False) & (pl.col("reason_cd") == "RD")).select(pl.col("ncid").alias("mid"))
                .unique().with_columns(pl.lit(True).alias("dead")))
        act = (ad.filter(pl.col("is_ai").fill_null(False) & (pl.col("status_cd") == "A")).select(pl.col("ncid").alias("mid"))
               .unique().with_columns(pl.lit(True).alias("act")))
        x = (pm.filter(pl.col("s_dt") == s).join(g0.rename({"akey": "key", "ncid": "mid"}), on="key", how="inner")
             .filter(pl.col("mid") != pl.col("ncid")))
        x = (x.join(gs.rename({"ncid": "mid"}), on="mid", how="left").join(dead, on="mid", how="left")
             .join(act, on="mid", how="left").with_columns(pl.col("act").fill_null(False)))
        mem_parts.append(x.group_by("ncid", "k", "policy").agg(
            pl.col("mid").n_unique().alias("n_mem"),
            pl.col("mid").filter(pl.col("akey_s").is_not_null()).n_unique().alias("n_nc"),
            pl.col("mid").filter(pl.col("akey_s") == pl.col("key")).n_unique().alias("n_old"),
            pl.col("mid").filter(pl.col("akey_s").is_not_null() & pl.col("act")).n_unique().alias("n_nc_act"),
            pl.col("mid").filter((pl.col("akey_s") == pl.col("key")) & pl.col("act")).n_unique().alias("n_old_act"),
            pl.col("mid").filter(pl.col("dead").fill_null(False)).n_unique().alias("n_dead"),
            pl.col("akey_s").drop_nulls().n_unique().alias("n_addr")))
    mem = pl.concat(mem_parts)
    e = pol.join(mem, on=["ncid", "k", "policy"], how="left").with_columns(
        pl.col("n_mem", "n_nc", "n_old", "n_dead", "n_addr", "n_nc_act", "n_old_act").fill_null(0))
    m, cl = pl.col("m"), pl.col("cls")

    def outcome(nc, old):
        return (pl.when(~m).then(pl.lit("none"))
                .when(cl == "RD").then(pl.lit("wrong"))
                .when((cl == "RS") & (pl.col(nc) > 0)).then(pl.lit("wrong"))
                .when((cl == "RL") & (pl.col(old) > 0)).then(pl.lit("wrong"))
                .when((pl.col("n_dead") > 0) | (pl.col("n_addr") >= 2)).then(pl.lit("wrong"))
                .when(pl.col("n_mem") == 0).then(pl.lit("unknown_alone"))
                .otherwise(pl.lit("unknown")))
    e = e.with_columns(outcome("n_nc", "n_old").alias("outcome"), outcome("n_nc_act", "n_old_act").alias("outcome_active"))
    e = e.with_columns(((cl == "RM") & pl.col("outcome").str.starts_with("unknown") & (pl.col("n_old") > 0)).alias("rmw"))
    tr = c.filter(pl.col("has_truth"))
    tp = pl.col("t_size") - 1
    s1 = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    fz = tr.select("k", pl.lit("last_known").alias("policy"),
                   pl.when(_exact(pl.col("f_size") - 1, pl.col("f_int") - 1, tp)).then(pl.lit("right")).otherwise(pl.lit("wrong"))
                   .alias("outcome"))
    nm = pl.col("matched").fill_null(False)
    nv = tr.select("k", pl.lit("never_rebuilt").alias("policy"),
                   pl.when(~nm).then(pl.lit("none"))
                   .when(_exact(pl.col("a_size") - s1, pl.col("a_int") - s1, tp)).then(pl.lit("right")).otherwise(pl.lit("wrong"))
                   .alias("outcome"))
    t_rows = pl.concat([fz, nv]).with_columns(pl.lit("TRUTH").alias("class"), pl.lit(False).alias("rmw"),
                                              pl.col("outcome").alias("outcome_active"))
    allr = pl.concat([t_rows.select("policy", "k", "class", "outcome", "outcome_active", "rmw"),
                      e.select("policy", "k", pl.col("cls").alias("class"), "outcome", "outcome_active", "rmw")])
    return (allr.group_by("policy", "k", "class", "outcome", "outcome_active")
            .agg(pl.len().cast(I).alias("n"), pl.col("rmw").cast(I).sum().alias("n_rm_mover_wrong"))
            .sort(["policy", "k", "class", "outcome", "outcome_active"]))


def three_outcome(ec: pl.DataFrame) -> pl.DataFrame:
    """three_outcome.csv: shares of the whole client file per (policy, k). right_lo counts identified right answers,
    right_hi adds every unknown exit; right_hi_rm_movers applies the RL rule to RM exits; *_rd_excluded drop the deceased
    from the file; *_active use the RS and RL checks on active returned members only; rd_share is the deceased share of the
    file. Exact integer ratios, rounded once."""
    rows = []
    for (pol, k), g in sorted(ec.group_by("policy", "k"), key=lambda x: x[0]):
        rs = list(g.iter_rows(named=True))
        N = sum(r["n"] for r in rs)

        def get(cl=None, o=None, col="outcome", exits_only=False):
            return sum(r["n"] for r in rs if (cl is None or r["class"] == cl) and (o is None or r[col] == o)
                       and not (exits_only and r["class"] == "TRUTH"))
        right, wrong, none = get(o="right"), get(o="wrong"), get(o="none")
        unk = get(o="unknown", exits_only=True) + get(o="unknown_alone", exits_only=True)
        wrong_a = get(o="wrong", col="outcome_active")
        unk_a = get(o="unknown", col="outcome_active", exits_only=True) + get(o="unknown_alone", col="outcome_active", exits_only=True)
        exits = N - get(cl="TRUTH")
        rmw = sum(r["n_rm_mover_wrong"] for r in rs)
        n_rd, rd_right = get(cl="RD"), get(cl="RD", o="right")
        Nx = N - n_rd
        rows.append({"policy": pol, "k": k, "n_file": N, "right_lo": right / N, "right_hi": (right + unk) / N,
                     "right_hi_rm_movers": (right + unk - rmw) / N, "right_lo_rd_excluded": (right - rd_right) / Nx,
                     "right_hi_rd_excluded": (right - rd_right + unk) / Nx, "wrong_lo": wrong / N, "none": none / N,
                     "unknown": unk / N, "exits": exits / N, "wrong_lo_active": wrong_a / N, "unknown_active": unk_a / N,
                     "right_hi_active": (right + unk_a) / N, "rd_share": n_rd / N})
    sch = {"policy": pl.String, "k": I, "n_file": I}
    sch.update({c: pl.Float64 for c in ("right_lo", "right_hi", "right_hi_rm_movers", "right_lo_rd_excluded", "right_hi_rd_excluded",
                                        "wrong_lo", "none", "unknown", "exits", "wrong_lo_active", "unknown_active",
                                        "right_hi_active", "rd_share")})
    return pl.DataFrame(rows, schema=sch)


# ---- reference measurement sensitivities (person anchor, persons with truth) ---------------------------------------------

def _member_counts(c: pl.DataFrame, sets: dict) -> pl.DataFrame:
    """sets: name -> frame (ncid, k, mid). Returns per (ncid, k) the size of each set and of each set's overlap with r."""
    u = pl.concat([f.select("ncid", "k", "mid", pl.lit(n).alias("src")) for n, f in sets.items()])
    u = u.group_by("ncid", "k", "mid").agg([(pl.col("src") == n).any().alias("in_" + n) for n in sets])
    return u


_BLDG = ["county_id", "zip5", "hn", "street_dir", "street_name", "street_type_cd", "street_sufx_cd"]


def mixed_buildings(st, d) -> pl.DataFrame:
    """NCIDs (active or inactive rows with a building key) whose building at snapshot d is mixed: some registrants carry a unit
    or a house-number suffix and some do not. Building = county, zip5, house digits (leading zeros dropped), street fields."""
    strip = lambda e: e.fill_null("").str.replace_all(r"[\s\-#\.]", "")
    a = (st.read("addr", d, columns=["ncid", "is_ai", "bldg_key", "county_id", "zip5", "street_dir", "street_name",
                                      "street_type_cd", "street_sufx_cd", "unit", "house_num", "half_code"])
         .filter(pl.col("is_ai").fill_null(False) & pl.col("bldg_key").is_not_null())
         .with_columns(pl.col("house_num").fill_null("").str.extract(r"^0*(\d+)", 1).alias("hn"),
                       (strip(pl.col("house_num").fill_null("").str.replace(r"^\d+", "")) + strip(pl.col("half_code"))).alias("suf"))
         .filter(pl.col("hn").is_not_null())
         .with_columns(((pl.col("suf") != "") | pl.col("unit").is_not_null()).alias("ud")))
    m = a.group_by(_BLDG).agg((pl.col("ud").any() & (~pl.col("ud")).any()).alias("mixed"))
    # an NCID with two building rows counts as mixed if either building is (the warehouse query took an arbitrary one)
    return (a.join(m, on=_BLDG, how="inner").group_by("ncid").agg(pl.col("mixed").any())
            .filter(pl.col("mixed")).select("ncid").sort("ncid"))


def ref_sensitivity(st, keep: pl.DataFrame, t0, reissue: pl.DataFrame | None = None, voted: pl.DataFrame | None = None) -> pl.DataFrame:
    """ref_sensitivity.csv, per k at build t0. Returned set R = the client's own cluster at t0 (last_known), graph rows of persons
    with one active-or-inactive row; clients with truth at s and such a row there. Exact = R equals the true set T, client
    excluded.
      base        T at s
      ghost       active-only truth: members whose row at s is inactive (status I*) are dropped from R and T
      slag        T at the January 1 snapshot after s (about two months later), clients with a row there; base_on_s2 is base
                  on the same clients
      tlag        base among clients who voted in the build election (voted: frame of ncid; null columns without it)
      relinked    clients without truth at s whose NCID was re-issued (reissue: old_ncid, new_ncid, new_first_dt) on or
                  before s, scored against the new NCID's household at s; reissue is base with them added
      unitkey_mixed_dropped  base without the clients whose building at t0 or at s is mixed (mixed_buildings); n_mixed
                  counts them
      agein_dropped  base with the members who were under 18 at t0 (age at s minus k below 18) dropped from R and T;
                  n_agein counts the clients whose T loses at least one such member
      tlag_all    base among clients who voted in the build election and whose every R member did too; n_v16_all counts them"""
    import datetime as dt
    c = (_never_rows(keep).filter(pl.col("has_truth")).select("ncid", "k", "s_dt", "last_known_key")
         .with_columns(pl.date(pl.col("s_dt").dt.year() + 1, 1, 1).alias("s2_dt")))
    g0 = _graph(st, t0, ["ncid", "akey"], single=True)
    r = (c.join(g0.rename({"akey": "last_known_key", "ncid": "mid"}), on="last_known_key", how="inner").filter(pl.col("mid") != pl.col("ncid"))
         .select("ncid", "k", "mid"))
    mx0 = mixed_buildings(st, t0)
    parts = []
    for s in sorted(set(c["s_dt"].to_list())):
        cs = c.filter(pl.col("s_dt") == s)
        mxs = pl.concat([mx0, mixed_buildings(st, s)]).unique()
        s2 = dt.date(s.year + 1, 1, 1)
        g1 = _graph(st, s, ["ncid", "tk"], single=True)
        g2 = _graph(st, s2, ["ncid", "tk"], single=True)
        a_s = st.read("addr", s, columns=["ncid", "is_ai", "status_cd", "age"]).filter(pl.col("is_ai").fill_null(False))
        ina = a_s.filter(pl.col("status_cd").str.starts_with("I")).select("ncid").unique()
        yng = (a_s.filter(pl.col("age").is_not_null()).group_by("ncid").agg(pl.col("age").min())
               .filter(pl.col("age") - (s.year - t0.year) < 18).select(pl.col("ncid").alias("mid")).with_columns(pl.lit(True).alias("young")))
        ck = (cs.join(g1.rename({"tk": "tk1"}), on="ncid", how="inner").join(g2.rename({"tk": "tk2"}), on="ncid", how="left"))
        t1 = (ck.join(g1.rename({"ncid": "mid", "tk": "tk1"}), on="tk1", how="inner").filter(pl.col("mid") != pl.col("ncid"))
              .select("ncid", "k", "mid"))
        t2 = (ck.filter(pl.col("tk2").is_not_null()).join(g2.rename({"ncid": "mid", "tk": "tk2"}), on="tk2", how="inner")
              .filter(pl.col("mid") != pl.col("ncid")).select("ncid", "k", "mid"))
        rs = r.join(ck.select("ncid", "k"), on=["ncid", "k"], how="semi")
        u = (_member_counts(ck, {"r": rs, "t1": t1, "t2": t2}).join(
            ina.rename({"ncid": "mid"}).with_columns(pl.lit(True).alias("inact")), on="mid", how="left")
            .join(yng, on="mid", how="left"))
        if voted is not None:
            u = u.join(voted.select(pl.col("ncid").alias("mid")).unique().with_columns(pl.lit(True).alias("mv")), on="mid", how="left")
        else:
            u = u.with_columns(pl.lit(None, dtype=pl.Boolean).alias("mv"))
        act = pl.col("inact").is_null()
        old = pl.col("young").is_null()
        cnt = lambda e: e.cast(I).sum()
        a = u.group_by("ncid", "k").agg(
            cnt(pl.col("in_r")).alias("r"), cnt(pl.col("in_t1")).alias("t1"), cnt(pl.col("in_r") & pl.col("in_t1")).alias("i1"),
            cnt(pl.col("in_t2")).alias("t2"), cnt(pl.col("in_r") & pl.col("in_t2")).alias("i2"),
            cnt(pl.col("in_r") & act).alias("ra"), cnt(pl.col("in_t1") & act).alias("t1a"),
            cnt(pl.col("in_r") & pl.col("in_t1") & act).alias("i1a"),
            cnt(pl.col("in_r") & old).alias("ro"), cnt(pl.col("in_t1") & old).alias("t1o"),
            cnt(pl.col("in_r") & pl.col("in_t1") & old).alias("i1o"), cnt(pl.col("in_t1") & ~old).alias("t1y"),
            cnt(pl.col("in_r") & ~pl.col("mv").fill_null(False)).alias("r_nv"))
        a = ck.select("ncid", "k", pl.col("tk2").is_not_null().alias("has2")).join(a, on=["ncid", "k"], how="left").with_columns(
            pl.col("r", "t1", "i1", "t2", "i2", "ra", "t1a", "i1a", "ro", "t1o", "i1o", "t1y", "r_nv").fill_null(0))
        a = a.join(mxs.with_columns(pl.lit(True).alias("mixed")), on="ncid", how="left").with_columns(pl.col("mixed").fill_null(False))
        parts.append(a)
    a = pl.concat(parts)
    if voted is not None:
        a = a.join(voted.select("ncid").unique().with_columns(pl.lit(True).alias("v")), on="ncid", how="left")
    ex = lambda rr, tt, ii: ((pl.col(rr) == 0) & (pl.col(tt) == 0)) | ((pl.col(rr) > 0) & (pl.col(ii) == pl.col(rr)) & (pl.col(ii) == pl.col(tt)))
    a = a.with_columns(ex("r", "t1", "i1").alias("e1"), ex("ra", "t1a", "i1a").alias("ea"), ex("r", "t2", "i2").alias("e2"),
                       ex("ro", "t1o", "i1o").alias("eo"))
    rel = relinked(st, keep, t0, reissue) if reissue is not None else None
    rows = []
    for k in sorted(set(a["k"].to_list())):
        x = a.filter(pl.col("k") == k)
        n, h2 = x.height, x.filter(pl.col("has2"))
        row = {"k": k, "n": n, "base": x["e1"].sum() / n, "ghost": x["ea"].sum() / n, "n_s2": h2.height, "n_v16_all": None,
               "tlag_all": None,
               "slag": h2["e2"].sum() / h2.height if h2.height else None,
               "base_on_s2": h2["e1"].sum() / h2.height if h2.height else None, "n_v16": None, "tlag": None,
               "n_relinked": None, "exact_relinked": None, "reissue": None}
        clean = x.filter(~pl.col("mixed"))
        row["n_mixed"] = n - clean.height
        row["unitkey_mixed_dropped"] = clean["e1"].sum() / clean.height if clean.height else None
        if voted is not None:
            v = x.filter(pl.col("v").fill_null(False))
            row["n_v16"], row["tlag"] = v.height, (v["e1"].sum() / v.height if v.height else None)
            va = v.filter(pl.col("r_nv") == 0)
            row["n_v16_all"], row["tlag_all"] = va.height, (va["e1"].sum() / va.height if va.height else None)
        row["n_agein"] = x.filter(pl.col("t1y") > 0).height
        row["agein_dropped"] = x["eo"].sum() / n
        if rel is not None and k in rel:
            nr, ne = rel[k]
            row["n_relinked"], row["exact_relinked"] = nr, ne / nr
            row["reissue"] = (x["e1"].sum() + ne) / (n + nr)
        rows.append(row)
    sch = {"k": I, "n": I, "base": pl.Float64, "ghost": pl.Float64, "n_s2": I, "slag": pl.Float64, "base_on_s2": pl.Float64,
           "n_v16": I, "tlag": pl.Float64, "n_relinked": I, "exact_relinked": pl.Float64, "reissue": pl.Float64,
           "n_mixed": I, "unitkey_mixed_dropped": pl.Float64, "n_agein": I, "agein_dropped": pl.Float64, "n_v16_all": I,
           "tlag_all": pl.Float64}
    return pl.DataFrame(rows, schema=sch)


def relinked(st, keep: pl.DataFrame, t0, reissue: pl.DataFrame) -> dict:
    """{k: (n, number exact)} for clients without truth at s whose NCID was re-issued on or before s and whose new NCID has
    a one-row graph record at s; R = the client's own cluster at t0, T = the new NCID's co-residents at s."""
    c = (_never_rows(keep).filter(~pl.col("has_truth")).select("ncid", "k", "s_dt", "last_known_key")
         .join(reissue.select(pl.col("old_ncid").alias("ncid"), "new_ncid", "new_first_dt"), on="ncid", how="inner")
         .filter(pl.col("new_first_dt") <= pl.col("s_dt")))
    g0 = _graph(st, t0, ["ncid", "akey"], single=True)
    out = []
    for s in sorted(set(c["s_dt"].to_list())):
        cs = c.filter(pl.col("s_dt") == s)
        g1 = _graph(st, s, ["ncid", "tk"], single=True)
        ck = cs.join(g1.rename({"ncid": "new_ncid"}), on="new_ncid", how="inner")
        r = (ck.join(g0.rename({"akey": "last_known_key", "ncid": "mid"}), on="last_known_key", how="inner").filter(pl.col("mid") != pl.col("ncid"))
             .select("ncid", "k", "mid"))
        t = (ck.join(g1.rename({"ncid": "mid"}), on="tk", how="inner").filter(pl.col("mid") != pl.col("new_ncid"))
             .select("ncid", "k", "mid"))
        u = _member_counts(ck, {"r": r, "t": t})
        a = u.group_by("ncid", "k").agg(pl.col("in_r").cast(I).sum().alias("r"), pl.col("in_t").cast(I).sum().alias("t"),
                                       (pl.col("in_r") & pl.col("in_t")).cast(I).sum().alias("i"))
        a = ck.select("ncid", "k").join(a, on=["ncid", "k"], how="left").with_columns(pl.col("r", "t", "i").fill_null(0))
        out.append(a)
    a = pl.concat(out)
    e = ((pl.col("r") == 0) & (pl.col("t") == 0)) | ((pl.col("r") > 0) & (pl.col("i") == pl.col("r")) & (pl.col("i") == pl.col("t")))
    res = {}
    for k, g in a.with_columns(e.alias("e")).group_by("k"):
        res[k[0]] = (g.height, int(g["e"].sum()))
    return res


# ---- level bands for later builds ----------------------------------------------------------------------------------------

BAND_TESTS = [(2018, 8, (2010, 2014)), (2020, 6, (2008, 2012, 2016)), (2022, 4, (2010, 2014)), (2024, 2, (2008, 2012, 2016))]
BAND_MARGIN = "0.05"


def currency_bands(acur: pl.DataFrame) -> pl.DataFrame:
    """currency_bands.csv: for each test pair (target build, k) and quantity (kP, kS, sigma), the last same-parity analog,
    the least-squares linear trend in build year through the analog builds evaluated at the target, and the band
    [min(last, trend) - 0.05, max(last, trend) + 0.05]. Computed in exact rational arithmetic from the point values and
    rounded once."""
    from fractions import Fraction as F
    v = {(r["t"], r["k"]): r for r in acur.iter_rows(named=True)}
    m = F(BAND_MARGIN)
    rows = []
    for T, k, an in BAND_TESTS:
        for q in ("kP", "kS", "sigma"):
            xs = [F(t) for t in an]
            ys = [F(v[(t, k)][q]) for t in an]
            xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
            sl = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
            tr = yb + sl * (F(T) - xb)
            last = ys[-1]
            rows.append({"target_t": T, "k": k, "quantity": q, "analogs": " ".join(map(str, an)), "last": float(last),
                         "trend": float(tr), "lo": float(min(last, tr) - m), "hi": float(max(last, tr) + m)})
    return pl.DataFrame(rows, schema={"target_t": I, "k": I, "quantity": pl.String, "analogs": pl.String, "last": pl.Float64,
                                      "trend": pl.Float64, "lo": pl.Float64, "hi": pl.Float64})


# Derived tables over stats.csv and inputs_by_k.csv. Plain double arithmetic: + - * / are correctly rounded under IEEE 754
# on every platform, so these are as exact as their inputs.

# The parts of the three primary hypotheses, identified by the start of their stat name in stats.csv:
# (name prefix, part, threshold, reciprocal of the value is linear in the exit score, has a tipping point).
PARTS = [
    ("gap G(2)", "gap", 0.0, False, True),
    ("widening G(8) - G(2)", "widening", 0.10, False, True),
    ("F1 margin", "f1_margin", 0.0, False, True),
    ("widening by build", "widening_by_build", 0.10, False, True),
    ("split-household share of loss", "split_share", 0.50, True, True),
    ("lagged-rebuild gap", "lagged_gap", 0.05, False, True),
    ("loss cut vs never_rebuilt", "loss_cut", 0.25, True, True),
    ("same-date rebuild F1", "same_date_f1", 0.95, False, False),  # secondary part: no tipping point
]


def _part(stat: str):
    """The PARTS row of a main-specification part row of stats.csv, or None (sensitivity rows carry ' | ')."""
    if "|" in stat:
        return None
    for p in PARTS:
        if stat.startswith(p[0]) and not (p[1] == "widening" and stat.startswith("widening by build")):
            return p
    return None


def small_cluster(stats_df: pl.DataFrame, g: int) -> pl.DataFrame:
    """small_cluster.csv: for each main-specification hypothesis part of stats.csv with a jackknife SE, point -/+
    t(0.995, G - 1) * jk_se, G the number of counties (delete-one-county jackknife with a t reference)."""
    tq = student_t.quantile("0.995", g - 1)
    rows = []
    for r in stats_df.sort("ord").iter_rows(named=True):
        if _part(r["stat"]) is None or r["jk_se"] is None:
            continue
        p, se = r["point"], r["jk_se"]
        rows.append({"stat": r["stat"], "point": p, "lo": p - tq * se, "hi": p + tq * se, "method": f"jackknife t{g - 1}"})
    return pl.DataFrame(rows, schema={"stat": pl.String, "point": pl.Float64, "lo": pl.Float64, "hi": pl.Float64,
                                      "method": pl.String})


def tipping_points(stats_df: pl.DataFrame) -> pl.DataFrame:
    """tipping_points.csv: the common exit score e* in [0, 1] at which each decided part (the same-date rebuild part
    excluded) reaches its threshold. G, its differences and the F1 margin are linear in e; the split-household share and
    the loss cut have a reciprocal linear in e, so e* follows exactly from the exits-scored-0 and exits-scored-1 rows.
    e_star is blank outside [0, 1]."""
    s = stats_df.sort("ord")
    pt = dict(zip(s["stat"].to_list(), s["point"].to_list()))
    rows = []
    for st in s["stat"].to_list():
        p = _part(st)
        if p is None or not p[4]:
            continue
        thr = p[2]
        v0, v1 = pt.get(f"{st} | exits scored 0"), pt.get(f"{st} | exits scored 1")
        e = None
        if v0 is not None and v1 is not None:
            a, b, t = (1 / v0, 1 / v1, 1 / thr) if p[3] else (v0, v1, thr)
            if b != a:
                x = (t - a) / (b - a)
                e = x if 0 <= x <= 1 else None
        rows.append({"stat": st, "point": pt[st], "threshold": thr, "exits0": v0, "exits1": v1, "e_star": e})
    return pl.DataFrame(rows, schema={"stat": pl.String, "point": pl.Float64, "threshold": pl.Float64, "exits0": pl.Float64,
                                      "exits1": pl.Float64, "e_star": pl.Float64})


TWO_SCORE = [("widening", "widening G(8) - G(2), never_rebuilt | exits scored 0", 0.10),
             ("f1_margin", "F1 margin, never_rebuilt | exits scored 0", 0.0)]


def tipping_two_score(stats_df: pl.DataFrame, ibk: pl.DataFrame) -> pl.DataFrame:
    """tipping_two_score.csv: worst case for the widening and the F1 margin with exits at k = 2 scored e2 and at k = 8
    scored e8. The part is value(e2, e8) = v00 + q2 e2 - q8 e8 (q = exit share); e8* is where it reaches the threshold, at
    e2 = 0 and e2 = 1."""
    pt = dict(zip(stats_df["stat"].to_list(), stats_df["point"].to_list()))
    q = dict(zip(ibk["k"].to_list(), ibk["exit_share"].to_list()))
    rows = []
    for part, name, thr in TWO_SCORE:
        v00 = pt[name]
        rows.append({"part": part, "value_at_00": v00, "threshold": thr, "e8_star_given_e2_0": (v00 - thr) / q[8],
                     "e8_star_given_e2_1": (v00 + q[2] - thr) / q[8]})
    return pl.DataFrame(rows, schema={"part": pl.String, "value_at_00": pl.Float64, "threshold": pl.Float64,
                                      "e8_star_given_e2_0": pl.Float64, "e8_star_given_e2_1": pl.Float64})


def envelopes(ibk: pl.DataFrame) -> pl.DataFrame:
    """envelopes.csv: whole-file ranges of the widening, the F1 margin and the loss cut when exits are scored anywhere in [0, 1].
    Fw(k, pol, e) = (1 - q) f1_pol + q e and Gw = mr_whole_pol - Fw, from inputs_by_k."""
    ib = {r["k"]: r for r in ibk.iter_rows(named=True)}

    def fw(k, pol, e):
        q = ib[k]["exit_share"]
        return (1 - q) * ib[k][f"f1_{pol}"] + q * e

    def gw(k, pol, e):
        return ib[k][f"mr_whole_{pol}"] - fw(k, pol, e)

    rows = [("widening", gw(8, "never", 1) - gw(2, "never", 0), gw(8, "never", 0) - gw(2, "never", 1))]
    margin = [(fw(2, "never", a) - fw(8, "never", b)) - 2 * (ib[2]["mr_whole_never"] - ib[8]["mr_whole_never"])
           for a, b in itertools.product((0, 1), repeat=2)]
    rows.append(("f1_margin", min(margin), max(margin)))
    for k in (4, 6, 8):
        cut = [((1 - fw(k, "never", a)) - (1 - fw(k, "refresh", b))) / (1 - fw(k, "never", a))
               for a, b in itertools.product((0, 1), repeat=2)]
        rows.append((f"loss_cut k={k}", min(cut), max(cut)))
    return pl.DataFrame(rows, schema={"part": pl.String, "lo": pl.Float64, "hi": pl.Float64}, orient="row")


def _exact_rows(df: pl.DataFrame) -> pl.Expr:
    """The records-table exact rule on a frame with t_size, a_size, a_int, self_in_a (null a_* counted as an empty set)."""
    sf = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    ap = (pl.col("a_size") - sf).fill_null(0)
    ai = (pl.col("a_int") - sf).fill_null(0)
    return _exact(ap, ai, pl.col("t_size") - 1)


def link_same(last_known: pl.DataFrame, fsr: pl.DataFrame) -> pl.DataFrame:
    """Integer counts for link_same_population.csv at one or more k: the last_known policy (perfect person link) against the
    probabilistic matcher on the same clients. last_known: keep rows of policy last_known, build 2016, lag 0, persons with truth; fsr: step 11 records."""
    f = last_known.select("k", "ncid", _exact_rows(last_known).alias("eP"))
    assert f.select("k", "ncid").is_duplicated().sum() == 0, "last_known rows repeat a client"
    m = fsr.filter(pl.col("has_truth")).select("k", "ncid", "w", _exact_rows(fsr).alias("e10"))
    assert m.select("k", "ncid").is_duplicated().sum() == 0, "matcher records repeat a client"
    j = f.join(m, on=["k", "ncid"], how="left")
    b = lambda e: e.fill_null(False).cast(I).sum()
    aggs = [pl.len().cast(I).alias("n"), b(pl.col("eP")).alias("eP")]
    for th in (0, 20):
        w = (pl.col("w") >= th).fill_null(False)
        aggs += [b(w).alias(f"n_w{th}"), b(w & pl.col("e10")).alias(f"e10_w{th}"), b(w & pl.col("eP")).alias(f"eP_w{th}")]
    return j.group_by("k").agg(aggs).sort("k")


def link_same_population(ls: pl.DataFrame) -> pl.DataFrame:
    """link_same_population.csv: kP over all clients, and the matcher's exact share beside kP on the clients it matches at W >= 0
    and W >= 20 (build 2016, lag 0, persons with truth)."""
    rows = []
    for r in ls.sort("k").iter_rows(named=True):
        row = {"k": r["k"], "n": r["n"], "kP_all": r["eP"] / r["n"]}
        for th in (0, 20):
            nw = r[f"n_w{th}"]
            if th == 0:
                row["n_w0"] = nw
            row[f"matcher_w{th}"] = r[f"e10_w{th}"] / nw if nw else None
            row[f"kP_on_w{th}"] = r[f"eP_w{th}"] / nw if nw else None
        row["n_w20"] = r["n_w20"]
        rows.append(row)
    return pl.DataFrame(rows, schema={"k": I, "n": I, "kP_all": pl.Float64, "n_w0": I, "matcher_w0": pl.Float64,
                                      "kP_on_w0": pl.Float64, "matcher_w20": pl.Float64, "kP_on_w20": pl.Float64, "n_w20": I})


def acs_movers(path: str) -> pl.DataFrame:
    """acs_movers.csv: ACS 1-year North Carolina mover rate, persons in a different house one year ago (same or different
    city) over total population, from inputs/acs_nc_movers.tsv (published Census aggregates; 2020 has no standard 1-year
    release). Integer over integer, so the rate is the correctly rounded double."""
    rows = []
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for ln in f:
            r = dict(zip(head, ln.rstrip("\n").split("\t")))
            a, b, p = int(r["moved_same_city"]), int(r["moved_different_city"]), int(r["total_pop"])
            rows.append({"year": int(r["year"]), "mover_rate": (a + b) / p, "movers": a + b, "total_pop": p})
    return pl.DataFrame(rows, schema={"year": I, "mover_rate": pl.Float64, "movers": I, "total_pop": I}).sort("year")


# ---- further diagnostics (descriptive; no decision depends on them) --------------------------------------------------------

def matcher_answer_diagnostics(keep: pl.DataFrame, fsr: pl.DataFrame) -> pl.DataFrame:
    """matcher_answer_diagnostics.csv: stayer and mover currency among the matcher's answers at the 2016 build, lag 0, persons with truth, per (k, rule, link).

    keep: never_rebuilt rows (mkey, last_known_key, f_size, f_int, t_size); fsr: step 11 records (w, same, a_size, a_int, self_in_a).
    Stayer: the client's address at s maps to the client's own cluster at t (mkey = last-known key), as in sigma. Person-anchor
    exact (eP): the client's cluster at t equals the true co-resident set at s. Rule probabilistic_matcher_w<theta>: the matcher answers when its weight
    is at least theta; V = its returned set is exact (absent members count as not co-resident). link: all, correct (the
    linked 2016 record is the client), wrong. Columns, as integers: n (clients), n_ans (answered), n_ans_stay, eV (answered
    and exact), eV_stay, eV_move, eP_ans_stay (person anchor exact among answered stayers), n_unans_stay, eP_unans_stay,
    n_ans_move, eP_ans_move; and the shares s_theta = n_ans_stay / n_ans, V, kS_ans, kS_unans, mu_ans, V_stay, V_move.
    Rule build_alone answers the clients whose cluster at t holds no one else (person anchor), link all."""
    nv = keep.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth")).select(
        "ncid", "k", (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay"),
        _exact(pl.col("f_size") - 1, pl.col("f_int") - 1, pl.col("t_size") - 1).alias("eP"), "f_size")
    s1 = pl.when(pl.col("self_in_a")).then(1).otherwise(0)
    fs = fsr.filter(pl.col("has_truth")).select(
        "ncid", "k", "w", "same", _exact(pl.col("a_size") - s1, pl.col("a_int") - s1, pl.col("t_size") - 1).alias("eV"))
    x = nv.join(fs, on=["ncid", "k"], how="inner")
    from .match_validity import THRESHOLDS
    parts = []
    b = lambda e: e.fill_null(False).cast(I).sum()
    for th, tv in THRESHOLDS:
        a = (pl.col("w") >= tv).fill_null(False)
        for link, lf in (("all", pl.lit(True)), ("correct", pl.col("same").fill_null(False)),
                         ("wrong", ~pl.col("same").fill_null(True))):
            y = x.with_columns(a.alias("ans"))
            y = y.filter(lf | ~pl.col("ans")) if link != "all" else y
            parts.append(y.group_by("k").agg(
                pl.len().cast(I).alias("n"), b(pl.col("ans")).alias("n_ans"), b(pl.col("ans") & pl.col("stay")).alias("n_ans_stay"),
                b(pl.col("ans") & pl.col("eV")).alias("eV"), b(pl.col("ans") & pl.col("stay") & pl.col("eV")).alias("eV_stay"),
                b(pl.col("ans") & ~pl.col("stay") & pl.col("eV")).alias("eV_move"),
                b(pl.col("ans") & pl.col("stay") & pl.col("eP")).alias("eP_ans_stay"),
                b(~pl.col("ans") & pl.col("stay")).alias("n_unans_stay"), b(~pl.col("ans") & pl.col("stay") & pl.col("eP")).alias("eP_unans_stay"),
                b(pl.col("ans") & ~pl.col("stay")).alias("n_ans_move"), b(pl.col("ans") & ~pl.col("stay") & pl.col("eP")).alias("eP_ans_move"))
                .with_columns(pl.lit(f"probabilistic_matcher_w{th}").alias("rule"), pl.lit(link).alias("link")))
    al = (pl.col("f_size") == 1).fill_null(False)
    parts.append(nv.with_columns(al.alias("ans")).group_by("k").agg(
        pl.len().cast(I).alias("n"), b(pl.col("ans")).alias("n_ans"), b(pl.col("ans") & pl.col("stay")).alias("n_ans_stay"),
        b(pl.col("ans") & pl.col("eP")).alias("eV"), b(pl.col("ans") & pl.col("stay") & pl.col("eP")).alias("eV_stay"),
        b(pl.col("ans") & ~pl.col("stay") & pl.col("eP")).alias("eV_move"),
        b(pl.col("ans") & pl.col("stay") & pl.col("eP")).alias("eP_ans_stay"),
        b(~pl.col("ans") & pl.col("stay")).alias("n_unans_stay"), b(~pl.col("ans") & pl.col("stay") & pl.col("eP")).alias("eP_unans_stay"),
        b(pl.col("ans") & ~pl.col("stay")).alias("n_ans_move"), b(pl.col("ans") & ~pl.col("stay") & pl.col("eP")).alias("eP_ans_move"))
        .with_columns(pl.lit("build_alone").alias("rule"), pl.lit("all").alias("link")))
    cols = ["n", "n_ans", "n_ans_stay", "eV", "eV_stay", "eV_move", "eP_ans_stay", "n_unans_stay", "eP_unans_stay", "n_ans_move",
            "eP_ans_move"]
    d = pl.concat([p.select("rule", "link", "k", *cols) for p in parts])
    r = lambda num, den: pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None)
    d = d.with_columns(r("n_ans", "n").alias("answer_rate"), r("n_ans_stay", "n_ans").alias("s_theta"), r("eV", "n_ans").alias("V"),
                       r("eP_ans_stay", "n_ans_stay").alias("kS_ans"), r("eP_unans_stay", "n_unans_stay").alias("kS_unans"),
                       r("eP_ans_move", "n_ans_move").alias("mu_ans"), r("eV_stay", "n_ans_stay").alias("V_stay"),
                       r("eV_move", "n_ans_move").alias("V_move"))
    order = {f"probabilistic_matcher_w{t}": i for i, (t, _) in enumerate(THRESHOLDS)} | {"build_alone": 99}
    lk = {"all": 0, "correct": 1, "wrong": 2}
    return (d.with_columns(pl.col("rule").replace_strict(order, return_dtype=I).alias("_r"), pl.col("link").replace_strict(lk, return_dtype=I).alias("_l"))
            .sort(["_r", "_l", "k"]).drop("_r", "_l"))


def mover_timing(st, keep: pl.DataFrame, t0) -> pl.DataFrame:
    """mover_timing.csv: mover currency by time since the move and for a fixed mover cohort, 2016 build, lag 0, persons with
    truth, person anchor (the client's cluster at t is exact at s).

    Move snapshot: the first snapshot after t at which the client is registered (active or inactive row) and none of the
    client's unit keys is the key of the client's cluster at t (the keys sigma compares; a county readdressing is a move).
    Movers at age k: not stayers (sigma). Rows:
      by_time   (k, years_since_move = whole years from the move snapshot to s, at most k): n, exact
      cohort    clients who are movers at k = 2, followed to k = 2, 4, 6, 8 (persons with truth at that k): n, exact
    Each row also splits exact into lone (the client's cluster at t holds no one else and the client lives alone at s) and
    group (everyone else exact: the build household, unchanged, at a new address); n_back counts movers whose registered
    address at s is again the build address's cluster (returned)."""
    from . import store as S
    nv = keep.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth")).select(
        "ncid", "k", "s_dt", "last_known_key", (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay"),
        _exact(pl.col("f_size") - 1, pl.col("f_int") - 1, pl.col("t_size") - 1).alias("eP"),
        ((pl.col("f_size") == 1) & (pl.col("t_size") == 1)).fill_null(False).alias("lone"))
    cl = nv.select("ncid", "last_known_key").unique().drop_nulls()
    last_s = max(nv["s_dt"].to_list())
    first = None
    for d in S.SNAPSHOTS:
        if d <= t0 or d > last_s:
            continue
        g = st.read("graph", d, columns=["ncid", "akey"]).join(cl.select("ncid"), on="ncid", how="semi")
        at = (g.join(cl, on="ncid", how="inner").group_by("ncid")
              .agg((pl.col("akey") == pl.col("last_known_key")).any().alias("home")))
        moved = at.filter(~pl.col("home")).select("ncid").with_columns(pl.lit(d).alias("move_dt"))
        first = moved if first is None else pl.concat([first, moved.join(first, on="ncid", how="anti")])
    # a move first seen after s does not date this mover (the client is home in the graph at s but not a stayer by sigma)
    mv = (nv.filter(~pl.col("stay")).join(first, on="ncid", how="left")
          .with_columns(pl.when(pl.col("move_dt") <= pl.col("s_dt")).then(pl.col("move_dt")).alias("move_dt")))
    yrs = ((pl.col("s_dt") - pl.col("move_dt")).dt.total_days() / 365.25).floor().cast(I)
    mv = mv.with_columns(yrs.alias("ysm"))
    b = lambda e: e.fill_null(False).cast(I).sum()
    agg = [pl.len().cast(I).alias("n"), b(pl.col("eP")).alias("exact"), b(pl.col("eP") & pl.col("lone")).alias("exact_lone"),
           b(pl.col("eP") & ~pl.col("lone")).alias("exact_group")]
    by_time = (mv.group_by("k", "ysm").agg(agg).with_columns(pl.lit("by_time").alias("table"), pl.lit(None, dtype=I).alias("cohort_k"))
               .rename({"ysm": "years_since_move"}))
    coh = nv.filter((pl.col("k") == 2) & ~pl.col("stay")).select("ncid")
    cohort = (nv.join(coh, on="ncid", how="semi").group_by("k").agg(agg + [b(pl.col("stay")).alias("n_back")])
              .with_columns(pl.lit("cohort").alias("table"), pl.lit(2).cast(I).alias("cohort_k"), pl.lit(None, dtype=I).alias("years_since_move")))
    cols = ["table", "cohort_k", "k", "years_since_move", "n", "exact", "exact_lone", "exact_group", "n_back"]
    out = pl.concat([by_time.with_columns(pl.lit(None, dtype=I).alias("n_back")).select(cols), cohort.select(cols)])
    out = out.with_columns((pl.col("exact") / pl.col("n")).alias("mu"))
    return out.sort(["table", "k", "years_since_move"], nulls_last=False)


def readdress_ablation(st, keep: pl.DataFrame, t0) -> pl.DataFrame:
    """readdress_ablation.csv: the reference with and without the readdressing rule, 2016 build, lag 0, persons with truth,
    per k. With the rule (tk, as everywhere else), co-residents at s share the reference household key, which maps a building
    that a county readdressed to its canonical building; without it (ak), co-residents share the registered unit key at s.
    Graph rows of persons with one active-or-inactive row, as in ref_sensitivity. R_P = the client's cluster at t0 (person
    anchor); R_A = the t0 cluster at the address the client's address at s matches (address anchor, matched clients only).
    Stayer as in sigma (mkey = last-known key; the matcher never uses the rule). n_affected counts clients whose true set differs
    between the two keys. Columns: n, n_affected, n_stay, n_matched, and exact counts and shares for kP, kS, mu and kA under
    each key."""
    c = (_never_rows(keep).filter(pl.col("has_truth"))
         .select("ncid", "k", "s_dt", "last_known_key", "mkey", pl.col("matched").fill_null(False).alias("m"),
                 (pl.col("mkey") == pl.col("last_known_key")).fill_null(False).alias("stay")))
    g0 = _graph(st, t0, ["ncid", "akey"], single=True)
    rp = (c.join(g0.rename({"akey": "last_known_key", "ncid": "mid"}), on="last_known_key", how="inner").filter(pl.col("mid") != pl.col("ncid"))
          .select("ncid", "k", "mid"))
    ra = (c.filter(pl.col("m")).join(g0.rename({"akey": "mkey", "ncid": "mid"}), on="mkey", how="inner")
          .filter(pl.col("mid") != pl.col("ncid")).select("ncid", "k", "mid"))
    parts = []
    for s in sorted(set(c["s_dt"].to_list())):
        cs = c.filter(pl.col("s_dt") == s)
        g1 = _graph(st, s, ["ncid", "tk", "akey"], single=True)
        ck = cs.join(g1, on="ncid", how="inner")
        tt = (ck.select("ncid", "k", "tk").join(g1.select(pl.col("ncid").alias("mid"), "tk"), on="tk", how="inner")
              .filter(pl.col("mid") != pl.col("ncid")).select("ncid", "k", "mid"))
        ta = (ck.select("ncid", "k", "akey").join(g1.select(pl.col("ncid").alias("mid"), "akey"), on="akey", how="inner")
              .filter(pl.col("mid") != pl.col("ncid")).select("ncid", "k", "mid"))
        keys = ck.select("ncid", "k")
        u = _member_counts(ck, {"rp": rp.join(keys, on=["ncid", "k"], how="semi"), "ra": ra.join(keys, on=["ncid", "k"], how="semi"),
                                "tt": tt, "ta": ta})
        cnt = lambda e: e.cast(I).sum()
        a = u.group_by("ncid", "k").agg(
            cnt(pl.col("in_rp")).alias("rp"), cnt(pl.col("in_ra")).alias("ra"), cnt(pl.col("in_tt")).alias("tt"),
            cnt(pl.col("in_ta")).alias("ta"), cnt(pl.col("in_tt") != pl.col("in_ta")).alias("tdiff"),
            cnt(pl.col("in_rp") & pl.col("in_tt")).alias("ipt"), cnt(pl.col("in_rp") & pl.col("in_ta")).alias("ipa"),
            cnt(pl.col("in_ra") & pl.col("in_tt")).alias("iat"), cnt(pl.col("in_ra") & pl.col("in_ta")).alias("iaa"))
        parts.append(ck.select("ncid", "k", "m", "stay").join(a, on=["ncid", "k"], how="left")
                     .with_columns(pl.col("rp", "ra", "tt", "ta", "tdiff", "ipt", "ipa", "iat", "iaa").fill_null(0)))
    a = pl.concat(parts)
    ex = lambda rr, tt, ii: ((pl.col(rr) == 0) & (pl.col(tt) == 0)) | ((pl.col(rr) > 0) & (pl.col(ii) == pl.col(rr)) & (pl.col(ii) == pl.col(tt)))
    a = a.with_columns(ex("rp", "tt", "ipt").alias("ePt"), ex("rp", "ta", "ipa").alias("ePa"),
                       (pl.col("m") & ex("ra", "tt", "iat")).alias("eAt"), (pl.col("m") & ex("ra", "ta", "iaa")).alias("eAa"))
    b = lambda e: e.cast(I).sum()
    out = a.group_by("k").agg(
        pl.len().cast(I).alias("n"), b(pl.col("tdiff") > 0).alias("n_affected"), b(pl.col("stay")).alias("n_stay"),
        b(pl.col("m")).alias("n_matched"),
        b(pl.col("ePt")).alias("eP_tk"), b(pl.col("ePa")).alias("eP_ak"),
        b(pl.col("stay") & pl.col("ePt")).alias("eS_tk"), b(pl.col("stay") & pl.col("ePa")).alias("eS_ak"),
        b(~pl.col("stay") & pl.col("ePt")).alias("eM_tk"), b(~pl.col("stay") & pl.col("ePa")).alias("eM_ak"),
        b(pl.col("eAt")).alias("eA_tk"), b(pl.col("eAa")).alias("eA_ak")).sort("k")
    nm = pl.col("n") - pl.col("n_stay")
    return out.with_columns(
        (pl.col("n_affected") / pl.col("n")).alias("affected_share"), (pl.col("n_stay") / pl.col("n")).alias("sigma"),
        (pl.col("eP_tk") / pl.col("n")).alias("kP_tk"), (pl.col("eP_ak") / pl.col("n")).alias("kP_ak"),
        (pl.col("eS_tk") / pl.col("n_stay")).alias("kS_tk"), (pl.col("eS_ak") / pl.col("n_stay")).alias("kS_ak"),
        (pl.col("eM_tk") / nm).alias("mu_tk"), (pl.col("eM_ak") / nm).alias("mu_ak"),
        (pl.col("eA_tk") / pl.col("n_matched")).alias("kA_tk"), (pl.col("eA_ak") / pl.col("n_matched")).alias("kA_ak"))


_BS_COLS = ["n", "eP", "st", "eS", "eM"]


def build_size_counts(rm0) -> pl.DataFrame:
    """Integer counts per (t year, k, single) over a run's lag 0 step 2 rows (policy never_rebuilt, persons with truth); single: the
    client's own cluster at t (the last-known key) has one member (f_size = 1)."""
    x = pl.scan_parquet(rm0) if isinstance(rm0, str) else rm0.lazy()
    x = x.filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0) & pl.col("has_truth"))
    eP = _exact(pl.col("f_size") - 1, pl.col("f_int") - 1, pl.col("t_size") - 1)
    st = (pl.col("mkey") == pl.col("last_known_key")).fill_null(False)
    one = lambda e: e.cast(I).sum()
    return (x.group_by(pl.col("t_dt").dt.year().alias("t"), "k", (pl.col("f_size") == 1).alias("single"))
            .agg(pl.len().cast(I).alias("n"), one(eP).alias("eP"), one(st).alias("st"), one(st & eP).alias("eS"), one(~st & eP).alias("eM"))
            .collect().sort(["t", "k", "single"]))


def _r4(v):
    """FORMAT of ROUND(v, 4) as the warehouse export wrote it: shortest decimal, trailing zeros dropped."""
    s = f"{v:.4f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def build_size_csv(parts, path):
    """currency_by_build_size.csv: person-anchor currency split by build household size (lives alone at t or not),
    every build t and age k. parts: [(split label, build_size_counts frame)] in output order."""
    import csv, os
    cols = ["split", "t", "k", "n", "single_sh", "kP", "kP_single", "kP_multi", "kS", "kS_single", "kS_multi", "mu", "mu_single",
            "mu_multi", "mover_single_sh", "mu_from_single"]
    rows = []
    for split, c in parts:
        g = {}
        for r in c.iter_rows(named=True):
            g.setdefault((r["t"], r["k"]), {})[r["single"]] = r
        for (t, k), d in sorted(g.items()):
            s, m = d[True], d[False]
            a = {x: s[x] + m[x] for x in _BS_COLS}
            mv = lambda r: r["n"] - r["st"]
            v = [s["n"] / a["n"], a["eP"] / a["n"], s["eP"] / s["n"], m["eP"] / m["n"], a["eS"] / a["st"], s["eS"] / s["st"],
                 m["eS"] / m["st"], a["eM"] / mv(a), s["eM"] / mv(s), m["eM"] / mv(m), mv(s) / mv(a), s["eM"] / a["eM"]]
            rows.append([split, t, k, a["n"]] + [_r4(x) for x in v])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        w.writerows(rows)
