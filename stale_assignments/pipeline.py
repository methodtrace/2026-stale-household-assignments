# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Runs the stages in order and writes the outputs and the manifest.

Stage 1 (snapshots.py): each published snapshot zip to a typed address table.
Stage 2 (prepare.py): household tags, building relabel edges and their components, graph records, age flags.
Stage 3 (steps 1 to 13): person and household events, re-match, stored ID, scores and county cells, statistics with
county-cluster intervals, build age, descriptive cells, the probabilistic matcher and the validity cells, and their
statistics.

Every output is sorted and written as text with a fixed format; the manifest lists each output's SHA-256 and an order-free
row fingerprint of every intermediate table, so two runs can be compared line by line.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import multiprocessing
import os
import time

import polars as pl

from . import calibration, config, currency, descriptives, events, match_validity, prepare, rematch, row_digest, score
from . import probabilistic_matcher as matcher
from . import snapshots, statistics, stored_assignment
from . import parallel as par
from . import store as S
from .common import T0

from .constants import POISSON_BOOTSTRAP_SALT as SEED
REPS = 2000


def log(msg, path=None):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---- fixed-format text outputs -------------------------------------------------------------------------------------------

def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (dt.date,)):
        return v.isoformat()
    return str(v)


def write_csv(df: pl.DataFrame, path: str):
    """CSV with LF line ends, minimal quoting, floats as the shortest text that reads back to the same double."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow(df.columns)
    for row in df.iter_rows():
        w.writerow([_fmt(v) for v in row])
    data = buf.getvalue().encode("utf-8")
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return hashlib.sha256(data).hexdigest()


def table_digest(df: pl.DataFrame) -> dict:
    """Order-free fingerprint of a table: row count, XOR and sum of the row fingerprints (row_digest)."""
    cols = list(df.columns)
    fx = df.with_columns(*(pl.col(c).map_elements(repr, return_dtype=pl.String) for c in cols if df.schema[c] == pl.Float64))
    r = row_digest.table_fp(fx, cols).row(0, named=True)
    return {"rows": int(r["n"]), "xor": int(r["x"] or 0), "sum": str(r["sm"] or 0), "columns": ",".join(cols)}


class Manifest:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def put(self, section, name, value):
        self.d.setdefault(section, {})[name] = value
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.d, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)


# ---- stage 2 ---------------------------------------------------------------------------------------------------------------

def stage2_date(st: S.Store, d, logf=None):
    """The part of stage 2 that needs one date only: its household tags, then drop the columns no later step reads for this
    date. reproduce_all.py calls it right after the date's stage 1 table is built, so the full tables never pile up on disk."""
    if not st.exists("tags", d):
        t = prepare.household_tags(st.read("addr", d))
        st.write(t, "tags", d)
        log(f"tags {d} rows={t.height}", logf)
    _reissue_apply(st, d)
    if d not in S.FULL_ADDR and S.SLIM_DROP[0] in pl.read_parquet_schema(st.path("addr", d)):
        st.write(st.read("addr", d).drop(S.SLIM_DROP), "addr", d)


def _reissue_apply(st: S.Store, d):
    """Fold date d into the re-issue crosswalk state, once, in date order (the state records the last date applied)."""
    done = st.read("reissue_state", columns=["last_applied"])["last_applied"][0] if st.exists("reissue_state") else None
    if done is not None and d <= done:
        return
    prev = [x for x in S.SNAPSHOTS if x < d]
    if (prev[-1] if prev else None) != done:
        raise RuntimeError(f"re-issue state is at {done}; snapshot {d} cannot be applied before {prev[-1]}. Delete the work "
                           "directory's tables and rerun.")
    cols = ["ncid", "is_ai", "registr_dt", "first_fp", "surname_norm_fp", "mid_init_fp", "name_sufx", "age", "county_id", "unit_key"]
    state = st.read("reissue_state").drop("last_applied") if done is not None else None
    state = prepare.reissue_update(state, st.read("addr", d, columns=cols), d)
    st.write(state.with_columns(pl.lit(d).alias("last_applied")), "reissue_state")


def _slim_stage3(st: S.Store, d):
    # disk: after its graph records exist, a date outside FULL_ADDR is read only for the ADDR_STAGE3 columns (age flags, stage 3)
    if d not in S.FULL_ADDR and len(pl.read_parquet_schema(st.path("addr", d))) > len(S.ADDR_STAGE3):
        st.write(st.read("addr", d, columns=S.ADDR_STAGE3), "addr", d)


GB_PER_PAIR = 8   # peak memory of one relabel pair (two dates' address columns) or one date's graph records

RELABEL_COLS = ["ncid", "is_ai", "bldg_key", "county_id", "house_num", "street_name", "half_code", "street_dir", "street_type_cd",
                "street_sufx_cd", "zip5"]


def _relabel_task(task):
    root, d0, d1 = task
    st = S.Store(root)
    e = prepare.relabel_pair(st.read("addr", d0, columns=RELABEL_COLS), st.read("addr", d1, columns=RELABEL_COLS), d0, d1)
    st.write(e, "relabel_pairs", d1)
    return e.height


def _graph_task(task):
    root, d, canon = task
    st = S.Store(root)
    g = prepare.graph_records(st.read("tags", d), st.read("addr", d), st.read(canon))
    st.write(g, "graph", d)
    return g.height


def stage2(st: S.Store, man: Manifest, logf=None):
    dates = S.SNAPSHOTS
    for d in dates:
        stage2_date(st, d, logf)
    if not st.exists("relabel_edges"):
        cols = ["ncid", "is_ai", "bldg_key", "county_id", "house_num", "street_name", "half_code", "street_dir", "street_type_cd",
                "street_sufx_cd", "zip5"]
        # one file per consecutive pair, named by the later date; pairs are independent and run side by side
        todo = [(st.root, d0, d1) for d0, d1 in zip(dates, dates[1:]) if not st.exists("relabel_pairs", d1)]
        for (_, d0, d1), h in zip(todo, par.pmap(_relabel_task, todo, n=par.jobs_for(GB_PER_PAIR, most=len(todo) or 1),
                                                share=True)):
            log(f"relabel {d0} {d1} edges={h}", logf)
        e = pl.concat([st.read("relabel_pairs", d1) for d1 in dates[1:]]).sort(["s0", "b0", "b1"])
        st.write(e, "relabel_edges")
        st.write(prepare.canon_from_edges(e), "canon")
        log(f"relabel edges {e.height}", logf)
    if later_file_present(st):
        stage2_later_file_edges(st, logf)  # before 2026-01-01 is slimmed: its relabel pair with the later file needs the street fields
    todo = [(st.root, d, "canon") for d in dates if not st.exists("graph", d)]
    for (_, d, _), h in zip(todo, par.pmap(_graph_task, todo, n=par.jobs_for(GB_PER_PAIR, most=len(todo) or 1), share=True)):
        log(f"graph {d} rows={h}", logf)
    for d in dates:
        _slim_stage3(st, d)
    if not st.exists("ncid_reissue"):
        x = prepare.reissue_crosswalk(st.read("reissue_state").drop("last_applied"))
        st.write(x, "ncid_reissue")
        log(f"ncid re-issue pairs {x.height}", logf)
    if not st.exists("age_flags"):
        a = prepare.age_flags([(d, (lambda d=d: st.read("addr", d, columns=["ncid", "is_ai", "age"]))) for d in dates])
        st.write(a, "age_flags")
        log(f"age flags {a.height}", logf)
    if later_file_present(st):
        stage2_later_file(st, logf)


# ---- the 2026-11-03 file (later_file runs) ---------------------------------------------------------------------------------
# The registered tables are never recomputed. The later file gets its own household tags; the relabel edges gain the pair
# (2026-01-01, 2026-11-03) and the components are rebuilt from all edges (canon_later_file), which the later
# file's graph records use; the age flags of the later file use all 41 files as age sources (age_flags_later_file = the
# registered rows plus these).

def later_file_present(st: S.Store) -> bool:
    return st.exists("addr", S.LATER_FILE_DATE)


def stage2_later_file_edges(st: S.Store, logf=None, prev_addr=None):
    if st.exists("relabel_pairs", S.LATER_FILE_DATE):
        return
    cols = ["ncid", "is_ai", "bldg_key", "county_id", "house_num", "street_name", "half_code", "street_dir", "street_type_cd",
            "street_sufx_cd", "zip5"]
    d0 = S.SNAPSHOTS[-1]
    a0 = prev_addr if prev_addr is not None else st.read("addr", d0, columns=cols)
    e = prepare.relabel_pair(a0.select(cols), st.read("addr", S.LATER_FILE_DATE, columns=cols), d0, S.LATER_FILE_DATE)
    st.write(e, "relabel_pairs", S.LATER_FILE_DATE)
    log(f"relabel {d0} {S.LATER_FILE_DATE} edges={e.height}", logf)


def stage2_later_file(st: S.Store, logf=None):
    d = S.LATER_FILE_DATE
    if not st.exists("tags", d):
        t = prepare.household_tags(st.read("addr", d))
        st.write(t, "tags", d)
        log(f"tags {d} rows={t.height}", logf)
    if not st.exists("canon_later_file"):
        e = pl.concat([st.read("relabel_edges"), st.read("relabel_pairs", d)]).sort(["s0", "b0", "b1"])
        st.write(e, "relabel_edges_later_file")
        st.write(prepare.canon_from_edges(e), "canon_later_file")
        log(f"relabel edges with the later file {e.height}", logf)
    if not st.exists("graph", d):
        g = prepare.graph_records(st.read("tags", d), st.read("addr", d), st.read("canon_later_file"))
        st.write(g, "graph", d)
        log(f"graph {d} rows={g.height}", logf)
    if not st.exists("age_flags_later_file"):
        dates = S.SNAPSHOTS + [d]
        a = prepare.age_flags([(x, (lambda x=x: st.read("addr", x, columns=["ncid", "is_ai", "age"]))) for x in dates], flag_dates={d})
        st.write(pl.concat([st.read("age_flags"), a]).sort(["snapshot_dt", "ncid", "age"]), "age_flags_later_file")
        log(f"age flags of the later file {a.height}", logf)


# ---- stage 3 ---------------------------------------------------------------------------------------------------------------

class Sink:
    """Receives step 2 pairs; scores them (step 4), sums county cells (step 5), keeps the t = 2016 lag 0 rows that later
    steps need, and fingerprints the step 2 rows and scored rows for checking."""
    RM_FP = ["t_dt", "ncid", "county0", "policy", "lag", "b_dt", "s_dt", "k", "crm_rec_dt", "has_truth", "matched", "name_hit",
             "self_in_a", "t_size", "a_size", "a_int", "f_size", "f_int", "mkey", "last_known_key"]
    REC_FP = ["t_dt", "ncid", "county0", "policy", "lag", "k", "has_truth", "matched", "t_size", "a_size", "a_int", "self_in_a",
              "fm", "fs", "bal", "event_list", "people_event", "src"]

    # columns of the kept rows that steps 9, 11 and 12 and the paper tables read
    KEEP_COLS = ["t_dt", "ncid", "county0", "policy", "lag", "b_dt", "s_dt", "k", "crm_rec_dt", "has_truth", "matched",
                 "t_size", "a_size", "a_int", "self_in_a", "f_size", "f_int", "f1c_n", "f1c_d", "mkey", "last_known_key", "event_list",
                 "people_event", "src"]

    def __init__(self, ev, fingerprint=False, keep_dir=None, rm0_dir=None):
        self.ev, self.fp, self.keep_dir, self.rm0_dir = ev, fingerprint, keep_dir, rm0_dir
        self.bal, self.cells, self.keep, self.anchor, self.rm0 = {}, [], [], [], []
        self.acc = {"rematch": [0, 0, 0], "records": [0, 0, 0]}

    def _acc(self, name, df, cols):
        r = row_digest.table_fp(df, cols).row(0, named=True)
        a = self.acc[name]
        a[0] += r["n"]
        a[1] ^= r["x"] or 0
        a[2] += r["sm"] or 0

    def __call__(self, t, p, df):
        tr = os.environ.get("STALE_TRACE")
        if tr:  # memory trace (debug): peak and current RSS per emitted frame
            import resource
            with open(tr, "a") as f:
                cur = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") >> 20
                f.write(f"{t} {p and (p['b_dt'], p['s_dt'], p['policy'], p['lag'])} rows={df.height} "
                        f"cur={cur} peak={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10}\n")
        if p is None:
            self.bal[t] = df
            return
        if self.fp:
            self._acc("rematch", df, self.RM_FP)
        if self.rm0_dir is not None and p["lag"] == 0:
            # the lag 0 step 2 rows of every build and policy (the warehouse's <run>_rematch where lag = 0), which the
            # analyses after stage 3 read (paper.py)
            path = os.path.join(self.rm0_dir, f"r{len(self.rm0):04d}.parquet")
            df.select(self.RM_FP).write_parquet(path, compression="zstd", statistics=False)
            self.rm0.append(path)
        self.anchor.append(currency.anchor_counts(df))
        self.add(df, t)

    def add(self, df, t, extra=None):
        for r in score.derived_rows(df):
            sc = score.score(r, self.bal[t], self.ev)
            if self.fp:
                self._acc("records", sc, self.REC_FP)
            self.cells.append(score.cells(sc))
            pol, lag = r["policy"][0], r["lag"][0]
            if t == T0 and lag == 0 and pol in score.KEEP_POLICIES and extra is None:
                q = df.select("ncid", "b_dt", "s_dt", "mkey", "last_known_key", "crm_rec_dt", "f_size", "f_int")
                kp = sc.join(q, on="ncid", how="left").select(self.KEEP_COLS)
                # memory: kept rows go to disk part files until step 2 ends
                path = os.path.join(self.keep_dir, f"{len(self.keep):04d}.parquet")
                kp.write_parquet(path, compression="zstd", statistics=False)
                self.keep.append(path)


def isolated(fn, *args):
    """Run fn(*args) in a new interpreter (spawn) and wait; raise if it fails. Each heavy step runs this way so that its
    memory goes back to the operating system when it ends, whatever the platform's allocator keeps."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=fn, args=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{getattr(fn, '__name__', fn)} failed with exit code {p.exitcode}")


def _digest(rs: S.Store, name: str):
    """Write the table's digest next to it (digests/<name>.json), computed in the process that wrote the table."""
    os.makedirs(os.path.join(rs.root, "digests"), exist_ok=True)
    with open(os.path.join(rs.root, "digests", name + ".json"), "w", encoding="utf-8") as f:
        json.dump(table_digest(rs.read(name)), f, sort_keys=True)


def _digest_only(out, name):
    _digest(S.Store(out), name)


def _events_src(rs):
    return score.events_src(rs.read("person_events"), rs.read("household_events"))


def _step1(root, out, counties, sample, fingerprint, logf):
    st, rs = S.Store(root), S.Store(out)
    pe = events.person_events(st, counties, sample).sort(["ncid", "k"])
    he = events.household_events(st, pe).sort(["dp0", "snapshot_dt"])
    rs.write(he, "household_events")
    rs.write(pe, "person_events")
    log(f"step 1 person_events={pe.height} household_events={he.height}", logf)
    del pe, he
    _digest(rs, "person_events")
    _digest(rs, "household_events")


def _concat_parts(parts, dest):
    pl.concat([pl.scan_parquet(f) for f in parts]).sink_parquet(dest + ".part", compression="zstd", statistics=False)
    os.replace(dest + ".part", dest)


def _step2_rematch(root, out, counties, sample, fingerprint, kd):
    """The re-match itself, in its own process: writes the keep and lag 0 part files and the sink's small results to kd,
    so that the process that sorts and concatenates them starts without the re-match's memory."""
    st, rs = S.Store(root), S.Store(out)
    ev = _events_src(rs)
    pe_c = rs.read("person_events", columns=["ncid", "client", "county0"])
    sink = Sink(ev, fingerprint, kd, kd)
    rematch.rematch(st, counties, sample, pe_c, sink)
    del pe_c
    sink.bal.get(T0, pl.DataFrame(schema={"ncid": pl.String, "bal": pl.Boolean})).write_parquet(os.path.join(kd, "_bal.parquet"))
    currency.anchor_cells(sink.anchor).write_parquet(os.path.join(kd, "_anchor.parquet"))
    score.sort_cells(pl.concat(sink.cells)).write_parquet(os.path.join(kd, "_cells.parquet"))
    with open(os.path.join(kd, "_sink.json"), "w") as f:
        json.dump({"keep": sink.keep, "rm0": sink.rm0, "acc": sink.acc}, f)


def _step2(root, out, counties, sample, fingerprint, logf):
    """Steps 2, 4 and 5 for the re-matched pairs. Writes keep, cells_step2, bal_step2 (and check_step2.json)."""
    rs = S.Store(out)
    kd = os.path.join(out, "keep_parts")
    if os.path.isdir(kd):
        for f in os.listdir(kd):
            os.remove(os.path.join(kd, f))
    os.makedirs(kd, exist_ok=True)
    isolated(_step2_rematch, root, out, counties, sample, fingerprint, kd)
    with open(os.path.join(kd, "_sink.json")) as f:
        sk = json.load(f)
    if sk["keep"]:
        keep = pl.concat([pl.read_parquet(f) for f in sk["keep"]]).sort(["policy", "k", "ncid"])
    else:  # no lag 0 rows of the main build kept (score-pair runs)
        keep = pl.DataFrame(schema={c: pl.Null for c in Sink.KEEP_COLS})
    rs.write(keep, "keep")
    del keep
    for f in sk["keep"]:
        os.remove(f)
    # in pair order (deterministic); readers never depend on row order; streamed, in a separate process
    isolated(_concat_parts, list(sk["rm0"]), rs.path("rematch0"))
    for f in sk["rm0"]:
        os.remove(f)
    if fingerprint:
        with open(os.path.join(out, "check_step2.json"), "w") as f:
            json.dump(sk["acc"], f)
    rs.write(pl.read_parquet(os.path.join(kd, "_bal.parquet")), "bal_step2")
    rs.write(pl.read_parquet(os.path.join(kd, "_anchor.parquet")), "anchor_cells")
    cells = pl.read_parquet(os.path.join(kd, "_cells.parquet"))
    for f in os.listdir(kd):
        os.remove(os.path.join(kd, f))
    os.rmdir(kd)
    rs.write(cells, "cells_step2")  # written last: marks step 2 complete
    log("step 2 done", logf)


def _step3(root, out, counties, sample, fingerprint, logf):
    """Step 3 (stored assignments), their scores and cells; joins step 2's cells into cells."""
    st, rs = S.Store(root), S.Store(out)
    sink = Sink(_events_src(rs), fingerprint)
    sink.bal[T0] = rs.read("bal_step2")
    sid = stored_assignment.stored_assignments(st, rs.read("person_events")).sort(["policy", "k", "ncid"])
    rs.write(sid, "stored_assignment")
    log(f"step 3 stored_assignment={sid.height}", logf)
    sid = sid.with_columns(pl.lit(None, dtype=pl.Int64).alias("t_size_u"), pl.lit(None, dtype=pl.Int64).alias("a_int_u"))
    for pol, k in sid.select("policy", "k").unique().sort(["policy", "k"]).iter_rows():  # memory: one cell group at a time
        sink.add(sid.filter((pl.col("policy") == pol) & (pl.col("k") == k)), T0, extra="stored_assignment")
    del sid
    cells = score.sort_cells(pl.concat([rs.read("cells_step2")] + sink.cells))
    if fingerprint:
        chk = json.load(open(os.path.join(out, "check_step2.json")))
        r, a = chk["records"], sink.acc["records"]
        chk["records"] = [r[0] + a[0], r[1] ^ a[1], r[2] + a[2]]
        with open(os.path.join(out, "check.json"), "w") as f:
            json.dump(chk, f)
    rs.write(cells, "cells")
    log(f"steps 4-5 cells={cells.height}", logf)
    for f in ("cells_step2", "bal_step2"):
        os.remove(rs.path(f))
    if os.path.exists(os.path.join(out, "check_step2.json")):
        os.remove(os.path.join(out, "check_step2.json"))
    for name in ("stored_assignment", "keep", "cells"):
        _digest(rs, name)


def _step8(root, out, counties, sample, fingerprint, logf):
    st, rs = S.Store(root), S.Store(out)
    age = descriptives.age_cells(st, counties, sample)
    rs.write(age, "age_cells")
    log(f"step 8 age_cells={age.height}", logf)
    _digest(rs, "age_cells")


def _step9(root, out, counties, sample, fingerprint, logf):
    st, rs = S.Store(root), S.Store(out)
    keep = rs.read("keep").filter((pl.col("policy") == "never_rebuilt") & (pl.col("lag") == 0))
    pe = rs.read("person_events", columns=["ncid", "client", "size0", "voter_tag0", "core"])
    dc = descriptives.descriptive_cells(st, pe, keep, rs.read("age_cells"))
    rs.write(dc, "descriptive_cells")
    log(f"step 9 descriptive_cells={dc.height}", logf)
    _digest(rs, "descriptive_cells")


def _step11(root, out, counties, sample, fingerprint, logf):
    st, rs = S.Store(root), S.Store(out)
    params = matcher.load_params()
    fg = matcher.matcher_graph(st)
    rm = rs.read("keep").filter(pl.col("policy") == "never_rebuilt")
    c = matcher.matcher_clients(st, rm, fg)
    src = rm.select("ncid", "k", "src")
    del rm
    W = matcher.Weights(fg, params)
    bx = matcher.best(c, fg, W)
    rs.write(bx, "matcher_best")
    fsr = matcher.matcher_records(st, c, bx, fg, src)
    rs.write(fsr, "matcher_records")
    log(f"step 11 matcher_graph={fg.height} clients={c.height} best={bx.height}", logf)
    _digest(rs, "matcher_best")
    _digest(rs, "matcher_records")


def _step12(root, out, counties, sample, fingerprint, logf):
    st, rs = S.Store(root), S.Store(out)
    parts = []
    for k in (2, 4, 6, 8):  # memory: one horizon at a time (every validity key carries its k)
        keep = (pl.scan_parquet(rs.path("keep")).filter((pl.col("k") == k) & (pl.col("lag") == 0) & pl.col("has_truth")
                                                        & pl.col("policy").is_in(["never_rebuilt", "last_known", "annual_rebuild"])).collect())
        if keep.height == 0:  # a later build has no horizon past the last snapshot
            continue
        fsr = pl.scan_parquet(rs.path("matcher_records")).filter(pl.col("k") == k).collect()
        parts.append(match_validity.validity_cells(st, keep, fsr))
        del keep, fsr
    vc = pl.concat(parts).sort(["key", "county0"])
    assert vc.select("key", "county0").is_duplicated().sum() == 0
    rs.write(vc, "validity_cells")
    log(f"step 12 validity_cells={vc.height}", logf)
    _digest(rs, "validity_cells")


# (step, function, the tables it writes; the last one marks the step complete)
def _step13(root, out, counties, sample, fingerprint, logf):
    """Whole-file exits by removal reason and the reference sensitivities, at the 2016 build (currency.py)."""
    st, rs = S.Store(root), S.Store(out)
    keep = rs.read("keep", columns=["policy", "lag", "ncid", "k", "s_dt", "has_truth", "matched", "self_in_a", "t_size", "a_size",
                                    "a_int", "f_size", "f_int", "mkey", "last_known_key"]).filter(pl.col("policy") == "never_rebuilt")
    ec = currency.exit_counts(st, keep, T0)
    rs.write(ec, "exit_counts")
    log(f"step 13 exit_counts={ec.height}", logf)
    parts = []
    for k in (2, 4, 6, 8):  # memory: one horizon at a time
        fz = (pl.scan_parquet(rs.path("keep")).filter((pl.col("k") == k) & (pl.col("policy") == "last_known") & (pl.col("t_dt") == T0)
                                                     & (pl.col("lag") == 0) & pl.col("has_truth"))
              .select("k", "ncid", "t_size", "a_size", "a_int", "self_in_a").collect())
        fsr = pl.scan_parquet(rs.path("matcher_records")).filter(pl.col("k") == k).collect()
        parts.append(currency.link_same(fz, fsr))
        del fz, fsr
    rs.write(pl.concat(parts).sort("k"), "link_same")
    log("step 13 link_same=4", logf)
    rx = st.read("ncid_reissue") if st.exists("ncid_reissue") else None
    voted = st.read("voted16") if st.exists("voted16") else None
    rf = currency.ref_sensitivity(st, keep, T0, reissue=rx, voted=voted)
    rs.write(rf, "ref_sensitivity")  # written last: marks step 13 complete
    log(f"step 13 ref_sensitivity={rf.height}", logf)


def _step14(root, out, counties, sample, fingerprint, logf):
    """Further diagnostics at the 2016 build (descriptive; no decision reads them): the matcher answer diagnostics,
    mover timing, the readdressing ablation and the mobility calibration (currency.py, calibration.py)."""
    st, rs = S.Store(root), S.Store(out)
    keep = rs.read("keep", columns=["policy", "lag", "ncid", "k", "s_dt", "has_truth", "matched", "self_in_a", "t_size", "a_size",
                                    "a_int", "f_size", "f_int", "mkey", "last_known_key"]).filter(pl.col("policy") == "never_rebuilt")
    fsr = rs.read("matcher_records", columns=["k", "ncid", "has_truth", "w", "same", "a_size", "a_int", "self_in_a", "t_size"])
    rs.write(currency.matcher_answer_diagnostics(keep, fsr), "matcher_answer_diagnostics")
    del fsr
    log("step 14 matcher_answer_diagnostics", logf)
    rs.write(currency.mover_timing(st, keep, T0), "mover_timing")
    log("step 14 mover_timing", logf)
    rs.write(currency.readdress_ablation(st, keep, T0), "readdress_ablation")
    log("step 14 readdress_ablation", logf)
    ages = (rs.read("person_events", columns=["client", "k", "age0"]).filter(pl.col("client") & (pl.col("k") == 2))
            .select(pl.col("age0").alias("age")))
    jan = [d for d in S.SNAPSHOTS if d.month == 1 and d.day == 1]
    pairs = [(a, b) for a, b in zip(jan, jan[1:]) if b.year == a.year + 1]
    acs = calibration.acs_by_age(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inputs",
                                        "acs_nc_mobility_by_age.tsv"))
    rs.write(calibration.mobility_calibration(st, pairs, acs, ages), "mobility_calibration")  # written last: marks step 14 complete
    log("step 14 mobility_calibration", logf)


STEPS = [("1", _step1, ["household_events", "person_events"]),
         ("2", _step2, ["rematch0", "keep", "bal_step2", "anchor_cells", "cells_step2"]),
         ("3", _step3, ["stored_assignment", "cells"]), ("8", _step8, ["age_cells"]), ("9", _step9, ["descriptive_cells"]),
         ("11", _step11, ["matcher_best", "matcher_records"]), ("12", _step12, ["validity_cells"]),
         ("13", _step13, ["exit_counts", "link_same", "ref_sensitivity"]),
         ("14", _step14, ["matcher_answer_diagnostics", "mover_timing", "readdress_ablation", "mobility_calibration"])]


def run(st: S.Store, out: str, counties, sample=0, seed=SEED, reps=REPS, fingerprint=False, logf=None, steps=None,
        outputs=None):
    """Stage 3 for one county list. Writes intermediate Parquet tables and the text outputs to out/. A rerun skips the
    steps whose tables exist. steps: the step names to run (default all); outputs: the text outputs to write (default all;
    [] for none)."""
    os.makedirs(out, exist_ok=True)
    rs = S.Store(out)
    man = Manifest(os.path.join(out, "manifest.json"))
    counties = sorted(int(c) for c in counties)
    man.put("run", "counties", ",".join(map(str, counties)))
    man.put("run", "sample", sample)
    man.put("run", "seed", seed)
    man.put("run", "reps", reps)
    if not config.DEFAULT:
        man.put("run", "config", {k: os.environ[k] for k in config.KEYS if k in os.environ})
    for name, fn, tables in STEPS:
        if steps is not None and name not in steps:
            continue
        if name == "2" and rs.exists("cells"):
            continue
        if not rs.exists(tables[-1]):
            isolated(fn, st.root, out, counties, sample, fingerprint, logf)
    for t in ("person_events", "household_events", "rematch0", "stored_assignment", "keep", "anchor_cells", "cells", "age_cells",
              "descriptive_cells", "matcher_best", "matcher_records", "validity_cells"):
        if not rs.exists(t):
            continue
        if not os.path.exists(os.path.join(out, "digests", t + ".json")):
            isolated(_digest_only, out, t)
        man.put("tables", t, json.load(open(os.path.join(out, "digests", t + ".json"))))
    if fingerprint and os.path.exists(os.path.join(out, "check.json")):
        chk = json.load(open(os.path.join(out, "check.json")))
        for k in ("rematch", "records"):
            man.put("check", k + "_fp", {a: str(v) for a, v in zip(("rows", "xor", "sum"), chk[k])})
    if outputs == []:
        return man

    rd = {}

    def tab(name):  # read each table once, when an output needs it
        if name not in rd:
            rd[name] = rs.read(name)
        return rd[name]
    res = {
        "stats.csv": lambda: statistics.main_stats(tab("cells"), seed, reps),
        "descriptive_stats.csv": lambda: statistics.ratio_stats(tab("descriptive_cells"), statistics.desc_derived(), seed, reps),
        "validity_stats.csv": lambda: statistics.ratio_stats(tab("validity_cells"), statistics.validity_derived(), seed, reps),
        "validity_frontier.csv": lambda: currency.validity_frontier(tab("validity_cells"), done["validity_stats.csv"]),
        "inputs_by_k.csv": lambda: currency.inputs_by_k(tab("cells")),
        "exit_counts.csv": lambda: rs.read("exit_counts"),
        "three_outcome.csv": lambda: currency.three_outcome(rs.read("exit_counts")),
        "ref_sensitivity.csv": lambda: rs.read("ref_sensitivity"),
        "anchor_currency.csv": lambda: currency.anchor_currency(
            tab("anchor_cells"),
            statistics.ratio_stats(tab("anchor_cells").filter(~pl.col("key").str.starts_with("anchor|n|")), [], seed, reps)),
        "currency_bands.csv": lambda: currency.currency_bands(done["anchor_currency.csv"]),
        "small_cluster.csv": lambda: currency.small_cluster(done["stats.csv"], len(counties)),
        "tipping_points.csv": lambda: currency.tipping_points(done["stats.csv"]),
        "tipping_two_score.csv": lambda: currency.tipping_two_score(done["stats.csv"], done["inputs_by_k.csv"]),
        "envelopes.csv": lambda: currency.envelopes(done["inputs_by_k.csv"]),
        "link_same_population.csv": lambda: currency.link_same_population(rs.read("link_same")),
        "matcher_answer_diagnostics.csv": lambda: rs.read("matcher_answer_diagnostics"),
        "mover_timing.csv": lambda: rs.read("mover_timing"),
        "readdress_ablation.csv": lambda: rs.read("readdress_ablation"),
        "mobility_calibration.csv": lambda: rs.read("mobility_calibration")}
    done = {}
    for name, f in res.items():
        if outputs is not None and name not in outputs:
            continue
        df = done[name] = f()
        h = write_csv(df, os.path.join(out, name))
        man.put("outputs", name, {"sha256": h, "rows": df.height})
        log(f"{name} rows={df.height} sha256={h}", logf)
    return man


def _run_child(env, root, out, counties, kw):
    os.environ.update(env)
    from . import config as c
    assert all(os.environ.get(k) == v for k, v in env.items()) and c.FILES == env.get("STALE_FILES", "registered")
    run(S.Store(root), out, counties, **kw)


def run_config(env: dict, st: S.Store, out: str, counties, **kw):
    """run() under a run configuration (config.py: STALE_* variables), in a new interpreter whose steps inherit it; this
    process's own configuration is untouched."""
    assert set(env) <= set(config.KEYS), env
    env = {k: str(v) for k, v in env.items()}
    # a spawned child imports config.py before _run_child runs, so it must inherit the variables at start
    saved = {k: os.environ.get(k) for k in config.KEYS}
    for k in config.KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        isolated(_run_child, env, st.root, out, list(counties), kw)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
