# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Every result of the paper from the published files, in five phases that can run separately and resume:

  tables   stages 1 and 2: the address table of every snapshot file, household tags, graphs, relabel components, age
           flags (and the later file's own components and flags), the 2016 voters and the unit-key audit counts
  runs     stage 3: every run of runs.py (a process pool here; on a Grid Engine cluster the job array of runs.py --task)
  build    the cell tables between the runs and the statistics (robustness packs, table cells, household-change panel and
           outcomes, refresh rows, curve cells, North Carolina households for the other-register comparison)
  analyze  the statistics and result files written from those cells (tables and their scores, household-change fit and
           held-out test, refresh test, robustness, later file, paper tables, build-error experiment)
  other_states  the Ohio and Washington lists (inputs/other_states.tsv): reading, household frames, scored cells, the
           section 9 evaluations and the household-change transfer; needs --other-states and the analyze phase's results

  python -m stale_assignments.reproduce --work WORK --snapshots DIR [--voter-history ZIP] [--address-points DIR]
         [--other-states DIR] [--phases tables,runs,build,analyze,other_states]

Work layout: WORK/tables (stages 1 and 2), WORK/results/results/<run> (stage 3 runs), and the result tree WORK/results
(layout.py). Each step skips when its outputs exist, so a rerun resumes. Every heavy step runs in its own interpreter
under the configuration it needs (registered files or the later file). The analyze and other_states phases write
WORK/results/OUTPUTS.sha256.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUTS = os.path.join(ROOT, "inputs")
LATER = "2026-11-03"
URL = "https://s3.amazonaws.com/dl.ncsbe.gov/data/Snapshots/"
T16 = dt.date(2016, 11, 8)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _tsv(name):
    with open(os.path.join(INPUTS, name), encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, ln.rstrip("\n").split("\t"))) for ln in f if ln.strip()]


def log(msg, logf):
    from . import pipeline
    pipeline.log(msg, logf)


# ---- configuration of a step's interpreter ------------------------------------------------------------------------------

def _child(env, fn, args):
    from . import config
    assert config.FILES == env.get("STALE_FILES", "registered")
    fn(*args)


_ENV_LOCK = __import__("threading").Lock()


def in_process(fn, *args, files="registered", threads=None):
    """fn(*args) in a new interpreter under STALE_FILES=files (and no other run variable), as pipeline.run_config. With
    threads, the interpreter's Polars threads and worker pools are limited to that many (steps running side by side).
    The interpreter takes its environment when it starts; the lock keeps steps started from several threads apart."""
    import multiprocessing
    from . import config
    env = {"STALE_FILES": files}
    if threads:
        env.update(POLARS_MAX_THREADS=str(threads), STALE_WORKERS=str(threads))
    with _ENV_LOCK:
        saved = {k: os.environ.get(k) for k in set(config.KEYS) | set(env)}
        for k in config.KEYS:
            os.environ.pop(k, None)
        os.environ.update(env)
        try:
            p = multiprocessing.get_context("spawn").Process(target=_child, args=(env, fn, args))
            p.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{getattr(fn, '__name__', fn)} failed with exit code {p.exitcode}")


def side_by_side(steps, gb_each, logf, what):
    """steps: [(name, fn, args, files)] that do not read each other's outputs. They run at the same time, as many as memory
    allows at gb_each GB, each in its own interpreter with its share of the cores. The first failure is raised."""
    from concurrent.futures import ThreadPoolExecutor
    from . import parallel
    if not steps:
        return
    n = parallel.jobs_for(gb_each, most=len(steps))
    threads = max(1, parallel.workers() // n)

    def one(step):
        name, fn, args, files = step
        t = time.time()
        in_process(fn, *args, files=files, threads=threads if n > 1 else None)
        log(f"{what} {name} {time.time() - t:.0f}s", logf)
    with ThreadPoolExecutor(n) as ex:
        for f in [ex.submit(one, st) for st in steps]:
            f.result()


# ---- phase 1: stages 1 and 2 -----------------------------------------------------------------------------------------

def _check_file(p, want, allow, logf, what):
    size, sha = os.path.getsize(p), sha256_file(p)
    if want is None:
        log(f"{what}: no recorded size and SHA-256; {os.path.basename(p)} bytes={size} sha256={sha}", logf)
    elif size != int(want["bytes"]) or sha != want["sha256"]:
        log(f"CHANGED {os.path.basename(p)} (size or SHA-256 differs from inputs/)", logf)
        if not allow:
            sys.exit(f"{what} changed; see {logf}")
    return size, sha


GB_PER_BUILD = 6      # peak memory of one snapshot build (decode, address table, household tags), with margin
BUILDS_MOST = 8       # downloads go one at a time, so more builds than this at once would only wait for files


def _stage1_task(task):
    """One snapshot file -> its address table and (registered files) its household tags, in a worker process."""
    p, tables, ds, tmp = task
    from . import pipeline, prepare, snapshots, store as S
    st, d = S.Store(tables), dt.date.fromisoformat(ds)
    out = st.path("addr", d)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rep = snapshots.build_snapshot(p, out + ".build", tmp)
    os.replace(out + ".build", out)
    addr = st.read("addr", d)
    digest = pipeline.table_digest(addr)
    tags = None
    if ds != LATER and not st.exists("tags", d):
        t = prepare.household_tags(addr)
        st.write(t, "tags", d)
        tags = t.height
    return rep, digest, tags


def tables_phase(a, logf):
    import collections
    from concurrent.futures import ThreadPoolExecutor
    import polars as pl
    from . import parallel, pipeline, snapshots, unit_key, voter_history
    from . import store as S
    st = S.Store(os.path.join(a.work, "tables"))
    prep = pipeline.Manifest(os.path.join(a.work, "results", "prep_manifest.json"))
    files = _tsv("snapshots.tsv")
    later = {r["date"]: r for r in _tsv("later_file.tsv")} if os.path.exists(os.path.join(INPUTS, "later_file.tsv")) else {}
    files.append(later.get(LATER) or {"date": LATER, "file": f"VR_Snapshot_{LATER.replace('-', '')}.zip"})
    built = lambda r: st.exists("addr", dt.date.fromisoformat(r["date"])) and r["date"] in prep.d.get("stage1_input", {})
    todo = [r for r in files if not built(r)]
    jobs = parallel.jobs_for(GB_PER_BUILD, reserve_gb=GB_PER_BUILD, most=min(BUILDS_MOST, max(1, len(todo))))
    if todo:
        log(f"stage 1: {len(todo)} files to build, {jobs} at a time, {max(1, parallel.workers() // jobs)} threads each", logf)

    def fetch(r):  # in the download thread: one file at a time, in date order
        p = os.path.join(a.snapshots, r["file"])
        if not os.path.exists(p) and a.fetch:
            os.makedirs(a.snapshots, exist_ok=True)
            log(f"download {r['file']}", logf)
            urllib.request.urlretrieve(URL + r["file"], p + ".part")
            os.replace(p + ".part", p)
        if not os.path.exists(p):
            sys.exit(f"missing input {p} (INPUT_DATA.md; --fetch downloads it)")
        size, sha = _check_file(p, r if "sha256" in r else None, a.allow_changed_inputs, logf, "snapshot file")
        return p, size, sha, time.time(), ex.submit(_stage1_task, (p, st.root, r["date"], os.path.join(a.work, "tmp")))

    # Builds run side by side; their results are recorded, and the date-ordered part of stage 2 applied, in date order.
    with parallel.executor(jobs) as ex, ThreadPoolExecutor(1) as dl:
        it = iter(todo)
        pend = collections.deque()

        def top_up():
            while len(pend) < jobs + 1:
                r = next(it, None)
                if r is None:
                    return
                pend.append((r, dl.submit(fetch, r)))
        top_up()
        for r in files:
            d = dt.date.fromisoformat(r["date"])
            if pend and pend[0][0] is r:
                _, f = pend.popleft()
                p, size, sha, t, bf = f.result()
                rep, digest, tags = bf.result()
                prep.put("stage1_lines", r["date"], {k: v for k, v in rep.items() if k != "quote_lines"})
                prep.put("addr", r["date"], digest)
                prep.put("stage1_input", r["date"], {"file": r["file"], "bytes": size, "sha256": sha})
                log(f"stage 1 {d} rows={rep['rows']} {time.time() - t:.0f}s", logf)
                if tags is not None:
                    log(f"tags {d} rows={tags}", logf)
                if a.discard_inputs:
                    os.remove(p)
                top_up()
            if r["date"] != LATER:
                pipeline.stage2_date(st, d, logf)
    if a.address_points:
        _unit_counts(st, prep, a, logf)
    pipeline.stage2(st, prep, logf)
    for d in S.SNAPSHOTS:
        for tbl in ("tags", "graph"):
            if d.isoformat() not in prep.d.get(tbl, {}):
                prep.put(tbl, d.isoformat(), pipeline.table_digest(st.read(tbl, d)))
    if a.voter_history and not st.exists("voted16"):
        want = _tsv("voter_history.tsv")[0]
        size, sha = _check_file(a.voter_history, want, a.allow_changed_inputs, logf, "voter history file")
        v = voter_history.voted(a.voter_history, "11/08/2016")
        st.write(v, "voted16")
        prep.put("voter_history", "input", {"file": os.path.basename(a.voter_history), "bytes": size, "sha256": sha})
        prep.put("voter_history", "voted16", pipeline.table_digest(v))
        log(f"voter history voted16={v.height}", logf)
    elif not st.exists("voted16"):
        log("no --voter-history: tlag in ref_sensitivity.csv is left blank", logf)
    del pl, unit_key


def _unit_counts(st, prep, a, logf):
    import polars as pl
    from . import pipeline, unit_key
    sums = os.path.join(a.address_points, "SHA256SUMS")
    want = dict(ln.split("  ", 1)[::-1] for ln in open(sums, encoding="utf-8").read().splitlines() if ln)
    have = sorted(f for f in os.listdir(a.address_points) if f.startswith("ap_") and f.endswith(".csv.gz"))
    if have != sorted(want) or any(sha256_file(os.path.join(a.address_points, f)) != want[f] for f in have):
        sys.exit("address point files do not match their SHA256SUMS")
    pin = _tsv("address_points.tsv")[0]
    if sha256_file(sums) != pin["sha256sums_sha256"] and not a.allow_changed_inputs:
        sys.exit("address points changed (SHA256SUMS differs from inputs/address_points.tsv)")
    todo = [d for d in unit_key.AUDIT_DATES if not st.exists("unit_counts", dt.date.fromisoformat(d))]
    if not todo:
        return
    ap = unit_key.load_address_points(a.address_points)
    cols = ["ncid", "county_id", "is_ai", "house_num", "half_code", "street_dir", "street_name", "street_type_cd",
            "street_sufx_cd", "unit", "zip5", "bldg_key"]
    for ds in todo:
        d = dt.date.fromisoformat(ds)
        if not set(cols) <= set(pl.read_parquet_schema(st.path("addr", d))):
            sys.exit(f"the {ds} address table was already slimmed by stage 2; run --address-points in a new work directory")
        c = unit_key.unit_counts(st.read("addr", d, columns=cols), ap, d)
        st.write(c, "unit_counts", d)
        prep.put("unit_counts", ds, pipeline.table_digest(c))
        log(f"unit-key audit {ds} buildings={c['bld_all'].sum()}", logf)


# ---- phase 2: the runs ----------------------------------------------------------------------------------------------------

def runs_phase(a, logf):
    from . import parallel, runs
    out = os.path.join(a.work, "results", "results")
    todo = [r for r in runs.RUNS if not a.only or r["name"] in a.only.split(",")]
    jobs = max(1, min(a.jobs or runs.default_jobs(), len(todo)))
    per_run = max(1, parallel.workers() // jobs)
    log(f"runs {len(todo)} at {jobs} a time, {per_run} workers each", logf)
    for name in parallel.pmap(runs._task, [(r, os.path.join(a.work, "tables"), out, per_run) for r in todo], n=jobs):
        log(f"run {name} done", logf)


# ---- phase 3: cell tables -------------------------------------------------------------------------------------------------

def _split():
    from . import runs
    return {c: s for s in ("development", "confirmation", "held_out") for c in runs.counties(s)}


def _rm0(R, run):
    import polars as pl
    return pl.scan_parquet(os.path.join(R, "results", run, "rematch0.parquet"))


def _read(R, run, t, **kw):
    import polars as pl
    return pl.read_parquet(os.path.join(R, "results", run, f"{t}.parquet"), **kw)


def _sorted(df):
    return df.sort(df.columns, nulls_last=True)


def _write(df, path):
    from .robustness.outputs import write_frame
    write_frame(df, path)


def _write_gz(df, path):
    from .files import gz_text_writer
    fmt = lambda v: "" if v is None else ("true" if v else "false") if isinstance(v, bool) else str(v)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gz_text_writer(path + ".part") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(df.columns)
        for r in df.iter_rows():
            w.writerow([fmt(v) for v in r])
    os.replace(path + ".part", path)


def b_county_split(T, R):
    from . import layout as L
    rows = sorted(((s, c) for c, s in _split().items()))
    os.makedirs(L.robustness(R), exist_ok=True)
    with open(L.robustness(R, "county_split.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["county_id", "split"])
        w.writerows([c, s] for s, c in rows)


def b_stale_listing(T, R, run, suffix):
    from . import layout as L, store as S
    from .robustness import data as D
    st = S.Store(T)
    rws = D.stale_rows(_rm0(R, run), _read(R, run, "matcher_records"))
    cells = _sorted(D.stale_cells(st, rws))
    del rws
    _write(cells, L.robustness(R, f"stale_listing_rows{suffix}.csv"))
    _write(_sorted(D.county_pack(cells)), L.robustness(R, f"stale_listing_county_pack{suffix}.csv"))


def b_confirmation_counts(T, R):
    """Graded classes, matcher by recorded-alone stratum, size split, age band and client universe, mobility reasons."""
    import polars as pl
    from . import layout as L, runs, store as S
    from .robustness import data as D, outputs as O
    st = S.Store(T)
    rm, fsr = _rm0(R, "confirmation"), pl.scan_parquet(os.path.join(R, "results", "confirmation", "matcher_records.parquet"))
    O.graded_csv(R, D.graded(rm))
    _write(D.matcher_alone_cells(fsr, rm), L.robustness(R, "probabilistic_matcher_alone_cells.csv"))
    _write(D.matcher_paired_county_pack(fsr, rm), L.robustness(R, "matcher_paired_county_pack.csv"))
    O.size_split_csv(R, [("confirmation", D.size_split(rm)), ("held_out", D.size_split(_rm0(R, "held_out")))])
    pe = _read(R, "confirmation", "person_events")
    _write(D.age_band(rm, pe), L.robustness(R, "age_band.csv"))
    O.universe(R, D.universe_counts(st, runs.counties("confirmation")), os.path.join(INPUTS, "acs_nc_mobility_by_age.tsv"))
    ages = pe.filter(pl.col("client") & (pl.col("k") == 2)).select(pl.col("age0").alias("age"))
    jan = [d for d in S.SNAPSHOTS if d.month == 1 and d.day == 1]
    pairs = [(x, y) for x, y in zip(jan, jan[1:]) if y.year == x.year + 1]
    _write(D.mobility_reasons(st, pairs, ages), L.robustness(R, "mobility_reasons.csv"))


ANCHOR_SOURCES = ([("confirmation", "confirmation", None, None), ("development", "development", None, None),
                   ("later_file_confirmation_b2016", "later_file_confirmation_b2016_k10", (2016,), (2, 4, 6, 8)),
                   ("confirmation_k10", "later_file_confirmation_b2016_k10", (2016,), None)]
                  + [("later_file_confirmation", f"later_file_confirmation_b{y}", (y,), None) for y in (2018, 2020, 2022, 2024)]
                  + [("development_k10", "later_file_development_b2016_k10", (2016,), None)]
                  + [("later_file_development", f"later_file_development_b{y}", (y,), None) for y in (2018, 2020, 2022, 2024)])


def b_anchor_pack(T, R):
    from . import layout as L
    from .robustness import data as D
    _write(D.anchor_currency_pack([(s, _rm0(R, run), ys, ks) for s, run, ys, ks in ANCHOR_SOURCES]),
           L.robustness(R, "anchor_currency_pack.csv"))


def b_build_size(T, R):
    from . import currency, layout as L
    currency.build_size_csv([("development", currency.build_size_counts(_rm0(R, "development"))),
                             ("confirmation", currency.build_size_counts(_rm0(R, "confirmation")))],
                            L.results(R, "currency_by_build_size.csv"))


def b_development_cells(T, R):
    import polars as pl
    from . import assignment_tables as E, layout as L, store as S
    st = S.Store(T)
    rm = _rm0(R, "development")
    parts = [E.fine_cells(r) for r in E.rows_by_build(st, rm)]
    _write(pl.concat(parts).sort(["t_dt", "k", "county0", "size6", "bt", "ab", "ny", "mb", "gb"]),
           L.table_cells(R, "development_fine_cells.csv"))
    _write(E.stale_probe(st, rm), L.tables(R, "stale_probe_development.csv"))


def b_held_out_cells(T, R):
    import polars as pl
    from . import assignment_tables as E, layout as L, store as S
    r = E.held_out_rows(S.Store(T), _rm0(R, "held_out"))
    _write(E.fine_cells(r), L.table_cells(R, "held_out_cells.csv"))
    _write(E.fine_cells(r, "e_lookahead1", pl.col("has_lookahead1")), L.table_cells(R, "held_out_lookahead_one_election_cells.csv"))


def b_later_file_cells(T, R):
    import polars as pl
    from . import assignment_tables as E, layout as L, store as S
    r = E.rows(S.Store(T), _rm0(R, "later_file_all_b2018"), pl.col("s_dt") == dt.date.fromisoformat(LATER))
    _write(E.fine_cells(r), L.table_cells(R, "later_file_cells.csv"))


def b_household_change(T, R):
    from . import layout as L, runs, store as S
    from .household_change import anchors as A
    st = S.Store(T)
    A.write_snapshots(L.household_change(R, "panel", "snapshots.csv"))
    _write_gz(A.transitions(st, runs.counties("development")), L.household_change(R, "panel", "transitions.csv.gz"))
    oc = A.score_cells(st, runs.counties("held_out")).sort(["t_dt", "cell", "bt", "county0", "k"])
    _write_gz(oc.with_columns(oc["t_dt"].cast(str)), L.household_change(R, "held_out", "outcomes.csv.gz"))
    _write(A.stayer_xtab(st, _rm0(R, "held_out"), runs.counties("held_out")),
           L.household_change(R, "held_out", "stayer_xtab_2016.csv"))


def b_refresh_rows(T, R):
    from . import layout as L, store as S
    from .refresh import rows as RF
    os.makedirs(L.refresh(R), exist_ok=True)
    RF.rows(S.Store(T), _rm0(R, "later_file_all_b2018")).write_parquet(L.refresh(R, "rows.parquet"))


def b_curve_cells(T, R):
    from . import layout as L
    from .refresh import rows as RF
    _write(RF.curve_cells(_rm0(R, "curve")), L.refresh(R, "curve_cells.csv"))


def b_north_carolina(T, R):
    """North Carolina scored cells for the other-register comparison and the build-error cells."""
    from . import layout as L, other_states as OS, store as S
    import polars as pl
    st = S.Store(T)
    pairs = [(dt.date.fromisoformat(r["t"]), dt.date.fromisoformat(r["s"]))
             for r in csv.DictReader(open(os.path.join(HERE, "data", "north_carolina_pairs.csv"), encoding="utf-8"))]
    hh, parts = {}, []
    for d in sorted({d for p in pairs for d in p}):
        hh[d] = OS.nc_households(st, d)
    for t, s in pairs:
        parts.append(OS.score_cells("NC", t, s, hh[t], hh[s], OS.nc_minors(st, t)))
    os.makedirs(L.other_states(R), exist_ok=True)
    pl.concat(parts).write_parquet(L.other_states(R, "north_carolina_cells.parquet"))
    del hh, parts
    OS.build_error_counties(OS.build_error(st), L.build_error(R, "counties.csv"))


def b_unit_key(T, R):
    import polars as pl
    from . import layout as L, pipeline, store as S, unit_key
    st = S.Store(T)
    ds = [dt.date.fromisoformat(d) for d in unit_key.AUDIT_DATES]
    if not all(st.exists("unit_counts", d) for d in ds):
        return
    uc = pl.concat([st.read("unit_counts", d) for d in ds]).sort("snapshot_dt", "county_id")
    pipeline.write_csv(uc, L.results(R, "unit_key_county.csv"))
    pipeline.write_csv(unit_key.statewide(uc, _split()), L.results(R, "unit_key_statewide.csv"))


def _builders(R):
    from . import layout as L
    rb = lambda *x: L.robustness(R, *x)
    return [  # (name, outputs, files, function, args)
        ("county split", [rb("county_split.csv")], "registered", b_county_split, ()),
        ("stale listing, confirmation", [rb("stale_listing_county_pack.csv")], "registered", b_stale_listing, ("confirmation", "")),
        ("stale listing, held out", [rb("stale_listing_county_pack_held_out.csv")], "registered", b_stale_listing,
         ("held_out", "_held_out")),
        ("confirmation counts", [rb("mobility_reasons.csv")], "registered", b_confirmation_counts, ()),
        ("anchor currency pack", [rb("anchor_currency_pack.csv")], "registered", b_anchor_pack, ()),
        ("currency by build size", [L.results(R, "currency_by_build_size.csv")], "registered", b_build_size, ()),
        ("development cells", [L.tables(R, "stale_probe_development.csv")], "registered", b_development_cells, ()),
        ("held-out cells", [L.table_cells(R, "held_out_lookahead_one_election_cells.csv")], "registered", b_held_out_cells, ()),
        ("later-file cells", [L.table_cells(R, "later_file_cells.csv")], "later_file", b_later_file_cells, ()),
        ("household change", [L.household_change(R, "held_out", "stayer_xtab_2016.csv")], "registered", b_household_change, ()),
        ("refresh rows", [L.refresh(R, "rows.parquet")], "later_file", b_refresh_rows, ()),
        ("curve cells", [L.refresh(R, "curve_cells.csv")], "registered", b_curve_cells, ()),
        ("North Carolina households", [L.build_error(R, "counties.csv")], "later_file", b_north_carolina, ()),
        ("unit-key audit", [L.results(R, "unit_key_statewide.csv")], "registered", b_unit_key, ()),
    ]


GB_PER_STEP = 10   # peak memory of one build or analyze step on the statewide tables, with margin


def build_phase(a, logf):
    """The builders read the tables and the stage 3 runs and write separate files, so they all run side by side."""
    T, R = os.path.join(a.work, "tables"), os.path.join(a.work, "results")
    side_by_side([(name, fn, (T, R, *args), files) for name, outs, files, fn, args in _builders(R)
                  if not all(os.path.exists(p) for p in outs)], GB_PER_STEP, logf, "build")


# ---- phase 4: statistics and result files ---------------------------------------------------------------------------------

def a_stats_tsv(T, R):
    """stats.tsv and validity_stats.tsv beside each run's CSV (tab-separated, the same values)."""
    from . import layout as L
    for run in sorted(os.listdir(L.results(R))):
        for name in ("stats", "validity_stats", "descriptive_stats"):
            p = L.results(R, run, f"{name}.csv")
            if os.path.isfile(p):
                rows = list(csv.reader(open(p, encoding="utf-8")))
                with open(L.results(R, run, f"{name}.tsv"), "w", newline="", encoding="utf-8") as f:
                    f.write("".join("\t".join(r) + "\n" for r in rows))


def a_tables(T, R):
    from . import layout as L, table_scores as TS
    TS.fit_tables(L.table_cells(R, "development_fine_cells.csv"), L.tables(R))
    for ref, pat, tabs in TS.REPORTS:
        cells = {"registered": "held_out_cells.csv", "lookahead_one_election": "held_out_lookahead_one_election_cells.csv",
                 "later_file": "later_file_cells.csv"}[ref]
        for tab in tabs:
            with open(L.tables(R, pat.format(tab)), "w", encoding="utf-8", newline="") as f:
                f.write(TS.score(L.tables(R, f"{tab}.csv"), L.table_cells(R, cells), tab))
    TS.table_results(L.tables(R), L.tables(R))


def a_later_file(T, R):
    import polars as pl
    from . import later_file as LF, layout as L
    from .pipeline import REPS, SEED
    for split in ("confirmation", "development"):
        run = lambda y: L.results(R, f"later_file_{split}_b{y}")
        LF.anchor_rows([(os.path.join(L.results(R, f"later_file_{split}_b2016_k10"), "anchor_currency.csv"), [2016])]
                       + [(os.path.join(run(y), "anchor_currency.csv"), [y]) for y in (2018, 2020, 2022, 2024)],
                       L.results(R, "later_file", f"anchor_currency_{split}.csv"))
        parts = [LF.validity_rows(pl.read_parquet(os.path.join(run(y), "validity_cells.parquet")), y, 2026 - y, SEED, REPS)
                 for y in (2018, 2020, 2022, 2024)]
        LF.validity_exact(parts, L.results(R, "later_file", f"validity_exact_{split}.csv"))
        k10 = L.results(R, f"later_file_{split}_b2016_k10")
        LF.ten_year(pl.read_parquet(os.path.join(k10, "cells.parquet")), pl.read_parquet(os.path.join(k10, "anchor_cells.parquet")),
                    SEED, REPS, L.results(R, "later_file", f"ten_year_2016_{split}.csv"),
                    L.results(R, "later_file", f"ten_year_2016_{split}_n.csv"))
    LF.gap_rows(os.path.join(L.results(R, "later_file_confirmation_b2018"), "stats.tsv"),
                L.results(R, "later_file", "gap_2018_confirmation.csv"))


def a_robustness(T, R):
    from . import layout as L
    from .robustness import outputs as O
    for sfx in ("", "_held_out"):
        O.summary(L.robustness(R, f"stale_listing_rows{sfx}.csv"), L.robustness(R, f"stale_listing_summary{sfx}.csv"))
        O.intervals(L.robustness(R, f"stale_listing_county_pack{sfx}.csv"), L.robustness(R, f"stale_listing_intervals{sfx}.csv"))
    O.compare(R)
    O.w20_alone(R)
    O.bracket(R)
    O.paired(R)
    O.anchor_full(R)
    O.matcher_alone(R)
    O.forecast_bands(R)
    O.skill(R)
    O.baselines(R)


def a_household_change(T, R):
    from . import layout as L
    from .household_change import fit as F, held_out as H, panel
    P = L.household_change(R, "panel")
    fit_path = L.household_change(R, "fit.json")
    if not os.path.exists(fit_path):
        F.fit(os.path.join(P, "transitions.csv.gz"), os.path.join(P, "snapshots.csv"), fit_path)
    out = L.household_change(R, "held_out")
    H.score(fit_path, os.path.join(P, "snapshots.csv"), os.path.join(out, "outcomes.csv.gz"), L.table_cells(R, "held_out_cells.csv"),
            L.table_cells(R, "development_fine_cells.csv"), L.tables(R), out)
    H.stayer_definitions(out, L.robustness(R, "stale_listing_summary_held_out.csv"))
    H.step_check(fit_path, panel.load(os.path.join(P, "transitions.csv.gz"), os.path.join(P, "snapshots.csv"), F.BUILDS),
                 L.household_change(R))


def a_refresh(T, R):
    import polars as pl
    from . import layout as L
    from .refresh import rows as RF, statistics as RS
    rw = pl.read_parquet(L.refresh(R, "rows.parquet"))
    pr = RF.predict(rw, L.tables(R), L.table_cells(R, "development_fine_cells.csv"))
    split = _split()
    RF.write_agg(RF.aggregate(rw, pr, split), L.refresh(R, "aggregate.csv"))
    RF.diag_posthoc(rw, pr, split, L.refresh(R, "refresh_diag_posthoc.csv"))
    RS.run(L.refresh(R, "aggregate.csv"), L.refresh(R))
    RS.curve_stats(L.refresh(R, "curve_cells.csv"), L.refresh(R, "curve_stats.csv"))


def a_paper(T, R):
    from . import currency, layout as L, paper_tables, pipeline
    paper_tables.write_tables(L.results(R))
    pipeline.write_csv(currency.acs_movers(os.path.join(INPUTS, "acs_nc_movers.tsv")), L.results(R, "acs_movers.csv"))


def a_north_carolina(T, R):
    import polars as pl
    from . import layout as L, other_states as OS
    cells = pl.read_parquet(L.other_states(R, "north_carolina_cells.parquet"))
    OS.nc_pair_sums(cells, L.other_states(R, "north_carolina_pair_sums.csv"))
    OS.listed_share(cells, L.other_states(R, "listed_share_north_carolina.csv"))
    OS.nc_tables_summary(OS.tables(cells, L.tables(R), L.table_cells(R, "development_fine_cells.csv")),
                         L.other_states(R, "north_carolina_tables.csv"))
    OS.build_error_summary(L.build_error(R, "counties.csv"), L.build_error(R, "summary.csv"))
    OS.exact_build_2018(L.build_error(R, "counties.csv"), L.robustness(R, "table_skill_cells.csv"),
                        L.build_error(R, "exact_build_2018.csv"))


def _analyses(R):
    """Waves of steps: a step reads only the outputs of earlier waves, so the steps of one wave run side by side."""
    from . import layout as L
    return [
        [("stats tsv", [], a_stats_tsv)],
        [("currency tables", [L.tables(R, "held_out_metrics.csv")], a_tables),             # stats.tsv: later file
         ("later file", [L.results(R, "later_file", "gap_2018_confirmation.csv")], a_later_file)],
        [("robustness", [L.robustness(R, "table_baselines.csv")], a_robustness),           # tables, later file
         ("refresh", [L.refresh(R, "curve_stats.csv")], a_refresh),                        # tables
         ("paper tables", [L.results(R, "acs_movers.csv")], a_paper)],                     # later file
        [("household change", [L.household_change(R, "step_check.csv")], a_household_change),  # tables, robustness
         ("North Carolina comparison and build error", [L.build_error(R, "exact_build_2018.csv")], a_north_carolina)],
    ]


def analyze_phase(a, logf):
    T, R = os.path.join(a.work, "tables"), os.path.join(a.work, "results")
    for wave in _analyses(R):
        side_by_side([(name, fn, (T, R), "registered") for name, outs, fn in wave
                      if not (outs and all(os.path.exists(p) for p in outs))], GB_PER_STEP, logf, "analyze")
    write_output_list(R)


def write_output_list(R):
    """Publish the released files (release_files.publish), then write OUTPUTS.sha256: one line per text result file
    (CSV, TSV, TXT, JSON, CSV.GZ), sorted by path. A file that is published under a descriptive name is listed under
    that name only."""
    from . import release_files
    release_files.publish(R)
    lines = []
    for root, _, files in os.walk(R):
        for fn in sorted(files):
            if fn.endswith((".csv", ".tsv", ".txt", ".json", ".csv.gz")) and fn not in ("OUTPUTS.sha256", "manifest.json",
                                                                                     "prep_manifest.json"):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, R).replace(os.sep, '/')
                if release_files.listed_path(rel):
                    lines.append(f"{sha256_file(p)}  {rel}")
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(R, "OUTPUTS.sha256"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ---- other registers: Ohio and Washington ------------------------------------------------------------------------------

def _write_csv(df, path):
    import polars as pl
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.with_columns([pl.col(c).cast(pl.String) for c, t in df.schema.items() if t == pl.Date]).write_csv(path, line_terminator="\n")


def _os_date_task(task):
    """One state and date: raw parts -> normalized rows, household frame and fuzzy address keys (worker process)."""
    T, st, d, raws = task
    import polars as pl
    from . import other_registers as OR
    raw = pl.concat([pl.read_parquet(x) for x in raws])
    persons = OR.normalize(raw, st, d)
    persons.write_parquet(os.path.join(T, f"persons_{st}_{d}.parquet"))
    h = OR.households(persons)
    OR.fuzzy_keys(persons).write_parquet(os.path.join(T, f"fuzzy_{st}_{d}.parquet"))
    p = os.path.join(T, f"households_{st}_{d}.parquet")
    h.write_parquet(p + ".part")
    os.replace(p + ".part", p)
    for x in raws:
        os.remove(x)
    return raw.height, persons.height, h.height, int(h["trackable"].sum()), int((h["trackable"] & h["minor"].fill_null(False)).sum())


def _os_pair_task(task):
    """One pair: identifier cells, crosswalk cells (when the pair has them) and transfer cells (worker process)."""
    T, st, t, s, xw = task
    import polars as pl
    from . import other_registers as OR, other_states as OS
    rd = lambda kind, d: pl.read_parquet(os.path.join(T, f"{kind}_{st}_{d}.parquet"))
    D = dt.date.fromisoformat
    ht, hs = rd("households", t), rd("households", s)
    ex = OS.minors(ht)
    cells = OS.score_cells(st, D(t), D(s), ht, hs, ex)
    xcells = None
    if xw:  # the crosswalk replaces vid, so its clients under 18 are found on the crosswalked frame
        xt = OR.with_crosswalk(ht, rd("xwkeys", t))
        xcells = OS.score_cells(st, D(t), D(s), xt, OR.with_crosswalk(hs, rd("xwkeys", s)), OS.minors(xt))
        del xt
    tc = OR.transfer_cells(st, t, s, ht, hs, rd("fuzzy", t), rd("fuzzy", s), ex)
    return cells, xcells, tc


def other_states_phase(a, logf):
    """Read the Ohio and Washington lists (inputs/other_states.tsv), build their household frames, score every pair and
    write the section 9 results: evaluation_identifier.csv, evaluation_crosswalk.csv, listed_share.csv and the transfer.
    Files are read side by side, dates are built side by side, and pairs are scored side by side; results are merged in
    pair order."""
    import polars as pl
    from . import layout as L, other_registers as OR, other_states as OS, parallel, pipeline
    from .constants import TRANSFER_BOOTSTRAP_SEED
    from .household_change import transfer as TR
    R = os.path.join(a.work, "results")
    T = os.path.join(a.work, "tables", "other_states")
    os.makedirs(T, exist_ok=True)
    prep = pipeline.Manifest(os.path.join(R, "prep_manifest.json"))
    files = _tsv("other_states.tsv")
    dates = sorted({(r["state"], r["date"]) for r in files})
    hh_path = lambda st, d, kind="households": os.path.join(T, f"{kind}_{st}_{d}.parquet")
    todo = [(st, d) for st, d in dates if not (os.path.exists(hh_path(st, d)) and os.path.exists(hh_path(st, d, "fuzzy")))]
    if todo:
        tasks = []
        for r in files:
            if (r["state"], r["date"]) not in todo:
                continue
            p = os.path.join(a.other_states, r["file"])
            if not os.path.exists(p):
                sys.exit(f"missing input {p} (INPUT_DATA.md)")
            size, sha = _check_file(p, r, a.allow_changed_inputs, logf, "other-state file")
            prep.put("other_states_input", r["file"], {"bytes": size, "sha256": sha})
            tasks.append((p, os.path.join(T, "raw_" + r["file"].replace("/", "_") + ".parquet"), r["state"], r["date"]))
        t0 = time.time()
        for (p, out, st, d), c in zip(tasks, parallel.pmap(OR.read_file, tasks, n=parallel.jobs_for(GB_PER_OTHER_FILE, most=len(tasks)))):
            prep.put("other_states_read", f"{st} {d} {c['file']}", c)
            log(f"read {st} {d} {c['file']}: lines={c['lines']} skipped={c['bad']} header_rows={c['header_rows']} rows={c['rows']}", logf)
        log(f"read {len(tasks)} files {time.time() - t0:.0f}s", logf)
        dt_tasks = [(T, st, d, [out for (p, out, s2, d2) in tasks if (s2, d2) == (st, d)]) for st, d in todo]
        for (_, st, d, _), (nr, npn, nh, ntr, nmi) in zip(dt_tasks, parallel.pmap(_os_date_task, dt_tasks,
                                                                            n=parallel.jobs_for(GB_PER_OTHER_DATE, most=len(dt_tasks)),
                                                                            share=True)):
            log(f"{st} {d}: rows={nr} persons={npn} listed_with_address={nh} trackable={ntr} trackable_under_18={nmi}", logf)
    for st in sorted({k[0] for k in dates}):
        keys = OR.crosswalk_keys({d: pl.read_parquet(hh_path(st, d, "persons")) for s2, d in dates if s2 == st})
        for d, f in keys.items():
            f.write_parquet(hh_path(st, d, "xwkeys"))
    xw = lambda t, s: "2018-06-30" not in (t, s)   # the crosswalk pairs: the long pairs that do not use 2018
    pt = [(T, st, t, s, xw(t, s)) for st, t, s in OR.PAIRS]
    res = parallel.pmap(_os_pair_task, pt, n=parallel.jobs_for(GB_PER_OTHER_PAIR, most=len(pt)), share=True)
    cells = pl.concat([r[0] for r in res])
    xcells = pl.concat([r[1] for r in res if r[1] is not None])
    tc = pl.concat([r[2] for r in res])
    D = dt.date.fromisoformat
    cells.write_parquet(L.other_states(R, "state_cells.parquet"))
    _write_csv(OS.county_sums(cells), L.other_states(R, "state_county_sums_identifier.csv"))
    _write_csv(OS.county_sums(xcells), L.other_states(R, "state_county_sums_crosswalk.csv"))
    long = cells.filter(pl.col("s") != D("2018-06-30"))
    tabs = OS.county_briers(OS.tables(long, L.tables(R), L.table_cells(R, "development_fine_cells.csv")))
    _write_csv(pl.DataFrame(tabs), L.other_states(R, "state_county_briers.csv"))
    OS.write_evaluation(OS.evaluate(L.other_states(R, "state_county_sums_identifier.csv"), L.other_states(R, "state_county_briers.csv")),
                        L.other_states(R, "evaluation_identifier.csv"))
    OS.write_evaluation(OS.evaluate(L.other_states(R, "state_county_sums_crosswalk.csv")), L.other_states(R, "evaluation_crosswalk.csv"))
    nc = pl.read_parquet(L.other_states(R, "north_carolina_cells.parquet"))
    OS.listed_share(pl.concat([nc, cells]), L.other_states(R, "listed_share.csv"))  # all four states' rows, one file
    log("other states: cells, crosswalk, evaluations, listed share", logf)
    out = L.household_change(R, "transfer")
    os.makedirs(out, exist_ok=True)
    _write_csv(OR.retention(tc), os.path.join(out, "retention.csv"))
    _write_gz(OR.transfer_outcomes(tc), os.path.join(out, "outcomes.csv.gz"))
    _write_gz(OR.transfer_table_cells(tc), os.path.join(out, "table_cells.csv.gz"))
    fit = L.household_change(R, "fit.json")
    TR.run(fit, os.path.join(out, "retention.csv"), out, log=lambda m: log(m, logf))
    TR.score(out, os.path.join(out, "outcomes.csv.gz"), os.path.join(out, "table_cells.csv.gz"), L.tables(R),
             os.path.join(out, "retention.csv"), out, TRANSFER_BOOTSTRAP_SEED)
    write_output_list(R)
    log("other states: transfer done", logf)


GB_PER_OTHER_FILE = 3    # reading one list file (raw columns buffered in chunks)
GB_PER_OTHER_DATE = 8    # normalizing one state's list at one date
GB_PER_OTHER_PAIR = 8    # scoring one pair (two household frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--snapshots", help="directory with the 41 VR_Snapshot_YYYYMMDD.zip files (INPUT_DATA.md)")
    ap.add_argument("--voter-history", help="ncvhis_Statewide.zip (INPUT_DATA.md); without it tlag in ref_sensitivity.csv is blank")
    ap.add_argument("--address-points", help="directory written by fetch_address_points.py; without it no unit-key audit")
    ap.add_argument("--phases", default="tables,runs,build,analyze,other_states")
    ap.add_argument("--only", help="runs phase: comma-separated run names")
    ap.add_argument("--jobs", type=int, help="runs phase: runs at the same time (default: memory / 12 GB)")
    ap.add_argument("--allow-changed-inputs", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="download a missing snapshot file from the NCSBE address in INPUT_DATA.md")
    ap.add_argument("--other-states", help="other_states phase: directory with the Ohio and Washington lists (INPUT_DATA.md)")
    ap.add_argument("--discard-inputs", action="store_true", help="delete each snapshot file once its address table is built")
    a = ap.parse_args()
    a.work = os.path.abspath(a.work)
    os.makedirs(os.path.join(a.work, "results"), exist_ok=True)
    import fcntl
    lock = open(os.path.join(a.work, "run.lock"), "w")  # held until this process ends; one run per work directory
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another run is using {a.work}; stop it first")
    logf = os.path.join(a.work, "run.log")
    import polars as pl
    log(f"start python={platform.python_version()} polars={pl.__version__} platform={platform.platform()} "
        f"machine={platform.machine()} phases={a.phases}", logf)
    ph = a.phases.split(",")
    if "tables" in ph:
        if not a.snapshots:
            sys.exit("--snapshots is needed for the tables phase")
        tables_phase(a, logf)
    if "runs" in ph:
        runs_phase(a, logf)
    if "build" in ph:
        build_phase(a, logf)
    if "analyze" in ph:
        analyze_phase(a, logf)
    if "other_states" in ph:
        if a.other_states:
            other_states_phase(a, logf)
        else:
            log("other_states not run: no --other-states directory (the Ohio and Washington lists, INPUT_DATA.md)", logf)
    log("done " + json.dumps(ph), logf)


if __name__ == "__main__":
    main()
