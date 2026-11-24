# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The stage 3 runs behind the paper's results, as independent tasks.

Each run is one county list under one run configuration (config.py). Runs share only the stage 1 and 2 tables, so they
can run at the same time: on a Grid Engine cluster as the tasks of one job array (task i runs RUNS[i - 1]), on one machine through a
process pool. Inside a run, Polars and the statistics use the cores the run is given. No result depends on how the runs
or the work inside them were split.

  python -m stale_assignments.runs --list
  python -m stale_assignments.runs --tables WORK/tables --out WORK/results --task 3      (Grid Engine: --task $SGE_TASK_ID)
  python -m stale_assignments.runs --tables WORK/tables --out WORK/results --all

With --all, runs go --jobs at a time (default: as many as memory allows at about 12 GB a run, at most the workers of
parallel.workers()), and the workers are shared out among them.
"""
import argparse
import os

from . import parallel, pipeline
from . import store as S

LATER_LAST = "2026-11-03"
LATER_AUX_2018 = "2020-11-03,2022-11-08,2024-11-05"
GB_PER_RUN = 12


def _run(name, counties, steps=None, outputs=None, **env):
    return dict(name=name, counties=counties, steps=steps, outputs=outputs, env={f"STALE_{k.upper()}": v for k, v in env.items()})


def _later(build, split, aux=""):
    return _run(f"later_file_{split}_b{build[:4]}", split, ["1", "2", "3", "11", "12"], ["anchor_currency.csv", "stats.csv"],
                files="later_file", build=build, aux=aux, last_s=LATER_LAST)


RUNS = [
    _run("development", "development"),
    _run("confirmation", "confirmation"),
    _run("held_out", "held_out", ["1", "2", "3", "11"], []),
    _run("curve", "confirmation", ["1", "2"], [], score_pairs="CURVE_PAIRS"),
    _run("later_file_development_b2016_k10", "development", ["1", "2", "3"], ["anchor_currency.csv"], files="later_file",
         build="2016-11-08", aux="", last_s=LATER_LAST, max_ed="5"),
    _run("later_file_confirmation_b2016_k10", "confirmation", ["1", "2", "3"], ["anchor_currency.csv"], files="later_file",
         build="2016-11-08", aux="", last_s=LATER_LAST, max_ed="5"),
    _later("2018-11-06", "development", LATER_AUX_2018),
    _later("2018-11-06", "confirmation", LATER_AUX_2018),
    _run("later_file_all_b2018", "all", ["1", "2"], [], files="later_file", build="2018-11-06", aux=LATER_AUX_2018,
         last_s=LATER_LAST),
] + [_later(b, s) for b in ("2020-11-03", "2022-11-08", "2024-11-05") for s in ("development", "confirmation")]


def counties(kind):
    """development (20), confirmation (60), held_out (20) or all (100), from data/counties.tsv."""
    out = []
    with open(os.path.join(os.path.dirname(__file__), "data", "counties.tsv"), encoding="utf-8") as f:
        f.readline()
        for ln in f:
            c, split = ln.rstrip("\n").split("\t")
            if kind == "all" or kind == split:
                out.append(int(c))
    return out


def run_one(r, tables, out, logf=None):
    env = dict(r["env"])
    if env.get("STALE_SCORE_PAIRS") == "CURVE_PAIRS":
        env["STALE_SCORE_PAIRS"] = os.path.join(os.path.dirname(__file__), "data", "curve_pairs.csv")
    dest = os.path.join(out, r["name"])
    pipeline.run_config(env, S.Store(tables), dest, counties(r["counties"]), steps=r["steps"], outputs=r["outputs"],
                        logf=logf or dest + ".log")
    return r["name"]


def _task(args):
    r, tables, out, per_run = args
    # each run's own processes inherit these: Polars threads and statistics workers shared out among concurrent runs
    os.environ["STALE_WORKERS"] = str(per_run)
    os.environ["POLARS_MAX_THREADS"] = str(per_run)
    return run_one(r, tables, out)


def memory_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return GB_PER_RUN


def default_jobs():
    return max(1, min(parallel.workers(), int(memory_gb() // GB_PER_RUN)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tables")
    ap.add_argument("--out")
    ap.add_argument("--task", type=int, help="1-based run index (Grid Engine: $SGE_TASK_ID)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", help="comma-separated run names (with --all)")
    ap.add_argument("--jobs", type=int, help="runs at the same time with --all (default: memory / 12 GB)")
    a = ap.parse_args()
    if a.list:
        for i, r in enumerate(RUNS, 1):
            print(i, r["name"], r["counties"], ",".join(r["steps"] or ["all"]), r["env"])
        return
    if a.task:
        print(run_one(RUNS[a.task - 1], a.tables, a.out))
    elif a.all:
        todo = [r for r in RUNS if not a.only or r["name"] in a.only.split(",")]
        jobs = max(1, min(a.jobs or default_jobs(), len(todo)))
        per_run = max(1, parallel.workers() // jobs)
        for name in parallel.pmap(_task, [(r, a.tables, a.out, per_run) for r in todo], n=jobs):
            print("done", name)


if __name__ == "__main__":
    main()
