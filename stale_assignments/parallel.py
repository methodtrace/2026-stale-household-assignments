# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Process parallelism with deterministic results: work is split into independent tasks whose results are returned in
task order and combined in that order, so the output does not depend on the number of workers."""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor


def workers():
    """Worker processes: STALE_WORKERS if set; else the slots a Grid Engine job was granted (NSLOTS); else every core but
    one, leaving one for the parent process and the desktop."""
    for k in ("STALE_WORKERS", "NSLOTS"):
        if os.environ.get(k):
            return max(1, int(os.environ[k]))
    return max(1, (os.cpu_count() or 2) - 1)


def chunks(xs, n):
    """xs split into at most n contiguous parts, in order."""
    xs = list(xs)
    n = max(1, min(n, len(xs)))
    q, r = divmod(len(xs), n)
    out, i = [], 0
    for j in range(n):
        m = q + (1 if j < r else 0)
        out.append(xs[i:i + m])
        i += m
    return out


def memory_gb():
    """Physical memory of the machine in GB."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return 16.0


def jobs_for(gb_each, reserve_gb=0.0, most=None):
    """How many tasks of about gb_each GB can run at once: memory less reserve_gb, at most the workers (and most)."""
    n = max(1, min(workers(), int((memory_gb() - reserve_gb) // gb_each)))
    return min(n, most) if most else n


def _share(threads):
    # in each worker: Polars threads and nested worker counts shared out, so concurrent tasks do not oversubscribe
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    os.environ["STALE_WORKERS"] = str(threads)


def executor(n):
    """A spawned process pool of n workers, each given workers() // n threads."""
    return ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"), initializer=_share,
                               initargs=(max(1, workers() // n),))


def pmap(fn, tasks, n=None, share=False):
    """[fn(t) for t in tasks], computed in spawned worker processes (serially when one worker or one task). With share,
    each worker gets workers() // n threads for Polars and its own nested pools."""
    tasks = list(tasks)
    n = workers() if n is None else n
    if n <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    n = min(n, len(tasks))
    if share:
        with executor(n) as ex:
            return list(ex.map(fn, tasks))
    with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn")) as ex:
        return list(ex.map(fn, tasks))


def pool(n, initializer=None, initargs=()):
    """A spawned pool of n worker processes (spawn, so workers start the same way on macOS and Linux)."""
    return multiprocessing.get_context("spawn").Pool(n, initializer=initializer, initargs=initargs)
