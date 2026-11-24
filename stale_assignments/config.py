# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Run configuration of stage 3, read from the environment once per process.

The default (no variable set) is the main design: the 40 registered snapshots, build 2016-11-08 with the further builds
2008 to 2014, Election Days to 2024-11-05, a window of four Election Days. Other runs set, before the stage 3 processes
start:
  STALE_FILES        registered | later_file: later_file adds the 2026-11-03 file (its own relabel components and age
                     flags; store.py)
  STALE_BUILD        the main build t (every policy; client-file lag 1 for never_rebuilt and same_date_rebuild)
  STALE_AUX          comma-separated further builds, policy never_rebuilt, lag 0 only ('' for none)
  STALE_LAST_S       the last scoring date of step 1's person events and step 3's carriers
  STALE_MAX_ED       Election Days after the main build that step 2 pairs it with (further builds always 4)
  STALE_SCORE_PAIRS  CSV t_dt,s_dt: step 2 scores exactly these pairs (policy never_rebuilt, lag 0), any loaded s
pipeline.run_config sets them for one run and starts every step in a fresh process, so a run never sees another's values.
"""
import csv
import datetime as dt
import os

_d = dt.date.fromisoformat

FILES = os.environ.get("STALE_FILES", "registered")
assert FILES in ("registered", "later_file"), FILES
BUILD = _d(os.environ.get("STALE_BUILD", "2016-11-08"))
AUX = [_d(x) for x in os.environ.get("STALE_AUX", "2008-11-04,2010-11-02,2012-11-06,2014-11-04").split(",") if x]
LAST_S = _d(os.environ.get("STALE_LAST_S", "2024-11-05"))
MAX_ED = int(os.environ.get("STALE_MAX_ED", "4"))
_sp = os.environ.get("STALE_SCORE_PAIRS", "")
SCORE_PAIRS = sorted({(_d(r["t_dt"]), _d(r["s_dt"])) for r in csv.DictReader(open(_sp))}) if _sp else None
DEFAULT = (FILES == "registered" and BUILD == dt.date(2016, 11, 8) and LAST_S == dt.date(2024, 11, 5) and MAX_ED == 4
           and not SCORE_PAIRS and AUX == [dt.date(2008, 11, 4), dt.date(2010, 11, 2), dt.date(2012, 11, 6), dt.date(2014, 11, 4)])
KEYS = ("STALE_FILES", "STALE_BUILD", "STALE_AUX", "STALE_LAST_S", "STALE_MAX_ED", "STALE_SCORE_PAIRS")
