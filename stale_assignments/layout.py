# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Where every result file lives under a results root R. One place for the names, so a module never spells a path."""
import os

POLICIES = ["last_known", "never_rebuilt", "lagged_rebuild", "same_date_rebuild", "four_year_rebuild", "annual_rebuild",
            "probabilistic_matcher_w0", "probabilistic_matcher_w20"]
REFERENCES = ["registered", "lookahead_one_election", "lookahead_two_elections", "inactive_dropped", "client_lookahead",
              "client_inactive"]                      # the registered reference and the five stale-listing corrections


def tables(R, *a):
    """Currency tables (table_scores.fit_tables)."""
    return os.path.join(R, "tables", *a)


def table_cells(R, *a):
    """Cell counts the tables are fitted and scored on: development_fine_cells.csv, held_out_cells.csv,
    held_out_lookahead_one_election_cells.csv, later_file_cells.csv."""
    return os.path.join(R, "table_cells", *a)


def robustness(R, *a):
    return os.path.join(R, "robustness", *a)


def results(R, *a):
    """Pipeline results by county set: confirmation/, development/, later_file/."""
    return os.path.join(R, "results", *a)


def household_change(R, *a):
    return os.path.join(R, "household_change", *a)


def other_states(R, *a):
    """Other registers: county sums, county Brier sums, evaluations, North Carolina comparison rows, listed share."""
    return os.path.join(R, "other_states", *a)


def build_error(R, *a):
    """The build-error experiment: counties.csv, summary.csv, exact_build_2018.csv."""
    return os.path.join(R, "build_error", *a)


def refresh(R, *a):
    """The refresh-decision test and the sub-two-year curve."""
    return os.path.join(R, "refresh", *a)
