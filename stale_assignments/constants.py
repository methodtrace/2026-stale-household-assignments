# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Fixed constants shared across the analysis. Changing a seed changes every interval drawn from it.

Every seed is a fixed purpose label. A string salt is the label itself; an integer seed is the first 8 bytes of the
SHA-256 of its label, read as an unsigned big-endian integer (seed_of)."""
import hashlib


def seed_of(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:8], "big")


# Seeds of the county-cluster bootstraps (random.Random; 2,000 replicates, counties drawn with replacement)
TABLE_BOOTSTRAP_SEED = seed_of("table-bootstrap")        # currency tables on held-out and later-file cells
MODEL_BOOTSTRAP_SEED = seed_of("model-bootstrap")        # household-change model against the tables, held-out counties
BOOTSTRAP_REPS = 2000
TRANSFER_BOOTSTRAP_SEED = seed_of("transfer-bootstrap")  # transfer to Ohio and Washington, counties within each pair
REFRESH_BOOTSTRAP_SEED = seed_of("refresh-bootstrap")    # refresh decision test and the sub-two-year curve
OTHER_STATES_BOOTSTRAP_SEED = seed_of("other-states")    # Ohio and Washington accounting and ranking, a stream per pair
# Seed of the build-error experiment's county bootstrap (numpy default_rng, one stream over the arms in ARMS order)
BUILD_ERROR_BOOTSTRAP_SEED = seed_of("build-error")
# String salts, hashed with FarmHash together with the record or county they apply to
POISSON_BOOTSTRAP_SALT = "county-weights"     # Poisson(1) county weights (bootstrap.weights)
CLIENT_SAMPLE_SALT = "client-sample"          # the 10% client sample (common.client_flag)
FIGURE_BANDS_SEED = "figure-bands"            # random.Random seed of the bands behind Figures 1 and 2 (paper_tables.py)
REFRESH_COHORT_SALT = "refresh-cohort"        # refresh test: cohort split
REFRESH_SAMPLE_SALT = "refresh-sample"        # refresh test: random rebuild order
MATCHER_TIE_SALT = "matcher-tie-break"        # probabilistic matcher: tie-break among equal weights
