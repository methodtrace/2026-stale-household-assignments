# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The household-change model (paper section 8): a latent continuous-time chain per anchor, fitted to consecutive voter
file snapshots and evaluated on held-out counties and later files.

  model     the chain, its rates and the forward step, used by the fit and by every prediction
  panel     the snapshot-pair counts the fit reads
  fit       the likelihood, its exact gradient, the parallel objective and the cold-start fit
  optimize  limited-memory BFGS in deterministic arithmetic
  transfer  the model carried to Ohio and Washington with each register's own listing process
"""
