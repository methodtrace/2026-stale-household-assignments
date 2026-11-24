#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Draw Figures 1 and 2 of the paper from a run's plotted-value files (paper_figures/ under the results root) or from
recorded/. Needs matplotlib (pip install matplotlib); the run itself does not.

    python3 make_figures.py [--data recorded] [--out figures]

Writes fig1_currency and fig2_frontier as PDF, SVG and PNG. The plotted values are checked by hash in OUTPUTS.sha256;
the images are not, because they depend on the installed fonts and renderer.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stale_assignments import paper_figures  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "recorded"), help="results root or recorded/ (default recorded/)")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    paper_figures.draw(a.data, a.out)
    print(f"wrote fig1_currency and fig2_frontier (pdf, svg, png) to {a.out}")


if __name__ == "__main__":
    main()
