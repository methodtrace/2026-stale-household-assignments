#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Create a private virtual environment in a workspace outside this repository, install the pinned dependencies, run the
unit tests, then run reproduce_all.py with that environment.

Usage:
    python3 bootstrap.py --workspace DIR --snapshots DIR [--allow-download | --wheelhouse DIR]
                         [--phases tables,runs,build,analyze,other_states] [--jobs N] [--fetch] [--discard-inputs]
                         [--voter-history FILE] [--address-points DIR] [--other-states DIR] [--setup-only]

The environment goes to <workspace>/venv and the working tables to <workspace>/work. Nothing is installed outside the
workspace and nothing is downloaded unless --allow-download is given (or a local wheelhouse is named). On CPython 3.13 on
Linux x86_64 or macOS arm64 an exact wheel lock (with hashes) is used; elsewhere the version pins in requirements.txt.
With --setup-only it stops after the unit tests, so the environment can be built where the network is (a cluster login
node) and the run started later where it is not; --snapshots is then not needed.
"""
import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--snapshots", type=Path)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--wheelhouse", type=Path)
    p.add_argument("--phases", default="tables,runs,build,analyze,other_states", help="passed to reproduce_all.py")
    p.add_argument("--jobs", type=int, help="passed to reproduce_all.py: stage 3 runs at the same time")
    p.add_argument("--fetch", action="store_true", help="passed to reproduce_all.py: download missing snapshot files")
    p.add_argument("--discard-inputs", action="store_true", help="passed to reproduce_all.py: delete each snapshot once used")
    p.add_argument("--voter-history", type=Path, help="passed to reproduce_all.py: ncvhis_Statewide.zip (INPUT_DATA.md)")
    p.add_argument("--address-points", type=Path, help="passed to reproduce_all.py: fetch_address_points.py output directory")
    p.add_argument("--other-states", type=Path, help="passed to reproduce_all.py: the Ohio and Washington lists (INPUT_DATA.md)")
    p.add_argument("--setup-only", action="store_true", help="build the environment and run the unit tests, then stop")
    a = p.parse_args()
    if a.snapshots is None and not a.setup_only and "tables" in a.phases.split(","):
        p.error("--snapshots is required for the tables phase")
    w = a.workspace.resolve()
    if w == ROOT or ROOT in w.parents:
        sys.exit("The workspace must be outside the repository.")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] not in ((3, 12), (3, 13)) or sys.flags.optimize:
        sys.exit("Use CPython 3.12 or 3.13, without -O (cityhash 0.4.10 publishes wheels up to CPython 3.13).")
    lock = None
    if sys.version_info[:2] == (3, 13):
        if sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64"):
            lock = "requirements-lock-linux-cp313.txt"
        elif sys.platform == "darwin" and platform.machine() == "arm64":
            lock = "requirements-lock-macos-arm64-cp313.txt"
    strict = lock is not None
    if not strict:
        print("No exact wheel lock for this interpreter and platform: installing the version pins of requirements.txt "
              "without hashes.", file=sys.stderr)
    w.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PIP_CACHE_DIR=str(w / "cache" / "pip"))
    py = w / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(w / "venv")
    ready = w / "venv" / "READY"
    if not ready.exists():
        if not a.allow_download and a.wheelhouse is None:
            sys.exit("Dependencies need --allow-download or --wheelhouse DIR. No network was used.")
        req = ROOT / (lock or "requirements.txt")
        cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:", "-r", str(req)]
        if strict:
            cmd.insert(5, "--require-hashes")
        if a.wheelhouse:
            cmd += ["--no-index", "--find-links", str(a.wheelhouse.resolve())]
        subprocess.run(cmd, env=env, check=True)
        ready.write_text("ok\n")
    subprocess.run([str(py), "-B", "-m", "unittest", "discover", "-s", "tests"], env=env, check=True,
                   cwd=str(ROOT))
    if a.setup_only:
        v = subprocess.run([str(py), "-c", "import sys, platform, polars; print(platform.python_implementation(), "
                            "platform.python_version(), sys.platform, platform.machine(), 'polars', polars.__version__)"],
                           env=env, check=True, capture_output=True, text=True).stdout.strip()
        print(f"SETUP OK: {v}; environment {w / 'venv'}; lock {lock or 'none (requirements.txt pins)'}")
        sys.exit(0)
    r = subprocess.run([str(py), "-B", str(ROOT / "reproduce_all.py"), "--work", str(w / "work"), "--phases", a.phases]
                       + (["--snapshots", str(a.snapshots.resolve())] if a.snapshots else [])
                       + (["--other-states", str(a.other_states.resolve())] if a.other_states else [])
                       + (["--jobs", str(a.jobs)] if a.jobs else [])
                       + ["--fetch"] * a.fetch + ["--discard-inputs"] * a.discard_inputs
                       + (["--voter-history", str(a.voter_history.resolve())] if a.voter_history else [])
                       + (["--address-points", str(a.address_points.resolve())] if a.address_points else []), env=env)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
