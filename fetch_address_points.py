#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Download the North Carolina address points (NC OneMap AddressNC, FeatureServer layer 0), attributes only, for the
unit-key audit (INPUT_DATA.md).

Usage:
    python3 fetch_address_points.py OUT_DIR

Writes ap_<objectid>.csv.gz files of 50,000 object IDs each and OUT_DIR/SHA256SUMS. The service is live and changes
without notice, so download once and copy OUT_DIR to every machine that should reproduce the same results. Standard
library only. A rerun skips the files already written.
"""
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://services.nconemap.gov/secure/rest/services/AddressNC/NC1Map_Addresses/FeatureServer/0/query"
FIELDS = ["objectid", "county", "countyfips", "add_number", "addnum_suf", "st_predir", "st_premod", "st_presep", "st_pretyp",
          "st_name", "st_posdir", "st_posmod", "st_postyp", "post_code", "post_comm", "unit", "unit_type", "building",
          "building_type", "floor", "floor_type", "placement", "landmkname", "dateupdate", "effective", "expire", "source",
          "st_address", "unique_id"]
BLOCK, PAGE = 50000, 2000


def get(params):
    q = urllib.parse.urlencode({**params, "f": "json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(BASE + "?" + q, timeout=120) as r:
                d = json.load(r)
            if "error" in d:
                raise RuntimeError(d["error"])
            if d.get("exceededTransferLimit"):
                raise RuntimeError("page exceeded the transfer limit")
            return d
        except Exception:
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)


def max_objectid():
    stat = [{"statisticType": "max", "onStatisticField": "objectid", "outStatisticFieldName": "m"}]
    d = get({"where": "1=1", "outStatistics": json.dumps(stat)})
    return int(d["features"][0]["attributes"]["m"])


def block(out, b, top):
    path = os.path.join(out, f"ap_{b:08d}.csv.gz")
    if os.path.exists(path):
        return 0
    rows = []
    for lo in range(b, min(b + BLOCK, top + 1), PAGE):
        d = get({"where": f"objectid>={lo} AND objectid<{lo + PAGE}", "outFields": ",".join(FIELDS), "returnGeometry": "false",
                 "orderByFields": "objectid", "resultRecordCount": PAGE})
        rows += [f["attributes"] for f in d["features"]]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FIELDS)
    for a in rows:
        w.writerow(["" if a.get(k) is None else a.get(k) for k in FIELDS])
    with open(path + ".part", "wb") as f:  # fixed gzip header (no name, mtime 0): the same rows give the same bytes
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as g:
            g.write(buf.getvalue().encode("utf-8"))
    os.replace(path + ".part", path)
    return len(rows)


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    out = argv[0]
    os.makedirs(out, exist_ok=True)
    top = max_objectid()
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(lambda b: block(out, b, top), range(1, top + 1, BLOCK)))
    names = sorted(f for f in os.listdir(out) if f.startswith("ap_") and f.endswith(".csv.gz"))
    with open(os.path.join(out, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        for name in names:
            with open(os.path.join(out, name), "rb") as g:
                f.write(f"{hashlib.sha256(g.read()).hexdigest()}  {name}\n")
    print(f"max objectid {top}; {n} rows written this run; {len(names)} files listed in SHA256SUMS")


if __name__ == "__main__":
    main(sys.argv[1:])
