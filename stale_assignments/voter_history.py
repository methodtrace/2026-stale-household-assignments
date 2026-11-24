# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""NCSBE statewide voter history (ncvhis_Statewide.zip): the NCIDs recorded as voting in one election.

The file is tab-separated text with every field in double quotes, one row per voter and election. Only two columns are
read, election_lbl (MM/DD/YYYY) and ncid; nothing else is kept. NCSBE replaces the file weekly, so a run records the
SHA-256 of the copy it read (INPUT_DATA.md)."""
import zipfile

import polars as pl


def voted(zip_path: str, election: str = "11/08/2016") -> pl.DataFrame:
    """Distinct NCIDs with a history row for the election, sorted."""
    tag = f'"{election}"'.encode("ascii")
    out = set()
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"expected one member in {zip_path}, found {len(names)}")
        with zf.open(names[0]) as f:
            head = [h.strip(b'"') for h in f.readline().rstrip(b"\r\n").split(b"\t")]
            ie, ic = head.index(b"election_lbl"), head.index(b"ncid")
            for ln in f:
                if tag not in ln:
                    continue
                x = ln.rstrip(b"\r\n").split(b"\t")
                if x[ie].strip(b'"') == tag[1:-1]:
                    v = x[ic].strip(b'"').strip()
                    if v:
                        out.add(v.decode("ascii"))
    return pl.DataFrame({"ncid": sorted(out)}, schema={"ncid": pl.String})
