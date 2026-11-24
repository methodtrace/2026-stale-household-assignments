# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""Output files with reproducible bytes."""
import gzip
import io


def gz_text_writer(path):
    """A text writer for a .gz file whose header carries no timestamp, so the same content gives the same bytes."""
    raw = open(path, "wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return _Closing(io.TextIOWrapper(gz, encoding="utf-8", newline=""), gz, raw)


class _Closing:
    def __init__(self, text, gz, raw):
        self.text, self.gz, self.raw = text, gz, raw

    def __enter__(self):
        return self.text

    def __exit__(self, *exc):
        self.text.close(); self.gz.close(); self.raw.close()
        return False
