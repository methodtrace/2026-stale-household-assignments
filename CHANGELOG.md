# Changelog

## 1.0.0

First public release. Python code that rebuilds every result in the paper from the public input files: the North
Carolina statistics for the development and confirmation counties, the held-out and later-file checks, the currency
tables, the household-change model, the refresh test, the robustness checks, the build-error experiment and the Ohio and
Washington comparison. Recorded output list of 320 files, byte-identical on macOS arm64 and Linux x86_64; 81 shipped aggregates under descriptive names with one README per folder, including the plotted values of Figures 1 and 2 (`make_figures.py` draws them).
