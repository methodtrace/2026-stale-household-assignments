# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The plotted values of Figures 1 and 2, taken from the run's own results, and the drawing of both figures.

figure_data(R) runs inside the run's last step (release_files.publish). It writes paper_figures/figure1_currency_by_age.csv
and paper_figures/figure2_validity_frontier.csv, which carry every plotted value as the text the source file holds, so
the two files are byte-identical on every platform and are listed in OUTPUTS.sha256.

draw(data_dir, out_dir) draws the figures from those two files (make_figures.py). It needs matplotlib, which the run does
not: image files depend on the installed fonts and renderer, so they are not part of the output list.

Figure 1 (section 5.1): the 2016 build, 60 confirmation counties, ages 2 to 10. Address match rate, address currency,
person currency, stayer currency and mover currency, and a band over person currency under the five stale-listing
corrections (last-known household; ages 2 to 8). Figure 2 (section 7.1): match rate against the exact share among
matched clients for the probabilistic matcher's threshold sweep at each age, with stayer currency and the annual rebuild.
"""
import csv
import os

K = (2, 4, 6, 8)
SWEEP = ["wnone", "w-10", "w-5", "w0", "w5", "w10", "w15", "w20", "w25"]
SERIES = [("mr_addr", "mr(10)"), ("kA", "kappa_A(10)"), ("kP", "kappa_P(10)"), ("kS", "kappa_S(10)"), ("mu", "mu(10)")]
CORRECTIONS = ["lookahead_one_election", "lookahead_two_elections", "inactive_dropped", "client_lookahead",
               "client_inactive"]
FIG1 = "paper_figures/figure1_currency_by_age.csv"
FIG2 = "paper_figures/figure2_validity_frontier.csv"
SOURCES = ["results/confirmation/anchor_currency.csv", "results/later_file/ten_year_2016_confirmation.csv",
           "robustness/stale_listing_summary.csv", "results/confirmation/validity_frontier.csv"]


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(path + ".part", path)


def figure_data(R):
    """Write the two plotted-value files; returns how many were written (0 when a source is missing)."""
    if not all(os.path.exists(os.path.join(R, p)) for p in SOURCES):
        return 0
    ac = {int(r["k"]): r for r in _rows(os.path.join(R, SOURCES[0])) if r["t"] == "2016"}
    k10 = {r["stat"]: r["point"] for r in _rows(os.path.join(R, SOURCES[1]))}
    st = [r for r in _rows(os.path.join(R, SOURCES[2]))
          if r["t"] == "2016" and r["policy"] == "last_known" and r["variant"] in CORRECTIONS]
    rows = []
    for name, k10name in SERIES:
        for k in K:
            rows.append(["line", name, k, ac[k][name]])
        rows.append(["line", name, 10, k10[k10name]])
    for k in K:
        ex = sorted((r for r in st if int(r["k"]) == k), key=lambda r: float(r["exact"]))
        rows.append(["band_lo", "kP_corrected", k, ex[0]["exact"]])
        rows.append(["band_hi", "kP_corrected", k, ex[-1]["exact"]])
    _write(os.path.join(R, FIG1), ["mark", "series", "k", "value"], rows)
    vf = {(int(r["k"]), r["rule"]): r for r in _rows(os.path.join(R, SOURCES[3]))}
    rows = []
    for k in K:
        for w in SWEEP:
            r = vf[k, f"probabilistic_matcher_{w}"]
            rows.append(["sweep", k, w, r["match_rate"], r["exact_share"]])
        r = vf[k, "annual_rebuild"]
        rows.append(["annual_rebuild", k, "", r["match_rate"], r["exact_share"]])
        rows.append(["stayer_currency", k, "", "", ac[k]["kS"]])
    _write(os.path.join(R, FIG2), ["mark", "k", "threshold", "match_rate", "exact_share"], rows)
    return 2


def draw(data_dir, out_dir):
    """Draw Figure 1 (fig1_currency) and Figure 2 (fig2_frontier) as PDF, SVG and PNG from the two plotted-value files
    in data_dir (a results root, or recorded/)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    lib = "/usr/share/fonts/opentype/linux-libertine"     # the acmart text face, when installed
    for f in ("LinLibertine_R.otf", "LinLibertine_RI.otf", "LinLibertine_RB.otf"):
        if os.path.exists(os.path.join(lib, f)):
            font_manager.fontManager.addfont(os.path.join(lib, f))
    age = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]
    ink, ink2, grid = "#111111", "#555555", "#e2e2e2"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Linux Libertine O", "DejaVu Serif"], "font.size": 8,
        "axes.edgecolor": ink2, "axes.labelcolor": ink, "axes.linewidth": 0.6, "xtick.color": ink2,
        "ytick.color": ink2, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "svg.hashsalt": "figures",
    })
    f1 = _rows(os.path.join(data_dir, FIG1))
    f2 = _rows(os.path.join(data_dir, FIG2))
    os.makedirs(out_dir, exist_ok=True)

    def save(fig, stem):
        for ext in ("pdf", "svg", "png"):
            kw = {"dpi": 220} if ext == "png" else {}
            kw["metadata"] = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else {}
            fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight", pad_inches=0.08, **kw)
        plt.close(fig)

    # Figure 1
    k10 = K + (10,)
    line = {(r["series"], int(r["k"])): float(r["value"]) for r in f1 if r["mark"] == "line"}
    lo = {int(r["k"]): float(r["value"]) for r in f1 if r["mark"] == "band_lo"}
    hi = {int(r["k"]): float(r["value"]) for r in f1 if r["mark"] == "band_hi"}
    ser = [("mr_addr", "Address match rate", "#08306b", "-"), ("kA", "Address currency", "#d94801", "-"),
           ("kP", "Person currency (registered)", "#fd8d3c", "--"), ("kS", "Stayer currency", "#238b45", "-"),
           ("mu", "Mover currency", "#756bb1", "-")]
    fig, b = plt.subplots(figsize=(5.4, 2.6))
    b.fill_between(K, [lo[k] for k in K], [hi[k] for k in K], color="#fd8d3c", alpha=0.18, lw=0,
                   label="Person currency, corrected")
    for name, lab, c, ls in ser:
        b.plot(k10, [line[name, k] for k in k10], ls, marker="o", ms=2.6, lw=1.1, color=c, label=lab)
    b.set_xticks(k10)
    b.set_xlabel("Age of the evidence (years)")
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.set_ylim(0, 1.0)
    b.set_ylabel("Share")
    b.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, handlelength=1.8, labelspacing=0.45)
    fig.tight_layout()
    save(fig, "fig1_currency")

    # Figure 2
    fig, a = plt.subplots(figsize=(5.4, 3.0))
    for k, col in zip(K, age):
        sw = [r for r in f2 if r["mark"] == "sweep" and int(r["k"]) == k]
        a.plot([float(r["match_rate"]) for r in sw], [float(r["exact_share"]) for r in sw], "-o", color=col, ms=2.4,
               lw=1.1, label=f"{k} years")
        ks = next(float(r["exact_share"]) for r in f2 if r["mark"] == "stayer_currency" and int(r["k"]) == k)
        a.hlines(ks, 0.22, 1.0, colors=col, linestyles=(0, (3, 2)), lw=0.8)
        an = next(r for r in f2 if r["mark"] == "annual_rebuild" and int(r["k"]) == k)
        a.plot(float(an["match_rate"]), float(an["exact_share"]), marker="D", ms=3.6, color=col, mec=ink, mew=0.4, ls="none")
    a.annotate("", xy=(0.60, 0.25), xytext=(0.90, 0.25), arrowprops=dict(arrowstyle="->", color=ink2, lw=0.6))
    a.text(0.75, 0.26, "stricter threshold", fontsize=6.5, color=ink2, ha="center", va="bottom")
    extra = [Line2D([], [], color=ink2, ls=(0, (3, 2)), lw=0.8, label="stayer currency"),
             Line2D([], [], color=ink2, marker="D", ms=3.6, mec=ink, mew=0.4, ls="none", label="annual rebuild")]
    a.set_xlim(0.22, 1.01)
    a.set_ylim(0.2, 0.95)
    a.set_xlabel("Match rate")
    a.set_ylabel("Exact household among matched")
    h, l = a.get_legend_handles_labels()
    a.legend(h + extra, l + [e.get_label() for e in extra], loc="upper left", fontsize=7, ncol=3, handlelength=1.6,
             columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig2_frontier")
