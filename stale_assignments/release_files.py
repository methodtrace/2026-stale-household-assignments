# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The released result files: descriptive names for the files the release ships, release copies of the four fitted
tables with no cell under 10 persons, and the plotted values of Figures 1 and 2 (paper_figures.py).

publish(R) runs at the end of the analyze and other_states phases, before the output list is written. It copies each
shipped result file to its descriptive path under R, writes the two figure files and the four release tables. The output list then names a
shipped file by its descriptive path only; every other result file, and the unmerged tables, keep their working path
and are checked by hash.

Release tables (the paper scores the unmerged tables; no printed number comes from these copies):
  1. By build. A coarse cell under 10 persons merges with the adjacent age band at the same evidence age, build, size and
     building type, under a combined label such as a65+a75. The co-member-age table relabels the same rows and sums
     rows whose keys then agree. Within each coarse cell, co-member-age cells under 10 pool into one row whose ny, mb
     and gb read "other"; when that row is still under 10, the smallest listed cells join it until it reaches 10.
  2. By parity. Each parity row is the sum of by-build rows of the same parity, so the parity tables carry no count the
     by-build tables do not already give. A parity cell takes the union of its builds' band merges. A co-member-age key
     is listed in a parity cell only when every build listing a person in that key lists it; the rest of the cell is
     its "other" row.
  3. p = (ne + pk) / (n + 1) from the summed counts, pk the pooled exact share at the evidence age.
Lookup: the co-member-age key if listed, else the cell's "other" row, else the coarse value, else pk.
"""
import csv
import os
import shutil
from collections import defaultdict

from . import paper_figures
from .table_scores import parity

MIN_PERSONS = 10
BANDS = ["a18", "a25", "a35", "a45", "a55", "a65", "a75"]
OTHER = "other"

# working path -> published path, for every shipped file copied as written
SHIPPED = {
    # North Carolina, 60 confirmation counties, 2016 build unless the file says otherwise
    "results/confirmation/anchor_currency.csv": "nc_confirmation_counties/currency_by_build_and_age.csv",
    "results/confirmation/envelopes.csv": "nc_confirmation_counties/exit_score_envelopes.csv",
    "results/confirmation/exit_counts.csv": "nc_confirmation_counties/exit_counts_by_reason.csv",
    "results/confirmation/inputs_by_k.csv": "nc_confirmation_counties/match_rate_inputs_by_age.csv",
    "results/confirmation/link_same_population.csv": "nc_confirmation_counties/matcher_same_population.csv",
    "results/confirmation/matcher_answer_diagnostics.csv": "nc_confirmation_counties/matcher_answer_diagnostics.csv",
    "results/confirmation/mover_timing.csv": "nc_confirmation_counties/mover_timing.csv",
    "results/confirmation/readdress_ablation.csv": "nc_confirmation_counties/readdressing_ablation.csv",
    "results/confirmation/ref_sensitivity.csv": "nc_confirmation_counties/reference_sensitivity.csv",
    "results/confirmation/small_cluster.csv": "nc_confirmation_counties/jackknife_t_intervals.csv",
    "results/confirmation/stats.tsv": "nc_confirmation_counties/hypothesis_statistics.tsv",
    "results/confirmation/three_outcome.csv": "nc_confirmation_counties/right_wrong_unknown_bounds.csv",
    "results/confirmation/tipping_points.csv": "nc_confirmation_counties/exit_score_tipping_points.csv",
    "results/confirmation/tipping_two_score.csv": "nc_confirmation_counties/exit_score_two_age_tipping.csv",
    "results/confirmation/validity_frontier.csv": "nc_confirmation_counties/match_validity_by_policy.csv",
    "results/confirmation/validity_stats.tsv": "nc_confirmation_counties/match_validity_statistics.tsv",
    # North Carolina, 20 development counties
    "results/development/anchor_currency.csv": "nc_development_counties/currency_by_build_and_age.csv",
    "results/development/link_same_population.csv": "nc_development_counties/matcher_same_population.csv",
    "results/development/ref_sensitivity.csv": "nc_development_counties/reference_sensitivity.csv",
    "results/development/stats.tsv": "nc_development_counties/hypothesis_statistics.tsv",
    "results/development/three_outcome.csv": "nc_development_counties/right_wrong_unknown_bounds.csv",
    "results/development/validity_frontier.csv": "nc_development_counties/match_validity_by_policy.csv",
    "results/development/validity_stats.tsv": "nc_development_counties/match_validity_statistics.tsv",
    # North Carolina, November 2026 file (the later file)
    "results/later_file/anchor_currency_confirmation.csv": "nc_november_2026_file/currency_by_build_confirmation.csv",
    "results/later_file/anchor_currency_development.csv": "nc_november_2026_file/currency_by_build_development.csv",
    "results/later_file/gap_2018_confirmation.csv": "nc_november_2026_file/gap_statistics_2018_build_confirmation.csv",
    "results/later_file/ten_year_2016_confirmation.csv": "nc_november_2026_file/ten_year_statistics_2016_confirmation.csv",
    "results/later_file/ten_year_2016_development.csv": "nc_november_2026_file/ten_year_statistics_2016_development.csv",
    "results/later_file/validity_exact_confirmation.csv": "nc_november_2026_file/exact_share_by_policy_confirmation.csv",
    "results/later_file/validity_exact_development.csv": "nc_november_2026_file/exact_share_by_policy_development.csv",
    # North Carolina, whole state
    "results/acs_movers.csv": "nc_statewide/acs_mover_rates.csv",
    "results/confirmation/mobility_calibration.csv": "nc_statewide/mobility_vs_acs_by_age_band.csv",
    "robustness/mobility_rates.csv": "nc_statewide/recorded_moves_and_exits_by_year.csv",
    "results/unit_key_statewide.csv": "nc_statewide/unit_key_audit.csv",
    "robustness/county_split.csv": "nc_statewide/county_sets.csv",
    # robustness and sensitivity checks
    "robustness/age_reweight.csv": "robustness_checks/age_reweighted_currency.csv",
    "robustness/anchor_currency_full.csv": "robustness_checks/currency_all_county_sets.csv",
    "robustness/client_universe.csv": "robustness_checks/client_universe.csv",
    "robustness/confirmation_vs_held_out.csv": "robustness_checks/confirmation_vs_held_out.csv",
    "robustness/forecast_bands.csv": "robustness_checks/forecast_bands.csv",
    "robustness/matcher_paired.csv": "robustness_checks/matcher_paired_differences.csv",
    "robustness/probabilistic_matcher_alone.csv": "robustness_checks/matcher_by_weight_threshold.csv",
    "robustness/probabilistic_matcher_w20_alone.csv": "robustness_checks/matcher_w20_by_reference.csv",
    "robustness/size_split.csv": "robustness_checks/currency_by_household_size.csv",
    "results/currency_by_build_size.csv": "robustness_checks/currency_by_building_type.csv",
    "robustness/stale_listing_intervals.csv": "robustness_checks/stale_listing_intervals.csv",
    "robustness/stale_listing_summary.csv": "robustness_checks/stale_listing_summary.csv",
    "robustness/stale_listing_summary_held_out.csv": "robustness_checks/stale_listing_summary_held_out.csv",
    "robustness/unrecorded_move_bracket.csv": "robustness_checks/unrecorded_move_scenario.csv",
    # fitted currency tables and their tests
    "tables/coarse_table_pk.csv": "fitted_tables/pooled_share_by_age.csv",
    "tables/held_out_cells.csv": "fitted_tables/held_out_cell_errors.csv",
    "tables/held_out_metrics.csv": "fitted_tables/held_out_scores.csv",
    "tables/stale_probe_development.csv": "fitted_tables/stale_listing_probe_development.csv",
    "robustness/table_baselines.csv": "fitted_tables/tables_vs_baselines.csv",
    "robustness/table_skill.csv": "fitted_tables/table_scores.csv",
    "robustness/table_skill_cells.csv": "fitted_tables/table_scores_by_build_and_age.csv",
    # household-change model
    "household_change/fit.json": "household_change_model/model_fit.json",
    "household_change/step_check.csv": "household_change_model/fit_step_check.csv",
    "household_change/held_out/build_age_cells.csv": "household_change_model/held_out_by_build_and_age.csv",
    "household_change/held_out/model_jackknife.csv": "household_change_model/held_out_jackknife_vs_tables.csv",
    "household_change/held_out/stayer_def.csv": "household_change_model/stayer_definitions.csv",
    "household_change/held_out/stayer_xtab_2016.csv": "household_change_model/stayer_crosstab_2016.csv",
    "household_change/held_out/summary.csv": "household_change_model/held_out_summary.csv",
    # Ohio and Washington
    "other_states/evaluation_identifier.csv": "ohio_washington/evaluation_voter_id.csv",
    "other_states/evaluation_crosswalk.csv": "ohio_washington/evaluation_address_crosswalk.csv",
    "other_states/listed_share.csv": "ohio_washington/listed_share_by_register.csv",
    "other_states/north_carolina_pair_sums.csv": "ohio_washington/nc_comparison_pair_sums.csv",
    "other_states/north_carolina_tables.csv": "ohio_washington/nc_comparison_table_scores.csv",
    "household_change/transfer/fit_listing.json": "ohio_washington/model_transfer_listing_fit.json",
    "household_change/transfer/summary_long.csv": "ohio_washington/model_transfer_summary.csv",
    # refresh decision
    "refresh/refresh_results.csv": "refresh_decision/refresh_results.csv",
    "refresh/curve_stats.csv": "refresh_decision/first_two_years_curve.csv",
    "refresh/refresh_diag_posthoc.csv": "refresh_decision/predictor_diagnostics_posthoc.csv",
    # build-error experiment
    "build_error/summary.csv": "build_error_experiment/currency_by_build_accuracy.csv",
    "build_error/exact_build_2018.csv": "build_error_experiment/exact_arm_check_2018.csv",
}

# release tables: published path -> working table it is made from
RELEASE_TABLES = {
    "fitted_tables/coarse_table_by_build.csv": "tables/coarse_table.csv",
    "fitted_tables/comember_ages_table_by_build.csv": "tables/comember_age_table.csv",
    "fitted_tables/coarse_table_by_parity.csv": "tables/coarse_table_parity.csv",
    "fitted_tables/comember_ages_table_by_parity.csv": "tables/comember_age_table_parity.csv",
}

FINE = ["ny", "mb", "gb"]


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n"] = int(r["n"]); r["ne"] = int(r["ne"])
    return rows


def _pk(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["k"]: float(r["pk"]) for r in csv.DictReader(f)}


def _pooled(coarse):
    """Pooled exact share by evidence age, computed as the fit computes it (sums over all cells, one division)."""
    t = defaultdict(lambda: [0, 0])
    for r in coarse:
        t[r["k"]][0] += r["n"]; t[r["k"]][1] += r["ne"]
    return {k: ne / n for k, (n, ne) in t.items()}


def band_groups(coarse):
    """{(k, build, size6, bt): list of band groups}: a band under MIN_PERSONS joins its lower neighbor (the upper one for
    the lowest band) until every group holds MIN_PERSONS. Bands absent from a cell are left out."""
    cells = defaultdict(dict)
    for r in coarse:
        cells[(r["k"], r["build"], r["size6"], r["bt"])][r["ab"]] = r["n"]
    out = {}
    for c, nb in cells.items():
        groups = [[b] for b in BANDS if b in nb]
        while len(groups) > 1:
            small = [i for i, g in enumerate(groups) if sum(nb[b] for b in g) < MIN_PERSONS]
            if not small:
                break
            i = small[0]
            j = i - 1 if i > 0 else i + 1
            lo, hi = min(i, j), max(i, j)
            groups[lo:hi + 1] = [groups[lo] + groups[hi]]
        out[c] = groups
    return out


def _label(groups):
    return {b: "+".join(g) for g in groups for b in g}


def _parity_groups(build_groups):
    """The union of the builds' band groups at each (k, parity, size6, bt)."""
    by = defaultdict(list)
    for (k, build, size6, bt), groups in build_groups.items():
        by[(k, parity(build), size6, bt)].append(groups)
    out = {}
    for c, gl in by.items():
        bands = sorted({b for groups in gl for g in groups for b in g}, key=BANDS.index)
        parent = {b: b for b in bands}

        def find(b):
            while parent[b] != b:
                b = parent[b]
            return b
        for groups in gl:
            for g in groups:
                for b in g[1:]:
                    parent[find(b)] = find(g[0])
        merged = defaultdict(list)
        for b in bands:
            merged[find(b)].append(b)
        out[c] = sorted(merged.values(), key=lambda g: BANDS.index(g[0]))
    return out


def _sum_rows(rows, keyf):
    """Sum n and ne by key, in order of first appearance."""
    tot = {}
    for r in rows:
        key = keyf(r)
        t = tot.setdefault(key, [0, 0])
        t[0] += r["n"]; t[1] += r["ne"]
    return tot


def _pool(cell):
    """cell: {fine key: [n, ne]} for one coarse cell -> (listed {fine: [n, ne]}, pooled [n, ne] or None)."""
    small = [f for f, v in cell.items() if v[0] < MIN_PERSONS]
    if not small:
        return dict(cell), None
    hidden = set(small)
    pooled = sum(cell[f][0] for f in hidden)
    for f in sorted((f for f in cell if f not in hidden), key=lambda f: (cell[f][0], f)):
        if pooled >= MIN_PERSONS:
            break
        hidden.add(f); pooled += cell[f][0]
    listed = {f: v for f, v in cell.items() if f not in hidden}
    return listed, [sum(cell[f][0] for f in hidden), sum(cell[f][1] for f in hidden)]


def release_tables(tables_dir):
    """The four release tables as {published path: (header, rows)} plus a check that each parity table equals the sum
    of its by-build counterpart."""
    coarse = _read(os.path.join(tables_dir, "coarse_table.csv"))
    pk = _pooled(coarse)
    if {k: f"{v:.10f}" for k, v in pk.items()} != {k: f"{v:.10f}" for k, v in _pk(os.path.join(tables_dir, "coarse_table_pk.csv")).items()}:
        raise SystemExit("coarse_table_pk.csv is not the pooled share of coarse_table")
    fine = _read(os.path.join(tables_dir, "comember_age_table.csv"))
    coarse_p = _read(os.path.join(tables_dir, "coarse_table_parity.csv"))
    fine_p = _read(os.path.join(tables_dir, "comember_age_table_parity.csv"))
    for name in ("comember_age_table", "coarse_table_parity", "comember_age_table_parity"):
        if _pk(os.path.join(tables_dir, f"{name}_pk.csv")) != _pk(os.path.join(tables_dir, "coarse_table_pk.csv")):
            raise SystemExit(f"{name}_pk.csv differs from coarse_table_pk.csv")
    bg = band_groups(coarse)
    pg = _parity_groups(bg)
    blab = {c: _label(g) for c, g in bg.items()}
    plab = {c: _label(g) for c, g in pg.items()}
    bkey = lambda r: (r["k"], r["build"], r["size6"], r["bt"], blab[(r["k"], r["build"], r["size6"], r["bt"])][r["ab"]])
    pkey = lambda r, by="parity": (r["k"], r[by] if by == "parity" else parity(r[by]), r["size6"], r["bt"],
                                   plab[(r["k"], r[by] if by == "parity" else parity(r[by]), r["size6"], r["bt"])][r["ab"]])

    # by build
    c_rel = _sum_rows(coarse, bkey)
    f_cells = defaultdict(dict)
    for key, v in _sum_rows(fine, lambda r: bkey(r) + tuple(r[x] for x in FINE)).items():
        f_cells[key[:5]][key[5:]] = v
    f_rel, f_listed = {}, {}
    for c, cell in f_cells.items():
        listed, pooled = _pool(cell)
        f_listed[c] = listed
        for f, v in listed.items():
            f_rel[c + f] = v
        if pooled:
            f_rel[c + (OTHER,) * 3] = pooled
        if sum(v[0] for v in cell.values()) != c_rel[c][0]:
            raise SystemExit(f"co-member-age cells do not sum to the coarse cell {c}")

    # by parity: sums of by-build rows
    cp_rel = _sum_rows(coarse, lambda r: pkey(r, "build"))
    if cp_rel != _sum_rows(coarse_p, pkey):
        raise SystemExit("coarse_table_parity is not the sum of coarse_table by parity")
    true_fp = defaultdict(dict)
    for key, v in _sum_rows(fine_p, lambda r: pkey(r) + tuple(r[x] for x in FINE)).items():
        true_fp[key[:5]][key[5:]] = v
    true_from_build = defaultdict(dict)
    for key, v in _sum_rows(fine, lambda r: pkey(r, "build") + tuple(r[x] for x in FINE)).items():
        true_from_build[key[:5]][key[5:]] = v
    if true_fp != true_from_build:
        raise SystemExit("comember_age_table_parity is not the sum of comember_age_table by parity")
    members = defaultdict(list)                      # parity cell -> by-build coarse cells in it
    for c in c_rel:
        k, build, size6, bt, ab = c
        p = (k, parity(build), size6, bt)
        members[p + (plab[p][ab.split("+")[0]],)].append(c)
    fp_rel = {}
    for C, cell in true_fp.items():
        listed = {}
        for f, v in cell.items():
            # listed only if no build holding persons in f hides them
            if all(f in f_listed[c] or f not in f_cells[c] for c in members[C]):
                listed[f] = v
        for f, v in listed.items():
            fp_rel[C + f] = v
        rest = [cp_rel[C][0] - sum(v[0] for v in listed.values()), cp_rel[C][1] - sum(v[1] for v in listed.values())]
        if rest[0]:
            fp_rel[C + (OTHER,) * 3] = rest

    def rows(tot):
        out = []
        for key, (n, ne) in tot.items():
            if n < MIN_PERSONS:
                raise SystemExit(f"release cell under {MIN_PERSONS}: {key}")
            out.append(list(key) + [n, ne, f"{(ne + pk[key[0]]) / (n + 1):.10f}"])
        return out

    def order(tot):
        def sk(item):
            key = item[0]
            fine_part = tuple((1, "") if x == OTHER else (0, x) for x in key[5:])
            return (int(key[0]), key[1], key[2], key[3], BANDS.index(key[4].split("+")[0])) + fine_part
        return dict(sorted(tot.items(), key=sk))
    hb = ["k", "build", "size6", "bt", "ab"]
    hp = ["k", "parity", "size6", "bt", "ab"]
    return {
        "fitted_tables/coarse_table_by_build.csv": (hb + ["n", "ne", "p"], rows(order(c_rel))),
        "fitted_tables/comember_ages_table_by_build.csv": (hb + FINE + ["n", "ne", "p"], rows(order(f_rel))),
        "fitted_tables/coarse_table_by_parity.csv": (hp + ["n", "ne", "p"], rows(order(cp_rel))),
        "fitted_tables/comember_ages_table_by_parity.csv": (hp + FINE + ["n", "ne", "p"], rows(order(fp_rel))),
    }


def lookup(coarse_rel, fine_rel, key, fine_key, pk):
    """Release-table lookup for one client: key (k, build or parity, size6, bt, band) and fine_key (ny, mb, gb).
    coarse_rel and fine_rel are {key tuple: p}. The band is mapped to its combined label when merged."""
    labels = {}
    for kk in coarse_rel:
        for b in kk[4].split("+"):
            labels[kk[:4] + (b,)] = kk[4]
    c = key[:4] + (labels.get(key[:4] + (key[4],), key[4]),)
    for k in (c + tuple(fine_key), c + (OTHER,) * 3):
        if k in fine_rel:
            return fine_rel[k]
    return coarse_rel.get(c, pk[key[0]])


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(path + ".part", path)


def publish(R):
    """Copy each shipped file that exists to its published path, and write the release tables when the fitted tables
    exist. Returns the number of files published."""
    n = 0
    for src, dst in SHIPPED.items():
        s = os.path.join(R, src)
        if os.path.exists(s):
            d = os.path.join(R, dst)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copyfile(s, d + ".part")
            os.replace(d + ".part", d)
            n += 1
    n += paper_figures.figure_data(R)
    if all(os.path.exists(os.path.join(R, p)) for p in RELEASE_TABLES.values()):
        for dst, (header, rows) in release_tables(os.path.join(R, "tables")).items():
            _write(os.path.join(R, dst), header, rows)
            n += 1
    return n


FIGURE_DATA = [paper_figures.FIG1, paper_figures.FIG2]   # written by paper_figures.figure_data


def published():
    """Every shipped path."""
    return sorted(list(SHIPPED.values()) + list(RELEASE_TABLES) + FIGURE_DATA)


def listed_path(rel):
    """None for a working path whose file is published under another name (it is listed there instead)."""
    return None if rel in SHIPPED else rel
