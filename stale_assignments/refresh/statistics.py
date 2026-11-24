# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Prashant Chaudhary
"""The refresh decision test: statistics, decisions and the second decision date, with the 99% county-cluster percentile
bootstrap; and the sub-two-year currency curve.

Input: the per-stratum export (CSV with header cost, td, file, cset, policy, bp, n_persons, n_records, n_units, cutoff,
n_counties, cells; cells = 'county|n_u|s_e|n_ref|rec_ref;...'). Output: refresh_results.csv (long format) and
refresh_summary.txt. Deterministic: integer sums, exact rational comparisons for the decisions, fixed formatting.

Statistics, for every cost rule, decision date, file, county set, policy and budget:
  E                    sum_county s_e / sum_county n_u (mean over U in the county set of e_fresh if refreshed, else e_stored)
  D_vs_age_x_size      E(pi, b) - E(age x size, b)
  wrong_avoided_per_1000_refreshed_vs_random   1000 (S_pi - S_random) / R_pi; undefined at b = 0
  D_per_1000_refreshed 1000 (S_pi - S_age_x_size) / R_pi; undefined at b = 0
  oracle_gap_share     (E(pi, b) - E(age x size, b)) / (E(oracle, b) - E(age x size, b)); undefined where the denominator
                       is 0 (b = 0 and b = 1)
  persons_refreshed, records_refreshed, n_U, n_units, cutoff   design counts, point only (fixed by the budget, no interval)
Intervals: 2,000 replicates; each replicate draws len(counties) county ids with replacement from the sorted integer
county ids of the county set; one draw sequence per county set (the generator is re-seeded for each set) shared by every
cost rule, decision date, file, policy, budget and statistic. Orderings and refreshed sets are fixed, so the intervals are
conditional on them. Bounds: the sorted replicates at indices 10 and 1989. A ratio undefined in some replicate gives no
bounds (n_rep_undefined says how many).
Decisions, stated before the test (primary county set, person cost, full file, decision date 2024-11-05),
D(pi, b) = E(pi, b) - E(age x size, b):
  comember_age_table: lo99 D(0.25) >= 0.002; hi99 D(0.10) >= 0 and hi99 D(0.50) >= 0; D(0.25) > 0 on the complete-case
  file (point). coarse_table: the same. Sanity check (no decision): E(age x size, 0.25) - E(oldest, 0.25) with its interval.
  The comparisons are exact: D = (S_pi - S_as) / N with integer sums (the same U, so the same N), compared as fractions.
"""
import argparse, csv, os, random
from collections import defaultdict
from fractions import Fraction
import numpy as np

from ..constants import REFRESH_BOOTSTRAP_SEED

SEED = REFRESH_BOOTSTRAP_SEED
REPS, LO, HI = 2000, 10, 1989
POLICIES = ['random', 'oldest', 'age_x_size', 'coarse_table', 'comember_age_table', 'lowest_p_comember_age_table', 'oracle']
BPS = [0, 1000, 2500, 5000, 10000]
BUDGET = {0: '0', 1000: '0.10', 2500: '0.25', 5000: '0.50', 10000: '1'}
COSTS, TDS, FILES, CSETS = ['person', 'record'], ['2024-11-05', '2022-11-08'], ['full', 'complete_case'], ['primary80', 'all100']
NCTY = {'primary80': 80, 'all100': 100}
PRIMARY = ('person', '2024-11-05', 'full', 'primary80')


def read_agg(path):
    """{(cost, td, file, cset, policy, bp): dict(counties, n_u, s_e, n_ref, rec_ref arrays, design counts)}."""
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            key = (r['cost'], r['td'], r['file'], r['cset'], r['policy'], int(r['bp']))
            assert key not in out, f'duplicate stratum {key}'
            cells = [tuple(int(x) for x in c.split('|')) for c in r['cells'].split(';')]
            a = np.array(cells, dtype=np.int64)
            assert len(a) == int(r['n_counties']), key
            out[key] = dict(counties=tuple(int(c) for c in a[:, 0]), n_u=a[:, 1], s_e=a[:, 2], n_ref=a[:, 3], rec_ref=a[:, 4],
                            n_persons=int(r['n_persons']), n_records=int(r['n_records']), n_units=int(r['n_units']),
                            cutoff=int(r['cutoff']))
    return out


def check(agg):
    """Structural checks; stops on the first failure."""
    want = {(c, t, f, s, p, b) for c in COSTS for t in TDS for f in FILES for s in CSETS for p in POLICIES for b in BPS}
    assert set(agg) == want, f'strata missing {sorted(want - set(agg))[:5]} or extra {sorted(set(agg) - want)[:5]}'
    for s in CSETS:
        cl = {v['counties'] for k, v in agg.items() if k[3] == s}
        assert len(cl) == 1, f'{s}: county lists differ between strata'
        c = next(iter(cl)); assert len(c) == NCTY[s] and list(c) == sorted(c), f'{s}: {len(c)} counties or unsorted'
    for c in COSTS:
        for t in TDS:
            for f in FILES:
                for s in CSETS:
                    base = agg[('person', t, f, s, 'random', 0)]
                    for p in POLICIES:
                        for b in BPS:
                            v = agg[(c, t, f, s, p, b)]
                            assert np.array_equal(v['n_u'], base['n_u']), ('n_u differs', c, t, f, s, p, b)
                            if b in (0, 10000):
                                assert np.array_equal(v['s_e'], agg[(c, t, f, s, 'random', b)]['s_e']), ('E at b = 0 or 1 differs', c, t, f, s, p, b)
                            refreshed = v['n_ref'].sum() if c == 'person' else v['rec_ref'].sum()
                            assert refreshed == v['cutoff'] == (v['n_units'] * b) // 10000, ('refreshed != floor(b x units)', c, t, f, s, p, b)
                            if b == 0: assert v['n_ref'].sum() == 0
                            if b == 10000: assert v['n_ref'].sum() == v['n_persons']


def draws(counties, seed=SEED, reps=REPS):
    """Replicate weights (reps x len(counties), int64), county ids sorted ascending."""
    rng = random.Random(seed)
    idx = {c: i for i, c in enumerate(counties)}
    W = np.zeros((reps, len(counties)), dtype=np.int64)
    for rep in range(reps):
        draw = defaultdict(int)
        for _ in counties: draw[rng.choice(counties)] += 1
        for c, m in draw.items(): W[rep, idx[c]] = m
    return W


def ratio(num, den):
    """Point or replicate ratio of integer sums; NaN where den == 0."""
    num = np.asarray(num, dtype=np.float64); den = np.asarray(den, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den != 0, num / np.where(den != 0, den, 1), np.nan)


def bounds(rep):
    rep = np.asarray(rep, dtype=np.float64)
    bad = int(np.isnan(rep).sum())
    if bad: return None, None, bad
    v = np.sort(rep, kind='stable'); return float(v[LO]), float(v[HI]), 0


def frac_bounds(nums, dens):
    """Exact 99% bounds of replicate ratios nums/dens (integers): sorted Fractions at indices 10 and 1989."""
    v = sorted(Fraction(int(n), int(d)) for n, d in zip(nums, dens))
    return v[LO], v[HI]


def run(agg_path, out_dir):
    """refresh_results.csv and refresh_summary.txt in out_dir from the per-stratum export at agg_path."""
    a = argparse.Namespace(agg=agg_path, out_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)
    agg = read_agg(a.agg); check(agg)
    W = {s: draws(agg[('person', TDS[0], 'full', s, 'random', 0)]['counties']) for s in CSETS}
    rows = []
    def add(section, key, policy, bp, stat, point, lo=None, hi=None, bad=0):
        c, t, f, s = key
        rows.append(dict(section=section, td=t, file=f, cost=c, county_set=s, policy=policy, budget=BUDGET[bp] if bp is not None else '',
                         stat=stat, point=point, lo99=lo, hi99=hi, n_rep_undefined=bad))
    exact = {}  # (key, policy, bp) -> (S, N, R, WS, WN, WR) integer sums for the decisions
    for c in COSTS:
        for t in TDS:
            for f in FILES:
                for s in CSETS:
                    key = (c, t, f, s); sec = 'statistics' if t == '2024-11-05' else 'second_date'
                    for b in BPS:
                        V = {p: agg[(c, t, f, s, p, b)] for p in POLICIES}
                        N = int(V['random']['n_u'].sum()); WN = W[s] @ V['random']['n_u']
                        S = {p: int(V[p]['s_e'].sum()) for p in POLICIES}; WS = {p: W[s] @ V[p]['s_e'] for p in POLICIES}
                        for p in POLICIES:
                            R = int(V[p]['n_ref'].sum()); WR = W[s] @ V[p]['n_ref']
                            exact[(key, p, b)] = (S[p], N, R, WS[p], WN, WR)
                            E = S[p] / N; lo, hi, bad = bounds(ratio(WS[p], WN)); add(sec, key, p, b, 'E', E, lo, hi, bad)
                            d = ratio(S[p] - S['age_x_size'], N); lo, hi, bad = bounds(ratio(WS[p] - WS['age_x_size'], WN))
                            add(sec, key, p, b, 'D_vs_age_x_size', float(d), lo, hi, bad)
                            pt = ratio(1000 * (S[p] - S['random']), R); lo, hi, bad = bounds(ratio(1000 * (WS[p] - WS['random']), WR))
                            add(sec, key, p, b, 'wrong_avoided_per_1000_refreshed_vs_random', float(pt), lo, hi, bad)
                            pt = ratio(1000 * (S[p] - S['age_x_size']), R); lo, hi, bad = bounds(ratio(1000 * (WS[p] - WS['age_x_size']), WR))
                            add(sec, key, p, b, 'D_per_1000_refreshed', float(pt), lo, hi, bad)
                            pt = ratio(S[p] - S['age_x_size'], S['oracle'] - S['age_x_size'])
                            lo, hi, bad = bounds(ratio(WS[p] - WS['age_x_size'], WS['oracle'] - WS['age_x_size']))
                            add(sec, key, p, b, 'oracle_gap_share', float(pt), lo, hi, bad)
                            add(sec, key, p, b, 'persons_refreshed', R)
                            add(sec, key, p, b, 'records_refreshed', int(V[p]['rec_ref'].sum()))
                            add(sec, key, p, b, 'n_U', N)
                            add(sec, key, p, b, 'n_units', V[p]['n_units'])
                            add(sec, key, p, b, 'cutoff', V[p]['cutoff'])
    # ---- decisions (exact) ----
    def D(key, p, b):
        S, N, _, WS, WN, _ = exact[(key, p, b)]; Sa, _, _, WSa, _, _ = exact[(key, 'age_x_size', b)]
        lo, hi = frac_bounds(WS - WSa, WN)
        return Fraction(S - Sa, N), lo, hi
    cc = ('person', '2024-11-05', 'complete_case', 'primary80')
    dec = {}
    for name, p in (('comember_age_table', 'comember_age_table'), ('coarse_table', 'coarse_table')):
        d25, lo25, hi25 = D(PRIMARY, p, 2500); d10, lo10, hi10 = D(PRIMARY, p, 1000); d50, lo50, hi50 = D(PRIMARY, p, 5000)
        dcc = D(cc, p, 2500)[0]
        c1 = lo25 >= Fraction(2, 1000); c2 = hi10 >= 0 and hi50 >= 0; c3 = dcc > 0
        dec[name] = dict(p=p, c1=c1, c2=c2, c3=c3, ok=c1 and c2 and c3, d25=d25, lo25=lo25, hi25=hi25, hi10=hi10, hi50=hi50, dcc=dcc)
        add('decision', PRIMARY, p, 2500, f'{name}.D', float(d25), float(lo25), float(hi25))
        add('decision', PRIMARY, p, 1000, f'{name}.D', float(d10), float(lo10), float(hi10))
        add('decision', PRIMARY, p, 5000, f'{name}.D', float(d50), float(lo50), float(hi50))
        add('decision', cc, p, 2500, f'{name}.D', float(dcc))
        add('decision', PRIMARY, p, 2500, f'{name}.c1 lo99 D(0.25) >= 0.002 (1 = met)', int(c1))
        add('decision', PRIMARY, p, None, f'{name}.c2 hi99 D(0.10) >= 0 and hi99 D(0.50) >= 0 (1 = met)', int(c2))
        add('decision', cc, p, 2500, f'{name}.c3 D(0.25) > 0 on the complete-case file (1 = met)', int(c3))
        add('decision', PRIMARY, p, None, f'{name}.pass (1 = pass)', int(dec[name]['ok']))
    S_as, N, _, WS_as, WN, _ = exact[(PRIMARY, 'age_x_size', 2500)]; S_old, _, _, WS_old, _, _ = exact[(PRIMARY, 'oldest', 2500)]
    lo, hi = frac_bounds(WS_as - WS_old, WN); d3 = Fraction(S_as - S_old, N)
    add('decision', PRIMARY, 'age_x_size', 2500, 'sanity.D E(age_x_size) - E(oldest) (no decision)', float(d3), float(lo), float(hi))
    os.makedirs(a.out_dir, exist_ok=True)
    fmt = lambda v: '' if v is None else (str(v) if isinstance(v, (int, np.integer)) else ('' if v != v else f'{v:.10f}'))
    cols = ['section', 'td', 'file', 'cost', 'county_set', 'policy', 'budget', 'stat', 'point', 'lo99', 'hi99', 'n_rep_undefined']
    with open(os.path.join(a.out_dir, 'refresh_results.csv'), 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(cols)
        for r in rows: w.writerow([fmt(r[c]) if c in ('point', 'lo99', 'hi99') else r[c] for c in cols])
    # ---- short text summary ----
    E = lambda key, p, b: exact[(key, p, b)][0] / exact[(key, p, b)][1]
    get = lambda key, p, b, stat: next(r for r in rows if (r['cost'], r['td'], r['file'], r['county_set']) == key
                                       and r['policy'] == p and r['budget'] == BUDGET[b] and r['stat'] == stat)
    L = ['Refresh decision test: primary county set (80 counties outside development), person cost,',
         f'full file, T = 2024-11-05, scored at 2026-11-03. |U| = {exact[(PRIMARY, "random", 0)][1]:,}; '
         f'|U_T| = {agg[("person",) + PRIMARY[1:] + ("random", 0)]["n_persons"]:,}. 99% county bootstrap, {REPS} reps.', '']
    for name in ('comember_age_table', 'coarse_table'):
        d = dec[name]
        L.append(f"{name}: {'PASS' if d['ok'] else 'FAIL'}. D(0.25) = {float(d['d25']):+.4f} "
                 f"[{float(d['lo25']):+.4f}, {float(d['hi25']):+.4f}] (lower bound >= 0.002: {'yes' if d['c1'] else 'no'}); "
                 f"upper bounds at 0.10 and 0.50: {float(d['hi10']):+.4f}, {float(d['hi50']):+.4f} (both >= 0: {'yes' if d['c2'] else 'no'}); "
                 f"complete-case D(0.25) = {float(d['dcc']):+.4f} (> 0: {'yes' if d['c3'] else 'no'}).")
    L.append(f'Sanity check (no decision): E(age x size, 0.25) - E(oldest first, 0.25) = {float(d3):+.4f} [{float(lo):+.4f}, {float(hi):+.4f}].')
    L += ['', 'E by policy and budget (primary stratum); in brackets D vs age x size, then the share of the oracle gap closed:']
    for b in BPS:
        parts = []
        for p in POLICIES:
            g = get(PRIMARY, p, b, 'oracle_gap_share')['point']
            parts.append(f"{p} {E(PRIMARY, p, b):.4f} ({get(PRIMARY, p, b, 'D_vs_age_x_size')['point']:+.4f}"
                         + ('' if g != g else f', {g:.2f}') + ')')
        L.append(f'  b = {BUDGET[b]}: ' + '; '.join(parts))
    L += ['', 'Every policy, budget, cost rule, county set, file and the second decision date T = 2022-11-08 (section 7 rows) are in',
          'refresh_results.csv, with persons and records refreshed.']
    with open(os.path.join(a.out_dir, 'refresh_summary.txt'), 'w', newline='') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


def curve_stats(src, out):
    """The sub-two-year curve (kappa_P, sigma, kappa_S, mu per (t, s), 99% county bootstrap) from
    curve_cells.csv; checks the (2016-11-08, 2018-11-06) kappa_P against Table 2's 0.671."""
    cells = defaultdict(dict); days = {}
    for r in csv.DictReader(open(src, newline='', encoding='utf-8')):
        key = (r['t_dt'], r['s_dt']); days[key] = int(r['days'])
        cells[key][int(r['county0'])] = np.array([int(r[c]) for c in ('n', 'n_exact', 'n_stay', 'n_exact_stay')], dtype=np.int64)
    counties = sorted(next(iter(cells.values())))
    assert len(counties) == 60 and all(sorted(v) == counties for v in cells.values()), 'county lists differ or not 60'
    W = draws(tuple(counties))
    rows = []
    for key in sorted(cells):
        A = np.stack([cells[key][c] for c in counties])  # counties x 4
        def stats(x):  # x: (..., 4) sums
            n, ne, ns, nes = (x[..., i].astype(np.float64) for i in range(4))
            return dict(kappa_P=ne / n, sigma=ns / n, kappa_S=nes / ns, mu=(ne - nes) / (n - ns))
        pt = stats(A.sum(0)); rep = stats(W @ A)
        for s in ('kappa_P', 'sigma', 'kappa_S', 'mu'):
            v = np.sort(rep[s], kind='stable')
            rows.append(dict(t_dt=key[0], s_dt=key[1], days=days[key], months=f'{days[key] / 30.4375:.4f}', n=int(A[:, 0].sum()),
                             stat=s, point=f'{float(pt[s]):.10f}', lo99=f'{float(v[LO]):.10f}', hi99=f'{float(v[HI]):.10f}'))
    chk = [r for r in rows if (r['t_dt'], r['s_dt'], r['stat']) == ('2016-11-08', '2018-11-06', 'kappa_P')][0]
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    return f'kappa_P(2016-11-08, 2018-11-06) = {float(chk["point"]):.4f} (Table 2 prints 0.671); {len(rows)} rows -> {out}'
