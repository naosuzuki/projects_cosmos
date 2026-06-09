#!/usr/bin/env python
"""
62_aggregate_neighbor_histograms.py — sum the per-tile neighbour-count and
radial histogram CSVs written by 61_ into the survey-wide "true distribution",
kept separate for the two reference populations (PSF stars / all point srcs).

Reads  : programs_star/csv_neighbors/<inst>_<tile>_{nhist,rhist}_<pop>.csv
Writes : programs_star/csv_neighbors/TOTAL_<pop>_nhist.csv   (summed, + N_ref, N_tiles)
         programs_star/csv_neighbors/TOTAL_<pop>_rhist.csv
         programs_star/csv_neighbors/TOTAL_neighbor_histograms.png

Usage  :  ./62_aggregate_neighbor_histograms.py
          ./62_aggregate_neighbor_histograms.py --missions euclid_vis,hst_acs_f814w
"""
from __future__ import annotations
import argparse, csv, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

DIR = Path('/Users/suzuki/github/projects_cosmos/programs_star/csv_neighbors')
POPS = ('psfstars', 'pointsrc')
NMAX = 30
rcParams['font.family'] = 'serif'; rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']


def read_nhist(path):
    nref = 0; rows = []
    for r in csv.reader(open(path)):
        if not r: continue
        if r[0].startswith('# n_ref'): nref = int(r[1]); continue
        if r[0] == 'n': continue
        rows.append([int(float(v)) for v in r])
    arr = np.array(rows, int)               # (31,4): n, w1, w3, w5
    return arr[:, 1:], nref                  # counts (31,3)


def read_rhist(path):
    nref = 0; rlo = []; rhi = []; cnt = []
    for r in csv.reader(open(path)):
        if not r: continue
        if r[0].startswith('# n_ref'): nref = int(r[1]); continue
        if r[0].startswith('r_lo'): continue
        rlo.append(float(r[0])); rhi.append(float(r[1])); cnt.append(int(r[2]))
    return np.array(rlo), np.array(rhi), np.array(cnt, int), nref


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--missions', default=None,
                   help='comma list of instrument prefixes to include (default: all)')
    return p.parse_args()


def main():
    a = parse_args()
    pref = a.missions.split(',') if a.missions else None
    def keep(fn):
        return pref is None or any(Path(fn).name.startswith(p + '_') for p in pref)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    col = {'psfstars': 'red', 'pointsrc': 'steelblue'}
    summary = []
    for pop in POPS:
        nfiles = [f for f in glob.glob(str(DIR / f'*_{pop}_nhist.csv')) if keep(f)] \
                 + [f for f in glob.glob(str(DIR / f'*_nhist_{pop}.csv')) if keep(f)]
        rfiles = [f for f in glob.glob(str(DIR / f'*_{pop}_rhist.csv')) if keep(f)] \
                 + [f for f in glob.glob(str(DIR / f'*_rhist_{pop}.csv')) if keep(f)]
        if not nfiles:
            continue
        ntot = np.zeros((NMAX + 1, 3), int); nref_tot = 0
        for f in nfiles:
            c, nref = read_nhist(f); ntot += c; nref_tot += nref
        rlo = rhi = None; rtot = None; rref = 0
        for f in rfiles:
            lo, hi, cnt, nref = read_rhist(f)
            if rtot is None:
                rlo, rhi, rtot = lo, hi, cnt.copy()
            else:
                rtot += cnt
            rref += nref
        ntiles = len(nfiles)
        # write TOTAL CSVs
        with open(DIR / f'TOTAL_{pop}_nhist.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['# pop', pop, '# N_ref', nref_tot, '# N_tiles', ntiles])
            w.writerow(['n', 'within_1arcsec', 'within_3arcsec', 'within_5arcsec'])
            for n in range(NMAX + 1): w.writerow([n, *ntot[n].tolist()])
        with open(DIR / f'TOTAL_{pop}_rhist.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(['# pop', pop, '# N_ref', rref, '# N_tiles', len(rfiles)])
            w.writerow(['r_lo_arcsec', 'r_hi_arcsec', 'n_objects'])
            for i in range(len(rtot)): w.writerow([round(rlo[i], 3), round(rhi[i], 3), int(rtot[i])])
        summary.append(f'{pop}: {ntiles} tiles, {nref_tot} reference objects')
        # plot
        ax[0].step(range(NMAX + 1), ntot[:, 2], where='mid', color=col[pop], label=f'{pop} ≤5″ (N={nref_tot})')
        ax[0].step(range(NMAX + 1), ntot[:, 0], where='mid', color=col[pop], ls='--', alpha=0.6, label=f'{pop} ≤1″')
        rc = 0.5 * (rlo + rhi)
        ax[1].step(rc, rtot, where='mid', color=col[pop], label=f'{pop} (N={rref})')
    ax[0].set_xlim(0, NMAX); ax[0].set_xlabel('Neighbour count'); ax[0].set_ylabel('N reference objects (all tiles)')
    ax[0].set_title('Summed neighbour-count histogram (x-max=30)', fontsize=11); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlim(0, 5); ax[1].set_xlabel('Radius [arcsec]'); ax[1].set_ylabel('N neighbours (0.1″ bins, all tiles)')
    ax[1].set_title('Summed radial neighbour histogram', fontsize=11); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(DIR / 'TOTAL_neighbor_histograms.png', dpi=110, bbox_inches='tight'); plt.close(fig)
    print('aggregated → ' + str(DIR))
    for s in summary: print('  ' + s)


if __name__ == '__main__':
    main()
