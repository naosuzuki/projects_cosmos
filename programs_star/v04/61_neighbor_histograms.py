#!/usr/bin/env python
"""
61_neighbor_histograms.py — neighbour-count + radial neighbour histograms for
the PSF QA, per (instrument, tile), for TWO reference populations, with
summable recording CSVs.  Standalone / decoupled from the PSF mass production
(does its own fast no-VIGNET SExtractor detection), so it never touches the
live pipeline and works uniformly on every tile.

Two reference populations (computed separately, per your "1 and 2 separately"):
  (1) PSF stars       — the selected PSF stars (the kept stars_<suffix>.fits)
  (2) all point srcs  — CLASS_STAR>0.8 & SNR_WIN>100 in the detection catalog
Neighbours = ALL detections within the radius (self excluded).

For each population it records, to programs_star/csv_neighbors/ :
  <inst>_<tile>_nhist_<pop>.csv  — neighbour-COUNT histogram, bins 0..30
        columns: n, within_1arcsec, within_3arcsec, within_5arcsec
        (how many reference objects have n neighbours within 1/3/5")
  <inst>_<tile>_rhist_<pop>.csv  — RADIAL histogram, 0.0–5.0", 0.1" bins
        columns: r_lo, r_hi, n_objects   (neighbour count summed over refs)
Both are raw counts → 62_aggregate_neighbor_histograms.py sums them across all
tiles into the "true distribution".  A per-tile QA PNG is also written.

x-axis of the neighbour-count histogram is fixed to 30 (uniform, your spec).

Usage:
  ./61_neighbor_histograms.py --instrument euclid_vis --tile 101538497
  ./61_neighbor_histograms.py --all          # every processed tile
  ./61_neighbor_histograms.py --all --jobs 4
"""
from __future__ import annotations
import argparse, csv, subprocess, sys, tempfile, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
from scipy.spatial import cKDTree

warnings.simplefilter('ignore')
PROJECT = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS = PROJECT / 'configs'
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
JWST    = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
SCIDIR  = JWST / 'scidir'
HSTDIR  = Path('/Volumes/exdisk1/data/HST/COSMOS_ACS2005')
EUC     = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')
OUTDIR  = PROJECT / 'programs_star' / 'csv_neighbors'
OUTDIR.mkdir(parents=True, exist_ok=True)

PIX = {'hst_acs_f814w': 0.030, 'euclid_vis': 0.10}   # arcsec/px; jwst added below
for _b in ('f115w', 'f150w', 'f277w', 'f444w'):
    PIX[f'jwst_nircam_{_b}'] = 0.030

NMAX = 30           # neighbour-count histogram x-max (uniform)
RBINS = np.round(np.arange(0.0, 5.0001, 0.1), 4)   # radial bins 0..5" / 0.1"
POPS = ('psfstars', 'pointsrc')

rcParams['font.family'] = 'serif'; rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']


def source_and_config(instrument, tile):
    """Return (sex_image, sex_config, filter_conv) for a fast detection."""
    if instrument.startswith('jwst_nircam_'):
        band = instrument.split('_')[-1]
        i2d = JWST / f'mosaic_nircam_{band}_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits'
        img = f'{i2d}[1]'
        if i2d.exists():
            try:
                with fits.open(i2d) as h:
                    _ = h['SCI'].header['NAXIS1']
            except Exception:
                img = None
        else:
            img = None
        if img is None:
            m = sorted(SCIDIR.glob(f'mosaic_nircam_{band}_COSMOS-Web_30mas_{tile}_v*_sci.fits'))
            if not m:
                return None, None, None
            img = str(m[0])
        return img, CONFIGS / f'{instrument}.sex', CONFIGS / 'default.conv'
    if instrument == 'hst_acs_f814w':
        m = sorted(HSTDIR.glob(f'mosaic_cosmos_web_*_30mas_tile_{tile}_hst_acs_wfc_f814w_drz.fits'))
        return (str(m[0]) if m else None), CONFIGS / 'hst_acs_f814w.sex', CONFIGS / 'default.conv'
    if instrument == 'euclid_vis':
        m = sorted(EUC.glob(f'EUC_MER_BGSUB-MOSAIC-VIS_TILE{tile}*.fits'))
        return (str(m[0]) if m else None), CONFIGS / 'euclid_vis.sex', CONFIGS / 'gauss_2.0_5x5.conv'
    return None, None, None


def detect(instrument, tile):
    """Fast no-VIGNET SExtractor detection → (x, y, cs, snr, flg) in pixels."""
    img, sexcfg, conv = source_and_config(instrument, tile)
    if not img:
        return None
    with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tf:
        cat = tf.name
    cmd = ['sex', img, '-c', str(sexcfg),
           '-CATALOG_NAME', cat, '-CATALOG_TYPE', 'FITS_LDAC',
           '-PARAMETERS_NAME', str(CONFIGS / 'pass1_novignet.param'),
           '-FILTER_NAME', str(conv), '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
           '-WEIGHT_TYPE', 'NONE', '-VERBOSE_TYPE', 'QUIET']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        Path(cat).unlink(missing_ok=True)
        return None
    d = fits.open(cat)[2].data
    out = (np.asarray(d['X_IMAGE'], float), np.asarray(d['Y_IMAGE'], float),
           np.asarray(d['CLASS_STAR'], float), np.asarray(d['SNR_WIN'], float),
           np.asarray(d['FLAGS'], int))
    Path(cat).unlink(missing_ok=True)
    return out


def ref_positions(instrument, tile, suffix, det):
    """(x,y) of the two reference populations."""
    x, y, cs, snr, flg = det
    pops = {}
    pops['pointsrc'] = (x[(cs > 0.8) & (snr > 100)], y[(cs > 0.8) & (snr > 100)])
    stars = WORK / instrument / tile / 'psf' / f'stars_{suffix}.fits'
    if stars.exists():
        s = fits.open(stars)[2].data
        pops['psfstars'] = (np.asarray(s['X_IMAGE'], float), np.asarray(s['Y_IMAGE'], float))
    else:
        pops['psfstars'] = (np.array([]), np.array([]))
    return pops


def histograms(refx, refy, tree, allxy, pix):
    """Return (ncount_hist[31,3], radial_hist[50]) for one reference set."""
    nhist = np.zeros((NMAX + 1, 3), int)          # rows n=0..30, cols 1/3/5"
    rhist = np.zeros(len(RBINS) - 1, int)
    if len(refx) == 0:
        return nhist, rhist, 0
    r5 = 5.0 / pix
    for xr, yr in zip(refx, refy):
        idx = tree.query_ball_point([xr, yr], r5)
        d = np.hypot(allxy[idx, 0] - xr, allxy[idx, 1] - yr) * pix   # arcsec
        d = d[d > 1e-6]                                              # drop self
        for j, rad in enumerate((1.0, 3.0, 5.0)):
            c = min(int(np.sum(d <= rad)), NMAX)
            nhist[c, j] += 1
        rhist += np.histogram(d, bins=RBINS)[0]
    return nhist, rhist, len(refx)


def write_csvs(instrument, tile, pop, nhist, rhist, nref):
    base = f'{instrument}_{tile}_{pop}'
    with open(OUTDIR / f'{base}_nhist.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['# n_ref', nref])
        w.writerow(['n', 'within_1arcsec', 'within_3arcsec', 'within_5arcsec'])
        for n in range(NMAX + 1):
            w.writerow([n, *nhist[n].tolist()])
    with open(OUTDIR / f'{base}_rhist.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['# n_ref', nref])
        w.writerow(['r_lo_arcsec', 'r_hi_arcsec', 'n_objects'])
        for i in range(len(rhist)):
            w.writerow([round(RBINS[i], 3), round(RBINS[i+1], 3), int(rhist[i])])


def plot_tile(instrument, tile, results):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    col = {'psfstars': 'red', 'pointsrc': 'steelblue'}
    for pop, (nhist, rhist, nref) in results.items():
        if nref == 0:
            continue
        ax[0].step(range(NMAX + 1), nhist[:, 2], where='mid', color=col[pop],
                   label=f'{pop} (N={nref}), ≤5″')
        ax[0].step(range(NMAX + 1), nhist[:, 0], where='mid', color=col[pop],
                   ls='--', alpha=0.6, label=f'{pop}, ≤1″')
        rc = 0.5 * (RBINS[:-1] + RBINS[1:])
        ax[1].step(rc, rhist, where='mid', color=col[pop], label=f'{pop} (N={nref})')
    ax[0].set_xlim(0, NMAX); ax[0].set_xlabel('Neighbour count'); ax[0].set_ylabel('N reference objects')
    ax[0].set_title(f'{instrument} {tile} — neighbour-count histogram (x-max=30)', fontsize=11)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlim(0, 5); ax[1].set_xlabel('Radius [arcsec]'); ax[1].set_ylabel('N neighbours (0.1″ bins)')
    ax[1].set_title(f'{instrument} {tile} — radial neighbour histogram', fontsize=11)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(WORK / instrument / tile / 'psf' / 'neighbor_hist.png', dpi=100, bbox_inches='tight')
    plt.close(fig)


def process(instrument, tile, suffix):
    det = detect(instrument, tile)
    if det is None:
        return f'{instrument}/{tile}: detect FAILED'
    x, y, cs, snr, flg = det
    allxy = np.column_stack([x, y])
    tree = cKDTree(allxy)
    pix = PIX[instrument]
    pops = ref_positions(instrument, tile, suffix, det)
    results = {}
    for pop in POPS:
        rx, ry = pops[pop]
        nh, rh, nref = histograms(rx, ry, tree, allxy, pix)
        write_csvs(instrument, tile, pop, nh, rh, nref)
        results[pop] = (nh, rh, nref)
    plot_tile(instrument, tile, results)
    return (f'{instrument}/{tile}: dets={len(x)} '
            f'psfstars={results["psfstars"][2]} pointsrc={results["pointsrc"][2]}')


def discover():
    """All (instrument, tile, suffix) with a finished PSF (meta present)."""
    items = []
    for inst_dir in sorted(WORK.glob('*/')):
        inst = inst_dir.name
        if inst not in PIX:
            continue
        for tdir in sorted(inst_dir.glob('*/')):
            tile = tdir.name
            metas = list((tdir / 'psf').glob('psf_*.meta.json'))
            if metas:
                suffix = metas[0].name[len('psf_'):-len('.meta.json')]
                items.append((inst, tile, suffix))
    return items


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument'); p.add_argument('--tile'); p.add_argument('--suffix')
    p.add_argument('--all', action='store_true')
    p.add_argument('--jobs', type=int, default=4)
    return p.parse_args()


def main():
    a = parse_args()
    if a.all:
        items = discover()
        print(f'processing {len(items)} tiles, {a.jobs}-way → {OUTDIR}')
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(process, i, t, s) for i, t, s in items]
            for k, f in enumerate(as_completed(futs), 1):
                print(f'  [{k}/{len(items)}] {f.result()}', flush=True)
    else:
        if not (a.instrument and a.tile):
            sys.exit('need --instrument and --tile (or --all)')
        suffix = a.suffix or (a.instrument.split('_')[-1]
                              if a.instrument.startswith('jwst_nircam_') else a.tile)
        print(process(a.instrument, a.tile, suffix))


if __name__ == '__main__':
    main()
