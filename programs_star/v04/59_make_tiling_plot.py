#!/usr/bin/env python
"""
59_make_tiling_plot.py — per-survey tiling map for the PSF-QA front page.

Adapted from projects_hsc/hostgalxy/programs/fig_tiling.py (Suzuki): one
small-multiple panel per survey showing that survey's on-sky tile footprints
(4 WCS corners from each mosaic's header), coloured/filled where a validated
PSF model exists, over the wide Gaia DR3 star field for context.  Every panel
carries the JWST COSMOS-Web A/B tile grid as a faint common reference so the
program's core scope is visible at any survey's scale.

Tile lists and source files come from the Step-3a meta JSONs
(/Volumes/exdisk1/data/photometry_v04/<instrument>/<tile>/psf/psf_*.meta.json,
key 'sci_source_file') — no hardcoded data paths.  Corner reads are header-only
and cached in programs_star/csv_footprints/tile_footprints.json, so re-runs
after new tiles are incremental.

Output: html/psf_qa/tiling_cosmos.png  (embedded at the top of index.html by
55_make_psf_qa_site.py).

Usage:  ./59_make_tiling_plot.py [--refresh-cache]
"""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

warnings.filterwarnings('ignore')

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')
GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/gaia_dr3_cosmos_wide.csv')
CACHE    = PROJECT / 'programs_star' / 'csv_footprints' / 'tile_footprints.json'
OUT_PNG  = PROJECT / 'html' / 'psf_qa' / 'tiling_cosmos.png'

# (panel title, [instrument prefixes to union], footprint-source instrument,
#  colour, label tiles?)  The footprint instrument is one band whose grid the
# survey shares; 'built' = tile has a PSF model in ANY of the union'd bands.
PANELS = [
    ('HST ACS $F814W$',            ['hst_acs_f814w'],   ['hst_acs_f814w'],           'tab:red',    True),
    ('JWST NIRCam (4 bands)',      ['jwst_nircam_'],    ['jwst_nircam_f115w',
                                                         'jwst_nircam_f277w'],       'tab:orange', True),
    ('Euclid VIS',                 ['euclid_vis'],      ['euclid_vis'],              'tab:purple', False),
    ('Euclid NISP $YJH$',          ['euclid_nisp_'],    ['euclid_nisp_y'],           'tab:pink',   False),
    ('HSC $grizy$ (tract 9813)',   ['hsc_'],            ['hsc_g', 'hsc_i'],          'tab:green',  False),
    ('LS DR10 $griz$ (DECam)',     ['lsdr10_'],         ['lsdr10_g', 'lsdr10_r'],    'tab:blue',   False),
    ('PS1 $grizy$ (rings.v3)',     ['ps1_'],            ['ps1_g', 'ps1_r'],          'tab:cyan',   False),
    ('SDSS $ugriz$ (DR17 frames)', ['sdss_'],           ['sdss_r', 'sdss_g'],        'tab:olive',  False),
    ('unWISE $W1\\,W2$ (neo7)',    ['unwise_'],         ['unwise_w1'],               'tab:brown',  True),
]
REF_INSTRUMENT = 'jwst_nircam_f115w'          # faint common reference grid


def footprint(path, ext):
    """4 sky corners (deg) of a FITS image, header-only (handles bz2/fz/gz)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    h = fits.getheader(path, ext)
    if h.get('NAXIS1') in (None, 0):
        h = fits.getheader(path, ext + 1)
    nx, ny = h['NAXIS1'], h['NAXIS2']
    w = WCS(h)
    c = w.all_pix2world([[0, 0], [nx, 0], [nx, ny], [0, ny]], 0)
    return np.asarray(c)[:, :2]


def load_metas():
    """{instrument: {tile: meta_dict}} for every meta JSON on disk."""
    out = {}
    for m in WORK.glob('*/*/psf/psf_*.meta.json'):
        inst, tile = m.parts[5], m.parts[6]
        try:
            out.setdefault(inst, {})[tile] = json.loads(m.read_text())
        except Exception:
            pass
    return out


def _source_for(metas, inst, tile):
    """(path, ext) for a tile's mosaic.  Pilot-era metas (JWST A3/A4/A5)
    predate the 'sci_source_file' key — reconstruct the path from a sibling
    tile of the SAME instrument by substituting the tile token (mosaics are
    uniformly named)."""
    d = metas.get(inst, {}).get(tile, {})
    src, ext = d.get('sci_source_file'), int(d.get('sci_source_ext') or 0)
    if src and Path(src).exists():
        return src, ext
    for t2, d2 in metas.get(inst, {}).items():
        s2 = d2.get('sci_source_file')
        if t2 == tile or not s2 or f'_{t2}_' not in Path(s2).name:
            continue
        cand = Path(s2).parent / Path(s2).name.replace(f'_{t2}_', f'_{tile}_')
        if cand.exists():
            return str(cand), int(d2.get('sci_source_ext') or 0)
    return None, 0


def build_cache(metas, refresh):
    """{instrument: {tile: {'corners': 4x2 list}}}.  Each panel may list
    several source bands: later ones only fill tiles the earlier ones missed
    (e.g. a no-coverage stub in PS1 g falls back to r)."""
    cache = {}
    if CACHE.exists() and not refresh:
        cache = json.loads(CACHE.read_text())
    n_read = 0
    for _, _, src_list, _, _ in PANELS + [(None, None, [REF_INSTRUMENT], None, None)]:
        primary = src_list[0]
        got = cache.setdefault(primary, {})
        # union of tiles known to any source band, footprint from first that works
        tiles = sorted({t for inst in src_list for t in metas.get(inst, {})})
        for tile in tiles:
            if tile in got:
                continue
            for inst in src_list:
                if tile not in metas.get(inst, {}):
                    continue
                src, ext = _source_for(metas, inst, tile)
                if not src:
                    continue
                try:
                    got[tile] = {'corners': footprint(src, ext).tolist()}
                    n_read += 1
                    break
                except Exception as e:                 # noqa: BLE001
                    print(f'  [warn] {inst}/{tile}: {e}')
            else:
                print(f'  [warn] {primary}/{tile}: no readable source in {src_list}')
    if n_read:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))
    print(f'footprints: {sum(len(v) for v in cache.values())} cached '
          f'({n_read} new header reads)')
    return cache


def built_tiles(metas, prefixes):
    """Tiles with a real PSF model (n_model_stars>0) in ANY matching band."""
    out = set()
    for inst, tiles in metas.items():
        if any(inst.startswith(p) for p in prefixes):
            out |= {t for t, d in tiles.items()
                    if (d.get('n_model_stars') or d.get('psf_accepted')
                        or d.get('n_psf_stars') or 0) > 0}
    return out


def make_tilemaps(cache, force=False):
    """Per-(survey grid, tile) mini map for the band pages' Tile column
    (their fig_tilemaps.py pattern): JWST A/B reference grid faint, all of
    this survey's tiles as thin outlines, THIS tile filled.  Written to
    html/psf_qa/tilemaps/<grid_instrument>__<tile>.png; bands of one survey
    share a grid so 55_ maps e.g. hsc_r -> hsc_g's maps."""
    outdir = OUT_PNG.parent / 'tilemaps'
    outdir.mkdir(parents=True, exist_ok=True)
    ref = [np.asarray(v['corners']) for v in cache.get(REF_INSTRUMENT, {}).values()]
    allc = np.vstack([np.asarray(v['corners'])
                      for inst in cache.values() for v in inst.values()])
    ramin, ramax = allc[:, 0].min() - 0.1, allc[:, 0].max() + 0.1
    dmin, dmax = allc[:, 1].min() - 0.1, allc[:, 1].max() + 0.1
    n = 0
    for _, _, src_list, col, _ in PANELS:
        grid = src_list[0]
        tiles = {t: np.asarray(v['corners']) for t, v in cache.get(grid, {}).items()}
        for tile, fp in tiles.items():
            png = outdir / f'{grid}__{tile}.png'
            if png.exists() and not force:
                continue
            fig, ax = plt.subplots(figsize=(2.3, 2.3))
            for p in ref:
                ax.add_patch(Polygon(p, closed=True, fill=False, ec='0.85', lw=0.4))
            for p2 in tiles.values():
                ax.add_patch(Polygon(p2, closed=True, fill=False, ec=col,
                                     lw=0.6, alpha=0.55))
            ax.add_patch(Polygon(fp, closed=True, facecolor=col, alpha=0.55,
                                 ec=col, lw=1.6))
            ax.set_xlim(ramax, ramin)
            ax.set_ylim(dmin, dmax)
            ax.set_aspect('equal', 'box')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('0.7')
            fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
            fig.savefig(png, dpi=220)
            plt.close(fig)
            n += 1
    print(f'tilemaps: {n} written to {outdir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh-cache', action='store_true')
    ap.add_argument('--tilemaps', action='store_true',
                    help='also render the per-tile mini maps for band pages')
    ap.add_argument('--force-tilemaps', action='store_true')
    a = ap.parse_args()
    t0 = time.time()

    metas = load_metas()
    cache = build_cache(metas, a.refresh_cache)
    if a.tilemaps or a.force_tilemaps:
        make_tilemaps(cache, force=a.force_tilemaps)

    # Gaia context points
    gra = gdec = np.array([])
    if GAIA_CSV.exists():
        import pandas as pd
        g = pd.read_csv(GAIA_CSV)
        gra, gdec = g['ra'].to_numpy(), g['dec'].to_numpy()

    ref = {t: np.asarray(v['corners']) for t, v in cache.get(REF_INSTRUMENT, {}).items()}

    # global extent over every survey's corners
    allc = np.vstack([np.asarray(v['corners'])
                      for inst in cache.values() for v in inst.values()])
    ramin, ramax = allc[:, 0].min() - 0.1, allc[:, 0].max() + 0.1
    dmin, dmax = allc[:, 1].min() - 0.1, allc[:, 1].max() + 0.1

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 15.0))
    axes = axes.ravel()
    fig.suptitle('COSMOS multi-mission stellar catalog — per-survey PSF-model tiling',
                 fontsize=15, y=0.995)
    for ax, (title, prefixes, src_list, col, label_tiles) in zip(axes, PANELS):
        src_inst = src_list[0]
        # faint common reference: the JWST COSMOS-Web A/B grid
        for t, p in ref.items():
            ax.add_patch(Polygon(p, closed=True, fill=False, ec='0.85', lw=0.5, zorder=1))
        if gra.size:
            ax.scatter(gra, gdec, s=0.4, c='0.6', alpha=0.30, lw=0, zorder=2,
                       label='Gaia DR3 stars')
        built = built_tiles(metas, prefixes)
        polys = cache.get(src_inst, {})
        n_on = 0
        many = len(polys) > 40                     # dense grids: thinner lines
        for t, v in sorted(polys.items()):
            p = np.asarray(v['corners'])
            on = t in built
            n_on += on
            ax.add_patch(Polygon(
                p, closed=True,
                facecolor=col, alpha=(0.10 if many else 0.22) if on else 0.0,
                ec=col if on else '0.55', lw=(0.6 if many else 1.3) if on else 0.7,
                ls='-' if on else ':', zorder=3))
            if label_tiles:
                ax.text(p[:, 0].mean(), p[:, 1].mean(), t, fontsize=6,
                        ha='center', va='center', color='0.25', zorder=5)
        ax.plot([], [], color=col, lw=6, alpha=0.4,
                label=f'{n_on} tiles with PSF model')
        ax.set_title(title, fontsize=11.5)
        ax.set_xlim(ramax, ramin)                  # RA increases left
        ax.set_ylim(dmin, dmax)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('R.A. (deg)')
        ax.set_ylabel('Dec. (deg)')
        ax.legend(loc='upper right', fontsize=7.5, framealpha=0.92, markerscale=4)
        ax.grid(True, color='0.93', lw=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {OUT_PNG}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
