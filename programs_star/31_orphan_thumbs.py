#!/usr/bin/env python
"""
31_orphan_thumbs.py — diagnostic cutouts for top orphan + high-PM
candidates.  Tile-grouped FITS opens for speed.

Top 30 per detection mission (HST/JWST/Euclid orphans) + top 30 high-PM
clean stars get 5"×5" cutouts in HST F814W + JWST F115W + Euclid VIS.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize

warnings.filterwarnings('ignore', category=FITSFixedWarning)

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
THUMB_DIR = HTML / 'orphan_thumbs'
THUMB_DIR.mkdir(parents=True, exist_ok=True)

HST_DIR    = Path('/Volumes/exdisk1/data/HST/COSMOS_v2.0')
JWST_DIR   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
JWST_SCI   = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8/scidir')
EUCLID_DIR = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

CUTOUT_AS = 5.0
TOP_N_PER_MISSION = 30
TOP_N_HIGHPM = 30


def _path_for(mission: str, band: str, tile: str, fname: str = ''):
    if mission == 'HST':
        return str(HST_DIR / f'acs_I_030mas_{int(tile):03d}_sci.fits')
    if mission == 'JWST':
        if tile == 'A10' and band == 'F115W':
            return str(JWST_SCI / 'mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits')
        return str(JWST_DIR / f'mosaic_nircam_{band.lower()}_COSMOS-Web_30mas_{tile}_v1.0_i2d.fits')
    if mission == 'Euclid':
        return str(EUCLID_DIR / fname)
    return None


def assign_tiles(ra, dec, tile_df, mission, band):
    """For each (ra, dec), return tile string + file name from tile_df, or (None, None)."""
    sel = tile_df[(tile_df['mission'] == mission) & (tile_df['band'] == band)]
    out_tile = np.full(len(ra), '', dtype=object)
    out_file = np.full(len(ra), '', dtype=object)
    for r in sel.itertuples():
        ra_min  = min(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        ra_max  = max(r.ra_c1, r.ra_c2, r.ra_c3, r.ra_c4)
        dec_min = min(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        dec_max = max(r.dec_c1, r.dec_c2, r.dec_c3, r.dec_c4)
        in_box = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max) & (out_tile == '')
        out_tile[in_box] = r.tile
        out_file[in_box] = r.file
    return out_tile, out_file


def render_cutout(data, w, ra, dec, out_png, title):
    try:
        center = SkyCoord(ra * u.deg, dec * u.deg)
        cu = Cutout2D(data, center, size=CUTOUT_AS * u.arcsec, wcs=w, mode='partial', fill_value=np.nan)
    except Exception:
        return False
    img = cu.data
    if not np.isfinite(img).any():
        return False
    z = ZScaleInterval()
    try:
        v1, v2 = z.get_limits(img[np.isfinite(img)])
    except Exception:
        return False
    norm = ImageNormalize(vmin=v1, vmax=v2, stretch=AsinhStretch())
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(img, origin='lower', cmap='gray', norm=norm)
    ax.set_xticks([]); ax.set_yticks([])
    n = img.shape[0] / 2
    ax.plot([n, n], [n - 8, n + 8], 'r-', lw=0.7, alpha=0.8)
    ax.plot([n - 8, n + 8], [n, n], 'r-', lw=0.7, alpha=0.8)
    ax.set_title(title, fontsize=8)
    plt.tight_layout(pad=0.05)
    plt.savefig(out_png, dpi=120)
    plt.close()
    return True


def process_band(targets, mission, band, tile_df):
    """Tile-grouped cutouts.  Returns dict (target_idx → png_path or None)."""
    import time
    paths = {}
    t0 = time.time()
    ra  = targets['ra'].values
    dec = targets['dec'].values
    tile, fname = assign_tiles(ra, dec, tile_df, mission, band)
    targets = targets.assign(_tile=tile, _file=fname)
    groups = targets.groupby('_tile')
    for tname, grp in groups:
        ti0 = time.time()
        if not tname:
            for idx in grp.index:
                paths[idx] = None
            continue
        p = _path_for(mission, band, tname, grp['_file'].iloc[0])
        if p is None or not Path(p).exists():
            for idx in grp.index:
                paths[idx] = None
            continue
        try:
            with fits.open(p, memmap=True) as h:
                d = None; hdr = None
                for hdu in h:
                    if hdu.header.get('NAXIS', 0) == 2 and hdu.data is not None:
                        d = hdu.data; hdr = hdu.header
                        break
                if d is None:
                    for idx in grp.index:
                        paths[idx] = None
                    continue
                w = WCS(hdr)
                for r in grp.itertuples():
                    slug = f'{r.kind}_{r.Index:04d}_{mission}_{band.replace("-","_")}.png'
                    out_png = THUMB_DIR / slug
                    if out_png.exists():
                        paths[r.Index] = slug
                        continue
                    title = f'{mission} {band} t={tname}'
                    ok = render_cutout(d, w, r.ra, r.dec, out_png, title)
                    paths[r.Index] = slug if ok else None
        except Exception as e:
            print(f'  [tile fail] {mission} {band} {tname}: {e}', flush=True)
            for idx in grp.index:
                paths[idx] = None
        print(f'  {mission:6s} {band:6s} tile {tname:>5s}: {len(grp)} cutouts in {time.time()-ti0:.1f}s', flush=True)
    return paths


def main():
    tile_df = pd.read_csv(OUT / 'footprints' / 'tile_polygons.csv')

    # ---------- 1. Top orphans (3 missions × 30) ----------
    orph = pd.read_parquet(OUT / 'orphans_highconf.parquet')
    orph_top = (
        orph.sort_values('snr', ascending=False)
            .groupby('found_in').head(TOP_N_PER_MISSION)
            .reset_index(drop=True)
    )
    orph_top['kind'] = 'orph_' + orph_top['found_in'].str.replace(' ', '_')
    print(f'orphans → top {len(orph_top):,}')

    # ---------- 2. Top high-PM clean stars ----------
    clean = pd.read_parquet(OUT / 'clean_stars_v01.parquet')
    pm_top = clean.sort_values('pm_tot_mas_yr', ascending=False).head(TOP_N_HIGHPM).reset_index(drop=True)
    pm_top['kind'] = 'pm'
    pm_top['ra']   = pm_top['ra_ref']
    pm_top['dec']  = pm_top['dec_ref']
    pm_top['star_id'] = pm_top['euclid_id'].astype(str)
    pm_top['snr']  = np.nan
    pm_top['mag']  = pm_top['mag_F115W']
    pm_top['tile'] = ''
    pm_top['missing_jwst']   = False
    pm_top['missing_hst']    = False
    pm_top['missing_euclid'] = False
    pm_top['found_in']       = 'PM clean'
    print(f'high-PM clean → top {len(pm_top):,}')

    cols_need = ['ra','dec','kind','star_id','snr','mag','tile','missing_jwst',
                 'missing_hst','missing_euclid','found_in']
    all_t = pd.concat(
        [orph_top[cols_need], pm_top[cols_need]],
        ignore_index=True
    )
    print(f'TOTAL targets: {len(all_t):,}')

    bands_show = [
        ('HST',    'F814W'),
        ('JWST',   'F115W'),
        ('JWST',   'F277W'),
        ('Euclid', 'VIS'),
        ('Euclid', 'NIR-J'),
    ]
    band_paths = {}
    for mi, ba in bands_show:
        print(f'\n--- {mi} {ba} ---')
        band_paths[(mi, ba)] = process_band(all_t, mi, ba, tile_df)
        print(f'  done: {sum(p is not None for p in band_paths[(mi,ba)].values()):,}/{len(all_t):,}')

    # Build HTML
    rows_html = []
    for i, r in all_t.iterrows():
        miss = ''.join([
            'J' if r.missing_jwst   else '_',
            'H' if r.missing_hst    else '_',
            'E' if r.missing_euclid else '_',
        ])
        info = (
            f'<td><b>{r.found_in}</b><br>id {r.star_id}<br>'
            f'RA={r.ra:.5f}<br>Dec={r.dec:.5f}<br>'
            f'mag={r.mag:.2f}<br>'
            f'{"SNR=%.1f<br>" % r.snr if not pd.isna(r.snr) else ""}'
            f'tile={r.tile}<br>missing[J/H/E]={miss}</td>'
        )
        cells = [info]
        for mi, ba in bands_show:
            slug = band_paths[(mi, ba)].get(i)
            if slug:
                cells.append(f'<td><img src="orphan_thumbs/{slug}" width="180"></td>')
            else:
                cells.append('<td style="color:#aaa;width:200px;text-align:center">—</td>')
        rows_html.append('<tr>' + ''.join(cells) + '</tr>')

    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>COSMOS star v01 — orphan + high-PM thumbnails</title>'
        '<style>body{font-family:sans-serif;margin:18px}'
        'table{border-collapse:collapse}td{vertical-align:top;border:1px solid #ddd;padding:4px;font-size:11px}'
        'td:first-child{width:170px;font-family:monospace}</style>'
        '</head><body>'
        '<h1>COSMOS star catalog v01 — diagnostic thumbnails</h1>'
        '<p>Rows 1-90: orphan candidates (top 30 per detection mission, '
        'SNR &gt; 20, sharp ∈ [0.5,0.75]).  Rows 91+: top 30 high-PM stars '
        'from clean_stars_v01 (sharp 0.5-0.75, ≥4 bands, σ_µ &lt; 5).</p>'
        '<p>5″ cutouts, ZScale + asinh, red crosshair at catalog position.</p>'
        '<table><thead><tr><th>info</th>' +
        ''.join(f'<th>{m} {b}</th>' for m, b in bands_show) +
        '</tr></thead><tbody>' + ''.join(rows_html) + '</tbody></table>'
        '</body></html>'
    )
    (HTML / 'orphans_thumbs.html').write_text(html)
    print(f'\nWrote {HTML/"orphans_thumbs.html"}')


if __name__ == '__main__':
    main()
