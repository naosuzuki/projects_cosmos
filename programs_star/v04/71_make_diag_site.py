#!/usr/bin/env python
"""
71_make_diag_site.py — static HTML site for the Step 3a-③/3b stellar
diagnostics (70_step3a_diag_plots.py), organized EXACTLY like the PSF
QA site (55_make_psf_qa_site.py):

  html/diag_qa/
    index.html               band-card grid (one card per band,
                              "N tiles" status)
    F814W.html, F115W.html, …, VIS.html, NISP-Y.html, …
                              per-band pages: table with ONE ROW PER
                              TILE × 4 diagnostic panels + 3b summary
    plots/<BAND>/<tile>/*.png symlinks to the canonical PNGs
    css/style.css             copied from psf_qa (same look)

Panels per (band, tile) row:
  1. composite PSF (×9 drizzle, sqrt stretch, ±1.2″)
  2. profile cut (peak-normalized)
  3. star RA/Dec map (color/size = mag, solid n₁=0 / open n₁>0)
  4. aperture-vs-PSF photometry (APER−PSFEx, APER−DAO, PSFEx−DAO)
  5. 3b envelope sheet (per-tile, all bands — same image each row)

Re-run after producing more tiles; symlinks only, instant.
"""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
HTML_DIR = PROJECT / 'html' / 'diag_qa'
PSFQA    = PROJECT / 'html' / 'psf_qa'
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

# (page name, display label, mission, band key) — order = index order
BANDS = [
    ('F814W',  'HST ACS F814W',          'hst',    'f814w'),
    ('F115W',  'JWST NIRCam F115W (SW)', 'jwst',   'f115w'),
    ('F150W',  'JWST NIRCam F150W (SW)', 'jwst',   'f150w'),
    ('F277W',  'JWST NIRCam F277W (LW)', 'jwst',   'f277w'),
    ('F444W',  'JWST NIRCam F444W (LW)', 'jwst',   'f444w'),
    ('VIS',    'Euclid VIS',             'euclid', 'vis'),
    ('NISP-Y', 'Euclid NISP Y',          'euclid', 'nisp_y'),
    ('NISP-J', 'Euclid NISP J',          'euclid', 'nisp_j'),
    ('NISP-H', 'Euclid NISP H',          'euclid', 'nisp_h'),
]

PANELS = [
    ('diag3_psf_composite', 'Composite PSF (×9 drizzle, √ stretch)'),
    ('diag4_psf_profile',   'Profile Cut (peak-normalized)'),
    ('diag1_stars_radec',   'Star Map RA/Dec (mag-coded)'),
    ('diag2_phot_compare',  'Aperture vs PSF / DAO Photometry'),
]


def tile_sort_key(t: str):
    if t[0] in 'AB' and t[1:].isdigit():
        return (t[0], int(t[1:]))
    return ('Z', t)


def star_meta(mission: str, tile: str) -> dict:
    p = WORK / f'{mission}_chi2' / tile / 'phot' / f'star_{tile}.meta.json'
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def link(src: Path, band_page: str, tile: str) -> str | None:
    if not src.exists():
        return None
    dst = HTML_DIR / 'plots' / band_page / tile
    dst.mkdir(parents=True, exist_ok=True)
    ln = dst / src.name
    if ln.is_symlink() or ln.exists():
        ln.unlink()
    ln.symlink_to(src)
    return f'plots/{band_page}/{tile}/{src.name}'


def band_page(page: str, label: str, mission: str, band: str) -> int:
    rows = []
    tiles = sorted((p.parent.parent.name for p in
                    WORK.glob(f'{mission}_chi2/*/phot/'
                              f'phot_*.meta.json')), key=tile_sort_key)
    for tile in tiles:
        phot = WORK / f'{mission}_chi2' / tile / 'phot'
        cells = []
        n_found = 0
        for stem, _ in PANELS:
            rel = link(phot / f'{stem}_{tile}_{band}.png', page, tile)
            if rel:
                n_found += 1
                cells.append(f'<td class="panel"><a href="{rel}" '
                             f'target="_blank"><span class="thumb-box">'
                             f'<img src="{rel}" loading="lazy"/></span>'
                             f'</a></td>')
            else:
                cells.append('<td class="panel">—</td>')
        rel = link(phot / f'star_envelopes_{tile}.png', page, tile)
        cells.append(f'<td class="panel"><a href="{rel}" target="_blank">'
                     f'<span class="thumb-box"><img src="{rel}" '
                     f'loading="lazy"/></span></a></td>'
                     if rel else '<td class="panel">—</td>')
        if n_found == 0 and rel is None:
            continue
        m = star_meta(mission, tile)
        info = (f"<br><span style='font-weight:400;font-size:11px;"
                f"color:#666'>{m.get('n_is_star', '—')}★ / "
                f"{m.get('n_sources', '—')}</span>" if m else '')
        rows.append(f'<tr><td class="tile-name">{tile}{info}</td>'
                    + ''.join(cells) + '</tr>')
    head = ''.join(f'<th>{t}</th>' for _, t in PANELS)
    page_html = dedent(f'''<!doctype html>
        <html><head>
          <meta charset="utf-8">
          <title>Stellar diagnostics — {label}</title>
          <link rel="stylesheet" href="css/style.css">
        </head><body>
        <header>
          <h1>Stellar diagnostics — {label}
          <span style="font-weight:400; font-size:14px;">
          [<a href="index.html">&larr; back</a>]</span></h1>
        </header>
        <div class="wrap">
          <p>Stars only (Step 3b is_star).  Solid = isolated (n&#8321;=0),
          open = neighbour within 1&Prime;.  Tile cell shows
          is_star / total sources.</p>
          <table class="tiles">
            <thead><tr><th>Tile</th>{head}<th>3b Envelopes
            (all bands)</th></tr></thead>
            <tbody>
            {''.join(rows)}
            </tbody>
          </table>
        </div>
        </body></html>''')
    (HTML_DIR / f'{page}.html').write_text(page_html)
    return len(rows)


def main():
    (HTML_DIR / 'css').mkdir(parents=True, exist_ok=True)
    css = PSFQA / 'css' / 'style.css'
    if css.exists():
        (HTML_DIR / 'css' / 'style.css').write_text(css.read_text())

    cards = []
    for page, label, mission, band in BANDS:
        n = band_page(page, label, mission, band)
        status = (f'<div class="status done">{n} tiles processed</div>'
                  if n else '<div class="status">no tiles yet</div>')
        cards.append(f'<div class="band-card">\n'
                     f'  <a href="{page}.html">{label}</a>\n'
                     f'  {status}\n</div>\n')
        print(f'{page:8s} {n} tiles')

    idx = dedent(f'''<!doctype html>
        <html><head>
          <meta charset="utf-8">
          <title>Stellar diagnostics QA — COSMOS cross-mission v04</title>
          <link rel="stylesheet" href="css/style.css">
        </head><body>
        <header>
          <h1>Stellar Diagnostics — COSMOS Cross-Mission Catalog v04 /
          Step 3a-③ &amp; 3b</h1>
        </header>
        <div class="wrap">
          <p>Per-band stellar diagnostics (stars = Step 3b is_star).
          Click a band to see one row per tile: composite PSF
          (&times;9 drizzle, sqrt stretch, &plusmn;1.2&Prime;), profile
          cut, star RA/Dec map, aperture-vs-PSF photometry, and the 3b
          classifier envelopes.  Click any panel for the full image.
          Sibling site: <a href="../psf_qa/index.html">PSF-model QA
          (Step 3a-①)</a>.</p>
          <div class="band-grid">
          {''.join(cards)}
          </div>
        </div>
        </body></html>''')
    (HTML_DIR / 'index.html').write_text(idx)
    print(f'[save] {HTML_DIR}/index.html')


if __name__ == '__main__':
    main()
