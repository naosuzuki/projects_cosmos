#!/usr/bin/env python
"""
55_make_psf_qa_site.py — generate a static HTML site for PSF QA.

Layout
------
  html/psf_qa/
    index.html                    band selector (HST, JWST 4 bands, Euclid)
    F814W.html, F115W.html, ...   per-band pages
    plots/<band>/<tile>/*.png     symlinks to canonical diagnostic PNGs
    css/style.css

For each (band, tile) we expose 6 thumbnails in a single row:
  1. mag_vs_halflight
  2. mag_vs_chi2
  3. saturation_peak
  4. psf_samples
  5. psf_residuals
  6. hist_n_means

Click any thumbnail → opens the full PNG in a new tab (vanilla HTML5,
no JS dependencies).

Source plots are taken from
  /Volumes/exdisk1/data/photometry_v04/<instrument>/<tile>/psf/
or, for the F115W A4 pilot, from /Volumes/exdisk1/data/photometry_v04/pilot/

Re-run this script after producing more (band, tile) outputs to refresh
the site.
"""
from __future__ import annotations
import argparse, shutil, os
from pathlib import Path
from textwrap import dedent

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
HTML_DIR = PROJECT / 'html' / 'psf_qa'
PILOT    = Path('/Volumes/exdisk1/data/photometry_v04/pilot')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

# (band, label, instrument_key) — order matters for index page
BANDS = [
    ('F814W', 'HST ACS F814W',         'hst_acs_f814w'),
    ('F115W', 'JWST NIRCam F115W (SW)', 'jwst_nircam_f115w'),
    ('F150W', 'JWST NIRCam F150W (SW)', 'jwst_nircam_f150w'),
    ('F277W', 'JWST NIRCam F277W (LW)', 'jwst_nircam_f277w'),
    ('F444W', 'JWST NIRCam F444W (LW)', 'jwst_nircam_f444w'),
    ('VIS',   'Euclid VIS',             'euclid_vis'),
]

# canonical plot filenames (after symlinking into the site)
PANELS = [
    ('mag_vs_halflight',  'Mag vs Half-Light Radius'),
    ('mag_vs_chi2',       'Mag vs χ² (neighbour-coded)'),
    ('saturation_peak',   'Peak Count vs Mag (saturation)'),
    ('psf_samples',       'PSF Samples Mosaic'),
    ('psf_residuals',     'PSF Residuals Mosaic'),
    ('hist_n_means',      'Neighbour Count Histograms'),
]

# Mapping from canonical panel name → source-file glob to copy/link.
# Picks the most-recent matching version in the pilot dir (for F115W A4)
PILOT_SRC = {
    'mag_vs_halflight':  'mag_vs_halflight_v9.png',
    'mag_vs_chi2':       'mag_vs_chi2_neighborcoded_v14_rainbow.png',
    'saturation_peak':   'saturation_peak_v4_f115w.png',
    'psf_samples':       'psf_samples_v19.png',
    'psf_residuals':     'psf_residuals_v19.png',
    'hist_n_means':      'hist_n_means_v3.png',
}


def _natkey(name: str):
    """Natural sort key so tiles order A1,A2,..,A9,A10,B1,..,B10 (numeric-aware)
    rather than lexicographic A1,A10,A2.  Euclid numeric IDs sort numerically too.
    Each chunk is a (rank,int,str) tuple so digit/non-digit chunks never compare
    str-vs-int (stray non-tile dir entries would otherwise raise TypeError)."""
    import re
    out = []
    for t in re.findall(r'\d+|\D+', name):
        out.append((0, int(t), '') if t.isdigit() else (1, 0, t))
    return out


def discover_tiles(instrument: str) -> list[str]:
    """Find tiles processed for a given instrument.

    A tile is "processed" if its psf/ dir exists and contains at least one
    canonical QA plot (produced by 56_make_psf_qa_plots.py).
    """
    inst_root = WORK / instrument
    if not inst_root.exists():
        return []
    tiles = []
    for p in sorted(inst_root.iterdir(), key=lambda p: _natkey(p.name)):
        if not p.is_dir(): continue
        psf = p / 'psf'
        if not psf.exists(): continue
        # require at least one canonical plot
        if any((psf / f'{panel}.png').exists() for panel, _ in PANELS):
            tiles.append(p.name)
    return tiles


def link_plots_for_tile(band: str, tile: str, instrument: str) -> dict[str, Path]:
    """Symlink the band/tile diagnostic PNGs into the site's plots/ tree.
    Returns dict {panel_name: relative_html_path} for what was linked."""
    out_dir = HTML_DIR / 'plots' / band / tile
    out_dir.mkdir(parents=True, exist_ok=True)
    linked = {}
    for panel, _ in PANELS:
        target = None
        # production location (preferred): produced by 56_make_psf_qa_plots.py
        cand = WORK / instrument / tile / 'psf' / f'{panel}.png'
        if cand.exists():
            target = cand
        # fallback to F115W A4 pilot if production is not yet there
        elif band == 'F115W' and tile == 'A4':
            src = PILOT / PILOT_SRC[panel]
            if src.exists():
                target = src
        if target is None:
            continue
        dst = out_dir / f'{panel}.png'
        # use absolute symlink so it survives moving the html dir
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(target, dst)
        linked[panel] = f'plots/{band}/{tile}/{panel}.png'
    return linked


def write_css():
    (HTML_DIR / 'css').mkdir(parents=True, exist_ok=True)
    (HTML_DIR / 'css' / 'style.css').write_text(dedent('''
        body {
          margin: 0; padding: 0;
          font-family: "Times New Roman", Times, serif;
          background: #fafafa; color: #222;
        }
        header { background:#2b3e50; color:#fff; padding:16px 28px; }
        header h1 { margin:0; font-size: 22px; }
        header a { color:#9ec3e6; text-decoration:none; }
        header a:hover { text-decoration:underline; }
        header .back-btn {
          display:inline-block; background:#4a90e2; color:#fff;
          padding:6px 14px; border-radius:5px; font-size:14px;
          margin-left:14px; vertical-align:middle;
        }
        header .back-btn:hover { background:#2c70c4; text-decoration:none; }
        .wrap { padding: 18px 28px; }
        .band-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 18px; margin-top: 18px;
        }
        .band-card {
          background:#fff; border:1px solid #ddd; border-radius:8px;
          padding:16px; text-align:center;
          transition: box-shadow .15s, transform .15s;
        }
        .band-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.12); transform: translateY(-1px); }
        .band-card a { color:#2b3e50; text-decoration:none; font-size:16px; font-weight:600; }
        .band-card .status { font-size:13px; color:#666; margin-top:6px; }
        .band-card .status.done { color:#2c8f5f; }
        .band-card .status.tbd  { color:#aa6; }
        .band-card .status.todo { color:#bbb; }

        table.tiles {
          width: 100%; border-collapse: collapse; margin-top: 8px;
          background: #fff;
        }
        table.tiles th, table.tiles td {
          border:1px solid #ddd; padding:6px; vertical-align: middle;
          text-align: center;
        }
        table.tiles thead th { background:#eef0f3; font-size:13px; }
        table.tiles tbody td.tile-name {
          font-weight:600; background:#f7f7f7; min-width:60px;
        }
        table.tiles tbody td.panel { width: 14%; }
        /* Uniform thumbnail box: fixed aspect-ratio container,
           images scaled with object-fit:contain so all panels render
           the same size regardless of their native aspect ratio. */
        table.tiles tbody td.panel .thumb-box {
          display:block; width:100%; aspect-ratio: 3 / 2;
          background:#fff; border:1px solid #eee; overflow:hidden;
          transition: box-shadow .15s;
        }
        table.tiles tbody td.panel .thumb-box img {
          width: 100%; height: 100%; object-fit: contain;
          display:block;
        }
        table.tiles tbody td.panel a:hover .thumb-box {
          box-shadow: 0 0 0 2px #4a90e2;
        }
        table.tiles tbody td.empty { color:#bbb; font-size: 12px; }
        .panel-label { font-size: 11px; color: #555; margin-top:2px; }

        /* Viewer page (full-size single image with back button) */
        .viewer-wrap { padding: 18px; text-align:center; }
        .viewer-wrap img {
          max-width: 100%; max-height: calc(100vh - 100px);
          border:1px solid #ccc; background:#fff;
        }
        .viewer-title { font-size:14px; color:#555; margin: 8px 0; }
    ''').strip())


def write_index(band_status: dict[str, int]):
    cards = []
    for band, label, _ in BANDS:
        n = band_status.get(band, 0)
        if n > 0:
            status_cls, status_txt = 'done', f'{n} tile{"s" if n!=1 else ""} processed'
        else:
            status_cls, status_txt = 'todo', 'not yet processed'
        cards.append(dedent(f'''
            <div class="band-card">
              <a href="{band}.html">{label}</a>
              <div class="status {status_cls}">{status_txt}</div>
            </div>
        '''))
    body = dedent('''
        <!doctype html>
        <html><head>
          <meta charset="utf-8">
          <title>PSF QA — COSMOS-Web cross-mission catalog v04</title>
          <link rel="stylesheet" href="css/style.css">
        </head><body>
        <header>
          <h1>PSF QA — COSMOS Cross-Mission Catalog v04 / Step 3a</h1>
        </header>
        <div class="wrap">
          <p>Per-band PSF model diagnostics.  Click a band to see one row
          per tile, with 6 diagnostic panels each.  Click any panel to
          open the full-resolution image.</p>
          <div class="band-grid">
    ''').strip()
    body += '\n'.join(cards)
    body += dedent('''
          </div>
        </div>
        </body></html>
    ''').strip()
    (HTML_DIR / 'index.html').write_text(body)


def write_viewer_page(band: str, tile: str, panel: str, panel_title: str,
                      img_rel: str):
    """Write a viewer page that shows a single full-size image with a back
    button to the band page."""
    viewer_dir = HTML_DIR / 'viewer' / band / tile
    viewer_dir.mkdir(parents=True, exist_ok=True)
    # img_rel is relative to html/psf_qa/  (e.g. plots/F115W/A4/foo.png),
    # but the viewer lives at html/psf_qa/viewer/F115W/A4/ → go up 3 dirs
    img_from_viewer = f'../../../{img_rel}'
    back_href       = f'../../../{band}.html'
    html = dedent(f'''
        <!doctype html>
        <html><head>
          <meta charset="utf-8">
          <title>PSF QA — {band} {tile} — {panel_title}</title>
          <link rel="stylesheet" href="../../../css/style.css">
        </head><body>
        <header>
          <h1>{band} / {tile} — {panel_title}
            <a class="back-btn" href="{back_href}">← Back to {band}</a>
          </h1>
        </header>
        <div class="viewer-wrap">
          <div class="viewer-title">Click image for native pixel size</div>
          <a href="{img_from_viewer}" target="_blank">
            <img src="{img_from_viewer}" alt="{panel}">
          </a>
        </div>
        </body></html>
    ''').strip()
    out = viewer_dir / f'{panel}.html'
    out.write_text(html)
    # return path relative to band page (which lives at html/psf_qa/)
    return f'viewer/{band}/{tile}/{panel}.html'


def write_band_page(band: str, label: str, instrument: str, tiles: list[str]):
    # build table header
    thead_panels = ''.join(f'<th>{title}</th>' for _, title in PANELS)
    thead = f'<tr><th>Tile</th>{thead_panels}</tr>'

    # build rows
    rows = []
    for tile in tiles:
        linked = link_plots_for_tile(band, tile, instrument)
        cells = []
        for panel, panel_title in PANELS:
            if panel in linked:
                rel = linked[panel]
                viewer_rel = write_viewer_page(band, tile, panel, panel_title, rel)
                cells.append(
                    f'<td class="panel"><a href="{viewer_rel}">'
                    f'<span class="thumb-box"><img src="{rel}" alt="{panel}"/></span>'
                    f'</a></td>'
                )
            else:
                cells.append(f'<td class="empty">—</td>')
        rows.append(f'<tr><td class="tile-name">{tile}</td>{"".join(cells)}</tr>')

    if not rows:
        body_extra = '<p style="color:#888; font-style:italic;">No tiles processed yet for this band.</p>'
        rows_html = ''
    else:
        body_extra = ''
        rows_html = '\n'.join(rows)

    html = dedent(f'''
        <!doctype html>
        <html><head>
          <meta charset="utf-8">
          <title>PSF QA — {label}</title>
          <link rel="stylesheet" href="css/style.css">
        </head><body>
        <header>
          <h1>PSF QA — {label}  <span style="font-weight:400; font-size:14px;">[<a href="index.html">← back</a>]</span></h1>
        </header>
        <div class="wrap">
          {body_extra}
          <table class="tiles">
            <thead>{thead}</thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        </body></html>
    ''').strip()
    (HTML_DIR / f'{band}.html').write_text(html)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean', action='store_true',
                   help='Wipe plots/ symlinks before rebuilding')
    args = p.parse_args()

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    if args.clean and (HTML_DIR / 'plots').exists():
        shutil.rmtree(HTML_DIR / 'plots')

    write_css()

    band_status = {}
    for band, label, instrument in BANDS:
        tiles = discover_tiles(instrument)
        band_status[band] = len(tiles)
        write_band_page(band, label, instrument, tiles)
        print(f'  {band:6s}  {label:32s}  tiles={tiles or "[]"}')

    write_index(band_status)
    print(f'\n  → site at: {HTML_DIR / "index.html"}')


if __name__ == '__main__':
    main()
