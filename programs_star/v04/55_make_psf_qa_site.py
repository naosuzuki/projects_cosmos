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
    ('NISP-Y', 'Euclid NISP Y',          'euclid_nisp_y'),
    ('NISP-J', 'Euclid NISP J',          'euclid_nisp_j'),
    ('NISP-H', 'Euclid NISP H',          'euclid_nisp_h'),
    ('HSC-g',  'HSC g (s23b deepCoadd)', 'hsc_g'),
    ('HSC-r',  'HSC r (s23b deepCoadd)', 'hsc_r'),
    ('HSC-i',  'HSC i (s23b deepCoadd)', 'hsc_i'),
    ('HSC-z',  'HSC z (s23b deepCoadd)', 'hsc_z'),
    ('HSC-y',  'HSC y (s23b deepCoadd)', 'hsc_y'),
    ('LS-g',   'LS DR10 g (DECam)',      'lsdr10_g'),
    ('LS-r',   'LS DR10 r (DECam)',      'lsdr10_r'),
    ('LS-i',   'LS DR10 i (DECam)',      'lsdr10_i'),
    ('LS-z',   'LS DR10 z (DECam)',      'lsdr10_z'),
    ('PS1-g',  'PS1 g (rings.v3 stack)', 'ps1_g'),
    ('PS1-r',  'PS1 r (rings.v3 stack)', 'ps1_r'),
    ('PS1-i',  'PS1 i (rings.v3 stack)', 'ps1_i'),
    ('PS1-z',  'PS1 z (rings.v3 stack)', 'ps1_z'),
    ('PS1-y',  'PS1 y (rings.v3 stack)', 'ps1_y'),
    ('SDSS-u', 'SDSS u (DR17 frame)',    'sdss_u'),
    ('SDSS-g', 'SDSS g (DR17 frame)',    'sdss_g'),
    ('SDSS-r', 'SDSS r (DR17 frame)',    'sdss_r'),
    ('SDSS-i', 'SDSS i (DR17 frame)',    'sdss_i'),
    ('SDSS-z', 'SDSS z (DR17 frame)',    'sdss_z'),
    ('unWISE-W1', 'unWISE W1 (neo7 coadd)', 'unwise_w1'),
    ('unWISE-W2', 'unWISE W2 (neo7 coadd)', 'unwise_w2'),
]

# canonical plot filenames (after symlinking into the site)
PANELS = [
    ('mag_vs_halflight',  'Mag vs Half-Light Radius'),
    ('mag_vs_chi2',       'Mag vs χ² (neighbour-coded)'),
    ('saturation_peak',   'Peak Count vs Mag (saturation)'),
    ('psf_samples',       'PSF Samples Mosaic'),
    ('psf_residuals',     'PSF Residuals Mosaic'),
    ('hist_n_means',      'Neighbour Δmag vs Separation'),
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


# ── EXTERNAL record import (user directive 2026-07-05): Euclid VIS/NISP and
# HSC pages SHOW the hostgalxy project's updated record (built there with the
# 10xFWHM/gate recipe; VIS 30 / NISP-Y 30 / J,H 28 tiles; HSC = 10 tracts)
# instead of rebuilding here.  Our v04 model files on exdisk1 are untouched
# (the photometry chain still consumes them).  Re-running this script re-links
# whatever their site currently holds — their record stays authoritative.
EXTERNAL_ROOT = Path('/Users/suzuki/github/projects_hsc/html/qa_psf_cosmos')
EXTERNAL_WORK = Path('/Users/suzuki/data/psf_v01')   # hostgalxy WORK root (SSD)
# COSMOS tracts (their HSC/CLAUDS work is tract-level and mixes SXDS tracts
# in the same WORK dirs — import only the COSMOS field)
_COSMOS_TRACTS = {'9569', '9570', '9571', '9572', '9812', '9813', '9814',
                  '10054', '10055', '10056'}
EXTERNAL_SRC = {
    'VIS':    ('site', 'euclid_vis'),
    'NISP-Y': ('site', 'euclid_y'),
    'NISP-J': ('site', 'euclid_j'),
    'NISP-H': ('site', 'euclid_h'),
    'HSC-g':  ('work', 'hsc_g'),
    'HSC-r':  ('work', 'hsc_r2'),
    'HSC-i':  ('work', 'hsc_i2'),
    'HSC-z':  ('work', 'hsc_z'),
    'HSC-y':  ('work', 'hsc_y'),
}


def external_dir(band: str, tile: str) -> Path:
    style, key = EXTERNAL_SRC[band]
    return (EXTERNAL_ROOT / 'plots' / key / tile if style == 'site'
            else EXTERNAL_WORK / key / tile / 'psf')


def external_tiles(band: str) -> list[str]:
    style, key = EXTERNAL_SRC[band]
    if style == 'site':
        root = EXTERNAL_ROOT / 'plots' / key
        if not root.is_dir():
            return []
        return [p.name for p in sorted(root.iterdir(), key=lambda p: _natkey(p.name))
                if p.is_dir()]
    root = EXTERNAL_WORK / key
    if not root.is_dir():
        return []
    return [p.name for p in sorted(root.iterdir(), key=lambda p: _natkey(p.name))
            if p.is_dir() and p.name in _COSMOS_TRACTS
            and (p / 'psf' / 'psf_samples.png').exists()]


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
    # external-record bands (Euclid/HSC) source every panel from the
    # hostgalxy site copy — see EXTERNAL_SRC.
    ext_dir = external_dir(band, tile) if band in EXTERNAL_SRC else None
    for panel, _ in PANELS:
        target = None
        if ext_dir is not None:
            for name in ([f'{panel}.png', 'neighbour_scatter.png']
                         if panel == 'hist_n_means' else [f'{panel}.png']):
                if (ext_dir / name).exists():
                    target = ext_dir / name; break
        # the neighbour slot PREFERS the hostgalxy Δmag-vs-separation scatter;
        # the old neighbour-count histogram is only a fallback for tiles not
        # yet regenerated.
        elif panel == 'hist_n_means' and \
                (WORK / instrument / tile / 'psf' / 'neighbour_scatter.png').exists():
            target = WORK / instrument / tile / 'psf' / 'neighbour_scatter.png'
        elif (WORK / instrument / tile / 'psf' / f'{panel}.png').exists():
            target = WORK / instrument / tile / 'psf' / f'{panel}.png'
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
        # mosaics: also link the ≤16-row *_thumb (band-page cell) when present
        if panel in ('psf_samples', 'psf_residuals'):
            tsrc = Path(str(target).replace('.png', '_thumb.png'))
            if tsrc.exists():
                tdst = out_dir / f'{panel}_thumb.png'
                if tdst.is_symlink() or tdst.exists():
                    tdst.unlink()
                os.symlink(tsrc, tdst)
                linked[f'{panel}_thumb'] = f'plots/{band}/{tile}/{panel}_thumb.png'
    # PSFEx/Piff variants of the two MODEL-DEPENDENT panels (73_make_piff_panels.py):
    #   mag_vs_chi2 → mag_vs_chi2_{psfex,piff}.png   (fair raw-stamp χ², both)
    #   psf_residuals → psf_residuals_piff.png (+_thumb); PSFEx state = default
    src_dir = WORK / instrument / tile / 'psf'
    for extra in ('mag_vs_chi2_psfex', 'mag_vs_chi2_piff',
                  'psf_residuals_piff', 'psf_residuals_piff_thumb'):
        s = src_dir / f'{extra}.png'
        if s.exists():
            d = out_dir / f'{extra}.png'
            if d.is_symlink() or d.exists():
                d.unlink()
            os.symlink(s, d)
            linked[extra] = f'plots/{band}/{tile}/{extra}.png'
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
        table.tiles tbody td.tile-name img.tilemap {
          display:block; width:104px; height:auto; margin:6px auto 0;
          border:1px solid #e2e2e2; border-radius:4px; background:#fff;
          cursor:zoom-in; transition:box-shadow .12s;
        }
        table.tiles tbody td.tile-name img.tilemap:hover {
          box-shadow:0 0 0 2px #4a90e2;
        }
        .psf-toggle { display:flex; gap:12px; align-items:center; flex-wrap:wrap;
          padding:8px 28px; background:#eef0f3; border-bottom:1px solid #ddd;
          font-size:14px; }
        .psf-toggle button { background:#5cb85c; color:#fff; border:0;
          border-radius:5px; padding:6px 13px; font-size:14px; cursor:pointer; }
        .psf-toggle button:hover { background:#449d44; }
        .psf-toggle button:disabled { background:#bbb; cursor:not-allowed; }
        .psf-toggle .hint { color:#666; }
        #psflbl.pex { color:#2c70c4; }
        #psflbl.pif { color:#c0392b; }
        .tm-lightbox { display:none; position:fixed; inset:0; z-index:1000;
          cursor:zoom-out; background:rgba(0,0,0,0.82);
          align-items:center; justify-content:center; }
        .tm-lightbox img { max-width:88vw; max-height:88vh; background:#fff;
          padding:6px; border:2px solid #fff; border-radius:6px;
          box-shadow:0 6px 30px rgba(0,0,0,0.5); }
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
    ''').strip()
    body += '\n  <div class="band-grid">\n'
    body += '\n'.join(cards)
    body += '\n  </div>\n'
    # per-survey tiling map (59_make_tiling_plot.py) BELOW the band buttons
    if (HTML_DIR / 'tiling_cosmos.png').exists():
        body += dedent('''
          <h2 style="margin:24px 0 6px;">Survey tiling</h2>
          <p style="margin-top:0;">On-sky tile footprints per survey
          (filled = PSF model built), over the wide Gaia DR3 star field.
          The faint grid in every panel is the JWST COSMOS-Web A/B tiling.
          Click to open full resolution.</p>
          <a href="tiling_cosmos.png" target="_blank"><img src="tiling_cosmos.png"
             alt="per-survey tiling map"
             style="width:100%;max-width:1200px;border:1px solid #ccc;
                    border-radius:6px;background:#fff;"></a>
        ''')
    body += dedent('''
        </div>
        </body></html>
    ''').strip()
    (HTML_DIR / 'index.html').write_text(body)


def write_viewer_page(band: str, tile: str, panel: str, panel_title: str,
                      img_rel: str, tiles: list[str]):
    """Write a viewer page showing one full-size image, with prev/next-tile
    pagination (same band + same panel, wrapping A1↔…↔B10) and a back button.
    `tiles` is the band's full, naturally-ordered tile list."""
    viewer_dir = HTML_DIR / 'viewer' / band / tile
    viewer_dir.mkdir(parents=True, exist_ok=True)
    # img_rel is relative to html/psf_qa/  (e.g. plots/F115W/A4/foo.png),
    # but the viewer lives at html/psf_qa/viewer/F115W/A4/ → go up 3 dirs
    img_from_viewer = f'../../../{img_rel}'
    back_href       = f'../../../{band}.html'
    # prev/next tile (wrapping), linking to the SAME panel one dir over
    i = tiles.index(tile)
    prev_tile = tiles[(i - 1) % len(tiles)]
    next_tile = tiles[(i + 1) % len(tiles)]
    prev_href = f'../{prev_tile}/{panel}.html'
    next_href = f'../{next_tile}/{panel}.html'
    # mosaic panels get the zoom + samples⇄residuals swap viewer
    if panel in ('psf_samples', 'psf_residuals'):
        return write_mosaic_viewer_page(band, tile, panel, panel_title,
                                        prev_tile, next_tile, prev_href,
                                        next_href, back_href, viewer_dir)
    # line plots: zoombox viewer with TOGGLE click — one click zooms in at the
    # cursor, the next click returns to fit (like the tile minimaps).
    html = dedent(f'''
        <!doctype html>
        <html><head>
          <meta charset="utf-8">
          <title>PSF QA — {band} {tile} — {panel_title}</title>
          <link rel="stylesheet" href="../../../css/style.css">
          <style>
            .tile-nav {{ display:flex; gap:10px; align-items:center; margin-top:8px; flex-wrap:wrap; }}
            .tile-nav .cur {{ font-weight:600; }}
            .tile-nav .spacer {{ flex:1; }}
            .zoomtools {{ display:flex; gap:8px; align-items:center; padding:7px 28px;
              background:#eef0f3; border-bottom:1px solid #ddd; font-size:14px; flex-wrap:wrap; }}
            .zoomtools button {{ background:#4a90e2; color:#fff; border:0; border-radius:5px;
              padding:6px 12px; font-size:14px; cursor:pointer; }}
            .zoomtools button:hover {{ background:#2c70c4; }}
            .zoomtools .pct {{ font-weight:600; min-width:46px; }}
            .zoomtools .hint {{ color:#666; }}
            .zoombox {{ overflow:auto; width:100%; height:calc(100vh - 152px); background:#fff; }}
            .zoombox img {{ display:block; width:100%; cursor:zoom-in; }}
          </style>
        </head><body>
        <header>
          <h1>{band} / {tile} — {panel_title}</h1>
          <div class="tile-nav">
            <a class="back-btn" href="{prev_href}">◀ Prev tile ({prev_tile})</a>
            <span class="cur">{tile}</span>
            <a class="back-btn" href="{next_href}">Next tile ({next_tile}) ▶</a>
            <span class="spacer"></span>
            <a class="back-btn" href="{back_href}">↑ All {band} tiles</a>
          </div>
        </header>
        <div class="zoomtools">
          <button id="zout">– Zoom out</button>
          <button id="zin">+ Zoom in</button>
          <button id="fit">Fit width</button>
          <button id="one">1:1 pixels</button>
          <span class="pct" id="pct">100%</span>
          <span class="spacer"></span>
          <span class="hint">click = zoom in · click again = back · scroll to pan · ←/→ flip tiles</span>
        </div>
        <div class="zoombox" id="box">
          <img id="plotimg" src="{img_from_viewer}" alt="{panel}">
        </div>
        <script>
          var img=document.getElementById('plotimg'), box=document.getElementById('box'),
              pct=document.getElementById('pct'); var scale=1, fit=true;
          function render(){{ img.style.width=(img.naturalWidth*scale)+'px';
            pct.textContent=Math.round(scale*100)+'%';
            img.style.cursor = fit ? 'zoom-in' : 'zoom-out'; }}
          function doFit(){{ fit=true; scale=box.clientWidth/img.naturalWidth; render(); }}
          function zoomTo(s, fx, fy){{ if(fx==null)fx=0.5; if(fy==null)fy=0.5;
            fit=false; scale=Math.max(0.05, Math.min(s, 8)); render();
            box.scrollLeft=fx*img.naturalWidth*scale - box.clientWidth/2;
            box.scrollTop =fy*img.naturalHeight*scale - box.clientHeight/2; }}
          document.getElementById('zin').onclick =function(){{ zoomTo(scale*1.4); }};
          document.getElementById('zout').onclick=function(){{ zoomTo(scale/1.4); }};
          document.getElementById('one').onclick =function(){{ zoomTo(1); }};
          document.getElementById('fit').onclick =doFit;
          img.onclick=function(e){{
            var r=img.getBoundingClientRect();
            if (fit) zoomTo(Math.max(1.0, scale*2.2),
                            (e.clientX-r.left)/img.clientWidth,
                            (e.clientY-r.top)/img.clientHeight);
            else doFit();                       // second click → back to fit
          }};
          window.addEventListener('resize', function(){{ if(fit) doFit(); }});
          if(img.complete && img.naturalWidth) doFit(); else img.onload=doFit;
          document.addEventListener('keydown', function (e) {{
            if (e.key === 'ArrowRight') location.href = '{next_href}';
            else if (e.key === 'ArrowLeft') location.href = '{prev_href}';
          }});
        </script>
        </body></html>
    ''').strip()
    out = viewer_dir / f'{panel}.html'
    out.write_text(html)
    # return path relative to band page (which lives at html/psf_qa/)
    return f'viewer/{band}/{tile}/{panel}.html'


def write_mosaic_viewer_page(band, tile, panel, panel_title, prev_tile,
                             next_tile, prev_href, next_href, back_href,
                             viewer_dir):
    """Zoom viewer for the PSF sample/residual mosaics (projects_hsc
    qa_site.py pattern): zoom in/out/fit/1:1 toolbar, click-to-zoom at the
    cursor, scroll-pan, and a swap button (space/x) flipping
    samples⇄residuals while PRESERVING the zoom and scroll position."""
    samp_rel = f'../../../plots/{band}/{tile}/psf_samples.png'
    resi_rel = f'../../../plots/{band}/{tile}/psf_residuals.png'
    start = 'samples' if panel == 'psf_samples' else 'residuals'
    start_src = samp_rel if panel == 'psf_samples' else resi_rel
    start_lbl = 'Samples' if panel == 'psf_samples' else 'Residuals'
    html = dedent(f'''
        <!doctype html>
        <html><head>
          <meta charset="utf-8">
          <title>PSF QA — {band} {tile} — {panel_title}</title>
          <link rel="stylesheet" href="../../../css/style.css">
          <style>
            .tile-nav {{ display:flex; gap:10px; align-items:center; margin-top:8px; flex-wrap:wrap; }}
            .tile-nav .cur {{ font-weight:600; }}
            .tile-nav .spacer {{ flex:1; }}
            .zoomtools {{ display:flex; gap:8px; align-items:center; padding:7px 28px;
              background:#eef0f3; border-bottom:1px solid #ddd; font-size:14px; flex-wrap:wrap; }}
            .zoomtools button {{ background:#4a90e2; color:#fff; border:0; border-radius:5px;
              padding:6px 12px; font-size:14px; cursor:pointer; }}
            .zoomtools button:hover {{ background:#2c70c4; }}
            .zoomtools button.swap {{ background:#5cb85c; }}
            .zoomtools button.swap:hover {{ background:#449d44; }}
            .zoomtools .pct {{ font-weight:600; min-width:46px; }}
            .zoomtools .hint {{ color:#666; }}
            .zoombox {{ overflow:auto; width:100%; height:calc(100vh - 152px); background:#111; }}
            .zoombox img {{ display:block; width:100%; cursor:zoom-in; }}
          </style>
        </head><body>
        <header>
          <h1>{band} / {tile} — {panel_title}</h1>
          <div class="tile-nav">
            <a class="back-btn" href="{prev_href}">◀ Prev tile ({prev_tile})</a>
            <span class="cur">{tile}</span>
            <a class="back-btn" href="{next_href}">Next tile ({next_tile}) ▶</a>
            <span class="spacer"></span>
            <a class="back-btn" href="{back_href}">↑ All {band} tiles</a>
          </div>
        </header>
        <div class="zoomtools">
          <button id="swapbtn" class="swap">&#8646; Samples / Residuals</button><span class="pct">showing&nbsp;<b id="whichlbl">{start_lbl}</b></span>
          <button id="zout">– Zoom out</button>
          <button id="zin">+ Zoom in</button>
          <button id="fit">Fit width</button>
          <button id="one">1:1 pixels</button>
          <span class="pct" id="pct">100%</span>
          <span class="spacer"></span>
          <span class="hint">scroll to pan · click to zoom · space/x = flip samples⇄residuals · ←/→ flip tiles</span>
        </div>
        <div class="zoombox" id="box">
          <img id="mosaic" src="{start_src}" alt="{panel}">
        </div>
        <script>
          var img=document.getElementById('mosaic'), box=document.getElementById('box'),
              pct=document.getElementById('pct'); var scale=1, fit=true;
          function render(){{ img.style.width=(img.naturalWidth*scale)+'px';
            pct.textContent=Math.round(scale*100)+'%'; }}
          function doFit(){{ fit=true; scale=box.clientWidth/img.naturalWidth; render(); }}
          function zoomTo(s, fx, fy){{ if(fx==null)fx=0.5; if(fy==null)fy=0.5;
            fit=false; scale=Math.max(0.05, Math.min(s, 8)); render();
            box.scrollLeft=fx*img.naturalWidth*scale - box.clientWidth/2;
            box.scrollTop =fy*img.naturalHeight*scale - box.clientHeight/2; }}
          document.getElementById('zin').onclick =function(){{ zoomTo(scale*1.4); }};
          document.getElementById('zout').onclick=function(){{ zoomTo(scale/1.4); }};
          document.getElementById('one').onclick =function(){{ zoomTo(1); }};
          document.getElementById('fit').onclick =doFit;
          img.onclick=function(e){{ var r=img.getBoundingClientRect();
            zoomTo(scale*1.5,(e.clientX-r.left)/img.clientWidth,(e.clientY-r.top)/img.clientHeight); }};
          window.addEventListener('resize', function(){{ if(fit) doFit(); }});
          if(img.complete && img.naturalWidth) doFit(); else img.onload=doFit;
          document.addEventListener('keydown', function (e) {{
            if (e.key === 'ArrowRight') location.href = '{next_href}';
            else if (e.key === 'ArrowLeft') location.href = '{prev_href}';
          }});
          var PAIR={{samples:'{samp_rel}', residuals:'{resi_rel}'}};
          var curImg='{start}', lbl=document.getElementById('whichlbl');
          function swapImg(){{
            var fx=(box.scrollLeft+box.clientWidth/2)/(img.naturalWidth*scale);
            var fy=(box.scrollTop+box.clientHeight/2)/(img.naturalHeight*scale);
            curImg=(curImg==='samples')?'residuals':'samples';
            img.onload=function(){{render();
              box.scrollLeft=fx*img.naturalWidth*scale-box.clientWidth/2;
              box.scrollTop =fy*img.naturalHeight*scale-box.clientHeight/2;
              lbl.textContent=(curImg==='samples')?'Samples':'Residuals';}};
            img.src=PAIR[curImg];
          }}
          document.getElementById('swapbtn').onclick=swapImg;
          document.addEventListener('keydown',function(e){{
            if(e.key===' '||e.key==='x'||e.key==='X'){{e.preventDefault();swapImg();}}
          }});
        </script>
        </body></html>
    ''').strip()
    (viewer_dir / f'{panel}.html').write_text(html)
    return f'viewer/{band}/{tile}/{panel}.html'


# band instrument → footprint-grid instrument whose tilemaps/ minimaps apply
# (bands of one survey share the tile grid; keys match 59_'s PANELS sources)
_TILEMAP_GRID = [
    ('hst_acs_f814w', 'hst_acs_f814w'), ('jwst_nircam_', 'jwst_nircam_f115w'),
    ('euclid_vis', 'euclid_vis'),       ('euclid_nisp_', 'euclid_nisp_y'),
    ('hsc_', 'hsc_g'),                  ('lsdr10_', 'lsdr10_g'),
    ('ps1_', 'ps1_g'),                  ('sdss_', 'sdss_r'),
    ('unwise_', 'unwise_w1'),
]


def tilemap_rel(instrument: str, tile: str, band: str | None = None) -> str | None:
    """Relative path of the tile's minimap (59_ --tilemaps), if rendered.
    External-record bands fall back to the hostgalxy site's tilemaps,
    symlinked into ours on first use."""
    for prefix, grid in _TILEMAP_GRID:
        if instrument.startswith(prefix):
            rel = f'tilemaps/{grid}__{tile}.png'
            if (HTML_DIR / rel).exists():
                return rel
            break
    if band in EXTERNAL_SRC:
        key = EXTERNAL_SRC[band][1]
        src = EXTERNAL_ROOT / 'tilemaps' / f'{key}__{tile}.png'
        if src.exists():
            rel = f'tilemaps/{key}__{tile}.png'
            dst = HTML_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not (dst.is_symlink() or dst.exists()):
                os.symlink(src, dst)
            return rel
    return None


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
                viewer_rel = write_viewer_page(band, tile, panel, panel_title, rel, tiles)
                # mosaics: the band-page cell shows the ≤16-row _thumb (the
                # full mosaic can be hundreds of rows); the viewer shows full.
                cell_img = linked.get(f'{panel}_thumb', rel)
                # the two MODEL-DEPENDENT panels carry PSFEx/Piff variants for
                # the top toggle; data-* attrs let the JS swap src per backend.
                data_attr = ''
                if panel == 'mag_vs_chi2' and 'mag_vs_chi2_piff' in linked:
                    data_attr = (f' class="switchable" '
                                 f'data-psfex="{linked.get("mag_vs_chi2_psfex", rel)}" '
                                 f'data-piff="{linked["mag_vs_chi2_piff"]}"')
                    cell_img = linked.get('mag_vs_chi2_psfex', rel)
                elif panel == 'psf_residuals' and 'psf_residuals_piff' in linked:
                    data_attr = (f' class="switchable" '
                                 f'data-psfex="{cell_img}" '
                                 f'data-piff="{linked.get("psf_residuals_piff_thumb", linked["psf_residuals_piff"])}"')
                cells.append(
                    f'<td class="panel"><a href="{viewer_rel}">'
                    f'<span class="thumb-box"><img src="{cell_img}"{data_attr} alt="{panel}"/></span>'
                    f'</a></td>'
                )
            else:
                cells.append(f'<td class="empty">—</td>')
        tm = tilemap_rel(instrument, tile, band)
        tm_html = (f'<br><img class="tilemap" src="{tm}" '
                   f'alt="{tile} sky location">' if tm else '')
        rows.append(f'<tr><td class="tile-name">{tile}{tm_html}</td>{"".join(cells)}</tr>')

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
        <div class="psf-toggle">
          <button id="psfbtn">&#8646; Switch PSF model</button>
          <span>PSF model shown for <b>Mag&nbsp;vs&nbsp;χ²</b> &amp;
            <b>PSF&nbsp;Residuals</b>:
            <b id="psflbl" class="pex">PSFEx</b></span>
          <span class="hint">(the other 4 panels are model-independent)</span>
        </div>
        <div class="wrap">
          {body_extra}
          <table class="tiles">
            <thead>{thead}</thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
        <div id="tmlb" class="tm-lightbox" onclick="this.style.display='none'">
          <img id="tmlbimg" alt="tile location (enlarged)">
        </div>
        <script>
          document.querySelectorAll('img.tilemap').forEach(function(im){{
            im.addEventListener('click', function(){{
              document.getElementById('tmlbimg').src = this.src;
              document.getElementById('tmlb').style.display = 'flex';
            }});
          }});
          document.addEventListener('keydown', function(e){{
            if (e.key === 'Escape') document.getElementById('tmlb').style.display = 'none';
          }});
          // PSFEx ⇄ Piff toggle for the two model-dependent panels
          var backend='psfex', lbl=document.getElementById('psflbl');
          var sw=document.querySelectorAll('img.switchable');
          var btn=document.getElementById('psfbtn');
          if(!sw.length){{ btn.disabled=true;
            btn.title='no Piff panels generated for this band yet'; }}
          btn.onclick=function(){{
            backend=(backend==='psfex')?'piff':'psfex';
            sw.forEach(function(im){{
              var s=im.getAttribute('data-'+backend); if(s) im.src=s; }});
            lbl.textContent=(backend==='psfex')?'PSFEx':'Piff';
            lbl.className=(backend==='psfex')?'pex':'pif';
          }};
        </script>
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
        tiles = (external_tiles(band) if band in EXTERNAL_SRC
                 else discover_tiles(instrument))
        band_status[band] = len(tiles)
        write_band_page(band, label, instrument, tiles)
        print(f'  {band:6s}  {label:32s}  tiles={tiles or "[]"}')

    write_index(band_status)
    print(f'\n  → site at: {HTML_DIR / "index.html"}')


if __name__ == '__main__':
    main()
