#!/usr/bin/env python
"""
71_make_diag_site.py — static HTML site for the Step 3a-③/3b stellar
diagnostics (70_step3a_diag_plots.py), in the style of the PSF QA site
(55_make_psf_qa_site.py).

Layout
------
  html/diag_qa/
    index.html                  mission selector + per-tile star counts
    jwst.html, hst.html, euclid.html
    plots/<mission>/<tile>/     symlinks to the canonical PNGs under
                                /Volumes/exdisk1/.../<tile>/phot/
    css/style.css               copied from psf_qa if present

Per (mission, tile) section: star-count header (3b meta), then one row
per band: diag3 composite PSF, diag4 profile, diag1 RA/Dec map, diag2
photometry comparison; plus the cross-band profile overlay and the 3b
envelope sheet.  Click any thumbnail → full PNG.

Re-run after producing more tiles (resumable; cheap — symlinks only).
"""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
HTML_DIR = PROJECT / 'html' / 'diag_qa'
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

MISSIONS = [('jwst', 'JWST NIRCam (F115W/F150W/F277W/F444W)'),
            ('hst', 'HST ACS (F814W)'),
            ('euclid', 'Euclid (VIS + NISP Y/J/H)')]

CSS = dedent('''
    body { font-family: -apple-system, Helvetica, sans-serif;
           margin: 1.2em 2em; background: #fafafa; color: #222; }
    h1 { font-size: 1.5em; } h2 { font-size: 1.2em; margin-top: 1.6em;
         border-bottom: 2px solid #888; padding-bottom: .2em; }
    h3 { font-size: 1.0em; margin: 1.0em 0 .3em; }
    table.idx { border-collapse: collapse; }
    table.idx td, table.idx th { border: 1px solid #ccc;
        padding: .35em .8em; font-size: .95em; }
    .thumbrow { white-space: nowrap; overflow-x: auto; margin: .3em 0; }
    .thumbrow a { display: inline-block; margin-right: .4em; }
    .thumbrow img { height: 200px; border: 1px solid #bbb;
        background: #fff; }
    .meta { color: #555; font-size: .9em; }
    a { color: #0645ad; text-decoration: none; }
    a:hover { text-decoration: underline; }
''')


def star_meta(mission: str, tile: str) -> dict:
    p = WORK / f'{mission}_chi2' / tile / 'phot' / f'star_{tile}.meta.json'
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def link_plots(mission: str, tile: str) -> list[str]:
    """Symlink every diag*/star_envelopes PNG; return names found."""
    src = WORK / f'{mission}_chi2' / tile / 'phot'
    dst = HTML_DIR / 'plots' / mission / tile
    names = []
    for p in sorted(src.glob('diag*.png')) + \
            sorted(src.glob('star_envelopes_*.png')):
        dst.mkdir(parents=True, exist_ok=True)
        ln = dst / p.name
        if ln.is_symlink() or ln.exists():
            ln.unlink()
        ln.symlink_to(p)
        names.append(p.name)
    return names


def tile_section(mission: str, tile: str, names: list[str],
                 meta: dict) -> str:
    rel = f'plots/{mission}/{tile}'
    bands = sorted({n.split(f'_{tile}_')[1][:-4] for n in names
                    if f'_{tile}_' in n and n.startswith('diag')})
    hdr = ''
    if meta:
        hdr = (f"<span class='meta'>sources {meta.get('n_sources', 0):,} "
               f"· is_star {meta.get('n_is_star', 0):,} "
               f"(votes {meta.get('n_by_votes', 0):,}, "
               f"Gaia-PM {meta.get('n_gaia_pm_significant', 0):,})</span>")
    out = [f'<h2 id="{tile}">{tile} {hdr}</h2>']
    for b in bands:
        row = []
        for k in (3, 4, 1, 2):
            n = f'diag{k}_' + {1: 'stars_radec', 2: 'phot_compare',
                               3: 'psf_composite', 4: 'psf_profile'}[k] + \
                f'_{tile}_{b}.png'
            if n in names:
                row.append(f'<a href="{rel}/{n}" target="_blank">'
                           f'<img src="{rel}/{n}" loading="lazy"></a>')
        if row:
            out.append(f'<h3>{b}</h3><div class="thumbrow">'
                       + ''.join(row) + '</div>')
    extras = [n for n in names
              if n == f'diag4_psf_profile_{tile}.png'
              or n.startswith('star_envelopes')]
    if extras:
        row = ''.join(f'<a href="{rel}/{n}" target="_blank">'
                      f'<img src="{rel}/{n}" loading="lazy"></a>'
                      for n in extras)
        out.append(f'<h3>cross-band profile + 3b envelopes</h3>'
                   f'<div class="thumbrow">{row}</div>')
    return '\n'.join(out)


def main():
    (HTML_DIR / 'css').mkdir(parents=True, exist_ok=True)
    old_css = PROJECT / 'html' / 'psf_qa' / 'css' / 'style.css'
    (HTML_DIR / 'css' / 'style.css').write_text(
        old_css.read_text() if old_css.exists() else CSS)

    index_rows = []
    for mission, label in MISSIONS:
        sections = []
        toc = []
        for tdir in sorted((WORK).glob(f'{mission}_chi2/*/phot')):
            tile = tdir.parent.name
            names = link_plots(mission, tile)
            if not names:
                continue
            meta = star_meta(mission, tile)
            sections.append(tile_section(mission, tile, names, meta))
            toc.append(f'<a href="#{tile}">{tile}</a>')
            index_rows.append(
                f'<tr><td><a href="{mission}.html#{tile}">{mission}'
                f'/{tile}</a></td>'
                f"<td>{meta.get('n_sources', '—')}</td>"
                f"<td>{meta.get('n_is_star', '—')}</td>"
                f"<td>{meta.get('n_by_votes', '—')}</td>"
                f"<td>{meta.get('n_gaia_pm_significant', '—')}</td></tr>")
        page = dedent(f'''<!DOCTYPE html><html><head>
            <meta charset="utf-8"><title>{label} — stellar diagnostics</title>
            <link rel="stylesheet" href="css/style.css"></head><body>
            <h1>{label} — Step 3a-③/3b stellar diagnostics</h1>
            <p><a href="index.html">&larr; index</a> &nbsp; tiles:
            {' · '.join(toc)}</p>
            {''.join(sections)}
            </body></html>''')
        (HTML_DIR / f'{mission}.html').write_text(page)
        print(f'{mission}: {len(toc)} tiles')

    idx = dedent(f'''<!DOCTYPE html><html><head>
        <meta charset="utf-8"><title>Stellar diagnostics QA</title>
        <link rel="stylesheet" href="css/style.css"></head><body>
        <h1>Step 3a-③ / 3b stellar diagnostics</h1>
        <p>Per (mission, tile): composite PSF (×9 drizzle, sqrt stretch,
        ±1.2&Prime;), profile cuts, star RA/Dec map, aperture-vs-PSF
        photometry; stars only (3b is_star), solid = isolated,
        open = n&#8321;&gt;0.  Sibling site:
        <a href="../psf_qa/index.html">PSF-model QA (3a-①)</a>.</p>
        <p>Missions: {' · '.join(f'<a href="{m}.html">{l}</a>'
                                 for m, l in MISSIONS)}</p>
        <table class="idx"><tr><th>tile</th><th>sources</th>
        <th>is_star</th><th>by votes</th><th>Gaia-PM</th></tr>
        {''.join(index_rows)}</table>
        </body></html>''')
    (HTML_DIR / 'index.html').write_text(idx)
    print(f'[save] {HTML_DIR}/index.html  ({len(index_rows)} tile rows)')


if __name__ == '__main__':
    main()
