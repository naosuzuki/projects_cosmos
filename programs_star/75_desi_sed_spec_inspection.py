#!/usr/bin/env python
"""
75_desi_sed_spec_inspection.py — paginated inspection page for the
DESI-spectroscopy subset of the master 4-way stars: SED fit (v08) on
the left, DESI spectrum on the right, one row per object.

Inputs:
  csvfiles_star/master_stars_4way_with_desi.parquet
  csvfiles_star/desi_dr1_download_log.parquet  (status='saved' rows)
  csvfiles_star/sed_blackbody_v08.parquet
  htmls/sed_v08/seds/sed_<id>.png              (per-object SED already
                                                generated)
  /Volumes/exdisk1/data/DESI/cosmos_<type>/DESI-<targetid>.fits

Outputs:
  htmls/desi_check/spec/spec_<id>.png        (per-object DESI spectrum plot)
  htmls/desi_check/index.html                (paginated inspection page)
  htmls/desi_check/index_<NN>.html
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

PROJ     = Path('/Users/suzuki/github/projects_cosmos')
OUT      = PROJ / 'csvfiles_star'
HTML_OUT = PROJ / 'htmls' / 'desi_check'
SPEC_DIR = HTML_OUT / 'spec'
DESI_DIR = Path('/Volumes/exdisk1/data/DESI')
SUBDIRS  = {'star': 'cosmos_star', 'quasar': 'cosmos_quasar', 'galaxy': 'cosmos_galaxy'}
SED_VER  = 'v08'
SED_PNG_DIR = PROJ / 'htmls' / f'sed_{SED_VER}' / 'seds'


def smooth(a, n=15):
    """Boxcar smooth with a window of n samples (edge-aware)."""
    if n < 2:
        return a.copy()
    a = np.asarray(a, dtype=float)
    w = np.ones(n) / n
    return np.convolve(a, w, mode='same')


def plot_desi_spectrum(spec_fits, out_path, title_extra=''):
    """Plot the DESI spectrum.  Returns False if the spectrum is essentially
    empty (zero/near-zero flux across the whole array — a DESI 'zombie'
    entry: target processed but no usable signal, typically ZWARN=514).
    Callers should drop those from the inspection page."""
    with fits.open(spec_fits) as h:
        hdr = h['PRIMARY'].header
        tab = h['SPECTRUM'].data
        wl = np.asarray(tab['WAVELENGTH'], dtype=float)
        fl = np.asarray(tab['FLUX'],       dtype=float)
        iv = None
        if 'IVAR' in tab.columns.names:
            iv = np.asarray(tab['IVAR'], dtype=float)
    # zombie / failed-extraction check
    nz = int(np.sum(np.abs(fl) > 1e-6))
    if nz < 100:
        return False
    sp_type = str(hdr.get('SPECTYPE', '?')).strip()
    z = float(hdr.get('Z', np.nan))
    zerr = float(hdr.get('ZERR', np.nan))
    targid = int(hdr.get('TARGETID', -1))
    jwstid = int(hdr.get('JWSTID', -1))
    survey = str(hdr.get('SURVEY', '?')).strip()
    program = str(hdr.get('PROGRAM', '?')).strip()

    flsm = smooth(fl, 15)
    # robust y-range from middle 98% of smoothed flux
    lo, hi = np.percentile(flsm, [1, 99])
    pad = 0.10 * (hi - lo + 1e-3)
    ymin = max(lo - pad, np.percentile(flsm, 0.1) - pad)
    ymax = hi + pad

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(wl, fl,   color='0.7', lw=0.35, label='raw', zorder=1)
    ax.plot(wl, flsm, color='C0',  lw=0.9,  label='smoothed (15 pix)', zorder=2)

    # rest-frame line markers (only for sensible z)
    if np.isfinite(z) and -1 < z < 8:
        if sp_type in ('STAR',):
            lines = {'CaIIH': 3933.66, 'CaIIK': 3968.47, 'Hδ': 4101.7, 'Hγ': 4340.5,
                     'Hβ': 4861.3, 'Mgb': 5175.0, 'NaD': 5893.0, 'Hα': 6562.8}
        elif sp_type in ('QSO', 'GALAXY'):
            lines = {'Lyα': 1215.7, 'CIV': 1549.5, 'CIII]': 1908.7, 'MgII': 2798.7,
                     '[OII]': 3727.3, 'Hβ': 4861.3, '[OIII]': 5006.8, 'Hα': 6562.8}
        else:
            lines = {}
        for name, lam0 in lines.items():
            obs = lam0 * (1 + z)
            if wl[0] < obs < wl[-1]:
                ax.axvline(obs, color='C3', lw=0.5, ls=':', alpha=0.6)
                ax.text(obs, ymax - 0.02*(ymax-ymin), name, color='C3',
                        rotation=90, fontsize=7, va='top', ha='right')

    ax.set_xlim(wl[0], wl[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Wavelength (Å, observed)')
    ax.set_ylabel(r'Flux (10$^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    ax.set_title(f'JWST {jwstid} — DESI {sp_type}  '
                 f'TARGETID {targid}  z={z:.4f}±{zerr:.4f}  '
                 f'({survey}/{program}) {title_extra}', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    plt.close(fig)
    return True


def main():
    t0 = time.time()
    HTML_OUT.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading master + DESI download log + SED v08 table ...')
    master = pd.read_parquet(OUT/'master_stars_4way_with_desi.parquet')
    log    = pd.read_parquet(OUT/'desi_dr1_download_log.parquet')
    sed    = pd.read_parquet(OUT/f'sed_blackbody_{SED_VER}.parquet')

    # master rows whose DESI spectrum has actually been saved
    saved_targets = set(log.loc[log['status']=='saved','targetid'].astype(np.int64))
    m_has_spec = master[master['has_desi_spec']
                        & master['desi_targetid'].astype('Int64').isin(saved_targets)].copy()
    print(f'  master rows with saved DESI spectrum: {len(m_has_spec):,}')

    # Restrict to those with v08 SED fit (id == jwst_id)
    sed_ids = set(sed['id'].astype(np.int64))
    m_has_spec['jwst_id_int'] = pd.to_numeric(m_has_spec['jwst_id'], errors='coerce').astype('Int64')
    keep = m_has_spec[m_has_spec['jwst_id_int'].astype('Int64').isin(sed_ids)].copy()
    print(f'  ... that also have a v08 SED fit:      {len(keep):,}')

    # sort by SED brightness for nicer browsing
    sed_idx = sed.set_index('id')
    keep['_F814W'] = keep['jwst_id_int'].astype('Int64').map(sed_idx['mag_obs_F814W'].to_dict())
    keep = keep.sort_values('_F814W').reset_index(drop=True)

    # Generate spectrum PNGs
    print(f'Generating {len(keep):,} DESI spectrum plots ...')
    spec_paths_ok = []
    for i, row in keep.iterrows():
        tid  = int(row['desi_targetid'])
        sp   = row['desi_type']
        sub  = SUBDIRS.get(sp)
        jid  = int(row['jwst_id_int'])
        if sub is None:
            spec_paths_ok.append(False); continue
        fits_path = DESI_DIR / sub / f'DESI-{tid}.fits'
        if not fits_path.exists():
            spec_paths_ok.append(False); continue
        try:
            ok = plot_desi_spectrum(fits_path, SPEC_DIR/f'spec_{jid}.png')
            spec_paths_ok.append(ok)
            if not ok:
                # zombie / empty spectrum — drop the stale PNG if present
                stale = SPEC_DIR/f'spec_{jid}.png'
                if stale.exists():
                    stale.unlink()
        except Exception as e:
            print(f'  plot failed for {jid}: {repr(e)[:80]}')
            spec_paths_ok.append(False)
    keep['_spec_ok'] = spec_paths_ok
    keep = keep[keep['_spec_ok']].reset_index(drop=True)
    print(f'  plotted {len(keep):,} spectra')

    # Build paginated HTML — one ROW per object (SED left, spectrum right)
    per_page = 20
    n_pages = (len(keep) + per_page - 1) // per_page
    def pname(p): return 'index.html' if p == 0 else f'index_{p:02d}.html'
    counts = keep['desi_type'].value_counts().to_dict()
    type_pill = (f'<span style="color:#1a5">{counts.get("star",0)} star</span> · '
                 f'<span style="color:#933">{counts.get("quasar",0)} quasar</span> · '
                 f'<span style="color:#633">{counts.get("galaxy",0)} galaxy</span>')

    for pg in range(n_pages):
        sub_df = keep.iloc[pg*per_page:(pg+1)*per_page]
        rows = []
        for _, r in sub_df.iterrows():
            jid = int(r['jwst_id_int'])
            sp  = r['desi_type']
            z   = r['desi_z']
            zwarn = int(r['desi_zwarn']) if pd.notna(r['desi_zwarn']) else 0
            our_class = 'AGN/QSO' if bool(r['is_agn_qso']) else 'star'
            badge_cls = {'star':'badge-star','quasar':'badge-qso','galaxy':'badge-gal'}[sp]
            sed_png = f'../sed_{SED_VER}/seds/sed_{jid}.png'
            spec_png = f'spec/spec_{jid}.png'
            rows.append(f'''
<div class="row">
  <div class="meta">
    <div class="ourid">JWST&nbsp;{jid}</div>
    <div class="cls">our: <b>{our_class}</b> &nbsp; <span class="badge {badge_cls}">DESI&nbsp;{sp}</span></div>
    <div class="z">z = {z:.4f}{" (ZWARN!)" if zwarn else ""}</div>
    <div class="sep">DESI sep = {r["desi_sep_arcsec"]:.2f}″</div>
  </div>
  <div class="sed"><img src="{sed_png}" loading="lazy" alt="SED v08"></div>
  <div class="spec"><img src="{spec_png}" loading="lazy" alt="DESI spectrum"></div>
</div>''')
        # nav
        def lk(p):
            return f'<b>{p+1}</b>' if p==pg else f'<a href="{pname(p)}">{p+1}</a>'
        win = sorted({0,1,n_pages-2,n_pages-1}|set(range(max(0,pg-2),min(n_pages,pg+3))))
        win = [p for p in win if 0<=p<n_pages]
        parts = []
        prev = None
        if pg>0: parts.append(f'<a href="{pname(pg-1)}">&larr; prev</a>')
        for p in win:
            if prev is not None and p != prev+1: parts.append('…')
            parts.append(lk(p)); prev = p
        if pg<n_pages-1: parts.append(f'<a href="{pname(pg+1)}">next &rarr;</a>')
        nav = ' '.join(parts)

        html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>DESI ↔ SED inspection — page {pg+1}/{n_pages}</title>
<style>
body{{font-family:sans-serif;margin:14px;background:#fafafa;color:#222;}}
h1{{font-size:18px;margin-bottom:4px;}}
.subtitle{{color:#666;font-size:13px;margin-bottom:14px;}}
.nav{{margin:10px 0;font-size:14px;}} .nav a{{margin:0 5px;}}
.row{{display:grid;grid-template-columns:200px 1fr 1fr;gap:8px;
     background:#fff;border:1px solid #ddd;border-radius:6px;
     padding:6px;margin-bottom:8px;align-items:center;}}
.row .meta{{font-family:monospace;font-size:11px;line-height:1.45;}}
.row .ourid{{font-size:13px;font-weight:bold;margin-bottom:2px;}}
.row .cls,.row .z,.row .sep{{margin:2px 0;}}
.row img{{width:100%;display:block;}}
.badge{{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;
       font-weight:bold;color:white;}}
.badge-star{{background:#1a5;}}
.badge-qso{{background:#a33;}}
.badge-gal{{background:#963;}}
</style></head><body>
<h1>DESI spectra ↔ SED v08 inspection
<span style="font-weight:normal;font-size:13px;color:#666;">
 — {len(keep):,} objects with both a downloaded DESI DR1 spectrum and a v08 SED fit (page {pg+1}/{n_pages})</span></h1>
<p class="subtitle">By DESI SPECTYPE: {type_pill}.  Rows show:
left meta panel, middle = v08 SED fit (rainbow per filter; circles = HST/JWST/Euclid;
squares = COSMOS-Web; ^=SDSS; v=PS1; × = blackbody prediction),
right = DESI DR1 spectrum (gray = raw; blue = 15-pix boxcar; dotted red lines = expected
rest-frame features at the DESI z).</p>
<div class="nav">{nav}</div>
{''.join(rows)}
<div class="nav">{nav}</div>
</body></html>'''
        (HTML_OUT/pname(pg)).write_text(html)

    print(f'\nWebform: {n_pages} pages → {HTML_OUT}/index.html')
    print(f'Wall time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
