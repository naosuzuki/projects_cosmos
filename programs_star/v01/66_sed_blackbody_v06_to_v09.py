#!/usr/bin/env python
"""
66_sed_blackbody_v06_to_v09.py — extend the v04 SED fit by adding SDSS
(u,g,r,i,z) and Pan-STARRS DR1 (g,r,i,z,y) photometry.

Sample: same 6,007 clean stars as v04/v05 (SNR>10 in all 9
HST+JWST+Euclid bands).  AGN/QSO excluded from the calibration
statistics & main webform but their photometry is kept (separate
index_agn.html), as in v04/v05.

PS1 and SDSS photometry are taken from
master_stars_4way_with_ps1_sdss.parquet, joined by JWST id.
Magnitudes are AB (PS1 from VizieR II/349 mean PSF; SDSS DR16 PhotoObj
psfMag).

Four versions in one run:
  v06 = v04 (20 bands) + SDSS  (25 bands, all democratically fit)
  v07 = v04            + PS1   (25 bands, all democratically fit)
  v08 = v04 + SDSS + PS1       (30 bands, all democratically fit)
  v09 = v08 bands, Euclid HELD OUT of fit (predicted & tested)

Each version writes:
  csvfiles_star/sed_blackbody_v0N.parquet (+ .csv)
  htmls/sed_v0N/band_offset_summary.png/.txt
  htmls/sed_v0N/seds/sed_<id>.png  (6,007)
  htmls/sed_v0N/index.html         (stars, paginated)
  htmls/sed_v0N/index_agn.html     (AGN/QSO, paginated)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

warnings.filterwarnings('ignore')
ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
FILTER_CACHE = OUT / 'sed_filters'
CW_PATH = '/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1.fits'

H = 6.62607015e-27; C_AA = 2.99792458e18; K = 1.380649e-16

# (key, svo_id, mission, source, valcol, errcol, errkind)
#   source 'ours' → clean-sample mag column
#   source 'cw'   → COSMOS-Web flux_model_<valcol> (μJy)
#   source 'sdss' → master_stars_4way_with_ps1_sdss mag column (AB)
#   source 'ps1'  → master_stars_4way_with_ps1_sdss mag column (AB)
ALL_BANDS = [
    # v04 base — ours (HST/JWST/Euclid)
    ('F814W',      'HST/ACS_WFC.F814W', 'HST',       'ours', 'cat_mag_F814W_hst',  'cat_magerr_F814W_hst',  'magerr'),
    ('F115W',      'JWST/NIRCam.F115W', 'JWST',      'ours', 'cat_mag_F115W_jwst', 'cat_snr_F115W_jwst',    'snr'),
    ('F150W',      'JWST/NIRCam.F150W', 'JWST',      'ours', 'cat_mag_F150W_jwst', 'cat_snr_F150W_jwst',    'snr'),
    ('F277W',      'JWST/NIRCam.F277W', 'JWST',      'ours', 'cat_mag_F277W_jwst', 'cat_snr_F277W_jwst',    'snr'),
    ('F444W',      'JWST/NIRCam.F444W', 'JWST',      'ours', 'cat_mag_F444W_jwst', 'cat_snr_F444W_jwst',    'snr'),
    ('Euclid_VIS', 'Euclid/VIS.vis',    'Euclid',    'ours', 'cat_mag_VIS_vis',    'cat_magerr_VIS_vis',    'magerr'),
    ('Euclid_Y',   'Euclid/NISP.Y',     'Euclid',    'ours', 'cat_mag_NIR_Y_nisp', 'cat_magerr_NIR_Y_nisp', 'magerr'),
    ('Euclid_J',   'Euclid/NISP.J',     'Euclid',    'ours', 'cat_mag_NIR_J_nisp', 'cat_magerr_NIR_J_nisp', 'magerr'),
    ('Euclid_H',   'Euclid/NISP.H',     'Euclid',    'ours', 'cat_mag_NIR_H_nisp', 'cat_magerr_NIR_H_nisp', 'magerr'),
    # v04 base — COSMOS-Web
    ('CFHT_u',     'CFHT/MegaCam.u',    'CFHT',      'cw',   'cfht-u',   None, None),
    ('HSC_g',      'Subaru/HSC.g',      'HSC',       'cw',   'hsc-g',    None, None),
    ('HSC_r',      'Subaru/HSC.r',      'HSC',       'cw',   'hsc-r',    None, None),
    ('HSC_i',      'Subaru/HSC.i',      'HSC',       'cw',   'hsc-i',    None, None),
    ('HSC_z',      'Subaru/HSC.z',      'HSC',       'cw',   'hsc-z',    None, None),
    ('HSC_y',      'Subaru/HSC.Y',      'HSC',       'cw',   'hsc-y',    None, None),
    ('IRAC_ch1',   'Spitzer/IRAC.I1',   'IRAC',      'cw',   'irac-ch1', None, None),
    ('IRAC_ch2',   'Spitzer/IRAC.I2',   'IRAC',      'cw',   'irac-ch2', None, None),
    ('IRAC_ch3',   'Spitzer/IRAC.I3',   'IRAC',      'cw',   'irac-ch3', None, None),
    ('IRAC_ch4',   'Spitzer/IRAC.I4',   'IRAC',      'cw',   'irac-ch4', None, None),
    ('MIRI_F770W', 'JWST/MIRI.F770W',   'JWST-MIRI', 'cw',   'f770w',    None, None),
    # SDSS DR16 (psfMag)
    ('SDSS_u',     'SLOAN/SDSS.u',      'SDSS',      'sdss', 'sdss_psfMag_u', 'sdss_psfMagErr_u', 'magerr'),
    ('SDSS_g',     'SLOAN/SDSS.g',      'SDSS',      'sdss', 'sdss_psfMag_g', 'sdss_psfMagErr_g', 'magerr'),
    ('SDSS_r',     'SLOAN/SDSS.r',      'SDSS',      'sdss', 'sdss_psfMag_r', 'sdss_psfMagErr_r', 'magerr'),
    ('SDSS_i',     'SLOAN/SDSS.i',      'SDSS',      'sdss', 'sdss_psfMag_i', 'sdss_psfMagErr_i', 'magerr'),
    ('SDSS_z',     'SLOAN/SDSS.z',      'SDSS',      'sdss', 'sdss_psfMag_z', 'sdss_psfMagErr_z', 'magerr'),
    # Pan-STARRS DR1 mean PSF
    ('PS1_g',      'PAN-STARRS/PS1.g',  'PS1',       'ps1',  'ps1_gmag',  'ps1_e_gmag', 'magerr'),
    ('PS1_r',      'PAN-STARRS/PS1.r',  'PS1',       'ps1',  'ps1_rmag',  'ps1_e_rmag', 'magerr'),
    ('PS1_i',      'PAN-STARRS/PS1.i',  'PS1',       'ps1',  'ps1_imag',  'ps1_e_imag', 'magerr'),
    ('PS1_z',      'PAN-STARRS/PS1.z',  'PS1',       'ps1',  'ps1_zmag',  'ps1_e_zmag', 'magerr'),
    ('PS1_y',      'PAN-STARRS/PS1.y',  'PS1',       'ps1',  'ps1_ymag',  'ps1_e_ymag', 'magerr'),
]
EUCLID_KEYS = {'Euclid_VIS','Euclid_Y','Euclid_J','Euclid_H'}
CW_SUFFIXES = [b[4] for b in ALL_BANDS if b[3]=='cw']

T_GRID  = np.geomspace(2000.0, 50000.0, 400)
SNR_MIN = 3.0
MAGERR_MAX = 1.0857 / SNR_MIN
FE_SENTINEL = 1e10
MIN_FIT_BANDS = 5

# Version → (subset_keys, in_fit_keys)
def version_config(ver):
    base = [b[0] for b in ALL_BANDS if b[3] in ('ours','cw')]
    sdss = [b[0] for b in ALL_BANDS if b[3]=='sdss']
    ps1  = [b[0] for b in ALL_BANDS if b[3]=='ps1']
    if ver == 'v06':  subset = base + sdss;        held = set()
    elif ver == 'v07': subset = base + ps1;        held = set()
    elif ver == 'v08': subset = base + sdss + ps1; held = set()
    elif ver == 'v09': subset = base + sdss + ps1; held = EUCLID_KEYS
    else: raise ValueError(ver)
    return subset, held


def load_filters(keys):
    FILTER_CACHE.mkdir(parents=True, exist_ok=True)
    from astroquery.svo_fps import SvoFps
    bmap = {b[0]: b for b in ALL_BANDS}
    filt = {}
    for k in keys:
        svo_id = bmap[k][1]
        cache = FILTER_CACHE / f'sed_{k}.npz'
        if cache.exists():
            d = np.load(cache); wl, tr = d['wl'], d['tr']
        else:
            print(f'[filters] fetch {svo_id}')
            t = SvoFps.get_transmission_data(svo_id)
            wl = np.asarray(t['Wavelength'], float); tr = np.asarray(t['Transmission'], float)
            np.savez(cache, wl=wl, tr=tr)
        good = tr > 0; wl, tr = wl[good], tr[good]
        piv = np.sqrt(np.trapezoid(tr, wl) / np.trapezoid(tr / wl**2, wl))
        filt[k] = {'wl': wl, 'tr': tr, 'pivot_AA': piv}
    return filt


def bb_fnu(wl_AA, T):
    nu = C_AA / wl_AA; x = H * nu / (K * T)
    with np.errstate(over='ignore'):
        return (nu**3) / np.expm1(x)


def precompute_shape(filt):
    grid = {k: np.zeros(len(T_GRID)) for k in filt}
    for i, T in enumerate(T_GRID):
        for k, fd in filt.items():
            wl, tr = fd['wl'], fd['tr']; fnu = bb_fnu(wl, T)
            grid[k][i] = -2.5*np.log10(np.trapezoid(fnu*tr*wl, wl)/np.trapezoid(tr*wl, wl))
    return grid


def fit_bb(mobs, merr, shape_grid, fit_keys):
    w = np.array([1.0/(merr[k]**2 + 0.03**2) for k in fit_keys])
    m = np.array([mobs[k] for k in fit_keys])
    smat = np.array([shape_grid[k] for k in fit_keys])
    chi2 = np.full(len(T_GRID), np.inf); dm = np.zeros(len(T_GRID))
    for i in range(len(T_GRID)):
        sh = smat[:, i]; d = np.sum(w*(m-sh))/np.sum(w)
        chi2[i] = np.sum(w*(m-sh-d)**2); dm[i] = d
    ib = np.argmin(chi2)
    return T_GRID[ib], dm[ib], chi2[ib], len(fit_keys)-2


def _filter_colors(filt):
    logp = {k: np.log10(filt[k]['pivot_AA']) for k in filt}
    lmin, lmax = min(logp.values()), max(logp.values())
    cmap = plt.cm.rainbow
    return {k: cmap((logp[k]-lmin)/(lmax-lmin)) for k in filt}


# Per-version global state for the worker (set by main)
_VER = None
_BANDS_SUB = None
_HELD = None


def make_sed_plot(rec, filt, shape_grid, path):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    T = rec['T_bb']; dm = rec['dm']; fcolor = _filter_colors(filt)
    SRC = {b[0]: b[3] for b in _BANDS_SUB}; MIS = {b[0]: b[2] for b in _BANDS_SUB}

    def marker(k):
        if k in _HELD: return 'D'        # Euclid held out
        s = SRC[k]
        if s == 'ours': return 'D' if k in EUCLID_KEYS else 'o'
        if s == 'cw':   return 's'
        if s == 'sdss': return '^'
        if s == 'ps1':  return 'v'
        return 'o'

    # model curve
    wl_um = np.geomspace(0.3, 9.0, 500); wl_AA = wl_um*1e4
    mmod = -2.5*np.log10(bb_fnu(wl_AA, T)) + dm
    anchor_b = 'F150W' if 'F150W' in filt else next(iter(filt))
    shp = np.interp(T, T_GRID, shape_grid[anchor_b]) + dm
    mod_at = -2.5*np.log10(bb_fnu(np.array([filt[anchor_b]['pivot_AA']]), T))[0] + dm
    ax.plot(wl_um, mmod + (shp-mod_at), '-', color='0.6', lw=1.1, zorder=1)

    ebar = dict(ecolor='0.25', elinewidth=1.0, capsize=3.0, capthick=1.0)
    mags=[]
    for b in _BANDS_SUB:
        k = b[0]; piv = filt[k]['pivot_AA']/1e4
        mo = rec.get(f'mag_obs_{k}', np.nan); me = rec.get(f'magerr_{k}', np.nan)
        if not np.isfinite(mo): continue
        ax.errorbar(piv, mo, yerr=me if np.isfinite(me) else None, fmt=marker(k),
                    color=fcolor[k], ms=7, mec='0.2', mew=0.5, zorder=3, **ebar)
        ax.plot(piv, rec[f'mag_pred_{k}'], 'x', color=fcolor[k], ms=8, mew=1.6, zorder=4)
        mags += [mo, rec[f'mag_pred_{k}']]

    # legend: sort by pivot
    order = sorted(filt, key=lambda k: filt[k]['pivot_AA'])
    handles=[]
    for k in order:
        s = SRC[k]
        if k in _HELD: tag = 'Euclid TEST'
        elif s=='ours' and k in EUCLID_KEYS: tag = 'Euclid'
        elif s=='ours': tag = 'HST' if k=='F814W' else 'JWST'
        elif s=='cw': tag = MIS[k]
        elif s=='sdss': tag = 'SDSS'
        elif s=='ps1': tag = 'PS1'
        else: tag = MIS[k]
        handles.append(Line2D([0],[0], marker=marker(k), color='w',
                       markerfacecolor=fcolor[k], markeredgecolor='0.2', markersize=7,
                       label=f'{k}  ({filt[k]["pivot_AA"]/1e4:.2f}μm, {tag})'))
    handles.append(Line2D([0],[0], marker='x', color='k', linestyle='none', markersize=8, mew=1.6, label='× = predicted'))
    handles.append(Line2D([0],[0], color='0.6', lw=1.1, label=f'blackbody T={T:.0f}K'))
    ax.legend(handles=handles, fontsize=5.6, loc='lower center', ncol=3, framealpha=0.9)

    ax.set_xscale('log')
    if mags:
        lo, hi = min(mags), max(mags); pad = max(0.3, 0.05*(hi-lo))
        ax.set_ylim(hi+pad, lo-pad)
    else:
        ax.invert_yaxis()
    ax.set_xlabel('wavelength (μm)'); ax.set_ylabel('AB magnitude')
    suff = f', Euclid held out' if _HELD else ', GLOBAL fit'
    ax.set_title(f'{int(rec["id"])}  —  T={T:.0f}K, χ²ν={rec["chi2_red"]:.2f}  '
                 f'({_VER}{suff}, {int(rec["n_fit_bands"])} bands)')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_worker(args):
    records, seddir, ver, bands_sub, held = args
    global _VER, _BANDS_SUB, _HELD
    _VER = ver; _BANDS_SUB = bands_sub; _HELD = held
    keys = [b[0] for b in bands_sub]
    filt = load_filters(keys); shape_grid = precompute_shape(filt)
    sd = Path(seddir)
    for rec in records:
        make_sed_plot(rec, filt, shape_grid, sd/f'sed_{int(rec["id"])}.png')
    return len(records)


def build_webform(res_plotted, res_all, html_dir, ver, per_page=60, prefix='index', banner=None):
    n=len(res_plotted); n_pages=(n+per_page-1)//per_page
    if n_pages==0:
        print(f'  Webform: no rows for prefix={prefix}'); return
    off=(html_dir/'band_offset_summary.txt').read_text() if (html_dir/'band_offset_summary.txt').exists() else ''
    def pname(p): return f'{prefix}.html' if p==0 else f'{prefix}_{p:02d}.html'
    bnr = f'<div style="background:#fee;border:1px solid #c88;padding:6px 10px;color:#900;margin-bottom:10px;">{banner}</div>' if banner else ''
    for pg in range(n_pages):
        sub=res_plotted.iloc[pg*per_page:(pg+1)*per_page]; cards=[]
        for _, r in sub.iterrows():
            cards.append(f'<div class="card"><img src="seds/sed_{int(r["id"])}.png" loading="lazy">'
                         f'<div class="meta">{int(r["id"])} &nbsp; T={r["T_bb"]:.0f}K &nbsp; χ²ν={r["chi2_red"]:.1f}</div></div>')
        def lk(p):
            return f'<b>{p+1}</b>' if p==pg else f'<a href="{pname(p)}">{p+1}</a>'
        win=sorted({0,1,n_pages-2,n_pages-1}|set(range(max(0,pg-3),min(n_pages,pg+4)))); win=[p for p in win if 0<=p<n_pages]
        parts=[]; prev=None
        if pg>0: parts.append(f'<a href="{pname(pg-1)}">&larr;prev</a>')
        for p in win:
            if prev is not None and p!=prev+1: parts.append('…')
            parts.append(lk(p)); prev=p
        if pg<n_pages-1: parts.append(f'<a href="{pname(pg+1)}">next&rarr;</a>')
        nav=' '.join(parts)
        html=f'''<!doctype html><html><head><meta charset="utf-8"><title>SED {ver} {prefix} — page {pg+1}/{n_pages}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fafafa;}}h1{{font-size:18px;}}
.summary{{background:#fff;border:1px solid #ddd;padding:8px 12px;font-family:monospace;white-space:pre;font-size:11px;margin-bottom:12px;}}
.nav{{margin:10px 0;}}.nav a{{margin:0 4px;}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}}
.card{{background:#fff;border:1px solid #ddd;border-radius:5px;padding:6px;}}.card img{{width:100%;}}
.meta{{font-family:monospace;font-size:11px;margin-top:4px;}}</style></head><body>
<h1>SED {ver} — multi-facility blackbody fit
<span style="font-weight:normal;font-size:13px;color:#666;">{len(res_all):,} sources; page {pg+1}/{n_pages}</span></h1>
{bnr}<div class="summary">{off}</div><div class="nav">{nav}</div><div class="grid">{''.join(cards)}</div><div class="nav">{nav}</div>
</body></html>'''
        (html_dir/pname(pg)).write_text(html)
    print(f'  Webform ({prefix}): {n_pages} pages → htmls/sed_{ver}/{prefix}.html')


def summarize(res, filt, html_dir, ver, bands_sub, held):
    lines = [f'SED {ver} offset test',
             f'Fit excludes Euclid: {bool(held)}.  N bands in version: {len(bands_sub)}.', '']
    lines.append(f'{"band":<12} {"mission":<11} {"source":<6} {"pivot_um":>8} {"N":>6} {"median Δ":>10} {"NMAD":>8}')
    lines.append('-'*68)
    bands_sorted = sorted(bands_sub, key=lambda b: filt[b[0]]['pivot_AA'])
    rows=[]
    for key, svo, mission, source, valcol, errcol, errkind in bands_sorted:
        if f'delta_{key}' not in res.columns: continue
        d = res[f'delta_{key}'].dropna().values
        if len(d)==0: continue
        med=np.median(d); nmad=1.4826*np.median(np.abs(d-med)); piv=filt[key]['pivot_AA']/1e4
        src_short = {'ours':'ours','cw':'CW','sdss':'SDSS','ps1':'PS1'}[source]
        lines.append(f'{key:<12} {mission:<11} {src_short:<6} {piv:>8.3f} {len(d):>6,} {med:>+10.4f} {nmad:>8.4f}')
        rows.append((key, mission, source, piv, med, nmad, len(d)))
    text='\n'.join(lines); print(text); (html_dir/'band_offset_summary.txt').write_text(text+'\n')

    fig, ax = plt.subplots(figsize=(12, 5.5))
    cmap = {'HST':'k','JWST':'C5','Euclid':'C6','CFHT':'C4','HSC':'C0','UltraVISTA':'C2',
            'IRAC':'C1','JWST-MIRI':'C3','SDSS':'C8','PS1':'C9'}
    for key, mission, source, piv, med, nmad, n in rows:
        mk = {'ours':'D' if key in EUCLID_KEYS else 'o','cw':'s','sdss':'^','ps1':'v'}[source]
        if key in held: mk='D'
        ax.errorbar(piv, med, yerr=nmad/np.sqrt(n), fmt=mk, ms=10,
                    color=cmap.get(mission,'C7'), capsize=4, mec='0.2')
        ax.annotate(key, (piv, med), fontsize=6.5, xytext=(0,9), textcoords='offset points', ha='center')
    ax.axhline(0, color='k', lw=0.8); ax.set_xscale('log')
    ax.set_xlabel('pivot wavelength (μm)'); ax.set_ylabel('median Δmag (obs − blackbody)')
    extra = ' (Euclid held out)' if held else ''
    ax.set_title(f'SED {ver} band offsets vs blackbody{extra}; circle=ours, square=CW, ^=SDSS, v=PS1')
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=9,label=m) for m,c in cmap.items()]
    ax.legend(handles=leg, fontsize=8, ncol=3)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(html_dir/'band_offset_summary.png', dpi=130); plt.close(fig)


def run_version(ver, df_master, filt_all, shape_all):
    print(f'\n========== {ver} ==========')
    subset_keys, held = version_config(ver)
    bands_sub = [b for b in ALL_BANDS if b[0] in subset_keys]
    keys = [b[0] for b in bands_sub]
    filt = {k: filt_all[k] for k in keys}
    shape_grid = {k: shape_all[k] for k in keys}
    in_fit = {k: (k not in held) for k in keys}

    html_dir = ROOT/'htmls'/f'sed_{ver}'
    seddir = html_dir/'seds'
    html_dir.mkdir(parents=True, exist_ok=True); seddir.mkdir(parents=True, exist_ok=True)

    # Fit each star
    t0 = time.time()
    recs = []
    for _, row in df_master.iterrows():
        mobs, merr = {}, {}
        # per-source CW fill detection
        cw_flux = {b[0]: row.get(f'f_{b[4]}', np.nan) for b in bands_sub if b[3]=='cw'}
        fcount = {}
        for v in cw_flux.values():
            if np.isfinite(v): fcount[v] = fcount.get(v,0)+1
        dup_flux = {v for v,c in fcount.items() if c>=2}
        valid = {}
        for b in bands_sub:
            key, svo, mission, source, valcol, errcol, errkind = b
            if source == 'ours':
                m = row[valcol]
                e = row[errcol] if errkind=='magerr' else (1.0857/row[errcol] if (np.isfinite(row[errcol]) and row[errcol]>0) else np.nan)
                mobs[key]=m; merr[key]=e
                valid[key] = np.isfinite(m) and np.isfinite(e) and (e < MAGERR_MAX)
            elif source == 'cw':
                f = row.get(f'f_{valcol}', np.nan); fe = row.get(f'fe_{valcol}', np.nan)
                with np.errstate(invalid='ignore', divide='ignore'):
                    m = -2.5*np.log10(f)+23.9 if (np.isfinite(f) and f>0) else np.nan
                    e = 1.0857*fe/f if (np.isfinite(f) and f>0 and np.isfinite(fe)) else np.nan
                mobs[key]=m; merr[key]=e
                valid[key] = (np.isfinite(m) and np.isfinite(e) and (f>0)
                              and np.isfinite(fe) and (fe < FE_SENTINEL)
                              and (e < MAGERR_MAX) and (f not in dup_flux))
            else:   # sdss / ps1
                m = row.get(valcol, np.nan); e = row.get(errcol, np.nan)
                mobs[key]=m; merr[key]=e
                valid[key] = (np.isfinite(m) and np.isfinite(e) and (m < 29) and (e < MAGERR_MAX))
        fit_keys = [k for k in keys if valid[k] and in_fit[k]]
        if len(fit_keys) < MIN_FIT_BANDS: continue
        T, dm, chi2, dof = fit_bb(mobs, merr, shape_grid, fit_keys)
        rec = {'id': int(row['jwst_id_int']), 'T_bb': T, 'dm': dm, 'chi2': chi2,
               'dof': dof, 'chi2_red': chi2/max(dof,1), 'n_fit_bands': len(fit_keys),
               'is_agn_qso': bool(row.get('is_agn_qso', False))}
        for b in bands_sub:
            key = b[0]
            shp = np.interp(T, T_GRID, shape_grid[key]); pred = shp + dm
            rec[f'mag_obs_{key}']  = mobs[key] if valid[key] else np.nan
            rec[f'magerr_{key}']   = merr[key] if valid[key] else np.nan
            rec[f'mag_pred_{key}'] = pred
            rec[f'delta_{key}']    = (mobs[key]-pred) if valid[key] else np.nan
        recs.append(rec)
    res = pd.DataFrame(recs)
    print(f'  Fit {len(res):,} sources in {time.time()-t0:.1f}s')

    res['id'] = res['id'].astype('int64')
    res.to_parquet(OUT/f'sed_blackbody_{ver}.parquet', index=False)
    res.to_csv(OUT/f'sed_blackbody_{ver}.csv', index=False)

    res_star = res[~res['is_agn_qso'].astype(bool)].reset_index(drop=True)
    res_agn  = res[res['is_agn_qso'].astype(bool)].reset_index(drop=True)
    print(f'  pure stars: {len(res_star):,}   AGN/QSO kept: {len(res_agn):,}')
    summarize(res_star, filt, html_dir, ver, bands_sub, held)

    # plot all (stars + AGN) so AGN photometry stays inspectable
    recs_list = res.sort_values('mag_obs_F814W').to_dict('records')
    nw=12; chunks=[recs_list[i::nw] for i in range(nw)]
    print(f'  Generating {len(recs_list):,} SED PNGs ...'); tp=time.time(); done=0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        for k in pool.map(_plot_worker, [(c, str(seddir), ver, bands_sub, held) for c in chunks]):
            done+=k
    print(f'    {done:,} PNGs in {time.time()-tp:.1f}s')
    build_webform(res_star.sort_values('mag_obs_F814W').reset_index(drop=True),
                  res_star, html_dir, ver, prefix='index')
    build_webform(res_agn.sort_values('mag_obs_F814W').reset_index(drop=True),
                  res_agn, html_dir, ver, prefix='index_agn',
                  banner='AGN/QSO — EXCLUDED from the calibration test; photometry kept for reference.')


def main():
    versions = sys.argv[1:] or ['v06', 'v07', 'v08', 'v09']
    print(f'Will run versions: {versions}')

    # Load filters & shape grid once (covers all 30 bands)
    print('Loading all filters ...')
    filt_all  = load_filters([b[0] for b in ALL_BANDS])
    print('Precomputing shape grid (all 30 bands × 400 T) ...')
    shape_all = precompute_shape(filt_all)

    # Load data once: clean sample + PS1/SDSS join + CW
    print('Loading clean sample, PS1/SDSS joined master, COSMOS-Web ...')
    clean = pd.read_parquet(OUT/'sed_clean_sample_v01.parquet')
    clean['jwst_id_int'] = pd.to_numeric(clean['jwst_id'], errors='coerce').astype('Int64')
    clean = clean[clean['jwst_id_int'].notna()].copy()
    print(f'  clean: {len(clean):,}')

    master = pd.read_parquet(OUT/'master_stars_4way_with_ps1_sdss.parquet')
    # bring PS1+SDSS cols from master to clean by jwst_id
    master['jid'] = pd.to_numeric(master['jwst_id'], errors='coerce').astype('Int64')
    ps_sd_cols = [c for c in master.columns if c.startswith('ps1_') or c.startswith('sdss_')]
    df = clean.merge(master[['jid']+ps_sd_cols], left_on='jwst_id_int', right_on='jid', how='left')

    # COSMOS-Web fluxes
    ids = set(int(i) for i in df['jwst_id_int'].values)
    with fits.open(CW_PATH) as h:
        d = h['PHOTOMETRY HOTCOLD AND SE++'].data
        cw_id = np.asarray(d['id']).astype(np.int64)
        sel = np.isin(cw_id, list(ids))
        cwtab = {'jwst_id_int': cw_id[sel].astype('int64')}
        for suf in CW_SUFFIXES:
            cwtab[f'f_{suf}']  = np.asarray(d[f'flux_model_{suf}']).astype(float)[sel]
            cwtab[f'fe_{suf}'] = np.asarray(d[f'flux_err-cal_model_{suf}']).astype(float)[sel]
    cwdf = pd.DataFrame(cwtab)
    df = df.merge(cwdf, on='jwst_id_int', how='left')
    print(f'  merged → {len(df):,} stars (each row has ours+CW+PS1+SDSS cols, NaN where no data)')

    # PS1/SDSS coverage
    n_ps  = int(df['ps1_imag'].notna().sum())
    n_sd  = int(df['sdss_psfMag_i'].notna().sum())
    print(f'  with PS1 match : {n_ps:,}')
    print(f'  with SDSS match: {n_sd:,}')

    for ver in versions:
        run_version(ver, df, filt_all, shape_all)

    print(f'\nAll done.')


if __name__ == '__main__':
    main()
