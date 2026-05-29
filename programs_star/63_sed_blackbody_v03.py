#!/usr/bin/env python
"""
63_sed_blackbody_v03.py — SED v03: blackbody anchored on OUR HST+JWST
photometry, testing OUR Euclid bands + COSMOS-Web's other-facility bands.

Per user (2026-05-27):
  - Start from OUR master table; use stars that have HST + JWST + Euclid
    photometry (= sed_clean_sample_v01, all 9 mission bands clean).
  - Fit a blackbody to the TRUSTED anchors: OUR HST F814W + JWST
    NIRCam F115/F150/F277/F444W (Step-3 catalog magnitudes).
  - Test bands:
      * Euclid VIS / NISP Y,J,H  — OUR photometry (cat_mag_*)
      * CFHT u; Subaru HSC g/r/i/z/y; Spitzer IRAC ch1-4; JWST MIRI F770W
        — looked up from COSMOS-Web flux_model (μJy) by JWST id, where
        a valid measurement is available.

Validity (drop fill / sentinel / low-SNR):
  flux>0, flux_err<1e10, SNR>3, and flux not bit-identical across ≥2
  COSMOS-Web bands (per-source fill).

Outputs:
  csvfiles_star/sed_blackbody_v03.parquet (+ .csv)
  htmls/sed_v03/band_offset_summary.png/.txt
  htmls/sed_v03/seds/sed_<id>.png
  htmls/sed_v03/index.html  (paginated)
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
HTML = ROOT / 'htmls' / 'sed_v03'
SEDDIR = HTML / 'seds'
FILTER_CACHE = OUT / 'sed_filters'
CW_PATH = '/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1.fits'

H = 6.62607015e-27; C_AA = 2.99792458e18; K = 1.380649e-16

# (key, svo_id, mission, source, valcol, errcol, errkind, is_anchor)
#  source 'ours' → value from clean-sample column (AB mag); errkind magerr/snr
#  source 'cw'   → value from COSMOS-Web flux_model_<valcol> (μJy)
BANDS = [
    ('F814W',      'HST/ACS_WFC.F814W', 'HST',       'ours', 'cat_mag_F814W_hst',  'cat_magerr_F814W_hst',  'magerr', True),
    ('F115W',      'JWST/NIRCam.F115W', 'JWST',      'ours', 'cat_mag_F115W_jwst', 'cat_snr_F115W_jwst',    'snr',    True),
    ('F150W',      'JWST/NIRCam.F150W', 'JWST',      'ours', 'cat_mag_F150W_jwst', 'cat_snr_F150W_jwst',    'snr',    True),
    ('F277W',      'JWST/NIRCam.F277W', 'JWST',      'ours', 'cat_mag_F277W_jwst', 'cat_snr_F277W_jwst',    'snr',    True),
    ('F444W',      'JWST/NIRCam.F444W', 'JWST',      'ours', 'cat_mag_F444W_jwst', 'cat_snr_F444W_jwst',    'snr',    True),
    ('Euclid_VIS', 'Euclid/VIS.vis',    'Euclid',    'ours', 'cat_mag_VIS_vis',    'cat_magerr_VIS_vis',    'magerr', False),
    ('Euclid_Y',   'Euclid/NISP.Y',     'Euclid',    'ours', 'cat_mag_NIR_Y_nisp', 'cat_magerr_NIR_Y_nisp', 'magerr', False),
    ('Euclid_J',   'Euclid/NISP.J',     'Euclid',    'ours', 'cat_mag_NIR_J_nisp', 'cat_magerr_NIR_J_nisp', 'magerr', False),
    ('Euclid_H',   'Euclid/NISP.H',     'Euclid',    'ours', 'cat_mag_NIR_H_nisp', 'cat_magerr_NIR_H_nisp', 'magerr', False),
    ('CFHT_u',     'CFHT/MegaCam.u',    'CFHT',      'cw',   'cfht-u',   None, None, False),
    ('HSC_g',      'Subaru/HSC.g',      'HSC',       'cw',   'hsc-g',    None, None, False),
    ('HSC_r',      'Subaru/HSC.r',      'HSC',       'cw',   'hsc-r',    None, None, False),
    ('HSC_i',      'Subaru/HSC.i',      'HSC',       'cw',   'hsc-i',    None, None, False),
    ('HSC_z',      'Subaru/HSC.z',      'HSC',       'cw',   'hsc-z',    None, None, False),
    ('HSC_y',      'Subaru/HSC.Y',      'HSC',       'cw',   'hsc-y',    None, None, False),
    ('IRAC_ch1',   'Spitzer/IRAC.I1',   'IRAC',      'cw',   'irac-ch1', None, None, False),
    ('IRAC_ch2',   'Spitzer/IRAC.I2',   'IRAC',      'cw',   'irac-ch2', None, None, False),
    ('IRAC_ch3',   'Spitzer/IRAC.I3',   'IRAC',      'cw',   'irac-ch3', None, None, False),
    ('IRAC_ch4',   'Spitzer/IRAC.I4',   'IRAC',      'cw',   'irac-ch4', None, None, False),
    ('MIRI_F770W', 'JWST/MIRI.F770W',   'JWST-MIRI', 'cw',   'f770w',    None, None, False),
]
ANCHORS = [b for b in BANDS if b[7]]
TESTS   = [b for b in BANDS if not b[7]]
CW_SUFFIXES = [b[4] for b in BANDS if b[3] == 'cw']
T_GRID  = np.geomspace(2000.0, 50000.0, 400)
SNR_MIN_TEST = 3.0
MAGERR_MAX_TEST = 1.0857 / SNR_MIN_TEST
FE_SENTINEL = 1e10


def load_filters():
    FILTER_CACHE.mkdir(parents=True, exist_ok=True)
    from astroquery.svo_fps import SvoFps
    filt = {}
    for key, svo_id, *_ in BANDS:
        cache = FILTER_CACHE / f'v03_{key}.npz'
        if cache.exists():
            d = np.load(cache); wl, tr = d['wl'], d['tr']
        else:
            print(f'[filters] fetch {svo_id}')
            t = SvoFps.get_transmission_data(svo_id)
            wl = np.asarray(t['Wavelength'], float); tr = np.asarray(t['Transmission'], float)
            np.savez(cache, wl=wl, tr=tr)
        good = tr > 0; wl, tr = wl[good], tr[good]
        piv = np.sqrt(np.trapezoid(tr, wl) / np.trapezoid(tr / wl**2, wl))
        filt[key] = {'wl': wl, 'tr': tr, 'pivot_AA': piv}
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


def fit_bb(mobs, merr, shape_grid, anchor_keys):
    w = np.array([1.0/(merr[k]**2 + 0.03**2) for k in anchor_keys])
    m = np.array([mobs[k] for k in anchor_keys])
    smat = np.array([shape_grid[k] for k in anchor_keys])
    chi2 = np.full(len(T_GRID), np.inf); dm = np.zeros(len(T_GRID))
    for i in range(len(T_GRID)):
        sh = smat[:, i]; d = np.sum(w*(m-sh))/np.sum(w)
        chi2[i] = np.sum(w*(m-sh-d)**2); dm[i] = d
    ib = np.argmin(chi2)
    return T_GRID[ib], dm[ib], chi2[ib], len(anchor_keys)-2


def _filter_colors(filt):
    logp = {k: np.log10(filt[k]['pivot_AA']) for k in filt}
    lmin, lmax = min(logp.values()), max(logp.values())
    cmap = plt.cm.rainbow
    return {k: cmap((logp[k]-lmin)/(lmax-lmin)) for k in filt}


def main(validate_only=False):
    t0 = time.time()
    HTML.mkdir(parents=True, exist_ok=True); SEDDIR.mkdir(parents=True, exist_ok=True)
    print('Loading filters ...'); filt = load_filters()
    print('Precomputing shape grid ...'); shape_grid = precompute_shape(filt)

    # sample: clean stars (HST+JWST+Euclid all present) from our master
    clean = pd.read_parquet(OUT / 'sed_clean_sample_v01.parquet')
    clean['jwst_id_int'] = pd.to_numeric(clean['jwst_id'], errors='coerce').astype('Int64')
    clean = clean[clean['jwst_id_int'].notna()].copy()
    print(f'Clean stars with HST+JWST+Euclid: {len(clean):,}')
    ids = set(int(i) for i in clean['jwst_id_int'].values)

    # COSMOS-Web fluxes for the cw test bands, keyed by id (=jwst id)
    print('Loading COSMOS-Web fluxes for CFHT/HSC/IRAC/MIRI ...')
    with fits.open(CW_PATH) as h:
        d = h['PHOTOMETRY HOTCOLD AND SE++'].data
        cw_id = np.asarray(d['id']).astype(np.int64)
        sel = np.isin(cw_id, list(ids))
        cwtab = {'jwst_id_int': cw_id[sel].astype('int64')}
        for suf in CW_SUFFIXES:
            cwtab[f'f_{suf}']  = np.asarray(d[f'flux_model_{suf}']).astype(float)[sel]
            cwtab[f'fe_{suf}'] = np.asarray(d[f'flux_err-cal_model_{suf}']).astype(float)[sel]
    cwdf = pd.DataFrame(cwtab)
    df = clean.merge(cwdf, on='jwst_id_int', how='left')
    print(f'  merged → {len(df):,} stars')

    anchor_keys = [b[0] for b in ANCHORS]
    recs = []
    for _, row in df.iterrows():
        mobs, merr = {}, {}
        # per-source fill detection on the CW bands (bit-identical flux)
        cw_flux = {b[0]: row.get(f'f_{b[4]}', np.nan) for b in BANDS if b[3]=='cw'}
        fcount = {}
        for v in cw_flux.values():
            if np.isfinite(v): fcount[v] = fcount.get(v,0)+1
        dup_flux = {v for v,c in fcount.items() if c>=2}
        valid = {}
        for key, svo, mission, source, valcol, errcol, errkind, anch in BANDS:
            if source == 'ours':
                m = row[valcol]
                if errkind == 'magerr':
                    e = row[errcol]
                else:
                    snr = row[errcol]; e = 1.0857/snr if (snr and snr>0) else np.nan
                mobs[key] = m; merr[key] = e
                valid[key] = np.isfinite(m) and np.isfinite(e) and (e < MAGERR_MAX_TEST)
            else:  # cw
                f = row.get(f'f_{valcol}', np.nan); fe = row.get(f'fe_{valcol}', np.nan)
                with np.errstate(invalid='ignore', divide='ignore'):
                    m = -2.5*np.log10(f)+23.9 if (np.isfinite(f) and f>0) else np.nan
                    e = 1.0857*fe/f if (np.isfinite(f) and f>0 and np.isfinite(fe)) else np.nan
                mobs[key] = m; merr[key] = e
                valid[key] = (np.isfinite(m) and np.isfinite(e) and (f>0)
                              and np.isfinite(fe) and (fe < FE_SENTINEL)
                              and (e < MAGERR_MAX_TEST) and (f not in dup_flux))
        # anchors: valid AND SNR>10
        if any((not valid[a]) or (merr[a] > 0.1086) for a in anchor_keys):
            continue
        T, dm, chi2, dof = fit_bb(mobs, merr, shape_grid, anchor_keys)
        rec = {'id': int(row['jwst_id_int']), 'T_bb': T, 'dm': dm,
               'chi2': chi2, 'dof': dof, 'chi2_red': chi2/max(dof,1),
               'is_agn_qso': bool(row.get('is_agn_qso', False))}
        for key, svo, mission, source, valcol, errcol, errkind, anch in BANDS:
            shp = np.interp(T, T_GRID, shape_grid[key]); pred = shp + dm
            rec[f'mag_obs_{key}']  = mobs[key] if valid[key] else np.nan
            rec[f'magerr_{key}']   = merr[key] if valid[key] else np.nan
            rec[f'mag_pred_{key}'] = pred
            if not anch:
                rec[f'delta_{key}'] = (mobs[key]-pred) if valid[key] else np.nan
        recs.append(rec)
    res = pd.DataFrame(recs)
    print(f'Fit {len(res):,} stars in {time.time()-t0:.1f}s')

    if validate_only:
        r = res.sort_values('mag_obs_F814W').iloc[0]
        print(f'\n[validate] id={int(r["id"])} T={r["T_bb"]:.0f}K chi2_red={r["chi2_red"]:.2f}')
        for b in TESTS:
            k=b[0]
            if np.isfinite(r.get(f'delta_{k}',np.nan)):
                print(f'  {k:<11} obs={r[f"mag_obs_{k}"]:.3f} pred={r[f"mag_pred_{k}"]:.3f} Δ={r[f"delta_{k}"]:+.3f}')
        make_sed_plot(r, filt, shape_grid, SEDDIR/f'sed_{int(r["id"])}.png')
        print(f'[validate] → seds/sed_{int(r["id"])}.png')
        return

    res['id'] = res['id'].astype('int64')
    res.to_parquet(OUT/'sed_blackbody_v03.parquet', index=False)
    res.to_csv(OUT/'sed_blackbody_v03.csv', index=False)
    print(f'Wrote sed_blackbody_v03.parquet ({len(res):,} rows; photometry kept for all)')

    # Exclude AGN/QSO from calibration stats & main webform; keep photometry.
    res_star = res[~res['is_agn_qso'].astype(bool)].reset_index(drop=True)
    res_agn  = res[res['is_agn_qso'].astype(bool)].reset_index(drop=True)
    print(f'Pure stars: {len(res_star):,}   AGN/QSO (kept, excluded from test): {len(res_agn):,}')
    summarize(res_star, filt)

    recs_list = res.sort_values('mag_obs_F814W').to_dict('records')
    nw=12; chunks=[recs_list[i::nw] for i in range(nw)]
    print(f'Generating {len(recs_list):,} SED PNGs ...'); tp=time.time(); done=0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        for k in pool.map(_plot_worker, [(c,str(SEDDIR)) for c in chunks]):
            done+=k
    print(f'  {done:,} PNGs in {time.time()-tp:.1f}s')
    build_webform(res_star.sort_values('mag_obs_F814W').reset_index(drop=True),
                  res_star, prefix='index')
    build_webform(res_agn.sort_values('mag_obs_F814W').reset_index(drop=True),
                  res_agn, prefix='index_agn',
                  banner='AGN/QSO — EXCLUDED from the calibration test; photometry kept for reference.')
    print(f'\nWall time: {time.time()-t0:.1f}s')


def _plot_worker(args):
    records, seddir = args
    filt = load_filters(); shape_grid = precompute_shape(filt); sd = Path(seddir)
    for rec in records:
        make_sed_plot(rec, filt, shape_grid, sd/f'sed_{int(rec["id"])}.png')
    return len(records)


def make_sed_plot(rec, filt, shape_grid, path):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    T = rec['T_bb']; dm = rec['dm']; fcolor = _filter_colors(filt)
    SRC = {b[0]: b[3] for b in BANDS}; MIS = {b[0]: b[2] for b in BANDS}
    ANCH = {b[0]: b[7] for b in BANDS}
    # marker: circle anchor, diamond Euclid(ours-test), square cw-test
    def marker(k):
        if ANCH[k]: return 'o'
        return 'D' if SRC[k]=='ours' else 's'

    wl_um = np.geomspace(0.3, 9.0, 500); wl_AA = wl_um*1e4
    mmod = -2.5*np.log10(bb_fnu(wl_AA, T)) + dm
    shp_anchor = np.interp(T, T_GRID, shape_grid['F150W']) + dm
    mod_at = -2.5*np.log10(bb_fnu(np.array([filt['F150W']['pivot_AA']]), T))[0] + dm
    ax.plot(wl_um, mmod + (shp_anchor-mod_at), '-', color='0.6', lw=1.1, zorder=1)

    ebar = dict(ecolor='0.25', elinewidth=1.1, capsize=3.5, capthick=1.1)
    mags = []
    for key, svo, mission, source, valcol, errcol, errkind, anch in BANDS:
        piv = filt[key]['pivot_AA']/1e4
        mo = rec[f'mag_obs_{key}']; me = rec[f'magerr_{key}']
        if not np.isfinite(mo): continue
        ax.errorbar(piv, mo, yerr=me if np.isfinite(me) else None, fmt=marker(key),
                    color=fcolor[key], ms=8, mec='0.2', mew=0.5, zorder=3, **ebar)
        mags.append(mo)
        if not anch:
            ax.plot(piv, rec[f'mag_pred_{key}'], 'x', color=fcolor[key], ms=9, mew=2, zorder=4)
            mags.append(rec[f'mag_pred_{key}'])

    order = sorted(filt, key=lambda k: filt[k]['pivot_AA'])
    handles=[]
    for k in order:
        tag = 'anchor' if ANCH[k] else ('Euclid/ours' if SRC[k]=='ours' else MIS[k])
        handles.append(Line2D([0],[0], marker=marker(k), color='w', markerfacecolor=fcolor[k],
                       markeredgecolor='0.2', markersize=8,
                       label=f'{k}  ({filt[k]["pivot_AA"]/1e4:.2f}μm, {tag})'))
    handles.append(Line2D([0],[0], marker='x', color='k', linestyle='none', markersize=9, mew=2, label='× = predicted'))
    handles.append(Line2D([0],[0], color='0.6', lw=1.1, label=f'blackbody T={T:.0f}K'))
    ax.legend(handles=handles, fontsize=6.6, loc='lower center', ncol=2, framealpha=0.9)

    ax.set_xscale('log')
    if mags:
        lo, hi = min(mags), max(mags); pad = max(0.3, 0.05*(hi-lo))
        ax.set_ylim(hi+pad, lo-pad)
    else:
        ax.invert_yaxis()
    ax.set_xlabel('wavelength (μm)'); ax.set_ylabel('AB magnitude')
    ax.set_title(f'{int(rec["id"])}  —  T={T:.0f}K, χ²ν={rec["chi2_red"]:.2f}  (anchors: HST+JWST, ours)')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def summarize(res, filt):
    lines = ['SED v03 offset test (observed - HST+JWST-blackbody-predicted)',
             'Anchor = OUR HST F814W + JWST NIRCam.  Test = OUR Euclid + COSMOS-Web CFHT/HSC/IRAC/MIRI.', '']
    lines.append(f'{"band":<12} {"mission":<11} {"source":<6} {"pivot_um":>8} {"N":>6} {"median Δ":>10} {"NMAD":>8}')
    lines.append('-'*68)
    test_sorted = sorted(TESTS, key=lambda b: filt[b[0]]['pivot_AA'])
    rows=[]
    for key, svo, mission, source, valcol, errcol, errkind, anch in test_sorted:
        d = res[f'delta_{key}'].dropna().values
        if len(d)==0: continue
        med=np.median(d); nmad=1.4826*np.median(np.abs(d-med)); piv=filt[key]['pivot_AA']/1e4
        src = 'ours' if source=='ours' else 'CW'
        lines.append(f'{key:<12} {mission:<11} {src:<6} {piv:>8.3f} {len(d):>6,} {med:>+10.4f} {nmad:>8.4f}')
        rows.append((key, mission, source, piv, med, nmad, len(d)))
    text='\n'.join(lines); print('\n'+text); (HTML/'band_offset_summary.txt').write_text(text+'\n')

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = {'CFHT':'C4','HSC':'C0','UltraVISTA':'C2','IRAC':'C1','JWST-MIRI':'C3','Euclid':'C6'}
    for key, mission, source, piv, med, nmad, n in rows:
        mk = 'D' if source=='ours' else 's'
        ax.errorbar(piv, med, yerr=nmad/np.sqrt(n), fmt=mk, ms=10,
                    color=cmap.get(mission,'C7'), capsize=4, mec='0.2')
        ax.annotate(key, (piv, med), fontsize=7, xytext=(0,9), textcoords='offset points', ha='center')
    ax.axhline(0, color='k', lw=0.8); ax.set_xscale('log')
    ax.set_xlabel('pivot wavelength (μm)'); ax.set_ylabel('median Δmag (obs − HST+JWST blackbody)')
    ax.set_title('SED v03 band offsets — Euclid (diamond) + COSMOS-Web (square), anchored on HST+JWST')
    from matplotlib.lines import Line2D
    leg = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=9,label=m) for m,c in cmap.items()]
    ax.legend(handles=leg, fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(HTML/'band_offset_summary.png', dpi=130); plt.close(fig)


def build_webform(res_plotted, res_all, per_page=60, prefix='index', banner=None):
    n=len(res_plotted); n_pages=(n+per_page-1)//per_page
    if n_pages==0:
        print(f'Webform: no rows for prefix={prefix}'); return
    off=(HTML/'band_offset_summary.txt').read_text() if (HTML/'band_offset_summary.txt').exists() else ''
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
        html=f'''<!doctype html><html><head><meta charset="utf-8"><title>SED v03 {prefix} — page {pg+1}/{n_pages}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fafafa;}}h1{{font-size:18px;}}
.summary{{background:#fff;border:1px solid #ddd;padding:8px 12px;font-family:monospace;white-space:pre;font-size:11px;margin-bottom:12px;}}
.nav{{margin:10px 0;}}.nav a{{margin:0 4px;}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}}
.card{{background:#fff;border:1px solid #ddd;border-radius:5px;padding:6px;}}.card img{{width:100%;}}
.meta{{font-family:monospace;font-size:11px;margin-top:4px;}}</style></head><body>
<h1>SED v03 — Euclid + COSMOS-Web offsets vs HST+JWST blackbody
<span style="font-weight:normal;font-size:13px;color:#666;">{len(res_all):,} sources; page {pg+1}/{n_pages}</span></h1>
{bnr}<div class="summary">{off}</div><div class="nav">{nav}</div><div class="grid">{''.join(cards)}</div><div class="nav">{nav}</div>
</body></html>'''
        (HTML/pname(pg)).write_text(html)
    print(f'Webform ({prefix}): {n_pages} pages → htmls/sed_v03/{prefix}.html')


if __name__ == '__main__':
    main(validate_only='--validate' in sys.argv)
