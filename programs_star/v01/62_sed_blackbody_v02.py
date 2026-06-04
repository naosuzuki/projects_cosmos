#!/usr/bin/env python
"""
62_sed_blackbody_v02.py — SED v02: test ALL COSMOS-Web telescope bands
against an HST+JWST-anchored blackbody.

The COSMOS-Web mastercatalog v1.1 records forced model photometry
(flux_model_<band>, μJy) from many facilities.  We:
  1. Cross-match the clean star sample (sed_clean_sample_v01) to COSMOS-Web
     by JWST id (== CW 'id').
  2. Fit a blackbody to the TRUSTED anchors — CW's own HST ACS F814W and
     JWST NIRCam F115/F150/F277/F444W flux_model.
  3. Predict every other band (CFHT u; Subaru HSC g/r/i/z/y; UltraVISTA
     Y/J/H/Ks; Spitzer IRAC ch1-4; JWST MIRI F770W) through its SVO
     filter curve and measure Δmag = observed - predicted.

This tests how well each facility's photometry (as recorded in COSMOS-Web)
agrees with the HST+JWST continuum, assuming HST and JWST are correct.

flux → AB:  m = -2.5*log10(flux_μJy) + 23.9
            magerr = 1.0857 * flux_err / flux

Outputs:
  csvfiles_star/sed_blackbody_v02.parquet  (+ .csv)
  htmls/sed_v02/band_offset_summary.png/.txt
  htmls/sed_v02/seds/sed_<id>.png
  htmls/sed_v02/index.html  (paginated)
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
HTML = ROOT / 'htmls' / 'sed_v02'
SEDDIR = HTML / 'seds'
FILTER_CACHE = OUT / 'sed_filters'
CW_PATH = '/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1.fits'

H = 6.62607015e-27; C_AA = 2.99792458e18; K = 1.380649e-16

# (cw_suffix, svo_id, mission, is_anchor)
BANDS = [
    ('cfht-u',    'CFHT/MegaCam.u',     'CFHT',       False),
    ('hsc-g',     'Subaru/HSC.g',       'HSC',        False),
    ('hsc-r',     'Subaru/HSC.r',       'HSC',        False),
    ('hst-f814w', 'HST/ACS_WFC.F814W',  'HST',        True),
    ('hsc-i',     'Subaru/HSC.i',       'HSC',        False),
    ('hsc-z',     'Subaru/HSC.z',       'HSC',        False),
    ('hsc-y',     'Subaru/HSC.Y',       'HSC',        False),
    ('uvista-y',  'Paranal/VISTA.Y',    'UltraVISTA', False),
    ('f115w',     'JWST/NIRCam.F115W',  'JWST',       True),
    ('uvista-j',  'Paranal/VISTA.J',    'UltraVISTA', False),
    ('f150w',     'JWST/NIRCam.F150W',  'JWST',       True),
    ('uvista-h',  'Paranal/VISTA.H',    'UltraVISTA', False),
    ('uvista-ks', 'Paranal/VISTA.Ks',   'UltraVISTA', False),
    ('f277w',     'JWST/NIRCam.F277W',  'JWST',       True),
    ('irac-ch1',  'Spitzer/IRAC.I1',    'IRAC',       False),
    ('f444w',     'JWST/NIRCam.F444W',  'JWST',       True),
    ('irac-ch2',  'Spitzer/IRAC.I2',    'IRAC',       False),
    ('irac-ch3',  'Spitzer/IRAC.I3',    'IRAC',       False),
    ('f770w',     'JWST/MIRI.F770W',    'JWST-MIRI',  False),
    ('irac-ch4',  'Spitzer/IRAC.I4',    'IRAC',       False),
]
ANCHORS = [b for b in BANDS if b[3]]
TESTS   = [b for b in BANDS if not b[3]]
T_GRID  = np.geomspace(2000.0, 50000.0, 400)
MISSION_COLORS = None  # set by rainbow


def load_filters():
    FILTER_CACHE.mkdir(parents=True, exist_ok=True)
    from astroquery.svo_fps import SvoFps
    filt = {}
    for suffix, svo_id, mission, anch in BANDS:
        cache = FILTER_CACHE / f'v02_{suffix}.npz'
        if cache.exists():
            d = np.load(cache); wl, tr = d['wl'], d['tr']
        else:
            print(f'[filters] fetch {svo_id}')
            t = SvoFps.get_transmission_data(svo_id)
            wl = np.asarray(t['Wavelength'], float); tr = np.asarray(t['Transmission'], float)
            np.savez(cache, wl=wl, tr=tr)
        good = tr > 0; wl, tr = wl[good], tr[good]
        piv = np.sqrt(np.trapezoid(tr, wl) / np.trapezoid(tr / wl**2, wl))
        filt[suffix] = {'wl': wl, 'tr': tr, 'pivot_AA': piv}
    return filt


def bb_fnu(wl_AA, T):
    nu = C_AA / wl_AA; x = H * nu / (K * T)
    with np.errstate(over='ignore'):
        return (nu**3) / np.expm1(x)


def shape_mag(filt, T):
    out = {}
    for b, fd in filt.items():
        wl, tr = fd['wl'], fd['tr']
        fnu = bb_fnu(wl, T)
        out[b] = -2.5 * np.log10(np.trapezoid(fnu*tr*wl, wl) / np.trapezoid(tr*wl, wl))
    return out


def precompute_shape(filt):
    grid = {b: np.zeros(len(T_GRID)) for b in filt}
    for i, T in enumerate(T_GRID):
        s = shape_mag(filt, T)
        for b in filt: grid[b][i] = s[b]
    return grid


def fit_bb(mobs, merr, shape_grid, anchor_suffixes):
    w = np.array([1.0/(merr[b]**2 + 0.03**2) for b in anchor_suffixes])
    m = np.array([mobs[b] for b in anchor_suffixes])
    smat = np.array([shape_grid[b] for b in anchor_suffixes])
    chi2 = np.full(len(T_GRID), np.inf); dm = np.zeros(len(T_GRID))
    for i in range(len(T_GRID)):
        sh = smat[:, i]
        d = np.sum(w*(m-sh))/np.sum(w)
        chi2[i] = np.sum(w*(m-sh-d)**2); dm[i] = d
    ib = np.argmin(chi2)
    return T_GRID[ib], dm[ib], chi2[ib], len(anchor_suffixes)-2


def _filter_colors(filt):
    logp = {b: np.log10(filt[b]['pivot_AA']) for b in filt}
    lmin, lmax = min(logp.values()), max(logp.values())
    cmap = plt.cm.rainbow
    return {b: cmap((logp[b]-lmin)/(lmax-lmin)) for b in filt}


def main(validate_only=False):
    t0 = time.time()
    HTML.mkdir(parents=True, exist_ok=True); SEDDIR.mkdir(parents=True, exist_ok=True)
    print('Loading filters ...'); filt = load_filters()
    print('Precomputing shape grid ...'); shape_grid = precompute_shape(filt)

    # clean sample → JWST ids
    clean = pd.read_parquet(OUT / 'sed_clean_sample_v01.parquet')
    ids = pd.to_numeric(clean['jwst_id'], errors='coerce').astype('Int64').dropna().unique()
    ids = set(int(i) for i in ids)
    print(f'Clean stars with JWST id: {len(ids):,}')

    # load COSMOS-Web flux_model for all bands
    print('Loading COSMOS-Web photometry ...')
    with fits.open(CW_PATH) as h:
        d = h['PHOTOMETRY HOTCOLD AND SE++'].data
        cw_id = np.asarray(d['id']).astype(np.int64)
        sel = np.isin(cw_id, list(ids))
        tab = {'id': cw_id[sel]}
        for suffix, *_ in BANDS:
            tab[f'f_{suffix}']  = np.asarray(d[f'flux_model_{suffix}']).astype(float)[sel]
            tab[f'fe_{suffix}'] = np.asarray(d[f'flux_err-cal_model_{suffix}']).astype(float)[sel]
    cw = pd.DataFrame(tab)
    print(f'  matched {len(cw):,} COSMOS-Web rows')

    # convert to AB mag + magerr
    for suffix, *_ in BANDS:
        f = cw[f'f_{suffix}'].values; fe = cw[f'fe_{suffix}'].values
        with np.errstate(invalid='ignore', divide='ignore'):
            mag = np.where(f > 0, -2.5*np.log10(f) + 23.9, np.nan)
            err = np.where((f > 0) & (fe > 0), 1.0857 * fe / f, np.nan)
        cw[f'm_{suffix}']  = mag
        cw[f'me_{suffix}'] = err

    anchor_suffixes = [b[0] for b in ANCHORS]

    # fit each star
    recs = []
    SNR_MIN_TEST = 3.0          # test bands must have SNR > 3
    MAGERR_MAX_TEST = 1.0857 / SNR_MIN_TEST   # ≈ 0.362 mag
    FE_SENTINEL = 1e10          # flux_err above this = "band not measured"
    for _, row in cw.iterrows():
        mobs = {b[0]: row[f'm_{b[0]}'] for b in BANDS}
        merr = {b[0]: row[f'me_{b[0]}'] for b in BANDS}
        # detect per-source fill value: a flux that is bit-identical across
        # ≥2 bands cannot be a real measurement (e.g. id 677035 has 2.57653
        # μJy repeated in 7 bands) → mark those bands invalid.
        fcount = {}
        for b in BANDS:
            v = row[f'f_{b[0]}']
            if np.isfinite(v):
                fcount[v] = fcount.get(v, 0) + 1
        dup_flux = {v for v, c in fcount.items() if c >= 2}
        # band validity: real measurement, not a fill/sentinel
        valid = {}
        for b in BANDS:
            s = b[0]
            f = row[f'f_{s}']; fe = row[f'fe_{s}']
            valid[s] = (np.isfinite(mobs[s]) and np.isfinite(merr[s])
                        and (f > 0) and np.isfinite(fe) and (fe < FE_SENTINEL)
                        and (merr[s] < MAGERR_MAX_TEST)
                        and (f not in dup_flux))
        # require all anchors valid AND SNR>10
        if any((not valid[a]) or (merr[a] > 0.1086) for a in anchor_suffixes):
            continue
        T, dm, chi2, dof = fit_bb(mobs, merr, shape_grid, anchor_suffixes)
        rec = {'id': int(row['id']), 'T_bb': T, 'dm': dm,
               'chi2': chi2, 'dof': dof, 'chi2_red': chi2/max(dof, 1)}
        for suffix, svo, mission, anch in BANDS:
            shp = np.interp(T, T_GRID, shape_grid[suffix])
            pred = shp + dm
            # store only valid measurements; fill/sentinel → NaN
            rec[f'mag_obs_{suffix}']  = mobs[suffix] if valid[suffix] else np.nan
            rec[f'magerr_{suffix}']   = merr[suffix] if valid[suffix] else np.nan
            rec[f'mag_pred_{suffix}'] = pred
            if not anch:
                rec[f'delta_{suffix}'] = (mobs[suffix] - pred) if valid[suffix] else np.nan
        recs.append(rec)
    res = pd.DataFrame(recs)
    print(f'Fit {len(res):,} stars in {time.time()-t0:.1f}s')

    if validate_only:
        r = res.sort_values('mag_obs_hst-f814w').iloc[0]   # brightest passing star
        print(f'\n[validate] id={r["id"]}  T={r["T_bb"]:.0f}K  chi2_red={r["chi2_red"]:.2f}')
        for b in TESTS:
            s = b[0]
            if np.isfinite(r.get(f'delta_{s}', np.nan)):
                print(f'  {s:<10} obs={r[f"mag_obs_{s}"]:.3f} pred={r[f"mag_pred_{s}"]:.3f} Δ={r[f"delta_{s}"]:+.3f}')
        make_sed_plot(r, filt, shape_grid, SEDDIR/f'sed_{int(r["id"])}.png')
        print(f'[validate] → seds/sed_{int(r["id"])}.png')
        return

    res['id'] = res['id'].astype('int64')
    res.to_parquet(OUT / 'sed_blackbody_v02.parquet', index=False)
    res.to_csv(OUT / 'sed_blackbody_v02.csv', index=False)
    print(f'Wrote sed_blackbody_v02.parquet ({len(res):,} rows)')
    summarize(res, filt)

    # mass-produce SED PNGs
    res_sorted = res.sort_values('mag_obs_hst-f814w').reset_index(drop=True)
    recs_list = res_sorted.to_dict('records')
    nw = 12; chunks = [recs_list[i::nw] for i in range(nw)]
    print(f'Generating {len(recs_list):,} SED PNGs ...')
    tp = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=nw) as pool:
        for k in pool.map(_plot_worker, [(c, str(SEDDIR)) for c in chunks]):
            done += k
    print(f'  {done:,} PNGs in {time.time()-tp:.1f}s')
    build_webform(res_sorted, res)
    print(f'\nWall time: {time.time()-t0:.1f}s')


def _plot_worker(args):
    records, seddir = args
    filt = load_filters(); shape_grid = precompute_shape(filt)
    sd = Path(seddir)
    for rec in records:
        make_sed_plot(rec, filt, shape_grid, sd/f'sed_{int(rec["id"])}.png')
    return len(records)


def make_sed_plot(rec, filt, shape_grid, path):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    T = rec['T_bb']; dm = rec['dm']
    fcolor = _filter_colors(filt)
    MISSION = {b[0]: b[2] for b in BANDS}
    ANCH = {b[0]: b[3] for b in BANDS}

    wl_um = np.geomspace(0.3, 9.0, 500); wl_AA = wl_um*1e4
    mmod = -2.5*np.log10(bb_fnu(wl_AA, T)) + dm
    anchor_b = 'f150w'
    shp_anchor = np.interp(T, T_GRID, shape_grid[anchor_b]) + dm
    mod_at = -2.5*np.log10(bb_fnu(np.array([filt[anchor_b]['pivot_AA']]), T))[0] + dm
    ax.plot(wl_um, mmod + (shp_anchor - mod_at), '-', color='0.6', lw=1.1, zorder=1)

    ebar = dict(ecolor='0.25', elinewidth=1.1, capsize=3.5, capthick=1.1)
    mags_for_ylim = []     # magnitude values only — NOT error-bar extents
    for suffix, svo, mission, anch in BANDS:
        piv = filt[suffix]['pivot_AA']/1e4
        mobs = rec[f'mag_obs_{suffix}']; merr = rec[f'magerr_{suffix}']
        if not np.isfinite(mobs):
            continue
        mk = 'o' if anch else 's'
        ax.errorbar(piv, mobs, yerr=merr if np.isfinite(merr) else None,
                    fmt=mk, color=fcolor[suffix], ms=8, mec='0.2', mew=0.5,
                    zorder=3, **ebar)
        mags_for_ylim.append(mobs)
        if not anch:
            pred = rec[f'mag_pred_{suffix}']
            ax.plot(piv, pred, 'x', color=fcolor[suffix], ms=9, mew=2, zorder=4)
            mags_for_ylim.append(pred)

    order = sorted(filt, key=lambda b: filt[b]['pivot_AA'])
    handles = []
    for b in order:
        mk = 'o' if ANCH[b] else 's'
        handles.append(Line2D([0],[0], marker=mk, color='w', markerfacecolor=fcolor[b],
                       markeredgecolor='0.2', markersize=8,
                       label=f'{b}  ({filt[b]["pivot_AA"]/1e4:.2f}μm, {MISSION[b]}'
                             + (', anchor)' if ANCH[b] else ')')))
    handles.append(Line2D([0],[0], marker='x', color='k', linestyle='none',
                   markersize=9, mew=2, label='× = predicted'))
    handles.append(Line2D([0],[0], color='0.6', lw=1.1, label=f'blackbody T={T:.0f}K'))
    ax.legend(handles=handles, fontsize=6.8, loc='lower center', ncol=2, framealpha=0.9)

    ax.set_xscale('log')
    # y-axis: follow the MAGNITUDE measurement range, ignore error-bar extent
    if mags_for_ylim:
        lo, hi = min(mags_for_ylim), max(mags_for_ylim)
        pad = max(0.3, 0.05 * (hi - lo))
        ax.set_ylim(hi + pad, lo - pad)     # inverted (brighter = up)
    else:
        ax.invert_yaxis()
    ax.set_xlabel('wavelength (μm)'); ax.set_ylabel('AB magnitude')
    ax.set_title(f'{int(rec["id"])}  —  T={T:.0f} K, χ²ν={rec["chi2_red"]:.2f}  '
                 f'(anchors: HST+JWST)')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def summarize(res, filt):
    lines = ['Multi-telescope offset test (observed - HST+JWST-blackbody-predicted)',
             'Blackbody anchored on COSMOS-Web HST F814W + JWST NIRCam flux_model.', '']
    lines.append(f'{"band":<11} {"mission":<11} {"pivot_um":>8} {"N":>6} '
                 f'{"median Δ":>10} {"NMAD":>8}')
    lines.append('-'*60)
    test_sorted = sorted(TESTS, key=lambda b: filt[b[0]]['pivot_AA'])
    plot_rows = []
    for suffix, svo, mission, anch in test_sorted:
        d = res[f'delta_{suffix}'].dropna().values
        if len(d) == 0: continue
        med = np.median(d); nmad = 1.4826*np.median(np.abs(d-med))
        piv = filt[suffix]['pivot_AA']/1e4
        lines.append(f'{suffix:<11} {mission:<11} {piv:>8.3f} {len(d):>6,} '
                     f'{med:>+10.4f} {nmad:>8.4f}')
        plot_rows.append((suffix, mission, piv, med, nmad, len(d)))
    text = '\n'.join(lines); print('\n'+text)
    (HTML/'band_offset_summary.txt').write_text(text+'\n')

    # offset-vs-wavelength plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = {'CFHT':'C4','HSC':'C0','UltraVISTA':'C2','IRAC':'C1','JWST-MIRI':'C3'}
    for suffix, mission, piv, med, nmad, n in plot_rows:
        ax.errorbar(piv, med, yerr=nmad/np.sqrt(n), fmt='o', ms=9,
                    color=cmap.get(mission,'C7'), capsize=4)
        ax.annotate(suffix, (piv, med), fontsize=7, xytext=(0,8),
                    textcoords='offset points', ha='center')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('pivot wavelength (μm)')
    ax.set_ylabel('median Δmag (observed − HST+JWST blackbody)')
    ax.set_title('COSMOS-Web band offsets vs HST+JWST-anchored blackbody')
    # mission legend
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,
              markersize=9,label=m) for m,c in cmap.items()], fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(HTML/'band_offset_summary.png', dpi=130); plt.close(fig)


def build_webform(res_plotted, res_all, per_page=60):
    n = len(res_plotted); n_pages = (n+per_page-1)//per_page
    off = (HTML/'band_offset_summary.txt').read_text() if (HTML/'band_offset_summary.txt').exists() else ''
    for pg in range(n_pages):
        sub = res_plotted.iloc[pg*per_page:(pg+1)*per_page]
        cards = []
        for _, r in sub.iterrows():
            cards.append(f'''
  <div class="card"><img src="seds/sed_{int(r['id'])}.png" loading="lazy">
  <div class="meta">{int(r['id'])} &nbsp; T={r['T_bb']:.0f}K &nbsp; χ²ν={r['chi2_red']:.1f}</div></div>''')
        def lk(p):
            href='index.html' if p==0 else f'index_{p:02d}.html'
            return f'<b>{p+1}</b>' if p==pg else f'<a href="{href}">{p+1}</a>'
        win = sorted({0,1,n_pages-2,n_pages-1} | set(range(max(0,pg-3),min(n_pages,pg+4))))
        win = [p for p in win if 0<=p<n_pages]
        parts=[]; prev=None
        if pg>0: parts.append(f'<a href="{"index.html" if pg-1==0 else f"index_{pg-1:02d}.html"}">&larr;prev</a>')
        for p in win:
            if prev is not None and p!=prev+1: parts.append('…')
            parts.append(lk(p)); prev=p
        if pg<n_pages-1: parts.append(f'<a href="index_{pg+1:02d}.html">next&rarr;</a>')
        nav=' '.join(parts)
        html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>SED v02 multi-telescope — page {pg+1}/{n_pages}</title>
<style>body{{font-family:sans-serif;margin:16px;background:#fafafa;}}
h1{{font-size:18px;}} .summary{{background:#fff;border:1px solid #ddd;padding:8px 12px;
font-family:monospace;white-space:pre;font-size:11px;margin-bottom:12px;}}
.nav{{margin:10px 0;}} .nav a{{margin:0 4px;}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}}
.card{{background:#fff;border:1px solid #ddd;border-radius:5px;padding:6px;}}
.card img{{width:100%;}} .meta{{font-family:monospace;font-size:11px;margin-top:4px;}}</style></head><body>
<h1>SED v02 — multi-telescope offset vs HST+JWST blackbody
<span style="font-weight:normal;font-size:13px;color:#666;">
{len(res_all):,} stars; page {pg+1}/{n_pages}</span></h1>
<div class="summary">{off}</div>
<div class="nav">{nav}</div><div class="grid">{''.join(cards)}</div><div class="nav">{nav}</div>
</body></html>'''
        (HTML/('index.html' if pg==0 else f'index_{pg:02d}.html')).write_text(html)
    print(f'Webform: {n_pages} pages → htmls/sed_v02/index.html')


if __name__ == '__main__':
    main(validate_only='--validate' in sys.argv)
