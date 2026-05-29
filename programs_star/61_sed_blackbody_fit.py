#!/usr/bin/env python
"""
61_sed_blackbody_fit.py — first-order blackbody SED fit of the clean star
sample, using only the trusted HST+JWST bands, to test Euclid photometric
offsets.

Method (Suzuki & Fukugita 2018 style, first order):
  - Fit a blackbody B_nu(T) to each star's 5 HST+JWST AB magnitudes
    (F814W, F115W, F150W, F277W, F444W).  Free parameters: T and a flux
    normalization.  For a fixed T the normalization is an additive mag
    offset solved analytically (inverse-variance weighted), so the fit is
    a 1-D grid scan over T.
  - From the best-fit blackbody, synthesize predicted AB magnitudes in the
    4 Euclid bands (VIS, NIR Y/J/H) through their SVO filter curves.
  - Delta = observed_Euclid - predicted_Euclid  → photometric offset test.

Synthetic AB photometry (photon-counting CCD/HgCdTe detectors):
  <f_nu> = ∫ f_nu(λ) R(λ) λ dλ / ∫ R(λ) λ dλ
  m_AB   = -2.5 log10(<f_nu>) - 48.6

Filters fetched from the SVO Filter Profile Service and cached locally.

Inputs:
  csvfiles_star/sed_clean_sample_v01.parquet   (6,007 stars, all 9 bands clean)
Outputs:
  csvfiles_star/sed_blackbody_v01.parquet      (fit params + Euclid offsets)
  htmls/sed_v01/euclid_offset_summary.png/.txt (aggregate offsets)
  htmls/sed_v01/seds/sed_<primary_id>.png      (per-star SED plots)
  htmls/sed_v01/index.html                     (paginated webform)
"""
from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'sed_v01'
SEDDIR = HTML / 'seds'
FILTER_CACHE = OUT / 'sed_filters'

# physical constants (cgs)
H = 6.62607015e-27      # erg s
C = 2.99792458e10       # cm/s
K = 1.380649e-16        # erg/K
C_AA = 2.99792458e18    # Angstrom/s

# Filters: name → (SVO id, fit?, mission_tag, magcol, errcol, errkind)
FILTERS = {
    'F814W': ('HST/ACS_WFC.F814W', True,  'hst',  'cat_mag_F814W_hst',  'cat_magerr_F814W_hst', 'magerr'),
    'F115W': ('JWST/NIRCam.F115W',  True,  'jwst', 'cat_mag_F115W_jwst', 'cat_snr_F115W_jwst',   'snr'),
    'F150W': ('JWST/NIRCam.F150W',  True,  'jwst', 'cat_mag_F150W_jwst', 'cat_snr_F150W_jwst',   'snr'),
    'F277W': ('JWST/NIRCam.F277W',  True,  'jwst', 'cat_mag_F277W_jwst', 'cat_snr_F277W_jwst',   'snr'),
    'F444W': ('JWST/NIRCam.F444W',  True,  'jwst', 'cat_mag_F444W_jwst', 'cat_snr_F444W_jwst',   'snr'),
    'VIS':   ('Euclid/VIS.vis',     False, 'vis',  'cat_mag_VIS_vis',    'cat_magerr_VIS_vis',   'magerr'),
    'NIR_Y': ('Euclid/NISP.Y',      False, 'nisp', 'cat_mag_NIR_Y_nisp', 'cat_magerr_NIR_Y_nisp','magerr'),
    'NIR_J': ('Euclid/NISP.J',      False, 'nisp', 'cat_mag_NIR_J_nisp', 'cat_magerr_NIR_J_nisp','magerr'),
    'NIR_H': ('Euclid/NISP.H',      False, 'nisp', 'cat_mag_NIR_H_nisp', 'cat_magerr_NIR_H_nisp','magerr'),
}
FIT_BANDS  = [b for b, v in FILTERS.items() if v[1]]
EUCLID_BANDS = [b for b, v in FILTERS.items() if not v[1]]

# T grid for the scan (K) — log-spaced, covers M dwarf to hot star
T_GRID = np.geomspace(2000.0, 50000.0, 400)


def load_filters() -> dict:
    """Fetch (and cache) SVO transmission curves for all 9 bands.
    Returns name → dict(wl[Angstrom], tr, pivot_AA)."""
    FILTER_CACHE.mkdir(parents=True, exist_ok=True)
    from astroquery.svo_fps import SvoFps
    filt = {}
    for name, (svo_id, *_rest) in FILTERS.items():
        cache = FILTER_CACHE / f'{name}.npz'
        if cache.exists():
            d = np.load(cache)
            wl, tr = d['wl'], d['tr']
        else:
            print(f'[filters] fetch {svo_id} ...')
            t = SvoFps.get_transmission_data(svo_id)
            wl = np.asarray(t['Wavelength'], dtype=float)   # Angstrom
            tr = np.asarray(t['Transmission'], dtype=float)
            np.savez(cache, wl=wl, tr=tr)
        # normalize transmission peak (shape only)
        good = tr > 0
        wl, tr = wl[good], tr[good]
        pivot = np.sqrt(np.trapezoid(tr, wl) / np.trapezoid(tr / wl**2, wl))
        filt[name] = {'wl': wl, 'tr': tr, 'pivot_AA': pivot}
    return filt


def bb_fnu(wl_AA: np.ndarray, T: float) -> np.ndarray:
    """Blackbody f_nu (arbitrary scale) at wavelengths wl_AA for temperature T."""
    nu = C_AA / wl_AA            # Hz
    x = H * nu / (K * T)
    # B_nu ∝ nu^3 / (exp(x) - 1); avoid overflow
    with np.errstate(over='ignore'):
        bnu = (nu**3) / np.expm1(x)
    return bnu


def synth_abmag_shape(filt: dict, T: float) -> dict:
    """Synthetic AB mag (arbitrary zero point) of a blackbody of temp T,
    per filter.  Photon-counting: <f_nu> = ∫ f_nu R λ dλ / ∫ R λ dλ."""
    out = {}
    for name, fd in filt.items():
        wl, tr = fd['wl'], fd['tr']
        fnu = bb_fnu(wl, T)
        num = np.trapezoid(fnu * tr * wl, wl)
        den = np.trapezoid(tr * wl, wl)
        mean_fnu = num / den
        out[name] = -2.5 * np.log10(mean_fnu)   # arbitrary ZP, fixed offset cancels in fit
    return out


def precompute_shape_grid(filt: dict) -> dict:
    """Precompute synthetic shape mags for every T in T_GRID per filter."""
    grid = {name: np.zeros(len(T_GRID)) for name in filt}
    for i, T in enumerate(T_GRID):
        s = synth_abmag_shape(filt, T)
        for name in filt:
            grid[name][i] = s[name]
    return grid


def fit_blackbody(mags, errs, shape_grid):
    """Fit T + normalization to the FIT_BANDS.
    mags, errs: dict band→value (AB mag, mag error) for FIT_BANDS.
    Returns (T_best, dm_best, chi2_best, dof).
    A 0.03 mag systematic floor is added in quadrature to the (raw) errors
    here, for weighting only — the plotted error bars use the raw errors."""
    w = np.array([1.0 / (errs[b]**2 + 0.03**2) for b in FIT_BANDS])
    mobs = np.array([mags[b] for b in FIT_BANDS])
    chi2 = np.full(len(T_GRID), np.inf)
    dm_arr = np.zeros(len(T_GRID))
    shape_mat = np.array([shape_grid[b] for b in FIT_BANDS])  # (nband, nT)
    for i in range(len(T_GRID)):
        mshape = shape_mat[:, i]
        # optimal additive offset dm = weighted mean of (mobs - mshape)
        dm = np.sum(w * (mobs - mshape)) / np.sum(w)
        resid = mobs - mshape - dm
        chi2[i] = np.sum(w * resid**2)
        dm_arr[i] = dm
    ibest = np.argmin(chi2)
    dof = len(FIT_BANDS) - 2
    return T_GRID[ibest], dm_arr[ibest], chi2[ibest], dof


def get_mag_err(row, band):
    """Return (mag, magerr) for a band from a catalog row."""
    _, _, tag, magcol, errcol, errkind = FILTERS[band]
    mag = row[magcol]
    if errkind == 'magerr':
        err = row[errcol]
    else:
        snr = row[errcol]
        err = 1.0857 / snr if snr and snr > 0 else np.nan
    # Return the RAW photometric error (for display / honest error bars).
    # A small floor (0.005) only guards against div-by-zero.  The 0.03 mag
    # systematic floor used to weight the fit is applied separately in
    # fit_blackbody so it doesn't inflate the plotted error bars.
    if np.isfinite(err):
        err = max(err, 0.005)
    return mag, err


def main(validate_only=False):
    t0 = time.time()
    HTML.mkdir(parents=True, exist_ok=True)
    SEDDIR.mkdir(parents=True, exist_ok=True)

    print('Loading filters ...')
    filt = load_filters()
    print('Precomputing blackbody shape grid ...')
    shape_grid = precompute_shape_grid(filt)

    print('Loading clean SED sample ...')
    df = pd.read_parquet(OUT / 'sed_clean_sample_v01.parquet').reset_index(drop=True)
    print(f'  {len(df):,} stars')

    if validate_only:
        # pick a bright, mid-temperature star for validation
        cand = df[df['gaia_phot_g_mean_mag'].notna()].copy()
        cand = cand.sort_values('cat_mag_F814W_hst').head(50)
        df = cand.head(1)
        print(f'[validate] star primary_id={df.iloc[0]["primary_id"]}')

    # Fit all
    results = []
    n = len(df)
    for ridx in range(n):
        row = df.iloc[ridx]
        mags, errs = {}, {}
        ok = True
        for b in FILTERS:
            m, e = get_mag_err(row, b)
            if not np.isfinite(m) or not np.isfinite(e):
                ok = False
            mags[b] = m; errs[b] = e
        if not ok:
            continue
        T, dm, chi2, dof = fit_blackbody(mags, errs, shape_grid)
        # predicted Euclid mags = shape(T_best) + dm   (interpolate shape at T)
        rec = {'primary_id': row['primary_id'], 'T_bb': T, 'dm': dm,
               'chi2': chi2, 'dof': dof, 'chi2_red': chi2 / max(dof, 1)}
        # interpolate shape grid at T for all bands
        for b in FILTERS:
            shp = np.interp(T, T_GRID, shape_grid[b])
            pred = shp + dm
            rec[f'mag_obs_{b}'] = mags[b]
            rec[f'magerr_{b}']  = errs[b]
            rec[f'mag_pred_{b}'] = pred
            if b in EUCLID_BANDS:
                rec[f'delta_{b}'] = mags[b] - pred   # observed - predicted
        results.append(rec)

    res = pd.DataFrame(results)
    print(f'Fit {len(res):,} stars in {time.time()-t0:.1f}s')

    if validate_only:
        r = res.iloc[0]
        print(f'\n[validate] T_bb = {r["T_bb"]:.0f} K, chi2_red = {r["chi2_red"]:.2f}')
        for b in EUCLID_BANDS:
            print(f'   {b}: obs={r[f"mag_obs_{b}"]:.3f}  pred={r[f"mag_pred_{b}"]:.3f}  '
                  f'Δ={r[f"delta_{b}"]:+.3f}')
        make_sed_plot(res.iloc[0], filt, shape_grid, SEDDIR / f'sed_{r["primary_id"]}.png')
        print(f'[validate] SED plot → seds/sed_{r["primary_id"]}.png')
        return

    # Save fit results
    res.to_parquet(OUT / 'sed_blackbody_v01.parquet', index=False)
    res.to_csv(OUT / 'sed_blackbody_v01.csv', index=False)
    print(f'Wrote sed_blackbody_v01.parquet ({len(res):,} rows)')

    # Aggregate Euclid offset summary
    summarize_offsets(res)

    # Mass production: per-star SED PNGs for ALL stars, sorted by brightness,
    # generated in parallel across cores.
    res_sorted = res.sort_values('mag_obs_F814W').reset_index(drop=True)
    recs = res_sorted.to_dict('records')
    n_workers = 12
    chunks = [recs[i::n_workers] for i in range(n_workers)]
    print(f'Generating {len(recs):,} SED PNGs across {n_workers} workers ...')
    from concurrent.futures import ProcessPoolExecutor
    t_plot = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for k in pool.map(_plot_worker, [(c, str(SEDDIR)) for c in chunks]):
            done += k
    print(f'  {done:,} PNGs in {time.time()-t_plot:.1f}s')
    build_webform(res_sorted, res)

    print(f'\nWall time: {time.time()-t0:.1f}s')


def _plot_worker(args):
    """Worker: render SED PNGs for a chunk of records."""
    records, seddir = args
    filt = load_filters()                  # cached → fast
    shape_grid = precompute_shape_grid(filt)
    sd = Path(seddir)
    for rec in records:
        make_sed_plot(rec, filt, shape_grid, sd / f'sed_{rec["primary_id"]}.png')
    return len(records)


def build_webform(res_plotted, res_all, per_page=60):
    """Paginated HTML grid of SED PNGs (orphan_star/sn_search style)."""
    n = len(res_plotted)
    n_pages = (n + per_page - 1) // per_page
    # aggregate offset line for the header
    off_txt = (HTML / 'euclid_offset_summary.txt').read_text() if (HTML / 'euclid_offset_summary.txt').exists() else ''
    for pg in range(n_pages):
        sub = res_plotted.iloc[pg*per_page:(pg+1)*per_page]
        cards = []
        for _, r in sub.iterrows():
            pid = r['primary_id']
            deltas = '  '.join(f'Δ{b}={r[f"delta_{b}"]:+.2f}' for b in EUCLID_BANDS)
            cards.append(f'''
  <div class="card">
    <img src="seds/sed_{pid}.png" loading="lazy">
    <div class="meta">{pid} &nbsp; T={r['T_bb']:.0f}K &nbsp; χ²ν={r['chi2_red']:.1f}<br>{deltas}</div>
  </div>''')
        # compact nav: first, prev, a window around current, next, last
        def pg_link(p):
            href = 'index.html' if p == 0 else f'index_{p:02d}.html'
            return f'<b>{p+1}</b>' if p == pg else f'<a href="{href}">{p+1}</a>'
        win = set(range(max(0, pg-3), min(n_pages, pg+4)))
        win |= {0, 1, n_pages-2, n_pages-1}
        win = sorted(p for p in win if 0 <= p < n_pages)
        parts = []
        if pg > 0:
            parts.append(f'<a href="{"index.html" if pg-1==0 else f"index_{pg-1:02d}.html"}">&larr;prev</a>')
        prev_p = None
        for p in win:
            if prev_p is not None and p != prev_p + 1:
                parts.append('…')
            parts.append(pg_link(p))
            prev_p = p
        if pg < n_pages - 1:
            parts.append(f'<a href="index_{pg+1:02d}.html">next&rarr;</a>')
        nav = ' '.join(parts)
        html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>SED blackbody fits — page {pg+1}/{n_pages}</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background:#fafafa; }}
h1 {{ font-size: 19px; }}
.summary {{ background:#fff; border:1px solid #ddd; padding:8px 12px;
            font-family:monospace; white-space:pre; font-size:12px; margin-bottom:12px; }}
.nav {{ margin: 10px 0; font-size: 15px; }} .nav a {{ margin:0 4px; }}
.grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }}
.card {{ background:#fff; border:1px solid #ddd; border-radius:5px; padding:6px; }}
.card img {{ width:100%; }}
.meta {{ font-family:monospace; font-size:11px; color:#333; margin-top:4px; }}
</style></head><body>
<h1>Blackbody SED fits (HST+JWST) — Euclid offset test &nbsp;
    <span style="font-weight:normal;font-size:13px;color:#666;">
    {len(res_all):,} stars fit; showing brightest {n} &mdash; page {pg+1}/{n_pages}</span></h1>
<div class="summary">{off_txt}</div>
<div class="nav">{nav}</div>
<div class="grid">{''.join(cards)}</div>
<div class="nav">{nav}</div>
</body></html>'''
        fname = 'index.html' if pg == 0 else f'index_{pg:02d}.html'
        (HTML / fname).write_text(html)
    print(f'Webform: {n_pages} pages → htmls/sed_v01/index.html')


def _filter_colors(filt):
    """Rainbow color per filter by log-pivot-wavelength (blue=short, red=long)."""
    from matplotlib.lines import Line2D
    logp = {b: np.log10(filt[b]['pivot_AA']) for b in FILTERS}
    lmin, lmax = min(logp.values()), max(logp.values())
    cmap = plt.cm.rainbow
    return {b: cmap((logp[b] - lmin) / (lmax - lmin)) for b in FILTERS}


def make_sed_plot(rec, filt, shape_grid, path):
    """Per-star SED: blackbody curve + per-filter rainbow-coloured points.
    Marker encodes data type: circle = HST/JWST (fit), square = Euclid
    (observed), x = Euclid (predicted).  Colour encodes filter (rainbow,
    blue=short λ → red=long λ).  Legend is per filter."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.5, 5))
    T = rec['T_bb']; dm = rec['dm']
    fcolor = _filter_colors(filt)

    # model curve over 0.5–5 um, anchored via the F150W synthetic mag
    wl_um = np.geomspace(0.5, 5.0, 400)
    wl_AA = wl_um * 1e4
    mmod = -2.5 * np.log10(bb_fnu(wl_AA, T)) + dm
    anchor_b = 'F150W'
    shp_anchor = np.interp(T, T_GRID, shape_grid[anchor_b]) + dm
    mod_at_anchor = -2.5*np.log10(bb_fnu(np.array([filt[anchor_b]['pivot_AA']]), T))[0] + dm
    curve_shift = shp_anchor - mod_at_anchor
    ax.plot(wl_um, mmod + curve_shift, '-', color='0.6', lw=1.1, zorder=1)

    # marker per data type; error bars use the RAW photometric error,
    # drawn dark and capped so they're visible behind the coloured markers.
    ebar = dict(ecolor='0.25', elinewidth=1.2, capsize=4, capthick=1.2)
    for b in FIT_BANDS:
        piv = filt[b]['pivot_AA'] / 1e4
        ax.errorbar(piv, rec[f'mag_obs_{b}'], yerr=rec[f'magerr_{b}'],
                    fmt='o', color=fcolor[b], ms=8, mec='0.2', mew=0.5,
                    zorder=3, **ebar)
    for b in EUCLID_BANDS:
        piv = filt[b]['pivot_AA'] / 1e4
        ax.errorbar(piv, rec[f'mag_obs_{b}'], yerr=rec[f'magerr_{b}'],
                    fmt='s', color=fcolor[b], ms=8, mec='0.2', mew=0.5,
                    zorder=3, **ebar)
        ax.plot(piv, rec[f'mag_pred_{b}'], 'x', color=fcolor[b], ms=10, mew=2.2,
                zorder=4)

    # ── Per-filter legend (sorted by wavelength), marker = data type ──
    MISSION = {'hst': 'HST', 'jwst': 'JWST', 'vis': 'Euclid', 'nisp': 'Euclid'}
    order = sorted(FILTERS, key=lambda b: filt[b]['pivot_AA'])
    handles = []
    for b in order:
        mk = 's' if b in EUCLID_BANDS else 'o'
        mission = MISSION[FILTERS[b][2]]   # per-filter mission name
        handles.append(Line2D([0], [0], marker=mk, color='w',
                              markerfacecolor=fcolor[b], markeredgecolor='0.2',
                              markersize=9,
                              label=f'{b}  ({filt[b]["pivot_AA"]/1e4:.2f} μm, {mission})'))
    # symbol-meaning entries
    handles.append(Line2D([0], [0], marker='x', color='k', linestyle='none',
                          markersize=9, mew=2, label='× = Euclid predicted'))
    handles.append(Line2D([0], [0], color='0.6', lw=1.1,
                          label=f'blackbody T={T:.0f} K'))
    ax.legend(handles=handles, fontsize=7.5, loc='best', framealpha=0.9)

    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('wavelength (μm)')
    ax.set_ylabel('AB magnitude')
    ax.set_title(f'{rec["primary_id"]}  —  T={T:.0f} K, χ²ν={rec["chi2_red"]:.2f}')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarize_offsets(res):
    """Aggregate Euclid offset stats + plot."""
    lines = ['Euclid photometric offset test (observed - blackbody-predicted)',
             'Blackbody fit to HST F814W + JWST F115/F150/F277/F444W only.', '']
    lines.append(f'{"band":<7} {"N":>7} {"median Δ":>10} {"NMAD":>8} {"mean Δ":>9} {"std":>8}')
    lines.append('-' * 55)
    fig, axes = plt.subplots(1, len(EUCLID_BANDS), figsize=(4*len(EUCLID_BANDS), 4), sharey=True)
    for ax, b in zip(axes, EUCLID_BANDS):
        d = res[f'delta_{b}'].dropna().values
        med = np.median(d); nmad = 1.4826*np.median(np.abs(d-med))
        lines.append(f'{b:<7} {len(d):>7,} {med:>+10.4f} {nmad:>8.4f} '
                     f'{np.mean(d):>+9.4f} {np.std(d):>8.4f}')
        ax.hist(d, bins=np.linspace(-0.5, 0.5, 80), color='C0', alpha=0.8)
        ax.axvline(0, color='k', lw=0.8)
        ax.axvline(med, color='C3', lw=1.5, label=f'median={med:+.3f}')
        ax.set_xlabel(f'Δ{b} (obs−pred) mag')
        ax.set_title(f'{b}  (NMAD={nmad:.3f})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('N stars')
    fig.suptitle('Euclid offset vs HST+JWST blackbody prediction', fontsize=12)
    fig.tight_layout()
    fig.savefig(HTML / 'euclid_offset_summary.png', dpi=130)
    plt.close(fig)
    text = '\n'.join(lines)
    print('\n' + text)
    (HTML / 'euclid_offset_summary.txt').write_text(text + '\n')


if __name__ == '__main__':
    main(validate_only='--validate' in sys.argv)
