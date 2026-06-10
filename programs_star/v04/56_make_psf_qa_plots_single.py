#!/usr/bin/env python
"""
56_make_psf_qa_plots.py — generate the 6 canonical PSF QA plots for a
processed (instrument, tile, band) using the artefacts that 54_ wrote.

Reads from
  WORK / <instrument> / <tile> / psf /
    pass1_<band>.fits         (detection catalog, FITS_LDAC w/ VIGNET)
    sci_<band>.fits           (single-HDU SCI image)
    stars_<band>.fits         (filtered LDAC PSF-star catalog)
    samp_stars_<band>.fits    (PSFEx SAMPLES check-image)
    resi_stars_<band>.fits    (PSFEx RESIDUALS check-image)

Re-runs PSFEx on stars_<band>.fits with OUTCAT_NAME so per-star
CHI2_PSF/FLAGS_PSF land in <psf>/outcat_<band>.fits.

Writes (canonical filenames, picked up by 55_make_psf_qa_site.py):
  WORK / <instrument> / <tile> / psf /
    mag_vs_halflight.png
    mag_vs_chi2.png
    saturation_peak.png
    psf_samples.png
    psf_residuals.png
    hist_n_means.png
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter, MultipleLocator
from astropy.io import fits
from astropy.visualization import simple_norm
from scipy.spatial import cKDTree

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS = PROJECT / 'configs'
WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
# Single-HDU instruments (Euclid VIS, HST ACS/WFC F814W).  PIX and the
# PSFEx config are selected per-instrument in main() and exposed as module
# globals so the plot/outcat helpers can read them.
INSTRUMENTS = {
    'euclid_vis':    {'pix': 0.10,  'psfex': 'psfex_euclid_vis.psfex', 'label': 'Euclid VIS'},
    'hst_acs_f814w': {'pix': 0.030, 'psfex': 'psfex_hst_acs.psfex',    'label': 'HST ACS F814W'},
    # Euclid NISP (Y/J/H) — MER mosaics resampled to 0.10"/px (native 0.30");
    # built by 54_..._euclid.py --band; uses the NISP-tuned PSFEx config.
    'euclid_nisp_y': {'pix': 0.10,  'psfex': 'psfex_euclid_nisp.psfex', 'label': 'Euclid NISP Y'},
    'euclid_nisp_j': {'pix': 0.10,  'psfex': 'psfex_euclid_nisp.psfex', 'label': 'Euclid NISP J'},
    'euclid_nisp_h': {'pix': 0.10,  'psfex': 'psfex_euclid_nisp.psfex', 'label': 'Euclid NISP H'},
    # HSC SSP s23b deepCoadd (LSST calexp), 0.168"/px; HST-ACS PSFEx config.
    'hsc_g': {'pix': 0.168, 'psfex': 'psfex_hsc.psfex', 'label': 'HSC g'},
    'hsc_r': {'pix': 0.168, 'psfex': 'psfex_hsc.psfex', 'label': 'HSC r'},
    'hsc_i': {'pix': 0.168, 'psfex': 'psfex_hsc.psfex', 'label': 'HSC i'},
    'hsc_z': {'pix': 0.168, 'psfex': 'psfex_hsc.psfex', 'label': 'HSC z'},
    'hsc_y': {'pix': 0.168, 'psfex': 'psfex_hsc.psfex', 'label': 'HSC y'},
}
PIX        = 0.10      # arcsec/pixel — overwritten per-instrument in main()
_PSFEX_CFG = 'psfex_euclid_vis.psfex'  # overwritten per-instrument in main()

rcParams['font.family']  = 'serif'
rcParams['font.serif']   = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', default='euclid_vis',
                   choices=list(INSTRUMENTS))
    p.add_argument('--tile', required=True,
                   help='tile id (Euclid 9-digit number, or COSMOS-Web '
                        'letter-number such as A4)')
    p.add_argument('--reuse-outcat', action='store_true',
                   help='Skip re-running PSFEx if outcat already exists')
    return p.parse_args()


def ensure_outcat(psf_dir: Path, band: str, reuse: bool) -> Path:
    """Re-run PSFEx to produce per-star OUTCAT in psf_dir.

    Uses the SAME per-band SAMPLE_FWHMRANGE that 54_ used (read from the
    meta JSON) so the OUTCAT acceptance matches the actual PSF model —
    otherwise the config default range would reject bright stars here
    that 54_ accepted (the FWHM-range bug).
    """
    import json, tempfile
    outcat = psf_dir / f'outcat_{band}.fits'
    if reuse and outcat.exists():
        print(f'  reuse OUTCAT: {outcat.name}')
        return outcat
    cmd = [
        'psfex', str(psf_dir / f'stars_{band}.fits'),
        '-c', str(CONFIGS / _PSFEX_CFG),
        '-OUTCAT_TYPE', 'FITS_LDAC',
        '-OUTCAT_NAME', str(outcat),
        '-CHECKIMAGE_TYPE', 'NONE',  # check images already exist from 54_
        # this re-run is only for per-star FLAGS_PSF; send its .psf to a temp
        # dir so it cannot clobber 54_'s pass-2 (bright-anchored) PSF model.
        '-PSF_DIR', tempfile.gettempdir(),
    ]
    # match 54_'s per-band FWHM range from the meta JSON
    meta_f = psf_dir / f'psf_{band}.meta.json'
    if meta_f.exists():
        m = json.loads(meta_f.read_text())
        fr = m.get('sample_fwhmrange')
        if fr:
            cmd += ['-SAMPLE_FWHMRANGE', f'{fr[0]:.2f},{fr[1]:.2f}']
        sm = m.get('snr_min')        # HSC's compressed SNR floor (else 100 rejects all)
        if sm:
            cmd += ['-SAMPLE_MINSN', f'{float(sm):.1f}']
    print(f'  PSFEx → {outcat.name}', flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr)
        sys.exit('PSFEx OUTCAT pass failed')
    return outcat


def read_outcat(outcat_path: Path):
    h = fits.open(outcat_path)
    for hdu in h:
        if hdu.data is not None and hasattr(hdu, 'columns') and len(hdu.columns) > 2:
            return hdu.data
    sys.exit(f'OUTCAT not readable: {outcat_path}')


def compute_locus_and_cut(fr, cs, snr, flg, elon, fwhm):
    """Iterative MAD-clipped stellar locus + 3·MAD artifact cut.

    Adds a FWHM lower-bound filter to the base sample to exclude
    single-pixel cosmic ray / hot-pixel artifacts.  The threshold is
    derived from the high-SNR subset (stars dominate at SNR>500),
    so it adapts to each band's PSF FWHM automatically.

    Without this filter, LW bands (F277W: 6000 CR artifacts at FR≈0.8
    px) produce a contaminated base where the iterative MAD doesn't
    converge tightly, giving cuts at negative half-light radius.
    """
    base = (cs > 0.8) & (snr > 100) & (flg < 2) & (elon < 1.5) & (fr > 0)
    if base.sum() < 20:
        return np.median(fr[fr > 0]), 0.0, 0.0, 0.0
    # Estimate PSF FWHM from the high-SNR subset (artifacts are usually
    # lower SNR; the median FWHM of bright sources gives a robust PSF estimate)
    hi = base & (snr > 1000)
    if hi.sum() < 5:
        hi = base & (snr > 500)
    psf_fwhm_est = np.median(fwhm[hi]) if hi.sum() >= 3 else np.median(fwhm[base])
    # Require FWHM > 0.5 × PSF FWHM (real stars are at PSF FWHM; CR/hot
    # pixels are sub-pixel).
    fwhm_min = 0.5 * psf_fwhm_est
    base = base & (fwhm > fwhm_min)
    if base.sum() < 20:
        return np.median(fr[fr > 0]), 0.0, 0.0, 0.0
    med = np.median(fr[base])
    mad = 1.4826 * np.median(np.abs(fr[base] - med))
    for _ in range(20):
        sel = np.abs(fr[base] - med) < 3.0 * mad
        nmed = np.median(fr[base][sel])
        nmad = 1.4826 * np.median(np.abs(fr[base][sel] - nmed))
        if abs(nmed - med) < 1e-4 and abs(nmad - mad) < 1e-4:
            med, mad = nmed, nmad; break
        med, mad = nmed, nmad
    cut = med - 3.0 * mad
    std0 = np.std(fr[base])
    return med, mad, cut, std0


def compute_peak_and_ncoremask(sci, xx, yy, half=2):
    """Per-source peak in 5×5 core box + masked-pixel count."""
    ny, nx = sci.shape
    peak = np.zeros(len(xx)); nm = np.zeros(len(xx), int)
    for k in range(len(xx)):
        xi = int(round(xx[k])); yi = int(round(yy[k]))
        if half <= xi < nx-half and half <= yi < ny-half:
            box = sci[yi-half:yi+half+1, xi-half:xi+half+1]
            peak[k] = np.nanmax(box)
            nm[k] = int(np.sum(box == 0))
    return peak, nm


def neighbour_counts(xx, yy, tree, radii_arcsec):
    """For each source, count detections within each radius (excluding self)."""
    out = {}
    for r in radii_arcsec:
        r_px = r / PIX
        out[r] = np.array([
            len(tree.query_ball_point([xx[k], yy[k]], r_px)) - 1
            for k in range(len(xx))
        ])
    return out


# ──────────────────────────────────────────────────────────────────────────
# plots
# ──────────────────────────────────────────────────────────────────────────

def plot_mag_vs_halflight(out_png, mag, fr, fwhm, ell, ncoremask, locus_px,
                          mad_px, cut_px, std_px, band_upper,
                          cs=None, snr=None, elon=None, flg=None, n1=None,
                          xx=None, yy=None, accepted_xy=None,
                          upper_bright=None, bright_pivot=None,
                          psf_fwhm_px=0.0, sat_onset_mag=None, snr_min=100.0):
    """Render mag-vs-half-light-radius with the SAME PSF-star definition
    that 54_ uses, so the red dots match the actual ~hundreds of PSF
    candidates (not the ~30k point-source-like detections in the field)."""
    good = np.isfinite(mag) & (mag > 13) & (mag < 32) & np.isfinite(ell)
    saturated = ncoremask > 0
    edge = (flg & 8) > 0 if flg is not None else np.zeros_like(good, bool)
    contaminated = ((n1 >= 2) & (flg < 2)) if (n1 is not None and flg is not None) \
                   else np.zeros_like(good, bool)
    art = good & (fr > 0) & ((fr < cut_px) | (fwhm <= 0)) & (~saturated)
    # Apply the full 54_ selection so this plot's red dots match the
    # actual PSF-candidate count (~317 for F115W A4), not 30k.
    psf_star = good & (fr > cut_px) & (fr < locus_px + 2.5*std_px) & (fwhm > 0) \
               & (~saturated) & (~edge) & (~contaminated)
    if cs is not None: psf_star &= (cs > 0.8)
    if snr is not None: psf_star &= (snr > snr_min)
    if elon is not None: psf_star &= (elon < 1.5)

    # Split into PSFEx-accepted (solid) vs PSFEx-rejected (open) using the
    # accepted star coordinates passed in from main().
    psf_accepted = np.zeros_like(psf_star)
    if accepted_xy is not None and xx is not None and yy is not None and len(accepted_xy):
        tree_acc = cKDTree(np.asarray(accepted_xy))
        d, _ = tree_acc.query(np.column_stack([xx, yy]), k=1)
        psf_accepted = psf_star & (d < 1.0)   # within 1 px = same source
    psf_rejected = psf_star & (~psf_accepted)

    # ── Bright-end shape override ──
    # PSFEx χ²-flags the highest-S/N stars because they out-resolve the
    # empirical PSF model (a few-% floor, irreducible for the resampled NISP
    # MER), NOT because they are bad.  Reinstate a bright (mag < bright_pivot)
    # PSFEx-rejected candidate as an accepted PSF anchor when its shape is
    # INDEPENDENTLY stellar: FLUX_RADIUS on the (bright) locus, FWHM consistent
    # with the PSF, and low ellipticity.  Faint rejections are left untouched.
    override = np.zeros_like(psf_star)
    if bright_pivot is not None and psf_fwhm_px > 0:
        ub = upper_bright if upper_bright is not None else (locus_px + 3.0*mad_px)
        override = (psf_rejected & (mag < bright_pivot)
                    & (fr > cut_px) & (fr < ub)
                    & (np.abs(fwhm - psf_fwhm_px) < 0.15 * psf_fwhm_px)
                    & (ell < 0.10))
        psf_accepted = psf_accepted | override
        psf_rejected = psf_star & (~psf_accepted)
    pure_acc = psf_accepted & (~override)

    fig, ax = plt.subplots(figsize=(11, 8.5))
    gd = good & (fr > 0)
    sc = ax.scatter(mag[gd], fr[gd]*PIX, s=10, c=ell[gd], cmap='rainbow_r',
                    vmin=0, vmax=0.8, alpha=0.6, rasterized=True, linewidths=0)
    ax.scatter(mag[pure_acc], fr[pure_acc]*PIX, s=28, c='red',
               edgecolor='k', lw=0.4, zorder=7,
               label=f'PSFEx accepted ({pure_acc.sum()})')
    if override.sum():
        ax.scatter(mag[override], fr[override]*PIX, s=48, c='red',
                   edgecolor='darkorange', lw=1.7, zorder=8,
                   label=f'Bright, shape-validated override ({override.sum()})')
    ax.scatter(mag[psf_rejected], fr[psf_rejected]*PIX, s=40, facecolor='none',
               edgecolor='red', lw=1.2, alpha=0.7, zorder=6,
               label=f'Candidate, PSFEx-rejected ({psf_rejected.sum()})')
    ax.scatter(mag[saturated & good], fr[saturated & good]*PIX, s=55,
               marker='x', c='black', linewidths=0.8, zorder=7,
               label=f'Saturated/masked-core ({(saturated & good).sum()})')
    ax.scatter(mag[art], fr[art]*PIX, s=70, facecolor='none', edgecolor='blue',
               lw=1.3, zorder=5, label=f'Artifacts ({art.sum()})')
    ax.axhline(locus_px*PIX, color='k', ls='--', lw=1.3, alpha=0.6,
               label=f'Stellar locus = {locus_px*PIX:.3f}″ ({locus_px:.2f} px)')
    ax.axhline(cut_px*PIX, color='blue', ls='-', lw=2.0, alpha=0.9,
               label=f'Artifact cut = locus−3·MAD = {cut_px*PIX:.3f}″ ({cut_px:.2f} px)')
    if sat_onset_mag is not None:
        ax.axvline(sat_onset_mag, color='purple', ls=':', lw=1.8, alpha=0.85,
                   label=f'Saturation onset = {sat_onset_mag:.2f} mag (bright limit)')
    ax.set_xlabel(f'MAG_AUTO ({band_upper}, AB)', fontsize=15, family='serif')
    ax.set_ylabel('Half-light radius FLUX_RADIUS [arcsec]', fontsize=15, family='serif')
    # ground-based data (HSC, 0.168"/px) can't resolve below ~0.3" half-light,
    # so floor the y-axis there; space missions keep 0.
    _ymin = 0.35 if PIX >= 0.15 else 0.0
    ax.set_ylim(_ymin, 0.55); ax.set_xlim(13.5, 30.5)
    ax.set_title(f'{band_upper} — Magnitude vs Half-Light Radius',
                 fontsize=13, family='serif')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.tick_params(which='both', direction='in', top=False, right=False,
                   labelsize=12, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0]/PIX, ax.get_ylim()[1]/PIX)
    ax2.set_ylabel('FLUX_RADIUS [pixel] (1 px = 30 mas)', fontsize=15, family='serif')
    ax2.tick_params(which='both', direction='in', labelsize=12, length=6)
    ax2.tick_params(which='minor', length=3); ax2.minorticks_on()
    fig.canvas.draw()
    bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
    right_edge_frac = bbox.x1 / fig.get_window_extent().width
    gap_frac = 1.0 / (11 * 25.4)
    cax = fig.add_axes([right_edge_frac+gap_frac, 0.14, 0.022, 0.79])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label('Ellipticity (0=round→elongated)', fontsize=12, family='serif')
    cb.ax.tick_params(direction='in')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_mag_vs_chi2(out_png, mag_acc, chi2_acc, n1_acc, n3_acc, flg_acc, band_upper):
    if np.asarray(mag_acc).size == 0:
        print(f'  [skip] {out_png.name}: no accepted stars'); return
    rainbow = ['#1f4eea','#5eb6ff','#00b300','#90ee90','#ff7f0e','#d62728']
    dcmap = ListedColormap(rainbow)
    bounds = [-0.5,0.5,1.5,2.5,3.5,4.5,5.5]
    dnorm = BoundaryNorm(bounds, dcmap.N)
    def dcol(v):
        vc = np.clip(v, 0, 5)
        return dcmap(vc.astype(int) if isinstance(vc, np.ndarray) else int(vc))
    bad = (n1_acc >= 2) & (flg_acc < 2)
    good = ~bad
    mean_post = chi2_acc[good].mean() if good.any() else float('nan')

    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.scatter(mag_acc, chi2_acc, s=320, marker='o', facecolor='none',
               edgecolor=dcol(n3_acc), linewidths=1.8, zorder=3)
    sc1 = ax.scatter(mag_acc[good], chi2_acc[good], s=55,
                     c=np.clip(n1_acc[good], 0, 5), cmap=dcmap, norm=dnorm,
                     edgecolor='k', linewidths=0.5, zorder=5)
    ax.scatter(mag_acc[bad], chi2_acc[bad], s=140, marker='o',
               facecolor='red', edgecolor='black', linewidths=1.2, zorder=6)
    ax.axhline(mean_post, color='red', ls='--', lw=1.8, alpha=0.8)
    ax.axhline(1.0, color='grey', ls=':', lw=1.0, alpha=0.6)
    ax.set_xlabel(f'MAG_AUTO ({band_upper}, AB)', fontsize=15, family='serif')
    ax.set_ylabel('CHI2_PSF (per-star reduced χ²)', fontsize=15, family='serif')
    ax.set_title(f'PSF-Fit Quality vs Brightness; Neighbour Count in 6 Discrete Bins ({band_upper}).',
                 fontsize=13, family='serif')
    ax.set_xlim(np.floor(mag_acc.min())-0.5, np.ceil(mag_acc.max())+0.5)
    # adaptive floor: ground data (HSC, WEIGHT NONE) has CHI2_PSF well below 1,
    # which the fixed 0.5 floor hid — drop the floor to the data when needed.
    _clo = float(np.nanmin(chi2_acc)) if np.asarray(chi2_acc).size else 0.5
    ax.set_ylim(min(0.5, max(_clo*0.7, 0.005)), max(np.nanmax(chi2_acc)*1.3, 2))
    ax.set_yscale('log')
    ax.grid(alpha=0.25, which='both')
    yticks = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{t}' for t in yticks]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=12, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()
    labels = ['0', '1', '2', '3', '4', '≥5']
    swatch = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                     markeredgecolor='k', markersize=10, label=f'  {l}')
              for l, c in zip(labels, rainbow)]
    extra = [
        Line2D([0],[0], color='none',
               label=r'$\bf{Filled\ dot}$ = $n_1$ (≤ 1″)'),
        Line2D([0],[0], color='none',
               label=r'$\bf{Outer\ ring}$ = $n_3$ (≤ 3″)'),
        Line2D([0],[0], color='none', label=' '),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='k', markersize=13,
               label=r'$\bf{Excluded}$: $n_1\geq2$ AND FLAGS<2'),
        Line2D([0],[0], color='none', label=f'  ({bad.sum()} contaminated)'),
        Line2D([0],[0], color='red', ls='--', lw=1.8,
               label=f'Mean χ² post-cut = {mean_post:.3f}'),
        Line2D([0],[0], color='grey', ls=':', lw=1.0, label='χ²=1.0 (ideal)'),
    ]
    ax.legend(handles=extra + swatch, loc='upper right', fontsize=10,
              framealpha=0.95, handlelength=1.6, labelspacing=0.45)
    cb = fig.colorbar(sc1, ax=ax, fraction=0.04, pad=0.015,
                      ticks=[0,1,2,3,4,5], extend='max')
    cb.set_label('Neighbour count (0,1,2,3,4,≥5)', fontsize=12, family='serif')
    cb.ax.set_yticklabels(['0','1','2','3','4','≥5'])
    cb.ax.tick_params(direction='in')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_saturation_peak(out_png, mag, peak, cs, snr, ncoremask, band_upper,
                         sat_peak_level=None, sat_onset_mag=None,
                         psf_star_mask=None):
    ok = np.isfinite(mag) & (mag > 13) & (mag < 30)
    # bright/high-SNR point sources — these trace the (saturated) bright
    # sequence and are EXCLUDED from the PSF model; do not confuse with the
    # PSF model stars (red, below).  Kept only as context for the onset.
    is_point = (cs > 0.8) & (snr > 100)
    sat = ncoremask > 0
    # the actual PSF-model star sample (matched back to the detection catalog
    # by the caller); these are what should be highlighted in red.
    if psf_star_mask is None:
        psf_star_mask = np.zeros(len(mag), bool)
    is_psf = psf_star_mask & ok
    has_pk = (sat_peak_level is not None) and np.isfinite(sat_peak_level)
    has_on = (sat_onset_mag is not None) and np.isfinite(sat_onset_mag)

    def draw_sat_lines(ax, label=True):
        # horizontal = peak saturation level; vertical = magnitude onset.
        if has_pk:
            ax.axhline(sat_peak_level, color='magenta', ls=':', lw=1.8, alpha=0.95,
                       zorder=6,
                       label=(f'Saturation peak = {sat_peak_level:.3g}' if label else None))
        if has_on:
            ax.axvline(sat_onset_mag, color='magenta', ls=':', lw=1.8, alpha=0.95,
                       zorder=6,
                       label=(f'Saturation onset = {sat_onset_mag:.2f} mag' if label else None))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.scatter(mag[ok], peak[ok], s=4, c='lightgray', alpha=0.4,
               label='All', rasterized=True)
    ax.scatter(mag[ok & is_point], peak[ok & is_point], s=8, c='darkorange',
               alpha=0.6,
               label=f'Bright/saturated, excluded ({(ok & is_point).sum()})',
               rasterized=True)
    ax.scatter(mag[is_psf], peak[is_psf], s=14, c='red', alpha=0.85, zorder=7,
               label=f'PSF model stars ({is_psf.sum()})', rasterized=True)
    draw_sat_lines(ax)
    ax.set_yscale('log')
    ax.set_xlabel(f'MAG_AUTO ({band_upper}, AB)', fontsize=13, family='serif')
    ax.set_ylabel('Peak Count FLUX_MAX [image units]', fontsize=13, family='serif')
    ax.set_title('Peak Counts vs Magnitude — PSF stars (red) sit faintward of saturation onset',
                 fontsize=11, family='serif')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.set_xlim(13.5, 30.5)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=11, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()

    ax = axes[1]
    # zoom that spans BOTH the saturated bright sequence and the PSF stars so
    # the user can see the red PSF stars relative to the onset line.
    brt = ok & (mag < 25.5)
    ax.scatter(mag[brt], peak[brt], s=5, c='lightgray', alpha=0.4,
               label='All', rasterized=True)
    ax.scatter(mag[brt & is_point], peak[brt & is_point], s=12, c='darkorange',
               alpha=0.7, label='Bright/saturated, excluded', rasterized=True)
    ax.scatter(mag[brt & sat], peak[brt & sat], s=44, marker='o',
               facecolor='none', edgecolor='blue', linewidths=1.5,
               label='Masked core (>0 px)', zorder=5, rasterized=True)
    ax.scatter(mag[is_psf & brt], peak[is_psf & brt], s=16, c='red', alpha=0.85,
               zorder=7, label='PSF model stars', rasterized=True)
    draw_sat_lines(ax)
    ax.set_yscale('log')
    ax.set_xlabel(f'MAG_AUTO ({band_upper}, AB)', fontsize=13, family='serif')
    ax.set_ylabel('Peak Count FLUX_MAX [image units]', fontsize=13, family='serif')
    ax.set_title('Zoom — saturated bright sequence (orange) vs PSF stars (red)',
                 fontsize=11, family='serif')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(13.5, 25.5)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(True, which='major', alpha=0.4)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=11, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_psf_mosaics(out_samp_png, out_resi_png, samp, resi, ocd,
                     in_mosaic_idx, det_cat, band_upper,
                     n_accepted_total=None, crop_cell=None):
    """psf_samples.png + psf_residuals.png — uses the same vmin/vmax (sqrt)
    derived from the samples mosaic.  All cells labelled, FLAGS_PSF≠0
    cells get a red frame; high-χ² accepted cells get an orange frame."""
    mos = ocd[in_mosaic_idx]
    N = len(mos)
    # Auto-detect cell size from the mosaic shape.  PSFEx mosaic cell size
    # depends on PSF_SIZE / sampling and the band's VIGNET:
    #   SW (F115W, F150W) → 101
    #   LW (F277W, F444W) → 151
    #   Euclid VIS        → 51
    # Try each candidate and use the one that divides BOTH axes cleanly.
    cs = None
    for candidate in (151, 101, 51):
        if samp.shape[0] % candidate == 0 and samp.shape[1] % candidate == 0:
            cs = candidate; break
    if cs is None:
        sys.exit(f'Cannot detect cell size from mosaic shape {samp.shape}')
    nrow = samp.shape[0] // cs
    ncol = samp.shape[1] // cs
    print(f'  mosaic cell size detected: {cs}×{cs}, grid {nrow}×{ncol}', flush=True)
    # Ground data: crop each cell to the small PSF-model region so distant
    # background galaxies (in the wide VIGNET but outside the model) don't show.
    if crop_cell and 8 < int(crop_cell) < cs:
        cw = int(crop_cell) | 1; off = (cs - cw) // 2
        def _crop(m):
            o = np.zeros((nrow*cw, ncol*cw), m.dtype)
            for rr in range(nrow):
                for cc in range(ncol):
                    o[rr*cw:(rr+1)*cw, cc*cw:(cc+1)*cw] = \
                        m[rr*cs+off:rr*cs+off+cw, cc*cs+off:cc*cs+off+cw]
            return o
        samp = _crop(samp); resi = _crop(resi); cs = cw
        print(f'  cropped cells to {cs}px (PSF-model region)', flush=True)
    ax_ = np.asarray(mos['X_IMAGE']); ay = np.asarray(mos['Y_IMAGE'])
    chi2_psf = np.asarray(mos['CHI2_PSF']); flgpsf = np.asarray(mos['FLAGS_PSF'])

    cx = np.asarray(det_cat['X_IMAGE']); cy = np.asarray(det_cat['Y_IMAGE'])
    mag_full = np.asarray(det_cat['MAG_AUTO'], float)
    flg_full = np.asarray(det_cat['FLAGS'], int)
    mag = np.zeros(N); flg_se = np.zeros(N, int)
    for k in range(N):
        j = ((cx-ax_[k])**2 + (cy-ay[k])**2).argmin()
        mag[k] = mag_full[j]; flg_se[k] = flg_full[j]

    tree = cKDTree(np.column_stack([cx, cy]))
    def nc(x, y, r): return len(tree.query_ball_point([x, y], r/PIX)) - 1
    n1 = np.array([nc(ax_[k], ay[k], 1.0) for k in range(N)])
    n3 = np.array([nc(ax_[k], ay[k], 3.0) for k in range(N)])
    n5 = np.array([nc(ax_[k], ay[k], 5.0) for k in range(N)])

    accepted = flgpsf == 0
    chi2_acc = chi2_psf[accepted]
    mean_c = chi2_acc.mean() if accepted.any() else 1.0
    std_c  = chi2_acc.std()  if accepted.any() else 0.0
    THRESH = mean_c + 2.0 * std_c

    VMIN = 0.0
    VMAX = np.percentile(samp[np.isfinite(samp)], 99.75)
    norm = simple_norm(samp, 'sqrt', vmin=VMIN, vmax=VMAX, clip=True)
    fs_num = 13.5; fs_lab = 11.0

    nrow_render = nrow
    samp_padded = samp
    resi_padded = resi

    def render(data, title, fname):
        # SQUARE cells guaranteed: aspect='equal' forces 1 data-unit-x ==
        # 1 data-unit-y, so each 1×1 cell renders square regardless of
        # nrow/ncol.  Figure size is matched to the data aspect
        # (width/ncol == height/nrow) so there's minimal whitespace, and
        # bbox_inches='tight' trims the rest.
        cell_in = 22.0 / ncol               # inches per cell column
        fig_w = 22.0
        fig_h = cell_in * nrow_render + 1.5  # +1.5" for title/axis labels
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(data, origin='upper', cmap='gray', norm=norm, aspect='equal',
                  extent=[0, ncol, nrow_render, 0])
        for c in range(ncol+1): ax.axvline(c, color='cyan', lw=0.5, alpha=0.5)
        for r in range(nrow_render+1): ax.axhline(r, color='cyan', lw=0.5, alpha=0.5)
        for k in range(N):
            r = k // ncol; c = k % ncol
            if r >= nrow_render: continue
            is_rej = flgpsf[k] != 0
            high_chi2 = (not is_rej) and chi2_psf[k] > THRESH
            if is_rej:
                ax.add_patch(Rectangle((c+0.015, r+0.015), 0.97, 0.97, fill=False,
                                       edgecolor='red', lw=2.5, zorder=5))
            if high_chi2:
                ax.add_patch(Rectangle((c+0.015, r+0.015), 0.97, 0.97, fill=False,
                                       edgecolor='orange', lw=2.5, zorder=5))
            col = 'red' if is_rej else ('orange' if high_chi2 else 'white')
            cell_lbl = f'{k+1}R' if is_rej else f'{k+1}'
            ax.text(c+0.04, r+0.07, cell_lbl, ha='left', va='top', color=col,
                    fontsize=fs_num, fontweight='bold', family='serif', zorder=6)
            ax.text(c+0.96, r+0.07, f'n={int(n1[k])}/{int(n3[k])}/{int(n5[k])}',
                    ha='right', va='top', color=col, fontsize=fs_lab,
                    family='serif', zorder=6)
            ax.text(c+0.96, r+0.20, f'F={flg_se[k]}', ha='right', va='top',
                    color=col, fontsize=fs_lab, family='serif', zorder=6)
            ax.text(c+0.04, r+0.93, f'm={mag[k]:.1f}', ha='left', va='bottom',
                    color=col, fontsize=fs_lab, family='serif', zorder=6)
            ax.text(c+0.96, r+0.93, f'χ²={chi2_psf[k]:.1f}',
                    ha='right', va='bottom', color=col, fontsize=fs_lab,
                    family='serif', zorder=6)
            # For PSFEx-rejected stars, write rejection reason at bottom-center
            # of the panel.  Short single-word code, red, bold so it stands out.
            if is_rej:
                v = int(flgpsf[k])
                if   v & 1:  rej_word = 'neighbor'
                elif v & 4:  rej_word = 'residual'
                elif v & 8:  rej_word = 'elong'
                elif v & 16: rej_word = 'low-snr'
                elif v & 32: rej_word = 'saturated'
                else:        rej_word = f'F={v}'
                ax.text(c+0.50, r+0.80, rej_word, ha='center', va='bottom',
                        color='red', fontsize=fs_lab, fontweight='bold',
                        family='serif', zorder=6)
        ax.set_xticks(np.arange(ncol)+0.5); ax.set_xticklabels(range(1, ncol+1),
            fontsize=11, family='serif')
        ax.set_yticks(np.arange(nrow_render)+0.5)
        ax.set_yticklabels(range(1, nrow_render+1), fontsize=11, family='serif')
        ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
        ax.set_xlabel('Column (1=left → 20)', fontsize=14, family='serif')
        ax.set_ylabel(f'Row (1=top → {nrow_render})', fontsize=14, family='serif')
        ax.set_title(title, fontsize=13, family='serif')
        # v19's exact save call:
        plt.tight_layout()
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)

    n_acc_in_mos = accepted.sum(); n_rej = (~accepted).sum()
    # Total accepted comes from the full OUTCAT; the mosaic only fits the
    # first N_cells, so n_acc_in_mos may be smaller than n_accepted_total.
    if n_accepted_total is None:
        n_accepted_total = n_acc_in_mos
    n_acc_outside_mos = n_accepted_total - n_acc_in_mos
    n_orange = ((chi2_psf > THRESH) & accepted).sum()
    extra_note = (f' (+{n_acc_outside_mos} more accepted but outside mosaic capacity)'
                  if n_acc_outside_mos > 0 else '')
    render(samp_padded,
           f'PSFEx {band_upper} — total PSFEx-accepted = {n_accepted_total}; '
           f'mosaic shows {n_acc_in_mos} accepted + {n_rej} rejected (red){extra_note}.  '
           f'sqrt vmin=0, vmax={VMAX:.3f}.  Orange = χ² > mean+2σ '
           f'= {THRESH:.2f} ({n_orange}/{n_acc_in_mos}).',
           out_samp_png)
    render(resi_padded,
           f'PSFEx {band_upper} — residuals (data−model).  Same scaling as samples '
           f'(vmin=0, vmax={VMAX:.3f}, sqrt).  '
           f'Total accepted = {n_accepted_total}; mosaic shows {n_acc_in_mos}.  '
           f'Red frame = PSFEx-rejected ({n_rej});  Orange = high χ² ({n_orange}).',
           out_resi_png)


def plot_neighbour_scatter(out_png, xx, yy, mag, psf_mask, tree, band_upper,
                           max_arcsec=5.0, iso_radius_arcsec=None,
                           seeing_fwhm_arcsec=None, nbr_fwhm=None):
    """Ground-based neighbour-contamination map.

    For every SELECTED PSF model star, every detected source within
    max_arcsec becomes one point:
        x = separation from the PSF star centre (arcsec, 0–max_arcsec)
        y = Δmag = (neighbour MAG_AUTO − PSF-star MAG_AUTO)
                   >0 → neighbour is fainter (harmless);  <0 → brighter
        colour = the PSF star's own magnitude, rainbow gradient
                 red (bright) → blue (faint)
    This replaces the neighbour-count histogram for ground data: it shows at a
    glance whether the surviving neighbours are faint/harmless and confirms the
    isolation radius is swept clear of any detected source."""
    idx = np.where(np.asarray(psf_mask))[0]
    if idx.size == 0:
        print(f'  [skip] {out_png.name}: no selected PSF stars'); return
    R = max_arcsec / PIX
    rr, dm, pm = [], [], []
    for k in idx:
        pmag = mag[k]
        if not np.isfinite(pmag):
            continue
        for j in tree.query_ball_point([xx[k], yy[k]], R):
            if j == k:
                continue
            mj = mag[j]
            if not np.isfinite(mj):
                continue
            sep = float(np.hypot(xx[j] - xx[k], yy[j] - yy[k])) * PIX
            rr.append(sep); dm.append(mj - pmag); pm.append(pmag)
    if not rr:
        print(f'  [skip] {out_png.name}: no neighbours within {max_arcsec}"'); return
    rr = np.asarray(rr); dm = np.asarray(dm); pm = np.asarray(pm)

    fig, ax = plt.subplots(figsize=(11, 7))
    # rainbow_r: vmin(bright)→red, vmax(faint)→violet/blue.
    sc = ax.scatter(rr, dm, c=pm, cmap='rainbow_r', s=26, alpha=0.85,
                    edgecolor='k', linewidths=0.25, zorder=3,
                    vmin=np.nanmin(pm), vmax=np.nanmax(pm), rasterized=True)
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label('PSF star magnitude  (red = bright → blue = faint)',
                 fontsize=12, family='serif')
    ax.axhline(0.0, color='0.35', ls=':', lw=1.6,
               label='Δmag = 0 (neighbour as bright as PSF star)')
    if seeing_fwhm_arcsec:
        ax.axvline(seeing_fwhm_arcsec, color='green', ls=':', lw=1.8, alpha=0.9,
                   label=f'seeing FWHM = {seeing_fwhm_arcsec:.2f}″')
    if iso_radius_arcsec:
        # derive the seeing for the label if it was not passed explicitly
        fw = seeing_fwhm_arcsec
        if fw is None and nbr_fwhm:
            fw = iso_radius_arcsec / nbr_fwhm
        nf = nbr_fwhm if nbr_fwhm else (iso_radius_arcsec / fw if fw else 0)
        lbl = (f'isolation cut = {iso_radius_arcsec:.2f}″ = {nf:.1f} × FWHM ({fw:.2f}″)'
               if fw else f'isolation cut = {iso_radius_arcsec:.2f}″')
        ax.axvline(iso_radius_arcsec, color='red', ls='--', lw=1.8, alpha=0.85,
                   label=lbl)
    ax.set_xlim(0, max_arcsec)
    ax.set_ylim(-3.5, 8.5)          # fixed Δmag range, uniform across all bands
    ax.set_xlabel('Separation from PSF star centre (arcsec)',
                  fontsize=14, family='serif')
    ax.set_ylabel('Δmag  (neighbour − PSF star)   ·   fainter ↑',
                  fontsize=14, family='serif')
    ax.set_title(f'Neighbour Contamination Map — {band_upper}   '
                 f'({idx.size} PSF stars, {rr.size} neighbours < {max_arcsec:.0f}″)',
                 fontsize=12.5, family='serif')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=12, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_hist_n(out_png, n_arr, mask_psf, band_upper, iso_radius_arcsec=None):
    """Neighbour-count distributions for the ACTUAL selected PSF model stars.

    Uses the real PSFEx-accepted star sample (not a stale snr>100 candidate
    re-derivation), so on HSC's compressed SNR it still renders.  Because the
    isolation cut rejects any star with a neighbour inside iso_radius_arcsec,
    the n_r histograms for radii < iso_radius should pile up at 0 — that is the
    visual proof the cut worked."""
    mask_psf = np.asarray(mask_psf)
    if int(mask_psf.sum()) == 0:
        print(f'  [skip] {out_png.name}: no selected PSF stars'); return
    N = int(mask_psf.sum())
    means = {r: n_arr[r][mask_psf].mean() for r in [1,2,3,4,5]}
    n1c, n3c, n5c = n_arr[1][mask_psf], n_arr[3][mask_psf], n_arr[5][mask_psf]
    fig, ax = plt.subplots(figsize=(11, 7))
    bins = np.arange(0, max(int(n5c.max()) if n5c.size else 0, 30)+1) - 0.5
    ax.hist(n1c, bins=bins, color='red',   alpha=0.55,
            label=f'$n_1$ (1″)  N={N}', edgecolor='darkred', lw=0.8)
    ax.hist(n3c, bins=bins, color='green', alpha=0.45,
            label=f'$n_3$ (3″)  N={N}', edgecolor='darkgreen', lw=0.8)
    ax.hist(n5c, bins=bins, color='blue',  alpha=0.35,
            label=f'$n_5$ (5″)  N={N}', edgecolor='darkblue', lw=0.8)
    ax.axvline(means[1], color='darkred',    ls=':', lw=2.0, label=f'mean $n_1$ = {means[1]:.2f}')
    ax.axvline(means[2], color='orange',     ls=':', lw=2.0, label=f'mean $n_2$ = {means[2]:.2f}')
    ax.axvline(means[3], color='darkgreen',  ls=':', lw=2.0, label=f'mean $n_3$ = {means[3]:.2f}')
    ax.axvline(means[4], color='teal',       ls=':', lw=2.0, label=f'mean $n_4$ = {means[4]:.2f}')
    ax.axvline(means[5], color='darkblue',   ls=':', lw=2.0, label=f'mean $n_5$ = {means[5]:.2f}')
    iso_txt = (f'isolation cut: no neighbour < {iso_radius_arcsec:.2f}″'
               if iso_radius_arcsec else 'isolation-cut sample')
    ax.axvline(0.0, color='black', ls='--', lw=1.5, alpha=0.6, label=iso_txt)
    ax.set_xlabel('Neighbour count', fontsize=15, family='serif')
    ax.set_ylabel('Number of selected PSF model stars', fontsize=15, family='serif')
    ax.set_title(f'Neighbour-Count Distributions for Selected PSF Model Stars ({band_upper}).  '
                 f'N={N}.', fontsize=13, family='serif')
    ax.set_xlim(-0.7, 30)        # uniform x-max=30 across HST/JWST/Euclid
    ax.legend(loc='upper right', fontsize=10, framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=12, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    global PIX, _PSFEX_CFG
    cfg = INSTRUMENTS[args.instrument]
    PIX = cfg['pix']
    _PSFEX_CFG = cfg['psfex']
    band = args.tile                       # filename suffix key (= tile id)
    band_upper = f"{cfg['label']} {args.tile}"
    psf_dir = WORK / args.instrument / args.tile / 'psf'
    if not psf_dir.exists():
        sys.exit(f'No 54_ output dir: {psf_dir}')
    print(f'  band={band}  tile={args.tile}  psf_dir={psf_dir}')

    # 1. ensure OUTCAT
    outcat = ensure_outcat(psf_dir, band, args.reuse_outcat)
    ocd = read_outcat(outcat)

    # 2. read pass-1 detection catalog
    pass1 = fits.open(psf_dir / f'pass1_{band}.fits')[2].data
    mag = np.asarray(pass1['MAG_AUTO'], float)
    fr  = np.asarray(pass1['FLUX_RADIUS'], float)
    fwhm = np.asarray(pass1['FWHM_IMAGE'], float)
    cs  = np.asarray(pass1['CLASS_STAR'], float)
    snr = np.asarray(pass1['SNR_WIN'], float)
    flg = np.asarray(pass1['FLAGS'], int)
    elon = np.asarray(pass1['ELONGATION'], float)
    ell  = np.asarray(pass1['ELLIPTICITY'], float)
    xx  = np.asarray(pass1['X_IMAGE'], float)
    yy  = np.asarray(pass1['Y_IMAGE'], float)
    print(f'  pass-1 detections: {len(pass1)}')

    # 3. SCI for peak + ncoremask — from the source recorded by 54_ (the
    #    original drz/MER mosaic); no sci_ copy is written anymore.
    import json as _json
    _m = _json.loads((psf_dir / f'psf_{band}.meta.json').read_text())
    _scf = _m.get('sci_source_file'); _sce = int(_m.get('sci_source_ext', 0))
    sci = (fits.getdata(_scf, ext=_sce) if _scf
           else fits.getdata(psf_dir / f'sci_{band}.fits')).astype(np.float32)
    print(f'  SCI shape: {sci.shape}')
    peak, ncoremask = compute_peak_and_ncoremask(sci, xx, yy, half=2)
    print(f'  masked-core sources: {(ncoremask>0).sum()}')

    # 4. stellar-locus + artifact-cut — READ FROM THE META (54_'s authoritative
    #    values, e.g. Gaia-anchored for NISP) so the plotted lines MATCH the
    #    build's selection.  Do NOT recompute here (that re-introduced the old
    #    CLASS_STAR locus, which disagreed with the Gaia-anchored build).
    locus = float(_m['stellar_locus_px'])
    mad   = float(_m.get('locus_mad_sigma_px', 0.0))
    cut   = float(_m['artifact_cut_px'])
    upper = float(_m.get('upper_px', locus + 2.5 * float(_m.get('locus_std_px', 0.0))))
    std0  = (upper - locus) / 2.5      # so the plot's (locus + 2.5*std0) == upper
    # bright-end shape-override params (NISP builds record these; absent → no override)
    upper_bright = float(_m.get('upper_bright_px', upper))
    bright_pivot = _m.get('bright_pivot_mag')      # None for non-NISP builds
    psf_fwhm_est = float(_m.get('psf_fwhm_est_px', _m.get('psf_fwhm_px', 0.0)))
    sat_onset    = _m.get('sat_onset_mag')
    snr_min_meta = float(_m.get('snr_min', 100.0))   # HSC uses a lower floor
    print(f'  (from meta) locus={locus:.3f} px = {locus*PIX:.3f}"  cut={cut:.3f} px  '
          f'upper={upper:.3f} px  anchor={_m.get("locus_anchor","?")}'
          + (f'  bright_pivot={bright_pivot} upper_bright={upper_bright:.3f}'
             if bright_pivot is not None else ''))

    # 5. neighbour counts
    tree = cKDTree(np.column_stack([xx, yy]))
    n_arr = neighbour_counts(xx, yy, tree, [1, 2, 3, 4, 5])

    # 6. mosaic mapping (OUTCAT entries that pass input filter)
    in_mosaic_mask = (ocd['FLAGS_PSF'] & 32) == 0
    samp = fits.getdata(psf_dir / f'samp_stars_{band}.fits').astype(float)
    resi = fits.getdata(psf_dir / f'resi_stars_{band}.fits').astype(float)
    # Auto-detect the mosaic cell size (HST/SW=101, LW=151, Euclid VIS=51)
    # the SAME way plot_psf_mosaics does — a hardcoded 101 would under-count
    # the 51 px Euclid cells (truncating in_mosaic_idx to a fraction of the
    # accepted stars) and over-count the 151 px LW cells.
    csize = None
    for candidate in (151, 101, 51):
        if samp.shape[0] % candidate == 0 and samp.shape[1] % candidate == 0:
            csize = candidate; break
    if csize is None:
        sys.exit(f'Cannot detect mosaic cell size from shape {samp.shape}')
    nrow = samp.shape[0]//csize
    ncol = samp.shape[1]//csize
    n_filled = sum(np.any(samp[r*csize:(r+1)*csize, c*csize:(c+1)*csize] != 0)
                   for r in range(nrow) for c in range(ncol))
    in_mosaic_idx = np.where(in_mosaic_mask)[0][:n_filled]
    print(f'  mosaic: {nrow}×{ncol} = {nrow*ncol},  filled={n_filled}, '
          f'accepted={(ocd["FLAGS_PSF"][in_mosaic_idx]==0).sum()}')

    # 7. accepted-only mag/n1/n3 for mag-vs-chi2 plot
    mos = ocd[in_mosaic_idx]
    ax_ = np.asarray(mos['X_IMAGE']); ay = np.asarray(mos['Y_IMAGE'])
    acc_mask = mos['FLAGS_PSF'] == 0
    cx = np.asarray(pass1['X_IMAGE']); cy = np.asarray(pass1['Y_IMAGE'])
    mag_acc = np.zeros(len(mos)); flg_acc = np.zeros(len(mos), int)
    for k in range(len(mos)):
        j = ((cx-ax_[k])**2 + (cy-ay[k])**2).argmin()
        mag_acc[k] = mag[j]; flg_acc[k] = flg[j]
    n1_acc = np.array([len(tree.query_ball_point([ax_[k], ay[k]], 1.0/PIX))-1
                       for k in range(len(mos))])
    n3_acc = np.array([len(tree.query_ball_point([ax_[k], ay[k]], 3.0/PIX))-1
                       for k in range(len(mos))])
    chi2_acc_arr = np.asarray(mos['CHI2_PSF'])

    # restrict to ACCEPTED stars for the mag-chi2 plot
    sel = acc_mask
    print(f'  accepted stars on mag-vs-chi2: {sel.sum()}')

    # 8. write the 6 plots
    print(f'  → plots:')
    # also compute n_1 for ALL detections so the plot can apply the
    # 54_ contamination cut (n_1≥2 AND FLAGS<2)
    n1_all = np.array([len(tree.query_ball_point([cx[k], cy[k]], 1.0/PIX))-1
                       for k in range(len(pass1))])
    # accepted-star XY for the candidate vs accepted split — use ALL OUTCAT
    # entries with FLAGS_PSF==0 (the mosaic only holds the first ~180, so
    # using just in-mosaic acc would under-count by ~18 stars on F115W A4).
    accepted_all_mask = ocd['FLAGS_PSF'] == 0
    accepted_xy = np.column_stack([
        np.asarray(ocd['X_IMAGE'])[accepted_all_mask],
        np.asarray(ocd['Y_IMAGE'])[accepted_all_mask],
    ])
    print(f'  total PSFEx-accepted (full OUTCAT): {accepted_all_mask.sum()}')
    plot_mag_vs_halflight(psf_dir / 'mag_vs_halflight.png',
                          mag, fr, fwhm, ell, ncoremask, locus, mad, cut, std0,
                          band_upper,
                          cs=cs, snr=snr, elon=elon, flg=flg, n1=n1_all,
                          xx=cx, yy=cy, accepted_xy=accepted_xy,
                          upper_bright=upper_bright, bright_pivot=bright_pivot,
                          psf_fwhm_px=psf_fwhm_est, sat_onset_mag=sat_onset,
                          snr_min=snr_min_meta)
    print(f'     mag_vs_halflight.png')

    # mag-vs-chi2 should plot ALL PSFEx-accepted stars (not just in-mosaic);
    # recompute arrays over the full OUTCAT to match mag-vs-halflight's count.
    acc_full_x = np.asarray(ocd['X_IMAGE'])[accepted_all_mask]
    acc_full_y = np.asarray(ocd['Y_IMAGE'])[accepted_all_mask]
    chi2_full  = np.asarray(ocd['CHI2_PSF'])[accepted_all_mask]
    mag_full2 = np.zeros(len(acc_full_x)); flg_full2 = np.zeros(len(acc_full_x), int)
    psf_match_idx = np.empty(len(acc_full_x), int)
    for k in range(len(acc_full_x)):
        j = ((cx-acc_full_x[k])**2 + (cy-acc_full_y[k])**2).argmin()
        mag_full2[k] = mag[j]; flg_full2[k] = flg[j]
        psf_match_idx[k] = j
    # boolean mask over the detection catalog marking the PSF model stars, so
    # the saturation plot can overplot the ACTUAL selected sample in red.
    psf_star_mask = np.zeros(len(mag), bool)
    psf_star_mask[psf_match_idx] = True
    n1_full = np.array([len(tree.query_ball_point([acc_full_x[k], acc_full_y[k]], 1.0/PIX))-1
                        for k in range(len(acc_full_x))])
    n3_full = np.array([len(tree.query_ball_point([acc_full_x[k], acc_full_y[k]], 3.0/PIX))-1
                        for k in range(len(acc_full_x))])
    plot_mag_vs_chi2(psf_dir / 'mag_vs_chi2.png',
                     mag_full2, chi2_full, n1_full, n3_full, flg_full2,
                     band_upper)
    print(f'     mag_vs_chi2.png  (n={len(acc_full_x)} accepted)')

    # saturation level from the builder's meta JSON (single source of truth)
    import json as _json
    _meta_f = psf_dir / f'psf_{band}.meta.json'
    _spk, _smag = None, None
    if _meta_f.exists():
        _m = _json.loads(_meta_f.read_text())
        _spk, _smag = _m.get('sat_peak_level'), _m.get('sat_onset_mag')
    plot_saturation_peak(psf_dir / 'saturation_peak.png',
                         mag, peak, cs, snr, ncoremask, band_upper,
                         sat_peak_level=_spk, sat_onset_mag=_smag,
                         psf_star_mask=psf_star_mask)
    print(f'     saturation_peak.png  (sat_peak={_spk}, onset_mag={_smag})')

    # total accepted comes from the FULL OUTCAT (matches mag_vs_halflight),
    # so the mosaic title can report consistent numbers.
    n_accepted_total = int(accepted_all_mask.sum())
    plot_psf_mosaics(psf_dir / 'psf_samples.png',
                     psf_dir / 'psf_residuals.png',
                     samp, resi, ocd, in_mosaic_idx, pass1, band_upper,
                     n_accepted_total=n_accepted_total,
                     crop_cell=_m.get('psf_size_px'))   # ground: show only the model region
    print(f'     psf_samples.png / psf_residuals.png  '
          f'(total accepted={n_accepted_total})')

    # neighbour diagnostic of the ACTUAL selected PSF model stars (same
    # psf_star_mask used by the saturation plot).  GROUND data (HSC, large
    # pixels, crowded COSMOS) gets the Δmag-vs-separation contamination map;
    # SPACE data keeps the neighbour-count histogram.
    if PIX >= 0.15:
        _fwhm_px = _m.get('fwhm_px')
        _seeing  = (_fwhm_px * PIX) if _fwhm_px else None
        plot_neighbour_scatter(psf_dir / 'neighbour_scatter.png',
                               xx, yy, mag, psf_star_mask, tree, band_upper,
                               max_arcsec=5.0,
                               iso_radius_arcsec=_m.get('nbr_radius_arcsec'),
                               seeing_fwhm_arcsec=_seeing,
                               nbr_fwhm=_m.get('nbr_fwhm'))
        print(f'     neighbour_scatter.png')
    else:
        plot_hist_n(psf_dir / 'hist_n_means.png',
                    {r: n_arr[r] for r in [1,2,3,4,5]},
                    psf_star_mask, band_upper,
                    iso_radius_arcsec=_m.get('nbr_radius_arcsec'))
        print(f'     hist_n_means.png')
    print(f'=== {band_upper} A4 QA plots done ===')


if __name__ == '__main__':
    main()
