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
PIX     = 0.10  # arcsec/pixel (Euclid VIS)        # arcsec / pixel (COSMOS-Web 30 mas mosaics)

rcParams['font.family']  = 'serif'
rcParams['font.serif']   = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--instrument', default='euclid_vis')
    p.add_argument('--tile', required=True,
                   help='Euclid tile id (the 9-digit number)')
    p.add_argument('--reuse-outcat', action='store_true')
    return p.parse_args()


def ensure_outcat(psf_dir: Path, band: str, reuse: bool) -> Path:
    """Re-run PSFEx to produce per-star OUTCAT in psf_dir."""
    outcat = psf_dir / f'outcat_{band}.fits'
    if reuse and outcat.exists():
        print(f'  reuse OUTCAT: {outcat.name}')
        return outcat
    cmd = [
        'psfex', str(psf_dir / f'stars_{band}.fits'),
        '-c', str(CONFIGS / 'psfex_euclid_vis.psfex'),
        '-OUTCAT_TYPE', 'FITS_LDAC',
        '-OUTCAT_NAME', str(outcat),
        '-CHECKIMAGE_TYPE', 'NONE',  # check images already exist from 54_
    ]
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
    """Iterative MAD-clipped stellar locus + 3·MAD artifact cut."""
    base = (cs > 0.8) & (snr > 100) & (flg < 2) & (elon < 1.5) & (fr > 0)
    if base.sum() < 20:
        return np.median(fr[fr > 0]), 0.0, 0.0
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
                          mad_px, cut_px, std_px, band_upper):
    good = np.isfinite(mag) & (mag > 13) & (mag < 32) & np.isfinite(ell)
    saturated = ncoremask > 0
    edge_or_bad = ~good
    art = good & (fr > 0) & ((fr < cut_px) | (fwhm <= 0)) & (~saturated)
    psf_star = (good & (fr > cut_px) & (fr < locus_px + 2.5*std_px) & (fwhm > 0) & (~saturated))

    fig, ax = plt.subplots(figsize=(11, 8.5))
    gd = good & (fr > 0)
    sc = ax.scatter(mag[gd], fr[gd]*PIX, s=10, c=ell[gd], cmap='rainbow_r',
                    vmin=0, vmax=0.8, alpha=0.6, rasterized=True, linewidths=0)
    ax.scatter(mag[psf_star], fr[psf_star]*PIX, s=24, c='red',
               edgecolor='k', lw=0.4, zorder=6, label=f'PSF stars ({psf_star.sum()})')
    ax.scatter(mag[saturated & good], fr[saturated & good]*PIX, s=55,
               marker='x', c='black', linewidths=0.8, zorder=7,
               label=f'Saturated/masked-core ({(saturated & good).sum()})')
    ax.scatter(mag[art], fr[art]*PIX, s=70, facecolor='none', edgecolor='blue',
               lw=1.3, zorder=5, label=f'Artifacts ({art.sum()})')
    ax.axhline(locus_px*PIX, color='k', ls='--', lw=1.3, alpha=0.6,
               label=f'Stellar locus = {locus_px*PIX:.3f}″ ({locus_px:.2f} px)')
    ax.axhline(cut_px*PIX, color='blue', ls='-', lw=2.0, alpha=0.9,
               label=f'Artifact cut = locus−3·MAD = {cut_px*PIX:.3f}″ ({cut_px:.2f} px)')
    ax.set_xlabel(f'MAG_AUTO ({band_upper}, AB)', fontsize=15, family='serif')
    ax.set_ylabel('Half-light radius FLUX_RADIUS [arcsec]', fontsize=15, family='serif')
    ax.set_ylim(0, 0.55); ax.set_xlim(13.5, 30.5)
    ax.set_title(f'JWST {band_upper} — Magnitude vs Half-Light Radius',
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
    ax.set_ylim(0.5, max(chi2_acc.max()*1.1, 5))
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


def plot_saturation_peak(out_png, mag, peak, cs, snr, ncoremask, band_upper):
    ok = np.isfinite(mag) & (mag > 13) & (mag < 30)
    is_point = (cs > 0.8) & (snr > 100)
    sat = ncoremask > 0
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.scatter(mag[ok], peak[ok], s=4, c='lightgray', alpha=0.4,
               label='All', rasterized=True)
    ax.scatter(mag[ok & is_point], peak[ok & is_point], s=8, c='red', alpha=0.7,
               label=f'Point sources ({(ok & is_point).sum()})', rasterized=True)
    ax.set_yscale('log')
    ax.set_xlabel(f'MAG_AUTO ({band_upper}, AB)', fontsize=13, family='serif')
    ax.set_ylabel('Peak Count FLUX_MAX [MJy/sr]', fontsize=13, family='serif')
    ax.set_title('Peak Counts vs Magnitude (Plateau at Bright End = SATURATION)',
                 fontsize=12, family='serif')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.set_xlim(13.5, 30.5)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=11, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()

    ax = axes[1]
    brt = ok & (mag < 22)
    ax.scatter(mag[brt & is_point], peak[brt & is_point], s=12, c='red',
               label='Point sources', rasterized=True)
    ax.scatter(mag[brt & sat], peak[brt & sat], s=40, marker='o',
               facecolor='none', edgecolor='blue', linewidths=1.5,
               label='Masked core (>0 px)', zorder=5, rasterized=True)
    ax.set_xlabel(f'MAG_AUTO ({band_upper}, AB)', fontsize=13, family='serif')
    ax.set_ylabel('Peak Count FLUX_MAX [MJy/sr]', fontsize=13, family='serif')
    ax.set_title('Bright-End Zoom — Look for the Flat Plateau', fontsize=12, family='serif')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(13.5, 22.5)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.grid(True, which='major', alpha=0.4)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=11, length=6)
    ax.tick_params(which='minor', length=3); ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)


def plot_psf_mosaics(out_samp_png, out_resi_png, samp, resi, ocd,
                     in_mosaic_idx, det_cat, band_upper):
    """psf_samples.png + psf_residuals.png — uses the same vmin/vmax (sqrt)
    derived from the samples mosaic.  All cells labelled, FLAGS_PSF≠0
    cells get a red frame; high-χ² accepted cells get an orange frame."""
    mos = ocd[in_mosaic_idx]
    N = len(mos)
    # Auto-detect cell size: Euclid VIS=51, SW=101, LW=151
    cs = None
    for candidate in (151, 101, 51):
        if samp.shape[0] % candidate == 0 and samp.shape[1] % candidate == 0:
            cs = candidate; break
    if cs is None:
        sys.exit(f'Cannot detect cell size from mosaic shape {samp.shape}')
    nrow = samp.shape[0] // cs
    ncol = samp.shape[1] // cs
    print(f'  mosaic cell size detected: {cs}×{cs}, grid {nrow}×{ncol}', flush=True)
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

    def render(data, title, fname):
        fig, ax = plt.subplots(figsize=(22, 10.5 * nrow / 9))
        ax.imshow(data, origin='upper', cmap='gray', norm=norm, aspect='auto',
                  extent=[0, ncol, nrow, 0])
        for c in range(ncol+1): ax.axvline(c, color='cyan', lw=0.5, alpha=0.5)
        for r in range(nrow+1): ax.axhline(r, color='cyan', lw=0.5, alpha=0.5)
        for k in range(N):
            r = k // ncol; c = k % ncol
            if r >= nrow: continue
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
            if is_rej:
                ax.text(c+0.04, r+0.80, 'neighbor', ha='left', va='bottom',
                        color='red', fontsize=fs_lab, fontweight='bold',
                        family='serif', zorder=6)
            ax.text(c+0.04, r+0.93, f'm={mag[k]:.1f}', ha='left', va='bottom',
                    color=col, fontsize=fs_lab, family='serif', zorder=6)
            ax.text(c+0.96, r+0.93, f'χ²={chi2_psf[k]:.1f}',
                    ha='right', va='bottom', color=col, fontsize=fs_lab,
                    family='serif', zorder=6)
        ax.set_xticks(np.arange(ncol)+0.5); ax.set_xticklabels(range(1, ncol+1),
            fontsize=11, family='serif')
        ax.set_yticks(np.arange(nrow)+0.5); ax.set_yticklabels(range(1, nrow+1),
            fontsize=11, family='serif')
        ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
        ax.set_xlabel('Column (1=left → 20)', fontsize=14, family='serif')
        ax.set_ylabel('Row (1=top → 9)', fontsize=14, family='serif')
        ax.set_title(title, fontsize=13, family='serif')
        fig.tight_layout()
        fig.savefig(fname, dpi=100, bbox_inches='tight')
        plt.close(fig)

    n_acc = accepted.sum(); n_rej = (~accepted).sum()
    n_orange = ((chi2_psf > THRESH) & accepted).sum()
    render(samp,
           f'PSFEx {band_upper} — {n_acc} accepted + {n_rej} PSFEx-rejected '
           f'(red).  sqrt vmin=0, vmax={VMAX:.3f}.  Orange = χ² > mean+2σ '
           f'= {THRESH:.2f} ({n_orange}/{n_acc}).',
           out_samp_png)
    render(resi,
           f'PSFEx {band_upper} — residuals (data−model).  Same scaling as samples '
           f'(vmin=0, vmax={VMAX:.3f}, sqrt).  '
           f'Red frame = PSFEx-rejected ({n_rej});  Orange = high χ² ({n_orange}).',
           out_resi_png)


def plot_hist_n(out_png, n_arr, mask_flags_lt2, band_upper):
    means = {r: n_arr[r][mask_flags_lt2].mean() for r in [1,2,3,4,5]}
    n1c, n3c, n5c = n_arr[1][mask_flags_lt2], n_arr[3][mask_flags_lt2], n_arr[5][mask_flags_lt2]
    fig, ax = plt.subplots(figsize=(11, 7))
    bins = np.arange(0, max(n5c.max(), 30)+1) - 0.5
    ax.hist(n1c, bins=bins, color='red',   alpha=0.55,
            label=f'$n_1$ (1″)  N={mask_flags_lt2.sum()}',
            edgecolor='darkred', lw=0.8)
    ax.hist(n3c, bins=bins, color='green', alpha=0.45,
            label=f'$n_3$ (3″)  N={mask_flags_lt2.sum()}',
            edgecolor='darkgreen', lw=0.8)
    ax.hist(n5c, bins=bins, color='blue',  alpha=0.35,
            label=f'$n_5$ (5″)  N={mask_flags_lt2.sum()}',
            edgecolor='darkblue', lw=0.8)
    ax.axvline(means[1], color='darkred',    ls=':', lw=2.0, label=f'mean $n_1$ = {means[1]:.2f}')
    ax.axvline(means[2], color='orange',     ls=':', lw=2.0, label=f'mean $n_2$ = {means[2]:.2f}')
    ax.axvline(means[3], color='darkgreen',  ls=':', lw=2.0, label=f'mean $n_3$ = {means[3]:.2f}')
    ax.axvline(means[4], color='teal',       ls=':', lw=2.0, label=f'mean $n_4$ = {means[4]:.2f}')
    ax.axvline(means[5], color='darkblue',   ls=':', lw=2.0, label=f'mean $n_5$ = {means[5]:.2f}')
    ax.axvline(1.5, color='red', ls='--', lw=1.5, alpha=0.7, label='cut: $n_1\\geq2$')
    ax.set_xlabel('Neighbour count', fontsize=15, family='serif')
    ax.set_ylabel('Number of PSF candidates (FLAGS<2)', fontsize=15, family='serif')
    ax.set_title(f'Neighbour-Count Distributions for FLAGS<2 PSF Candidates ({band_upper}).  '
                 f'N={mask_flags_lt2.sum()}.', fontsize=13, family='serif')
    ax.set_xlim(-0.7, max(20, n5c.max()+1)+0.5)
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
    band = args.tile  # use tile id as filename suffix key
    band_upper = f'VIS {args.tile}'  # for plot titles
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

    # 3. SCI for peak + ncoremask
    sci = fits.getdata(psf_dir / f'sci_{band}.fits').astype(np.float32)
    print(f'  SCI shape: {sci.shape}')
    peak, ncoremask = compute_peak_and_ncoremask(sci, xx, yy, half=2)
    print(f'  masked-core sources: {(ncoremask>0).sum()}')

    # 4. stellar-locus + artifact-cut
    locus, mad, cut, std0 = compute_locus_and_cut(fr, cs, snr, flg, elon, fwhm)
    print(f'  locus={locus:.3f} px, MAD={mad:.3f} px, cut={cut:.3f} px')

    # 5. neighbour counts
    tree = cKDTree(np.column_stack([xx, yy]))
    n_arr = neighbour_counts(xx, yy, tree, [1, 2, 3, 4, 5])

    # 6. mosaic mapping (OUTCAT entries that pass input filter)
    in_mosaic_mask = (ocd['FLAGS_PSF'] & 32) == 0
    samp = fits.getdata(psf_dir / f'samp_stars_{band}.fits').astype(float)
    resi = fits.getdata(psf_dir / f'resi_stars_{band}.fits').astype(float)
    ncol, nrow, csize = 20, 9, 101
    if samp.shape != (nrow*csize, ncol*csize):
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
    plot_mag_vs_halflight(psf_dir / 'mag_vs_halflight.png',
                          mag, fr, fwhm, ell, ncoremask, locus, mad, cut, std0,
                          band_upper)
    print(f'     mag_vs_halflight.png')

    plot_mag_vs_chi2(psf_dir / 'mag_vs_chi2.png',
                     mag_acc[sel], chi2_acc_arr[sel],
                     n1_acc[sel], n3_acc[sel], flg_acc[sel], band_upper)
    print(f'     mag_vs_chi2.png')

    plot_saturation_peak(psf_dir / 'saturation_peak.png',
                         mag, peak, cs, snr, ncoremask, band_upper)
    print(f'     saturation_peak.png')

    plot_psf_mosaics(psf_dir / 'psf_samples.png',
                     psf_dir / 'psf_residuals.png',
                     samp, resi, ocd, in_mosaic_idx, pass1, band_upper)
    print(f'     psf_samples.png / psf_residuals.png')

    # FLAGS<2 PSF candidates
    base = (cs > 0.8) & (snr > 100) & (flg < 2) & (elon < 1.5) & (fr > 0)
    locus_m, mad_m, cut_m, std0_m = compute_locus_and_cut(fr, cs, snr, flg, elon, fwhm)
    saturated = ncoremask > 0
    edge = (flg & 8) > 0
    candidate = ((cs > 0.8) & (snr > 100) & (elon < 1.5) & (fwhm > 0)
                 & (fr > cut_m) & (fr < locus_m + 2.5*std0_m)
                 & (~saturated) & (~edge))
    cand_flg_lt2 = candidate & (flg < 2)
    plot_hist_n(psf_dir / 'hist_n_means.png',
                {r: n_arr[r] for r in [1,2,3,4,5]},
                cand_flg_lt2, band_upper)
    print(f'     hist_n_means.png')
    print(f'=== {band_upper} A4 QA plots done ===')


if __name__ == '__main__':
    main()
