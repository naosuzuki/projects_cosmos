#!/usr/bin/env python
"""
51_step3a_chi2_detect.py — Step 3a-② mission-generic χ²₊ hot+cold detection.

Builds the χ₊ detection image per (mission, tile) and runs the
SExtractor cold + hot passes, then merges via the cold-Kron-ellipse
criterion.  Follows the Shuntov+ 2025 (COSMOS2025) recipe verified
against Galametz+ 2013 (CANDELS); locked plan in ms.tex §B.1 Step 3a,
eq. (chi2plus), in its SQUARE-ROOT (chi) form:

    χ₊(x,y) = sqrt( (1/N) Σ_b [ max( SCI^h_b(x,y)·√WHT_b(x,y), 0 ) ]² )

IMPORTANT — why the sqrt (2026-06-10, validated on JWST A4):
ms.tex eq. (chi2plus) writes the straight sum of squares, but detecting
on the SQUARED image is numerically unstable: its noise floor is so
skewed that SExtractor's mesh-clipped background-RMS map blows up
around sources (p99 of the RMS map 107 vs 242 for two kernel sets that
differ by only ~15%), and the A4 cold count swung 10,913 → 2,163 from
that kernel tweak alone.  The sqrt form is what SWarp's CHI2 combine
(Szalay+ 1999) actually produces — i.e. what Shuntov+ 2025 and
Galametz+ 2013 really ran SExtractor on — and its background is
near-Gaussian: same A4 cutout, cold counts 315 vs 323 (2.5%) and a
stable RMS map (p99 0.69 vs 0.70) across the same kernel change.
DETECT_THRESH in σ is only meaningful on this form.  (ms.tex §B.1
equation to be amended accordingly.)

Missions (detection stacks per ms.tex §B.1):
  jwst    F115W+F150W+F277W+F444W → homogenized to F444W   (i2d SCI/WHT)
  euclid  VIS + NISP Y/J/H        → homogenized to NISP H  (MER BGSUB,
          no RMS maps on disk → per-band robust σ, measured AFTER the
          homogenization convolution, replaces √WHT)
  hst     F814W single band (N=1 stack; drz × √wht)

PSF homogenization uses a Gaussian kernel with σ²_k = σ²_tgt − σ²_src,
where each band's FWHM is the EMPIRICAL PSFEx value from the Step 3a-①
meta JSON of the same tile (fallback: per-instrument median over all
tiles, then a static estimate).

Per tile the script is resumable: the χ²₊ image is reused only when its
sidecar meta matches the requested (bands, target, fwhm) configuration,
and a completed run (detect meta + merged catalog present) is skipped
unless --force.

Outputs under /Volumes/exdisk1/data/photometry_v04/<mission>_chi2/<TILE>/:
  chi2_<TILE>.fits            χ²₊ detection image (float32, target-band WCS)
  chi2_<TILE>.meta.json       stack provenance (bands, σ/WHT mode, FWHMs)
  cat_cold_<TILE>.fits        FITS_LDAC cold catalog
  cat_hot_<TILE>.fits         FITS_LDAC hot catalog
  seg_cold_<TILE>.fits        cold segmentation (drives the merge QA)
  seg_hot_<TILE>.fits         hot segmentation
  merged_<TILE>.fits          merged catalog + DETECT_MODE cold/hot + SOURCE_ID
  detect_<TILE>.meta.json     counts, timings, parameters (resume marker)
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')
JWST_DIR = Path('/Volumes/exdisk1/data/JWST/COSMOS_v0.8')
HST_DIR  = Path('/Volumes/exdisk1/data/HST/COSMOS_ACS2005')
EUC_DIR  = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')

AB_TILES = [f'{r}{n}' for r in 'AB' for n in range(1, 11)]

# mission → (band list, homogenization target, pixel scale ["/px],
#            instrument-dir name per band [for the 3a-① meta lookup],
#            hot_thresh: per-mission HOT DETECT_THRESH override).
#
# hot_thresh provenance: the Shuntov hot pass (3.0σ, MINAREA 8) is
# spurious-DOMINATED on our stacks because the PSF-homogenization
# kernels correlate the noise (negative-image test on JWST A4 cutout:
# neg/pos = 1.39 @3.0σ → 0.056 @5.0σ; cold pass is pure, neg = 0).
# Each mission's value is calibrated from its own per-tile
# negative-image QA (negqa_* in the chi2 meta); None = config default
# (3.0) until calibrated.
MISSIONS = {
    'jwst': dict(
        bands=['f115w', 'f150w', 'f277w', 'f444w'], target='f444w',
        pixscale=0.030, hot_thresh=5.0,
        inst={b: f'jwst_nircam_{b}' for b in
              ('f115w', 'f150w', 'f277w', 'f444w')}),
    'hst': dict(
        bands=['f814w'], target='f814w', pixscale=0.030, hot_thresh=None,
        inst={'f814w': 'hst_acs_f814w'}),
    'euclid': dict(
        bands=['vis', 'nisp_y', 'nisp_j', 'nisp_h'], target='nisp_h',
        pixscale=0.100, hot_thresh=None,
        inst={'vis': 'euclid_vis', 'nisp_y': 'euclid_nisp_y',
              'nisp_j': 'euclid_nisp_j', 'nisp_h': 'euclid_nisp_h'}),
}

RECIPE_VERSION = 2          # 2 = chi (sqrt) statistic + negative-image QA
NEGQA_WINDOW = 8192         # central window (px) for the per-tile negative QA

# last-resort FWHM estimates (arcsec) if NO 3a-① meta exists anywhere
STATIC_FWHM_ARCSEC = {
    'f115w': 0.073, 'f150w': 0.085, 'f277w': 0.120, 'f444w': 0.165,
    'f814w': 0.095,
    'vis': 0.160, 'nisp_y': 0.350, 'nisp_j': 0.370, 'nisp_h': 0.400,
}

MAX_COLD_ELLIPSE_PX = 2000.0   # sanity cap on a cold Kron semi-major axis


def step(msg: str):
    print(f'\n── {msg} ──', flush=True)


def euclid_tiles() -> list[str]:
    return sorted({p.name.split('TILE')[1].split('-')[0]
                   for p in EUC_DIR.glob('EUC_MER_BGSUB-MOSAIC-VIS_TILE*.fits')})


# ----------------------------------------------------------------------
# empirical PSF FWHM lookup (Step 3a-① meta JSONs)
# ----------------------------------------------------------------------
_median_cache: dict[str, float | None] = {}


def _meta_path(inst: str, tile: str, band: str) -> Path:
    # jwst metas are keyed by band, all others by tile (see 60_)
    suffix = band if inst.startswith('jwst') else tile
    return WORK / inst / tile / 'psf' / f'psf_{suffix}.meta.json'


def _meta_fwhm_px(p: Path) -> float | None:
    try:
        m = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    v = m.get('psf_fwhm_px') or m.get('psf_fwhm_est_px')
    return float(v) if v else None


def _instrument_median_fwhm_px(inst: str) -> float | None:
    if inst not in _median_cache:
        vals = [v for p in WORK.glob(f'{inst}/*/psf/psf_*.meta.json')
                if (v := _meta_fwhm_px(p)) is not None]
        _median_cache[inst] = float(np.median(vals)) if vals else None
    return _median_cache[inst]


def empirical_fwhm_arcsec(mission: str, tile: str, band: str) -> tuple[float, str]:
    """Return (FWHM ["], provenance) for a band on a tile."""
    spec = MISSIONS[mission]
    inst, scale = spec['inst'][band], spec['pixscale']
    v = _meta_fwhm_px(_meta_path(inst, tile, band))
    if v:
        return v * scale, 'tile_meta'
    v = _instrument_median_fwhm_px(inst)
    if v:
        return v * scale, 'instrument_median'
    return STATIC_FWHM_ARCSEC[band], 'static_estimate'


def kernel_sigma_px(fwhm_src: float, fwhm_tgt: float, pixscale: float) -> float:
    """σ (px) of the Gaussian that brings FWHM src ["] up to tgt ["]."""
    if fwhm_src >= fwhm_tgt:
        return 0.0
    s_src = fwhm_src / 2.355 / pixscale
    s_tgt = fwhm_tgt / 2.355 / pixscale
    return float(np.sqrt(s_tgt**2 - s_src**2))


# ----------------------------------------------------------------------
# per-mission band loaders → noise-equalized, homogenized NSCI
# ----------------------------------------------------------------------
def _strip_structural(hdr) -> dict:
    drop = ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND',
            'XTENSION', 'PCOUNT', 'GCOUNT', 'EXTNAME', 'HISTORY', 'COMMENT')
    return {k: v for k, v in dict(hdr).items() if k not in drop}


def load_jwst_nsci(tile: str, band: str, sigma_px: float):
    """JWST: SCI homogenized then × √WHT (the i2d weight map)."""
    src = JWST_DIR / (f'mosaic_nircam_{band}_COSMOS-Web_30mas_'
                      f'{tile}_v1.0_i2d.fits')
    if not src.exists():
        sys.exit(f'missing JWST mosaic: {src}')
    t0 = time.time()
    with fits.open(src) as h:
        hdr = _strip_structural(h['SCI'].header)
        sci = h['SCI'].data.astype(np.float32, copy=True)
        wht = h['WHT'].data.astype(np.float32, copy=True)
    if sigma_px > 0.1:
        sci = gaussian_filter(sci, sigma_px, mode='constant', cval=0)
    np.sqrt(np.maximum(wht, 0, out=wht), out=wht)
    sci *= wht
    del wht
    print(f'    {band:7s} shape {sci.shape}  σ_k={sigma_px:.2f} px '
          f'noise=√WHT  ({time.time()-t0:.0f}s)', flush=True)
    return sci, hdr, dict(noise='sqrt_WHT')


def load_hst_nsci(tile: str, band: str, sigma_px: float):
    """HST F814W: drz × √wht; A tiles are 2023apr, B tiles 2024jan."""
    matches = sorted(HST_DIR.glob(
        f'mosaic_cosmos_web_*_30mas_tile_{tile}_hst_acs_wfc_f814w_drz.fits'))
    if not matches:
        sys.exit(f'missing HST drz for tile {tile}')
    drz = matches[0]
    whtf = Path(str(drz).replace('_drz.fits', '_wht.fits'))
    if not whtf.exists():
        sys.exit(f'missing HST wht: {whtf}')
    t0 = time.time()
    with fits.open(drz) as h:
        hdr = _strip_structural(h[0].header)
        sci = h[0].data.astype(np.float32, copy=True)
    with fits.open(whtf) as h:
        wht = h[0].data.astype(np.float32, copy=True)
    if sigma_px > 0.1:
        sci = gaussian_filter(sci, sigma_px, mode='constant', cval=0)
    np.sqrt(np.maximum(wht, 0, out=wht), out=wht)
    sci *= wht
    del wht
    print(f'    {band:7s} shape {sci.shape}  σ_k={sigma_px:.2f} px '
          f'noise=√wht ({drz.name.split("_30mas")[0][-7:]})  '
          f'({time.time()-t0:.0f}s)', flush=True)
    return sci, hdr, dict(noise='sqrt_WHT', drz=drz.name)


def _euclid_mosaic(tile: str, band: str) -> Path | None:
    tag = {'vis': 'VIS', 'nisp_y': 'NIR-Y',
           'nisp_j': 'NIR-J', 'nisp_h': 'NIR-H'}[band]
    m = sorted(EUC_DIR.glob(f'EUC_MER_BGSUB-MOSAIC-{tag}_TILE{tile}-*.fits'))
    return m[0] if m else None


def load_euclid_nsci(tile: str, band: str, sigma_px: float):
    """Euclid MER BGSUB (single HDU, no RMS map on disk): homogenize,
    then noise-equalize by a single robust σ measured on the CONVOLVED
    image over covered pixels (uniform-depth approximation; swap in the
    MER RMS mosaics if they are downloaded later)."""
    src = _euclid_mosaic(tile, band)
    if src is None:
        return None, None, None          # band genuinely absent
    t0 = time.time()
    with fits.open(src) as h:
        hdr = _strip_structural(h[0].header)
        sci = h[0].data.astype(np.float32, copy=True)
    cov = np.isfinite(sci) & (sci != 0)
    n_cov = int(cov.sum())
    if n_cov < 1000:                     # empty NISP tile edge case
        print(f'    {band:7s} coverage {n_cov} px — skipped', flush=True)
        return None, None, None
    np.nan_to_num(sci, copy=False)
    if sigma_px > 0.1:
        sci = gaussian_filter(sci, sigma_px, mode='constant', cval=0)
    samp = sci[::4, ::4][cov[::4, ::4]]
    med = float(np.median(samp))
    sigma = float(1.4826 * np.median(np.abs(samp - med)))
    if sigma <= 0:
        print(f'    {band:7s} σ_rob=0 — skipped', flush=True)
        return None, None, None
    sci /= sigma
    sci[~cov] = 0
    print(f'    {band:7s} shape {sci.shape}  σ_k={sigma_px:.2f} px '
          f'σ_rob={sigma:.4g}  cov={n_cov/cov.size:.1%}  '
          f'({time.time()-t0:.0f}s)', flush=True)
    return sci, hdr, dict(noise='robust_sigma', sigma_robust=sigma,
                          coverage_frac=round(n_cov / cov.size, 4))


LOADERS = {'jwst': load_jwst_nsci, 'hst': load_hst_nsci,
           'euclid': load_euclid_nsci}


# ----------------------------------------------------------------------
# χ²₊ stack
# ----------------------------------------------------------------------
def build_chi2(mission: str, tile: str, work: Path,
               force: bool) -> tuple[Path, dict]:
    spec = MISSIONS[mission]
    chi2_path = work / f'chi2_{tile}.fits'
    meta_path = work / f'chi2_{tile}.meta.json'

    fwhms = {}
    for b in spec['bands']:
        f, prov = empirical_fwhm_arcsec(mission, tile, b)
        fwhms[b] = dict(fwhm_arcsec=round(f, 4), provenance=prov)
    fwhm_tgt = fwhms[spec['target']]['fwhm_arcsec']

    want = dict(mission=mission, tile=tile, bands=spec['bands'],
                target=spec['target'], fwhms=fwhms, cutout=0,
                statistic='chi_plus = sqrt(mean positive square)',
                recipe_version=RECIPE_VERSION)
    if chi2_path.exists() and meta_path.exists() and not force:
        try:
            have = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            have = {}
        if {k: have.get(k) for k in want} == want:
            print(f'  reusing {chi2_path.name} (meta matches)')
            return chi2_path, have
        print(f'  existing {chi2_path.name} meta MISMATCH — rebuilding')

    chi2 = None          # positive stack (full frame)
    neg = None           # negative stack (central QA window)
    win = None
    ref_hdr = None
    band_info = {}
    used = []
    for b in spec['bands']:
        sigma_px = kernel_sigma_px(fwhms[b]['fwhm_arcsec'], fwhm_tgt,
                                   spec['pixscale'])
        nsci, hdr, info = LOADERS[mission](tile, b, sigma_px)
        if nsci is None:
            band_info[b] = dict(used=False)
            continue
        info.update(kernel_sigma_px=round(sigma_px, 3), used=True,
                    **fwhms[b])
        band_info[b] = info
        used.append(b)
        if win is None:
            ny, nx = nsci.shape
            wy, wx = min(NEGQA_WINDOW, ny), min(NEGQA_WINDOW, nx)
            win = (slice((ny - wy)//2, (ny - wy)//2 + wy),
                   slice((nx - wx)//2, (nx - wx)//2 + wx))
        # negative half of the QA window, BEFORE truncation
        nw = np.minimum(nsci[win], 0).astype(np.float32)
        np.square(nw, out=nw)
        neg = nw if neg is None else neg + nw
        # positive full-frame stack
        np.maximum(nsci, 0, out=nsci)
        np.square(nsci, out=nsci)
        if chi2 is None:
            chi2, ref_hdr = nsci, hdr
        else:
            if nsci.shape != chi2.shape:
                sys.exit(f'{mission}/{tile}: band {b} shape {nsci.shape} '
                         f'!= stack {chi2.shape} — grids must match')
            chi2 += nsci
            del nsci
    if chi2 is None:
        sys.exit(f'{mission}/{tile}: no usable band')
    n_used = np.float32(len(used))
    chi2 /= n_used
    np.sqrt(chi2, out=chi2)
    neg /= n_used
    np.sqrt(neg, out=neg)
    print(f'  χ₊ bands used: {used}   '
          f'med {np.median(chi2):.3g}  max {chi2.max():.3g}')

    h = fits.PrimaryHDU(data=chi2)
    for k, v in ref_hdr.items():
        try:
            h.header[k] = v
        except Exception:
            pass
    h.header['BUNIT'] = ('chi+', 'sqrt of mean positive square (Szalay+99)')
    h.header['HISTORY'] = f'chi+ stack: {"+".join(used)}'
    h.header['HISTORY'] = (f'homogenized to {spec["target"]} '
                           f'FWHM={fwhm_tgt}"')
    h.writeto(chi2_path, overwrite=True)
    print(f'  [save] {chi2_path}  ({chi2_path.stat().st_size/1e9:.2f} GB)')

    negqa = run_negqa(mission, tile, work, chi2[win], neg)
    del chi2

    meta = dict(want, bands_used=used, band_info=band_info, negqa=negqa,
                created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    meta_path.write_text(json.dumps(meta, indent=2))
    return chi2_path, meta


def run_negqa(mission: str, tile: str, work: Path,
              pos_win: np.ndarray, neg_win: np.ndarray) -> dict:
    """Per-tile spurious-detection QA: run the cold and hot passes on the
    positive and negative chi windows; the negative image contains only
    (correlated) noise + artifacts, so n_neg/n_pos estimates the spurious
    fraction at the current thresholds."""
    spec = MISSIONS[mission]
    seeing = empirical_fwhm_arcsec(mission, tile, spec['target'])[0]
    out = dict(window_px=list(pos_win.shape),
               hot_thresh=spec['hot_thresh'] or 3.0)
    for tag, img in (('pos', pos_win), ('neg', neg_win)):
        f = work / f'_negqa_{tag}_{tile}.fits'
        fits.PrimaryHDU(data=np.ascontiguousarray(img)).writeto(
            f, overwrite=True)
        for mode in ('cold', 'hot'):
            cat = work / f'_negqa_{tag}_{mode}_{tile}.fits'
            run_sex_mode(f, mode, work, tile, spec['pixscale'], seeing,
                         hot_thresh=spec['hot_thresh'],
                         cat_override=cat, seg=False, quiet=True)
            out[f'n_{tag}_{mode}'] = len(fits.open(cat)[2].data)
            cat.unlink()
        f.unlink()
    out['spurious_frac_hot'] = round(
        out['n_neg_hot'] / max(out['n_pos_hot'], 1), 4)
    out['spurious_frac_cold'] = round(
        out['n_neg_cold'] / max(out['n_pos_cold'], 1), 4)
    print(f"  [negqa] window {out['window_px']}  "
          f"cold {out['n_pos_cold']}/{out['n_neg_cold']} (pos/neg)  "
          f"hot {out['n_pos_hot']}/{out['n_neg_hot']}  "
          f"spurious_hot={out['spurious_frac_hot']:.3f}")
    return out


# ----------------------------------------------------------------------
# SExtractor cold / hot passes
# ----------------------------------------------------------------------
def run_sex_mode(image: Path, mode: str, work: Path, tile: str,
                 pixscale: float, seeing: float,
                 hot_thresh: float | None = None,
                 cat_override: Path | None = None,
                 seg: bool = True, quiet: bool = False) -> Path:
    cfg = CONFIGS / f'chi2_{mode}.sex'
    out_cat = cat_override or work / f'cat_{mode}_{tile}.fits'
    cmd = ['sex', str(image),
           '-c',               str(cfg),
           '-CATALOG_NAME',    str(out_cat),
           '-PARAMETERS_NAME', str(CONFIGS / 'chi2_detect.param'),
           '-STARNNW_NAME',    str(CONFIGS / 'default.nnw'),
           '-FILTER_NAME',     str(CONFIGS / (
               'tophat_9.0_9x9.conv' if mode == 'cold'
               else 'gauss_3.0_5x5.conv')),
           '-PIXEL_SCALE',     f'{pixscale:.4f}',
           '-SEEING_FWHM',     f'{seeing:.4f}',
           ]
    if seg:
        cmd += ['-CHECKIMAGE_NAME', str(work / f'seg_{mode}_{tile}.fits')]
    else:
        cmd += ['-CHECKIMAGE_TYPE', 'NONE']
    if mode == 'hot' and hot_thresh is not None:
        cmd += ['-DETECT_THRESH', f'{hot_thresh:.2f}']
    if not quiet:
        print(f'  sex {image.name} -c {cfg.name} → {out_cat.name}',
              flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit(f'SExtractor {mode} pass failed (rc={r.returncode})')
    if not quiet:
        print(f'  wall: {time.time()-t0:.0f}s')
    return out_cat


# ----------------------------------------------------------------------
# merge: all cold + hot outside every cold Kron ellipse
# ----------------------------------------------------------------------
def merge_catalogs(cold_cat: Path, hot_cat: Path,
                   out_path: Path) -> tuple[int, int, int]:
    from astropy.table import Table, vstack
    from scipy.spatial import cKDTree

    cold = fits.open(cold_cat)[2].data
    hot = fits.open(hot_cat)[2].data
    n_cold, n_hot = len(cold), len(hot)

    if n_hot == 0 or n_cold == 0:
        cold_t = Table(cold)
        cold_t['DETECT_MODE'] = np.full(n_cold, 'cold')
        hot_t = Table(hot)
        hot_t['DETECT_MODE'] = np.full(n_hot, 'hot')
        merged = vstack([cold_t, hot_t]) if n_hot else cold_t
        merged['SOURCE_ID'] = np.arange(1, len(merged) + 1)
        merged.write(out_path, format='fits', overwrite=True)
        print(f'  degenerate merge: cold {n_cold:,}  hot {n_hot:,}')
        return n_cold, n_hot, len(merged)

    xc = cold['X_IMAGE'].astype(np.float64)
    yc = cold['Y_IMAGE'].astype(np.float64)
    a = cold['A_IMAGE'].astype(np.float64)
    b = cold['B_IMAGE'].astype(np.float64)
    th = np.deg2rad(cold['THETA_IMAGE'].astype(np.float64))
    kron = cold['KRON_RADIUS'].astype(np.float64)
    kron[kron <= 0] = 2.5                       # SEx writes 0 when unfit
    a_k = np.minimum(a * kron, MAX_COLD_ELLIPSE_PX)
    b_k = np.minimum(b * kron, MAX_COLD_ELLIPSE_PX)

    hx = hot['X_IMAGE'].astype(np.float64)
    hy = hot['Y_IMAGE'].astype(np.float64)
    tree = cKDTree(np.column_stack([hx, hy]))
    inside_any = np.zeros(n_hot, dtype=bool)
    # per-cold query radius = its Kron semi-major axis
    neigh = tree.query_ball_point(np.column_stack([xc, yc]), r=a_k)
    for i, idx in enumerate(neigh):
        if not idx or a_k[i] <= 0 or b_k[i] <= 0:
            continue
        idx = np.asarray(idx)
        dx, dy = hx[idx] - xc[i], hy[idx] - yc[i]
        ct, st = np.cos(th[i]), np.sin(th[i])
        xp = ct * dx + st * dy
        yp = -st * dx + ct * dy
        inside_any[idx[(xp / a_k[i])**2 + (yp / b_k[i])**2 < 1.0]] = True

    keep = ~inside_any
    n_kept = int(keep.sum())
    cold_t = Table(cold)
    cold_t['DETECT_MODE'] = np.full(n_cold, 'cold')
    hot_t = Table(hot[keep])
    hot_t['DETECT_MODE'] = np.full(n_kept, 'hot')
    merged = vstack([cold_t, hot_t])
    merged['SOURCE_ID'] = np.arange(1, len(merged) + 1)
    merged.write(out_path, format='fits', overwrite=True)
    print(f'  cold {n_cold:,}  hot {n_hot:,}  hot kept {n_kept:,}  '
          f'→ {len(merged):,} merged')
    return n_cold, n_kept, len(merged)


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mission', required=True, choices=sorted(MISSIONS))
    p.add_argument('--tile', required=True,
                   help='A1..B10 (jwst/hst) or MER tile id (euclid)')
    p.add_argument('--force', action='store_true',
                   help='rebuild even if detect meta exists')
    p.add_argument('--skip-stack', action='store_true',
                   help='reuse existing chi2 image regardless of meta (debug)')
    return p.parse_args()


def main():
    args = parse_args()
    spec = MISSIONS[args.mission]
    work = WORK / f'{args.mission}_chi2' / args.tile
    work.mkdir(parents=True, exist_ok=True)
    detect_meta = work / f'detect_{args.tile}.meta.json'
    merged_path = work / f'merged_{args.tile}.fits'

    if detect_meta.exists() and merged_path.exists() and not args.force:
        print(f'{args.mission}/{args.tile}: already done '
              f'({detect_meta.name} present) — use --force to redo')
        return

    print(f'mission {args.mission}   tile {args.tile}   workdir {work}')
    t0 = time.time()

    step('1. χ²₊ detection image')
    if args.skip_stack and (work / f'chi2_{args.tile}.fits').exists():
        chi2_path = work / f'chi2_{args.tile}.fits'
        chi2_meta = {}
    else:
        chi2_path, chi2_meta = build_chi2(args.mission, args.tile, work,
                                          args.force)
    t_stack = time.time()

    seeing = empirical_fwhm_arcsec(args.mission, args.tile,
                                   spec['target'])[0]
    step('2. SExtractor cold pass')
    cold_cat = run_sex_mode(chi2_path, 'cold', work, args.tile,
                            spec['pixscale'], seeing)
    t_cold = time.time()
    step('3. SExtractor hot pass')
    hot_cat = run_sex_mode(chi2_path, 'hot', work, args.tile,
                           spec['pixscale'], seeing,
                           hot_thresh=spec['hot_thresh'])
    t_hot = time.time()

    step('4. merge cold + hot (Kron-ellipse criterion)')
    n_cold, n_hot_kept, n_total = merge_catalogs(cold_cat, hot_cat,
                                                 merged_path)
    t_end = time.time()

    detect_meta.write_text(json.dumps(dict(
        mission=args.mission, tile=args.tile,
        n_cold=n_cold, n_hot_kept=n_hot_kept, n_total=n_total,
        hot_thresh=spec['hot_thresh'] or 3.0,
        seeing_fwhm_arcsec=round(seeing, 4),
        stack_s=round(t_stack - t0, 1), cold_s=round(t_cold - t_stack, 1),
        hot_s=round(t_hot - t_cold, 1), merge_s=round(t_end - t_hot, 1),
        total_s=round(t_end - t0, 1),
        chi2_meta=chi2_meta,
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    ), indent=2))
    print(f'\n=== {args.mission}/{args.tile} detect complete ===')
    print(f'  cold {n_cold:,}   hot kept {n_hot_kept:,}   '
          f'total {n_total:,}   wall {t_end-t0:.0f}s')
    print(f'  merged catalog: {merged_path}')


if __name__ == '__main__':
    main()
