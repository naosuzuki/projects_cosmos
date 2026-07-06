#!/usr/bin/env python
"""
54_step3a_build_psf_model_hst.py — HST ACS/WFC F814W PSF model builder
on the COSMOS-Web 30 mas drizzled tiles.

v2 (2026-07): star selection made CONSISTENT with the hostgalxy PSF
recipe (projects_hsc/hostgalxy/programs/psf/build_psf.py):

  - SPACE-TELESCOPE ISOLATION = 10 x PSF_FWHM (user rule): any other
    detected source within R_nbr = 10*psf_fwhm_est native px rejects the
    candidate (was: n_1>=2 within 1" AND FLAGS<2, which passed single
    companions at 0.55" and everything at 1-1.5" inside the +-1.5" stamp).
  - 7-detector STAMP-QUALITY GATE, bits into stars_rejected_<tile>.fits:
       1 edge      near frame border (< psf_size/2) or SExtractor edge flag
       2 nbr       catalogue neighbour within 10 x FWHM
       4 bad-pix   masked/zero pixels in the VIGNET footprint
       8 blend     >=2 local maxima in the raw core stamp (sub-FWHM blends
                   SExtractor cannot deblend; catches undetected companions)
      16 sat       masked core OR core peak on the bright-end plateau
      32 asym      integrated L/R / T/B residual flux imbalance vs the
                   provisional PSF (shoulder companion, no 2nd peak);
                   data-driven threshold max(5%, 98th pct)
      64 outer     >8 sigma residual blob in the r=12..psf/2 annulus
                   (faint wing companion, not a catalogue source)
  - two-stage PSFEx: pass-2a (gate-1 survivors -> provisional .psf, used
    only by the asym/outer gates), pass-2b (final model + OUTCAT from the
    SAME run so cell<->star mapping stays aligned; no check-images -- the
    QA mosaics are raw-cutout based).

Format / calibration (unchanged): single-HDU drz, WEIGHT NONE, ZP from
PHOTFLAM/PHOTPLAM, 0.030"/px, VIGNET 101 px, tiles A1..B10 from
/Volumes/exdisk1/data/HST/COSMOS_ACS2005.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter, shift as ndshift, zoom
from psf_saturation import core_peak, detect_saturation

PROJECT  = Path('/Users/suzuki/github/projects_cosmos')
CONFIGS  = PROJECT / 'configs'
HST_ROOT = Path('/Volumes/exdisk1/data/HST/COSMOS_ACS2005')
WORK     = Path('/Volumes/exdisk1/data/photometry_v04')

PIX_HST      = 0.030   # arcsec / pix (COSMOS-Web 30 mas drizzle of ACS/WFC)
VIG_HALF     = 50      # VIGNET(101,101) half-size
PSF_SIZE     = 101     # native model stamp (= VIGNET)
NBR_ISO_FWHM = 10.0    # space-telescope isolation radius in PSF-FWHM units


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instrument', default='hst_acs_f814w')
    p.add_argument('--tile', required=True,
                   help='COSMOS-Web tile id (e.g. A4)')
    p.add_argument('--core-box', type=int, default=3,
                   help='central NxN box to test for masked (=0) pixels')
    p.add_argument('--reuse-pass1', action='store_true')
    return p.parse_args()


def find_tile_image(tile_id: str) -> Path:
    # A-tiles are "2023apr", B-tiles "2024jan" — glob the date token.
    matches = sorted(HST_ROOT.glob(
        f'mosaic_cosmos_web_*_30mas_tile_{tile_id}_hst_acs_wfc_f814w_drz.fits'))
    if not matches:
        sys.exit(f'No HST ACS/WFC F814W image matches tile {tile_id} in {HST_ROOT}')
    return matches[0]


def run_psfex(star_cat, fwhm_lo, fwhm_hi, extra):
    cmd = ['psfex', str(star_cat),
           '-c', str(CONFIGS / 'psfex_hst_acs.psfex'),
           '-SAMPLE_FWHMRANGE', f'{fwhm_lo:.2f},{fwhm_hi:.2f}'] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr); sys.exit('PSFEx failed')
    psf = Path(star_cat).with_suffix('.psf')
    if not psf.exists():
        alt = Path.cwd() / psf.name
        if alt.exists(): shutil.move(alt, psf)
    return psf


def main():
    args = parse_args()
    img = find_tile_image(args.tile)
    out = WORK / args.instrument / args.tile / 'psf'
    out.mkdir(parents=True, exist_ok=True)

    with fits.open(img) as h:
        sci = h[0].data.astype(np.float32)
        hdr = h[0].header
    photflam = float(hdr['PHOTFLAM'])
    photplam = float(hdr['PHOTPLAM'])
    zp = -2.5*np.log10(photflam) - 5*np.log10(photplam) - 2.408
    print(f'Instrument : {args.instrument}')
    print(f'Tile       : {args.tile}   ZP_AB={zp:.4f}   PIX={PIX_HST}″')
    print(f'Image      : {img.name}  shape={sci.shape}')

    cat1 = out / f'pass1_{args.tile}.fits'
    if not (args.reuse_pass1 and cat1.exists()):
        print('\n── SExtractor pass 1 ──', flush=True)
        cmd = ['sex', str(img),
               '-c', str(CONFIGS / 'hst_acs_f814w.sex'),
               '-CATALOG_NAME', str(cat1),
               '-PARAMETERS_NAME', str(CONFIGS / 'pass1_hst_acs.param'),
               '-FILTER_NAME', str(CONFIGS / 'default.conv'),
               '-STARNNW_NAME', str(CONFIGS / 'default.nnw'),
               '-MAG_ZEROPOINT', f'{zp:.4f}',
               '-WEIGHT_TYPE', 'NONE',
               '-VERBOSE_TYPE', 'QUIET']
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], file=sys.stderr); sys.exit('SExtractor failed')
        print(f'  wall {time.time()-t0:.1f}s')

    # ── star selection ──
    print('\n── star selection ──', flush=True)
    hcat = fits.open(cat1)
    obj  = hcat[2].data
    mag  = np.asarray(obj['MAG_AUTO'], float)
    fr   = np.asarray(obj['FLUX_RADIUS'], float)
    fwhm = np.asarray(obj['FWHM_IMAGE'], float)
    cs   = np.asarray(obj['CLASS_STAR'], float)
    snr  = np.asarray(obj['SNR_WIN'], float)
    flg  = np.asarray(obj['FLAGS'], int)
    elon = np.asarray(obj['ELONGATION'], float)
    ell_arr = np.asarray(obj['ELLIPTICITY'], float)
    xx   = np.asarray(obj['X_IMAGE'], float)
    yy   = np.asarray(obj['Y_IMAGE'], float)
    ny, nx = sci.shape

    # FWHM artifact floor + iterative MAD locus (unchanged from v1)
    base0 = (cs > 0.8) & (snr > 100) & (flg < 2) & (elon < 1.5) & (fr > 0)
    hi = base0 & (snr > 1000)
    if hi.sum() < 5:
        hi = base0 & (snr > 500)
    psf_fwhm_est = float(np.median(fwhm[hi])) if hi.sum() >= 3 else float(np.median(fwhm[base0]))
    fwhm_min = 0.5 * psf_fwhm_est
    base = base0 & (fwhm > fwhm_min)
    print(f'  PSF FWHM estimate (high-SNR median): {psf_fwhm_est:.2f} px '
          f'→ base FWHM-min = {fwhm_min:.2f} px '
          f'(removes {(base0 & ~base).sum()} CR/hot-pixel artifacts)')
    std0 = np.std(fr[base])
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

    # masked-core + bright-end peak plateau (bit 16 'sat', unchanged physics)
    half = args.core_box // 2
    xi0 = np.clip(np.round(xx).astype(int), half, nx-half-1)
    yi0 = np.clip(np.round(yy).astype(int), half, ny-half-1)
    masked_core = np.zeros(len(obj), int)
    for k in range(len(obj)):
        box = sci[yi0[k]-half:yi0[k]+half+1, xi0[k]-half:xi0[k]+half+1]
        masked_core[k] = np.sum(box == 0)
    is_point = (cs > 0.8) & (snr > 100)
    peak = core_peak(sci, xx, yy, half=2, idx=np.where(is_point)[0])
    sat_peak_level, onset_mag = detect_saturation(mag, peak, masked_core, is_point)
    peak_saturated = np.isfinite(peak) & (peak >= sat_peak_level)
    saturated = (masked_core > 0) | peak_saturated
    print(f'  saturation: masked-core={int((masked_core>0).sum())}  '
          f'peak-plateau={int(peak_saturated.sum())}  '
          f'(onset_mag={onset_mag if onset_mag is None else round(onset_mag,2)})')

    # candidate stars = base morphology cuts only; every exclusion below is a
    # GATE with a recorded reason (hostgalxy structure)
    cand = (base & (fr > cut) & (fr < med + 2.5*std0) & (ell_arr < 0.20))
    midx = np.where(cand)[0]
    print(f'  candidates after locus/ellipticity cuts: {len(midx)}')
    if len(midx) < 10:
        sys.exit(f'only {len(midx)} candidates — too sparse for a PSF model')

    # ── stamp-quality gate-1 (catalog/image detectors) ──
    Rs = PSF_SIZE // 2
    # (1) edge: near frame border or SExtractor edge flag
    edge_b = ((xx[midx] < Rs) | (xx[midx] > nx-Rs) | (yy[midx] < Rs)
              | (yy[midx] > ny-Rs) | ((flg[midx] & 8) > 0))
    # (2) nbr: SPACE rule — any other source within 10 x PSF_FWHM
    R_nbr = NBR_ISO_FWHM * psf_fwhm_est
    _ok = np.isfinite(xx) & np.isfinite(yy)
    tree = cKDTree(np.column_stack([xx[_ok], yy[_ok]]))
    nbr_d = np.full(len(midx), np.inf)
    _q = np.isfinite(xx[midx]) & np.isfinite(yy[midx])
    if _q.any():
        nbr_d[_q] = tree.query(
            np.column_stack([xx[midx][_q], yy[midx][_q]]), k=2)[0][:, 1]
    nbr_b = nbr_d < R_nbr
    print(f'  isolation: R_nbr = {NBR_ISO_FWHM:.0f} x {psf_fwhm_est:.2f} px '
          f'= {R_nbr:.1f} px = {R_nbr*PIX_HST:.2f}"')
    # (4) bad-pix: masked/zero pixels in the VIGNET footprint
    xiv = np.clip(np.round(xx[midx]).astype(int), VIG_HALF, nx-VIG_HALF-1)
    yiv = np.clip(np.round(yy[midx]).astype(int), VIG_HALF, ny-VIG_HALF-1)
    vmask_b = np.zeros(len(midx), bool)
    for i in range(len(midx)):
        box = sci[yiv[i]-VIG_HALF:yiv[i]+VIG_HALF+1,
                  xiv[i]-VIG_HALF:xiv[i]+VIG_HALF+1]
        vmask_b[i] = np.sum(box == 0) > 5
    # (8) blend: >=2 local maxima in the minimally-smoothed raw core stamp
    Hc = 17
    rc = np.hypot(*np.mgrid[-Hc:Hc+1, -Hc:Hc+1]); em = rc > 14; smr = rc < 15
    peak_b = np.zeros(len(midx), bool)
    for i in range(len(midx)):
        xi = int(round(xx[midx[i]] - 1)); yi = int(round(yy[midx[i]] - 1))
        if yi-Hc < 0 or xi-Hc < 0 or yi+Hc+1 > ny or xi+Hc+1 > nx:
            continue
        cc = np.nan_to_num(sci[yi-Hc:yi+Hc+1, xi-Hc:xi+Hc+1].astype(float))
        ss = gaussian_filter(cc, 0.4)
        bgv = np.median(ss[em]); nzv = 1.4826*np.median(np.abs(ss[em]-bgv)) + 1e-9
        loc = ss == maximum_filter(ss, size=3)
        peak_b[i] = int((loc & (ss > bgv+8*nzv) & (ss > 0.04*ss.max()) & smr).sum()) >= 2
    # (16) sat
    sat_b = saturated[midx]

    g1_bad = edge_b | nbr_b | vmask_b | peak_b | sat_b
    g1_reason = (edge_b.astype(int) + 2*nbr_b.astype(int) + 4*vmask_b.astype(int)
                 + 8*peak_b.astype(int) + 16*sat_b.astype(int))
    midx1 = midx[~g1_bad]
    print(f'  gate-1: edge {int(edge_b.sum())}, nbr {int(nbr_b.sum())}, '
          f'bad-pix {int(vmask_b.sum())}, blend {int(peak_b.sum())}, '
          f'sat {int(sat_b.sum())} → {len(midx1)} survive')
    if len(midx1) < 10:
        sys.exit(f'only {len(midx1)} stars survive gate-1 — no PSF model')

    # ── PSFEx pass-2a: provisional PSF (gates the asym/outer detectors) ──
    fwhm_lo = max(0.8, 0.6 * psf_fwhm_est)
    fwhm_hi = 2.5 * psf_fwhm_est
    star_cat = out / f'stars_{args.tile}.fits'
    hcat[2].data = obj[midx1]
    hcat.writeto(star_cat, overwrite=True)
    print('\n── PSFEx pass-2a (provisional, for asym/outer gates) ──', flush=True)
    psf = run_psfex(star_cat, fwhm_lo, fwhm_hi, ['-CHECKIMAGE_TYPE', 'NONE'])

    # ── gate-2: (32) asymmetry + (64) outer companion vs provisional PSF ──
    _hp = fits.open(psf)[1]; _hd = _hp.header
    _mk = np.asarray(_hp.data['PSF_MASK'][0], float)
    _dg = int(_hd.get('POLDEG1', 0))
    _x0 = float(_hd.get('POLZERO1', 0.0)); _xs = float(_hd.get('POLSCAL1', 1.0))
    _y0 = float(_hd.get('POLZERO2', 0.0)); _ys = float(_hd.get('POLSCAL2', 1.0))
    _ps = float(_hd.get('PSF_SAMP', 1.0))
    if _mk.ndim == 2:
        _mk = _mk[None, ...]

    def _psf_at(x, y):
        dx = (x-_x0)/_xs; dy = (y-_y0)/_ys; t = []
        for jj in range(_dg+1):
            for ii in range(_dg+1-jj):
                t.append(dx**ii * dy**jj)
        p = np.tensordot(np.array(t), _mk, axes=(0, 0))
        return zoom(p, PSF_SIZE/p.shape[0], order=3) if abs(_ps-1.0) > 1e-3 else p

    _Hp = _psf_at(float(np.median(xx[midx1])), float(np.median(yy[midx1]))).shape[0]//2
    _gy, _gx = np.mgrid[-_Hp:_Hp+1, -_Hp:_Hp+1]; _rd = np.hypot(_gx, _gy)
    _fm = _rd < 16; _ap = _rd < 15; _ann = (_rd >= 12) & (_rd <= _Hp)
    _has_ann = int(_ann.sum()) >= 8
    asym_val = np.full(len(midx1), np.nan)
    outer_b = np.zeros(len(midx1), bool)
    for i in range(len(midx1)):
        # SExtractor X/Y_IMAGE are 1-indexed; numpy cutout is 0-indexed
        X = xx[midx1[i]] - 1.0; Y = yy[midx1[i]] - 1.0
        xi = int(round(X)); yi = int(round(Y))
        if yi-_Hp < 0 or xi-_Hp < 0 or yi+_Hp+1 > ny or xi+_Hp+1 > nx:
            continue
        c = np.nan_to_num(sci[yi-_Hp:yi+_Hp+1, xi-_Hp:xi+_Hp+1].astype(float))
        P = ndshift(_psf_at(X, Y), (Y-yi, X-xi), order=3)
        g2y, g2x = np.gradient(P)
        co = np.linalg.lstsq(np.vstack([P[_fm], g2x[_fm], g2y[_fm],
                                        np.ones(_fm.sum())]).T,
                             c[_fm], rcond=None)[0]
        Rfull = c - (co[0]*P + co[1]*g2x + co[2]*g2y + co[3])
        flux = float(co[0]*P[_ap].sum())
        if flux <= 0:
            continue
        Rr = Rfull * _ap
        hh = abs(float(Rr[_gx > 0].sum() - Rr[_gx < 0].sum())) / flux
        vv = abs(float(Rr[_gy > 0].sum() - Rr[_gy < 0].sum())) / flux
        asym_val[i] = max(hh, vv)
        if _has_ann:
            Rsm = gaussian_filter(Rfull, 1.0)
            _av = Rsm[_ann]
            nz = 1.4826*float(np.median(np.abs(_av - np.median(_av)))) + 1e-9
            outer_b[i] = (float(_av.max()) / nz) > 8.0
    _fin = np.isfinite(asym_val)
    asym_thr = max(0.05, float(np.percentile(asym_val[_fin], 98))) \
        if int(_fin.sum()) >= 50 else 0.05
    asym_b = _fin & (asym_val > asym_thr)
    post_bad = asym_b | outer_b
    post_reason = (32*asym_b.astype(int) + 64*outer_b.astype(int))
    midx2 = midx1[~post_bad]
    print(f'  gate-2: asym {int(asym_b.sum())}@{asym_thr*100:.1f}%, '
          f'outer {int(outer_b.sum())} → {len(midx2)} model stars')
    if len(midx2) < 5:
        sys.exit(f'only {len(midx2)} model stars survive the gate — no PSF model')

    # persist rejects (bits 1 edge / 2 nbr / 4 bad-pix / 8 blend / 16 sat /
    # 32 asym / 64 outer) — consumed by 56_'s raw mosaics (red frame + reason)
    rej = np.concatenate([midx[g1_bad], midx1[post_bad]])
    rreason = np.concatenate([g1_reason[g1_bad],
                              post_reason[post_bad]]).astype('int16')
    fits.BinTableHDU.from_columns([
        fits.Column(name='X_IMAGE',  format='E', array=xx[rej]),
        fits.Column(name='Y_IMAGE',  format='E', array=yy[rej]),
        fits.Column(name='MAG_AUTO', format='E', array=mag[rej]),
        fits.Column(name='REJ_REASON', format='I', array=rreason),
    ]).writeto(out / f'stars_rejected_{args.tile}.fits', overwrite=True)
    print(f'  wrote stars_rejected_{args.tile}.fits ({len(rej)} rejects)')

    # ── PSFEx pass-2b: final model + aligned OUTCAT ──
    hcat[2].data = obj[midx2]
    hcat.writeto(star_cat, overwrite=True)
    print(f'  wrote {star_cat.name} ({len(midx2)} model stars)')
    print('\n── PSFEx pass-2b (final) ──', flush=True)
    print(f'  SAMPLE_FWHMRANGE = {fwhm_lo:.2f},{fwhm_hi:.2f} px')
    t0 = time.time()
    psf = run_psfex(star_cat, fwhm_lo, fwhm_hi,
                    ['-OUTCAT_TYPE', 'FITS_LDAC',
                     '-OUTCAT_NAME', str(out / f'outcat_{args.tile}.fits'),
                     '-CHECKIMAGE_TYPE', 'NONE'])
    ph = fits.open(psf)[1].header
    print(f'  wall {time.time()-t0:.1f}s')
    print(f'  PSF model: chi2={ph.get("CHI2",-1):.3f}  '
          f'FWHM={ph.get("PSF_FWHM",-1):.2f} px  accepted={ph.get("ACCEPTED","?")}')

    (out / f'psf_{args.tile}.meta.json').write_text(json.dumps({
        'created_utc_iso': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'instrument': args.instrument, 'tile': args.tile,
        'zp_ab': zp, 'pixel_scale_arcsec': PIX_HST,
        'sci_source_file': str(img), 'sci_source_ext': 0,
        'psf_fwhm_est_px': float(psf_fwhm_est),
        'sample_fwhmrange': [float(fwhm_lo), float(fwhm_hi)],
        'stellar_locus_px': float(med), 'locus_std_px': float(std0),
        'locus_mad_sigma_px': float(mad),
        'artifact_cut_px': float(cut),
        'nbr_iso_fwhm': NBR_ISO_FWHM,
        'r_nbr_px': float(R_nbr), 'r_nbr_arcsec': float(R_nbr*PIX_HST),
        'n_candidates': int(len(midx)),
        'n_psf_stars': int(len(midx2)),
        'n_model_stars': int(len(midx2)),
        'n_rejected_gate': int(len(rej)),
        'gate_counts': {'edge': int(edge_b.sum()), 'nbr': int(nbr_b.sum()),
                        'bad_pix': int(vmask_b.sum()), 'blend': int(peak_b.sum()),
                        'sat': int(sat_b.sum()), 'asym': int(asym_b.sum()),
                        'outer': int(outer_b.sum())},
        'asym_threshold': float(asym_thr),
        'n_saturated_excluded': int(saturated.sum()),
        'sat_peak_level': (None if not np.isfinite(sat_peak_level) else float(sat_peak_level)),
        'sat_onset_mag': onset_mag,
        'n_peak_saturated': int(peak_saturated.sum()),
        'psf_chi2': float(ph.get('CHI2', -1)),
        'psf_fwhm_px': float(ph.get('PSF_FWHM', -1)),
        'psf_accepted': int(ph.get('ACCEPTED', -1)),
        'psf_model': str(psf),
        'recipe': 'hostgalxy-consistent v2: 10xFWHM isolation + 7-bit stamp gate',
    }, indent=2))
    print(f'\n[save] {out / f"psf_{args.tile}.meta.json"}')
    print('=== HST ACS/WFC F814W PSF build complete ===')


if __name__ == '__main__':
    main()
