#!/usr/bin/env python
"""
psf_gate.py — hostgalxy-consistent stamp-quality gate for the space-telescope
PSF builders (ported from projects_hsc/hostgalxy/programs/psf/build_psf.py).

Bits (REJ_REASON in stars_rejected_<suffix>.fits):
   1 edge      near frame border (< psf_size/2) or SExtractor edge flag
   2 nbr       catalogue neighbour within nbr_iso_fwhm x PSF_FWHM (space: 10)
   4 bad-pix   masked/zero pixels in the VIGNET footprint (caller-provided)
   8 blend     >=2 local maxima in the raw core stamp (sub-FWHM blends)
  16 sat       saturated (caller-provided: masked core / peak plateau)
  32 asym      integrated L/R / T/B residual imbalance vs a provisional PSF
  64 outer     >8 sigma residual blob in the r=12..psf/2 annulus

Usage (see 54_step3a_build_psf_model_hst.py for the inline original):
    g1 = gate1(sci, xx, yy, flg, midx, psf_fwhm_est, psf_size,
               sat_flags=saturated[midx], badpix_flags=gap_masked[midx],
               nbr_iso_fwhm=10.0)
    ... run provisional PSFEx on midx[~g1['bad']] ...
    g2 = gate2(sci, xx, yy, midx1, psf_path, psf_size)
    write_rejects(out_fits, xx, yy, mag, midx, g1, midx1, g2)
"""
from __future__ import annotations
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, maximum_filter, shift as ndshift, zoom
from scipy.spatial import cKDTree


def gate1(sci, xx, yy, flg, midx, psf_fwhm_est, psf_size,
          sat_flags, badpix_flags, nbr_iso_fwhm=10.0):
    """Catalog/image detectors (bits 1,2,4,8,16) on candidate indices midx."""
    ny, nx = sci.shape
    Rs = psf_size // 2
    edge_b = ((xx[midx] < Rs) | (xx[midx] > nx - Rs) | (yy[midx] < Rs)
              | (yy[midx] > ny - Rs) | ((flg[midx] & 8) > 0))
    R_nbr = nbr_iso_fwhm * psf_fwhm_est
    _ok = np.isfinite(xx) & np.isfinite(yy)
    tree = cKDTree(np.column_stack([xx[_ok], yy[_ok]]))
    nbr_d = np.full(len(midx), np.inf)
    _q = np.isfinite(xx[midx]) & np.isfinite(yy[midx])
    if _q.any():
        nbr_d[_q] = tree.query(
            np.column_stack([xx[midx][_q], yy[midx][_q]]), k=2)[0][:, 1]
    nbr_b = nbr_d < R_nbr
    vmask_b = np.asarray(badpix_flags, bool)
    # (8) blend: >=2 local maxima in the minimally-smoothed raw core stamp
    Hc = 17
    rc = np.hypot(*np.mgrid[-Hc:Hc+1, -Hc:Hc+1]); em = rc > 14; smr = rc < 15
    peak_b = np.zeros(len(midx), bool)
    for i in range(len(midx)):
        xi = int(round(xx[midx[i]] - 1)); yi = int(round(yy[midx[i]] - 1))
        if yi - Hc < 0 or xi - Hc < 0 or yi + Hc + 1 > ny or xi + Hc + 1 > nx:
            continue
        cc = np.nan_to_num(sci[yi-Hc:yi+Hc+1, xi-Hc:xi+Hc+1].astype(float))
        ss = gaussian_filter(cc, 0.4)
        bgv = np.median(ss[em])
        nzv = 1.4826 * np.median(np.abs(ss[em] - bgv)) + 1e-9
        loc = ss == maximum_filter(ss, size=3)
        peak_b[i] = int((loc & (ss > bgv + 8*nzv)
                         & (ss > 0.04*ss.max()) & smr).sum()) >= 2
    sat_b = np.asarray(sat_flags, bool)
    bad = edge_b | nbr_b | vmask_b | peak_b | sat_b
    reason = (edge_b.astype(int) + 2*nbr_b.astype(int) + 4*vmask_b.astype(int)
              + 8*peak_b.astype(int) + 16*sat_b.astype(int))
    return dict(bad=bad, reason=reason, R_nbr=float(R_nbr),
                counts=dict(edge=int(edge_b.sum()), nbr=int(nbr_b.sum()),
                            bad_pix=int(vmask_b.sum()), blend=int(peak_b.sum()),
                            sat=int(sat_b.sum())))


def psf_evaluator(psf_path, psf_size):
    """Return psf_at(x, y) evaluating the PSFEx model at NATIVE psf_size px."""
    hp = fits.open(psf_path)[1]
    hd = hp.header
    mk = np.asarray(hp.data['PSF_MASK'][0], float)
    dg = int(hd.get('POLDEG1', 0))
    x0 = float(hd.get('POLZERO1', 0.0)); xs = float(hd.get('POLSCAL1', 1.0))
    y0 = float(hd.get('POLZERO2', 0.0)); ys = float(hd.get('POLSCAL2', 1.0))
    ps = float(hd.get('PSF_SAMP', 1.0))
    if mk.ndim == 2:
        mk = mk[None, ...]

    def psf_at(x, y):
        dx = (x - x0) / xs; dy = (y - y0) / ys; t = []
        for jj in range(dg + 1):
            for ii in range(dg + 1 - jj):
                t.append(dx**ii * dy**jj)
        p = np.tensordot(np.array(t), mk, axes=(0, 0))
        return (zoom(p, psf_size / p.shape[0], order=3)
                if abs(ps - 1.0) > 1e-3 else p)
    return psf_at


def gate2(sci, xx, yy, midx1, psf_path, psf_size):
    """PSF-residual detectors vs a provisional model (bits 32 asym, 64 outer)."""
    ny, nx = sci.shape
    psf_at = psf_evaluator(psf_path, psf_size)
    Hp = psf_at(float(np.median(xx[midx1])), float(np.median(yy[midx1]))).shape[0] // 2
    gy, gx = np.mgrid[-Hp:Hp+1, -Hp:Hp+1]
    rd = np.hypot(gx, gy)
    fm = rd < 16; ap = rd < 15; ann = (rd >= 12) & (rd <= Hp)
    has_ann = int(ann.sum()) >= 8
    asym_val = np.full(len(midx1), np.nan)
    outer_b = np.zeros(len(midx1), bool)
    for i in range(len(midx1)):
        # SExtractor X/Y_IMAGE are 1-indexed; numpy cutout is 0-indexed
        X = xx[midx1[i]] - 1.0; Y = yy[midx1[i]] - 1.0
        xi = int(round(X)); yi = int(round(Y))
        if yi - Hp < 0 or xi - Hp < 0 or yi + Hp + 1 > ny or xi + Hp + 1 > nx:
            continue
        c = np.nan_to_num(sci[yi-Hp:yi+Hp+1, xi-Hp:xi+Hp+1].astype(float))
        P = ndshift(psf_at(X, Y), (Y - yi, X - xi), order=3)
        g2y, g2x = np.gradient(P)
        co = np.linalg.lstsq(np.vstack([P[fm], g2x[fm], g2y[fm],
                                        np.ones(fm.sum())]).T,
                             c[fm], rcond=None)[0]
        Rfull = c - (co[0]*P + co[1]*g2x + co[2]*g2y + co[3])
        flux = float(co[0] * P[ap].sum())
        if flux <= 0:
            continue
        Rr = Rfull * ap
        hh = abs(float(Rr[gx > 0].sum() - Rr[gx < 0].sum())) / flux
        vv = abs(float(Rr[gy > 0].sum() - Rr[gy < 0].sum())) / flux
        asym_val[i] = max(hh, vv)
        if has_ann:
            Rsm = gaussian_filter(Rfull, 1.0)
            av = Rsm[ann]
            nz = 1.4826 * float(np.median(np.abs(av - np.median(av)))) + 1e-9
            outer_b[i] = (float(av.max()) / nz) > 8.0
    fin = np.isfinite(asym_val)
    asym_thr = max(0.05, float(np.percentile(asym_val[fin], 98))) \
        if int(fin.sum()) >= 50 else 0.05
    asym_b = fin & (asym_val > asym_thr)
    bad = asym_b | outer_b
    reason = 32*asym_b.astype(int) + 64*outer_b.astype(int)
    return dict(bad=bad, reason=reason, asym_thr=float(asym_thr),
                counts=dict(asym=int(asym_b.sum()), outer=int(outer_b.sum())))


def write_rejects(out_fits, xx, yy, mag, midx, g1, midx1, g2):
    """Persist all gated stars (bits 1..64) for the QA mosaics."""
    rej = np.concatenate([midx[g1['bad']], midx1[g2['bad']]])
    rreason = np.concatenate([g1['reason'][g1['bad']],
                              g2['reason'][g2['bad']]]).astype('int16')
    fits.BinTableHDU.from_columns([
        fits.Column(name='X_IMAGE',  format='E', array=xx[rej]),
        fits.Column(name='Y_IMAGE',  format='E', array=yy[rej]),
        fits.Column(name='MAG_AUTO', format='E', array=mag[rej]),
        fits.Column(name='REJ_REASON', format='I', array=rreason),
    ]).writeto(out_fits, overwrite=True)
    return len(rej)
