#!/usr/bin/env python
"""
68_make_phot_psf.py — derive PHOTOMETRY-sized PSFEx models from the
Step 3a-① morphology models.

The 3a-① models keep 201–301-sample rasters so the NIRCam diffraction
spikes + wings live in the model (the Tanaka+2023 science requirement).
But SExtractor's per-source PSF fit costs ∝ raster area: with the big
stamps a single dual-image pass on one JWST tile burned >2 CPU-hours
with the catalog still empty (measured 2026-06-11).  Photometry only
needs the core: this tool crops every stars_<suffix>.psf to a
6×FWHM-diameter raster (odd, clamped to [25, 75] samples) and writes
stars_<suffix>_phot.psf alongside — same empirical model, same spatial
polynomial, truncated wings.

The wing flux lost by truncation is a constant fraction per band →
a per-band ZP offset in MAG_PSF, absorbed by the Step 4 empirical
anchor calibration (Gaia/2MASS), exactly like any aperture correction.

53_step3a_dual_photometry.py prefers stars_*_phot.psf when present.

Usage:
  ./68_make_phot_psf.py            # all models under photometry_v04
  ./68_make_phot_psf.py --force    # re-crop even if _phot.psf exists
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

WORK = Path('/Volumes/exdisk1/data/photometry_v04')
CLAMP = (25, 75)        # crop size clamp, in PSF samples (odd)


def crop_psf(src: Path, dst: Path) -> tuple[int, int, float]:
    """Crop the PSF_MASK raster to 6×FWHM diameter; return
    (old_size, new_size, kept_flux_fraction_of_mean_psf)."""
    with fits.open(src) as h:
        hdr = h[1].header.copy()
        data = h[1].data['PSF_MASK'][0]          # (ncoeff, ny, nx)
    ny, nx = data.shape[1], data.shape[2]
    fwhm_px = float(hdr['PSF_FWHM'])             # native image px
    samp = float(hdr['PSF_SAMP'])                # native px per sample
    size = int(round(6.0 * fwhm_px / samp))
    size = max(CLAMP[0], min(CLAMP[1], size))
    if size % 2 == 0:
        size += 1
    if size >= min(ny, nx):
        size = min(ny, nx) | 1                   # nothing to crop
    cy, cx = ny // 2, nx // 2
    half = size // 2
    cut = data[:, cy-half:cy+half+1, cx-half:cx+half+1].copy()
    # flux kept by the constant (mean-PSF) term — QA number only
    tot = float(np.abs(data[0]).sum())
    kept = float(np.abs(cut[0]).sum()) / tot if tot > 0 else 1.0

    col = fits.Column(name='PSF_MASK', format=f'{cut.size}E',
                      dim=f'({size}, {size}, {cut.shape[0]})',
                      array=cut.reshape(1, -1))
    hdu = fits.BinTableHDU.from_columns([col])
    for k in hdr:
        if k in ('XTENSION', 'BITPIX', 'PCOUNT', 'GCOUNT', 'TFIELDS',
                 'TTYPE1', 'TFORM1', 'TDIM1') or k.startswith('NAXIS'):
            continue
        try:
            hdu.header[k] = hdr[k]
        except Exception:
            pass
    hdu.header['PSFAXIS1'] = size
    hdu.header['PSFAXIS2'] = size
    hdu.header['EXTNAME'] = 'PSF_DATA'
    hdu.header['HISTORY'] = (f'cropped {nx}->{size} samples (6xFWHM) '
                             f'for photometry; 68_make_phot_psf.py')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(dst, overwrite=True)
    return nx, size, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    srcs = sorted(WORK.glob('*/*/psf/stars_*.psf'))
    srcs = [s for s in srcs if not s.stem.endswith('_phot')]
    print(f'{len(srcs)} morphology models found')
    done = skip = 0
    sizes = {}
    for s in srcs:
        d = s.with_name(s.stem + '_phot.psf')
        if d.exists() and not args.force:
            skip += 1
            continue
        try:
            old, new, kept = crop_psf(s, d)
        except Exception as e:
            print(f'  [fail] {s}: {type(e).__name__} {e}')
            continue
        inst = s.parts[-4]
        sizes.setdefault(inst, (old, new, kept))
        done += 1
    print(f'cropped {done}, skipped {skip}')
    for inst, (old, new, kept) in sorted(sizes.items()):
        print(f'  {inst:22s} {old:3d} -> {new:2d} samples  '
              f'core flux kept {kept:.3f}  (fit cost /{(old/new)**2:.0f})')


if __name__ == '__main__':
    main()
