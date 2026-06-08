"""Prototype: does a PSF-subtracted residual isolate on-host SNe?

For ONE JWST band+tile:
  1. Build an empirical PSF (EPSF) from bright, isolated stars in the tile.
  2. At each test position, forced-fit the EPSF (amplitude only, position fixed
     to the catalog SN coord) and subtract → residual.
  3. Compare residual at KNOWN on-host SN positions vs random galaxy positions.

If on-host SNe leave a clean positive PSF in the residual while galaxies leave
~flat residual, the residual is a strong CNN channel for the on-host wall.

This is a CHEAP concept test (one tile), not the full pipeline.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_cnnsn/src")
import recut_3_hst as R
from tile_lookup import TileResolver
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import EPSFBuilder, extract_stars
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    band, btile = "F150W", "A9"   # A9 hosts known SNe 318858, 320233
    res = TileResolver()
    _, path = res.resolve("jwst", band, 150.146, 1.866)
    log(f"band {band} tile {btile}: {Path(path).name}")
    with fits.open(path, memmap=True) as h:
        sci = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = WCS(sci.header)
        data = np.asarray(sci.data, dtype=np.float32)   # whole tile (one band, one time)
    ny, nx = data.shape
    log(f"tile shape {data.shape}")

    # --- 1. find bright isolated stars to build the EPSF ---
    mean, med, std = sigma_clipped_stats(data, sigma=3.0, maxiters=3)
    dao = DAOStarFinder(fwhm=3.0, threshold=50.0*std, sharplo=0.5, sharphi=1.0)
    src = dao(data - med)
    if src is None or len(src) < 20:
        log("too few bright stars; lowering threshold");
        dao = DAOStarFinder(fwhm=3.0, threshold=20.0*std); src = dao(data-med)
    src = src[(src["xcentroid"]>40)&(src["xcentroid"]<nx-40)&
              (src["ycentroid"]>40)&(src["ycentroid"]<ny-40)]
    # keep the brightest ~80, well-separated
    src.sort("flux"); src.reverse()
    stars_tbl = Table()
    stars_tbl["x"] = src["xcentroid"][:80]; stars_tbl["y"] = src["ycentroid"][:80]
    log(f"EPSF stars: {len(stars_tbl)}")
    nd = NDData(data=data - med)
    stars = extract_stars(nd, stars_tbl, size=25)
    t0=time.time()
    epsf, fitted = EPSFBuilder(oversampling=2, maxiters=8, progress_bar=False)(stars)
    log(f"EPSF built in {time.time()-t0:.1f}s, shape {epsf.data.shape}")

    # --- 2. forced PSF fit (amplitude only) + subtract at test positions ---
    from photutils.psf import PSFPhotometry, ImagePSF
    # build a fittable model from the EPSF data
    psf_model = ImagePSF(epsf.data, flux=1.0, x_0=0, y_0=0,
                         oversampling=epsf.oversampling)

    def residual_at(ra, dec, n=25, verbose=False):
        sx, sy = wcs.all_world2pix(ra, dec, 0); sx=float(sx); sy=float(sy)
        if not (40<sx<nx-40 and 40<sy<ny-40):
            if verbose: log(f"    off-tile sx={sx:.0f} sy={sy:.0f}")
            return None
        half=n//2; x0,y0=int(round(sx))-half,int(round(sy))-half
        cut = (data[y0:y0+n, x0:x0+n] - med).astype(np.float64)
        # forced fit: fix position to (sx-x0, sy-y0), fit flux only.
        # FRESH model each call (carry EPSF oversampling correctly).
        m = ImagePSF(epsf.data, flux=1.0, x_0=sx-x0, y_0=sy-y0,
                     oversampling=epsf.oversampling)
        m.x_0.fixed=True; m.y_0.fixed=True
        phot = PSFPhotometry(m, fit_shape=(7,7), aperture_radius=4)
        init = Table({"x_0":[sx-x0],"y_0":[sy-y0]})
        try:
            r = phot(cut, init_params=init)
            resid = phot.make_residual_image(cut)
            flux = float(r["flux_fit"][0])
            ferr = float(r["flux_err"][0]) if "flux_err" in r.colnames else np.nan
        except Exception as e:
            if verbose: log(f"    fit error: {type(e).__name__}: {e}")
            return None
        # central peak of cut vs residual (does subtraction remove a point source?)
        c=n//2; box=slice(c-3,c+4)
        cut_peak=float(np.nanmax(cut[box,box])); res_peak=float(np.nanmax(resid[box,box]))
        return dict(flux=flux, ferr=ferr, cut_peak=cut_peak, res_peak=res_peak,
                    removed_frac=1-res_peak/cut_peak if cut_peak>0 else np.nan)

    # known SNe in A9 (from master) + a few random galaxy positions in-tile
    known = {"318858":(150.185337,1.842646), "320233":(150.146358,1.866410)}
    log("\n=== forced PSF fit at KNOWN SN positions (tile A9) ===")
    for sn,(ra,dec) in known.items():
        r = residual_at(ra,dec,verbose=True)
        if r: log(f"  SN {sn}: fit_flux={r['flux']:.1f} cut_peak={r['cut_peak']:.3f} "
                  f"res_peak={r['res_peak']:.3f} removed={100*r['removed_frac']:.0f}%")
        else: log(f"  SN {sn}: off-tile or fit failed")

    # random galaxy positions: sample catalog sources in-tile (not the SNe)
    log("\n=== forced PSF fit at random in-tile positions (galaxies/blank) ===")
    rng=np.random.default_rng(1)
    for k in range(5):
        # random pixel → world
        px,py=rng.integers(100,nx-100),rng.integers(100,ny-100)
        ra,dec=wcs.all_pix2world(px,py,0); ra=float(ra);dec=float(dec)
        r=residual_at(ra,dec)
        if r: log(f"  rand{k}: fit_flux={r['flux']:.1f} cut_peak={r['cut_peak']:.3f} "
                  f"res_peak={r['res_peak']:.3f} removed={100*r['removed_frac']:.0f}%")

if __name__ == "__main__":
    main()
