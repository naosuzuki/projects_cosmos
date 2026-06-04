"""Apply 4 SN-position corrections to lookup_sn17_v31 → lookup_sn17_v32.

User-confirmed picks:
  ID=19931  band=F150W  direction=SE  expected sep ~0.17"  (only PSF-clean candidate)
  ID=53669  band=F115W  direction=SW  expected sep ~0.62"
  ID=63919  band=F115W  direction=SW  expected sep ~0.44"
  ID=430352 band=F115W  direction=SE  expected sep ~0.32"  (brighter of the two)

For each, scan DAO around the catalog (=host) position, find the candidate
that best matches the specified direction and approximate separation.
"""
import warnings; warnings.filterwarnings("ignore")
import csv, math, sys, shutil
from pathlib import Path
import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R

IN_LOOKUP  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/lookup_sn17_v31.csv")
OUT_LOOKUP = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/lookup_sn17_v32.csv")

# (cid, band, fwhm, direction, expected_sep, tolerance, sigma)
CORRECTIONS = [
    (19931,  "F150W", 0.057, "SE", 0.17, 0.15, 3.0),
    (53669,  "F115W", 0.057, "SW", 0.62, 0.20, 5.0),
    (63919,  "F115W", 0.057, "SW", 0.44, 0.20, 5.0),
    (430352, "F115W", 0.057, "SE", 0.32, 0.15, 5.0),
]
BOX_AS = 4.0


def in_direction(d_ra_as, d_dec_as, direction):
    """direction ∈ {N,S,E,W,NE,NW,SE,SW}."""
    ns = "N" if d_dec_as > 0.05 else ("S" if d_dec_as < -0.05 else "")
    ew = "E" if d_ra_as  > 0.05 else ("W" if d_ra_as  < -0.05 else "")
    if direction in ("N","S"): return ns == direction
    if direction in ("E","W"): return ew == direction
    return (ns + ew) == direction or (ew + ns) == direction


with IN_LOOKUP.open() as f:
    rows = list(csv.DictReader(f))

corrections_by_id = {c[0]: c for c in CORRECTIONS}

for r in rows:
    cid = int(r["id"])
    if cid not in corrections_by_id:
        continue
    _, band, fwhm, want_dir, want_sep, tol, sigma = corrections_by_id[cid]
    host_ra = float(r["host_ra"]); host_dec = float(r["host_dec"])
    sci_path = R._jwst_band_path_fn(band.lower())(dict(jwst=r["jwst"]))
    h = R._open_cached(sci_path)
    sci = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    wcs = R._wcs_cached(sci_path, sci)
    ps = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    fwhm_px = max(1.5, fwhm / ps)

    sx, sy = wcs.all_world2pix(host_ra, host_dec, 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci.data.shape[-2:]
    half = int(np.ceil(BOX_AS / ps)) + 1
    x0 = max(0, int(sx) - half); x1 = min(nx, int(sx) + half + 1)
    y0 = max(0, int(sy) - half); y1 = min(ny, int(sy) + half + 1)
    sub = sci.data[y0:y1, x0:x1].astype(np.float64)
    finite = np.isfinite(sub) & (sub != 0)
    _, bg_med, bg_std = sigma_clipped_stats(sub[finite], sigma=3.0, maxiters=3)
    res = DAOStarFinder(fwhm=fwhm_px, threshold=sigma*bg_std)(sub - bg_med)

    hpx, hpy = sx - x0, sy - y0
    matches = []
    if res is not None:
        for s in res:
            dx = float(s["xcentroid"]) - hpx
            dy = float(s["ycentroid"]) - hpy
            sep = math.hypot(dx, dy) * ps
            if abs(sep - want_sep) > tol: continue
            ra, dec = wcs.all_pix2world(float(s["xcentroid"]) + x0,
                                        float(s["ycentroid"]) + y0, 0)
            d_ra = (ra - host_ra) * 3600.0 * math.cos(math.radians(host_dec))
            d_dec = (dec - host_dec) * 3600.0
            if not in_direction(d_ra, d_dec, want_dir): continue
            matches.append(dict(ra=float(ra), dec=float(dec),
                                sep=sep, d_ra=d_ra, d_dec=d_dec,
                                peak=float(s["peak"]),
                                sharp=float(s["sharpness"]),
                                rnd1=float(s["roundness1"]),
                                rnd2=float(s["roundness2"]),
                                _diff=abs(sep - want_sep)))
    if matches:
        best = min(matches, key=lambda c: c["_diff"])
        old_ra, old_dec = float(r["sn_ra"]), float(r["sn_dec"])
        r["sn_ra"]  = f"{best['ra']:.8f}"
        r["sn_dec"] = f"{best['dec']:.8f}"
        print(f"  ID={cid}: {band} {want_dir} sep={best['sep']:.2f}\" "
              f"(want {want_sep:.2f}\")  peak={best['peak']:.3f}  "
              f"sharp={best['sharp']:+.2f} r1={best['rnd1']:+.2f} r2={best['rnd2']:+.2f}")
        print(f"           sn_ra  {old_ra:.6f} -> {best['ra']:.6f}  "
              f"(Δra = {(best['ra']-old_ra)*3600*math.cos(math.radians(old_dec)):+.3f}\")")
        print(f"           sn_dec {old_dec:.6f} -> {best['dec']:.6f}  "
              f"(Δdec = {(best['dec']-old_dec)*3600:+.3f}\")")
    else:
        print(f"  ID={cid}: NO MATCH for {band} {want_dir} sep~{want_sep:.2f}\"; "
              f"keeping v31 position")

# write new lookup
cols = ["id","telescope","sn_ra","sn_dec","host_ra","host_dec","hst","jwst","euclid"]
with OUT_LOOKUP.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in cols})
print(f"\nwrote {OUT_LOOKUP}  ({len(rows)} sources)")
