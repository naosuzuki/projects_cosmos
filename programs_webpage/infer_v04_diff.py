"""v04: Difference imaging detector.

For each candidate position, use HST F814W as a 2005-2008 SN-free template
(pre-dates JWST + Euclid epochs). For each modern-epoch science band
(JWST F115W/F150W/F277W/F444W, Euclid VIS, NIR-Y/J/H), do:

  1. Reproject HST cutout onto the science image's WCS grid.
  2. PSF-match the SHARPER of (HST, science) up to the wider PSF using
     a Gaussian kernel: σ_kernel² = σ_wide² − σ_sharp².
  3. Scale HST to match the science image's background level in an
     annulus around the candidate (captures galaxy colour roughly).
  4. Subtract: diff = science − scale·HST_matched.
  5. Aperture-phot the diff at the candidate position; require positive
     residual ≥ 5σ AND PSF morphology (G1-G4 on the diff).

Real modern-epoch transients should appear as a strong positive PSF in
the difference; HST-SN (130972/371996/471959) won't — they're in the
template — but those 3 are not the target of v04.

Output: csvfiles_sn/_partial/infer_v04_diff_<band>.parquet per band.
        csvfiles_sn/tbl_sn_candidates_v04.csv via consolidate_v04.py.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, os, math
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, set_start_method
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR = CSV_DIR / "_partial"
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"
CUT_PIX  = 64
SCI_BANDS = [
    ("jwst", "F115W", "f115w", 0.057),
    ("jwst", "F150W", "f150w", 0.057),
    ("jwst", "F277W", "f277w", 0.130),
    ("jwst", "F444W", "f444w", 0.160),
    ("euclid","VIS",  "VIS",   0.194),
    ("euclid","Y",    "NIR-Y", 0.524),
    ("euclid","J",    "NIR-J", 0.537),
    ("euclid","H",    "NIR-H", 0.567),
]
FWHM_HST_AS = 0.134


def log(msg, prefix="V04"):
    print(f"[{time.strftime('%H:%M:%S')}] {prefix}: {msg}", flush=True)


def worker_diff(args):
    """One worker handles one (hst_tile, sci_tile, mission). For EACH science
    band: reproject the whole HST tile onto the science grid ONCE, PSF-match
    once, then extract per-source cutouts from the in-memory arrays and do
    the difference + aperture photometry. This makes reproject O(tiles) not
    O(sources) — the key speedup over the per-source version."""
    import warnings; warnings.filterwarnings("ignore")
    import sys, os, time, math
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
    import recut_3_hst as R_local
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import gaussian_filter
    from reproject import reproject_interp
    import numpy as np

    (hst_tile, sci_tile, sources, mission) = args
    out_rows = []
    hst_path = f"{R_local.HST_DIR}/acs_I_030mas_{hst_tile}_sci.fits"
    if not os.path.exists(hst_path):
        return out_rows
    try:
        hst_h = fits.open(hst_path, memmap=True)
        hst_hdu = hst_h["SCI"] if "SCI" in [x.name for x in hst_h] else hst_h[0]
        hst_wcs = WCS(hst_hdu.header)
        hst_data = hst_hdu.data
    except Exception as e:
        print(f"[v04] open HST failed {hst_tile}: {e}", flush=True)
        return out_rows

    for band_info in SCI_BANDS:
        m, blabel, R_band, fwhm_sci_as = band_info
        if m != mission: continue
        if mission == "jwst":
            sci_path = R_local.resolve_jwst_path(sci_tile, R_band.lower())
        else:
            sci_path = R_local.resolve_euclid_path(sci_tile, R_band)
        if not sci_path or not os.path.exists(sci_path):
            continue
        try:
            sci_h = fits.open(sci_path, memmap=True)
            sci_hdu = sci_h["SCI"] if "SCI" in [x.name for x in sci_h] else sci_h[0]
            sci_wcs = WCS(sci_hdu.header)
            sci_data = sci_hdu.data   # KEEP as memmap — slice before astype (per CLAUDE.md)
            sci_hdr = sci_hdu.header
            sci_ps = float(np.sqrt(np.abs(np.linalg.det(sci_wcs.pixel_scale_matrix)))) * 3600.0
        except Exception as e:
            print(f"[v04] open {mission}/{blabel} failed {sci_tile}: {e}", flush=True)
            continue

        from scipy.ndimage import map_coordinates
        ny, nx = sci_data.shape[-2:]
        hny, hnx = hst_data.shape[-2:]

        # PSF-match parameters (applied to the small per-source cutouts below)
        sig_sci_pix = fwhm_sci_as / sci_ps / 2.3548
        sig_hst_pix = FWHM_HST_AS / sci_ps / 2.3548
        if fwhm_sci_as > FWHM_HST_AS:
            k_hst = math.sqrt(max(0.0, sig_sci_pix**2 - sig_hst_pix**2)); k_sci = 0.0
        else:
            k_hst = 0.0; k_sci = math.sqrt(max(0.0, sig_hst_pix**2 - sig_sci_pix**2))

        half = CUT_PIX // 2
        aper_r = fwhm_sci_as / sci_ps
        r_in   = 2.0 * fwhm_sci_as / sci_ps
        r_out  = 3.5 * fwhm_sci_as / sci_ps
        yy, xx = np.indices((CUT_PIX, CUT_PIX))
        rr = np.sqrt((xx - half)**2 + (yy - half)**2)
        in_ap = rr <= aper_r
        ann   = (rr >= r_in) & (rr <= r_out)
        n_ap  = int(in_ap.sum())
        # sci cutout pixel grid (relative), reused per source via integer shift
        gx = xx.ravel(); gy = yy.ravel()

        bunit = (sci_hdr.get("BUNIT") or "").strip()
        is_mjy = (mission == "jwst" or "MJy/sr" in bunit)
        pix_sr = sci_hdr.get("PIXAR_SR") or ((sci_ps / 206265.0) ** 2)
        zp = (sci_hdr.get("ZP") or sci_hdr.get("MAGZP") or sci_hdr.get("PHOTZP") or 23.9)

        try:
            ras  = np.array([s[1] for s in sources])
            decs = np.array([s[2] for s in sources])
            sx, sy = sci_wcs.all_world2pix(ras, decs, 0)
            # ---- BATCHED per-source affine: map sci-pixel -> HST-pixel via a
            # local 2x2 Jacobian computed from finite differences (5 vectorised
            # WCS calls total instead of a 4096-point transform per source). ----
            hcx, hcy = hst_wcs.all_world2pix(ras, decs, 0)               # HST coord of center
            wra_dx, wdec_dx = sci_wcs.all_pix2world(sx + 1.0, sy, 0)
            hx_dx, hy_dx = hst_wcs.all_world2pix(wra_dx, wdec_dx, 0)
            wra_dy, wdec_dy = sci_wcs.all_pix2world(sx, sy + 1.0, 0)
            hx_dy, hy_dy = hst_wcs.all_world2pix(wra_dy, wdec_dy, 0)
            # Jacobian columns: d(hx,hy)/d(sx) and d(hx,hy)/d(sy)
            Jxx = hx_dx - hcx; Jyx = hy_dx - hcy   # d/d sx
            Jxy = hx_dy - hcx; Jyy = hy_dy - hcy   # d/d sy
        except Exception:
            try: sci_h.close()
            except Exception: pass
            continue

        for si, (gidx, ra, dec) in enumerate(sources):
            cx = sx[si]; cy = sy[si]
            if not (np.isfinite(cx) and np.isfinite(cy)): continue
            ix = int(round(cx)); iy = int(round(cy))
            x0 = ix - half; y0 = iy - half
            if x0 < 0 or y0 < 0 or x0+CUT_PIX > nx or y0+CUT_PIX > ny:
                continue
            sci_arr = sci_data[y0:y0+CUT_PIX, x0:x0+CUT_PIX].astype(np.float32)
            if not np.any(sci_arr): continue
            # Map each sci cutout pixel to HST pixel via the local affine:
            #   hst = hst_center + J @ (sci_pix - sci_center)
            if not (np.isfinite(hcx[si]) and np.isfinite(Jxx[si])): continue
            dsx = (gx + x0) - cx; dsy = (gy + y0) - cy
            hx = hcx[si] + Jxx[si]*dsx + Jxy[si]*dsy
            hy = hcy[si] + Jyx[si]*dsx + Jyy[si]*dsy
            # Sample HST at those fractional pixel coords via map_coordinates.
            # Load a bounding-box HST region into memory first.
            hxmin = int(np.floor(np.nanmin(hx))) - 2; hxmax = int(np.ceil(np.nanmax(hx))) + 2
            hymin = int(np.floor(np.nanmin(hy))) - 2; hymax = int(np.ceil(np.nanmax(hy))) + 2
            if hxmax <= 0 or hymax <= 0 or hxmin >= hnx or hymin >= hny:
                continue
            hxmin = max(0, hxmin); hymin = max(0, hymin)
            hxmax = min(hnx, hxmax); hymax = min(hny, hymax)
            if hxmax - hxmin < 2 or hymax - hymin < 2: continue
            hst_region = hst_data[hymin:hymax, hxmin:hxmax].astype(np.float32)
            coords = np.vstack([hy - hymin, hx - hxmin])
            hst_samp = map_coordinates(hst_region, coords, order=1,
                                       mode="constant", cval=0.0).reshape(CUT_PIX, CUT_PIX)
            hst_arr = np.nan_to_num(hst_samp, nan=0.0)
            if not np.any(hst_arr): continue
            # PSF-match the small cutouts
            if k_hst > 0: hst_arr = gaussian_filter(hst_arr, k_hst)
            if k_sci > 0: sci_arr = gaussian_filter(sci_arr, k_sci)
            if ann.sum() < 5: continue
            hh = hst_arr[ann]; ss = sci_arr[ann]
            good = np.abs(hh) > 1e-12
            if good.sum() < 3:
                scale = 0.0
            else:
                ratios = ss[good] / hh[good]
                ratios = ratios[np.isfinite(ratios)]
                scale = float(np.median(ratios)) if len(ratios) else 0.0
                scale = max(-10.0, min(10.0, scale))
            diff = sci_arr - scale * hst_arr
            flux = float(np.nansum(diff[in_ap]))
            sigma = float(np.nanstd(diff[ann])) * math.sqrt(n_ap)
            snr = flux / (sigma + 1e-30)
            if is_mjy:
                f_jy = flux * float(pix_sr) * 1.0e6
                mag = (-2.5 * math.log10(f_jy / 3631.0)) if f_jy > 0 else -1.0
            else:
                mag = (float(zp) - 2.5 * math.log10(flux)) if flux > 0 else -1.0
            out_rows.append({
                "idx": int(gidx), "band": blabel,
                "diff_flux": float(flux), "diff_snr": float(snr),
                "diff_mag": float(mag) if (mag is not None and mag != -1.0) else -1.0,
                "scale": float(scale),
            })
        try: sci_h.close()
        except Exception: pass
    try: hst_h.close()
    except Exception: pass
    return out_rows


def main():
    t0 = time.time()
    log("=== v04 difference imaging ===")
    lk = pq.read_table(LOOK)
    pid = np.array(lk["primary_id"].to_pylist(), dtype=object)
    ra  = lk["ra"].to_numpy(zero_copy_only=False)
    dec = lk["dec"].to_numpy(zero_copy_only=False)
    th  = np.array(lk["tile_hst"].to_pylist(), dtype=object)
    tj  = np.array(lk["tile_jwst"].to_pylist(), dtype=object)
    te  = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(lk["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(lk["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    N = len(pid)
    full = in_h & in_j & in_e
    log(f"full-coverage sources: {int(full.sum()):,}")

    # JWST jobs grouped by (hst_tile, jwst_tile)
    jwst_groups = defaultdict(list)
    eu_groups   = defaultdict(list)
    for i in np.where(full)[0]:
        if not th[i] or not tj[i] or not te[i]: continue
        jwst_groups[(str(th[i]), str(tj[i]))].append((int(i), float(ra[i]), float(dec[i])))
        eu_groups[(str(th[i]),  str(te[i]))].append((int(i), float(ra[i]), float(dec[i])))
    log(f"JWST groups: {len(jwst_groups)}   Euclid groups: {len(eu_groups)}")

    jobs = []
    for (htile, jtile), srcs in jwst_groups.items():
        jobs.append((htile, jtile, srcs, "jwst"))
    for (htile, etile), srcs in eu_groups.items():
        jobs.append((htile, etile, srcs, "euclid"))
    log(f"Dispatching {len(jobs)} (HST,sci,mission) jobs")

    PART_DIR.mkdir(parents=True, exist_ok=True)
    # Output accumulators per band
    acc = {b[1]: {"idx": [], "diff_flux": [], "diff_snr": [], "diff_mag": [], "scale": []}
           for b in SCI_BANDS}

    try: set_start_method("spawn", force=True)
    except RuntimeError: pass
    n_workers = max(2, os.cpu_count() // 2)
    log(f"Pool workers: {n_workers}")
    t_par = time.time()
    done = 0
    with Pool(processes=n_workers) as pool:
        for rows in pool.imap_unordered(worker_diff, jobs, chunksize=1):
            for r in rows:
                b = r["band"]
                acc[b]["idx"].append(r["idx"])
                acc[b]["diff_flux"].append(r["diff_flux"])
                acc[b]["diff_snr"].append(r["diff_snr"])
                acc[b]["diff_mag"].append(r["diff_mag"])
                acc[b]["scale"].append(r["scale"])
            done += 1
            if done % 10 == 0 or done == len(jobs):
                log(f"  [{done}/{len(jobs)}] elapsed={time.time()-t_par:.1f}s")
                # Flush partials so the consolidator can run in parallel
                for b in acc:
                    if not acc[b]["idx"]: continue
                    table = pa.table({k: np.array(v) for k, v in acc[b].items()})
                    tmp = PART_DIR / f"infer_v04_diff_{b}.parquet.tmp"
                    pq.write_table(table, tmp, compression="zstd")
                    tmp.replace(PART_DIR / f"infer_v04_diff_{b}.parquet")

    # Final flush
    for b in acc:
        if not acc[b]["idx"]: continue
        table = pa.table({k: np.array(v) for k, v in acc[b].items()})
        pq.write_table(table, PART_DIR / f"infer_v04_diff_{b}.parquet", compression="zstd")
    log(f"=== v04 inference done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
