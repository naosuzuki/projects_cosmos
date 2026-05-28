"""Step 3 v05: training set with SYNTHETIC SN INJECTIONS.

Key v05 change vs v03: instead of relying on 60 rotation+mag-scaled
augmentations of the same 3-17 real SNe (which the CNN ended up
memorising, hence the 1/17 in-sky recovery in v03), v05 injects
THOUSANDS of synthetic point sources at random sky positions in real
candidate cutouts. This gives the CNN INDEPENDENT positive examples
spanning the full mag/host-context distribution.

Per survey:
  * Real SNe: ALL 17 known positions, in EVERY survey they're covered by.
    Treat them as positives in every survey so the CNN learns to detect
    SN signal regardless of which discovery telescope flagged it.
  * Synthetic injections: 2000 per survey. Pick a random non-SN source,
    take its cutout, ADD a Gaussian PSF at the centre with mag drawn from
    dN/dm ∝ 10^(0.4·m) in the survey's mag range.
  * Negatives: 5000 random non-SN sources (no injection).

Outputs:
  csvfiles_sn/training_set_v05.npz
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv, time, math, os
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R

CSV_DIR     = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
LOOKUP_SN   = CSV_DIR / "lookup_sn17_v32.csv"
TBL_SN      = CSV_DIR / "tbl_sn17_v32.csv"
FITS_LOOKUP = CSV_DIR / "fits_lookup_v03.parquet"
OUT_NPZ     = CSV_DIR / "training_set_v05.npz"

CUT_PIX     = 64
N_INJ       = 2000        # synthetic injections per survey
N_NEG       = 5000
RNG_SEED    = 105
N_WORKERS   = max(2, cpu_count() // 2)

SURVEY_BANDS = {
    "hst":  [("F814W", (22.0, 27.0))],
    "jwst": [("F115W", (22.0, 27.2)), ("F150W", (22.0, 27.1)),
             ("F277W", (22.0, 27.3)), ("F444W", (22.0, 27.6))],
    "vis":  [("VIS",   (21.0, 26.0))],
    "nisp": [("Y",     (19.2, 24.2)), ("J", (19.3, 24.3)), ("H", (19.2, 24.2))],
}
BAND_R = {"F814W":"F814W", "F115W":"f115w", "F150W":"f150w", "F277W":"f277w", "F444W":"f444w",
          "VIS":"VIS", "Y":"NIR-Y", "J":"NIR-J", "H":"NIR-H"}
# Per-band FWHM in arcsec → for synthetic PSF injection
FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}
# Per-band ZP for converting mag→flux
ZP_FALLBACK = {"F814W": 25.937,  # ACS PHOTFLAM-derived ABMAG_ZP
               "VIS": 23.9,
               "Y": 23.9, "J": 23.9, "H": 23.9,
               # JWST uses MJy/sr — handled in inject function
              }


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def resolve_path(survey, R_band, tile):
    if survey == "hst":
        return f"{R.HST_DIR}/acs_I_030mas_{tile}_sci.fits"
    if survey == "jwst":
        return R.resolve_jwst_path(tile, R_band.lower())
    return R.resolve_euclid_path(tile, R_band)


def cutout_at(sci_data, wcs, ra, dec, n=CUT_PIX):
    try:
        sx, sy = wcs.all_world2pix(ra, dec, 0)
    except Exception:
        return np.full((n, n), np.nan, dtype=np.float32)
    sx = float(sx); sy = float(sy)
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return np.full((n, n), np.nan, dtype=np.float32)
    half = n // 2
    cx = int(round(sx)); cy = int(round(sy))
    ny, nx = sci_data.shape[-2:]
    x0 = cx - half; x1 = x0 + n; y0 = cy - half; y1 = y0 + n
    if x0 >= 0 and y0 >= 0 and x1 <= nx and y1 <= ny:
        return sci_data[y0:y1, x0:x1].astype(np.float32)
    out = np.full((n, n), np.nan, dtype=np.float32)
    xs0 = max(0, -x0); ys0 = max(0, -y0)
    xs1 = n - max(0, x1 - nx); ys1 = n - max(0, y1 - ny)
    src_x0 = max(0, x0); src_y0 = max(0, y0)
    src_x1 = min(nx, x1); src_y1 = min(ny, y1)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[ys0:ys1, xs0:xs1] = sci_data[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    return out


def worker_extract(args):
    import warnings; warnings.filterwarnings("ignore")
    import sys, os
    sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
    import recut_3_hst as R_local
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    survey, band, R_band, tile, reqs = args
    path = resolve_path(survey, R_band, tile)
    if not path or not os.path.exists(path):
        return [(r["key"], None, None, None) for r in reqs]
    try:
        with fits.open(path, memmap=True) as h:
            sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
            wcs = WCS(sci_hdu.header)
            hdr = sci_hdu.header
            # Pixel scale in arcsec
            ps = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
            data = sci_hdu.data
            try:
                ras  = np.array([r["ra"]  for r in reqs])
                decs = np.array([r["dec"] for r in reqs])
                _, ys = wcs.all_world2pix(ras, decs, 0)
                order = np.argsort(np.where(np.isfinite(ys), ys, 0.0))
            except Exception:
                order = np.arange(len(reqs))
            return [(reqs[i]["key"], cutout_at(data, wcs, reqs[i]["ra"], reqs[i]["dec"]),
                     ps, dict(hdr)) for i in order]
    except Exception:
        return [(r["key"], None, None, None) for r in reqs]


def sample_mag_dndm(lo, hi, n, rng, alpha=0.4):
    u = rng.uniform(0, 1, n)
    bias = 10**(alpha * lo)
    span = 10**(alpha * hi) - bias
    return np.log10(bias + u * span) / alpha


def mag_to_flux(mag, hdr_dict, kind, pixel_scale_as):
    """Convert AB mag to instrument flux units appropriate to this image."""
    if kind == "jwst" or (hdr_dict.get("BUNIT", "").strip() == "MJy/sr"):
        # f_lam in MJy/sr; pix area = PIXAR_SR
        pix_sr = hdr_dict.get("PIXAR_SR") or ((pixel_scale_as / 206265.0) ** 2)
        f_jy = 3631.0 * 10**(-0.4 * mag)         # AB
        f_mjy_per_sr = f_jy / (float(pix_sr) * 1.0e6)
        return float(f_mjy_per_sr)
    # HST / Euclid: header ZP works on the raw pixel sum in aperture
    zp = (hdr_dict.get("ABMAG_ZP") or hdr_dict.get("ABMAGZP") or
          hdr_dict.get("ZP") or hdr_dict.get("MAGZP") or hdr_dict.get("PHOTZP"))
    if zp is None:
        zp = ZP_FALLBACK.get("F814W" if kind == "hst" else "VIS", 23.9)
    return 10**(0.4 * (float(zp) - mag))


def inject_psf(cutout, hdr_dict, kind, pixel_scale_as, fwhm_as, mag, rng):
    """Add a Gaussian PSF to `cutout` (in-place return new) at the centre
    with the given AB magnitude. Returns the modified cutout."""
    if cutout is None or not np.any(np.isfinite(cutout)):
        return None
    total_flux = mag_to_flux(mag, hdr_dict, kind, pixel_scale_as)
    sigma_pix = fwhm_as / pixel_scale_as / 2.3548
    n = cutout.shape[0]
    cy, cx = n // 2, n // 2
    # 3-sigma kernel
    rk = max(3, int(round(4 * sigma_pix)))
    yy, xx = np.mgrid[-rk:rk+1, -rk:rk+1]
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma_pix**2))
    g *= total_flux / g.sum()
    out = cutout.copy()
    y0 = max(0, cy - rk); y1 = min(n, cy + rk + 1)
    x0 = max(0, cx - rk); x1 = min(n, cx + rk + 1)
    gy0 = y0 - (cy - rk); gy1 = gy0 + (y1 - y0)
    gx0 = x0 - (cx - rk); gx1 = gx0 + (x1 - x0)
    out[y0:y1, x0:x1] += g[gy0:gy1, gx0:gx1].astype(np.float32)
    return out


def main():
    t_start = time.time()
    log("=== Step 3 v05: synthetic SN injection training set ===")
    log(f"Output: {OUT_NPZ}")
    log(f"Workers: {N_WORKERS}")

    with LOOKUP_SN.open() as f:
        sn_rows = list(csv.DictReader(f))
    with TBL_SN.open() as f:
        sn_mags_tbl = {int(r["id"]): r for r in csv.DictReader(f)}
    log(f"Loaded {len(sn_rows)} known SN positions")

    # ALL_BANDS: (survey, band, R_band, mag_range)
    ALL_BANDS = []
    for s, bs in SURVEY_BANDS.items():
        for b, mr in bs:
            ALL_BANDS.append((s, b, BAND_R[b], mr))

    # Real-SN requests: ALL 17 SNe in EVERY survey
    real_reqs = []
    for r in sn_rows:
        sid = int(r["id"])
        ra, dec = float(r["sn_ra"]), float(r["sn_dec"])
        tiles = {"hst": r["hst"], "jwst": r["jwst"],
                 "vis": r["euclid"], "nisp": r["euclid"]}
        for (s, b, rb, _) in ALL_BANDS:
            t = tiles[s]
            if not t: continue
            real_reqs.append(dict(key=("real", sid, s, b), survey=s, band=b,
                                  R_band=rb, tile=t, ra=ra, dec=dec))

    # Negative pool: full-coverage candidates not near known SNe
    lk = pq.read_table(FITS_LOOKUP).to_pandas()
    eligible = (lk["ra"].notna() & lk["dec"].notna() &
                lk["in_hst"] & lk["in_jwst"] & lk["in_euclid"]).values
    cos_d = np.cos(np.deg2rad(lk["dec"].values))
    excl = np.zeros(len(lk), dtype=bool)
    for r in sn_rows:
        sra, sdec = float(r["sn_ra"]), float(r["sn_dec"])
        dra  = (lk["ra"].values  - sra)  * cos_d * 3600
        ddec = (lk["dec"].values - sdec) * 3600
        excl |= (np.sqrt(dra**2 + ddec**2) < 2.0)
    pool = np.where(eligible & ~excl)[0]
    rng = np.random.default_rng(RNG_SEED)
    # Pull a big random pool — INJECT positives use it AND negatives use it
    needed = N_INJ + N_NEG
    sel = rng.choice(pool, size=needed, replace=False)
    inj_idx = sel[:N_INJ]; neg_idx = sel[N_INJ:N_INJ+N_NEG]
    log(f"Negative pool eligible={len(pool):,}.  Injection positions: {len(inj_idx):,}, "
        f"negatives: {len(neg_idx):,}")

    # Build INJECTION requests (same source position used to extract host cutout)
    inj_reqs = []
    for i, ii_ in enumerate(inj_idx):
        ii = int(ii_)
        tiles = {"hst": str(lk["tile_hst"].iloc[ii]) if lk["tile_hst"].iloc[ii] else None,
                 "jwst": str(lk["tile_jwst"].iloc[ii]) if lk["tile_jwst"].iloc[ii] else None,
                 "vis": str(lk["tile_euclid"].iloc[ii]) if lk["tile_euclid"].iloc[ii] else None,
                 "nisp": str(lk["tile_euclid"].iloc[ii]) if lk["tile_euclid"].iloc[ii] else None}
        for (s, b, rb, _) in ALL_BANDS:
            t = tiles[s]
            if not t: continue
            inj_reqs.append(dict(key=("inj", i, s, b), survey=s, band=b, R_band=rb,
                                 tile=t, ra=float(lk["ra"].iloc[ii]),
                                 dec=float(lk["dec"].iloc[ii])))

    # Negative requests
    neg_reqs = []
    for i, ii_ in enumerate(neg_idx):
        ii = int(ii_)
        tiles = {"hst": str(lk["tile_hst"].iloc[ii]) if lk["tile_hst"].iloc[ii] else None,
                 "jwst": str(lk["tile_jwst"].iloc[ii]) if lk["tile_jwst"].iloc[ii] else None,
                 "vis": str(lk["tile_euclid"].iloc[ii]) if lk["tile_euclid"].iloc[ii] else None,
                 "nisp": str(lk["tile_euclid"].iloc[ii]) if lk["tile_euclid"].iloc[ii] else None}
        for (s, b, rb, _) in ALL_BANDS:
            t = tiles[s]
            if not t: continue
            pid = str(lk["primary_id"].iloc[ii])
            neg_reqs.append(dict(key=("neg", pid, s, b), survey=s, band=b, R_band=rb,
                                 tile=t, ra=float(lk["ra"].iloc[ii]),
                                 dec=float(lk["dec"].iloc[ii])))

    all_reqs = real_reqs + inj_reqs + neg_reqs
    log(f"Total IO requests: {len(all_reqs):,} "
        f"(real={len(real_reqs):,}, inj={len(inj_reqs):,}, neg={len(neg_reqs):,})")

    by_tile = defaultdict(list)
    for r in all_reqs:
        by_tile[(r["survey"], r["band"], r["R_band"], r["tile"])].append(r)
    jobs = [(s, b, rb, t, lst) for (s, b, rb, t), lst in by_tile.items()]
    log(f"Unique (survey, band, tile) groups: {len(jobs)}")

    cutouts = {}   # key -> (arr, pixel_scale, hdr_dict)
    t_io = time.time()
    with Pool(processes=N_WORKERS) as pool_:
        for ji, results in enumerate(pool_.imap_unordered(worker_extract, jobs)):
            for key, arr, ps, hdr_dict in results:
                cutouts[key] = (arr, ps, hdr_dict)
            if (ji + 1) % 30 == 0 or ji == len(jobs) - 1:
                log(f"  [{ji+1}/{len(jobs)}] elapsed={time.time()-t_io:.1f}s")
    log(f"IO done in {time.time()-t_io:.1f}s")

    # Assemble per-survey training arrays
    rng2 = np.random.default_rng(RNG_SEED + 1)
    out = {s: dict(X=[], y=[], group=[], id=[], mag=[]) for s in SURVEY_BANDS}
    for survey, bands_specs in SURVEY_BANDS.items():
        bands = [b for b, _ in bands_specs]
        n_ch = len(bands)
        kind = {"hst":"hst","jwst":"jwst","vis":"euclid","nisp":"euclid"}[survey]

        # REAL SN positives (all 17 in this survey if covered)
        for r in sn_rows:
            sid = int(r["id"])
            stk = np.full((n_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            ok = False
            for ci, b in enumerate(bands):
                key = ("real", sid, survey, b)
                if key in cutouts and cutouts[key][0] is not None:
                    stk[ci] = cutouts[key][0]; ok = True
            if not ok: continue
            out[survey]["X"].append(stk)
            out[survey]["y"].append(1)
            out[survey]["group"].append("known")
            out[survey]["id"].append(str(sid))
            out[survey]["mag"].append(-1.0)
            # also a few rotated versions
            for k_rot in (1, 2, 3):
                arr2 = np.rot90(stk, k=k_rot, axes=(1,2)).copy()
                out[survey]["X"].append(arr2)
                out[survey]["y"].append(1); out[survey]["group"].append("known_rot")
                out[survey]["id"].append(str(sid)); out[survey]["mag"].append(-1.0)

        # SYNTHETIC INJECTIONS
        for i in range(N_INJ):
            # Build a stack: each channel = host cutout + injected PSF at center
            stk = np.full((n_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            inj_ok = False
            lo, hi = bands_specs[0][1]
            m_target = float(sample_mag_dndm(lo, hi, 1, rng2, alpha=0.4)[0])
            for ci, b in enumerate(bands):
                key = ("inj", i, survey, b)
                if key not in cutouts: continue
                arr, ps, hdr_dict = cutouts[key]
                if arr is None or ps is None: continue
                mag = m_target  # use same mag across bands (assumes flat SED)
                stk[ci] = inject_psf(arr, hdr_dict, kind, ps, FWHM_AS[b], mag, rng2)
                inj_ok = True
            if not inj_ok: continue
            out[survey]["X"].append(stk)
            out[survey]["y"].append(1)
            out[survey]["group"].append("inj")
            out[survey]["id"].append(f"inj_{i}")
            out[survey]["mag"].append(m_target)

        # NEGATIVES (no injection)
        neg_by_pid = defaultdict(dict)
        for r in neg_reqs:
            if r["survey"] != survey: continue
            neg_by_pid[r["key"][1]][r["band"]] = r["key"]
        for pid, band_map in neg_by_pid.items():
            stk = np.full((n_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            ok = False
            for ci, b in enumerate(bands):
                if b in band_map and band_map[b] in cutouts:
                    arr = cutouts[band_map[b]][0]
                    if arr is not None:
                        stk[ci] = arr; ok = True
            if not ok: continue
            out[survey]["X"].append(stk)
            out[survey]["y"].append(0)
            out[survey]["group"].append("neg")
            out[survey]["id"].append(pid)
            out[survey]["mag"].append(-1.0)

        n_total = len(out[survey]["X"])
        n_pos = sum(out[survey]["y"])
        log(f"  [{survey}] X count={n_total:,}  pos={n_pos:,}  neg={n_total-n_pos:,}")

    log(f"Writing {OUT_NPZ} ...")
    save_dict = {}
    for s in SURVEY_BANDS:
        if not out[s]["X"]: continue
        save_dict[f"X_{s}"]     = np.stack(out[s]["X"]).astype(np.float32)
        save_dict[f"y_{s}"]     = np.array(out[s]["y"], dtype=np.int8)
        save_dict[f"{s}_group"] = np.array(out[s]["group"])
        save_dict[f"{s}_id"]    = np.array(out[s]["id"])
        save_dict[f"{s}_mag"]   = np.array(out[s]["mag"], dtype=np.float32)
    np.savez_compressed(OUT_NPZ, **save_dict)
    log(f"Wrote {OUT_NPZ}  ({OUT_NPZ.stat().st_size/1e6:.1f} MB)")
    log(f"=== Step 3 v05 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
