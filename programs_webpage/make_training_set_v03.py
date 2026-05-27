"""Step 3 v03: SN training set with magnitude-scaled augmentation + dN/dm prior.

Inputs:
  csvfiles_sn/lookup_sn17_v32.csv            (17 known SN positions + discovery telescope)
  csvfiles_sn/tbl_sn17_v32.csv               (17 known SN with per-band magnitudes)
  csvfiles_sn/sn_candidates_v03.parquet      (1.11M v03 candidates — used for negatives)
  csvfiles_sn/fits_lookup_v03.parquet        (per-source tile IDs)

Output:
  csvfiles_sn/training_set_v03.npz           (per-survey X, y, metadata)

Design (v03 specific changes vs v01):

  1. Per-survey CORRECT labels (NISP-discovery for Euclid SN per user 2026-05-27):
       hst:  3 HST-discovered SNe (130972, 371996, 471959)
       jwst: 13 JWST-discovered SNe
       vis:  0 (no SN ever discovered in VIS) → skip CNN
       nisp: 1 EUCLID-discovered SN (63924)
  2. Magnitude-scaled augmentation: for each real SN cutout, generate N variants
     by multiplying pixel values by 10^(0.4·(mag_real - mag_target)), with
     mag_target drawn from dN/dm ∝ 10^(0.4·m) in per-band range:
       HST F814W:     22 – 27.0
       JWST F115W:    22 – 27.2
       JWST F150W:    22 – 27.1
       JWST F277W:    22 – 27.3
       JWST F444W:    22 – 27.6
       Euclid NIR-Y:  19.2 – 24.2
       Euclid NIR-J:  19.3 – 24.3
       Euclid NIR-H:  19.2 – 24.2
       (VIS skipped; 21 – 26 range noted for inference)
  3. Rotation (4 angles) × flip (2) × mag scaling (60 random samples per SN)
     gives 8 × 60 = 480 augmented variants per known SN per band.
  4. Negatives: 5000 per survey, random non-SN candidates (excluded from
     2″ of any known SN). Cutouts centered on the source catalog position.
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

CSV_DIR    = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
LOOKUP_SN  = CSV_DIR / "lookup_sn17_v32.csv"
TBL_SN     = CSV_DIR / "tbl_sn17_v32.csv"
CANDIDATES = CSV_DIR / "sn_candidates_v03.parquet"
FITS_LOOKUP = CSV_DIR / "fits_lookup_v03.parquet"
OUT_NPZ    = CSV_DIR / "training_set_v03.npz"

CUT_PIX    = 64
N_NEG      = 5000
N_AUG_MAG  = 60         # mag-scale variants per (SN, rot, flip)
N_ROT_FLIP = 8          # 4 rotations × 2 flips
RNG_SEED   = 42
N_WORKERS  = max(2, cpu_count() // 2)

# Per-survey channel definitions + mag range for dN/dm-weighted sampling
SURVEY_BANDS = {
    "hst":  [("F814W", (22.0, 27.0))],
    "jwst": [("F115W", (22.0, 27.2)), ("F150W", (22.0, 27.1)),
              ("F277W", (22.0, 27.3)), ("F444W", (22.0, 27.6))],
    "vis":  [("VIS",   (21.0, 26.0))],   # no CNN trained for VIS — kept for IO consistency only
    "nisp": [("Y",     (19.2, 24.2)), ("J", (19.3, 24.3)), ("H", (19.2, 24.2))],
}
ALL_BANDS = []   # (survey, band_label, R_band_string, mag_range)
BAND_R = {"F814W":"F814W", "F115W":"f115w", "F150W":"f150w", "F277W":"f277w", "F444W":"f444w",
          "VIS":"VIS", "Y":"NIR-Y", "J":"NIR-J", "H":"NIR-H"}
for s, bs in SURVEY_BANDS.items():
    for b, mr in bs:
        ALL_BANDS.append((s, b, BAND_R[b], mr))

# Discovery telescope → which survey gets this SN as a POSITIVE
SURVEY_TEL = {"hst": {"HST"}, "jwst": {"JWST"}, "vis": set(), "nisp": {"EUCLID"}}


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
    x0 = cx - half; x1 = x0 + n
    y0 = cy - half; y1 = y0 + n
    if x0 >= 0 and y0 >= 0 and x1 <= nx and y1 <= ny:
        # Slice FIRST, then astype (master CLAUDE.md §4)
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
    """Process one (survey, band, R_band, tile) group of cutout requests."""
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
        return [(r["array_key"], np.full((CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)) for r in reqs]
    try:
        with fits.open(path, memmap=True) as h:
            sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
            wcs = WCS(sci_hdu.header)
            # Sort by Y for sequential disk-page access
            try:
                ras  = np.array([r["ra"]  for r in reqs])
                decs = np.array([r["dec"] for r in reqs])
                _, sys_ = wcs.all_world2pix(ras, decs, 0)
                order = np.argsort(np.where(np.isfinite(sys_), sys_, 0.0))
            except Exception:
                order = np.arange(len(reqs))
            data = sci_hdu.data
            return [(reqs[i]["array_key"], cutout_at(data, wcs, reqs[i]["ra"], reqs[i]["dec"]))
                    for i in order]
    except Exception:
        return [(r["array_key"], np.full((CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)) for r in reqs]


def sample_mag_dndm(lo, hi, n, rng, alpha=0.4):
    """Draw n magnitudes from p(m) ∝ 10^(α m) in [lo, hi]. α=0.4 gives realistic SN dN/dm."""
    # CDF inversion: F(m) = (10^(α m) - 10^(α lo)) / (10^(α hi) - 10^(α lo))
    u = rng.uniform(0, 1, n)
    bias = 10**(alpha * lo)
    span = 10**(alpha * hi) - bias
    return np.log10(bias + u * span) / alpha


def main():
    t_start = time.time()
    log(f"=== Step 3 v03: training set with dN/dm-weighted mag-scaling ===")
    log(f"Output: {OUT_NPZ}")
    log(f"Workers: {N_WORKERS}")

    # Load known SNe + their mags
    with LOOKUP_SN.open() as f:
        sn_rows = list(csv.DictReader(f))
    with TBL_SN.open() as f:
        sn_mags = {int(r["id"]): r for r in csv.DictReader(f)}
    log(f"Loaded {len(sn_rows)} known SNe; {len(sn_mags)} with measured mags")

    # Per-survey discovery SN ids (CORRECTED labels)
    tel_to_survey = {"HST":"hst", "JWST":"jwst", "EUCLID":"nisp"}  # per user 2026-05-27
    survey_known = {s: [] for s in SURVEY_BANDS}
    for r in sn_rows:
        tel = r["telescope"].strip().upper()
        s = tel_to_survey.get(tel)
        if s and s in survey_known:
            survey_known[s].append(int(r["id"]))
    log("Per-survey discovery SN counts (CORRECTED):")
    for s, ids in survey_known.items():
        log(f"  {s}: {len(ids)} SNe ({ids})")

    # === Stage 1: cutouts for known SNe (one cutout per SN per band) ===
    # (Need real measurements as templates for mag-scaling)
    known_reqs = []
    for sn_id in {int(r["id"]) for r in sn_rows}:
        row = next(r for r in sn_rows if int(r["id"]) == sn_id)
        ra = float(row["sn_ra"]); dec = float(row["sn_dec"])
        tiles = {"hst": row["hst"], "jwst": row["jwst"], "vis": row["euclid"], "nisp": row["euclid"]}
        for (survey, band, R_band, _) in ALL_BANDS:
            t = tiles[survey]
            if not t: continue
            known_reqs.append(dict(group="known", id=sn_id, survey=survey, band=band,
                                   R_band=R_band, tile=t, ra=ra, dec=dec))

    # === Stage 2: cutouts for negative samples ===
    lk = pq.read_table(FITS_LOOKUP).to_pandas()
    finite = lk["ra"].notna() & lk["dec"].notna()
    # exclude near known SNe
    cos_d = np.cos(np.deg2rad(lk["dec"].values))
    excl = np.zeros(len(lk), dtype=bool)
    for r in sn_rows:
        sra, sdec = float(r["sn_ra"]), float(r["sn_dec"])
        dra  = (lk["ra"].values  - sra)  * cos_d * 3600
        ddec = (lk["dec"].values - sdec) * 3600
        excl |= (np.sqrt(dra**2 + ddec**2) < 2.0)
    # Want negatives covered by all 3 surveys (full v03 inference universe)
    eligible = finite.values & ~excl & lk["in_hst"].values & lk["in_jwst"].values & lk["in_euclid"].values
    idx = np.where(eligible)[0]
    rng = np.random.default_rng(RNG_SEED + 1)
    if len(idx) > N_NEG:
        idx = rng.choice(idx, size=N_NEG, replace=False)
    log(f"Negative pool: {int(eligible.sum()):,} eligible, sampling {len(idx):,}")
    neg_reqs = []
    for i in idx:
        ii = int(i)
        tiles = {"hst": str(lk["tile_hst"].iloc[ii]) if lk["tile_hst"].iloc[ii] else None,
                 "jwst": str(lk["tile_jwst"].iloc[ii]) if lk["tile_jwst"].iloc[ii] else None,
                 "vis": str(lk["tile_euclid"].iloc[ii]) if lk["tile_euclid"].iloc[ii] else None,
                 "nisp": str(lk["tile_euclid"].iloc[ii]) if lk["tile_euclid"].iloc[ii] else None}
        for (survey, band, R_band, _) in ALL_BANDS:
            tt = tiles[survey]
            if not tt: continue
            neg_reqs.append(dict(group="neg", id=str(lk["primary_id"].iloc[ii]),
                                 survey=survey, band=band, R_band=R_band,
                                 tile=tt, ra=float(lk["ra"].iloc[ii]),
                                 dec=float(lk["dec"].iloc[ii])))

    all_reqs = known_reqs + neg_reqs
    for i, r in enumerate(all_reqs):
        r["array_key"] = i
    log(f"Total IO requests: {len(all_reqs):,} (known={len(known_reqs):,}, neg={len(neg_reqs):,})")

    # Group by (survey, band, R_band, tile) for tile-major parallel IO
    by_tile = defaultdict(list)
    for r in all_reqs:
        by_tile[(r["survey"], r["band"], r["R_band"], r["tile"])].append(r)
    jobs = [(s, b, rb, t, lst) for (s, b, rb, t), lst in by_tile.items()]
    log(f"Unique (survey, band, tile) groups: {len(jobs)}")

    # Parallel IO
    t_io = time.time()
    cutouts = {}
    n_done = 0
    with Pool(processes=N_WORKERS) as pool:
        for ji, results in enumerate(pool.imap_unordered(worker_extract, jobs)):
            for key, arr in results:
                cutouts[key] = arr
            n_done += len(results)
            if (ji + 1) % 30 == 0 or ji == len(jobs) - 1:
                log(f"  [{ji+1}/{len(jobs)}] cutouts={n_done:,}  elapsed={time.time()-t_io:.1f}s")
    log(f"Parallel IO done in {time.time()-t_io:.1f}s")

    # === Stage 3: build per-survey training sets ===
    # For each survey:
    #   POSITIVES: for each discovery SN, take its cutouts in this survey's bands.
    #     Augment each: rotation/flip + mag scaling (dN/dm prior).
    #   NEGATIVES: use neg cutouts as-is in this survey's bands.
    log("Assembling per-survey training arrays ...")

    out = {s: dict(X=[], y=[], group=[], id=[], mag=[]) for s in SURVEY_BANDS if s != "vis"}
    # (skip vis — no CNN to train)

    rng2 = np.random.default_rng(RNG_SEED + 2)
    for survey, bands_specs in SURVEY_BANDS.items():
        if survey == "vis":
            log(f"  [{survey}] SKIPPING — no positives in training (0 SNe ever discovered in VIS)")
            continue
        bands = [b for b, _ in bands_specs]
        n_ch = len(bands)
        # Discovery SNe for this survey
        for sn_id in survey_known[survey]:
            sn_row = next((r for r in sn_rows if int(r["id"]) == sn_id), None)
            if sn_row is None: continue
            # Stack the SN's cutouts in this survey's bands
            stk = np.full((n_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            real_mag = {}
            for ci, b in enumerate(bands):
                # find cutout for (known, sn_id, survey, b)
                for r in known_reqs:
                    if r["id"] == sn_id and r["survey"] == survey and r["band"] == b:
                        stk[ci] = cutouts[r["array_key"]]
                        break
                # SN's measured mag in band b
                if sn_id in sn_mags:
                    try:
                        v = float(sn_mags[sn_id].get(f"mag_{b}", "-1"))
                        if v > 0: real_mag[b] = v
                    except (ValueError, TypeError):
                        pass
            if not np.any(np.isfinite(stk)):
                continue
            # Add the original (real) SN cutout as one positive sample
            out[survey]["X"].append(stk.copy())
            out[survey]["y"].append(1)
            out[survey]["group"].append("known")
            out[survey]["id"].append(str(sn_id))
            out[survey]["mag"].append(real_mag.get(bands[0], -1.0))

            # Augment: rotation×flip × mag-scaling
            mag_range_per_band = {b: mr for b, mr in bands_specs}
            for _ in range(N_AUG_MAG):
                # draw target mag in the first band's range
                lo, hi = mag_range_per_band[bands[0]]
                m_target = float(sample_mag_dndm(lo, hi, 1, rng2, alpha=0.4)[0])
                # scale factor: SN flux multiplied by 10^(0.4 * (m_real - m_target))
                # If real_mag known, use it; else assume real_mag = lo+1 (typical bright known SN)
                m_real = real_mag.get(bands[0], lo + 1.0)
                scale = 10**(0.4 * (m_real - m_target))
                # rotate/flip
                rf = int(rng2.integers(0, N_ROT_FLIP))
                arr = stk.copy()
                # Subtract local background per channel so we scale only the SN flux
                # (rough: use median of corner pixels as background estimate)
                for ci in range(n_ch):
                    if np.any(np.isfinite(arr[ci])):
                        bg = float(np.nanmedian(arr[ci, :8, :8]))
                        arr[ci] = (arr[ci] - bg) * scale + bg
                if rf & 4: arr = arr[:, :, ::-1].copy()
                k_rot = rf % 4
                if k_rot: arr = np.rot90(arr, k=k_rot, axes=(1, 2)).copy()
                out[survey]["X"].append(arr)
                out[survey]["y"].append(1)
                out[survey]["group"].append("aug")
                out[survey]["id"].append(str(sn_id))
                out[survey]["mag"].append(m_target)

        # Negatives for this survey
        neg_by_id = defaultdict(dict)
        for r in neg_reqs:
            if r["survey"] != survey: continue
            neg_by_id[r["id"]][r["band"]] = r["array_key"]
        for pid, band_map in neg_by_id.items():
            stk = np.full((n_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for ci, b in enumerate(bands):
                if b in band_map:
                    stk[ci] = cutouts[band_map[b]]
            if not np.any(np.isfinite(stk)):
                continue
            out[survey]["X"].append(stk)
            out[survey]["y"].append(0)
            out[survey]["group"].append("neg")
            out[survey]["id"].append(pid)
            out[survey]["mag"].append(-1.0)

        n_pos = sum(out[survey]["y"]); n_total = len(out[survey]["y"])
        log(f"  [{survey}] X count={n_total:,}  pos={n_pos:,}  neg={n_total-n_pos:,}")

    # Save
    log(f"Writing {OUT_NPZ} ...")
    save_dict = {}
    for s in ("hst", "jwst", "nisp"):
        if not out[s]["X"]: continue
        Xarr = np.stack(out[s]["X"]).astype(np.float32)
        save_dict[f"X_{s}"]      = Xarr
        save_dict[f"y_{s}"]      = np.array(out[s]["y"], dtype=np.int8)
        save_dict[f"{s}_group"]  = np.array(out[s]["group"])
        save_dict[f"{s}_id"]     = np.array(out[s]["id"])
        save_dict[f"{s}_mag"]    = np.array(out[s]["mag"], dtype=np.float32)
    np.savez_compressed(OUT_NPZ, **save_dict)
    size_mb = OUT_NPZ.stat().st_size / 1e6
    log(f"  wrote {size_mb:.1f} MB")
    log(f"=== Step 3 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
