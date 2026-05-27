"""Step 3 v01: build SN training set (cutouts).

Inputs:
  csvfiles_sn/lookup_sn17_v32.csv    — 17 known SN positions
  csvfiles_sn/sn_candidates_v01.parquet — 1.12M candidates (negatives sampled here)
  csvfiles_sn/fits_lookup_v01.parquet   — per-source tile IDs

Output:
  csvfiles_sn/training_set_v01.npz   — compressed cutouts + labels

Design (IO-optimised):
  Tile-major outer loop: open each (survey, band, tile) FITS once, slice all
  required sources from it. This is the only place in the pipeline where we
  touch FITS for training data.

  Cutout = 64×64 px, native pixel scale. Centred on RA/Dec via the tile WCS.

  Positives:
    known_sn:    17 ground truth SN, one cutout per band (153 cutouts)
    augmented:   for each known SN, 8 rotations × 2 flips × 3 magnitude jitters
                 = 48 variants; cutouts stored per band → 17×48×9 = 7,344 cutouts
    psf_inject:  empirical PSF point sources injected at random nearby positions
                 inside a tile we already have open; ~50 per band
  Negatives:
    random:      ~10,000 candidate sources from sn_candidates_v01 (excluded
                 from 2" of any known SN), one cutout per band as available

All cutouts saved as float32 to keep size manageable. Missing-band cutouts
filled with NaN. Labels are integer arrays per survey.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv, time, random, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS

# repo path
sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R

CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
LOOKUP_SN = CSV_DIR / "lookup_sn17_v32.csv"
CANDIDATES = CSV_DIR / "sn_candidates_v01.parquet"
FITS_LOOKUP = CSV_DIR / "fits_lookup_v01.parquet"
OUT_NPZ = CSV_DIR / "training_set_v01.npz"
STATUS_LOG = CSV_DIR / "run_v01_status.log"

CUT_PIX = 64               # cutout pixel size per band
N_NEG = 10000              # number of negative candidate sources
RNG_SEED = 42

# Bands map: (survey_short, band_name_for_R, channel_index_in_survey, kind_for_R)
# Each survey stack defined separately so CNN inputs are clean.
HST_BANDS  = [("F814W", "F814W")]
JWST_BANDS = [("F115W", "f115w"), ("F150W", "f150w"), ("F277W", "f277w"), ("F444W", "f444w")]
VIS_BANDS  = [("VIS",   "VIS")]
NISP_BANDS = [("Y",     "NIR-Y"), ("J", "NIR-J"), ("H", "NIR-H")]

ALL_BANDS = []  # (survey, band_label, R_band, kind)
for b, br in HST_BANDS:  ALL_BANDS.append(("hst",   b, br, "hst"))
for b, br in JWST_BANDS: ALL_BANDS.append(("jwst",  b, br, "jwst"))
for b, br in VIS_BANDS:  ALL_BANDS.append(("vis",   b, br, "euclid"))
for b, br in NISP_BANDS: ALL_BANDS.append(("nisp",  b, br, "euclid"))


def log(msg):
    t = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with STATUS_LOG.open("a") as f:
        f.write(line + "\n")


def resolve_path(survey, R_band, tile):
    """Resolve FITS path for a given (survey, band, tile)."""
    if survey == "hst":
        return f"{R.HST_DIR}/acs_I_030mas_{tile}_sci.fits"
    if survey == "jwst":
        return R.resolve_jwst_path(tile, R_band.lower())
    if survey in ("vis", "nisp"):
        return R.resolve_euclid_path(tile, R_band)
    raise ValueError(f"unknown survey: {survey}")


def cutout_at(sci_hdu, wcs, ra, dec, n=CUT_PIX):
    """Slice a small cutout, centred on (ra, dec). Returns float32 array
    of shape (n, n), NaN-filled if out of bounds.
    Uses memmap slice + astype (the 80KB-not-30GB trick)."""
    try:
        sx, sy = wcs.all_world2pix(ra, dec, 0)
    except Exception:
        return np.full((n, n), np.nan, dtype=np.float32)
    sx = float(sx); sy = float(sy)
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return np.full((n, n), np.nan, dtype=np.float32)
    half = n // 2
    cx = int(round(sx)); cy = int(round(sy))
    ny, nx = sci_hdu.data.shape[-2:]
    x0 = cx - half; x1 = x0 + n
    y0 = cy - half; y1 = y0 + n
    # full overlap path
    if x0 >= 0 and y0 >= 0 and x1 <= nx and y1 <= ny:
        return sci_hdu.data[y0:y1, x0:x1].astype(np.float32)
    # partial overlap: build with NaN padding
    out = np.full((n, n), np.nan, dtype=np.float32)
    xs0 = max(0, -x0); ys0 = max(0, -y0)
    xs1 = n - max(0, x1 - nx); ys1 = n - max(0, y1 - ny)
    src_x0 = max(0, x0); src_y0 = max(0, y0)
    src_x1 = min(nx, x1); src_y1 = min(ny, y1)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[ys0:ys1, xs0:xs1] = sci_hdu.data[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    return out


def open_band_path(path):
    """Return (sci_hdu, wcs) using R's caches; resilient to missing files."""
    if path is None or not os.path.exists(path):
        return None, None
    try:
        h = R._open_cached(path)
        sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = R._wcs_cached(path, sci_hdu)
        return sci_hdu, wcs
    except Exception as e:
        log(f"  open failed: {Path(path).name}: {type(e).__name__}: {e}")
        return None, None


def build_requests_for_known_sn(lookup_rows):
    """For each known SN, request a cutout in every band, augmented set."""
    reqs = []   # each: dict with id, label, survey, band, tile, ra, dec, augment_id
    for row in lookup_rows:
        sn_id = int(row["id"])
        ra = float(row["sn_ra"]); dec = float(row["sn_dec"])
        tiles = dict(hst=row["hst"], jwst=row["jwst"], vis=row["euclid"], nisp=row["euclid"])
        for (survey, band_label, R_band, kind) in ALL_BANDS:
            t = tiles[survey]
            if not t or t in ("", "nan", "None"):
                continue
            reqs.append(dict(group="known_pos", id=sn_id, label=1,
                             survey=survey, band=band_label, R_band=R_band,
                             tile=t, ra=ra, dec=dec, aug=0,
                             augment_op="none"))
    return reqs


def build_augmented_requests(lookup_rows, n_aug=48):
    """Augmented positives: re-use known SN tiles, jitter position by ±0.05",
    keep them grouped per (survey,band,tile) for IO efficiency.
    The actual rotation/flip/mag-scale happens at consume time on the cutout
    pixels, so the FITS slice is identical to the known one — we just record
    augment_id so the consumer applies a deterministic transform per id."""
    reqs = []
    rng = np.random.default_rng(RNG_SEED)
    for row in lookup_rows:
        sn_id = int(row["id"])
        ra0 = float(row["sn_ra"]); dec0 = float(row["sn_dec"])
        tiles = dict(hst=row["hst"], jwst=row["jwst"], vis=row["euclid"], nisp=row["euclid"])
        for aug_i in range(n_aug):
            # jitter (in arcsec → degrees)
            d_ra_as  = rng.uniform(-0.05, 0.05)
            d_dec_as = rng.uniform(-0.05, 0.05)
            cos_d = math.cos(math.radians(dec0))
            ra = ra0 + d_ra_as / 3600.0 / cos_d
            dec = dec0 + d_dec_as / 3600.0
            # rotation + flip code in aug_id (0..7 = 4 rot × 2 flip)
            rot_flip = aug_i % 8
            for (survey, band_label, R_band, kind) in ALL_BANDS:
                t = tiles[survey]
                if not t or t in ("", "nan", "None"):
                    continue
                reqs.append(dict(group="aug_pos", id=sn_id, label=1,
                                 survey=survey, band=band_label, R_band=R_band,
                                 tile=t, ra=ra, dec=dec, aug=aug_i,
                                 augment_op=f"rotflip_{rot_flip}"))
    return reqs


def build_negative_requests(n_neg=N_NEG, exclude_radius_as=2.0):
    """Sample N random sources from the candidate catalog, NOT within
    `exclude_radius_as` of any known SN."""
    # Load fits_lookup_v01 to get tile assignments + ra/dec
    t = pq.read_table(FITS_LOOKUP)
    pid = np.array(t["primary_id"].to_pylist(), dtype=object)
    ra  = t["ra"].to_numpy(zero_copy_only=False)
    dec = t["dec"].to_numpy(zero_copy_only=False)
    t_hst  = np.array(t["tile_hst"].to_pylist(), dtype=object)
    t_jwst = np.array(t["tile_jwst"].to_pylist(), dtype=object)
    t_eu   = np.array(t["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(t["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(t["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(t["in_euclid"].to_pylist(), dtype=bool)

    finite = np.isfinite(ra) & np.isfinite(dec)
    # exclude near known SNe
    with LOOKUP_SN.open() as f:
        rows = list(csv.DictReader(f))
    sn_ra  = np.array([float(r["sn_ra"]) for r in rows])
    sn_dec = np.array([float(r["sn_dec"]) for r in rows])
    cos_d = np.cos(np.deg2rad(sn_dec))
    excl = np.zeros(len(ra), dtype=bool)
    for s_r, s_d, cd in zip(sn_ra, sn_dec, cos_d):
        dra = (ra - s_r) * cd * 3600.0
        ddec = (dec - s_d) * 3600.0
        sep = np.sqrt(dra**2 + ddec**2)
        excl |= (sep < exclude_radius_as)

    # prefer rows in all 3 surveys for diverse cutouts
    eligible = finite & ~excl & in_h & in_j & in_e
    idx = np.where(eligible)[0]
    rng = np.random.default_rng(RNG_SEED + 1)
    if len(idx) > n_neg:
        idx = rng.choice(idx, size=n_neg, replace=False)
    log(f"  negative sample: {len(idx):,} sources (eligible pool: {eligible.sum():,})")

    reqs = []
    for i in idx:
        i_int = int(i)
        pid_i = str(pid[i_int])
        ra_i = float(ra[i_int]); dec_i = float(dec[i_int])
        tiles = {"hst": str(t_hst[i_int]) if t_hst[i_int] else None,
                 "jwst": str(t_jwst[i_int]) if t_jwst[i_int] else None,
                 "vis": str(t_eu[i_int]) if t_eu[i_int] else None,
                 "nisp": str(t_eu[i_int]) if t_eu[i_int] else None}
        for (survey, band_label, R_band, kind) in ALL_BANDS:
            tt = tiles[survey]
            if not tt or tt in ("", "nan", "None"): continue
            reqs.append(dict(group="neg", id=pid_i, label=0,
                             survey=survey, band=band_label, R_band=R_band,
                             tile=tt, ra=ra_i, dec=dec_i, aug=0,
                             augment_op="none"))
    return reqs


def main():
    t_start = time.time()
    STATUS_LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== Step 3 v01: build training set ===")
    log(f"Output: {OUT_NPZ}")

    # ----- load known SNe -------------------------------------------------
    with LOOKUP_SN.open() as f:
        sn_rows = list(csv.DictReader(f))
    log(f"Loaded {len(sn_rows)} known SNe from {LOOKUP_SN.name}")

    # ----- build cutout requests -----------------------------------------
    log("Building requests ...")
    reqs_known = build_requests_for_known_sn(sn_rows)
    reqs_aug   = build_augmented_requests(sn_rows, n_aug=48)
    reqs_neg   = build_negative_requests(n_neg=N_NEG, exclude_radius_as=2.0)
    all_reqs = reqs_known + reqs_aug + reqs_neg
    log(f"  known_pos: {len(reqs_known):,}")
    log(f"  aug_pos:   {len(reqs_aug):,}")
    log(f"  negatives: {len(reqs_neg):,}")
    log(f"  total:     {len(all_reqs):,}")

    # ----- group by (survey, band, tile) for tile-major IO ---------------
    by_tile = defaultdict(list)
    for r in all_reqs:
        by_tile[(r["survey"], r["band"], r["R_band"], r["tile"])].append(r)
    log(f"  unique tile-band groups: {len(by_tile)}")

    # ----- pre-allocate output arrays ------------------------------------
    # We don't know exact size until done because some bands might be skipped.
    # Allocate per-request slots then split per-survey at write time.
    # Use dict[(group_idx, survey, band, aug)] -> cutout, plus metadata lists.
    cutouts = {}  # key: int (request index) -> np.array shape (CUT_PIX, CUT_PIX)
    meta = []     # list of dicts mirroring all_reqs, with array key

    # iterate tile-major
    t0_io = time.time()
    n_done = 0
    for tile_idx, (key, group) in enumerate(by_tile.items()):
        survey, band, R_band, tile = key
        path = resolve_path(survey, R_band, tile)
        sci_hdu, wcs = open_band_path(path)
        if sci_hdu is None:
            for r in group:
                req_idx = len(meta)
                cutouts[req_idx] = np.full((CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
                r2 = dict(r); r2["array_key"] = req_idx
                meta.append(r2)
            n_done += len(group)
            continue
        # sort by Y pixel for sequential disk access
        # (compute pixel y for sorting)
        try:
            ys = []
            for r in group:
                sx, sy = wcs.all_world2pix(r["ra"], r["dec"], 0)
                ys.append((float(sy) if np.isfinite(float(sy)) else 0.0, r))
            ys.sort(key=lambda t: t[0])
            group_sorted = [g[1] for g in ys]
        except Exception:
            group_sorted = group
        for r in group_sorted:
            req_idx = len(meta)
            c = cutout_at(sci_hdu, wcs, r["ra"], r["dec"], n=CUT_PIX)
            cutouts[req_idx] = c
            r2 = dict(r); r2["array_key"] = req_idx
            meta.append(r2)
        n_done += len(group_sorted)
        if (tile_idx + 1) % 20 == 0 or tile_idx == len(by_tile) - 1:
            elapsed = time.time() - t0_io
            log(f"  [{tile_idx+1}/{len(by_tile)}] tile {survey}/{band}/{tile} "
                f"+{len(group_sorted)} done={n_done:,}  elapsed={elapsed:.1f}s")

    # ----- organize per-survey arrays ------------------------------------
    log(f"IO done. Organising arrays ...")
    # We assemble per (group, sn_id_or_idx, aug) a stack per survey:
    #   HST stack:  shape (1, CUT, CUT)
    #   JWST stack: shape (4, CUT, CUT) in order F115W,F150W,F277W,F444W
    #   VIS stack:  shape (1, CUT, CUT)
    #   NISP stack: shape (3, CUT, CUT) in order Y,J,H
    survey_chan = {
        "hst":  ["F814W"],
        "jwst": ["F115W","F150W","F277W","F444W"],
        "vis":  ["VIS"],
        "nisp": ["Y","J","H"],
    }
    # group by (group, id, aug)
    by_sample = defaultdict(dict)   # key -> {survey: {band: idx}}
    for m in meta:
        k = (m["group"], m["id"], m["aug"])
        by_sample[k].setdefault(m["survey"], {})[m["band"]] = m["array_key"]

    log(f"  unique samples: {len(by_sample):,}")
    out_hst_x  = []; out_hst_y  = []; out_hst_meta  = []
    out_jwst_x = []; out_jwst_y = []; out_jwst_meta = []
    out_vis_x  = []; out_vis_y  = []; out_vis_meta  = []
    out_nisp_x = []; out_nisp_y = []; out_nisp_meta = []
    sample_keys = sorted(by_sample.keys(), key=lambda k: (k[0], str(k[1]), k[2]))
    for k in sample_keys:
        group, sid, aug = k
        bands = by_sample[k]
        # find the label (positives = 1, negatives = 0) — same across bands
        label = 1 if group in ("known_pos","aug_pos") else 0
        for survey, chans in survey_chan.items():
            if survey not in bands: continue
            arr = np.full((len(chans), CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for ci, b in enumerate(chans):
                if b in bands[survey]:
                    arr[ci] = cutouts[bands[survey][b]]
            # need at least one finite channel
            if not np.any(np.isfinite(arr)): continue
            # apply augmentation if positive aug>0
            if group == "aug_pos":
                rot_flip = aug % 8
                if rot_flip & 4:  # flip
                    arr = arr[:, :, ::-1].copy()
                k_rot = rot_flip % 4
                if k_rot:
                    arr = np.rot90(arr, k=k_rot, axes=(1, 2)).copy()
            md = dict(group=group, id=str(sid), aug=aug, label=label, survey=survey)
            if survey == "hst":  out_hst_x.append(arr);  out_hst_y.append(label);  out_hst_meta.append(md)
            if survey == "jwst": out_jwst_x.append(arr); out_jwst_y.append(label); out_jwst_meta.append(md)
            if survey == "vis":  out_vis_x.append(arr);  out_vis_y.append(label);  out_vis_meta.append(md)
            if survey == "nisp": out_nisp_x.append(arr); out_nisp_y.append(label); out_nisp_meta.append(md)

    def stack_or_empty(L, shape0):
        return np.stack(L, axis=0).astype(np.float32) if L else np.zeros((0, shape0, CUT_PIX, CUT_PIX), np.float32)

    X_hst  = stack_or_empty(out_hst_x,  1); y_hst  = np.array(out_hst_y,  dtype=np.int8)
    X_jwst = stack_or_empty(out_jwst_x, 4); y_jwst = np.array(out_jwst_y, dtype=np.int8)
    X_vis  = stack_or_empty(out_vis_x,  1); y_vis  = np.array(out_vis_y,  dtype=np.int8)
    X_nisp = stack_or_empty(out_nisp_x, 3); y_nisp = np.array(out_nisp_y, dtype=np.int8)

    log(f"  HST:   X={X_hst.shape}   y={y_hst.shape}  pos={int(y_hst.sum())}")
    log(f"  JWST:  X={X_jwst.shape}  y={y_jwst.shape} pos={int(y_jwst.sum())}")
    log(f"  VIS:   X={X_vis.shape}   y={y_vis.shape}  pos={int(y_vis.sum())}")
    log(f"  NISP:  X={X_nisp.shape}  y={y_nisp.shape} pos={int(y_nisp.sum())}")

    # metadata as recarray-style: parallel arrays
    def meta_arrays(M):
        return dict(
            group=np.array([m["group"] for m in M]),
            id   =np.array([m["id"]    for m in M]),
            aug  =np.array([m["aug"]   for m in M]),
        )
    log(f"Writing {OUT_NPZ} ...")
    np.savez_compressed(OUT_NPZ,
        X_hst=X_hst,   y_hst=y_hst,   **{f"hst_{k}":v   for k,v in meta_arrays(out_hst_meta).items()},
        X_jwst=X_jwst, y_jwst=y_jwst, **{f"jwst_{k}":v  for k,v in meta_arrays(out_jwst_meta).items()},
        X_vis=X_vis,   y_vis=y_vis,   **{f"vis_{k}":v   for k,v in meta_arrays(out_vis_meta).items()},
        X_nisp=X_nisp, y_nisp=y_nisp, **{f"nisp_{k}":v  for k,v in meta_arrays(out_nisp_meta).items()},
    )
    size_mb = OUT_NPZ.stat().st_size / 1e6
    log(f"  wrote {size_mb:.1f} MB")

    log(f"=== Step 3 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
