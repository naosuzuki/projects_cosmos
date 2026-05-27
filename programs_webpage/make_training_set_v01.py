"""Step 3 v01 (parallel): build SN training set (cutouts).

PARALLELISED via multiprocessing.Pool — each worker opens its own FITS
and extracts its assigned cutouts in a tile-major loop. Memory-mapped IO,
slice-then-cast.

Inputs:
  csvfiles_sn/lookup_sn17_v32.csv         (17 known SN positions)
  csvfiles_sn/sn_candidates_v01.parquet   (1.12M candidates; negatives sampled)
  csvfiles_sn/fits_lookup_v01.parquet     (per-source tile IDs)

Output:
  csvfiles_sn/training_set_v01.npz        (compressed cutouts + labels)

Cutout: 64×64 px, native pixel scale per band.
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
FITS_LOOKUP = CSV_DIR / "fits_lookup_v01.parquet"
OUT_NPZ    = CSV_DIR / "training_set_v01.npz"

CUT_PIX    = 64
N_NEG      = 10000
N_AUG      = 48
RNG_SEED   = 42
N_WORKERS  = max(2, cpu_count() // 2)   # leave headroom; FITS IO benefits from concurrency but not unboundedly

HST_BANDS  = [("F814W", "F814W")]
JWST_BANDS = [("F115W", "f115w"), ("F150W", "f150w"), ("F277W", "f277w"), ("F444W", "f444w")]
VIS_BANDS  = [("VIS",   "VIS")]
NISP_BANDS = [("Y",     "NIR-Y"), ("J", "NIR-J"), ("H", "NIR-H")]
ALL_BANDS  = ([("hst",  b, r, "hst")    for b, r in HST_BANDS]  +
              [("jwst", b, r, "jwst")   for b, r in JWST_BANDS] +
              [("vis",  b, r, "euclid") for b, r in VIS_BANDS]  +
              [("nisp", b, r, "euclid") for b, r in NISP_BANDS])


def log(msg):
    """Print only — orchestrator handles file logging."""
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
    """Process one (survey, band, R_band, tile) group of requests.

    Args is (survey, band, R_band, tile, requests). Each request is a dict with
    keys ra, dec, array_key. Returns list of (array_key, cutout_array).
    """
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
            # Sort by pixel-Y for sequential page-cache hits.
            try:
                ras  = np.array([r["ra"]  for r in reqs])
                decs = np.array([r["dec"] for r in reqs])
                _, sys_ = wcs.all_world2pix(ras, decs, 0)
                order = np.argsort(np.where(np.isfinite(sys_), sys_, 0.0))
            except Exception:
                order = np.arange(len(reqs))
            out = []
            data = sci_hdu.data
            for i in order:
                r = reqs[i]
                c = cutout_at(data, wcs, r["ra"], r["dec"], n=CUT_PIX)
                out.append((r["array_key"], c))
        return out
    except Exception as e:
        return [(r["array_key"], np.full((CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)) for r in reqs]


def build_requests_for_known_sn(rows):
    reqs = []
    for row in rows:
        sn_id = int(row["id"])
        ra = float(row["sn_ra"]); dec = float(row["sn_dec"])
        tiles = dict(hst=row["hst"], jwst=row["jwst"], vis=row["euclid"], nisp=row["euclid"])
        for (survey, band, R_band, _kind) in ALL_BANDS:
            t = tiles[survey]
            if not t or t in ("", "nan", "None"):
                continue
            reqs.append(dict(group="known_pos", id=sn_id, label=1,
                             survey=survey, band=band, R_band=R_band,
                             tile=t, ra=ra, dec=dec, aug=0))
    return reqs


def build_augmented_requests(rows, n_aug=N_AUG):
    reqs = []
    rng = np.random.default_rng(RNG_SEED)
    for row in rows:
        sn_id = int(row["id"])
        ra0 = float(row["sn_ra"]); dec0 = float(row["sn_dec"])
        tiles = dict(hst=row["hst"], jwst=row["jwst"], vis=row["euclid"], nisp=row["euclid"])
        for aug_i in range(n_aug):
            d_ra_as  = rng.uniform(-0.05, 0.05)
            d_dec_as = rng.uniform(-0.05, 0.05)
            cos_d = math.cos(math.radians(dec0))
            ra = ra0 + d_ra_as / 3600.0 / cos_d
            dec = dec0 + d_dec_as / 3600.0
            for (survey, band, R_band, _kind) in ALL_BANDS:
                t = tiles[survey]
                if not t or t in ("", "nan", "None"):
                    continue
                reqs.append(dict(group="aug_pos", id=sn_id, label=1,
                                 survey=survey, band=band, R_band=R_band,
                                 tile=t, ra=ra, dec=dec, aug=aug_i))
    return reqs


def build_negative_requests(n_neg=N_NEG, exclude_radius_as=2.0):
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
    with LOOKUP_SN.open() as f:
        rows = list(csv.DictReader(f))
    sn_ra  = np.array([float(r["sn_ra"])  for r in rows])
    sn_dec = np.array([float(r["sn_dec"]) for r in rows])
    cos_d  = np.cos(np.deg2rad(sn_dec))
    excl = np.zeros(len(ra), dtype=bool)
    for s_r, s_d, cd in zip(sn_ra, sn_dec, cos_d):
        dra  = (ra  - s_r) * cd * 3600.0
        ddec = (dec - s_d) * 3600.0
        sep = np.sqrt(dra**2 + ddec**2)
        excl |= (sep < exclude_radius_as)
    eligible = finite & ~excl & in_h & in_j & in_e
    idx = np.where(eligible)[0]
    rng = np.random.default_rng(RNG_SEED + 1)
    if len(idx) > n_neg:
        idx = rng.choice(idx, size=n_neg, replace=False)
    log(f"  negative sample: {len(idx):,} sources (eligible pool: {eligible.sum():,})")

    reqs = []
    for i in idx:
        ii = int(i)
        tiles = {"hst": str(t_hst[ii]) if t_hst[ii] else None,
                 "jwst": str(t_jwst[ii]) if t_jwst[ii] else None,
                 "vis": str(t_eu[ii]) if t_eu[ii] else None,
                 "nisp": str(t_eu[ii]) if t_eu[ii] else None}
        for (survey, band, R_band, _kind) in ALL_BANDS:
            tt = tiles[survey]
            if not tt or tt in ("", "nan", "None"): continue
            reqs.append(dict(group="neg", id=str(pid[ii]), label=0,
                             survey=survey, band=band, R_band=R_band,
                             tile=tt, ra=float(ra[ii]), dec=float(dec[ii]), aug=0))
    return reqs


def main():
    t_start = time.time()
    log("=== Step 3 v01 (parallel): build training set ===")
    log(f"Output: {OUT_NPZ}")
    log(f"Workers: {N_WORKERS}")

    with LOOKUP_SN.open() as f:
        sn_rows = list(csv.DictReader(f))
    log(f"Loaded {len(sn_rows)} known SNe")

    log("Building requests ...")
    reqs_known = build_requests_for_known_sn(sn_rows)
    reqs_aug   = build_augmented_requests(sn_rows, n_aug=N_AUG)
    reqs_neg   = build_negative_requests(n_neg=N_NEG, exclude_radius_as=2.0)
    all_reqs = reqs_known + reqs_aug + reqs_neg
    log(f"  known_pos: {len(reqs_known):,}")
    log(f"  aug_pos:   {len(reqs_aug):,}")
    log(f"  negatives: {len(reqs_neg):,}")
    log(f"  total:     {len(all_reqs):,}")

    # assign global array_keys
    for i, r in enumerate(all_reqs):
        r["array_key"] = i

    # group by (survey, band, R_band, tile)
    by_tile = defaultdict(list)
    for r in all_reqs:
        by_tile[(r["survey"], r["band"], r["R_band"], r["tile"])].append(r)
    log(f"  tile-band groups: {len(by_tile)}  (parallelising IO across {N_WORKERS} workers)")

    # dispatch jobs
    jobs = [(s, b, rb, t, lst) for (s, b, rb, t), lst in by_tile.items()]

    t_io = time.time()
    cutouts = {}
    n_done = 0
    with Pool(processes=N_WORKERS) as pool:
        for tile_idx, results in enumerate(pool.imap_unordered(worker_extract, jobs)):
            for key, arr in results:
                cutouts[key] = arr
            n_done += len(results)
            if (tile_idx + 1) % 20 == 0 or tile_idx == len(jobs) - 1:
                log(f"  [{tile_idx+1}/{len(jobs)}] cutouts so far: {n_done:,}  "
                    f"elapsed={time.time()-t_io:.1f}s")
    log(f"Parallel IO done in {time.time()-t_io:.1f}s")

    # assemble per-survey arrays (same as before)
    log("Assembling per-survey arrays ...")
    survey_chan = {"hst":["F814W"], "jwst":["F115W","F150W","F277W","F444W"],
                   "vis":["VIS"], "nisp":["Y","J","H"]}

    # by (group, id, aug) collect band cutouts
    by_sample = defaultdict(dict)   # k -> {survey: {band: array_key}}
    for r in all_reqs:
        k = (r["group"], r["id"], r["aug"])
        by_sample[k].setdefault(r["survey"], {})[r["band"]] = r["array_key"]

    out = {s: dict(X=[], y=[], group=[], id=[], aug=[]) for s in survey_chan}
    for k in sorted(by_sample.keys(), key=lambda x: (x[0], str(x[1]), x[2])):
        group, sid, aug = k
        label = 1 if group in ("known_pos", "aug_pos") else 0
        bands = by_sample[k]
        for survey, chans in survey_chan.items():
            if survey not in bands: continue
            arr = np.full((len(chans), CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for ci, b in enumerate(chans):
                if b in bands[survey]:
                    arr[ci] = cutouts[bands[survey][b]]
            if not np.any(np.isfinite(arr)): continue
            if group == "aug_pos":
                rot_flip = aug % 8
                if rot_flip & 4:
                    arr = arr[:, :, ::-1].copy()
                k_rot = rot_flip % 4
                if k_rot:
                    arr = np.rot90(arr, k=k_rot, axes=(1, 2)).copy()
            out[survey]["X"].append(arr)
            out[survey]["y"].append(label)
            out[survey]["group"].append(group)
            out[survey]["id"].append(str(sid))
            out[survey]["aug"].append(aug)

    def stack(L, C):
        return np.stack(L).astype(np.float32) if L else np.zeros((0, C, CUT_PIX, CUT_PIX), np.float32)

    X_hst  = stack(out["hst"]["X"],  1); y_hst  = np.array(out["hst"]["y"],  dtype=np.int8)
    X_jwst = stack(out["jwst"]["X"], 4); y_jwst = np.array(out["jwst"]["y"], dtype=np.int8)
    X_vis  = stack(out["vis"]["X"],  1); y_vis  = np.array(out["vis"]["y"],  dtype=np.int8)
    X_nisp = stack(out["nisp"]["X"], 3); y_nisp = np.array(out["nisp"]["y"], dtype=np.int8)
    log(f"  HST:  X={X_hst.shape}  pos={int(y_hst.sum())}")
    log(f"  JWST: X={X_jwst.shape} pos={int(y_jwst.sum())}")
    log(f"  VIS:  X={X_vis.shape}  pos={int(y_vis.sum())}")
    log(f"  NISP: X={X_nisp.shape} pos={int(y_nisp.sum())}")

    log(f"Writing {OUT_NPZ} ...")
    np.savez_compressed(OUT_NPZ,
        X_hst=X_hst,   y_hst=y_hst,
        hst_group=np.array(out["hst"]["group"]),
        hst_id   =np.array(out["hst"]["id"]),
        hst_aug  =np.array(out["hst"]["aug"]),
        X_jwst=X_jwst, y_jwst=y_jwst,
        jwst_group=np.array(out["jwst"]["group"]),
        jwst_id   =np.array(out["jwst"]["id"]),
        jwst_aug  =np.array(out["jwst"]["aug"]),
        X_vis=X_vis,   y_vis=y_vis,
        vis_group=np.array(out["vis"]["group"]),
        vis_id   =np.array(out["vis"]["id"]),
        vis_aug  =np.array(out["vis"]["aug"]),
        X_nisp=X_nisp, y_nisp=y_nisp,
        nisp_group=np.array(out["nisp"]["group"]),
        nisp_id   =np.array(out["nisp"]["id"]),
        nisp_aug  =np.array(out["nisp"]["aug"]),
    )
    log(f"  wrote {OUT_NPZ.stat().st_size/1e6:.1f} MB")
    log(f"=== Step 3 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
