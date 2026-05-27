"""Add VIS to v03 search retroactively.

Reality check: there ARE VIS supernovae — we just lack a labeled
VIS-DISCOVERED example in this catalog. Many JWST/NISP-discovered SNe ARE
in the VIS images (Euclid VIS depth ~26 mag goes deep enough). So:

  1. Treat all 17 known SN positions as VIS positives.
  2. Extract VIS cutouts at those positions + 5000 negatives.
  3. Augment with rotation × flip × magnitude-scaling (same dN/dm prior).
  4. Train a VIS-only CNN (1 channel, MPS, ~2-3 min).
  5. Add it to cnn_models_v03.pt and start a VIS inference worker in
     parallel with the running HST/JWST/NISP workers. The watcher already
     supports VIS — partials get picked up automatically.

Outputs:
  csvfiles_sn/training_set_v03_vis.npz   (VIS-only training arrays)
  csvfiles_sn/cnn_models_v03.pt          (updated with "vis" entry)
  csvfiles_sn/cnn_thresholds_v03.json    (updated)
  csvfiles_sn/_partial/infer_v03_vis.parquet  (worker checkpoints)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv, time, math, os, json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R
from cnn_models_v03 import SmallCNN, best_device

CSV_DIR     = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
LOOKUP_SN   = CSV_DIR / "lookup_sn17_v32.csv"
TBL_SN      = CSV_DIR / "tbl_sn17_v32.csv"
FITS_LOOKUP = CSV_DIR / "fits_lookup_v03.parquet"
OUT_NPZ     = CSV_DIR / "training_set_v03_vis.npz"
MODELS_PT   = CSV_DIR / "cnn_models_v03.pt"
THRESH_JSON = CSV_DIR / "cnn_thresholds_v03.json"
PART_DIR    = CSV_DIR / "_partial"

CUT_PIX     = 64
N_NEG       = 5000
N_AUG_MAG   = 60
N_ROT_FLIP  = 8
RNG_SEED    = 42
N_WORKERS   = max(2, cpu_count() // 2)
MAG_RANGE   = (21.0, 26.0)
EPOCHS, BATCH, LR = 30, 256, 1e-3
TARGET_FPR  = 1e-3
BATCH_CNN   = 1024


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] VIS: {msg}", flush=True)


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
    tile, reqs = args
    path = R_local.resolve_euclid_path(tile, "VIS")
    if not path or not os.path.exists(path):
        return [(r["key"], np.full((CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)) for r in reqs]
    try:
        with fits.open(path, memmap=True) as h:
            sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
            wcs = WCS(sci_hdu.header)
            try:
                ras = np.array([r["ra"] for r in reqs])
                decs = np.array([r["dec"] for r in reqs])
                _, sys_ = wcs.all_world2pix(ras, decs, 0)
                order = np.argsort(np.where(np.isfinite(sys_), sys_, 0.0))
            except Exception:
                order = np.arange(len(reqs))
            data = sci_hdu.data
            return [(reqs[i]["key"], cutout_at(data, wcs, reqs[i]["ra"], reqs[i]["dec"]))
                    for i in order]
    except Exception:
        return [(r["key"], np.full((CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)) for r in reqs]


def sample_mag_dndm(lo, hi, n, rng, alpha=0.4):
    u = rng.uniform(0, 1, n)
    bias = 10**(alpha * lo)
    span = 10**(alpha * hi) - bias
    return np.log10(bias + u * span) / alpha


def per_channel_stats(X):
    C = X.shape[1]
    med = np.zeros(C, dtype=np.float32); sig = np.ones(C, dtype=np.float32)
    for c in range(C):
        v = X[:, c]; v = v[np.isfinite(v)]
        if v.size == 0: continue
        m = np.median(v); mad = np.median(np.abs(v - m))
        med[c] = float(m); sig[c] = float(1.4826 * mad) if mad > 0 else 1.0
    return med, sig


# ---------- Stage 1: build VIS training set ----------
def build_vis_training_set():
    log("=== Stage 1: VIS training set ===")
    t0 = time.time()
    # Load 17 known SNe
    with LOOKUP_SN.open() as f:
        sn_rows = list(csv.DictReader(f))
    with TBL_SN.open() as f:
        sn_mags_tbl = {int(r["id"]): r for r in csv.DictReader(f)}
    log(f"Loaded {len(sn_rows)} known SN positions (treating ALL as VIS positives)")

    # known SN VIS requests
    by_tile = defaultdict(list)
    pos_keys = []
    for ri, r in enumerate(sn_rows):
        ra, dec = float(r["sn_ra"]), float(r["sn_dec"])
        tile = r["euclid"]
        if not tile: continue
        key = ("pos", int(r["id"]))
        pos_keys.append(key)
        by_tile[tile].append(dict(key=key, ra=ra, dec=dec))

    # negatives from full-coverage candidates, excluding 2" near known SNe
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
    idx = np.where(eligible & ~excl)[0]
    rng = np.random.default_rng(RNG_SEED + 7)
    if len(idx) > N_NEG:
        idx = rng.choice(idx, size=N_NEG, replace=False)
    log(f"Negative pool eligible={int((eligible & ~excl).sum()):,}, sampled {len(idx):,}")
    neg_keys = []
    for i in idx:
        ii = int(i); tile = str(lk["tile_euclid"].iloc[ii])
        if not tile or tile in ("nan", "None", ""): continue
        key = ("neg", str(lk["primary_id"].iloc[ii]))
        neg_keys.append(key)
        by_tile[tile].append(dict(key=key,
                                  ra=float(lk["ra"].iloc[ii]),
                                  dec=float(lk["dec"].iloc[ii])))

    jobs = list(by_tile.items())
    log(f"Tiles to read: {len(jobs)}  (workers={N_WORKERS})")

    # parallel IO
    cutouts = {}
    t_io = time.time()
    with Pool(processes=N_WORKERS) as pool:
        for results in pool.imap_unordered(worker_extract, jobs):
            for k, arr in results:
                cutouts[k] = arr
    log(f"IO done in {time.time()-t_io:.1f}s — {len(cutouts):,} cutouts")

    # Build arrays: positives = real + augmentations
    rng2 = np.random.default_rng(RNG_SEED + 8)
    X_list, y_list, group_list, id_list, mag_list = [], [], [], [], []
    n_pos_real = 0
    for k in pos_keys:
        arr = cutouts.get(k)
        if arr is None or not np.any(np.isfinite(arr)): continue
        sid = k[1]
        # try to get mag_VIS from TBL_SN
        m_real = -1.0
        if sid in sn_mags_tbl:
            try:
                v = float(sn_mags_tbl[sid].get("mag_VIS", "-1"))
                if v > 0: m_real = v
            except (ValueError, TypeError): pass
        if m_real <= 0:
            m_real = MAG_RANGE[0] + 1.0
        X_list.append(arr[None, :, :].copy())
        y_list.append(1); group_list.append("known"); id_list.append(str(sid))
        mag_list.append(m_real)
        n_pos_real += 1
        # augmentations
        for _ in range(N_AUG_MAG):
            lo, hi = MAG_RANGE
            m_target = float(sample_mag_dndm(lo, hi, 1, rng2, alpha=0.4)[0])
            scale = 10**(0.4 * (m_real - m_target))
            rf = int(rng2.integers(0, N_ROT_FLIP))
            a = arr.copy()
            if np.any(np.isfinite(a)):
                bg = float(np.nanmedian(a[:8, :8]))
                a = (a - bg) * scale + bg
            if rf & 4: a = a[:, ::-1].copy()
            k_rot = rf % 4
            if k_rot: a = np.rot90(a, k=k_rot).copy()
            X_list.append(a[None, :, :])
            y_list.append(1); group_list.append("aug"); id_list.append(str(sid))
            mag_list.append(m_target)

    n_pos_total = len(X_list)
    for k in neg_keys:
        arr = cutouts.get(k)
        if arr is None or not np.any(np.isfinite(arr)): continue
        X_list.append(arr[None, :, :])
        y_list.append(0); group_list.append("neg"); id_list.append(k[1])
        mag_list.append(-1.0)
    log(f"VIS arrays: real_pos={n_pos_real}, aug_pos={n_pos_total-n_pos_real}, "
        f"neg={len(X_list)-n_pos_total}, total={len(X_list)}")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int8)
    np.savez_compressed(OUT_NPZ,
                        X_vis=X, y_vis=y,
                        vis_group=np.array(group_list),
                        vis_id=np.array(id_list),
                        vis_mag=np.array(mag_list, dtype=np.float32))
    log(f"Wrote {OUT_NPZ}  ({OUT_NPZ.stat().st_size/1e6:.1f} MB)  in {time.time()-t0:.1f}s")
    return X, y, np.array(group_list), np.array(id_list)


# ---------- Stage 2: train VIS CNN ----------
def train_vis_cnn(X, y):
    log("=== Stage 2: train VIS CNN ===")
    t0 = time.time()
    device = best_device()
    log(f"Device: {device}  X={X.shape}  pos={int(y.sum())}  neg={int((y==0).sum())}")
    model = SmallCNN(in_ch=1).to(device)
    med, sig = per_channel_stats(X)
    model.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    ds = TensorDataset(X_t, y_t)
    n_pos = int(y_t.sum().item()); n_neg = len(y_t) - n_pos
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad(); l = crit(model(xb), yb); l.backward(); opt.step()
    # threshold @ FPR=1e-3 on negatives
    model.eval()
    with torch.no_grad():
        neg_idx = np.where(y == 0)[0]
        scores = []
        for i in range(0, len(neg_idx), BATCH_CNN):
            xb = torch.tensor(X[neg_idx[i:i+BATCH_CNN]], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
        neg_scores = np.concatenate(scores)
    thr = float(np.quantile(neg_scores, 1.0 - TARGET_FPR))
    log(f"VIS threshold@FPR={TARGET_FPR:.0e}: P>={thr:.4f}  trained in {time.time()-t0:.1f}s")
    return model, med, sig, thr


# ---------- Stage 3: update model bundle ----------
def add_to_bundle(state_dict, med, sig, thr):
    log("=== Stage 3: add VIS to model bundle ===")
    bundle = torch.load(MODELS_PT, map_location="cpu", weights_only=False)
    bundle["models"]["vis"] = {k: v.detach().cpu() for k, v in state_dict.items()}
    bundle["norm_mean"]["vis"] = med
    bundle["norm_std"]["vis"]  = sig
    bundle["thresholds"]["vis"] = thr
    torch.save(bundle, MODELS_PT)
    thresholds_now = bundle["thresholds"]
    THRESH_JSON.write_text(json.dumps(thresholds_now, indent=2))
    log(f"Bundle updated.  Thresholds now: {thresholds_now}")


# ---------- Stage 4: VIS inference worker (parallel to running pipeline) ----------
def run_vis_inference():
    log("=== Stage 4: VIS inference (parallel to running pipeline) ===")
    t0 = time.time()
    from astropy.io import fits
    from astropy.wcs import WCS
    sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
    from infer_sn_v03 import cutout_and_aper, atomic_write_parquet, FWHM_AS
    import pyarrow as pa
    # Load lookup + model
    lk = pq.read_table(FITS_LOOKUP)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    te   = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    ra   = lk["ra"].to_numpy(zero_copy_only=False)
    dec  = lk["dec"].to_numpy(zero_copy_only=False)
    idxs = np.where(in_e)[0]
    sources = [(int(i), float(ra[i]), float(dec[i]), str(te[i])) for i in idxs if te[i]]
    log(f"VIS sources to process: {len(sources):,}")

    bundle = torch.load(MODELS_PT, map_location="cpu", weights_only=False)
    device = best_device()
    model = SmallCNN(in_ch=1).to(device)
    model.load_state_dict(bundle["models"]["vis"])
    model.set_norm(torch.tensor(bundle["norm_mean"]["vis"], device=device),
                   torch.tensor(bundle["norm_std"]["vis"], device=device))
    model.eval()

    # Group by tile
    by_tile = defaultdict(list)
    for (idx, r, d, t) in sources:
        by_tile[t].append((idx, r, d))

    PART_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = PART_DIR / "infer_v03_vis.parquet"
    acc = {"idx": [], "P": [],
           "mag_VIS": [], "snr_VIS": [],
           "sharp_VIS": [], "rnd1_VIS": [], "rnd2_VIS": [],
           "sep_VIS": [], "nanfrac_VIS": []}

    sorted_tiles = sorted(by_tile.items())
    for ti, (tile, tile_sources) in enumerate(sorted_tiles):
        path = R.resolve_euclid_path(tile, "VIS")
        if not path or not os.path.exists(path):
            for (gidx, _, _) in tile_sources:
                acc["idx"].append(gidx); acc["P"].append(float("nan"))
                acc["mag_VIS"].append(-1.0); acc["snr_VIS"].append(0.0)
                acc["sharp_VIS"].append(float("nan"))
                acc["rnd1_VIS"].append(float("nan"))
                acc["rnd2_VIS"].append(float("nan"))
                acc["sep_VIS"].append(float("nan"))
                acc["nanfrac_VIS"].append(1.0)
            continue
        try:
            h = fits.open(path, memmap=True)
            sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
            wcs = WCS(sci_hdu.header)
            hdr = sci_hdu.header
            data = sci_hdu.data
        except Exception as e:
            log(f"  open failed {Path(path).name}: {e}")
            continue
        # Y-sort
        try:
            ras = np.array([s[1] for s in tile_sources])
            decs = np.array([s[2] for s in tile_sources])
            _, ys = wcs.all_world2pix(ras, decs, 0)
            order = np.argsort(np.where(np.isfinite(ys), ys, 0.0))
        except Exception:
            order = np.arange(len(tile_sources))
        ordered = [tile_sources[i] for i in order]

        # batched inference
        for bstart in range(0, len(ordered), BATCH_CNN):
            batch = ordered[bstart:bstart+BATCH_CNN]
            cutouts = np.full((len(batch), 1, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            mag_b = []; snr_b = []
            sharp_b = []; rnd1_b = []; rnd2_b = []; sep_b = []; nf_b = []
            for k, (gidx, r, d) in enumerate(batch):
                c, mag, snr, sharp, rnd1, rnd2, sep_sn, nan_frac = cutout_and_aper(
                    data, hdr, wcs, r, d, FWHM_AS["VIS"], kind="euclid")
                cutouts[k, 0] = c
                mag_b.append(mag); snr_b.append(snr)
                sharp_b.append(sharp); rnd1_b.append(rnd1); rnd2_b.append(rnd2)
                sep_b.append(sep_sn); nf_b.append(nan_frac)
            with torch.no_grad():
                xb = torch.tensor(cutouts, dtype=torch.float32, device=device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
            for k, (gidx, _, _) in enumerate(batch):
                acc["idx"].append(gidx)
                acc["P"].append(float(p[k]))
                acc["mag_VIS"].append(mag_b[k]); acc["snr_VIS"].append(snr_b[k])
                acc["sharp_VIS"].append(sharp_b[k])
                acc["rnd1_VIS"].append(rnd1_b[k])
                acc["rnd2_VIS"].append(rnd2_b[k])
                acc["sep_VIS"].append(sep_b[k])
                acc["nanfrac_VIS"].append(nf_b[k])
        try: h.close()
        except Exception: pass
        table = pa.table({k: np.array(v) for k, v in acc.items()})
        atomic_write_parquet(table, checkpoint)
        log(f"  tile [{ti+1}/{len(sorted_tiles)}] {tile} acc={len(acc['idx']):,}/"
            f"{len(sources):,} elapsed={time.time()-t0:.1f}s")
    # final
    table = pa.table({k: np.array(v) for k, v in acc.items()})
    atomic_write_parquet(table, checkpoint)
    log(f"VIS inference done in {time.time()-t0:.1f}s")


def main():
    t_total = time.time()
    X, y, group, ids = build_vis_training_set()
    model, med, sig, thr = train_vis_cnn(X, y)
    add_to_bundle(model.state_dict(), med, sig, thr)
    # Note: free the GPU model before forking is unnecessary — run_vis_inference
    # creates its own. Just delete the trainer's state.
    del model
    run_vis_inference()
    log(f"=== ALL VIS STAGES done in {time.time()-t_total:.1f}s ===")


if __name__ == "__main__":
    main()
