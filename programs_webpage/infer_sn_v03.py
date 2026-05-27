"""Step 6 v01 (parallel + streaming): inference + incremental output.

Architecture:
  - 4 survey worker processes (HST / JWST / VIS / NISP), each on its own MPS
    context. Each iterates over its survey's tiles tile-major.
  - After EACH tile completes, the worker appends per-source results
    (primary_idx, P, mag_<band>, snr_<band>) to a per-survey checkpoint
    parquet file (atomic rename).
  - One watcher process runs alongside. Every WATCH_INTERVAL seconds it
    reads all 4 checkpoint parquets, applies the coverage-aware 1-of-N
    rule, and rewrites tbl_sn_candidates_v03.csv + the webpage.

Coverage-aware rule:
  Per source, look only at surveys WITH coverage (in_hst / in_jwst /
  in_euclid). A source is a SN candidate iff among the surveys we have
  processed and have coverage, EXACTLY ONE has P >= threshold and the
  others have P < threshold. Sources still missing some surveys' P show
  up as "pending" (not yet emitted) until all covered surveys are done.

Outputs:
  csvfiles_sn/_partial/infer_v03_<survey>.parquet      (worker checkpoints)
  csvfiles_sn/tbl_sn_candidates_v03.csv                (final + updated each cycle)
  csvfiles_sn/sn_candidates_v03_all_scored.parquet     (all sources, final)
  htmls/sn_search/v03/index.html                       (updated each watcher tick)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, os, csv, math, json
from pathlib import Path
from multiprocessing import Pool, Process, Event, set_start_method
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R

CSV_DIR   = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR  = CSV_DIR / "_partial"
HTML_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v03")
LOOK      = CSV_DIR / "fits_lookup_v03.parquet"
MODELS    = CSV_DIR / "cnn_models_v03.pt"
OUT_CSV   = CSV_DIR / "tbl_sn_candidates_v03.csv"
OUT_PARQ  = CSV_DIR / "sn_candidates_v03_all_scored.parquet"

CUT_PIX     = 64
BATCH_CNN   = 1024
TOP_N_WEB   = 500
WATCH_INTERVAL = 15   # seconds between webpage refreshes

# Magnitude floor — reject anything brighter than this in ANY band (saturation
# guard). 21.0 chosen by user 2026-05-27 to avoid saturated stars while keeping
# faint-SN sensitivity.
MAG_FLOOR = 21.0

HST_BANDS  = [("F814W","F814W")]
JWST_BANDS = [("F115W","f115w"),("F150W","f150w"),("F277W","f277w"),("F444W","f444w")]
VIS_BANDS  = [("VIS","VIS")]
NISP_BANDS = [("Y","NIR-Y"),("J","NIR-J"),("H","NIR-H")]

FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}

CSV_BANDS = ["F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W"]
SURVEY_BANDS = {"hst":  ["F814W"],
                "jwst": ["F115W","F150W","F277W","F444W"],
                "vis":  ["VIS"],
                "nisp": ["Y","J","H"]}


def log(msg, prefix="MAIN"):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix}: {msg}", flush=True)


# -------- shared helpers (used in workers) --------
def cutout_and_aper(sci_data, hdr, wcs, ra, dec, fwhm_as, kind, n=CUT_PIX):
    """v03: returns (cutout, mag, snr, sharp, rnd1, rnd2, sep_sn_as, nan_frac).

    sharp/rnd1/rnd2/sep_sn_as = DAO morphology of the nearest peak to the
    source position (NaN if no peak in 3" search radius). Allow downstream
    G2-G4 gating without re-opening the FITS.

    nan_frac = fraction of NaN OR zero pixels in the central 20x20 region
    of the cutout. >0.25 is the "edge/incomplete coverage" veto.
    """
    import numpy as np, math
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder
    NAN_RESULT = (np.full((n, n), np.nan, dtype=np.float32),
                  -1.0, 0.0, float("nan"), float("nan"), float("nan"),
                  float("nan"), 1.0)
    try:
        sx, sy = wcs.all_world2pix(ra, dec, 0)
    except Exception:
        return NAN_RESULT
    sx = float(sx); sy = float(sy)
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return NAN_RESULT
    ps = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    half = n // 2
    cx = int(round(sx)); cy = int(round(sy))
    ny, nx = sci_data.shape[-2:]
    x0 = cx - half; x1 = x0 + n
    y0 = cy - half; y1 = y0 + n
    cutout = np.full((n, n), np.nan, dtype=np.float32)
    if x1 > 0 and y1 > 0 and x0 < nx and y0 < ny:
        xs0 = max(0, -x0); ys0 = max(0, -y0)
        xs1 = n - max(0, x1 - nx); ys1 = n - max(0, y1 - ny)
        src_x0 = max(0, x0); src_y0 = max(0, y0)
        src_x1 = min(nx, x1); src_y1 = min(ny, y1)
        if src_x1 > src_x0 and src_y1 > src_y0:
            cutout[ys0:ys1, xs0:xs1] = sci_data[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    if not np.any(np.isfinite(cutout)):
        return NAN_RESULT
    # NaN/zero fraction in central 20x20
    cyc, cxc = cutout.shape[0]//2, cutout.shape[1]//2
    cen = cutout[max(0,cyc-10):cyc+10, max(0,cxc-10):cxc+10]
    nan_frac = float(((~np.isfinite(cen)) | (cen == 0)).sum()) / max(1, cen.size)

    sub = np.where(np.isfinite(cutout), cutout, 0.0).astype(np.float64)
    cy_p = (sy - (cy - half)); cx_p = (sx - (cx - half))
    aper_r = fwhm_as / ps; ring_in = 2.0 * fwhm_as / ps; ring_out = 3.5 * fwhm_as / ps
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - cx_p)**2 + (yy - cy_p)**2)
    in_aper = rr <= aper_r; in_ring = (rr >= ring_in) & (rr <= ring_out)
    if not in_aper.any() or not in_ring.any():
        return cutout, -1.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"), nan_frac
    bg = float(np.nanmedian(sub[in_ring]))
    sub2 = sub - bg
    n_aper = int(in_aper.sum())
    flux = float(np.nansum(sub2[in_aper]))
    sig = float(np.nanstd(sub[in_ring])) * math.sqrt(n_aper)
    snr = flux / (sig + 1e-30)
    # AB magnitude
    bunit = (hdr.get("BUNIT") or "").strip()
    if kind == "jwst" or "MJy/sr" in bunit:
        pix_sr = hdr.get("PIXAR_SR") or ((ps / 206265.0) ** 2)
        f_jy = flux * float(pix_sr) * 1.0e6
        mag = (-2.5 * math.log10(f_jy / 3631.0)) if f_jy > 0 else -1.0
    elif kind == "hst":
        zp = hdr.get("ABMAG_ZP") or hdr.get("ABMAGZP") or hdr.get("MAGZP") or 25.937
        mag = (float(zp) - 2.5 * math.log10(flux)) if flux > 0 else -1.0
    else:
        zp = hdr.get("ZP") or hdr.get("MAGZP") or hdr.get("PHOTZP") or 23.9
        mag = (float(zp) - 2.5 * math.log10(flux)) if flux > 0 else -1.0
    mag = float(mag) if (mag is not None and mag != -1.0) else -1.0

    # DAO morphology — fast skip if snr_aper < 3 (no point source likely)
    sharp = float("nan"); rnd1 = float("nan"); rnd2 = float("nan"); sep_sn = float("nan")
    if snr >= 3.0:
        try:
            fwhm_px = max(1.5, fwhm_as / ps)
            _, _, bg_std = sigma_clipped_stats(sub2, sigma=3.0, maxiters=2)
            res = DAOStarFinder(fwhm=fwhm_px, threshold=3.0 * max(bg_std, 1e-6))(sub2)
            if res is not None and len(res) > 0:
                # find nearest peak to (cx_p, cy_p)
                dxs = np.asarray(res["xcentroid"]) - cx_p
                dys = np.asarray(res["ycentroid"]) - cy_p
                dist_px = np.sqrt(dxs**2 + dys**2)
                k = int(np.argmin(dist_px))
                sep_sn = float(dist_px[k] * ps)  # arcsec
                if sep_sn <= max(0.10, 1.5 * fwhm_as):
                    sharp = float(res["sharpness"][k])
                    rnd1  = float(res["roundness1"][k])
                    rnd2  = float(res["roundness2"][k])
        except Exception:
            pass
    return cutout, mag, float(snr), sharp, rnd1, rnd2, sep_sn, nan_frac


def atomic_write_parquet(table, path):
    """Write parquet atomically via temp + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def survey_worker(args):
    """Run inference for ONE survey, checkpointing after each tile."""
    import warnings; warnings.filterwarnings("ignore")
    import sys, os, time
    # Allow numpy/scipy/photutils to use multiple BLAS threads PER worker.
    # We have 4 worker processes on M-series (8 perf cores); 2 threads each
    # gives 8 threads total without oversubscribing.
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
    import recut_3_hst as R_local
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    import pyarrow as pa
    import torch
    from cnn_models_v03 import SmallCNN

    survey = args["survey"]; in_ch = args["in_ch"]
    sources = args["sources"]   # list of (idx, ra, dec, tile)
    bands = SURVEY_BANDS[survey]
    kind  = {"hst":"hst","jwst":"jwst","vis":"euclid","nisp":"euclid"}[survey]
    band_specs = {"hst":HST_BANDS,"jwst":JWST_BANDS,"vis":VIS_BANDS,"nisp":NISP_BANDS}[survey]
    checkpoint_path = PART_DIR / f"infer_v03_{survey}.parquet"
    PART_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = SmallCNN(in_ch=in_ch).to(device)
    model.load_state_dict(args["model_state"])
    model.set_norm(torch.tensor(args["norm_mean"], device=device),
                   torch.tensor(args["norm_std"],  device=device))
    model.eval()

    # group sources by tile
    from collections import defaultdict
    by_tile = defaultdict(list)
    for (idx, ra, dec, tile) in sources:
        by_tile[tile].append((idx, ra, dec))

    N_total = len(sources)
    # accumulator: per-source dict; flushed to parquet after each tile
    acc = {"idx": [], "P": []}
    for b in bands:
        acc[f"mag_{b}"] = []
        acc[f"snr_{b}"] = []
        acc[f"sharp_{b}"] = []
        acc[f"rnd1_{b}"]  = []
        acc[f"rnd2_{b}"]  = []
        acc[f"sep_{b}"]   = []
        acc[f"nanfrac_{b}"] = []

    t0 = time.time()
    sorted_tiles = sorted(by_tile.items())
    for ti, (tile, tile_sources) in enumerate(sorted_tiles):
        # open all bands for this tile
        hdus = {}; wcss = {}; hdrs = {}
        for blabel, R_band in band_specs:
            if survey == "hst":
                path = f"{R_local.HST_DIR}/acs_I_030mas_{tile}_sci.fits"
            elif survey == "jwst":
                path = R_local.resolve_jwst_path(tile, R_band.lower())
            else:
                path = R_local.resolve_euclid_path(tile, R_band)
            if not path or not os.path.exists(path):
                hdus[blabel] = None; continue
            try:
                h = fits.open(path, memmap=True)
                sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
                hdus[blabel] = sci_hdu
                wcss[blabel] = WCS(sci_hdu.header)
                hdrs[blabel] = sci_hdu.header
            except Exception as e:
                print(f"[{survey}] open failed {Path(path).name}: {e}", flush=True)
                hdus[blabel] = None
        ref_wcs = next((w for w in wcss.values() if w is not None), None)
        if ref_wcs is None:
            # everyone in this tile gets P=nan, mags=-1
            for gidx, _, _ in tile_sources:
                acc["idx"].append(gidx); acc["P"].append(float("nan"))
                for b in bands:
                    acc[f"mag_{b}"].append(-1.0); acc[f"snr_{b}"].append(0.0)
                    acc[f"sharp_{b}"].append(float("nan"))
                    acc[f"rnd1_{b}"].append(float("nan"))
                    acc[f"rnd2_{b}"].append(float("nan"))
                    acc[f"sep_{b}"].append(float("nan"))
                    acc[f"nanfrac_{b}"].append(1.0)
            continue
        # Y-sort
        try:
            ras  = np.array([s[1] for s in tile_sources])
            decs = np.array([s[2] for s in tile_sources])
            _, ys = ref_wcs.all_world2pix(ras, decs, 0)
            order = np.argsort(np.where(np.isfinite(ys), ys, 0.0))
        except Exception:
            order = np.arange(len(tile_sources))
        ordered = [tile_sources[i] for i in order]
        # batched inference
        for batch_start in range(0, len(ordered), BATCH_CNN):
            batch = ordered[batch_start:batch_start + BATCH_CNN]
            cutouts = np.full((len(batch), in_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            mag_b   = {b: [] for b in bands}
            snr_b   = {b: [] for b in bands}
            sharp_b = {b: [] for b in bands}
            rnd1_b  = {b: [] for b in bands}
            rnd2_b  = {b: [] for b in bands}
            sep_b   = {b: [] for b in bands}
            nf_b    = {b: [] for b in bands}
            for k, (gidx, ra_v, dec_v) in enumerate(batch):
                for ci, (blabel, _) in enumerate(band_specs):
                    if hdus[blabel] is None:
                        mag_b[blabel].append(-1.0); snr_b[blabel].append(0.0)
                        sharp_b[blabel].append(float("nan"))
                        rnd1_b[blabel].append(float("nan"))
                        rnd2_b[blabel].append(float("nan"))
                        sep_b[blabel].append(float("nan"))
                        nf_b[blabel].append(1.0)
                        continue
                    c, mag, snr, sharp, rnd1, rnd2, sep_sn, nan_frac = cutout_and_aper(
                        hdus[blabel].data, hdrs[blabel], wcss[blabel],
                        ra_v, dec_v, FWHM_AS[blabel], kind=kind)
                    cutouts[k, ci] = c
                    mag_b[blabel].append(mag); snr_b[blabel].append(snr)
                    sharp_b[blabel].append(sharp)
                    rnd1_b[blabel].append(rnd1)
                    rnd2_b[blabel].append(rnd2)
                    sep_b[blabel].append(sep_sn)
                    nf_b[blabel].append(nan_frac)
            with torch.no_grad():
                xb = torch.tensor(cutouts, dtype=torch.float32, device=device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
            for k, (gidx, _, _) in enumerate(batch):
                acc["idx"].append(gidx)
                acc["P"].append(float(p[k]))
                for b in bands:
                    acc[f"mag_{b}"].append(mag_b[b][k])
                    acc[f"snr_{b}"].append(snr_b[b][k])
                    acc[f"sharp_{b}"].append(sharp_b[b][k])
                    acc[f"rnd1_{b}"].append(rnd1_b[b][k])
                    acc[f"rnd2_{b}"].append(rnd2_b[b][k])
                    acc[f"sep_{b}"].append(sep_b[b][k])
                    acc[f"nanfrac_{b}"].append(nf_b[b][k])
        # close handles
        for blabel in list(hdus):
            try:
                if hdus[blabel] is not None:
                    hdus[blabel].fileinfo()['file'].close()
            except Exception:
                pass
        # checkpoint after this tile
        table = pa.table({k: np.array(v) for k, v in acc.items()})
        try:
            atomic_write_parquet(table, checkpoint_path)
        except Exception as e:
            print(f"[{survey}] checkpoint write failed: {e}", flush=True)
        print(f"[{survey}] tile [{ti+1}/{len(sorted_tiles)}] {tile} "
              f"acc={len(acc['idx']):,}/{N_total:,} elapsed={time.time()-t0:.1f}s",
              flush=True)
    # final checkpoint
    table = pa.table({k: np.array(v) for k, v in acc.items()})
    atomic_write_parquet(table, checkpoint_path)
    print(f"[{survey}] DONE total elapsed={time.time()-t0:.1f}s wrote {checkpoint_path.name}", flush=True)
    return dict(survey=survey, elapsed=time.time()-t0, n=len(acc["idx"]))


# -------- watcher (separate process) --------
def watcher(N_total, pid_arr, psrc_arr, ra_arr, dec_arr, th_arr, tj_arr, te_arr,
            in_h, in_j, in_e, thresholds, stop_event, interval=WATCH_INTERVAL):
    """Periodic consolidation: read partials, apply coverage-aware 1-of-N rule,
    update tbl_sn_candidates_v03.csv + webpage."""
    import time
    from pathlib import Path
    import numpy as np
    import pyarrow.parquet as pq
    while True:
        time.sleep(interval)
        try:
            update_outputs(N_total, pid_arr, psrc_arr, ra_arr, dec_arr,
                           th_arr, tj_arr, te_arr, in_h, in_j, in_e, thresholds)
        except Exception as e:
            print(f"[WATCHER] update failed: {e}", flush=True)
        if stop_event.is_set():
            break
    # final pass after workers done
    try:
        update_outputs(N_total, pid_arr, psrc_arr, ra_arr, dec_arr,
                       th_arr, tj_arr, te_arr, in_h, in_j, in_e, thresholds, final=True)
    except Exception as e:
        print(f"[WATCHER] FINAL update failed: {e}", flush=True)


def update_outputs(N, pid, psrc, ra, dec, th, tj, te, in_h, in_j, in_e, thresholds, final=False):
    """Read all partial parquets, apply rule, write CSV + HTML."""
    # Gather per-survey P + mags/snrs/morphology into N-shaped arrays
    P = {s: np.full(N, np.nan, dtype=np.float32) for s in ("hst","jwst","vis","nisp")}
    MAG  = {b: np.full(N, -1.0, dtype=np.float32) for b in CSV_BANDS}
    SNR  = {b: np.full(N,  0.0, dtype=np.float32) for b in CSV_BANDS}
    SHARP = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    RND1  = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    RND2  = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    SEP   = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    NFR   = {b: np.full(N,  1.0, dtype=np.float32) for b in CSV_BANDS}
    surveys_done = {s: False for s in P}   # whether the partial parquet exists at all
    surveys_processed = {s: 0 for s in P}
    for survey in ("hst","jwst","vis","nisp"):
        cp = PART_DIR / f"infer_v03_{survey}.parquet"
        if not cp.exists(): continue
        try:
            t = pq.read_table(cp)
        except Exception:
            continue
        idxs = t["idx"].to_numpy(zero_copy_only=False)
        Ps   = t["P"].to_numpy(zero_copy_only=False)
        if len(idxs) == 0: continue
        valid = idxs >= 0
        ii = idxs[valid].astype(np.int64)
        P[survey][ii] = Ps[valid].astype(np.float32)
        cols = set(t.column_names)
        for b in SURVEY_BANDS[survey]:
            MAG[b][ii] = t[f"mag_{b}"].to_numpy(zero_copy_only=False)[valid].astype(np.float32)
            SNR[b][ii] = t[f"snr_{b}"].to_numpy(zero_copy_only=False)[valid].astype(np.float32)
            if f"sharp_{b}" in cols:
                SHARP[b][ii] = t[f"sharp_{b}"].to_numpy(zero_copy_only=False)[valid].astype(np.float32)
                RND1[b][ii]  = t[f"rnd1_{b}"].to_numpy(zero_copy_only=False)[valid].astype(np.float32)
                RND2[b][ii]  = t[f"rnd2_{b}"].to_numpy(zero_copy_only=False)[valid].astype(np.float32)
                SEP[b][ii]   = t[f"sep_{b}"].to_numpy(zero_copy_only=False)[valid].astype(np.float32)
                NFR[b][ii]   = t[f"nanfrac_{b}"].to_numpy(zero_copy_only=False)[valid].astype(np.float32)
        surveys_done[survey] = True
        surveys_processed[survey] = int(valid.sum())

    if not any(surveys_done.values()):
        return

    # Coverage-aware detection rule.
    # v03 update (user 2026-05-27): Euclid VIS and NISP are taken at the same
    # epoch, so a real Euclid SN typically fires in BOTH (or ONE if depth
    # differs). Group them: det_euclid = det_vis OR det_nisp. Per-telescope
    # 1-of-3 rule across HST / JWST / Euclid.
    thr_hst  = thresholds.get("hst",  0.5)
    thr_jwst = thresholds.get("jwst", 0.5)
    thr_vis  = thresholds.get("vis",  0.5)
    thr_nisp = thresholds.get("nisp", 0.5)
    det_hst  = np.where(in_h & np.isfinite(P["hst"]),  P["hst"]  >= thr_hst,  False)
    det_jwst = np.where(in_j & np.isfinite(P["jwst"]), P["jwst"] >= thr_jwst, False)
    det_vis  = np.where(in_e & np.isfinite(P["vis"]),  P["vis"]  >= thr_vis,  False)
    det_nisp = np.where(in_e & np.isfinite(P["nisp"]), P["nisp"] >= thr_nisp, False)
    det_euclid = det_vis | det_nisp   # NEW: VIS OR NISP counts as one Euclid detection
    # coverage AND processed
    cov_hst  = in_h & np.isfinite(P["hst"])
    cov_jwst = in_j & np.isfinite(P["jwst"])
    cov_vis  = in_e & np.isfinite(P["vis"])
    cov_nisp = in_e & np.isfinite(P["nisp"])
    # nondet where covered and processed but below threshold
    nondet_hst  = cov_hst  & ~det_hst
    nondet_jwst = cov_jwst & ~det_jwst
    nondet_vis  = cov_vis  & ~det_vis
    nondet_nisp = cov_nisp & ~det_nisp
    # 1-of-3 telescope rule (HST / JWST / Euclid). VIS and NISP collapse.
    n_det = det_hst.astype(int) + det_jwst.astype(int) + det_euclid.astype(int)
    n_cov = cov_hst.astype(int) + cov_jwst.astype(int) + cov_vis.astype(int) + cov_nisp.astype(int)
    # A source is "evaluated" (rule applicable) when every covered survey
    # WITH A TRAINED MODEL is processed. VIS is intentionally skipped (no
    # positives ever discovered there → no model trained), so we do NOT
    # require it for the evaluated flag.
    evaluated = (
        (~in_h | np.isfinite(P["hst"])) &
        (~in_j | np.isfinite(P["jwst"])) &
        (~in_e | np.isfinite(P["nisp"]))
    )
    # v03 RULES:
    #   1. Coverage: REQUIRE full HST + JWST + Euclid coverage
    #   2. Detection asymmetry: exactly 1 of 3 TELESCOPES says "detected"
    #      (HST / JWST / Euclid, where Euclid = VIS OR NISP)
    #   3. Magnitude floor: best_snr-band mag ≥ MAG_FLOOR
    #   4. snr floor: best_snr ≥ 5 in detection band
    full_cov = in_h & in_j & in_e
    is_sn = (n_det == 1) & evaluated & full_cov

    # composite confidence — best detected P × (1 − best non-detected P), with
    # VIS+NISP collapsed: best Euclid-detected P is max over VIS and NISP if
    # either is detected; non-detection penalty is also computed at the
    # telescope level.
    P_hst_d  = np.where(det_hst,  np.nan_to_num(P["hst"],  nan=0.0), 0.0)
    P_jwst_d = np.where(det_jwst, np.nan_to_num(P["jwst"], nan=0.0), 0.0)
    P_vis_d  = np.where(det_vis,  np.nan_to_num(P["vis"],  nan=0.0), 0.0)
    P_nisp_d = np.where(det_nisp, np.nan_to_num(P["nisp"], nan=0.0), 0.0)
    P_eu_d   = np.maximum(P_vis_d, P_nisp_d)
    det_survey_p = np.maximum.reduce([P_hst_d, P_jwst_d, P_eu_d])
    # non-detection penalty: max P in the telescope(s) that did NOT fire
    P_hst_nd  = np.where(~det_hst,  np.nan_to_num(P["hst"],  nan=0.0), 0.0)
    P_jwst_nd = np.where(~det_jwst, np.nan_to_num(P["jwst"], nan=0.0), 0.0)
    P_eu_nd   = np.maximum(
        np.where(~det_euclid, np.nan_to_num(P["vis"],  nan=0.0), 0.0),
        np.where(~det_euclid, np.nan_to_num(P["nisp"], nan=0.0), 0.0),
    )
    nondet_max_p = np.maximum.reduce([P_hst_nd, P_jwst_nd, P_eu_nd])
    comp_conf = det_survey_p * (1.0 - nondet_max_p)

    # Telescope label, preserving Euclid sub-label by which band(s) fired
    which = np.full(N, "", dtype=object)
    which[det_hst  & is_sn] = "HST"
    which[det_jwst & is_sn] = "JWST"
    eu_both = det_vis  & det_nisp & is_sn
    eu_visonly  = det_vis  & ~det_nisp & is_sn
    eu_nisponly = det_nisp & ~det_vis  & is_sn
    which[eu_both]      = "EUCLID-VIS+NISP"
    which[eu_visonly]   = "EUCLID-VIS"
    which[eu_nisponly]  = "EUCLID-NISP"

    # ---- CSV (apply v03 hard rules at emission) ----
    sn_idx = np.where(is_sn)[0]
    sn_idx = sn_idx[np.argsort(-comp_conf[sn_idx])]
    # Map the source's telescope label to the bands we should scan for best
    # SNR / mag. For Euclid the band set is union of VIS + NISP — covers all
    # three label variants.
    band_to_survey = {
        "F814W":"HST",
        "F115W":"JWST","F150W":"JWST","F277W":"JWST","F444W":"JWST",
        "VIS":"EUCLID-VIS","Y":"EUCLID-NISP","J":"EUCLID-NISP","H":"EUCLID-NISP",
    }
    # Which bands belong to each telescope label for best-band picking
    bands_per_label = {
        "HST":              ["F814W"],
        "JWST":             ["F115W","F150W","F277W","F444W"],
        "EUCLID-VIS":       ["VIS"],
        "EUCLID-NISP":      ["Y","J","H"],
        "EUCLID-VIS+NISP":  ["VIS","Y","J","H"],
    }

    # dN/dm prior: p(mag) ∝ 10^(0.4·mag) over the search range [MAG_FLOOR, 27].
    # Normalised so P_prior(MAG_FLOOR) = 10^(0.4·(MAG_FLOOR-27)) and P_prior(27) = 1.0.
    def dndm_prior(mag):
        if mag is None or not np.isfinite(mag) or mag <= 0: return 0.0
        if mag < MAG_FLOOR: return 0.0   # bright sources have ~0 SN prior
        if mag > 27: return 1.0           # capped; depth-limited regime
        return 10**(0.4 * (mag - 27))

    # FWHM per band for G1 (sep_sn) gate
    G1_TOL = {b: max(0.10, 1.5 * FWHM_AS[b]) for b in CSV_BANDS}
    # G2-G4 thresholds per programs_webpage/CLAUDE.md §2.1
    SHARP_LO, SHARP_HI = 0.40, 0.85
    RND_LIM = 0.50
    NF_LIM  = 0.25   # NaN/zero fraction veto on central 20x20

    def band_passes_morph(b, i):
        """G1 sep + G2 sharp + G3 rnd1 + G4 rnd2 + NaN-frac for source i band b."""
        sp = float(SHARP[b][i]); r1 = float(RND1[b][i]); r2 = float(RND2[b][i])
        sg = float(SEP[b][i]);   nf = float(NFR[b][i])
        if nf > NF_LIM: return False
        if not np.isfinite(sg) or sg > G1_TOL[b]: return False
        if not np.isfinite(sp) or sp < SHARP_LO or sp > SHARP_HI: return False
        if not np.isfinite(r1) or abs(r1) > RND_LIM: return False
        if not np.isfinite(r2) or abs(r2) > RND_LIM: return False
        return True

    rows = []
    n_filt_snr = 0; n_filt_mag = 0; n_filt_morph = 0
    n_filt_cross = 0; n_filt_nan = 0; n_filt_sat = 0
    for cand_no, i in enumerate(sn_idx, start=1):
        det_name = which[i]
        bands_for_det = bands_per_label.get(det_name, [])
        # Saturation guard: REJECT if the source's mag in any HIGH-RESOLUTION
        # band (HST F814W / JWST / Euclid VIS — pixel scale 30-100 mas) is
        # brighter than MAG_FLOOR with finite detection (snr ≥ 3). NISP YJH
        # bands are EXCLUDED because their 300-mas aperture sums host-galaxy
        # flux, giving mag 17-19 even for normal galaxies — that's not
        # saturation, just host-blending.
        SAT_BANDS = ["F814W","F115W","F150W","F277W","F444W","VIS"]
        saturated = False
        for b in SAT_BANDS:
            mv = float(MAG[b][i]); sv = float(SNR[b][i])
            if np.isfinite(mv) and 0 < mv < MAG_FLOOR and np.isfinite(sv) and sv >= 3.0:
                saturated = True; break
        if saturated:
            n_filt_sat += 1; continue
        # Pick best band by SNR among bands passing morphology G1-G4 + NaN veto
        best_band = ""; best_snr = 0.0; best_mag = -1.0
        morph_ok_any = False
        for b in bands_for_det:
            sv = float(SNR[b][i])
            if not np.isfinite(sv) or sv <= 0: continue
            if not band_passes_morph(b, i): continue
            morph_ok_any = True
            if sv > best_snr:
                best_snr = sv; best_band = b; best_mag = float(MAG[b][i])
        if not morph_ok_any:
            n_filt_morph += 1; continue
        # Cross-band consistency for multi-band telescopes (JWST, Euclid):
        # the brightest band's SNR must not exceed 10× the second-brightest
        # morph-passing band (rejects single-band flares — CR strikes / hot
        # pixels masquerading as point sources). Single-band detection labels
        # (HST, EUCLID-VIS) are exempt — no second band to compare against.
        if det_name in ("JWST", "EUCLID-NISP", "EUCLID-VIS+NISP"):
            morph_snrs = sorted(
                [float(SNR[b][i]) for b in bands_for_det
                 if band_passes_morph(b, i) and np.isfinite(SNR[b][i]) and SNR[b][i] > 0],
                reverse=True,
            )
            if len(morph_snrs) >= 2:
                if morph_snrs[0] > 10.0 * morph_snrs[1]:
                    n_filt_cross += 1; continue
            elif len(morph_snrs) == 1:
                # single-band-only detection in a multi-band telescope is
                # suspicious — require strong PSF source (snr ≥ 8).
                if morph_snrs[0] < 8.0:
                    n_filt_cross += 1; continue
        # NaN-frac veto already applied via band_passes_morph; but also
        # require the *detection* band's central 20x20 to be clean.
        if not np.isfinite(NFR[best_band][i]) or NFR[best_band][i] > NF_LIM:
            n_filt_nan += 1; continue
        # v03 hard rules
        if best_snr < 5.0:
            n_filt_snr += 1; continue
        if best_mag <= 0 or best_mag < MAG_FLOOR:
            n_filt_mag += 1; continue
        # apply dN/dm prior to composite confidence
        comp_conf[i] = float(comp_conf[i]) * dndm_prior(best_mag)
        r = {
            "id": f"cand_{cand_no:05d}",
            "primary_id": str(pid[i]),
            "telescope": det_name,
            "hst_tile": str(th[i]) if th[i] else "",
            "jwst_tile": str(tj[i]) if tj[i] else "",
            "euclid_tile": str(te[i]) if te[i] else "",
            "host_ra": f"{float(ra[i]):.7f}", "host_dec": f"{float(dec[i]):.7f}",
            "sn_ra": f"{float(ra[i]):.7f}", "sn_dec": f"{float(dec[i]):.7f}",
            "sn_host_sep_arcsec": "0.000", "host_z": "-1",
            "best_band": best_band, "best_snr": f"{best_snr:.2f}",
            "combined_sigma": f"{float(comp_conf[i]):.3f}",
            "max_snr_detect": f"{best_snr:.2f}",
            "n_3sig_detect": "1", "quality_flag": "cnn_candidate",
        }
        for b in CSV_BANDS:
            v = float(MAG[b][i])
            r[f"mag_{b}"] = f"{v:.2f}" if v != -1.0 else "-1"
        for b in CSV_BANDS:
            r[f"snr_{b}"] = f"{float(SNR[b][i]):.2f}"
        for s in ("hst","jwst","vis","nisp"):
            v = float(P[s][i]) if np.isfinite(P[s][i]) else float("nan")
            r[f"cnn_conf_{s}"] = f"{v:.4f}" if v == v else "nan"   # NaN check
        r["composite_confidence"] = f"{float(comp_conf[i]):.4f}"
        rows.append(r)

    # Re-sort rows by prior-adjusted composite confidence (descending).
    rows.sort(key=lambda r: -float(r["composite_confidence"]))
    # Re-number cand IDs so rank-1 = top candidate.
    for new_rank, r in enumerate(rows, start=1):
        r["id"] = f"cand_{new_rank:05d}"
    if rows:
        cols = list(rows[0].keys())
        tmp = OUT_CSV.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)
        tmp.replace(OUT_CSV)

    # ---- HTML ----
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    page = ["<!doctype html><html><head><title>SN search v01</title>",
            "<meta http-equiv='refresh' content='30'>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee} td.lab{text-align:left;font-weight:bold}",
            "h1{color:#333} .det{background:#e0f5e0}",
            ".prog{background:#fff5b0;padding:8px 12px;border:1px solid #d4a017;margin:10px 0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v03 &mdash; 1-of-3 telescope detection (HST / JWST / Euclid). VIS+NISP collapsed; either or both bands counts as 1 Euclid detection.</h1>",
            f"<p>Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}{' (FINAL)' if final else ' (running — refreshes every 30s)'}</p>",
            "<div class='prog'>",
            f"<b>Progress:</b><br>",
            f"HST processed: {surveys_processed['hst']:,}{' ✔' if surveys_done['hst'] else ''}<br>",
            f"JWST processed: {surveys_processed['jwst']:,}{' ✔' if surveys_done['jwst'] else ''}<br>",
            f"Euclid VIS processed: {surveys_processed['vis']:,}{' ✔' if surveys_done['vis'] else ''}<br>",
            f"Euclid NISP processed: {surveys_processed['nisp']:,}{' ✔' if surveys_done['nisp'] else ''}<br>",
            f"<br><b>Sources fully evaluated</b> (all covered surveys processed): {int(evaluated.sum()):,}<br>",
            f"<b>1-of-N raw detections:</b> {int(is_sn.sum()):,}<br>",
            f"&nbsp;&nbsp;&minus; saturation guard (any-band mag&lt;{MAG_FLOOR:.0f}): {n_filt_sat:,}<br>",
            f"&nbsp;&nbsp;&minus; morphology G1-G4 rejects: {n_filt_morph:,}<br>",
            f"&nbsp;&nbsp;&minus; cross-band consistency rejects: {n_filt_cross:,}<br>",
            f"&nbsp;&nbsp;&minus; NaN/edge veto rejects: {n_filt_nan:,}<br>",
            f"&nbsp;&nbsp;&minus; SNR&lt;5 rejects: {n_filt_snr:,}<br>",
            f"&nbsp;&nbsp;&minus; mag&lt;{MAG_FLOOR:.0f} rejects: {n_filt_mag:,}<br>",
            f"<b>SN candidates after v03 rules:</b> <span style='color:#d80'>{len(rows):,}</span>",
            "</div>",
            f"<p>Per-survey detections (overall, may include incomplete-coverage rows): "
            f"HST={int(det_hst.sum()):,}, JWST={int(det_jwst.sum()):,}, "
            f"VIS={int(det_vis.sum()):,}, NISP={int(det_nisp.sum()):,}.</p>",
            f"<p>Thresholds (LOO @ FPR=10<sup>&minus;3</sup>): "
            f"HST&ge;{thr_hst:.4f}, JWST&ge;{thr_jwst:.4f}, "
            f"VIS&ge;{thr_vis:.4f}, NISP&ge;{thr_nisp:.4f}.</p>",
            f"<p>Top {min(TOP_N_WEB, len(rows))} candidates (sorted by composite confidence):</p>",
            "<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
            "<th>comp_conf</th><th>best band</th><th>best snr</th>"
            "<th>P_hst</th><th>P_jwst</th><th>P_vis</th><th>P_nisp</th></tr></thead><tbody>"]
    for rank, r in enumerate(rows[:TOP_N_WEB], start=1):
        page.append(
            f"<tr><td>{rank}</td><td>{r['primary_id']}</td>"
            f"<td class='lab'>{r['telescope']}</td>"
            f"<td>{r['sn_ra']}</td><td>{r['sn_dec']}</td>"
            f"<td>{r['composite_confidence']}</td>"
            f"<td>{r['best_band']}</td><td>{r['best_snr']}</td>"
            f"<td>{r['cnn_conf_hst']}</td>"
            f"<td>{r['cnn_conf_jwst']}</td>"
            f"<td>{r['cnn_conf_vis']}</td>"
            f"<td>{r['cnn_conf_nisp']}</td></tr>")
    page.append("</tbody></table></div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))


def main():
    t_start = time.time()
    log("=== Step 6 v01 (parallel + streaming): inference ===")

    import torch
    log(f"Loading models from {MODELS}")
    bundle = torch.load(MODELS, map_location="cpu", weights_only=False)
    thresholds = bundle["thresholds"]
    log(f"Thresholds: {thresholds}")

    log(f"Reading lookup {LOOK}")
    lk = pq.read_table(LOOK)
    pid = np.array(lk["primary_id"].to_pylist(), dtype=object)
    psrc = np.array(lk["primary_source"].to_pylist(), dtype=object)
    ra  = lk["ra"].to_numpy(zero_copy_only=False)
    dec = lk["dec"].to_numpy(zero_copy_only=False)
    th  = np.array(lk["tile_hst"].to_pylist(), dtype=object)
    tj  = np.array(lk["tile_jwst"].to_pylist(), dtype=object)
    te  = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(lk["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(lk["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    N = len(pid)
    log(f"  {N:,} candidates")

    # clean stale partials so we don't apply old data
    PART_DIR.mkdir(parents=True, exist_ok=True)
    for survey in ("hst","jwst","vis","nisp"):
        cp = PART_DIR / f"infer_v03_{survey}.parquet"
        if cp.exists(): cp.unlink()

    # build jobs
    survey_specs = [
        ("hst",  1, in_h, th),
        ("jwst", 4, in_j, tj),
        ("vis",  1, in_e, te),
        ("nisp", 3, in_e, te),
    ]
    jobs = []
    for survey, in_ch, in_mask, tiles in survey_specs:
        if survey not in bundle["models"]:
            log(f"  [warn] model for {survey} not in bundle; skipping")
            continue
        idxs = np.where(in_mask)[0]
        sources = [(int(i), float(ra[i]), float(dec[i]), str(tiles[i])) for i in idxs if tiles[i]]
        log(f"  {survey}: {len(sources):,} sources to process")
        jobs.append(dict(
            survey=survey, in_ch=in_ch,
            model_state={k: v.detach().cpu() for k, v in bundle["models"][survey].items()},
            norm_mean=bundle["norm_mean"][survey],
            norm_std =bundle["norm_std"][survey],
            sources=sources,
        ))

    # launch watcher
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    stop_event = Event()
    watcher_p = Process(target=watcher, args=(N, pid, psrc, ra, dec, th, tj, te,
                                              in_h, in_j, in_e, thresholds, stop_event))
    watcher_p.start()
    log(f"Watcher process started (PID {watcher_p.pid}), updates every {WATCH_INTERVAL}s")

    # run survey workers in parallel
    log(f"\nDispatching {len(jobs)} survey workers ...")
    t_par = time.time()
    with Pool(processes=len(jobs)) as pool:
        results = pool.map(survey_worker, jobs)
    log(f"\nAll survey workers done in {time.time()-t_par:.1f}s")
    for res in results:
        log(f"  {res['survey']} elapsed: {res['elapsed']:.1f}s n={res['n']:,}")

    # signal watcher and wait
    stop_event.set()
    watcher_p.join(timeout=120)
    if watcher_p.is_alive():
        log("watcher hang on join — terminating")
        watcher_p.terminate()
    log(f"=== Step 6 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
