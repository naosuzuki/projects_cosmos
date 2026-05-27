"""Step 6 v01 (parallel + MPS): inference at 1.12M scale.

PARALLELISED via multiprocessing.Pool — one worker per survey
(hst / jwst / vis / nisp). Each worker:
  - loads its CNN onto MPS (independent context)
  - iterates over its survey's tiles in a tile-major loop
  - extracts cutouts + runs CNN forward pass in batched MPS calls
  - returns per-source P + aperture photometry per band

The 4 surveys run truly in parallel (1 process each, separate MPS contexts).
On a Mac Mini Apple Silicon the GPU is shared, so the 4 processes will queue
at the GPU driver — but their IO + CPU overlap. Net: ~2-4× speedup vs serial.

Outputs:
  csvfiles_sn/tbl_sn_candidates_v01.csv             (v32-schema candidates only)
  csvfiles_sn/sn_candidates_v01_all_scored.parquet  (per-source scores for ALL)
  htmls/sn_search/v01/index.html                    (top-N webpage)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, os, csv, math, json
from pathlib import Path
from multiprocessing import Pool, set_start_method
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
HTML_DIR = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v01")
LOOK     = CSV_DIR / "fits_lookup_v01.parquet"
MODELS   = CSV_DIR / "cnn_models_v01.pt"
OUT_CSV  = CSV_DIR / "tbl_sn_candidates_v01.csv"
OUT_PARQ = CSV_DIR / "sn_candidates_v01_all_scored.parquet"

CUT_PIX     = 64
BATCH_CNN   = 1024     # larger batches for MPS
TOP_N_WEB   = 500

HST_BANDS  = [("F814W","F814W")]
JWST_BANDS = [("F115W","f115w"),("F150W","f150w"),("F277W","f277w"),("F444W","f444w")]
VIS_BANDS  = [("VIS","VIS")]
NISP_BANDS = [("Y","NIR-Y"),("J","NIR-J"),("H","NIR-H")]

FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}

CSV_BANDS = ["F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W"]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# -------- shared helpers (used in workers) --------
def cutout_and_aper(sci_data, hdr, wcs, ra, dec, fwhm_as, kind, n=CUT_PIX):
    try:
        sx, sy = wcs.all_world2pix(ra, dec, 0)
    except Exception:
        return np.full((n, n), np.nan, dtype=np.float32), -1.0, 0.0
    sx = float(sx); sy = float(sy)
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return np.full((n, n), np.nan, dtype=np.float32), -1.0, 0.0
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
        return cutout, -1.0, 0.0
    sub = np.where(np.isfinite(cutout), cutout, 0.0).astype(np.float64)
    cy_p = (sy - (cy - half))
    cx_p = (sx - (cx - half))
    aper_r = fwhm_as / ps
    ring_in = 2.0 * fwhm_as / ps
    ring_out = 3.5 * fwhm_as / ps
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - cx_p)**2 + (yy - cy_p)**2)
    in_aper = rr <= aper_r
    in_ring = (rr >= ring_in) & (rr <= ring_out)
    if not in_aper.any() or not in_ring.any():
        return cutout, -1.0, 0.0
    bg = float(np.nanmedian(sub[in_ring]))
    sub2 = sub - bg
    n_aper = int(in_aper.sum())
    flux = float(np.nansum(sub2[in_aper]))
    sig = float(np.nanstd(sub[in_ring])) * math.sqrt(n_aper)
    snr = flux / (sig + 1e-30)
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
    return cutout, float(mag) if mag != -1.0 else -1.0, float(snr)


def survey_worker(args):
    """Run inference for ONE entire survey in this process.

    args = dict with:
      survey, in_ch, model_state, norm_mean, norm_std
      sources: list of (idx, ra, dec, tile)
    Returns dict with:
      survey, idxs (N,), P (N,), mag_dict (band -> (N,)), snr_dict (band -> (N,))
    """
    import warnings; warnings.filterwarnings("ignore")
    import sys, os, math, time
    sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
    import recut_3_hst as R_local
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    import torch
    from cnn_models_v01 import SmallCNN

    survey = args["survey"]; in_ch = args["in_ch"]
    sources = args["sources"]   # list of (idx, ra, dec, tile)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # load model
    model = SmallCNN(in_ch=in_ch).to(device)
    model.load_state_dict(args["model_state"])
    model.set_norm(torch.tensor(args["norm_mean"], device=device),
                   torch.tensor(args["norm_std"],  device=device))
    model.eval()

    if survey == "hst":   band_specs = HST_BANDS;  kind = "hst"
    elif survey == "jwst": band_specs = JWST_BANDS; kind = "jwst"
    elif survey == "vis":  band_specs = VIS_BANDS;  kind = "euclid"
    elif survey == "nisp": band_specs = NISP_BANDS; kind = "euclid"
    else: raise ValueError(survey)

    # group sources by tile
    from collections import defaultdict
    by_tile = defaultdict(list)
    for (idx, ra, dec, tile) in sources:
        by_tile[tile].append((idx, ra, dec))

    N_total = len(sources)
    out_idx = np.full(N_total, -1, dtype=np.int64)
    out_P   = np.full(N_total, np.nan, dtype=np.float32)
    out_mag = {b: np.full(N_total, -1.0, dtype=np.float32) for (b, _) in band_specs}
    out_snr = {b: np.full(N_total, 0.0,  dtype=np.float32) for (b, _) in band_specs}
    has_band = {b: np.zeros(N_total, dtype=bool) for (b, _) in band_specs}

    n_filled = 0
    t0 = time.time()
    for ti, (tile, tile_sources) in enumerate(sorted(by_tile.items())):
        # Open all bands for this tile.
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
            n_filled += len(tile_sources)
            continue
        # sort by pixel-Y
        try:
            ras = np.array([s[1] for s in tile_sources])
            decs = np.array([s[2] for s in tile_sources])
            _, ys = ref_wcs.all_world2pix(ras, decs, 0)
            order = np.argsort(np.where(np.isfinite(ys), ys, 0.0))
        except Exception:
            order = np.arange(len(tile_sources))
        ordered = [tile_sources[i] for i in order]
        # batched inference over this tile
        for batch_start in range(0, len(ordered), BATCH_CNN):
            batch = ordered[batch_start:batch_start + BATCH_CNN]
            cutouts = np.full((len(batch), in_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for k, (gidx, ra_v, dec_v) in enumerate(batch):
                for ci, (blabel, _) in enumerate(band_specs):
                    if hdus[blabel] is None: continue
                    c, mag, snr = cutout_and_aper(hdus[blabel].data, hdrs[blabel], wcss[blabel],
                                                  ra_v, dec_v, FWHM_AS[blabel], kind=kind)
                    cutouts[k, ci] = c
                    if np.any(np.isfinite(c)):
                        has_band[blabel][n_filled + k] = True
                        out_mag[blabel][n_filled + k] = mag
                        out_snr[blabel][n_filled + k] = snr
            # CNN inference
            with torch.no_grad():
                xb = torch.tensor(cutouts, dtype=torch.float32, device=device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
            # save back in original n_filled-based indexing
            for k, (gidx, _, _) in enumerate(batch):
                out_idx[n_filled + k] = gidx
                out_P[n_filled + k]   = p[k]
            n_filled += len(batch)
        # close FITS handles
        for blabel in list(hdus):
            try:
                if hdus[blabel] is not None:
                    hdus[blabel].fileinfo()['file'].close()
            except Exception:
                pass
        if (ti + 1) % 5 == 0 or ti == len(by_tile) - 1:
            print(f"[{survey}] tile [{ti+1}/{len(by_tile)}] {tile}  "
                  f"sources={n_filled:,}/{N_total:,}  elapsed={time.time()-t0:.1f}s",
                  flush=True)
    return dict(survey=survey, idxs=out_idx, P=out_P,
                mag={b: out_mag[b] for (b, _) in band_specs},
                snr={b: out_snr[b] for (b, _) in band_specs},
                elapsed=time.time()-t0)


def main():
    t_start = time.time()
    log("=== Step 6 v01 (parallel): inference ===")

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

    survey_specs = [
        ("hst",  1, in_h, th),
        ("jwst", 4, in_j, tj),
        ("vis",  1, in_e, te),
        ("nisp", 3, in_e, te),
    ]

    # Build jobs: each gets the source subset that is in-coverage for its survey.
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
            model_state={k: v.detach().cpu() for k, v in
                          {**bundle["models"][survey]}.items()},
            norm_mean=bundle["norm_mean"][survey],
            norm_std =bundle["norm_std"][survey],
            sources=sources,
        ))

    # Run 4 surveys in parallel.
    log(f"\nDispatching {len(jobs)} survey workers ...")
    t_par = time.time()
    # Use 'spawn' for clean MPS contexts
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    with Pool(processes=len(jobs)) as pool:
        results = pool.map(survey_worker, jobs)
    log(f"\nAll survey workers done in {time.time()-t_par:.1f}s")
    for res in results:
        log(f"  {res['survey']} elapsed: {res['elapsed']:.1f}s")

    # Stitch results into global N arrays.
    P = {s: np.full(N, np.nan, dtype=np.float32) for s in ("hst","jwst","vis","nisp")}
    MAG = {b: np.full(N, -1.0, dtype=np.float32) for b in CSV_BANDS}
    SNR = {b: np.full(N, 0.0,  dtype=np.float32) for b in CSV_BANDS}
    for res in results:
        s = res["survey"]
        valid = res["idxs"] >= 0
        P[s][res["idxs"][valid]] = res["P"][valid]
        for b, arr in res["mag"].items():
            MAG[b][res["idxs"][valid]] = arr[valid]
        for b, arr in res["snr"].items():
            SNR[b][res["idxs"][valid]] = arr[valid]

    # 1-of-4 detection rule
    thr_hst = thresholds.get("hst",  0.5)
    thr_jwst = thresholds.get("jwst", 0.5)
    thr_vis = thresholds.get("vis",  0.5)
    thr_nisp = thresholds.get("nisp", 0.5)
    det_hst  = np.where(np.isfinite(P["hst"]),  P["hst"]  >= thr_hst,  False)
    det_jwst = np.where(np.isfinite(P["jwst"]), P["jwst"] >= thr_jwst, False)
    det_vis  = np.where(np.isfinite(P["vis"]),  P["vis"]  >= thr_vis,  False)
    det_nisp = np.where(np.isfinite(P["nisp"]), P["nisp"] >= thr_nisp, False)
    n_det = det_hst.astype(int) + det_jwst.astype(int) + det_vis.astype(int) + det_nisp.astype(int)
    is_sn = (n_det == 1)
    log(f"\n=== detection summary ===")
    log(f"  HST detected:        {int(det_hst.sum()):>10,}")
    log(f"  JWST detected:       {int(det_jwst.sum()):>10,}")
    log(f"  VIS detected:        {int(det_vis.sum()):>10,}")
    log(f"  NISP detected:       {int(det_nisp.sum()):>10,}")
    log(f"  SN candidates (1-of-4): {int(is_sn.sum()):>10,}")

    # composite confidence
    P_arr = np.stack([np.nan_to_num(P[s], nan=0.0) for s in ("hst","jwst","vis","nisp")], axis=1)
    det_arr = np.stack([det_hst,det_jwst,det_vis,det_nisp], axis=1)
    det_survey_p = np.where(det_arr, P_arr, 0.0).max(axis=1)
    nondet_max_p = np.where(~det_arr, P_arr, 0.0).max(axis=1)
    comp_conf = det_survey_p * (1.0 - nondet_max_p)

    survey_names = np.array(["HST","JWST","EUCLID-VIS","EUCLID-NISP"])
    which = np.full(N, "", dtype=object)
    for k, name in enumerate(survey_names):
        col = det_arr[:, k] & is_sn
        which[col] = name

    # Write parquet (all sources)
    log(f"\nWriting all-scored parquet → {OUT_PARQ}")
    out_table = pa.table({
        "primary_id": pid.tolist(),
        "primary_source": psrc.tolist(),
        "ra": ra, "dec": dec,
        "tile_hst": th.tolist(),
        "tile_jwst": tj.tolist(),
        "tile_euclid": te.tolist(),
        "P_hst":  P["hst"], "P_jwst": P["jwst"], "P_vis": P["vis"], "P_nisp": P["nisp"],
        "det_hst": det_hst, "det_jwst": det_jwst, "det_vis": det_vis, "det_nisp": det_nisp,
        "n_surveys_detected": n_det.astype(np.int8),
        "is_sn_candidate": is_sn,
        "composite_confidence": comp_conf,
        "telescope_detected": which.tolist(),
        **{f"mag_{b}": MAG[b] for b in CSV_BANDS},
        **{f"snr_{b}": SNR[b] for b in CSV_BANDS},
    })
    pq.write_table(out_table, OUT_PARQ, compression="zstd")

    # CSV in v32 schema
    log(f"Writing CSV → {OUT_CSV}")
    sn_idx = np.where(is_sn)[0]
    sn_idx = sn_idx[np.argsort(-comp_conf[sn_idx])]
    band_to_survey = {"F814W":"HST","F115W":"JWST","F150W":"JWST","F277W":"JWST","F444W":"JWST",
                      "VIS":"EUCLID-VIS","Y":"EUCLID-NISP","J":"EUCLID-NISP","H":"EUCLID-NISP"}
    rows = []
    for cand_no, i in enumerate(sn_idx, start=1):
        det_name = which[i]
        bands_for_det = [b for b, s in band_to_survey.items() if s == det_name]
        best_band = ""; best_snr = 0.0
        for b in bands_for_det:
            s = float(SNR[b][i])
            if np.isfinite(s) and s > best_snr:
                best_snr = s; best_band = b
        r = {
            "id": f"cand_{cand_no:05d}",
            "primary_id": str(pid[i]),
            "telescope": det_name,
            "hst_tile": str(th[i]) if th[i] else "",
            "jwst_tile": str(tj[i]) if tj[i] else "",
            "euclid_tile": str(te[i]) if te[i] else "",
            "host_ra": f"{float(ra[i]):.7f}",
            "host_dec": f"{float(dec[i]):.7f}",
            "sn_ra": f"{float(ra[i]):.7f}",
            "sn_dec": f"{float(dec[i]):.7f}",
            "sn_host_sep_arcsec": "0.000",
            "host_z": "-1",
            "best_band": best_band,
            "best_snr": f"{best_snr:.2f}",
            "combined_sigma": f"{float(comp_conf[i]):.3f}",
            "max_snr_detect": f"{best_snr:.2f}",
            "n_3sig_detect": "1",
            "quality_flag": "cnn_candidate",
        }
        for b in CSV_BANDS:
            v = float(MAG[b][i])
            r[f"mag_{b}"] = f"{v:.2f}" if v != -1.0 else "-1"
        for b in CSV_BANDS:
            v = float(SNR[b][i])
            r[f"snr_{b}"] = f"{v:.2f}"
        r["cnn_conf_hst"]  = f"{float(P['hst'][i]):.4f}"  if np.isfinite(P['hst'][i])  else "nan"
        r["cnn_conf_jwst"] = f"{float(P['jwst'][i]):.4f}" if np.isfinite(P['jwst'][i]) else "nan"
        r["cnn_conf_vis"]  = f"{float(P['vis'][i]):.4f}"  if np.isfinite(P['vis'][i])  else "nan"
        r["cnn_conf_nisp"] = f"{float(P['nisp'][i]):.4f}" if np.isfinite(P['nisp'][i]) else "nan"
        r["composite_confidence"] = f"{float(comp_conf[i]):.4f}"
        rows.append(r)

    if rows:
        cols = list(rows[0].keys())
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)
        log(f"Wrote {len(rows):,} candidates → {OUT_CSV}")
    else:
        log("No SN candidates found.")

    # webpage
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Writing webpage → {HTML_DIR}/index.html")
    page = ["<!doctype html><html><head><title>SN search v01</title>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee} td.lab{text-align:left;font-weight:bold}",
            "h1{color:#333} .det{background:#e0f5e0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v01 — 1-of-4 detection across HST / JWST / Euclid-VIS / Euclid-NISP</h1>",
            f"<p>Built: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>Total candidates evaluated: {N:,}<br>",
            f"Detected in exactly one survey (= SN candidate): <b>{int(is_sn.sum()):,}</b><br>",
            f"Top {min(TOP_N_WEB, len(rows))} shown below (by composite confidence).</p>",
            f"<p>Per-survey detection counts: HST={int(det_hst.sum()):,}, "
            f"JWST={int(det_jwst.sum()):,}, VIS={int(det_vis.sum()):,}, "
            f"NISP={int(det_nisp.sum()):,}.</p>",
            f"<p>Thresholds (LOO @ FPR=10<sup>&minus;3</sup>): "
            f"HST&ge;{thr_hst:.4f}, JWST&ge;{thr_jwst:.4f}, "
            f"VIS&ge;{thr_vis:.4f}, NISP&ge;{thr_nisp:.4f}.</p>",
            "<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
            "<th>comp_conf</th><th>best band</th><th>best snr</th>"
            "<th>P_hst</th><th>P_jwst</th><th>P_vis</th><th>P_nisp</th></tr></thead><tbody>"]
    for rank, i in enumerate(sn_idx[:TOP_N_WEB], start=1):
        page.append(f"<tr><td>{rank}</td><td>{pid[i]}</td><td class='lab'>{which[i]}</td>"
                    f"<td>{ra[i]:.6f}</td><td>{dec[i]:.6f}</td>"
                    f"<td>{comp_conf[i]:.3f}</td>"
                    f"<td>{rows[rank-1]['best_band']}</td>"
                    f"<td>{rows[rank-1]['best_snr']}</td>"
                    f"<td class='{'det' if det_hst[i] else ''}'>{P['hst'][i]:.3f}</td>"
                    f"<td class='{'det' if det_jwst[i] else ''}'>{P['jwst'][i]:.3f}</td>"
                    f"<td class='{'det' if det_vis[i] else ''}'>{P['vis'][i]:.3f}</td>"
                    f"<td class='{'det' if det_nisp[i] else ''}'>{P['nisp'][i]:.3f}</td></tr>")
    page.append("</tbody></table></div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))
    log(f"=== Step 6 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
