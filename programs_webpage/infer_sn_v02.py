"""Step 6 v02: inference using v02 (correctly-labeled) models.

Identical to infer_sn_v01.py architecture (parallel survey workers +
streaming watcher → CSV + webpage) but reads cnn_models_v02.pt and
writes to *_v02 outputs / htmls/sn_search/v02/.
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
PART_DIR  = CSV_DIR / "_partial_v02"
HTML_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v02")
LOOK      = CSV_DIR / "fits_lookup_v01.parquet"          # unchanged
MODELS    = CSV_DIR / "cnn_models_v02.pt"
OUT_CSV   = CSV_DIR / "tbl_sn_candidates_v02.csv"

CUT_PIX     = 64
BATCH_CNN   = 1024
TOP_N_WEB   = 500
WATCH_INTERVAL = 30

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


# -------- copy of the helper functions from v01 --------
def cutout_and_aper(sci_data, hdr, wcs, ra, dec, fwhm_as, kind, n=CUT_PIX):
    import numpy as np, math
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
    cy_p = (sy - (cy - half)); cx_p = (sx - (cx - half))
    aper_r = fwhm_as / ps; ring_in = 2.0 * fwhm_as / ps; ring_out = 3.5 * fwhm_as / ps
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - cx_p)**2 + (yy - cy_p)**2)
    in_aper = rr <= aper_r; in_ring = (rr >= ring_in) & (rr <= ring_out)
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


def atomic_write_parquet(table, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def survey_worker(args):
    import warnings; warnings.filterwarnings("ignore")
    import sys, os, time
    sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
    import recut_3_hst as R_local
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    import pyarrow as pa
    import torch
    from cnn_models_v01 import SmallCNN

    survey = args["survey"]; in_ch = args["in_ch"]
    sources = args["sources"]
    bands = SURVEY_BANDS[survey]
    kind  = {"hst":"hst","jwst":"jwst","vis":"euclid","nisp":"euclid"}[survey]
    band_specs = {"hst":HST_BANDS,"jwst":JWST_BANDS,"vis":VIS_BANDS,"nisp":NISP_BANDS}[survey]
    checkpoint_path = PART_DIR / f"infer_v02_{survey}.parquet"
    PART_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = SmallCNN(in_ch=in_ch).to(device)
    model.load_state_dict(args["model_state"])
    model.set_norm(torch.tensor(args["norm_mean"], device=device),
                   torch.tensor(args["norm_std"],  device=device))
    model.eval()

    from collections import defaultdict
    by_tile = defaultdict(list)
    for (idx, ra, dec, tile) in sources:
        by_tile[tile].append((idx, ra, dec))

    N_total = len(sources)
    acc = {"idx": [], "P": []}
    for b in bands:
        acc[f"mag_{b}"] = []; acc[f"snr_{b}"] = []

    t0 = time.time()
    sorted_tiles = sorted(by_tile.items())
    for ti, (tile, tile_sources) in enumerate(sorted_tiles):
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
            for gidx, _, _ in tile_sources:
                acc["idx"].append(gidx); acc["P"].append(float("nan"))
                for b in bands:
                    acc[f"mag_{b}"].append(-1.0); acc[f"snr_{b}"].append(0.0)
            continue
        try:
            ras  = np.array([s[1] for s in tile_sources])
            decs = np.array([s[2] for s in tile_sources])
            _, ys = ref_wcs.all_world2pix(ras, decs, 0)
            order = np.argsort(np.where(np.isfinite(ys), ys, 0.0))
        except Exception:
            order = np.arange(len(tile_sources))
        ordered = [tile_sources[i] for i in order]
        for batch_start in range(0, len(ordered), BATCH_CNN):
            batch = ordered[batch_start:batch_start + BATCH_CNN]
            cutouts = np.full((len(batch), in_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            mag_b = {b: [] for b in bands}
            snr_b = {b: [] for b in bands}
            for k, (gidx, ra_v, dec_v) in enumerate(batch):
                for ci, (blabel, _) in enumerate(band_specs):
                    if hdus[blabel] is None:
                        mag_b[blabel].append(-1.0); snr_b[blabel].append(0.0); continue
                    c, mag, snr = cutout_and_aper(hdus[blabel].data, hdrs[blabel], wcss[blabel],
                                                  ra_v, dec_v, FWHM_AS[blabel], kind=kind)
                    cutouts[k, ci] = c
                    mag_b[blabel].append(mag); snr_b[blabel].append(snr)
            with torch.no_grad():
                xb = torch.tensor(cutouts, dtype=torch.float32, device=device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
            for k, (gidx, _, _) in enumerate(batch):
                acc["idx"].append(gidx); acc["P"].append(float(p[k]))
                for b in bands:
                    acc[f"mag_{b}"].append(mag_b[b][k])
                    acc[f"snr_{b}"].append(snr_b[b][k])
        for blabel in list(hdus):
            try:
                if hdus[blabel] is not None:
                    hdus[blabel].fileinfo()['file'].close()
            except Exception:
                pass
        table = pa.table({k: np.array(v) for k, v in acc.items()})
        try:
            atomic_write_parquet(table, checkpoint_path)
        except Exception as e:
            print(f"[{survey}] ckpt write failed: {e}", flush=True)
        print(f"[{survey}] tile [{ti+1}/{len(sorted_tiles)}] {tile} "
              f"acc={len(acc['idx']):,}/{N_total:,} elapsed={time.time()-t0:.1f}s",
              flush=True)
    table = pa.table({k: np.array(v) for k, v in acc.items()})
    atomic_write_parquet(table, checkpoint_path)
    print(f"[{survey}] DONE total elapsed={time.time()-t0:.1f}s", flush=True)
    return dict(survey=survey, elapsed=time.time()-t0, n=len(acc["idx"]))


def watcher(N, pid, psrc, ra, dec, th, tj, te, in_h, in_j, in_e, thresholds, stop_event, interval=WATCH_INTERVAL):
    import time, numpy as np
    while True:
        time.sleep(interval)
        try:
            update_outputs(N, pid, psrc, ra, dec, th, tj, te, in_h, in_j, in_e, thresholds)
        except Exception as e:
            print(f"[WATCHER] update failed: {e}", flush=True)
        if stop_event.is_set():
            break
    try:
        update_outputs(N, pid, psrc, ra, dec, th, tj, te, in_h, in_j, in_e, thresholds, final=True)
    except Exception as e:
        print(f"[WATCHER] FINAL update failed: {e}", flush=True)


def update_outputs(N, pid, psrc, ra, dec, th, tj, te, in_h, in_j, in_e, thresholds, final=False):
    P = {s: np.full(N, np.nan, dtype=np.float32) for s in ("hst","jwst","vis","nisp")}
    MAG = {b: np.full(N, -1.0, dtype=np.float32) for b in CSV_BANDS}
    SNR = {b: np.full(N, 0.0, dtype=np.float32) for b in CSV_BANDS}
    surveys_done = {s: False for s in P}
    surveys_processed = {s: 0 for s in P}
    for survey in ("hst","jwst","vis","nisp"):
        cp = PART_DIR / f"infer_v02_{survey}.parquet"
        if not cp.exists(): continue
        try:
            t = pq.read_table(cp)
        except Exception: continue
        idxs = t["idx"].to_numpy(zero_copy_only=False)
        Ps = t["P"].to_numpy(zero_copy_only=False)
        if len(idxs) == 0: continue
        valid = idxs >= 0
        P[survey][idxs[valid].astype(np.int64)] = Ps[valid].astype(np.float32)
        for b in SURVEY_BANDS[survey]:
            mcol = t[f"mag_{b}"].to_numpy(zero_copy_only=False)
            scol = t[f"snr_{b}"].to_numpy(zero_copy_only=False)
            MAG[b][idxs[valid].astype(np.int64)] = mcol[valid].astype(np.float32)
            SNR[b][idxs[valid].astype(np.int64)] = scol[valid].astype(np.float32)
        surveys_done[survey] = True
        surveys_processed[survey] = int(valid.sum())

    if not any(surveys_done.values()): return

    thr_hst  = thresholds.get("hst",  0.5)
    thr_jwst = thresholds.get("jwst", 0.5)
    thr_vis  = thresholds.get("vis",  0.5)
    thr_nisp = thresholds.get("nisp", 0.5)
    det_hst  = np.where(in_h & np.isfinite(P["hst"]),  P["hst"]  >= thr_hst,  False)
    det_jwst = np.where(in_j & np.isfinite(P["jwst"]), P["jwst"] >= thr_jwst, False)
    det_vis  = np.where(in_e & np.isfinite(P["vis"]),  P["vis"]  >= thr_vis,  False)
    det_nisp = np.where(in_e & np.isfinite(P["nisp"]), P["nisp"] >= thr_nisp, False)
    cov_hst  = in_h & np.isfinite(P["hst"])
    cov_jwst = in_j & np.isfinite(P["jwst"])
    cov_vis  = in_e & np.isfinite(P["vis"])
    cov_nisp = in_e & np.isfinite(P["nisp"])
    n_det = det_hst.astype(int) + det_jwst.astype(int) + det_vis.astype(int) + det_nisp.astype(int)
    n_cov = cov_hst.astype(int) + cov_jwst.astype(int) + cov_vis.astype(int) + cov_nisp.astype(int)
    evaluated = (
        (~in_h | np.isfinite(P["hst"])) &
        (~in_j | np.isfinite(P["jwst"])) &
        (~in_e | np.isfinite(P["vis"])) &
        (~in_e | np.isfinite(P["nisp"]))
    )
    is_sn = (n_det == 1) & evaluated & (n_cov >= 2)

    P_arr = np.stack([np.nan_to_num(P[s], nan=0.0) for s in ("hst","jwst","vis","nisp")], axis=1)
    det_arr = np.stack([det_hst, det_jwst, det_vis, det_nisp], axis=1)
    det_survey_p = np.where(det_arr, P_arr, 0.0).max(axis=1)
    nondet_max_p = np.where(~det_arr, P_arr, 0.0).max(axis=1)
    comp_conf = det_survey_p * (1.0 - nondet_max_p)

    survey_names = np.array(["HST","JWST","EUCLID-VIS","EUCLID-NISP"])
    which = np.full(N, "", dtype=object)
    for k, name in enumerate(survey_names):
        col = det_arr[:, k] & is_sn
        which[col] = name

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
            sv = float(SNR[b][i])
            if np.isfinite(sv) and sv > best_snr:
                best_snr = sv; best_band = b
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
            r[f"cnn_conf_{s}"] = f"{v:.4f}" if v == v else "nan"
        r["composite_confidence"] = f"{float(comp_conf[i]):.4f}"
        rows.append(r)

    if rows:
        cols = list(rows[0].keys())
        tmp = OUT_CSV.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)
        tmp.replace(OUT_CSV)

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    page = ["<!doctype html><html><head><title>SN search v02</title>",
            "<meta http-equiv='refresh' content='30'>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee} td.lab{text-align:left;font-weight:bold}",
            "h1{color:#333} .det{background:#e0f5e0}",
            ".prog{background:#fff5b0;padding:8px 12px;border:1px solid #d4a017;margin:10px 0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v02 — corrected per-survey labels</h1>",
            f"<p>Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}{' (FINAL)' if final else ' (running — refreshes every 30s)'}</p>",
            "<div class='prog'>",
            f"<b>Progress:</b><br>",
            f"HST processed: {surveys_processed['hst']:,}{' ✔' if surveys_done['hst'] else ''}<br>",
            f"JWST processed: {surveys_processed['jwst']:,}{' ✔' if surveys_done['jwst'] else ''}<br>",
            f"Euclid VIS processed: {surveys_processed['vis']:,}{' ✔' if surveys_done['vis'] else ''}<br>",
            f"Euclid NISP processed: {surveys_processed['nisp']:,}{' ✔' if surveys_done['nisp'] else ''}<br>",
            f"<br><b>Sources fully evaluated:</b> {int(evaluated.sum()):,}<br>",
            f"<b>SN candidates so far:</b> <span style='color:#d80'>{int(is_sn.sum()):,}</span>",
            "</div>",
            f"<p>Per-survey detections (overall): HST={int(det_hst.sum()):,}, "
            f"JWST={int(det_jwst.sum()):,}, VIS={int(det_vis.sum()):,}, "
            f"NISP={int(det_nisp.sum()):,}.</p>",
            f"<p>Thresholds: HST&ge;{thr_hst:.4f}, JWST&ge;{thr_jwst:.4f}, "
            f"VIS&ge;{thr_vis:.4f}, NISP&ge;{thr_nisp:.4f}.</p>",
            f"<p>Top {min(TOP_N_WEB, len(rows))} candidates:</p>",
            "<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
            "<th>comp_conf</th><th>best band</th><th>best snr</th>"
            "<th>P_hst</th><th>P_jwst</th><th>P_vis</th><th>P_nisp</th></tr></thead><tbody>"]
    for rank, i in enumerate(sn_idx[:TOP_N_WEB], start=1):
        p_hst  = P['hst'][i]  if np.isfinite(P['hst'][i])  else 0.0
        p_jwst = P['jwst'][i] if np.isfinite(P['jwst'][i]) else 0.0
        p_vis  = P['vis'][i]  if np.isfinite(P['vis'][i])  else 0.0
        p_nisp = P['nisp'][i] if np.isfinite(P['nisp'][i]) else 0.0
        cls_h = "det" if det_hst[i]  else ""
        cls_j = "det" if det_jwst[i] else ""
        cls_v = "det" if det_vis[i]  else ""
        cls_n = "det" if det_nisp[i] else ""
        page.append(f"<tr><td>{rank}</td><td>{pid[i]}</td><td class='lab'>{which[i]}</td>"
                    f"<td>{ra[i]:.6f}</td><td>{dec[i]:.6f}</td>"
                    f"<td>{comp_conf[i]:.3f}</td>"
                    f"<td>{rows[rank-1]['best_band']}</td>"
                    f"<td>{rows[rank-1]['best_snr']}</td>"
                    f"<td class='{cls_h}'>{p_hst:.3f}</td>"
                    f"<td class='{cls_j}'>{p_jwst:.3f}</td>"
                    f"<td class='{cls_v}'>{p_vis:.3f}</td>"
                    f"<td class='{cls_n}'>{p_nisp:.3f}</td></tr>")
    page.append("</tbody></table></div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))


def main():
    t_start = time.time()
    log("=== Step 6 v02: inference with corrected models ===")

    import torch
    bundle = torch.load(MODELS, map_location="cpu", weights_only=False)
    thresholds = bundle["thresholds"]
    log(f"Thresholds: {thresholds}")

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

    PART_DIR.mkdir(parents=True, exist_ok=True)
    for survey in ("hst","jwst","vis","nisp"):
        cp = PART_DIR / f"infer_v02_{survey}.parquet"
        if cp.exists(): cp.unlink()

    survey_specs = [("hst",1,in_h,th),("jwst",4,in_j,tj),("vis",1,in_e,te),("nisp",3,in_e,te)]
    jobs = []
    for survey, in_ch, in_mask, tiles in survey_specs:
        if survey not in bundle["models"]:
            continue
        idxs = np.where(in_mask)[0]
        sources = [(int(i), float(ra[i]), float(dec[i]), str(tiles[i])) for i in idxs if tiles[i]]
        log(f"  {survey}: {len(sources):,} sources")
        jobs.append(dict(
            survey=survey, in_ch=in_ch,
            model_state={k: v.detach().cpu() for k, v in bundle["models"][survey].items()},
            norm_mean=bundle["norm_mean"][survey],
            norm_std =bundle["norm_std"][survey],
            sources=sources,
        ))

    try: set_start_method("spawn", force=True)
    except RuntimeError: pass
    stop_event = Event()
    watcher_p = Process(target=watcher, args=(N, pid, psrc, ra, dec, th, tj, te,
                                              in_h, in_j, in_e, thresholds, stop_event))
    watcher_p.start()
    log(f"Watcher started PID {watcher_p.pid}, updates every {WATCH_INTERVAL}s")
    t_par = time.time()
    with Pool(processes=len(jobs)) as pool:
        results = pool.map(survey_worker, jobs)
    log(f"All workers done in {time.time()-t_par:.1f}s")
    for res in results:
        log(f"  {res['survey']} elapsed: {res['elapsed']:.1f}s n={res['n']:,}")
    stop_event.set()
    watcher_p.join(timeout=120)
    if watcher_p.is_alive():
        watcher_p.terminate()
    log(f"=== Step 6 v02 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
