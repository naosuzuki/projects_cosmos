"""Step 6 v01: inference at 1.12M scale + CSV + webpage.

Inputs:
  csvfiles_sn/sn_candidates_v01.parquet
  csvfiles_sn/fits_lookup_v01.parquet
  csvfiles_sn/cnn_models_v01.pt        (state dicts + thresholds)

Outputs:
  csvfiles_sn/tbl_sn_candidates_v01.csv
  csvfiles_sn/sn_candidates_v01_all_scored.parquet   (per-source scores for all 1.12M)
  htmls/sn_search/v01/  (top-N webpage with cutout PNGs)

Tile-major inner loop:
  For each (survey, tile):
    open the relevant FITS files once (HST=1 band, JWST=4 bands, VIS=1, NISP=3)
    extract 64×64 cutouts for every candidate in that tile, sorted by Y pix
    run CNN inference per survey
    compute simple aperture photometry for each band
    accumulate per-source result
  After all tiles: combine per-survey probs and apply the 1-of-4 detection rule.

The single most expensive part is the cutout slicing. Cutout = small (16KB)
slice from a memmapped FITS — with sources sorted by Y pix the OS page cache
gives near-RAM-speed reads. CNN inference is batched per tile.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, os, json, math, csv, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from astropy.io import fits
from astropy.wcs import WCS
import torch

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R
from cnn_models_v01 import SmallCNN, best_device

CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
HTML_DIR = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v01")
CAND = CSV_DIR / "sn_candidates_v01.parquet"
LOOK = CSV_DIR / "fits_lookup_v01.parquet"
MODELS = CSV_DIR / "cnn_models_v01.pt"
OUT_CSV = CSV_DIR / "tbl_sn_candidates_v01.csv"
OUT_PARQ = CSV_DIR / "sn_candidates_v01_all_scored.parquet"
STATUS_LOG = CSV_DIR / "run_v01_status.log"

CUT_PIX = 64
BATCH_CNN = 256
TOP_N_WEB = 500

# Band specs (survey, band_label, R_band)
HST_BANDS  = [("F814W","F814W")]
JWST_BANDS = [("F115W","f115w"),("F150W","f150w"),("F277W","f277w"),("F444W","f444w")]
VIS_BANDS  = [("VIS","VIS")]
NISP_BANDS = [("Y","NIR-Y"),("J","NIR-J"),("H","NIR-H")]

# PSF FWHM (arcsec) per band — for aperture photometry sizing
FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}

# Order of bands in the CSV per the v32 schema
CSV_BANDS = ["F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W"]


def log(msg):
    t = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with STATUS_LOG.open("a") as f:
        f.write(line + "\n")


def cutout_and_aper(sci_hdu, wcs, ra, dec, fwhm_as, n=CUT_PIX, kind="jwst"):
    """Return (cutout_64x64_float32, mag_aper, snr_aper).
    Aperture: r=1·FWHM, sky ring 2·–3.5·FWHM, all at native scale.
    NaN-fill cutout if out of bounds; (-1, 0) for mag/snr if can't measure."""
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
    ny, nx = sci_hdu.data.shape[-2:]
    x0 = cx - half; x1 = x0 + n
    y0 = cy - half; y1 = y0 + n
    cutout = np.full((n, n), np.nan, dtype=np.float32)
    if x1 > 0 and y1 > 0 and x0 < nx and y0 < ny:
        xs0 = max(0, -x0); ys0 = max(0, -y0)
        xs1 = n - max(0, x1 - nx); ys1 = n - max(0, y1 - ny)
        src_x0 = max(0, x0); src_y0 = max(0, y0)
        src_x1 = min(nx, x1); src_y1 = min(ny, y1)
        if src_x1 > src_x0 and src_y1 > src_y0:
            cutout[ys0:ys1, xs0:xs1] = sci_hdu.data[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    # aperture photometry on the cutout
    if not np.any(np.isfinite(cutout)):
        return cutout, -1.0, 0.0
    sub = np.where(np.isfinite(cutout), cutout, 0.0).astype(np.float64)
    cy_p = (sy - (cy - half))   # subpixel offset within cutout
    cx_p = (sx - (cx - half))
    aper_px = max(1.0, FWHM_AS.get("DEFAULT", fwhm_as) / ps)
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
    # mag (AB)
    hdr = sci_hdu.header
    bunit = (hdr.get("BUNIT") or "").strip()
    if kind == "jwst" or "MJy/sr" in bunit:
        pix_sr = hdr.get("PIXAR_SR") or ((ps / 206265.0) ** 2)
        f_jy = flux * float(pix_sr) * 1.0e6
        mag = (-2.5 * math.log10(f_jy / 3631.0)) if f_jy > 0 else -1.0
    elif kind == "hst":
        zp = hdr.get("ABMAG_ZP") or hdr.get("ABMAGZP") or hdr.get("MAGZP") or 25.937
        mag = (float(zp) - 2.5 * math.log10(flux)) if flux > 0 else -1.0
    else:  # euclid
        zp = hdr.get("ZP") or hdr.get("MAGZP") or hdr.get("PHOTZP") or 23.9
        mag = (float(zp) - 2.5 * math.log10(flux)) if flux > 0 else -1.0
    return cutout, float(mag) if mag != -1.0 else -1.0, float(snr)


def open_or_none(path):
    if not path or not os.path.exists(path): return None, None
    try:
        h = R._open_cached(path)
        sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = R._wcs_cached(path, sci_hdu)
        return sci_hdu, wcs
    except Exception as e:
        log(f"  open failed: {Path(path).name}: {type(e).__name__}: {e}")
        return None, None


def predict_batch(model, X, device):
    model.eval()
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), BATCH_CNN):
            xb = torch.tensor(X[i:i+BATCH_CNN], dtype=torch.float32, device=device)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            out[i:i+BATCH_CNN] = p
    return out


def main():
    t_start = time.time()
    log("=== Step 6 v01: inference ===")
    device = best_device()
    log(f"Device: {device}")

    # ---- load models ----
    log(f"Loading models from {MODELS}")
    bundle = torch.load(MODELS, map_location=device, weights_only=False)
    models = {}
    for survey, in_ch in (("hst",1),("jwst",4),("vis",1),("nisp",3)):
        if survey not in bundle["models"]:
            log(f"  [warn] model for {survey} missing — skipping that survey")
            continue
        m = SmallCNN(in_ch=in_ch).to(device)
        m.load_state_dict(bundle["models"][survey])
        med = torch.tensor(bundle["norm_mean"][survey], device=device)
        sig = torch.tensor(bundle["norm_std"][survey],  device=device)
        m.set_norm(med, sig)
        m.eval()
        models[survey] = m
    thresholds = bundle["thresholds"]
    log(f"Thresholds: {thresholds}")

    # ---- load lookup + candidates ----
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

    # initialise per-source results
    P = {s: np.full(N, np.nan, dtype=np.float32) for s in ("hst","jwst","vis","nisp")}
    MAG = {b: np.full(N, -1.0, dtype=np.float32) for b in CSV_BANDS}
    SNR = {b: np.full(N, 0.0, dtype=np.float32) for b in CSV_BANDS}
    HAS_BAND = {b: np.zeros(N, dtype=bool) for b in CSV_BANDS}

    # ---- HST pass (per HST tile) ----
    if "hst" in models:
        hst_tiles = sorted(set(t for t in th if t and str(t) != "nan"))
        log(f"\n--- HST inference: {len(hst_tiles)} tiles ---")
        t0 = time.time()
        for ti, tile in enumerate(hst_tiles):
            idx = np.where(th == tile)[0]
            path = f"{R.HST_DIR}/acs_I_030mas_{tile}_sci.fits"
            sci, wcs = open_or_none(path)
            if sci is None:
                log(f"  HST tile {tile}: no path; {len(idx)} sources skipped")
                continue
            # sort by Y for sequential disk access
            try:
                _, ys = wcs.all_world2pix(ra[idx], dec[idx], 0)
                order = np.argsort(ys)
                idx = idx[order]
            except Exception:
                pass
            cutouts = np.full((len(idx), 1, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for k, i in enumerate(idx):
                c, mag, snr = cutout_and_aper(sci, wcs, ra[i], dec[i], FWHM_AS["F814W"], kind="hst")
                cutouts[k, 0] = c
                if np.any(np.isfinite(c)):
                    HAS_BAND["F814W"][i] = True
                    MAG["F814W"][i] = mag
                    SNR["F814W"][i] = snr
            probs = predict_batch(models["hst"], cutouts, device)
            for k, i in enumerate(idx):
                P["hst"][i] = probs[k]
            if (ti + 1) % 5 == 0 or ti == len(hst_tiles) - 1:
                log(f"  HST [{ti+1}/{len(hst_tiles)}] tile {tile} sources={len(idx)} elapsed={time.time()-t0:.1f}s")
        log(f"HST pass done in {time.time()-t0:.1f}s")

    # ---- JWST pass (per JWST tile) ----
    if "jwst" in models:
        jwst_tiles = sorted(set(t for t in tj if t and str(t) != "nan"))
        log(f"\n--- JWST inference: {len(jwst_tiles)} tiles ---")
        t0 = time.time()
        for ti, tile in enumerate(jwst_tiles):
            idx = np.where(tj == tile)[0]
            # open all 4 NIRCam bands for this tile (single source loop)
            hdus = {}; wcss = {}
            for blabel, R_band in JWST_BANDS:
                path = R.resolve_jwst_path(tile, R_band.lower())
                sci, wcs = open_or_none(path)
                hdus[blabel] = sci; wcss[blabel] = wcs
            ref_wcs = next((w for w in wcss.values() if w is not None), None)
            if ref_wcs is None:
                log(f"  JWST tile {tile}: no paths; {len(idx)} sources skipped")
                continue
            try:
                _, ys = ref_wcs.all_world2pix(ra[idx], dec[idx], 0)
                order = np.argsort(ys)
                idx = idx[order]
            except Exception:
                pass
            cutouts = np.full((len(idx), 4, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for k, i in enumerate(idx):
                for ci, (blabel, _) in enumerate(JWST_BANDS):
                    if hdus[blabel] is None: continue
                    c, mag, snr = cutout_and_aper(hdus[blabel], wcss[blabel], ra[i], dec[i],
                                                  FWHM_AS[blabel], kind="jwst")
                    cutouts[k, ci] = c
                    if np.any(np.isfinite(c)):
                        HAS_BAND[blabel][i] = True
                        MAG[blabel][i] = mag
                        SNR[blabel][i] = snr
            probs = predict_batch(models["jwst"], cutouts, device)
            for k, i in enumerate(idx):
                P["jwst"][i] = probs[k]
            if (ti + 1) % 2 == 0 or ti == len(jwst_tiles) - 1:
                log(f"  JWST [{ti+1}/{len(jwst_tiles)}] tile {tile} sources={len(idx)} elapsed={time.time()-t0:.1f}s")
        log(f"JWST pass done in {time.time()-t0:.1f}s")

    # ---- Euclid pass per tile (VIS + NISP YJH share the tile id) ----
    eu_tiles = sorted(set(t for t in te if t and str(t) != "nan"))
    log(f"\n--- Euclid inference: {len(eu_tiles)} tiles ---")
    t0 = time.time()
    for ti, tile in enumerate(eu_tiles):
        idx = np.where(te == tile)[0]
        # open VIS + 3 NISP
        hdus = {}; wcss = {}
        for blabel, R_band in VIS_BANDS + NISP_BANDS:
            path = R.resolve_euclid_path(tile, R_band)
            sci, wcs = open_or_none(path)
            hdus[blabel] = sci; wcss[blabel] = wcs
        ref_wcs = next((w for w in wcss.values() if w is not None), None)
        if ref_wcs is None:
            log(f"  Euclid tile {tile}: no paths; {len(idx)} sources skipped")
            continue
        try:
            _, ys = ref_wcs.all_world2pix(ra[idx], dec[idx], 0)
            order = np.argsort(ys)
            idx = idx[order]
        except Exception:
            pass
        # VIS cutouts (1 channel)
        if "vis" in models:
            cuts_vis = np.full((len(idx), 1, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for k, i in enumerate(idx):
                if hdus["VIS"] is None: continue
                c, mag, snr = cutout_and_aper(hdus["VIS"], wcss["VIS"], ra[i], dec[i],
                                              FWHM_AS["VIS"], kind="euclid")
                cuts_vis[k, 0] = c
                if np.any(np.isfinite(c)):
                    HAS_BAND["VIS"][i] = True
                    MAG["VIS"][i] = mag
                    SNR["VIS"][i] = snr
            probs = predict_batch(models["vis"], cuts_vis, device)
            for k, i in enumerate(idx):
                P["vis"][i] = probs[k]
        # NISP cutouts (3 channels)
        if "nisp" in models:
            cuts_nisp = np.full((len(idx), 3, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            for k, i in enumerate(idx):
                for ci, (blabel, _) in enumerate(NISP_BANDS):
                    if hdus[blabel] is None: continue
                    c, mag, snr = cutout_and_aper(hdus[blabel], wcss[blabel], ra[i], dec[i],
                                                  FWHM_AS[blabel], kind="euclid")
                    cuts_nisp[k, ci] = c
                    if np.any(np.isfinite(c)):
                        HAS_BAND[blabel][i] = True
                        MAG[blabel][i] = mag
                        SNR[blabel][i] = snr
            probs = predict_batch(models["nisp"], cuts_nisp, device)
            for k, i in enumerate(idx):
                P["nisp"][i] = probs[k]
        if (ti + 1) % 5 == 0 or ti == len(eu_tiles) - 1:
            log(f"  Euclid [{ti+1}/{len(eu_tiles)}] tile {tile} sources={len(idx)} elapsed={time.time()-t0:.1f}s")
    log(f"Euclid pass done in {time.time()-t0:.1f}s")

    # ---- combine: 1-of-4 detection rule ----
    thr_hst = thresholds.get("hst", 0.5)
    thr_jwst = thresholds.get("jwst", 0.5)
    thr_vis = thresholds.get("vis", 0.5)
    thr_nisp = thresholds.get("nisp", 0.5)
    det_hst  = np.where(np.isfinite(P["hst"]),  P["hst"]  >= thr_hst,  False)
    det_jwst = np.where(np.isfinite(P["jwst"]), P["jwst"] >= thr_jwst, False)
    det_vis  = np.where(np.isfinite(P["vis"]),  P["vis"]  >= thr_vis,  False)
    det_nisp = np.where(np.isfinite(P["nisp"]), P["nisp"] >= thr_nisp, False)
    n_det = (det_hst.astype(int) + det_jwst.astype(int) + det_vis.astype(int) + det_nisp.astype(int))
    is_sn = (n_det == 1)
    log(f"\n=== detection summary ===")
    log(f"  HST detected:        {int(det_hst.sum()):>10,}")
    log(f"  JWST detected:       {int(det_jwst.sum()):>10,}")
    log(f"  Euclid-VIS detected: {int(det_vis.sum()):>10,}")
    log(f"  Euclid-NISP det:     {int(det_nisp.sum()):>10,}")
    log(f"  SN candidates (1-of-4): {int(is_sn.sum()):>10,}")

    # composite confidence: (detected-survey P) * (1 - max of other 3 surveys' P)
    P_arr = np.stack([np.nan_to_num(P[s], nan=0.0) for s in ("hst","jwst","vis","nisp")], axis=1)
    det_arr = np.stack([det_hst,det_jwst,det_vis,det_nisp], axis=1)
    det_survey_p = np.where(det_arr, P_arr, 0.0).max(axis=1)
    nondet_max_p = np.where(~det_arr, P_arr, 0.0).max(axis=1)
    comp_conf = det_survey_p * (1.0 - nondet_max_p)

    # which survey detected? (only meaningful for is_sn=True)
    survey_names = np.array(["HST","JWST","EUCLID-VIS","EUCLID-NISP"])
    which = np.full(N, "", dtype=object)
    for k, name in enumerate(survey_names):
        col = det_arr[:, k] & is_sn
        which[col] = name

    # ---- write full scored parquet (all sources) ----
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

    # ---- write CSV in v32 schema (SN candidates only) ----
    log(f"Writing CSV → {OUT_CSV}")
    sn_idx = np.where(is_sn)[0]
    # sort by composite confidence desc
    sn_idx = sn_idx[np.argsort(-comp_conf[sn_idx])]
    # best band: among bands with snr>3 in the detected survey, highest snr_aper
    band_to_survey = {"F814W":"HST", "F115W":"JWST","F150W":"JWST","F277W":"JWST","F444W":"JWST",
                      "VIS":"EUCLID-VIS","Y":"EUCLID-NISP","J":"EUCLID-NISP","H":"EUCLID-NISP"}
    rows = []
    for cand_no, i in enumerate(sn_idx, start=1):
        det_name = which[i]
        # candidate bands for "best band"
        bands_for_det = [b for b, s in band_to_survey.items() if s == det_name]
        best_band = ""; best_snr = 0.0
        for b in bands_for_det:
            s = SNR[b][i]
            if np.isfinite(s) and s > best_snr:
                best_snr = float(s); best_band = b
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

    # ---- minimal v01 webpage ----
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Writing webpage → {HTML_DIR}/index.html")
    page = ["<!doctype html><html><head><title>SN search v01</title>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee}",
            "td.lab{text-align:left;font-weight:bold}",
            "h1{color:#333}",
            ".det{background:#e0f5e0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v01 — 1-of-4 detection across HST / JWST / Euclid-VIS / Euclid-NISP</h1>",
            f"<p>Built: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
            f"<p>Total candidates evaluated: {N:,}<br>",
            f"Detected in exactly one survey (= SN candidate): <b>{int(is_sn.sum()):,}</b><br>",
            f"Top {min(TOP_N_WEB, len(rows))} shown below (sorted by composite confidence).</p>",
            f"<p>Per-survey detection counts (overall): HST={int(det_hst.sum()):,}, "
            f"JWST={int(det_jwst.sum()):,}, VIS={int(det_vis.sum()):,}, NISP={int(det_nisp.sum()):,}.</p>",
            f"<p>Thresholds used: HST P&ge;{thr_hst:.4f}, JWST P&ge;{thr_jwst:.4f}, "
            f"VIS P&ge;{thr_vis:.4f}, NISP P&ge;{thr_nisp:.4f} (LOO-tuned at FPR=10<sup>&minus;3</sup>).</p>",
            "<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
            "<th>comp_conf</th><th>best band</th><th>best snr</th>"
            "<th>P_hst</th><th>P_jwst</th><th>P_vis</th><th>P_nisp</th></tr></thead><tbody>"]
    for rank, i in enumerate(sn_idx[:TOP_N_WEB], start=1):
        page.append(f"<tr><td>{rank}</td><td>{pid[i]}</td><td class='lab'>{which[i]}</td>"
                    f"<td>{ra[i]:.6f}</td><td>{dec[i]:.6f}</td>"
                    f"<td>{comp_conf[i]:.3f}</td>"
                    f"<td>{rows[rank-1]['best_band']}</td>"
                    f"<td>{rows[rank-1]['best_snr']}</td>"
                    f"<td class='{'det' if det_hst[i] else ''}'>"
                    f"{P['hst'][i]:.3f}</td>"
                    f"<td class='{'det' if det_jwst[i] else ''}'>"
                    f"{P['jwst'][i]:.3f}</td>"
                    f"<td class='{'det' if det_vis[i] else ''}'>"
                    f"{P['vis'][i]:.3f}</td>"
                    f"<td class='{'det' if det_nisp[i] else ''}'>"
                    f"{P['nisp'][i]:.3f}</td></tr>")
    page.append("</tbody></table>")
    page.append("</div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))
    log(f"=== Step 6 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
