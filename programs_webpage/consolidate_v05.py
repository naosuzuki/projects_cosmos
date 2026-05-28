"""v05 consolidator: read v05 partial parquets, apply 1-of-3 telescope rule
+ morphology G1-G4 + cross-band consistency + dN/dm prior, write
csvfiles_sn/tbl_sn_candidates_v05.csv and htmls/sn_search/v05/index.html.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv, time, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR = CSV_DIR / "_partial"
HTML_DIR = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v05")
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"
MODELS   = CSV_DIR / "cnn_models_v05.pt"
OUT_CSV  = CSV_DIR / "tbl_sn_candidates_v05.csv"

FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}
CSV_BANDS = ["F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W"]
SURVEY_BANDS = {"hst": ["F814W"], "jwst": ["F115W","F150W","F277W","F444W"],
                "vis": ["VIS"], "nisp": ["Y","J","H"]}
SAT_BANDS = ["F814W","F115W","F150W","F277W","F444W","VIS"]
MAG_FLOOR = 21.0
SHARP_LO, SHARP_HI = 0.40, 0.85
RND_LIM = 0.50
NF_LIM  = 0.25
TOP_N_WEB = 500


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dndm_prior(mag):
    if mag is None or not np.isfinite(mag) or mag <= 0: return 0.0
    if mag < MAG_FLOOR: return 0.0
    if mag > 27: return 1.0
    return 10**(0.4 * (mag - 27))


def main():
    t0 = time.time()
    log("=== v05 consolidate ===")
    bundle = torch.load(MODELS, map_location="cpu", weights_only=False)
    thr_hst  = bundle["thresholds"].get("hst",  0.5)
    thr_jwst = bundle["thresholds"].get("jwst", 0.5)
    thr_vis  = bundle["thresholds"].get("vis",  0.5)
    thr_nisp = bundle["thresholds"].get("nisp", 0.5)
    log(f"thresholds: hst={thr_hst:.4f} jwst={thr_jwst:.4f} vis={thr_vis:.4f} nisp={thr_nisp:.4f}")

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

    P = {s: np.full(N, np.nan, dtype=np.float32) for s in ("hst","jwst","vis","nisp")}
    MAG  = {b: np.full(N, -1.0, dtype=np.float32) for b in CSV_BANDS}
    SNR  = {b: np.full(N,  0.0, dtype=np.float32) for b in CSV_BANDS}
    SHARP = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    RND1  = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    RND2  = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    SEP   = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    NFR   = {b: np.full(N,  1.0, dtype=np.float32) for b in CSV_BANDS}
    surveys_done = {}; surveys_processed = {}
    for sv in ("hst","jwst","vis","nisp"):
        cp = PART_DIR / f"infer_v05_{sv}.parquet"
        if not cp.exists():
            surveys_done[sv] = False; surveys_processed[sv] = 0; continue
        t = pq.read_table(cp)
        idxs = t["idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        Ps   = t["P"].to_numpy(zero_copy_only=False).astype(np.float32)
        ii = idxs[idxs >= 0]
        P[sv][ii] = Ps[idxs >= 0]
        cols = set(t.column_names)
        for b in SURVEY_BANDS[sv]:
            MAG[b][ii] = t[f"mag_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            SNR[b][ii] = t[f"snr_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            if f"sharp_{b}" in cols:
                SHARP[b][ii] = t[f"sharp_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                RND1[b][ii]  = t[f"rnd1_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                RND2[b][ii]  = t[f"rnd2_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                SEP[b][ii]   = t[f"sep_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                NFR[b][ii]   = t[f"nanfrac_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
        surveys_done[sv] = True
        surveys_processed[sv] = len(ii)
    log(f"processed: {surveys_processed}")

    det_hst = np.where(in_h & np.isfinite(P["hst"]),  P["hst"]  >= thr_hst,  False)
    det_jwst = np.where(in_j & np.isfinite(P["jwst"]), P["jwst"] >= thr_jwst, False)
    det_vis  = np.where(in_e & np.isfinite(P["vis"]),  P["vis"]  >= thr_vis,  False)
    det_nisp = np.where(in_e & np.isfinite(P["nisp"]), P["nisp"] >= thr_nisp, False)
    det_euclid = det_vis | det_nisp
    n_det = det_hst.astype(int) + det_jwst.astype(int) + det_euclid.astype(int)
    log(f"per-telescope det: HST={int(det_hst.sum()):,} JWST={int(det_jwst.sum()):,} "
        f"VIS={int(det_vis.sum()):,} NISP={int(det_nisp.sum()):,} Euclid={int(det_euclid.sum()):,}")

    evaluated = (
        (~in_h | np.isfinite(P["hst"])) &
        (~in_j | np.isfinite(P["jwst"])) &
        (~in_e | np.isfinite(P["nisp"])) &
        (~in_e | np.isfinite(P["vis"]))
    )
    full_cov = in_h & in_j & in_e
    is_sn = (n_det == 1) & evaluated & full_cov

    which = np.full(N, "", dtype=object)
    which[det_hst  & is_sn] = "HST"
    which[det_jwst & is_sn] = "JWST"
    eu_both = det_vis & det_nisp & is_sn
    eu_v = det_vis & ~det_nisp & is_sn
    eu_n = det_nisp & ~det_vis & is_sn
    which[eu_both] = "EUCLID-VIS+NISP"
    which[eu_v]    = "EUCLID-VIS"
    which[eu_n]    = "EUCLID-NISP"

    P_arr = np.stack([np.nan_to_num(P[s], nan=0.0) for s in ("hst","jwst","vis","nisp")], axis=1)
    det_arr = np.stack([det_hst, det_jwst, det_vis, det_nisp], axis=1)
    det_survey_p = np.where(det_arr, P_arr, 0.0).max(axis=1)
    nondet_max_p = np.where(~det_arr, P_arr, 0.0).max(axis=1)
    comp_conf = det_survey_p * (1.0 - nondet_max_p)

    bands_per_label = {
        "HST":              ["F814W"],
        "JWST":             ["F115W","F150W","F277W","F444W"],
        "EUCLID-VIS":       ["VIS"], "EUCLID-NISP":      ["Y","J","H"],
        "EUCLID-VIS+NISP":  ["VIS","Y","J","H"],
    }
    G1_TOL = {b: max(0.10, 1.5 * FWHM_AS[b]) for b in CSV_BANDS}

    def band_morph_ok(b, i):
        sp = float(SHARP[b][i]); r1 = float(RND1[b][i]); r2 = float(RND2[b][i])
        sg = float(SEP[b][i]);   nf = float(NFR[b][i])
        if nf > NF_LIM: return False
        if not np.isfinite(sg) or sg > G1_TOL[b]: return False
        if not np.isfinite(sp) or sp < SHARP_LO or sp > SHARP_HI: return False
        if not np.isfinite(r1) or abs(r1) > RND_LIM: return False
        if not np.isfinite(r2) or abs(r2) > RND_LIM: return False
        return True

    sn_idx = np.where(is_sn)[0]
    sn_idx = sn_idx[np.argsort(-comp_conf[sn_idx])]
    rows = []
    n_sat = 0; n_morph = 0; n_cross = 0; n_snr = 0; n_mag = 0
    for i in sn_idx:
        det_name = which[i]
        bfd = bands_per_label.get(det_name, [])
        # saturation
        sat = False
        for b in SAT_BANDS:
            mv = float(MAG[b][i]); sv = float(SNR[b][i])
            if np.isfinite(mv) and 0 < mv < MAG_FLOOR and np.isfinite(sv) and sv >= 3.0:
                sat = True; break
        if sat: n_sat += 1; continue
        # morphology + best band
        best_band = ""; best_snr = 0.0; best_mag = -1.0
        morph_snrs = []
        for b in bfd:
            sv = float(SNR[b][i])
            if not np.isfinite(sv) or sv <= 0: continue
            if not band_morph_ok(b, i): continue
            morph_snrs.append(sv)
            if sv > best_snr:
                best_snr = sv; best_band = b; best_mag = float(MAG[b][i])
        if not morph_snrs: n_morph += 1; continue
        if det_name in ("JWST","EUCLID-NISP","EUCLID-VIS+NISP"):
            ms = sorted(morph_snrs, reverse=True)
            if len(ms) >= 2 and ms[0] > 10*ms[1]: n_cross += 1; continue
            elif len(ms) == 1 and ms[0] < 8: n_cross += 1; continue
        if best_snr < 5.0: n_snr += 1; continue
        if best_mag <= 0 or best_mag < MAG_FLOOR: n_mag += 1; continue
        comp_conf[i] = float(comp_conf[i]) * dndm_prior(best_mag)
        r = {"id": "cand_xxxxx", "primary_id": str(pid[i]),
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
             "n_3sig_detect": "1", "quality_flag": "v05_cnn_candidate"}
        for b in CSV_BANDS:
            v = float(MAG[b][i]); r[f"mag_{b}"] = f"{v:.2f}" if v != -1.0 else "-1"
        for b in CSV_BANDS:
            r[f"snr_{b}"] = f"{float(SNR[b][i]):.2f}"
        for s in ("hst","jwst","vis","nisp"):
            v = float(P[s][i]) if np.isfinite(P[s][i]) else float("nan")
            r[f"cnn_conf_{s}"] = f"{v:.4f}" if v == v else "nan"
        r["composite_confidence"] = f"{float(comp_conf[i]):.4f}"
        rows.append(r)
    rows.sort(key=lambda r: -float(r["composite_confidence"]))
    for rk, r in enumerate(rows, start=1):
        r["id"] = f"cand_{rk:05d}"
    log(f"raw 1-of-3: {int(is_sn.sum()):,}  filters: sat={n_sat} morph={n_morph} "
        f"cross={n_cross} snr={n_snr} mag={n_mag}  FINAL: {len(rows)}")

    if rows:
        cols = list(rows[0].keys())
        tmp = OUT_CSV.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)
        tmp.replace(OUT_CSV)
        log(f"wrote {OUT_CSV}")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    page = ["<!doctype html><html><head><title>SN search v05</title>",
            "<meta http-equiv='refresh' content='30'>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee} h1{color:#333}",
            ".prog{background:#fff5b0;padding:8px 12px;border:1px solid #d4a017;margin:10px 0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v05 &mdash; synthetic-injection-trained CNN, recall-based thresholds, 1-of-3 telescope rule</h1>",
            f"<p>{time.strftime('%Y-%m-%d %H:%M:%S')}. Thresholds @ 80% LOO recall: "
            f"HST={thr_hst:.3f} JWST={thr_jwst:.3f} VIS={thr_vis:.3f} NISP={thr_nisp:.3f}.</p>",
            "<div class='prog'>",
            f"<b>Processed:</b> HST={surveys_processed.get('hst',0):,} "
            f"JWST={surveys_processed.get('jwst',0):,} VIS={surveys_processed.get('vis',0):,} "
            f"NISP={surveys_processed.get('nisp',0):,}<br>",
            f"<b>Per-telescope detections:</b> HST={int(det_hst.sum()):,} "
            f"JWST={int(det_jwst.sum()):,} VIS={int(det_vis.sum()):,} "
            f"NISP={int(det_nisp.sum()):,} Euclid={int(det_euclid.sum()):,}<br>",
            f"<b>1-of-3 raw:</b> {int(is_sn.sum()):,}<br>",
            f"&minus;sat {n_sat}, &minus;morph {n_morph}, &minus;cross {n_cross}, "
            f"&minus;snr {n_snr}, &minus;mag {n_mag}<br>",
            f"<b>FINAL candidates:</b> <span style='color:#d80'>{len(rows):,}</span>",
            "</div>",
            "<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
            "<th>comp_conf</th><th>best</th><th>snr</th><th>mag</th>"
            "<th>P_hst</th><th>P_jwst</th><th>P_vis</th><th>P_nisp</th></tr></thead><tbody>"]
    for rk, r in enumerate(rows[:TOP_N_WEB], start=1):
        best_mag = r.get(f"mag_{r['best_band']}", "-1") if r['best_band'] else "-1"
        page.append(f"<tr><td>{rk}</td><td>{r['primary_id']}</td>"
                    f"<td>{r['telescope']}</td><td>{r['sn_ra']}</td><td>{r['sn_dec']}</td>"
                    f"<td>{r['composite_confidence']}</td>"
                    f"<td>{r['best_band']}</td><td>{r['best_snr']}</td>"
                    f"<td>{best_mag}</td>"
                    f"<td>{r['cnn_conf_hst']}</td><td>{r['cnn_conf_jwst']}</td>"
                    f"<td>{r['cnn_conf_vis']}</td><td>{r['cnn_conf_nisp']}</td></tr>")
    page.append("</tbody></table></div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))
    log(f"=== done {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
