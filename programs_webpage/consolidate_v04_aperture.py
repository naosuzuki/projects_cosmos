"""Consolidate v04 aperture-diff residuals into a candidate list.

Rule:
  Real modern-epoch SN = strong positive residual in ≥1 science band
  AND morphology G1-G4 passes in that band AND HST-side mag in
  [MAG_FLOOR, MAG_FAINT].

  Multi-band agreement bonus: residual > 5σ in JWST AND Euclid (both
  modern epochs) is much more confident than single-band.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR = CSV_DIR / "_partial"
HTML_DIR = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v04")
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"
OUT_CSV  = CSV_DIR / "tbl_sn_candidates_v04.csv"
RES_PARQ = PART_DIR / "infer_v04_aper_diff.parquet"

SCI_BANDS = ["F115W","F150W","F277W","F444W","VIS","Y","J","H"]
SNR_FLOOR = 5.0
MAG_FLOOR = 21.0
MAG_FAINT = 28.0
TOP_N_WEB = 500

FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}
SHARP_LO, SHARP_HI = 0.40, 0.85; RND_LIM = 0.50; NF_LIM = 0.25


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("=== v04 aperture-diff consolidate ===")
    if not RES_PARQ.exists():
        raise RuntimeError(f"Run infer_v04_aperture_diff.py first ({RES_PARQ} missing)")

    lk = pq.read_table(LOOK)
    pid = np.array(lk["primary_id"].to_pylist(), dtype=object)
    ra  = lk["ra"].to_numpy(zero_copy_only=False); dec = lk["dec"].to_numpy(zero_copy_only=False)
    th  = np.array(lk["tile_hst"].to_pylist(), dtype=object)
    tj  = np.array(lk["tile_jwst"].to_pylist(), dtype=object)
    te  = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(lk["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(lk["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    N = len(pid)

    # Load residuals (v04)
    res = pq.read_table(RES_PARQ)
    res_idx = res["idx"].to_numpy(zero_copy_only=False).astype(np.int64)
    DSNR = {}; DMAG = {}; RES = {}
    for b in SCI_BANDS:
        col = f"diff_snr_{b}"
        if col in res.column_names:
            arr = np.full(N, 0.0, dtype=np.float32)
            arr[res_idx] = res[col].to_numpy(zero_copy_only=False).astype(np.float32)
            DSNR[b] = arr
            arr2 = np.full(N, -1.0, dtype=np.float32)
            arr2[res_idx] = res[f"diff_mag_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            DMAG[b] = arr2
            arr3 = np.full(N, 0.0, dtype=np.float32)
            arr3[res_idx] = res[f"residual_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            RES[b]  = arr3
    log(f"  loaded residuals for: {sorted(DSNR.keys())}")

    # Load per-band morphology + per-band mag (from v05 partial) for the morph gate
    SHARP = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    RND1  = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    RND2  = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    SEP   = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    NFR   = {b: np.full(N,  1.0, dtype=np.float32) for b in SCI_BANDS}
    SNR_SCI = {b: np.full(N, 0.0, dtype=np.float32) for b in SCI_BANDS}
    MAG_SCI = {b: np.full(N, -1.0, dtype=np.float32) for b in SCI_BANDS}
    band_sv = {"F115W":"jwst","F150W":"jwst","F277W":"jwst","F444W":"jwst",
               "VIS":"vis","Y":"nisp","J":"nisp","H":"nisp"}
    for b in SCI_BANDS:
        cp = PART_DIR / f"infer_v05_{band_sv[b]}.parquet"
        if not cp.exists(): continue
        t = pq.read_table(cp)
        idxs = t["idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        cols = set(t.column_names)
        if f"sharp_{b}" in cols:
            SHARP[b][idxs] = t[f"sharp_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            RND1[b][idxs]  = t[f"rnd1_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            RND2[b][idxs]  = t[f"rnd2_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            SEP[b][idxs]   = t[f"sep_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            NFR[b][idxs]   = t[f"nanfrac_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
        if f"snr_{b}" in cols:
            SNR_SCI[b][idxs] = t[f"snr_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            MAG_SCI[b][idxs] = t[f"mag_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)

    G1_TOL = {b: max(0.10, 1.5 * FWHM_AS[b]) for b in SCI_BANDS}

    def passes_morph(b):
        """Vectorised G1-G4 + NaN-frac (returns bool array over N)."""
        sp = SHARP[b]; r1 = RND1[b]; r2 = RND2[b]; sg = SEP[b]; nf = NFR[b]
        return (np.isfinite(sp) & (sp >= SHARP_LO) & (sp <= SHARP_HI) &
                np.isfinite(r1) & (np.abs(r1) <= RND_LIM) &
                np.isfinite(r2) & (np.abs(r2) <= RND_LIM) &
                np.isfinite(sg) & (sg <= G1_TOL[b]) &
                np.isfinite(nf) & (nf <= NF_LIM))

    # Detection: positive residual ≥ 5σ AND morphology G1-G4 AND
    # measured science mag in [MAG_FLOOR, MAG_FAINT]
    det = {}
    for b in SCI_BANDS:
        if b not in DSNR: det[b] = np.zeros(N, dtype=bool); continue
        det[b] = (
            (DSNR[b] >= SNR_FLOOR) & (RES[b] > 0) &
            passes_morph(b) &
            np.isfinite(MAG_SCI[b]) &
            (MAG_SCI[b] >= MAG_FLOOR) & (MAG_SCI[b] <= MAG_FAINT)
        )
        log(f"  [{b}] det count: {int(det[b].sum()):,}")

    # JWST = any of 4 bands fires; Euclid = any of 4 bands
    det_jwst   = det.get("F115W", False) | det.get("F150W", False) | det.get("F277W", False) | det.get("F444W", False)
    det_euclid = det.get("VIS", False)   | det.get("Y", False)     | det.get("J", False)     | det.get("H", False)
    # v04 candidate = ≥1 modern-epoch detection with full coverage
    full_cov = in_h & in_j & in_e
    is_sn = full_cov & (det_jwst | det_euclid)
    log(f"v04 raw candidates (full cov, ≥1 band ≥5σ residual): {int(is_sn.sum()):,}")

    # Composite confidence: number of bands firing × max diff_snr / (max+50)
    nb = sum(det[b].astype(np.int8) for b in SCI_BANDS if b in det)
    max_dsnr = np.maximum.reduce([DSNR[b] for b in SCI_BANDS if b in DSNR])
    comp_conf = nb.astype(np.float32) * (max_dsnr / (max_dsnr + 50.0))

    sn_idx = np.where(is_sn)[0]
    sn_idx = sn_idx[np.argsort(-comp_conf[sn_idx])]
    rows = []
    for i in sn_idx:
        best_band = ""; best_snr = 0.0; best_mag = -1.0
        for b in SCI_BANDS:
            if b not in det or not det[b][i]: continue
            sv = float(DSNR[b][i])
            if sv > best_snr:
                best_snr = sv; best_band = b
                best_mag = float(MAG_SCI[b][i])
        is_j = bool(det_jwst[i]); is_v = bool(det.get("VIS", np.zeros(N, dtype=bool))[i])
        is_n = bool((det.get("Y", False) | det.get("J", False) | det.get("H", False))[i])
        if is_j and (is_v or is_n): tel = "JWST+EUCLID"
        elif is_j: tel = "JWST"
        elif is_v and is_n: tel = "EUCLID-VIS+NISP"
        elif is_v: tel = "EUCLID-VIS"
        elif is_n: tel = "EUCLID-NISP"
        else: tel = "?"
        r = {"id": "cand_xxxxx", "primary_id": str(pid[i]), "telescope": tel,
             "hst_tile": str(th[i]) if th[i] else "",
             "jwst_tile": str(tj[i]) if tj[i] else "",
             "euclid_tile": str(te[i]) if te[i] else "",
             "host_ra": f"{float(ra[i]):.7f}", "host_dec": f"{float(dec[i]):.7f}",
             "sn_ra": f"{float(ra[i]):.7f}", "sn_dec": f"{float(dec[i]):.7f}",
             "sn_host_sep_arcsec": "0.000", "host_z": "-1",
             "best_band": best_band, "best_snr": f"{best_snr:.2f}",
             "n_bands_detected": str(int(nb[i])),
             "combined_sigma": f"{float(comp_conf[i]):.3f}",
             "max_snr_detect": f"{best_snr:.2f}",
             "quality_flag": "v04_aper_diff_candidate"}
        for b in SCI_BANDS:
            r[f"diff_snr_{b}"] = f"{float(DSNR[b][i]):.2f}" if b in DSNR else "nan"
            r[f"mag_{b}"]      = f"{float(MAG_SCI[b][i]):.2f}" if MAG_SCI[b][i] != -1.0 else "-1"
        r["composite_confidence"] = f"{float(comp_conf[i]):.4f}"
        rows.append(r)
    for rk, r in enumerate(rows, start=1):
        r["id"] = f"cand_{rk:05d}"

    if rows:
        cols = list(rows[0].keys())
        tmp = OUT_CSV.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows: w.writerow(r)
        tmp.replace(OUT_CSV)
        log(f"  wrote {OUT_CSV}  ({len(rows)} candidates)")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    page = ["<!doctype html><html><head><title>SN search v04 aperture-diff</title>",
            "<meta http-equiv='refresh' content='30'>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee} h1{color:#333}",
            ".prog{background:#fff5b0;padding:8px 12px;border:1px solid #d4a017;margin:10px 0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v04 &mdash; aperture-level difference imaging</h1>",
            f"<p>{time.strftime('%Y-%m-%d %H:%M:%S')}. Residual flux = "
            f"sci_flux &minus; K(HST_mag) &middot; HST_F814W_flux; K estimated per HST-mag bin "
            f"from population median ratio. SN candidate if residual SNR &ge; {SNR_FLOOR:.0f} "
            f"+ G1-G4 morphology + {MAG_FLOOR:.0f} &le; sci_mag &le; {MAG_FAINT:.0f}.</p>",
            "<div class='prog'>",
            f"<b>Per-band detections (positive residual ≥5σ + morph):</b><br>"]
    for b in SCI_BANDS:
        page.append(f"&nbsp;&nbsp;{b}: {int(det.get(b, np.zeros(N)).sum()):,}<br>")
    page.append(f"<b>FINAL candidates (full coverage, &ge;1 band):</b> "
                f"<span style='color:#d80'>{len(rows):,}</span></div>")
    page.append("<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
                "<th>comp_conf</th><th>n_bands</th><th>best</th><th>diff_snr</th><th>mag</th></tr></thead><tbody>")
    for rk, r in enumerate(rows[:TOP_N_WEB], start=1):
        best_mag = r.get(f"mag_{r['best_band']}", "-1") if r['best_band'] else "-1"
        page.append(f"<tr><td>{rk}</td><td>{r['primary_id']}</td>"
                    f"<td>{r['telescope']}</td><td>{r['sn_ra']}</td><td>{r['sn_dec']}</td>"
                    f"<td>{r['composite_confidence']}</td>"
                    f"<td>{r['n_bands_detected']}</td>"
                    f"<td>{r['best_band']}</td>"
                    f"<td>{r['best_snr']}</td>"
                    f"<td>{best_mag}</td></tr>")
    page.append("</tbody></table></div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))
    log(f"=== done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
