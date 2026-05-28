"""v04 consolidator: read per-band diff-imaging parquets, apply selection,
write csvfiles_sn/tbl_sn_candidates_v04.csv + webpage.

Rule:
  Real modern-epoch transient = positive residual ≥ 5σ in at least 1
  science band. Higher confidence if multiple bands fire.

Note: this is a coarse first pass — depth-matched, PSF-matched difference
imaging is hard. The scale fit may absorb some SN flux for sources whose
galaxy colour is very atypical; the per-band SNR floor + multi-band
agreement compensate.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR = CSV_DIR / "_partial"
HTML_DIR = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v04")
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"
OUT_CSV  = CSV_DIR / "tbl_sn_candidates_v04.csv"

SCI_BANDS = ["F115W","F150W","F277W","F444W","VIS","Y","J","H"]
SNR_FLOOR = 5.0
MAG_FLOOR = 21.0
MAG_FAINT = 28.0
TOP_N_WEB = 500


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log("=== v04 consolidate ===")
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

    # Read per-band diff parquets
    SNR = {b: np.full(N, 0.0, dtype=np.float32) for b in SCI_BANDS}
    MAG = {b: np.full(N, -1.0, dtype=np.float32) for b in SCI_BANDS}
    FLUX = {b: np.full(N, 0.0, dtype=np.float32) for b in SCI_BANDS}
    avail = []
    for b in SCI_BANDS:
        p = PART_DIR / f"infer_v04_diff_{b}.parquet"
        if not p.exists(): continue
        t = pq.read_table(p)
        idxs = t["idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        SNR[b][idxs] = t["diff_snr"].to_numpy(zero_copy_only=False).astype(np.float32)
        MAG[b][idxs] = t["diff_mag"].to_numpy(zero_copy_only=False).astype(np.float32)
        FLUX[b][idxs] = t["diff_flux"].to_numpy(zero_copy_only=False).astype(np.float32)
        avail.append((b, len(idxs)))
    log(f"available bands: {avail}")

    # Detection in each science band: positive residual snr >= 5 AND mag in range
    det = {b: ((SNR[b] >= SNR_FLOOR) & np.isfinite(MAG[b]) &
               (MAG[b] >= MAG_FLOOR) & (MAG[b] <= MAG_FAINT))
           for b in SCI_BANDS}
    jwst_count = sum(det[b].astype(np.int8) for b in ["F115W","F150W","F277W","F444W"])
    nisp_count = sum(det[b].astype(np.int8) for b in ["Y","J","H"])
    # Real SNe leave a positive residual in MULTIPLE bands; single-band
    # positives are usually subtraction artefacts (registration/PSF residual).
    #   JWST detected  = >=2 of 4 bands
    #   Euclid detected = VIS positive, OR >=2 of 3 NISP bands
    det_jwst = jwst_count >= 2
    det_euclid = det["VIS"] | (nisp_count >= 2)
    log(f"JWST diff dets (>=2 bands): {int(det_jwst.sum()):,}   Euclid: {int(det_euclid.sum()):,}")

    full_cov = in_h & in_j & in_e
    is_sn = full_cov & (det_jwst | det_euclid)

    sn_idx = np.where(is_sn)[0]
    log(f"raw v04 candidates (full coverage, ≥1 band positive residual ≥5σ): {len(sn_idx):,}")

    # Composite confidence: weighted by number of bands firing and max SNR
    n_bands = sum(det[b].astype(np.int8) for b in SCI_BANDS)
    max_snr = np.maximum.reduce([SNR[b] for b in SCI_BANDS])
    comp_conf = n_bands.astype(np.float32) * (max_snr / (max_snr + 20.0))
    sn_idx = sn_idx[np.argsort(-comp_conf[sn_idx])]

    rows = []
    for i in sn_idx:
        # Pick best band by SNR
        best_band = ""; best_snr = 0.0; best_mag = -1.0
        for b in SCI_BANDS:
            sv = float(SNR[b][i])
            if sv > best_snr and det[b][i]:
                best_snr = sv; best_band = b; best_mag = float(MAG[b][i])
        # Determine telescope label
        is_j = bool((det["F115W"] | det["F150W"] | det["F277W"] | det["F444W"])[i])
        is_v = bool(det["VIS"][i])
        is_n = bool((det["Y"] | det["J"] | det["H"])[i])
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
             "n_bands_detected": str(int(n_bands[i])),
             "combined_sigma": f"{float(comp_conf[i]):.3f}",
             "max_snr_detect": f"{best_snr:.2f}",
             "quality_flag": "v04_diff_candidate"}
        for b in SCI_BANDS:
            v = float(MAG[b][i]); r[f"diff_mag_{b}"] = f"{v:.2f}" if v != -1.0 else "-1"
        for b in SCI_BANDS:
            r[f"diff_snr_{b}"] = f"{float(SNR[b][i]):.2f}"
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
        log(f"wrote {OUT_CSV}  ({len(rows)} candidates)")

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    page = ["<!doctype html><html><head><title>SN search v04 (diff imaging)</title>",
            "<meta http-equiv='refresh' content='30'>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee} h1{color:#333}",
            ".prog{background:#fff5b0;padding:8px 12px;border:1px solid #d4a017;margin:10px 0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v04 &mdash; difference imaging (HST F814W = static template)</h1>",
            f"<p>{time.strftime('%Y-%m-%d %H:%M:%S')}. Method: per-source, "
            f"reproject HST onto each science band, PSF-match, scale to local annulus, subtract; "
            f"detect positive residual with aperture SNR &ge; {SNR_FLOOR:.0f} and "
            f"{MAG_FLOOR:.0f} &le; mag &le; {MAG_FAINT:.0f}.</p>",
            "<div class='prog'>",
            f"<b>Per-band positive residuals:</b><br>",
    ]
    for b in SCI_BANDS:
        page.append(f"&nbsp;&nbsp;{b}: {int(det[b].sum()):,}<br>")
    page.append(f"<b>FINAL candidates (≥1 band ≥{SNR_FLOOR:.0f}σ + full coverage):</b> "
                f"<span style='color:#d80'>{len(rows):,}</span></div>")
    page.append("<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
                "<th>comp_conf</th><th>n_bands</th><th>best</th><th>snr</th><th>mag</th></tr></thead><tbody>")
    for rk, r in enumerate(rows[:TOP_N_WEB], start=1):
        page.append(f"<tr><td>{rk}</td><td>{r['primary_id']}</td>"
                    f"<td>{r['telescope']}</td><td>{r['sn_ra']}</td><td>{r['sn_dec']}</td>"
                    f"<td>{r['composite_confidence']}</td>"
                    f"<td>{r['n_bands_detected']}</td>"
                    f"<td>{r['best_band']}</td>"
                    f"<td>{r['best_snr']}</td>"
                    f"<td>{r.get('diff_mag_'+r['best_band'], '-1') if r['best_band'] else '-1'}</td></tr>")
    page.append("</tbody></table></div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))
    log(f"=== done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
