"""Consolidate v03 partials using MORPHOLOGY + SNR only (NO CNN).

Rationale: with only 3-13 known SNe per survey, the per-survey CNNs over-
fit augmentation patterns and produce both false positives (firing on
host galaxies / artifacts) and false negatives (missing 16/17 known SNe
at the in-sky validation). Use the per-band photometry + DAO morphology
directly — these are far more robust on small training data.

Rule for "telescope detected":
  ANY band of that telescope has a clean PSF point source at the catalog
  position AND aperture SNR ≥ SNR_FLOOR AND detection-band mag in
  [MAG_FLOOR, MAG_FAINT].
  Clean PSF = G1 (sep_sn ≤ 1.5·FWHM) ∧ G2 (0.40 ≤ sharp ≤ 0.85) ∧
              G3 (|rnd1| ≤ 0.5) ∧ G4 (|rnd2| ≤ 0.5) ∧
              NaN-fraction ≤ 0.25 in central 20×20.

Final rule:  exactly 1 of {HST, JWST, Euclid} detected (VIS+NISP collapse).
Saturation guard: ANY HST/JWST/VIS band with finite mag<MAG_FLOOR & snr≥3
  marks the source as likely-saturated (NISP excluded — host blending).
dN/dm prior:  ∝ 10^(0.4·m)  in [MAG_FLOOR, 27].

Outputs:
  csvfiles_sn/tbl_sn_candidates_v03.csv  (same schema as the CNN run)
  htmls/sn_search/v03/index.html         (stats page)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv, time, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR = CSV_DIR / "_partial"
HTML_DIR = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v03")
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"
OUT_CSV  = CSV_DIR / "tbl_sn_candidates_v03.csv"

MAG_FLOOR = 21.0
MAG_FAINT = 28.0
# Single-band telescopes (HST F814W, Euclid VIS): must clear a high SNR floor
# because we can't cross-check with another band of the same telescope.
SNR_FLOOR_HST = 15.0
SNR_FLOOR_VIS = 50.0      # very strict for single-band Euclid VIS (else compact galaxies dominate)
SNR_FLOOR_MULTI  = 5.0
MIN_BANDS_JWST   = 2
MIN_BANDS_NISP   = 2
# Tighter PSF morphology — real point sources are near sharp~0.65, not 0.4-0.85
SHARP_LO, SHARP_HI = 0.40, 0.85
RND_LIM = 0.50
NF_LIM  = 0.25
TOP_N_WEB = 500

FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}
SURVEY_BANDS = {"hst": ["F814W"],
                "jwst": ["F115W","F150W","F277W","F444W"],
                "vis":  ["VIS"],
                "nisp": ["Y","J","H"]}
CSV_BANDS = ["F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W"]
SAT_BANDS = ["F814W","F115W","F150W","F277W","F444W","VIS"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dndm_prior(mag):
    if mag is None or not np.isfinite(mag) or mag <= 0: return 0.0
    if mag < MAG_FLOOR: return 0.0
    if mag > 27: return 1.0
    return 10**(0.4 * (mag - 27))


def main():
    t0 = time.time()
    log("=== morphology+SNR consolidate (no CNN) ===")
    lk = pq.read_table(LOOK)
    pid  = np.array(lk["primary_id"].to_pylist(), dtype=object)
    psrc = np.array(lk["primary_source"].to_pylist(), dtype=object)
    ra   = lk["ra"].to_numpy(zero_copy_only=False)
    dec  = lk["dec"].to_numpy(zero_copy_only=False)
    th   = np.array(lk["tile_hst"].to_pylist(), dtype=object)
    tj   = np.array(lk["tile_jwst"].to_pylist(), dtype=object)
    te   = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(lk["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(lk["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    N = len(pid)
    log(f"  lookup {N:,} candidates")

    # Per-band arrays
    MAG   = {b: np.full(N, -1.0, dtype=np.float32) for b in CSV_BANDS}
    SNR   = {b: np.full(N,  0.0, dtype=np.float32) for b in CSV_BANDS}
    SHARP = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    RND1  = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    RND2  = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    SEP   = {b: np.full(N, np.nan, dtype=np.float32) for b in CSV_BANDS}
    NFR   = {b: np.full(N,  1.0, dtype=np.float32) for b in CSV_BANDS}
    processed = {s: 0 for s in SURVEY_BANDS}
    for sv in ("hst","jwst","vis","nisp"):
        path = PART_DIR / f"infer_v03_{sv}.parquet"
        if not path.exists(): continue
        t = pq.read_table(path)
        idxs = t["idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        ii = idxs[idxs >= 0]
        cols = set(t.column_names)
        for b in SURVEY_BANDS[sv]:
            MAG[b][ii]   = t[f"mag_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            SNR[b][ii]   = t[f"snr_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            if f"sharp_{b}" in cols:
                SHARP[b][ii] = t[f"sharp_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                RND1[b][ii]  = t[f"rnd1_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                RND2[b][ii]  = t[f"rnd2_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                SEP[b][ii]   = t[f"sep_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                NFR[b][ii]   = t[f"nanfrac_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
        processed[sv] = len(ii)
    log(f"  loaded partials  processed={processed}")

    G1_TOL = {b: max(0.10, 1.5 * FWHM_AS[b]) for b in CSV_BANDS}

    def band_pass(b, snr_floor):
        """Vectorised per-band G1-G4 + NaN-frac + snr + mag pass (full N)."""
        sp = SHARP[b]; r1 = RND1[b]; r2 = RND2[b]
        sg = SEP[b];   nf = NFR[b]
        sn = SNR[b];   mg = MAG[b]
        return (
            np.isfinite(sp) & (sp >= SHARP_LO) & (sp <= SHARP_HI) &
            np.isfinite(r1) & (np.abs(r1) <= RND_LIM) &
            np.isfinite(r2) & (np.abs(r2) <= RND_LIM) &
            np.isfinite(sg) & (sg <= G1_TOL[b]) &
            np.isfinite(nf) & (nf <= NF_LIM) &
            np.isfinite(sn) & (sn >= snr_floor) &
            np.isfinite(mg) & (mg >= MAG_FLOOR) & (mg <= MAG_FAINT)
        )

    # PRE-COMPUTE per-band pass arrays (no recomputation in the candidate loop)
    log("  precompute per-band pass arrays")
    PASS = {}
    PASS["F814W"] = band_pass("F814W", SNR_FLOOR_HST)
    for b in SURVEY_BANDS["jwst"]:
        PASS[b] = band_pass(b, SNR_FLOOR_MULTI)
    PASS["VIS"] = band_pass("VIS", SNR_FLOOR_VIS)
    for b in SURVEY_BANDS["nisp"]:
        PASS[b] = band_pass(b, SNR_FLOOR_MULTI)

    # Per-telescope detection
    log(f"  per-telescope detection: HST snr≥{SNR_FLOOR_HST:.0f}; JWST ≥{MIN_BANDS_JWST}/4 "
        f"bands snr≥{SNR_FLOOR_MULTI:.0f}; VIS snr≥{SNR_FLOOR_VIS:.0f}; "
        f"NISP ≥{MIN_BANDS_NISP}/3 bands snr≥{SNR_FLOOR_MULTI:.0f}")
    det_hst = PASS["F814W"] & in_h
    jwst_count = sum(PASS[b].astype(np.int8) for b in SURVEY_BANDS["jwst"])
    det_jwst = (jwst_count >= MIN_BANDS_JWST) & in_j
    det_vis = PASS["VIS"] & in_e
    nisp_count = sum(PASS[b].astype(np.int8) for b in SURVEY_BANDS["nisp"])
    det_nisp = (nisp_count >= MIN_BANDS_NISP) & in_e
    det_euclid = det_vis | det_nisp
    n_det = det_hst.astype(int) + det_jwst.astype(int) + det_euclid.astype(int)
    log(f"  raw telescope detections: HST={int(det_hst.sum()):,}  "
        f"JWST={int(det_jwst.sum()):,}  VIS={int(det_vis.sum()):,}  "
        f"NISP={int(det_nisp.sum()):,}  Euclid={int(det_euclid.sum()):,}")

    # Coverage + evaluated (we treat every source as evaluated since this is
    # the FINAL pass after inference completed)
    full_cov = in_h & in_j & in_e
    is_sn = (n_det == 1) & full_cov
    log(f"  raw 1-of-3 detections (full coverage): {int(is_sn.sum()):,}")

    # Telescope label
    which = np.full(N, "", dtype=object)
    which[det_hst  & is_sn] = "HST"
    which[det_jwst & is_sn] = "JWST"
    which[det_vis  & ~det_nisp & is_sn] = "EUCLID-VIS"
    which[det_nisp & ~det_vis  & is_sn] = "EUCLID-NISP"
    which[det_vis  &  det_nisp & is_sn] = "EUCLID-VIS+NISP"

    bands_per_label = {
        "HST":              ["F814W"],
        "JWST":             ["F115W","F150W","F277W","F444W"],
        "EUCLID-VIS":       ["VIS"],
        "EUCLID-NISP":      ["Y","J","H"],
        "EUCLID-VIS+NISP":  ["VIS","Y","J","H"],
    }

    # ---- per-candidate post-rule filters + best band picking ----
    sn_idx = np.where(is_sn)[0]
    log(f"  applying saturation + composite confidence + dN/dm prior to {len(sn_idx):,} sources")
    rows = []
    n_filt_sat = 0; n_filt_cross = 0
    comp_conf = np.zeros(N, dtype=np.float32)
    for i in sn_idx:
        det_name = which[i]
        bands_for_det = bands_per_label.get(det_name, [])
        # Saturation guard (HST/JWST/VIS bands only)
        saturated = False
        for b in SAT_BANDS:
            mv = float(MAG[b][i]); sv = float(SNR[b][i])
            if np.isfinite(mv) and 0 < mv < MAG_FLOOR and np.isfinite(sv) and sv >= 3.0:
                saturated = True; break
        if saturated:
            n_filt_sat += 1; continue
        # Pick best band by SNR among morph-passing bands of the detection telescope
        best_band = ""; best_snr = 0.0; best_mag = -1.0
        morph_snrs = []
        for b in bands_for_det:
            if not PASS[b][i]: continue
            sv = float(SNR[b][i])
            morph_snrs.append(sv)
            if sv > best_snr:
                best_snr = sv; best_band = b; best_mag = float(MAG[b][i])
        # Cross-band consistency for multi-band telescopes
        if det_name in ("JWST", "EUCLID-NISP", "EUCLID-VIS+NISP"):
            ms = sorted(morph_snrs, reverse=True)
            if len(ms) >= 2:
                if ms[0] > 10.0 * ms[1]:
                    n_filt_cross += 1; continue
            elif len(ms) == 1 and ms[0] < 8.0:
                n_filt_cross += 1; continue
        # Composite confidence: best_snr * dN/dm prior, normalized by 1+best_snr
        prior = dndm_prior(best_mag)
        # Use snr-saturating composite: snr/(snr+10) so values in [0,1)
        snr_score = best_snr / (best_snr + 10.0)
        comp_conf[i] = float(prior * snr_score)
        r = {
            "id": f"cand_xxxxx",     # rank assigned after sort
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
            "n_3sig_detect": "1", "quality_flag": "morph_candidate",
            "_idx": int(i),
        }
        for b in CSV_BANDS:
            v = float(MAG[b][i]); r[f"mag_{b}"] = f"{v:.2f}" if v != -1.0 else "-1"
        for b in CSV_BANDS:
            r[f"snr_{b}"] = f"{float(SNR[b][i]):.2f}"
        r["cnn_conf_hst"] = "nan"; r["cnn_conf_jwst"] = "nan"
        r["cnn_conf_vis"] = "nan"; r["cnn_conf_nisp"] = "nan"
        r["composite_confidence"] = f"{float(comp_conf[i]):.4f}"
        rows.append(r)

    # Sort by composite confidence, assign ranks
    rows.sort(key=lambda r: -float(r["composite_confidence"]))
    for rank, r in enumerate(rows, start=1):
        r["id"] = f"cand_{rank:05d}"
    log(f"  saturation rejects: {n_filt_sat:,}   cross-band rejects: {n_filt_cross:,}")
    log(f"  FINAL: {len(rows)} candidates")

    # Write CSV
    if rows:
        cols = [c for c in rows[0] if c != "_idx"]
        tmp = OUT_CSV.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
        tmp.replace(OUT_CSV)
        log(f"  wrote {OUT_CSV}")

    # Write HTML stats page
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    page = ["<!doctype html><html><head><title>SN search v03 (morph-only)</title>",
            "<meta http-equiv='refresh' content='30'>",
            "<style>body{font-family:serif;background:#fafafa}",
            ".wrap{max-width:1300px;margin:0 auto;padding:20px}",
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}",
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
            "th{background:#eee} td.lab{text-align:left;font-weight:bold}",
            "h1{color:#333} .prog{background:#fff5b0;padding:8px 12px;border:1px solid #d4a017;margin:10px 0}",
            "</style></head><body><div class='wrap'>",
            "<h1>SN search v03 &mdash; morphology + SNR (NO CNN). 1-of-3 telescope rule.</h1>",
            f"<p>Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. Detection rule: "
            f"HST F814W snr&ge;{SNR_FLOOR_HST:.0f}; JWST &ge;{MIN_BANDS_JWST}/4 bands snr&ge;{SNR_FLOOR_MULTI:.0f}; "
            f"VIS snr&ge;{SNR_FLOOR_VIS:.0f}; NISP &ge;{MIN_BANDS_NISP}/3 bands snr&ge;{SNR_FLOOR_MULTI:.0f}. "
            f"Plus G1-G4 morphology, {MAG_FLOOR:.0f}&le;mag&le;{MAG_FAINT:.0f}, nan-frac&le;{NF_LIM}.</p>",
            "<div class='prog'>",
            f"<b>Inputs:</b> {N:,} candidates &middot; "
            f"{int(full_cov.sum()):,} with full HST+JWST+Euclid coverage<br>",
            f"<b>Per-telescope detections:</b> HST={int(det_hst.sum()):,} "
            f"JWST={int(det_jwst.sum()):,} VIS={int(det_vis.sum()):,} NISP={int(det_nisp.sum()):,} "
            f"Euclid(VIS|NISP)={int(det_euclid.sum()):,}<br>",
            f"<b>Raw 1-of-3 detections (full coverage):</b> {int(is_sn.sum()):,}<br>",
            f"&nbsp;&minus; saturation rejects: {n_filt_sat:,}<br>",
            f"&nbsp;&minus; cross-band consistency rejects: {n_filt_cross:,}<br>",
            f"<b>Final candidates after morph+SNR rules:</b> <span style='color:#d80'>{len(rows):,}</span>",
            "</div>",
            f"<p>Top {min(TOP_N_WEB, len(rows))} candidates (sorted by composite confidence "
            f"= dN/dm(best_mag) &times; snr/(snr+10)):</p>",
            "<table><thead><tr><th>rank</th><th>id</th><th>tel</th><th>ra</th><th>dec</th>"
            "<th>comp_conf</th><th>best band</th><th>best snr</th><th>best mag</th></tr></thead><tbody>"]
    for rank, r in enumerate(rows[:TOP_N_WEB], start=1):
        bm = MAG[r["best_band"]][r["_idx"]] if r["best_band"] else -1.0
        page.append(f"<tr><td>{rank}</td><td>{r['primary_id']}</td>"
                    f"<td class='lab'>{r['telescope']}</td>"
                    f"<td>{r['sn_ra']}</td><td>{r['sn_dec']}</td>"
                    f"<td>{r['composite_confidence']}</td>"
                    f"<td>{r['best_band']}</td><td>{r['best_snr']}</td>"
                    f"<td>{bm:.2f}</td></tr>")
    page.append("</tbody></table></div></body></html>")
    (HTML_DIR / "index.html").write_text("\n".join(page))
    log(f"=== done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
