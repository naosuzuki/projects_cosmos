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
import sys, os, time, csv, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
MASTER = Path("/Users/suzuki/github/projects_cosmos/csvfiles_star/master_or_catalog_v03.parquet")

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

# v04 patches 2026-05-28 (mirror of v05 patches after user's top-25 review).
# v04's partials only have diff_snr/diff_mag; we load RAW aperture phot +
# morphology from the v05 partials (same source positions, same FITS, just
# the raw aperture sums instead of the bg-subtracted diff).
MAG_FAINT_HST = 27.8         # HST F814W 3σ point-source depth (COSMOS-Web)
HST_VIS_SNR_VETO = 3.0       # reject if HST or Euclid-VIS aperture snr ≥ this
SHARP_LO, SHARP_HI = 0.40, 0.75
RND_LIM = 0.50
NF_LIM  = 0.25
FWHM_AS = {"F814W":0.134, "F115W":0.057, "F150W":0.057, "F277W":0.130, "F444W":0.160,
           "VIS":0.194, "Y":0.524, "J":0.537, "H":0.567}
BAND_SURVEY = {"F814W":"hst", "F115W":"jwst","F150W":"jwst","F277W":"jwst","F444W":"jwst",
               "VIS":"vis","Y":"nisp","J":"nisp","H":"nisp"}

# Mirrors v05 patches 9, 10, 11 (added 2026-05-28 after v05 review found
# high-z dropouts and high-PM stars slipping through the basic vetoes).
NISP_SNR_VETO = 3.0              # wide-aperture NISP veto (raw from v05 NISP partial)
NISP_TIGHT_APER_MULT = 0.3       # tighter aperture for blending-corrected re-measure
NISP_TIGHT_SNR_VETO = 3.0
HST_PM_SEARCH_RADIUS_AS = 5.0    # HST DAO neighbor search radius
HST_PM_NEIGHBOR_SNR_MIN = 5.0


def load_hst_dao_kdtree():
    """Load HST DAO positions from master catalog, build KDTree."""
    t = pq.read_table(MASTER, columns=["primary_id","dao_F814W_ra_hst",
                                       "dao_F814W_dec_hst","dao_F814W_snr_hst"])
    ra = t["dao_F814W_ra_hst"].to_numpy(zero_copy_only=False)
    dec = t["dao_F814W_dec_hst"].to_numpy(zero_copy_only=False)
    snr = t["dao_F814W_snr_hst"].to_numpy(zero_copy_only=False)
    pids = np.array(t["primary_id"].to_pylist(), dtype=object)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(snr) & (snr >= HST_PM_NEIGHBOR_SNR_MIN)
    ra, dec, snr, pids = ra[ok], dec[ok], snr[ok], pids[ok]
    med_dec = float(np.median(dec))
    cosd = math.cos(math.radians(med_dec))
    coords = np.column_stack([ra * cosd, dec])
    return cKDTree(coords), ra, dec, pids, snr, cosd


def nisp_tight_aper(ra, dec, sci_data, wcs, hdr, fwhm_as, aper_mult):
    try:
        sx, sy = wcs.all_world2pix(ra, dec, 0)
    except Exception:
        return -1.0, 0.0
    sx, sy = float(sx), float(sy)
    if not (np.isfinite(sx) and np.isfinite(sy)): return -1.0, 0.0
    ps = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    cy, cx = int(round(sy)), int(round(sx))
    n = 30; half = n//2
    y0, y1 = cy-half, cy+half; x0, x1 = cx-half, cx+half
    ny, nx = sci_data.shape[-2:]
    if y0<0 or x0<0 or y1>ny or x1>nx: return -1.0, 0.0
    sub = sci_data[y0:y1, x0:x1].astype(np.float64)
    cy2, cx2 = sy-y0, sx-x0
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx-cx2)**2 + (yy-cy2)**2)
    aper_r = aper_mult * fwhm_as / ps
    ring_in = 2.0*fwhm_as/ps; ring_out = 3.5*fwhm_as/ps
    in_aper = rr <= aper_r; in_ring = (rr >= ring_in) & (rr <= ring_out)
    if int(in_aper.sum())<1 or int(in_ring.sum())<5: return -1.0, 0.0
    bg = float(np.nanmedian(sub[in_ring])); sub2 = sub - bg
    n_ap = int(in_aper.sum())
    flux = float(np.nansum(sub2[in_aper]))
    sig = float(np.nanstd(sub[in_ring])) * math.sqrt(n_ap)
    snr = flux / (sig + 1e-30)
    zp = (hdr.get("ZP") or hdr.get("MAGZP") or hdr.get("PHOTZP") or 23.9)
    mag = (float(zp) - 2.5*math.log10(flux)) if flux > 0 else -1.0
    return mag, float(snr)


def measure_nisp_tight(survivors, te_arr, ra_arr, dec_arr):
    by_tile = defaultdict(list)
    for i in survivors:
        t = str(te_arr[i]) if te_arr[i] else None
        if t: by_tile[t].append((i, float(ra_arr[i]), float(dec_arr[i])))
    nisp_fwhm = {"Y": FWHM_AS["Y"], "J": FWHM_AS["J"], "H": FWHM_AS["H"]}
    nisp_R = {"Y":"NIR-Y","J":"NIR-J","H":"NIR-H"}
    out = {i: {b: (-1.0, 0.0) for b in ("Y","J","H")} for i in survivors}
    for tile, srcs in by_tile.items():
        for b in ("Y","J","H"):
            path = R.resolve_euclid_path(tile, nisp_R[b])
            if not path or not os.path.exists(path): continue
            try:
                with fits.open(path, memmap=True) as h:
                    sci = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
                    wcs = WCS(sci.header); data = sci.data; hdr = sci.header
                    for (i, r, d) in srcs:
                        m, s = nisp_tight_aper(r, d, data, wcs, hdr,
                                               nisp_fwhm[b], NISP_TIGHT_APER_MULT)
                        out[i][b] = (m, s)
            except Exception:
                continue
    return out


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
    log(f"available diff bands: {avail}")

    # --- Load RAW aperture photometry + morphology from v05 partials.
    # v04 patches need raw snr_F814W / snr_VIS (persistence vetoes), raw
    # mag in detection band (HST 3σ depth cap), and sharp/rnd morphology
    # (G1-G4 gate to reject extended galaxies).
    raw_SNR = {b: np.full(N, 0.0, dtype=np.float32) for b in ["F814W","VIS"]+SCI_BANDS}
    raw_MAG = {b: np.full(N, -1.0, dtype=np.float32) for b in ["F814W"]+SCI_BANDS}
    SHARP = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    RND1  = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    RND2  = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    SEP   = {b: np.full(N, np.nan, dtype=np.float32) for b in SCI_BANDS}
    NFR   = {b: np.full(N,  1.0, dtype=np.float32) for b in SCI_BANDS}
    SURVEY_BANDS = {"hst": ["F814W"], "jwst": ["F115W","F150W","F277W","F444W"],
                    "vis": ["VIS"], "nisp": ["Y","J","H"]}
    for sv, bands in SURVEY_BANDS.items():
        cp = PART_DIR / f"infer_v05_{sv}.parquet"
        if not cp.exists():
            log(f"  [warn] v05 {sv} partial missing — patches will skip {sv}"); continue
        tt = pq.read_table(cp)
        ii = tt["idx"].to_numpy(zero_copy_only=False).astype(np.int64)
        cols = set(tt.column_names)
        for b in bands:
            if f"snr_{b}" in cols:
                raw_SNR[b][ii] = tt[f"snr_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                raw_MAG[b][ii] = tt[f"mag_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
            # morphology only stored for SCI_BANDS (no F814W — HST is template)
            if b in SHARP and f"sharp_{b}" in cols:
                SHARP[b][ii] = tt[f"sharp_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                RND1[b][ii]  = tt[f"rnd1_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                RND2[b][ii]  = tt[f"rnd2_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                SEP[b][ii]   = tt[f"sep_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
                NFR[b][ii]   = tt[f"nanfrac_{b}"].to_numpy(zero_copy_only=False).astype(np.float32)
    log(f"loaded raw aperture phot + morph from v05 partials")

    G1_TOL = {b: max(0.10, 1.5 * FWHM_AS[b]) for b in SCI_BANDS}
    def band_morph_ok(b, i):
        sp = float(SHARP[b][i]); r1 = float(RND1[b][i]); r2 = float(RND2[b][i])
        sg = float(SEP[b][i]);   nf = float(NFR[b][i])
        if nf > NF_LIM: return False
        if not np.isfinite(sg) or sg > G1_TOL[b]: return False
        if not np.isfinite(sp) or sp < SHARP_LO or sp > SHARP_HI: return False
        if not np.isfinite(r1) or abs(r1) > RND_LIM: return False
        if not np.isfinite(r2) or abs(r2) > RND_LIM: return False
        return True

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

    # PATCH 11: HST DAO KDTree for high-PM star check
    log("  building HST DAO KDTree for PM neighbor check ...")
    t_pm0 = time.time()
    pm_tree, pm_ra, pm_dec, pm_pid, pm_snr, pm_cosd = load_hst_dao_kdtree()
    log(f"  HST DAO tree: {len(pm_ra):,} entries in {time.time()-t_pm0:.1f}s")

    rows = []
    n_hst_visible = 0; n_vis_visible = 0; n_too_faint = 0; n_morph = 0
    n_nisp_wide_visible = 0; n_pm_star = 0
    pm_info = {}
    for i in sn_idx:
        # Pick best band by SNR
        best_band = ""; best_snr = 0.0; best_mag = -1.0
        for b in SCI_BANDS:
            sv = float(SNR[b][i])
            if sv > best_snr and det[b][i]:
                best_snr = sv; best_band = b; best_mag = float(MAG[b][i])
        # --- v04 PATCHES (mirror of v05): persistence vetoes, morphology,
        # and HST 3σ depth cap. These hide the worst FPs that the diff
        # imaging + multi-band rule still let through.
        # (1) HST persistence veto: source visible in HST aperture → not SN
        if float(raw_SNR["F814W"][i]) >= HST_VIS_SNR_VETO:
            n_hst_visible += 1; continue
        # (2) Euclid-VIS persistence veto: same logic
        if float(raw_SNR["VIS"][i]) >= HST_VIS_SNR_VETO:
            n_vis_visible += 1; continue
        # (3) Morphology gate on the detection band (G1-G4 + NaN-frac)
        if best_band and not band_morph_ok(best_band, i):
            n_morph += 1; continue
        # (4) HST 3σ depth cap on the RAW mag in the detection band.
        # If source is too faint for HST to see, HST non-detection is
        # uninformative — can't verify it's a transient.
        raw_best_mag = float(raw_MAG[best_band][i]) if best_band else -1.0
        if raw_best_mag <= 0 or raw_best_mag > MAG_FAINT_HST:
            n_too_faint += 1; continue
        # (5) NISP wide-aperture persistence veto (mirror of v05 patch 9)
        n_nb = sum(1 for b in ("Y","J","H") if float(raw_SNR[b][i]) >= NISP_SNR_VETO)
        if n_nb >= 2:
            n_nisp_wide_visible += 1; continue
        # (6) High-PM star check (mirror of v05 patch 11)
        ra0, dec0 = float(ra[i]), float(dec[i])
        this_pid = str(pid[i])
        pt = np.array([ra0*pm_cosd, dec0])
        nb_idx = pm_tree.query_ball_point(pt, HST_PM_SEARCH_RADIUS_AS/3600.0)
        pm_hit = False
        for k in nb_idx:
            if pm_pid[k] == this_pid: continue
            dra = (pm_ra[k]-ra0)*pm_cosd*3600
            ddec = (pm_dec[k]-dec0)*3600
            sep = math.sqrt(dra*dra + ddec*ddec)
            if 0.5 < sep < HST_PM_SEARCH_RADIUS_AS:
                pm_info[i] = (sep, str(pm_pid[k]), float(pm_snr[k]))
                pm_hit = True; break
        if pm_hit:
            n_pm_star += 1; continue
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
        r["_idx"] = int(i)
        rows.append(r)

    # PATCH 10 (mirror v05): NISP tight-aperture re-measure for survivors
    log(f"  re-measuring NISP tight aperture ({NISP_TIGHT_APER_MULT}xFWHM) for {len(rows)} survivors ...")
    t_t = time.time()
    tight = measure_nisp_tight([r["_idx"] for r in rows], te, ra, dec)
    log(f"  tight NISP done in {time.time()-t_t:.1f}s")
    final_rows = []; n_nisp_tight_visible = 0
    for r in rows:
        i = r["_idx"]
        tn = tight.get(i, {"Y":(-1.0,0.0),"J":(-1.0,0.0),"H":(-1.0,0.0)})
        n_hit = sum(1 for b in ("Y","J","H") if tn[b][1] >= NISP_TIGHT_SNR_VETO)
        if n_hit >= 2:
            n_nisp_tight_visible += 1; continue
        for b in ("Y","J","H"):
            m, s = tn[b]
            r[f"mag_{b}_tight"] = f"{m:.2f}" if m != -1.0 else "-1"
            r[f"snr_{b}_tight"] = f"{s:.2f}"
        if i in pm_info:
            sep, npid, nsnr = pm_info[i]
            r["hst_pm_neighbor_sep"] = f"{sep:.2f}"
            r["hst_pm_neighbor_pid"] = npid
            r["hst_pm_neighbor_snr"] = f"{nsnr:.1f}"
        else:
            r["hst_pm_neighbor_sep"] = ""
            r["hst_pm_neighbor_pid"] = ""
            r["hst_pm_neighbor_snr"] = ""
        r.pop("_idx")
        final_rows.append(r)
    rows = final_rows

    for rk, r in enumerate(rows, start=1):
        r["id"] = f"cand_{rk:05d}"

    log(f"v04 patches: hst_visible(snr>={HST_VIS_SNR_VETO})={n_hst_visible}, "
        f"vis_visible={n_vis_visible}, morph={n_morph}, "
        f"too_faint(raw_mag>{MAG_FAINT_HST})={n_too_faint}, "
        f"nisp_wide(>=2of3@{NISP_SNR_VETO})={n_nisp_wide_visible}, "
        f"nisp_tight(0.3xFWHM)={n_nisp_tight_visible}, "
        f"pm_star(HST_neighbor_within_5\")={n_pm_star}  ->  FINAL {len(rows)}")
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
    page.append(f"<b>Raw v04 diff candidates:</b> {len(sn_idx):,}<br>")
    page.append(f"&minus;hst_visible(snr&ge;{HST_VIS_SNR_VETO}) {n_hst_visible:,}, "
                f"&minus;vis_visible {n_vis_visible:,}, "
                f"&minus;morph {n_morph:,}, "
                f"&minus;too_faint(raw_mag&gt;{MAG_FAINT_HST}) {n_too_faint:,}, "
                f"&minus;nisp_wide(&ge;2of3@{NISP_SNR_VETO}&sigma;) {n_nisp_wide_visible:,}, "
                f"&minus;nisp_tight(0.3&times;FWHM) {n_nisp_tight_visible:,}, "
                f"&minus;pm_star(HST_DAO_within_{HST_PM_SEARCH_RADIUS_AS:.0f}&Prime;) {n_pm_star:,}<br>")
    page.append(f"<b>FINAL candidates after v04 patches:</b> "
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
