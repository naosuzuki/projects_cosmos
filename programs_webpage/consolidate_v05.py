"""v05 consolidator: read v05 partial parquets, apply 1-of-3 telescope rule
+ morphology G1-G4 + cross-band consistency + dN/dm prior, write
csvfiles_sn/tbl_sn_candidates_v05.csv and htmls/sn_search/v05/index.html.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, csv, time, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
MASTER = Path("/Users/suzuki/github/projects_cosmos/csvfiles_star/master_or_catalog_v03.parquet")

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
# HST F814W 5σ depth in COSMOS ≈ 27.0 mag. A candidate fainter than this in
# its DETECTION band cannot be cross-validated against HST (HST literally
# cannot see it), so HST non-detection becomes uninformative. Cap accepted
# candidates at HST depth — primarily restricts JWST detections (JWST F115W
# goes to ~27.2, F444W to ~27.6) which can find faint sources HST misses.
# User-set 2026-05-28 after reviewing v05 cand_00001 (mag_F814W=27.86) and
# cand_00002 (mag_F814W=28.32) as JWST-only-because-too-faint, not transients.
# User 2026-05-28 (after reviewing v05 top 25 — 0/25 were real SNe, all
# persistent multi-survey or extended galaxies). Four interim patches:
#   (1) MAG_FAINT_HST = HST 3σ depth (was 5σ): visually we can see down to
#       3σ; sources fainter than 3σ can't be cross-checked by HST.
#   (2) SHARP_HI tightened 0.85 → 0.75: rejects extended galaxies with
#       compact knots that previously slipped through.
#   (3) HST persistence veto: snr_F814W ≥ 3 → reject (visible in HST = not
#       a transient).
#   (4) Euclid-VIS persistence veto: snr_VIS ≥ 3 → reject (same reasoning).
# Underlying CNN is still flawed (it ranks persistent compacts highly); v06
# multi-epoch training will fix that. These patches just hide the worst FPs.
MAG_FAINT_HST = 27.8        # HST F814W 3σ point-source depth (COSMOS-Web)
HST_VIS_SNR_VETO = 3.0       # reject if HST or Euclid-VIS aperture snr ≥ this
# 2026-05-28 (later): NISP persistence veto. The blending fear that originally
# excluded NISP from the veto turned out NOT to be the dominant case for the
# top candidates — for #2/#4/#5/#7/#8 the NISP DAO peak is squarely at the
# candidate position (sep_Y ≤ 0.3") with consistent Y/J/H mags ≈ 20 (high-z
# dropout galaxies, bright IR, faint optical). Use ≥2 of 3 NISP bands above
# NISP_SNR_VETO to qualify as persistent (the "2-of-3 bands" requirement
# protects against single-band noise spikes that could mimic NISP detection).
NISP_SNR_VETO = 3.0
# 2026-05-28 patch 6: NISP TIGHT-APERTURE re-measurement. Original NISP
# aperture (1.0 × FWHM ≈ 0.55″) sums host-galaxy + nearby-source flux,
# inflating mags by 2-3 mag (see candidates 671543, 481171, 548679: tight
# 0.3 × FWHM aperture gives mag ~23-24, wide gives mag ~20-21). The veto
# above uses the WIDE-aperture snr (still useful for catching strong cases).
# For BORDERLINE cases — wide-aperture snr below veto threshold but real
# faint source AT-position — we re-measure with TIGHT aperture and apply
# the veto on tight snr too.
NISP_TIGHT_APER_MULT = 0.3   # fraction of FWHM for tight aperture
NISP_TIGHT_SNR_VETO = 3.0    # apply same SNR threshold on tight measurement

# 2026-05-28 patch 7: high-PM star check. Some stars have proper motion
# 2-5″ over the HST-to-JWST baseline (17 yr) — they appear as TWO
# separate catalog entries (different positions, never matched). The
# HST persistence veto (snr_F814W at JWST position) misses them since the
# HST flux moved away. Catch them by searching the HST DAO catalog for
# ANY detection within HST_PM_SEARCH_RADIUS_AS of the candidate position
# that is not the candidate itself. User identified examples: #1 (270470)
# has HST DAO at 2.5″, #16 #22 same pattern.
HST_PM_SEARCH_RADIUS_AS = 5.0
HST_PM_NEIGHBOR_SNR_MIN = 5.0    # require HST DAO snr above this to flag

# Morphology gates (G2-G4 in CLAUDE.md §2.1) — accidentally dropped earlier
SHARP_LO, SHARP_HI = 0.40, 0.75
RND_LIM = 0.50
NF_LIM  = 0.25
RND_LIM = 0.50
NF_LIM  = 0.25
TOP_N_WEB = 500


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_hst_dao_kdtree():
    """Load HST DAO positions from master catalog, build KDTree for fast 5\"
    neighbor search. Returns (tree, ra_arr, dec_arr, pid_arr, snr_arr)."""
    t = pq.read_table(MASTER, columns=["primary_id","dao_F814W_ra_hst",
                                       "dao_F814W_dec_hst","dao_F814W_snr_hst"])
    ra = t["dao_F814W_ra_hst"].to_numpy(zero_copy_only=False)
    dec = t["dao_F814W_dec_hst"].to_numpy(zero_copy_only=False)
    snr = t["dao_F814W_snr_hst"].to_numpy(zero_copy_only=False)
    pids = np.array(t["primary_id"].to_pylist(), dtype=object)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(snr) & (snr >= HST_PM_NEIGHBOR_SNR_MIN)
    ra = ra[ok]; dec = dec[ok]; snr = snr[ok]; pids = pids[ok]
    # KDTree on equirectangular projection (RA × cos(median dec), Dec) in deg,
    # then convert query radius to deg via simple cos(dec) factor at median dec
    med_dec = float(np.median(dec))
    cosd = math.cos(math.radians(med_dec))
    coords = np.column_stack([ra * cosd, dec])
    tree = cKDTree(coords)
    return tree, ra, dec, pids, snr, cosd


def nisp_tight_aper(ra, dec, sci_data, wcs, hdr, fwhm_as, aper_mult):
    """Aperture photometry at smaller aperture for one source.
    Returns (mag, snr)."""
    try:
        sx, sy = wcs.all_world2pix(ra, dec, 0)
    except Exception:
        return -1.0, 0.0
    sx, sy = float(sx), float(sy)
    if not (np.isfinite(sx) and np.isfinite(sy)): return -1.0, 0.0
    ps = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    cy, cx = int(round(sy)), int(round(sx))
    n = 30; half = n // 2
    y0, y1 = cy-half, cy+half; x0, x1 = cx-half, cx+half
    ny, nx = sci_data.shape[-2:]
    if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx: return -1.0, 0.0
    sub = sci_data[y0:y1, x0:x1].astype(np.float64)
    cy2 = sy - y0; cx2 = sx - x0
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx-cx2)**2 + (yy-cy2)**2)
    aper_r = aper_mult * fwhm_as / ps
    ring_in = 2.0 * fwhm_as / ps; ring_out = 3.5 * fwhm_as / ps
    in_aper = rr <= aper_r; in_ring = (rr >= ring_in) & (rr <= ring_out)
    if int(in_aper.sum()) < 1 or int(in_ring.sum()) < 5: return -1.0, 0.0
    bg = float(np.nanmedian(sub[in_ring])); sub2 = sub - bg
    n_ap = int(in_aper.sum())
    flux = float(np.nansum(sub2[in_aper]))
    sig = float(np.nanstd(sub[in_ring])) * math.sqrt(n_ap)
    snr = flux / (sig + 1e-30)
    zp = (hdr.get("ZP") or hdr.get("MAGZP") or hdr.get("PHOTZP") or 23.9)
    mag = (float(zp) - 2.5 * math.log10(flux)) if flux > 0 else -1.0
    return mag, float(snr)


def measure_nisp_tight(survivors, te_array, ra_array, dec_array):
    """Re-measure NISP photometry with tight aperture for the candidate
    indices in `survivors`. Returns dict keyed by source index: per-band
    (mag, snr). Groups requests by tile to minimize FITS opens."""
    by_tile = defaultdict(list)
    for i in survivors:
        t = str(te_array[i]) if te_array[i] else None
        if t: by_tile[t].append((i, float(ra_array[i]), float(dec_array[i])))
    nisp_fwhm = {"Y": FWHM_AS["Y"], "J": FWHM_AS["J"], "H": FWHM_AS["H"]}
    nisp_R = {"Y": "NIR-Y", "J": "NIR-J", "H": "NIR-H"}
    out = {i: {b: (-1.0, 0.0) for b in ("Y","J","H")} for i in survivors}
    for tile, srcs in by_tile.items():
        for b in ("Y","J","H"):
            path = R.resolve_euclid_path(tile, nisp_R[b])
            if not path or not os.path.exists(path): continue
            try:
                with fits.open(path, memmap=True) as h:
                    sci = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
                    wcs = WCS(sci.header)
                    data = sci.data
                    hdr = sci.header
                    for (i, r, d) in srcs:
                        m, s = nisp_tight_aper(r, d, data, wcs, hdr,
                                               nisp_fwhm[b], NISP_TIGHT_APER_MULT)
                        out[i][b] = (m, s)
            except Exception:
                continue
    return out


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

    # PATCH 7: load HST DAO KDTree for the high-PM star neighbor check
    log("  building HST DAO KDTree for high-PM neighbor search ...")
    t_pm0 = time.time()
    pm_tree, pm_ra, pm_dec, pm_pid, pm_snr, pm_cosd = load_hst_dao_kdtree()
    log(f"  HST DAO tree: {len(pm_ra):,} entries, built in {time.time()-t_pm0:.1f}s")
    rows = []
    n_sat = 0; n_morph = 0; n_cross = 0; n_snr = 0; n_mag = 0; n_too_faint = 0
    n_hst_visible = 0; n_vis_visible = 0; n_nisp_visible = 0   # persistence vetoes
    n_pm_star = 0   # PATCH 7: high-PM HST neighbor
    pm_info = {}    # idx -> (sep_arcsec, neighbor_pid, neighbor_snr) for diagnostics
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
        # HST-depth cap: reject if detection-band mag is fainter than HST can
        # cross-check. Mostly bites JWST detections (the only telescope that
        # goes deeper than HST). Without this we accept "JWST sees it, HST
        # doesn't" cases that are actually "HST can't see anything that faint".
        if best_mag > MAG_FAINT_HST: n_too_faint += 1; continue
        # PERSISTENCE VETOES (added 2026-05-28). A real SN is absent in
        # both the HST epoch (2005-2008) and Euclid VIS depth. If either
        # shows the source at ≥ 3σ aperture SNR, it's a persistent host —
        # not a transient.
        if float(SNR["F814W"][i]) >= HST_VIS_SNR_VETO:
            n_hst_visible += 1; continue
        if float(SNR["VIS"][i]) >= HST_VIS_SNR_VETO:
            n_vis_visible += 1; continue
        # NISP persistence veto: high-z dropout galaxies (bright in NISP IR,
        # faint in optical HST/VIS) mimic the SN signature in HST + VIS. If
        # NISP catches a source at this position in ≥2 of 3 bands at ≥3σ,
        # it's a real IR source — reject.
        n_nisp_bands_hit = sum(1 for b in ("Y","J","H") if float(SNR[b][i]) >= NISP_SNR_VETO)
        if n_nisp_bands_hit >= 2:
            n_nisp_visible += 1; continue
        # PATCH 7: high-PM star check — search HST DAO within 5″ for an
        # unmatched detection (catches stars that moved >0.5″ between HST and
        # JWST epochs, so they decouple as TWO catalog entries).
        ra0, dec0 = float(ra[i]), float(dec[i])
        this_pid = str(pid[i])
        pt = np.array([ra0*pm_cosd, dec0])
        nb_idx = pm_tree.query_ball_point(pt, HST_PM_SEARCH_RADIUS_AS/3600.0)
        pm_hit = False
        for k in nb_idx:
            if pm_pid[k] == this_pid: continue   # candidate's own HST entry
            dra = (pm_ra[k]-ra0)*pm_cosd*3600
            ddec = (pm_dec[k]-dec0)*3600
            sep = math.sqrt(dra*dra + ddec*ddec)
            if 0.5 < sep < HST_PM_SEARCH_RADIUS_AS:
                pm_info[i] = (sep, str(pm_pid[k]), float(pm_snr[k]))
                pm_hit = True; break
        if pm_hit:
            n_pm_star += 1; continue
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
        r["_idx"] = int(i)   # private — popped before CSV write
        rows.append(r)

    # PATCH 6: NISP TIGHT-APERTURE re-measurement for survivors. Big aperture
    # was used for the wide-aperture veto (caught the obvious cases); now
    # re-measure NISP at 0.3 × FWHM aperture and apply the same SNR threshold.
    log(f"  re-measuring NISP at tight aperture ({NISP_TIGHT_APER_MULT}xFWHM) for {len(rows)} survivors ...")
    t_t = time.time()
    tight = measure_nisp_tight([r["_idx"] for r in rows], te, ra, dec)
    log(f"  tight NISP done in {time.time()-t_t:.1f}s")
    final_rows = []; n_nisp_tight_visible = 0
    for r in rows:
        i = r["_idx"]
        tnphot = tight.get(i, {"Y":(-1.0,0.0),"J":(-1.0,0.0),"H":(-1.0,0.0)})
        n_hit = sum(1 for b in ("Y","J","H") if tnphot[b][1] >= NISP_TIGHT_SNR_VETO)
        if n_hit >= 2:
            n_nisp_tight_visible += 1
            continue
        # add tight measurements + PM info to the row for the CSV
        for b in ("Y","J","H"):
            m, s = tnphot[b]
            r[f"mag_{b}_tight"]  = f"{m:.2f}" if m != -1.0 else "-1"
            r[f"snr_{b}_tight"]  = f"{s:.2f}"
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
    rows.sort(key=lambda r: -float(r["composite_confidence"]))
    for rk, r in enumerate(rows, start=1):
        r["id"] = f"cand_{rk:05d}"
    log(f"raw 1-of-3: {int(is_sn.sum()):,}  filters: sat={n_sat} morph={n_morph} "
        f"too_faint(>HST_depth_{MAG_FAINT_HST}) = {n_too_faint}; "
        f"hst_visible(snr>={HST_VIS_SNR_VETO})={n_hst_visible} "
        f"vis_visible={n_vis_visible} "
        f"nisp_visible_wide(>=2of3@{NISP_SNR_VETO}sigma)={n_nisp_visible} "
        f"nisp_visible_tight(0.3xFWHM)={n_nisp_tight_visible} "
        f"pm_star({HST_PM_SEARCH_RADIUS_AS:.0f}\"_HST_neighbor)={n_pm_star}")
    log(f"  ... continued: "
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
            f"&minus;too_faint(>HST_3&sigma;_depth_{MAG_FAINT_HST}) {n_too_faint}, "
            f"&minus;hst_visible(snr&ge;{HST_VIS_SNR_VETO}) {n_hst_visible}, "
            f"&minus;vis_visible(snr&ge;{HST_VIS_SNR_VETO}) {n_vis_visible}, "
            f"&minus;nisp_visible_wide(&ge;2of3@{NISP_SNR_VETO}&sigma;) {n_nisp_visible}, "
            f"&minus;nisp_visible_tight(0.3&times;FWHM) {n_nisp_tight_visible}, "
            f"&minus;pm_star(HST_DAO_within_{HST_PM_SEARCH_RADIUS_AS:.0f}&Prime;) {n_pm_star}, "
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
