"""Optimized polished-webpage generator for top-N v01 SN candidates.

Replaces make_v01_polished_web.py (which opened 9 FITS files per candidate)
with tile-major batched IO: each unique (band, tile) FITS file is opened
exactly once, all candidate cutouts in it are sliced together, then a
single pool of workers parallelises across tiles.

For N=100 candidates expect ~150 file opens (vs 900) and ~60-120 s
wall-clock end-to-end.

Reuses:
  - recut_single_source.render_{gray,rgb_jwst,rgb_nisp}  (band recipes)
  - 04_polish_labels.overlay_crosshair / repaint_labels   (crosshair+labels)
  - recut_3_hst.resolve_jwst_path / resolve_euclid_path   (A10/superseded)
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, csv, time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R
from recut_single_source import render_gray, render_rgb_jwst, render_rgb_nisp
from importlib import import_module
polish = import_module("04_polish_labels")

CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v01.csv")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v01/polished_top100")
TMP_DIR  = OUT_DIR / "_raw"

TOP_N    = 100
CUT_AS   = 6.0   # arcsec
N_WORKERS = max(2, cpu_count() // 2)

PANELS = ["jwst1", "jwst2", "hst", "euvis", "eunisp"]
PANEL_LABEL = {
    "jwst1":  "JWST F115/F150/F277 (blue)",
    "jwst2":  "JWST F150/F277/F444 (red)",
    "hst":    "HST F814W",
    "euvis":  "Euclid VIS",
    "eunisp": "Euclid NISP YJH",
}

# All (survey, band_label, R_band) we extract per candidate
BAND_REQS = [
    ("hst",    "F814W", "F814W"),
    ("jwst",   "F115W", "f115w"),
    ("jwst",   "F150W", "f150w"),
    ("jwst",   "F277W", "f277w"),
    ("jwst",   "F444W", "f444w"),
    ("euclid", "VIS",   "VIS"),
    ("euclid", "Y",     "NIR-Y"),
    ("euclid", "J",     "NIR-J"),
    ("euclid", "H",     "NIR-H"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_path(survey, R_band, tile):
    if not tile or tile in ("", "nan", "None"):
        return None
    if survey == "hst":
        return f"{R.HST_DIR}/acs_I_030mas_{tile}_sci.fits"
    if survey == "jwst":
        return R.resolve_jwst_path(tile, R_band.lower())
    return R.resolve_euclid_path(tile, R_band)


def worker_extract(args):
    """Open one (survey, band_label, R_band, tile) FITS once, return cutout
    arrays for all (cand_idx, ra, dec) in it. Returns dict keyed by cand_idx."""
    import warnings; warnings.filterwarnings("ignore")
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.nddata import Cutout2D
    import numpy as np

    survey, band, R_band, tile, reqs = args
    path = resolve_path(survey, R_band, tile)
    out = {}
    if path is None or not os.path.exists(path):
        for cand_idx, _, _ in reqs:
            out[(cand_idx, survey, band)] = None
        return out
    try:
        with fits.open(path, memmap=True) as h:
            sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
            wcs = WCS(sci_hdu.header)
            data = sci_hdu.data
            for cand_idx, ra, dec in reqs:
                pos = SkyCoord(ra*u.deg, dec*u.deg)
                try:
                    c = Cutout2D(data.astype(np.float32), pos,
                                 size=CUT_AS * u.arcsec, wcs=wcs,
                                 mode="partial", fill_value=0.0)
                    out[(cand_idx, survey, band)] = c.data
                except Exception:
                    out[(cand_idx, survey, band)] = None
    except Exception as e:
        for cand_idx, _, _ in reqs:
            out[(cand_idx, survey, band)] = None
    return out


def _write_black(path):
    Image.new("RGB", (480, 480), (10, 10, 10)).save(str(path))


def render_candidate_pngs(cand_idx, cand, all_cuts, tmp_dir, out_dir):
    """Given the dict of all cached cutouts, render 5 raw PNGs then apply
    crosshair + labels and save final polished PNGs."""
    cid = cand["id"]
    seq = cand_idx + 1
    # gather cutouts (could be None)
    def get(survey, band):
        return all_cuts.get((cand_idx, survey, band))
    hst    = get("hst", "F814W")
    f115   = get("jwst", "F115W")
    f150   = get("jwst", "F150W")
    f277   = get("jwst", "F277W")
    f444   = get("jwst", "F444W")
    vis    = get("euclid", "VIS")
    y_     = get("euclid", "Y")
    j_     = get("euclid", "J")
    h_     = get("euclid", "H")

    # raw paths (intermediate)
    hst_png    = tmp_dir / f"sn_v01_{seq:04d}_hst.png"
    vis_png    = tmp_dir / f"sn_v01_{seq:04d}_euvis.png"
    nisp_png   = tmp_dir / f"sn_v01_{seq:04d}_eunisp.png"
    jwst1_png  = tmp_dir / f"sn_v01_{seq:04d}_jwst1.png"
    jwst2_png  = tmp_dir / f"sn_v01_{seq:04d}_jwst2.png"
    blank = []
    if hst is not None: render_gray(hst, hst_png, blank)
    else: _write_black(hst_png)
    if vis is not None: render_gray(vis, vis_png, blank)
    else: _write_black(vis_png)
    if y_ is not None and j_ is not None and h_ is not None:
        render_rgb_nisp(y_, j_, h_, nisp_png, blank)
    else: _write_black(nisp_png)
    if f115 is not None and f150 is not None and f277 is not None:
        render_rgb_jwst(f115, f150, f277, jwst1_png, blank)
    else: _write_black(jwst1_png)
    if f150 is not None and f277 is not None and f444 is not None:
        render_rgb_jwst(f150, f277, f444, jwst2_png, blank)
    else: _write_black(jwst2_png)

    # apply polish
    telescope = cand["telescope"].strip().upper()
    if telescope == "HST":         det_panels = {"hst"}
    elif telescope.startswith("EUCLID-NISP") or telescope == "NISP": det_panels = {"eunisp"}
    elif telescope.startswith("EUCLID-VIS") or telescope == "VIS":  det_panels = {"euvis"}
    else: det_panels = {"jwst1", "jwst2"}
    arms = ["right", "up"]

    def loadc(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    def loadg(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0

    jw1 = loadc(jwst1_png)
    jw2 = loadc(jwst2_png)
    nisp_rgb = loadc(nisp_png)
    hst_g = loadg(hst_png)
    vis_g = loadg(vis_png)
    h_im, w_im = jw1.shape[:2]
    cx, cy = w_im // 2, h_im // 2
    panels_rgb = {
        "jwst1":  (jw1 * 255).clip(0, 255).astype(np.uint8),
        "jwst2":  (jw2 * 255).clip(0, 255).astype(np.uint8),
        "hst":    np.stack([(polish.resize_to(hst_g, w_im, h_im) * 255).astype(np.uint8)] * 3, axis=-1),
        "euvis":  np.stack([(polish.resize_to(vis_g, w_im, h_im) * 255).astype(np.uint8)] * 3, axis=-1),
        "eunisp": (polish.resize_to(nisp_rgb, w_im, h_im) * 255).clip(0, 255).astype(np.uint8),
    }

    def gm(k):
        try:
            v = float(cand[k])
            return v if (np.isfinite(v) and v > 0) else float("nan")
        except (ValueError, TypeError, KeyError):
            return float("nan")
    sn_mags = {b: gm(f"mag_{b}") for b in
               ("F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W")}
    try:
        best_snr = float(cand["best_snr"])
        if best_snr <= 0: best_snr = None
    except (ValueError, TypeError): best_snr = None
    best_band = cand["best_band"] or None
    z = float("nan")

    for name in PANELS:
        img = panels_rgb[name]
        spec = polish.PANEL_SPEC[name]
        is_det = name in det_panels
        style = "solid" if is_det else "dotted"
        out_img = polish.overlay_crosshair(img, (cx, cy), arms, style=style)
        header = f"{spec['survey']} {spec['filter_str']}"
        bluest = spec["bands"][0]
        host_m = sn_mags.get(bluest, float("nan"))
        sn_band_list = spec["bands"]
        sn_mag_list  = [sn_mags.get(b, -1.0) for b in sn_band_list]
        out_img = polish.repaint_labels(
            out_img, cid, z, header, bluest, host_m,
            sn_band_list, sn_mag_list,
            show_sn_line=is_det,
            best_snr=best_snr if is_det else None,
            best_band=best_band if is_det else None,
            show_scale_bar=(name == "hst"),
        )
        Image.fromarray(out_img).save(str(out_dir / f"polished_{cid}_{name}.png"))


def main():
    t_total = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Loading {CAND_CSV.name}")
    with CAND_CSV.open() as f:
        cands = list(csv.DictReader(f))
    top = cands[:TOP_N]
    log(f"Top {len(top)} candidates")

    # ---- build requests grouped by (survey, band, R_band, tile) ----
    t_plan = time.time()
    by_tile = defaultdict(list)   # (survey, band, R_band, tile) -> list of (cand_idx, ra, dec)
    for ci, cand in enumerate(top):
        ra = float(cand["sn_ra"]); dec = float(cand["sn_dec"])
        tiles = {"hst": cand["hst_tile"] or None,
                 "jwst": cand["jwst_tile"] or None,
                 "euclid": cand["euclid_tile"] or None}
        for (survey, band, R_band) in BAND_REQS:
            t = tiles[survey]
            if not t or t in ("", "nan", "None"): continue
            by_tile[(survey, band, R_band, t)].append((ci, ra, dec))
    jobs = [(s, b, rb, t, lst) for (s, b, rb, t), lst in by_tile.items()]
    log(f"Planning: {len(jobs)} unique (survey,band,tile) groups for {len(top)} candidates "
        f"(naive would be {len(top)*len(BAND_REQS)} opens) [{time.time()-t_plan:.2f}s]")

    # ---- parallel IO ----
    t_io = time.time()
    log(f"Extracting cutouts with {N_WORKERS} workers ...")
    all_cuts = {}
    with Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_extract, jobs)):
            all_cuts.update(result)
            if (i+1) % 50 == 0 or i == len(jobs)-1:
                log(f"  IO: [{i+1}/{len(jobs)}] groups done  ({time.time()-t_io:.1f}s)")
    log(f"IO phase: {time.time()-t_io:.1f}s  (cutouts stored: {len(all_cuts):,})")

    # ---- render PNGs ----
    t_ren = time.time()
    log("Rendering polished PNGs ...")
    for ci, cand in enumerate(top):
        render_candidate_pngs(ci, cand, all_cuts, TMP_DIR, OUT_DIR)
        if (ci+1) % 25 == 0 or ci == len(top)-1:
            log(f"  render: [{ci+1}/{len(top)}]  ({time.time()-t_ren:.1f}s)")
    log(f"Render phase: {time.time()-t_ren:.1f}s")

    # ---- v33-style index.html ----
    t_html = time.time()
    html = ["<!doctype html><html><head>",
            f"<title>v01 top {TOP_N} SN candidates — v33-style polished</title>",
            "<style>",
            "body{font-family:serif;background:#fafafa;margin:0;padding:0}",
            ".wrap{max-width:1500px;margin:0 auto;padding:20px}",
            ".back{font-size:13px;color:#666}",
            ".meta{color:#666;font-style:italic;margin-bottom:16px}",
            ".src{margin-bottom:30px;padding:12px;background:white;border:1px solid #ddd}",
            ".src h3{margin:0 0 8px 0;font-size:15px;color:#222}",
            ".bug{background:#fee;border:1px solid #d00;padding:10px 14px;margin:14px 0}",
            ".panels{display:flex;flex-wrap:wrap;gap:6px}",
            ".panel{text-align:center;font-size:11px;color:#666}",
            ".panel img{width:240px;display:block;border:1px solid #eee}",
            ".panel .lab{font-weight:bold;color:#333;margin-top:3px}",
            ".panel .fn{color:#999;font-family:monospace;font-size:10px}",
            "</style></head><body><div class='wrap'>",
            f"<p class='back'><a href='../index.html'>&larr; sn_search v01</a></p>",
            f"<h1>v01 &mdash; top {TOP_N} SN candidates (visual inspection)</h1>",
            f"<p class='meta'>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} &mdash; "
            f"layout matches <code>htmls/_crosshair_test/v33/</code>.</p>",
            "<div class='bug'><b>v01 had a labeling bug:</b> all 17 known SNe were "
            "marked positive in every survey. Top candidates are dominated by "
            "host-galaxy false positives. v02 with corrected per-survey labels "
            "is pending visual feedback from this page.</div>"]
    for cand in top:
        cid = cand["id"]
        tel = cand["telescope"]
        bb = cand["best_band"] or "?"
        bs = cand["best_snr"]
        cc = cand["composite_confidence"]
        html.append(f"<div class='src'><h3>{cid} &nbsp; "
                    f"<span style='font-weight:normal;color:#666'>"
                    f"{tel} &middot; best {bb}@{bs}&sigma; &middot; "
                    f"comp_conf={cc} &middot; primary_id={cand['primary_id']}"
                    f"</span></h3><div class='panels'>")
        for panel in PANELS:
            html.append(f"<div class='panel'><img src='polished_{cid}_{panel}.png'>"
                        f"<div class='lab'>{PANEL_LABEL[panel]}</div>"
                        f"<div class='fn'>polished_{cid}_{panel}.png</div></div>")
        html.append("</div></div>")
    html.append("</div></body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(html))
    log(f"HTML: {time.time()-t_html:.1f}s")
    log(f"TOTAL: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()
