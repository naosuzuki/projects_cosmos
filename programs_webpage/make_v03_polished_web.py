"""Live polished-gallery for v03 SN candidates.

Streams from tbl_sn_candidates_v03.csv as the inference workers update it.
For each new candidate (keyed by primary_id, so stable across re-rankings),
renders 5 polished PNGs once and writes an index.html that reflects the
current top-N ranking. Old candidate PNGs are kept so re-ranking is free.

Layout / styling matches htmls/sn_search/v01/polished_top100/index.html.

Outputs:
  htmls/sn_search/v03/polished_top100/index.html
  htmls/sn_search/v03/polished_top100/polished_<primary_id>_<panel>.png

Usage:
    python make_v03_polished_web.py             # loop forever (REFRESH s)
    python make_v03_polished_web.py --once      # one pass
    python make_v03_polished_web.py --top 200   # top-N override
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, csv, time, re
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

CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v03.csv")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v03/polished_top100")
TMP_DIR  = OUT_DIR / "_raw"

REFRESH   = 30                       # seconds between re-checks
DEFAULT_TOP = 200
CUT_AS    = 6.0                      # arcsec
N_WORKERS = max(2, cpu_count() // 2)

PANELS = ["jwst1", "jwst2", "hst", "euvis", "eunisp"]
PANEL_LABEL = {
    "jwst1":  "JWST F115/F150/F277 (blue)",
    "jwst2":  "JWST F150/F277/F444 (red)",
    "hst":    "HST F814W",
    "euvis":  "Euclid VIS",
    "eunisp": "Euclid NISP YJH",
}

# All (survey, band_label, R_band) extracted per candidate
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


def safe_pid(pid):
    """File-safe primary_id token."""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(pid))


def resolve_path(survey, R_band, tile):
    if not tile or tile in ("", "nan", "None"):
        return None
    if survey == "hst":
        return f"{R.HST_DIR}/acs_I_030mas_{tile}_sci.fits"
    if survey == "jwst":
        return R.resolve_jwst_path(tile, R_band.lower())
    return R.resolve_euclid_path(tile, R_band)


def worker_extract(args):
    """Open one (survey, band, tile) FITS once, return cutouts for all sources."""
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
        for key, _, _ in reqs:
            out[(key, survey, band)] = None
        return out
    try:
        with fits.open(path, memmap=True) as h:
            sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
            wcs = WCS(sci_hdu.header)
            data = sci_hdu.data
            for key, ra, dec in reqs:
                pos = SkyCoord(ra*u.deg, dec*u.deg)
                try:
                    c = Cutout2D(data, pos, size=CUT_AS * u.arcsec, wcs=wcs,
                                 mode="partial", fill_value=0.0)
                    out[(key, survey, band)] = c.data.astype(np.float32)
                except Exception:
                    out[(key, survey, band)] = None
    except Exception:
        for key, _, _ in reqs:
            out[(key, survey, band)] = None
    return out


def _write_black(path):
    Image.new("RGB", (480, 480), (10, 10, 10)).save(str(path))


def render_candidate_pngs(key, cand, all_cuts, tmp_dir, out_dir):
    """Render the 5 polished PNGs for a candidate keyed by `key` (= safe_pid)."""
    def get(survey, band):
        return all_cuts.get((key, survey, band))
    hst    = get("hst", "F814W")
    f115   = get("jwst", "F115W")
    f150   = get("jwst", "F150W")
    f277   = get("jwst", "F277W")
    f444   = get("jwst", "F444W")
    vis    = get("euclid", "VIS")
    y_     = get("euclid", "Y")
    j_     = get("euclid", "J")
    h_     = get("euclid", "H")

    hst_png   = tmp_dir / f"raw_{key}_hst.png"
    vis_png   = tmp_dir / f"raw_{key}_euvis.png"
    nisp_png  = tmp_dir / f"raw_{key}_eunisp.png"
    jwst1_png = tmp_dir / f"raw_{key}_jwst1.png"
    jwst2_png = tmp_dir / f"raw_{key}_jwst2.png"

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

    telescope = cand["telescope"].strip().upper()
    if telescope == "HST":
        det_panels = {"hst"}
    elif telescope == "EUCLID-VIS+NISP":
        det_panels = {"euvis", "eunisp"}   # both Euclid bands fired
    elif telescope.startswith("EUCLID-NISP") or telescope == "NISP":
        det_panels = {"eunisp"}
    elif telescope.startswith("EUCLID-VIS") or telescope == "VIS":
        det_panels = {"euvis"}
    else:
        det_panels = {"jwst1", "jwst2"}
    arms = ["right", "up"]

    def loadc(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    def loadg(p): return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0

    jw1 = loadc(jwst1_png); jw2 = loadc(jwst2_png)
    nisp_rgb = loadc(nisp_png)
    hst_g = loadg(hst_png); vis_g = loadg(vis_png)
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
            v = float(cand[k]);  return v if (np.isfinite(v) and v > 0) else float("nan")
        except (ValueError, TypeError, KeyError):
            return float("nan")
    sn_mags = {b: gm(f"mag_{b}") for b in
               ("F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W")}
    try:
        best_snr = float(cand["best_snr"]);  best_snr = best_snr if best_snr > 0 else None
    except (ValueError, TypeError):
        best_snr = None
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
            out_img, key, z, header, bluest, host_m,
            sn_band_list, sn_mag_list,
            show_sn_line=is_det,
            best_snr=best_snr if is_det else None,
            best_band=best_band if is_det else None,
            show_scale_bar=(name == "hst"),
        )
        Image.fromarray(out_img).save(str(out_dir / f"polished_{key}_{name}.png"))


def build_html(top_cands, top_n, n_total, run_status):
    """Render the v33-style polished gallery index.html."""
    html = ["<!doctype html><html><head>",
            f"<title>v03 top {top_n} SN candidates &mdash; live polished gallery</title>",
            "<meta http-equiv='refresh' content='30'>",
            "<style>",
            "body{font-family:serif;background:#fafafa;margin:0;padding:0}",
            ".wrap{max-width:1500px;margin:0 auto;padding:20px}",
            ".back{font-size:13px;color:#666}",
            ".meta{color:#666;font-style:italic;margin-bottom:16px}",
            ".src{margin-bottom:30px;padding:12px;background:white;border:1px solid #ddd}",
            ".src h3{margin:0 0 8px 0;font-size:15px;color:#222}",
            ".note{background:#fff5b0;border:1px solid #d4a017;padding:10px 14px;margin:14px 0}",
            ".panels{display:flex;flex-wrap:wrap;gap:6px}",
            ".panel{text-align:center;font-size:11px;color:#666}",
            ".panel img{width:240px;display:block;border:1px solid #eee}",
            ".panel .lab{font-weight:bold;color:#333;margin-top:3px}",
            ".panel .fn{color:#999;font-family:monospace;font-size:10px}",
            "</style></head><body><div class='wrap'>",
            "<p class='back'><a href='../index.html'>&larr; sn_search v03 stats page</a></p>",
            f"<h1>v03 &mdash; live top {top_n} SN candidates (visual inspection)</h1>",
            f"<p class='meta'>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} &mdash; "
            f"showing {len(top_cands)} of {n_total} candidates currently surviving v03 "
            f"rules (saturation guard mag&lt;21, morphology G1-G4, cross-band consistency, "
            f"SNR&ge;5, mag&ge;21, dN/dm prior). Page auto-refreshes every 30s.</p>",
            f"<div class='note'>{run_status}</div>"]
    for cand in top_cands:
        pid = cand["primary_id"]; key = safe_pid(pid)
        cid = cand["id"]
        tel = cand["telescope"]
        bb = cand["best_band"] or "?"
        bs = cand["best_snr"]
        cc = cand["composite_confidence"]
        html.append(f"<div class='src'><h3>{cid} &nbsp; "
                    f"<span style='font-weight:normal;color:#666'>"
                    f"{tel} &middot; best {bb}@{bs}&sigma; &middot; "
                    f"comp_conf={cc} &middot; primary_id={pid}"
                    f"</span></h3><div class='panels'>")
        for panel in PANELS:
            html.append(f"<div class='panel'><img src='polished_{key}_{panel}.png'>"
                        f"<div class='lab'>{PANEL_LABEL[panel]}</div>"
                        f"<div class='fn'>polished_{key}_{panel}.png</div></div>")
        html.append("</div></div>")
    html.append("</div></body></html>")
    return "\n".join(html)


def run_status_line():
    """Read the run status log tail to figure out worker progress."""
    log_path = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/run_v03_status.log")
    if not log_path.exists():
        return "Inference run status: log not yet written."
    progress = {}
    try:
        with log_path.open() as f:
            for line in f:
                m = re.search(r"\[(\w+)\] tile \[(\d+)/(\d+)\] .* acc=([\d,]+)/([\d,]+)", line)
                if m:
                    sv, ti, total, acc, ntot = m.groups()
                    progress[sv] = (int(ti), int(total),
                                    int(acc.replace(",", "")), int(ntot.replace(",", "")))
    except Exception:
        return "Inference status unavailable."
    if not progress:
        return "Inference status: workers warming up."
    parts = []
    for sv in ("hst","jwst","nisp"):
        if sv in progress:
            ti, total, acc, ntot = progress[sv]
            parts.append(f"{sv.upper()} {ti}/{total} ({100*ti/total:.0f}%, {acc:,}/{ntot:,} sources)")
    return ("<b>Inference progress:</b> " + " &middot; ".join(parts)) if parts else "warming up"


def one_pass(top_n, rendered_keys):
    """One pass: read CSV, render missing PNGs, rebuild HTML."""
    if not CAND_CSV.exists():
        log(f"  {CAND_CSV.name} not present yet")
        return rendered_keys

    with CAND_CSV.open() as f:
        all_cands = list(csv.DictReader(f))
    n_total = len(all_cands)
    top = all_cands[:top_n]
    log(f"  CSV has {n_total} candidates; rendering top {len(top)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Figure out which candidates need rendering (PNGs missing)
    pending = []
    for cand in top:
        key = safe_pid(cand["primary_id"])
        if key in rendered_keys: continue
        if all((OUT_DIR / f"polished_{key}_{p}.png").exists() for p in PANELS):
            rendered_keys.add(key)
            continue
        pending.append((key, cand))
    log(f"  {len(pending)} new PNGs to render ({len(rendered_keys)} already on disk)")

    if pending:
        # Build per-(survey, band, tile) request list
        by_tile = defaultdict(list)
        for key, cand in pending:
            try:
                ra = float(cand["sn_ra"]); dec = float(cand["sn_dec"])
            except (ValueError, TypeError):
                continue
            tiles = {"hst": cand["hst_tile"] or None,
                     "jwst": cand["jwst_tile"] or None,
                     "euclid": cand["euclid_tile"] or None}
            for (survey, band, R_band) in BAND_REQS:
                t = tiles[survey]
                if not t or t in ("", "nan", "None"): continue
                by_tile[(survey, band, R_band, t)].append((key, ra, dec))
        jobs = [(s, b, rb, t, lst) for (s, b, rb, t), lst in by_tile.items()]
        log(f"  IO plan: {len(jobs)} (survey,band,tile) groups, {N_WORKERS} workers")

        all_cuts = {}
        if jobs:
            t_io = time.time()
            with Pool(processes=min(N_WORKERS, max(1, len(jobs)))) as pool:
                for result in pool.imap_unordered(worker_extract, jobs):
                    all_cuts.update(result)
            log(f"  IO done in {time.time()-t_io:.1f}s")

        t_ren = time.time()
        for key, cand in pending:
            try:
                render_candidate_pngs(key, cand, all_cuts, TMP_DIR, OUT_DIR)
                rendered_keys.add(key)
            except Exception as e:
                log(f"  render failed {key}: {e}")
        log(f"  render done in {time.time()-t_ren:.1f}s")

    # Rebuild HTML
    html = build_html(top, top_n, n_total, run_status_line())
    (OUT_DIR / "index.html").write_text(html)
    log(f"  wrote {OUT_DIR / 'index.html'}")
    return rendered_keys


def main():
    once = "--once" in sys.argv
    top_n = DEFAULT_TOP
    if "--top" in sys.argv:
        i = sys.argv.index("--top")
        if i + 1 < len(sys.argv):
            try: top_n = int(sys.argv[i+1])
            except ValueError: pass

    log(f"v03 polished gallery — top {top_n}, refresh every {REFRESH}s")
    rendered_keys = set()
    # Seed rendered_keys from disk to survive restarts
    for png in OUT_DIR.glob("polished_*_hst.png"):
        m = re.match(r"polished_(.+)_hst\.png$", png.name)
        if m: rendered_keys.add(m.group(1))
    log(f"  found {len(rendered_keys)} previously-rendered candidates on disk")
    while True:
        try:
            t0 = time.time()
            rendered_keys = one_pass(top_n, rendered_keys)
            log(f"pass done in {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"PASS ERROR: {e}")
        if once: return
        time.sleep(REFRESH)


if __name__ == "__main__":
    main()
