"""Top-N v01 SN candidates → v33-style polished webpage.

Reuses the existing render recipes (recut_single_source.render_*) and
crosshair/label code (04_polish_labels.overlay_crosshair, repaint_labels)
to produce 5 PNGs per candidate (hst, euvis, eunisp, jwst1, jwst2) in
the same layout as htmls/_crosshair_test/v33/.

Candidate IDs in v01 are mixed (EuclidID_*, HSTID_*, numeric), so we do
NOT look up host_mags in the COSMOSWeb master catalog (it only knows
numeric COSMOSWeb IDs). Instead we use the per-band mags from
tbl_sn_candidates_v01.csv directly. host_z is set to NaN (renders as 'z=?').

Output: htmls/sn_search/v01/polished/
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv
from pathlib import Path
import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R
from recut_single_source import render_gray, render_rgb_jwst, render_rgb_nisp
from importlib import import_module
polish = import_module("04_polish_labels")

CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v01.csv")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v01/polished")
TMP_DIR  = OUT_DIR / "_raw"   # raw PNGs before crosshair+label overlay
TOP_N    = 5
CUTOUT_SIZE = 6.0 * u.arcsec

PANELS = ["jwst1", "jwst2", "hst", "euvis", "eunisp"]
PANEL_LABEL = {
    "jwst1":  "JWST F115/F150/F277 (blue)",
    "jwst2":  "JWST F150/F277/F444 (red)",
    "hst":    "HST F814W",
    "euvis":  "Euclid VIS",
    "eunisp": "Euclid NISP YJH",
}


def cut(path, pos):
    """6\" cutout; returns Cutout2D-like object with .data (float32)."""
    with fits.open(path, memmap=True) as h:
        sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        wcs = WCS(sci_hdu.header)
        data = sci_hdu.data
        c = Cutout2D(data.astype(np.float32), pos, size=CUTOUT_SIZE, wcs=wcs,
                     mode="partial", fill_value=0.0)
    return c


def render_raw_pngs(cand, seq, tmp_dir):
    """Cut FITS for this candidate and render the 5 raw PNGs (no crosshair/label
    overlay yet). Returns dict of {panel: Path} for the raw PNG paths."""
    cid = cand["id"]
    ra = float(cand["sn_ra"]); dec = float(cand["sn_dec"])
    hst_tile = cand["hst_tile"] or None
    jwst_tile = cand["jwst_tile"] or None
    eu_tile = cand["euclid_tile"] or None
    pos = SkyCoord(ra*u.deg, dec*u.deg)

    out = {}
    blank_label = []  # we add labels in the polish step

    # HST F814W
    hst_png = tmp_dir / f"sn_v01cand_{seq:04d}_hst.png"
    out["hst"] = hst_png
    if hst_tile:
        try:
            p = f"{R.HST_DIR}/acs_I_030mas_{hst_tile}_sci.fits"
            c = cut(p, pos)
            render_gray(c.data, hst_png, blank_label)
        except Exception as e:
            print(f"  HST cut failed for {cid}: {e}", flush=True)
            _write_black(hst_png)
    else:
        _write_black(hst_png)

    # Euclid VIS
    vis_png = tmp_dir / f"sn_v01cand_{seq:04d}_euclid_vis.png"
    out["euvis"] = vis_png
    if eu_tile:
        try:
            p = R.resolve_euclid_path(eu_tile, "VIS")
            c = cut(p, pos)
            render_gray(c.data, vis_png, blank_label)
        except Exception as e:
            print(f"  VIS cut failed for {cid}: {e}", flush=True)
            _write_black(vis_png)
    else:
        _write_black(vis_png)

    # Euclid NISP YJH
    nisp_png = tmp_dir / f"sn_v01cand_{seq:04d}_euclid_nisp.png"
    out["eunisp"] = nisp_png
    if eu_tile:
        try:
            cy = cut(R.resolve_euclid_path(eu_tile, "NIR-Y"), pos)
            cj = cut(R.resolve_euclid_path(eu_tile, "NIR-J"), pos)
            ch = cut(R.resolve_euclid_path(eu_tile, "NIR-H"), pos)
            render_rgb_nisp(cy.data, cj.data, ch.data, nisp_png, blank_label)
        except Exception as e:
            print(f"  NISP cut failed for {cid}: {e}", flush=True)
            _write_black(nisp_png)
    else:
        _write_black(nisp_png)

    # JWST jwst1 (F115/F150/F277) and jwst2 (F150/F277/F444)
    jw1_png = tmp_dir / f"sn_v01cand_{seq:04d}_jwst1.png"
    jw2_png = tmp_dir / f"sn_v01cand_{seq:04d}_jwst2.png"
    out["jwst1"] = jw1_png; out["jwst2"] = jw2_png
    if jwst_tile:
        try:
            c115 = cut(R.resolve_jwst_path(jwst_tile, "f115w"), pos)
            c150 = cut(R.resolve_jwst_path(jwst_tile, "f150w"), pos)
            c277 = cut(R.resolve_jwst_path(jwst_tile, "f277w"), pos)
            c444 = cut(R.resolve_jwst_path(jwst_tile, "f444w"), pos)
            render_rgb_jwst(c115.data, c150.data, c277.data, jw1_png, blank_label)
            render_rgb_jwst(c150.data, c277.data, c444.data, jw2_png, blank_label)
        except Exception as e:
            print(f"  JWST cut failed for {cid}: {e}", flush=True)
            _write_black(jw1_png); _write_black(jw2_png)
    else:
        _write_black(jw1_png); _write_black(jw2_png)

    return out


def _write_black(path):
    Image.new("RGB", (480, 480), (10, 10, 10)).save(str(path))


def _load_color(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0

def _load_gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def polish_candidate(cand, seq, raw_paths, out_dir):
    """Apply crosshair + labels to the 5 raw PNGs and write polished_*.png."""
    cid = cand["id"]
    # Telescope → which panel(s) are 'detection' (solid crosshair); others dotted
    telescope = cand["telescope"].strip().upper()
    if telescope == "HST":         det_panels = {"hst"}
    elif telescope.startswith("EUCLID-NISP") or telescope == "NISP":
        det_panels = {"eunisp"}
    elif telescope.startswith("EUCLID-VIS") or telescope == "VIS":
        det_panels = {"euvis"}
    else: det_panels = {"jwst1", "jwst2"}

    # SN is at center of cutout (we recentered on sn_ra/sn_dec at cut time);
    # arms point NE by default (we don't have a separate host position).
    arms = ["right", "up"]

    # Per-panel images
    jw1 = _load_color(raw_paths["jwst1"])
    jw2 = _load_color(raw_paths["jwst2"])
    nisp_rgb = _load_color(raw_paths["eunisp"])
    hst_g = _load_gray(raw_paths["hst"])
    vis_g = _load_gray(raw_paths["euvis"])
    h, w = jw1.shape[:2]
    cx, cy = w // 2, h // 2

    panels_rgb = {
        "jwst1":  (jw1 * 255).clip(0, 255).astype(np.uint8),
        "jwst2":  (jw2 * 255).clip(0, 255).astype(np.uint8),
        "hst":    np.stack([(polish.resize_to(hst_g, w, h) * 255).astype(np.uint8)] * 3, axis=-1),
        "euvis":  np.stack([(polish.resize_to(vis_g, w, h) * 255).astype(np.uint8)] * 3, axis=-1),
        "eunisp": (polish.resize_to(nisp_rgb, w, h) * 255).clip(0, 255).astype(np.uint8),
    }

    # mags + snr from CSV (NOT master catalog — IDs may not be COSMOSWeb)
    def gm(k):
        try:
            v = float(cand[k])
            return v if (np.isfinite(v) and v > 0) else float("nan")
        except (ValueError, TypeError, KeyError):
            return float("nan")
    sn_mags = {b: gm(f"mag_{b}") for b in ("F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W")}
    try:
        best_snr = float(cand["best_snr"])
        if best_snr <= 0: best_snr = None
    except (ValueError, TypeError): best_snr = None
    best_band = cand["best_band"] or None
    z = float("nan")  # we don't have z for arbitrary primary_ids

    for name in PANELS:
        img = panels_rgb[name]
        spec = polish.PANEL_SPEC[name]
        is_det = name in det_panels
        style = "solid" if is_det else "dotted"
        out = polish.overlay_crosshair(img, (cx, cy), arms, style=style)
        header = f"{spec['survey']} {spec['filter_str']}"
        bluest = spec["bands"][0]
        host_m = sn_mags.get(bluest, float("nan"))
        sn_band_list = spec["bands"]
        sn_mag_list  = [sn_mags.get(b, -1.0) for b in sn_band_list]
        out = polish.repaint_labels(
            out, cid, z, header, bluest, host_m,
            sn_band_list, sn_mag_list,
            show_sn_line=is_det,
            best_snr=best_snr if is_det else None,
            best_band=best_band if is_det else None,
            show_scale_bar=(name == "hst"),
        )
        Image.fromarray(out).save(str(out_dir / f"polished_{cid}_{name}.png"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    with CAND_CSV.open() as f:
        cands = list(csv.DictReader(f))
    top = cands[:TOP_N]
    print(f"Top {len(top)} candidates from {CAND_CSV.name}")

    for i, cand in enumerate(top, start=1):
        cid = cand["id"]
        print(f"\n==== {cid} ({i}/{len(top)}) ====")
        raw = render_raw_pngs(cand, i, TMP_DIR)
        polish_candidate(cand, i, raw, OUT_DIR)
        print(f"  wrote polished_{cid}_{{hst,euvis,eunisp,jwst1,jwst2}}.png")

    # Build v33-style index.html
    import time
    html = ["<!doctype html><html><head>",
            "<title>v01 top SN candidates — v33-style polished</title>",
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
            "<div class='bug'><b>v01 had a labeling bug:</b> all 17 known SNe "
            "were marked positive in every survey, so the CNN learned host "
            "morphology. Top candidates are likely host galaxies, not real "
            "supernovae. v02 with corrected per-survey labels is pending.</div>"]
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
    print(f"\nWrote {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
