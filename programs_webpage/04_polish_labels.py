"""Polished SN-centered PNG generator for the two test sources.

Drives off sn_test_2sources.csv (id, host_ra/dec, sn_ra/dec, sn_dx_pil,
sn_dy_pil, per-band mags).  For each source picks the appropriate cutout
set (the SN-recentered known34c_0001 for id=318858; the original
known34_0024 for id=320233, whose master-catalog position is already
within 1 PIL px of the SN), repaints labels + draws crosshair, saves to
the _crosshair_test/ directory.

Per panel:
  upper-left line 1: ID=<id>
  upper-left line 2: survey + filter description
  lower-left:        magnitude (SN mag on JWST panels, host mag on HST+Euclid)
  lower-right:       z=<lephare zfinal>
  crosshair: solid white on JWST (detection), dotted white on HST+Euclid
             (non-detection); 4-case partial L pointing away from the host
"""
import csv
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from astropy.table import Table

# ============================================================
HTMLS_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls")
JWST_PNG   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")
HST_PNG    = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png")
EUCLID_PNG = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png")

OUT_DIR = HTMLS_DIR / "_crosshair_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEPHARE_FITS = Path("/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1_lephare.fits")
MASTER_FITS  = Path("/Volumes/exdisk1/data/catalog/COSMOSWeb_mastercatalog_v1.1.fits")

# Crosshair / label parameters (same as test_sn_crosshair.py)
ARM_INNER = 0.020
ARM_OUTER = 0.075
SEARCH_FRAC = 0.15
LABEL_FONT_PT  = 22
LABEL_PAD_PX   = 28
CROSSHAIR_COLOR = (255, 255, 255)

DETECTION_PANELS = {"jwst1", "jwst2"}

PANELS = ["hst", "euvis", "eunisp", "jwst1", "jwst2"]

# Per panel:
#   survey: short telescope label
#   filter_str: filter combo string used in the upper-left header
#   bands: ordered list of bands used by this panel (bluest first for color)
#   single_band: True if grayscale (label uses just one filter+mag)
PANEL_SPEC = {
    "hst":    dict(survey="HST",    filter_str="F814W",          bands=["F814W"],                       single_band=True),
    "euvis":  dict(survey="Euclid", filter_str="VIS",            bands=["VIS"],                          single_band=True),
    "eunisp": dict(survey="Euclid", filter_str="Y/J/H",          bands=["Y", "J", "H"],                  single_band=False),
    "jwst1":  dict(survey="JWST",   filter_str="F115/F150/F277", bands=["F115W", "F150W", "F277W"],      single_band=False),
    "jwst2":  dict(survey="JWST",   filter_str="F150/F277/F444", bands=["F150W", "F277W", "F444W"],      single_band=False),
}


# ============================================================
# Drawing helpers (mostly copied from test_sn_crosshair.py)

def _load_font(size):
    for path in [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _draw_segment(d, p0, p1, color, width, style="solid",
                  dot_len=3, gap_len=3, dash_len=8, dash_gap=5):
    if style == "solid":
        d.line([p0, p1], fill=color, width=width); return
    seg, gap = (dot_len, gap_len) if style == "dotted" else (dash_len, dash_gap)
    x0, y0 = p0; x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = (dx*dx + dy*dy) ** 0.5
    if length < 1: return
    ux, uy = dx/length, dy/length
    step = seg + gap
    n = int(length // step) + 1
    for i in range(n):
        s = i * step
        e = min(s + seg, length)
        if e <= s: break
        d.line([(x0 + ux*s, y0 + uy*s), (x0 + ux*e, y0 + uy*e)],
               fill=color, width=width)


def quadrant_arms(dx, dy):
    """SN-host vector (in PIL coords): arms point AWAY from host (NE/NW/SW/SE)."""
    dy_math = -dy
    if   dx >= 0 and dy_math >= 0: return ["right", "up"]
    elif dx <  0 and dy_math >= 0: return ["left",  "up"]
    elif dx <= 0 and dy_math <= 0: return ["left",  "down"]
    else:                          return ["right", "down"]


def overlay_crosshair(img_rgb, sn_xy, arms, color=CROSSHAIR_COLOR, style="solid"):
    h, w = img_rgb.shape[:2]
    sx, sy = sn_xy
    # +7 PNG px extra gap (user request 2026-05-26 clarified):
    # 3 HST native pixels = 3 × 0.030" = 0.09" of sky.  On the 480-px PNG
    # of a 6" field that's 0.09 / (6/480) ≈ 7 PNG pixels — same across
    # all 5 panel types since they all share the 480×480 / 6" rendering.
    inner = max(2, int(ARM_INNER * min(h, w))) + 7
    outer = max(inner + 4, int(ARM_OUTER * min(h, w)) + 7)
    thickness = max(1, min(h, w) // 240)
    pil = Image.fromarray(img_rgb).convert("RGB")
    d = ImageDraw.Draw(pil)
    rgb = (int(color[0]), int(color[1]), int(color[2]))
    if "right" in arms:
        _draw_segment(d, (sx + inner, sy), (sx + outer, sy), rgb, thickness, style)
    if "left" in arms:
        _draw_segment(d, (sx - inner, sy), (sx - outer, sy), rgb, thickness, style)
    if "up" in arms:
        _draw_segment(d, (sx, sy - inner), (sx, sy - outer), rgb, thickness, style)
    if "down" in arms:
        _draw_segment(d, (sx, sy + inner), (sx, sy + outer), rgb, thickness, style)
    return np.asarray(pil)


def _fmt_mag(label, val):
    """`label`=val if val is a valid positive float, else `label`=?"""
    if val is None or not np.isfinite(val) or val <= 0:
        return f"{label}=?"
    return f"{label}={val:.2f}"


def repaint_labels(img_rgb, source_id, z, header, host_mag_label, host_mag,
                   sn_bands, sn_mags, show_sn_line=True,
                   best_snr=None, best_band=None,
                   font_pt=LABEL_FONT_PT, color=(255, 255, 255)):
    """New 3-line corner layout (no lower-right).

    Upper-left line 1:  ID=<source_id>    z=<z>           (4 spaces between)
    Upper-left line 2:  <header> <bluest_band>=<host_mag>
    Lower-left line:    SN <b1>=<m1> <b2>=<m2> ...        (only if show_sn_line)

    `show_sn_line=False` is used on non-detection panels (HST + Euclid) so
    those panels stay clean and only the upper-left labels carry the host
    info.
    """
    pil = Image.fromarray(img_rgb).convert("RGB")
    d = ImageDraw.Draw(pil)
    font = _load_font(font_pt)
    h, w = img_rgb.shape[:2]
    line_step = int(font_pt * 1.25)
    # NOTE: no wash rectangles -- cutouts are now label-free so we stamp the
    # text directly onto the full-frame image.

    def _stamp(x, y, txt, anchor="lt"):
        try:
            d.text((x, y), txt, font=font, fill=color, anchor=anchor)
        except TypeError:
            d.text((x, y), txt, font=font, fill=color)

    # Line 1: ID    z=...    SN=Xσ
    z_str = f"z={z:.3f}" if (z is not None and np.isfinite(z)) else "z=?"
    line1 = f"ID={source_id}     {z_str}"
    if (best_snr is not None and np.isfinite(best_snr) and best_snr > 0
            and best_band):
        line1 += f"       SN={best_snr:.1f}σ ({best_band})"
    _stamp(LABEL_PAD_PX, LABEL_PAD_PX, line1, "lt")
    # Line 2: <header> [<bluest band>=<host mag>]
    # If the host mag is unavailable (None / NaN / <= 0), drop the suffix.
    if host_mag is not None and np.isfinite(host_mag) and host_mag > 0:
        line2 = f"{header} {host_mag_label}={host_mag:.2f}"
    else:
        line2 = header
    _stamp(LABEL_PAD_PX, LABEL_PAD_PX + line_step, line2, "lt")
    if show_sn_line:
        parts = ["SN"] + [_fmt_mag(b, m) for b, m in zip(sn_bands, sn_mags)]
        sn_line = " ".join(parts)
        _stamp(LABEL_PAD_PX, h - LABEL_PAD_PX, sn_line, "ls")

    # 1" scale bar in lower-right corner.
    # PNG pixel scale = 6 arcsec / 480 px = 12.5 mas/px → 1" = 80 px.
    bar_px = int(round(1.0 / (6.0 / w)))   # works for any cutout/PNG ratio
    bar_thickness = max(2, int(font_pt * 0.18))
    bar_x_right = w - LABEL_PAD_PX
    bar_y       = h - LABEL_PAD_PX
    bar_x_left  = bar_x_right - bar_px
    d.rectangle([(bar_x_left, bar_y - bar_thickness),
                 (bar_x_right, bar_y)], fill=color)
    # Label "1\"" just above the bar, right-aligned
    try:
        d.text((bar_x_right, bar_y - bar_thickness - 4),
               '1"', font=font, fill=color, anchor="rs")
    except TypeError:
        d.text((bar_x_right - int(font_pt * 0.7),
                bar_y - bar_thickness - font_pt - 4),
               '1"', font=font, fill=color)
    return np.asarray(pil)


# ============================================================
# Per-source helpers

# Catalog caches — loaded lazily on first access, re-used across sources.
# Without this, each source caused 2 reads of a ~hundreds-of-MB FITS.
_MASTER_TABLE = None
_LEPHARE_TABLE = None


def _master_table():
    global _MASTER_TABLE
    if _MASTER_TABLE is None:
        _MASTER_TABLE = Table.read(MASTER_FITS)
    return _MASTER_TABLE


def _lephare_table():
    global _LEPHARE_TABLE
    if _LEPHARE_TABLE is None:
        _LEPHARE_TABLE = Table.read(LEPHARE_FITS)
    return _LEPHARE_TABLE


def host_mags_for_id(cid):
    """Return dict of host mags from the master catalog: F814W, VIS?, Y?, F115W, F150W."""
    t = _master_table()
    idx = (t["id"] == cid).nonzero()[0][0]
    row = t[idx]
    out = dict()
    out["F814W"] = float(row["mag_auto_hst-f814w"])
    out["F115W"] = float(row["mag_auto_f115w"])
    out["F150W"] = float(row["mag_auto_f150w"])
    out["F277W"] = float(row["mag_auto_f277w"])
    out["F444W"] = float(row["mag_auto_f444w"])
    # Euclid not in this catalog -> NaN, will render as "?"
    out["VIS"] = np.nan
    out["Y"]   = np.nan
    return out


def lephare_z_for_id(cid):
    """Look up zfinal for `cid` in the lephare catalog."""
    master  = _master_table()
    lephare = _lephare_table()
    idx = (master["id"] == cid).nonzero()[0][0]
    return float(lephare["zfinal"][idx])


def resize_to(img, w, h):
    if img.ndim == 2:
        pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(pil.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def load_color(path):
    return np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32) / 255.0


def load_gray(path):
    return np.asarray(Image.open(str(path)).convert("L"), dtype=np.float32) / 255.0


# ============================================================
def process_source(cid, row, list_name, seq, out_dir):
    print(f"\n==== ID={cid} ====")
    tag = f"{list_name}_{seq:04d}"

    # Locate the 5 cutout files
    files = {
        "hst":    HST_PNG    / f"sn_{tag}_hst.png",
        "euvis":  EUCLID_PNG / f"sn_{tag}_euclid_vis.png",
        "eunisp": EUCLID_PNG / f"sn_{tag}_euclid_nisp.png",
        "jwst1":  JWST_PNG   / f"sn_{tag}_jwst1.png",
        "jwst2":  JWST_PNG   / f"sn_{tag}_jwst2.png",
    }
    for k, p in files.items():
        if not p.exists():
            sys.exit(f"missing {k}: {p}")

    # Mags
    sn_mags   = {b: row.get(f"mag_{b}", -1.0) for b in
                 ("F814W","VIS","Y","J","H","F115W","F150W","F277W","F444W")}

    # Telescope from CSV.  Detection panel(s) per source type:
    #   JWST           -> jwst1 + jwst2 (both NIRCam combos are detection)
    #   HST            -> hst
    #   EUCLID / NISP  -> eunisp (Euclid NISP-only SN: VIS at different epoch
    #                              shows host only, NISP shows SN excess)
    #   EUCLID-VIS     -> euvis  (Euclid VIS-only SN — for future use)
    # Default JWST for back-compat.
    telescope = str(row.get("telescope", "JWST")).strip().upper()
    if telescope == "HST":
        det_panels_for_this_source = {"hst"}
    elif telescope in ("EUCLID", "NISP", "EUCLID-NISP"):
        det_panels_for_this_source = {"eunisp"}
    elif telescope in ("EUCLID-VIS", "VIS"):
        det_panels_for_this_source = {"euvis"}
    else:
        det_panels_for_this_source = {"jwst1", "jwst2"}

    # Best-band sigma comes directly from the CSV (computed by measure_sn_5.py).
    best_snr = row.get("best_snr", -1.0)
    best_band = row.get("best_band", "")
    try: best_snr = float(best_snr)
    except (ValueError, TypeError): best_snr = -1.0
    if best_snr <= 0 or not best_band:
        best_snr, best_band = None, None

    host_mags = host_mags_for_id(cid)
    z = lephare_z_for_id(cid)
    print(f"  z (lephare zfinal) = {z:.4f}  telescope={telescope}  "
          f"detection panels = {det_panels_for_this_source}")

    # Arm quadrant: in the recentered cutout the host is opposite the
    # original SN offset.  Use the stored sn_dx_pil / sn_dy_pil from the CSV.
    dx_pil = int(row.get("sn_dx_pil", 0))
    dy_pil = int(row.get("sn_dy_pil", 0))
    if abs(dx_pil) < 2 and abs(dy_pil) < 2:
        # SN essentially at center -- default to NE
        arms = ["right", "up"]
    else:
        arms = quadrant_arms(dx_pil, dy_pil)
    print(f"  SN-host PIL dx={dx_pil:+d} dy={dy_pil:+d}  arms={arms}")

    # Load images
    hst_g   = load_gray(files["hst"])
    euvis_g = load_gray(files["euvis"])
    jw1     = load_color(files["jwst1"])
    jw2     = load_color(files["jwst2"])
    eunisp  = load_color(files["eunisp"])
    h, w = jw1.shape[:2]
    cx, cy = w // 2, h // 2

    panels = {
        "jwst1":  (jw1 * 255).clip(0, 255).astype(np.uint8),
        "jwst2":  (jw2 * 255).clip(0, 255).astype(np.uint8),
        "hst":    np.stack([(resize_to(hst_g,   w, h) * 255).astype(np.uint8)] * 3, axis=-1),
        "euvis":  np.stack([(resize_to(euvis_g, w, h) * 255).astype(np.uint8)] * 3, axis=-1),
        "eunisp": (resize_to(eunisp, w, h) * 255).clip(0, 255).astype(np.uint8),
    }

    for name in PANELS:
        img = panels[name]
        spec = PANEL_SPEC[name]
        is_detection = (name in det_panels_for_this_source)
        style = "solid" if is_detection else "dotted"
        out = overlay_crosshair(img, (cx, cy), arms, style=style)

        # Header for upper-left line 2: "HST F814W" / "JWST F115/F150/F277" / etc.
        header = f"{spec['survey']} {spec['filter_str']}"
        # Host mag = bluest band's host mag (band[0]).  For grayscale (single
        # band), bluest == only band.
        bluest = spec["bands"][0]
        host_m = host_mags.get(bluest, np.nan)
        # SN mags: one per band in this panel
        sn_band_list = spec["bands"]
        sn_mag_list  = [sn_mags.get(b, -1) for b in sn_band_list]

        # SN=X.Xσ label only on detection (JWST) panels
        out = repaint_labels(out, cid, z, header, bluest, host_m,
                             sn_band_list, sn_mag_list,
                             show_sn_line=is_detection,
                             best_snr=best_snr if is_detection else None,
                             best_band=best_band if is_detection else None)

        out_path = out_dir / f"polished_{cid}_{name}.png"
        Image.fromarray(out).save(str(out_path))
        kind = "SOLID" if is_detection else "DOTTED"
        print(f"  {out_path.name}  [{kind}, host {bluest}="
              f"{host_m if not np.isfinite(host_m) else f'{host_m:.2f}'}; "
              f"SN " + " ".join(f"{b}={m if m is None or m<0 else m:.2f}"
                                 for b, m in zip(sn_band_list, sn_mag_list)) + "]")


def _maybe_float(v):
    """Coerce to float when possible (so '-1' becomes -1.0)."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="CSV with id, ra, dec, mag_<band>, snr_<band> columns")
    ap.add_argument("--list", required=True,
                    help="Cutout list name (PNG prefix 'sn_<list>_<seq:04d>_*')")
    ap.add_argument("--out", default=str(OUT_DIR),
                    help=f"Output directory (default {OUT_DIR})")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.csv) as fh:
        rdr = csv.DictReader(fh)
        rows = [{k: _maybe_float(v) for k, v in r.items()} for r in rdr]
    print(f"{len(rows)} sources from {args.csv}", flush=True)
    for i, row in enumerate(rows, 1):
        cid = int(row["id"])
        # Use row index as seq (matches the 1-based per-list seq the cutout
        # scripts assign).  Ignore any 'seq' column in the CSV.
        process_source(cid, row, args.list, i, out_dir)


if __name__ == "__main__":
    main()
