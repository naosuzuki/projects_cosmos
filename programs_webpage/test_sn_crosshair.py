"""Single-target test of the SN-crosshair detector.

Workflow:
  1. Load the 5 cutout PNGs for one ID (HST F814W, Euclid VIS, Euclid NISP,
     JWST F115/F150/F277, JWST F150/F277/F444).
  2. Detect bright point sources in each (HST + Euclid VIS as grayscale,
     JWST color images via the BLUE channel since SN are blue-dominated
     when F115W/F150W is in the B slot).
  3. The SN is a source that appears in BOTH JWST images but is absent
     from HST and Euclid VIS (the SN exploded between HST/Euclid epoch
     and the JWST epoch).
  4. Compute (dx, dy) = SN_pos - cutout_center (the cutout is centered on
     the host galaxy = master-catalog id position).
  5. Draw the 4-case partial crosshair (see hsc_plot.draw_crosshair):
     only 2 of the 4 arms are drawn, pointing AWAY from the host so the
     host is never obscured.

Output: same 5 cutouts but with the crosshair overlaid, saved under
    /Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/

Usage:
    python test_sn_crosshair.py
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-panel info for the lower-left label.
# For color composites we use the BLUEST band's mag (the SN drops out of bluer
# light fastest for low-z, and stays brightest at high-z).
#   jwst1 (F115/F150/F277) -> bluest = F115W
#   jwst2 (F150/F277/F444) -> bluest = F150W
#   eunisp (Y/J/H)         -> bluest = Y
# For each panel:
#   header     -> upper-left line 2 (under "ID=...")
#   mag_label  -> lower-left, with magnitude
#   For color composites, the lower-left mag is the BLUEST band of the trio
#   (F115W for jwst1, F150W for jwst2, Y for eunisp).
PANEL_LABELS = {
    "hst":    dict(header="HST F814W",            mag_label="F814W", mag=22.26),
    "euvis":  dict(header="Euclid VIS",           mag_label="VIS",   mag=22.55),
    "eunisp": dict(header="Euclid Y/J/H",         mag_label="Y",     mag=22.54),
    "jwst1":  dict(header="JWST F115/F150/F277",  mag_label="F115W", mag=21.83),
    "jwst2":  dict(header="JWST F150/F277/F444",  mag_label="F150W", mag=21.64),
}

# Source ID (one for the whole row; appears as upper-left line 1)
SOURCE_ID = "318858"

# Photo-z stamped on every panel.  Source = lephare zfinal from
# COSMOSWeb_mastercatalog_v1.1_lephare.fits (the COSMOS2025 / Shuntov+25
# LePHARE photo-z catalog built on the SE++ 34-band model photometry).
# In production this is looked up per source from the lephare catalog.
LEPHARE_Z = 0.5763
PANEL_Z = {
    "hst":    LEPHARE_Z,
    "euvis":  LEPHARE_Z,
    "eunisp": LEPHARE_Z,
    "jwst1":  LEPHARE_Z,
    "jwst2":  LEPHARE_Z,
}

# Which panels show the SN ("detection"); detection panels get a SOLID
# crosshair (the SN itself is the proof), non-detection panels get a DASHED
# crosshair (clearly marks the position when there's nothing there).
DETECTION_PANELS = {"jwst1", "jwst2"}

CROSSHAIR_COLOR = (255, 255, 255)   # white

# Corner-label style.  Font size matches the original baked-in matplotlib
# labels (~22 px tall); padding is ~6% of image side from each edge so the
# labels have comfortable breathing room (was 14 = 3%).
LABEL_FONT_PT  = 22
LABEL_PAD_PX   = 28

# ============================================================
TARGET_SEQ = 1    # ID=318858 in known34c (SN-centered), seq 0001
LIST_NAME  = "known34c"

# When the cutout has been recentered on the SN, the SN sits at PNG center
# with sub-pixel residual.  Re-detecting it would land on noisy coordinates
# and give the wrong arm quadrant.  Instead, use the ORIGINAL SN-to-host
# offset (stored as audit columns in the recentered FITS): the host is now
# at -offset from the new center, so the arms still point in +offset direction
# (the same NE quadrant as the original).
RECENTERED = True                # True if the cutout is SN-centered
ORIG_SN_DX_PIL = 5               # SN's offset in the OLD host-centered PNG
ORIG_SN_DY_PIL = -62

HTMLS_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls")
JWST_PNG   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")
HST_PNG    = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png")
EUCLID_PNG = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png")

OUT_DIR = HTMLS_DIR / "_crosshair_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Crosshair geometry (in image pixels)
ARM_INNER = 0.020   # gap from SN center (frac of image side) — small so the L hugs the SN
ARM_OUTER = 0.075   # arm length (frac of image side) — matches hsc_plot proportion
MATCH_RADIUS_FRAC = 0.04   # max separation to call a "match" between images (frac of image side)
SEARCH_FRAC = 0.15         # only look for SN inside central 30% box (frac from center to edge)

# Mask the label region (upper-left corner where "ID=..." text sits)
# Empirically the text sits within roughly the top 18% × left 38% of the image.
LABEL_MASK_FRAC_Y = 0.20
LABEL_MASK_FRAC_X = 0.40

# ============================================================


def load_color(path):
    """Load a PNG as RGB float [0,1] (H, W, 3)."""
    img = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32) / 255.0
    return img


def load_gray(path):
    img = np.asarray(Image.open(str(path)).convert("L"), dtype=np.float32) / 255.0
    return img


def resize_to(img, w, h):
    """Resize 2-D or 3-D image to (w, h) using PIL high-quality downsampling."""
    if img.ndim == 2:
        pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8), mode="L")
        out = np.asarray(pil.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    else:
        pil = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8), mode="RGB")
        out = np.asarray(pil.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    return out


def mask_label_region(img2d):
    """Zero out the upper-left label region so it doesn't trigger as a peak."""
    out = img2d.copy()
    h, w = out.shape
    y_to = int(LABEL_MASK_FRAC_Y * h)
    x_to = int(LABEL_MASK_FRAC_X * w)
    out[:y_to, :x_to] = 0.0
    return out


def detect_peaks(img2d, n_top=15, min_sigma=2.5, footprint=5):
    """Detect bright local-maxima above min_sigma*MAD in img2d.

    Returns list of (x, y, peak_value) sorted by brightness (desc).
    The upper-left label region is masked out before peak-finding.
    """
    img2d = mask_label_region(img2d)
    # Background-subtract by median, scale by MAD-derived sigma
    med = float(np.median(img2d))
    mad = float(np.median(np.abs(img2d - med)))
    sigma = 1.4826 * mad if mad > 0 else 1e-6
    thresh = med + min_sigma * sigma

    # Local max via maximum filter
    dilated = maximum_filter(img2d, size=footprint)
    is_peak = (img2d == dilated) & (img2d > thresh)
    ys, xs = np.where(is_peak)
    vals = img2d[ys, xs]
    order = np.argsort(vals)[::-1][:n_top]
    return [(int(xs[i]), int(ys[i]), float(vals[i])) for i in order]


def match(p, peaks, radius):
    """Return True if any peak in `peaks` is within `radius` of p (x,y)."""
    px, py = p[0], p[1]
    for (x, y, _) in peaks:
        if (x - px)**2 + (y - py)**2 <= radius**2:
            return True
    return False


def find_sn_position(hst_g, eu_vis_g, jw1_rgb, jw2_rgb):
    """Find the SN: a blue-anomaly peak in both JWST images NOT in HST/EuclidVIS.

    "Blue anomaly" = B - 0.5*(R+G).  This isolates pixels that are
    bluer than the host's warmer disk light, which is the SN signature
    (F115W/F150W dominated).

    Returns (x, y) in cutout pixel coords, or None.
    """
    h, w = jw1_rgb.shape[:2]

    def blue_anomaly(rgb):
        b, g, r = rgb[..., 2], rgb[..., 1], rgb[..., 0]
        return b - 0.5 * (r + g)

    jw1_b = blue_anomaly(jw1_rgb)
    jw2_b = blue_anomaly(jw2_rgb)

    # Confine search to central box (SN is near the host)
    cx, cy = w // 2, h // 2
    half_box = int(SEARCH_FRAC * min(h, w))

    def in_search(p):
        return abs(p[0] - cx) <= half_box and abs(p[1] - cy) <= half_box

    jw1_peaks = [p for p in detect_peaks(jw1_b) if in_search(p)]
    jw2_peaks = [p for p in detect_peaks(jw2_b) if in_search(p)]

    # HST + Euclid VIS reference peaks (these define what was already there)
    # Resize to JWST resolution first (so pixel coords are comparable).
    hst_r   = resize_to(hst_g,    w, h)
    euvis_r = resize_to(eu_vis_g, w, h)
    hst_peaks   = detect_peaks(hst_r)
    euvis_peaks = detect_peaks(euvis_r)

    radius = int(MATCH_RADIUS_FRAC * min(h, w))

    # Candidate SNe: in jwst1 AND jwst2, NOT in HST AND NOT in EuclidVIS
    candidates = []
    for p1 in jw1_peaks:
        if not match(p1, jw2_peaks, radius):
            continue
        if match(p1, hst_peaks, radius):
            continue
        if match(p1, euvis_peaks, radius):
            continue
        candidates.append(p1)

    if not candidates:
        # Diagnostic dump
        print(f"  jwst1 peaks (in box): {jw1_peaks}")
        print(f"  jwst2 peaks (in box): {jw2_peaks}")
        print(f"  hst peaks: {hst_peaks[:5]}")
        print(f"  euclidvis peaks: {euvis_peaks[:5]}")
        return None

    # Pick the brightest SN candidate
    candidates.sort(key=lambda p: -p[2])
    best = candidates[0]
    print(f"  found {len(candidates)} SN candidate(s); picked brightest at ({best[0]},{best[1]}) val={best[2]:.3f}")
    return best[0], best[1]


def quadrant_arms(dx, dy):
    """Given SN offset (dx, dy) from cutout center in IMAGE coords
    (y grows DOWNWARD as in cv2), return which 2 arms to draw.

    To match the hsc_plot convention (matplotlib y grows UP), flip dy.

    Returns list of arm directions:
        'right' = +x,  'left' = -x,  'up' = -dy_image,  'down' = +dy_image
    """
    # hsc_plot logic with matplotlib y-up:
    #   1st (dx>=0, dy_math>=0):  +x, +y_math   -> 'right', 'up'
    #   2nd (dx<0,  dy_math>=0):  -x, +y_math   -> 'left',  'up'
    #   3rd (dx<=0, dy_math<=0):  -x, -y_math   -> 'left',  'down'
    #   4th (dx>0,  dy_math<0):   +x, -y_math   -> 'right', 'down'
    dy_math = -dy   # flip image-down to math-up
    if   dx >= 0 and dy_math >= 0: return ['right', 'up']
    elif dx <  0 and dy_math >= 0: return ['left',  'up']
    elif dx <= 0 and dy_math <= 0: return ['left',  'down']
    else:                          return ['right', 'down']


def _draw_segment(d, p0, p1, color, width, style="solid",
                  dot_len=3, gap_len=3, dash_len=8, dash_gap=5):
    """Draw a line from p0 to p1.

    style: 'solid' | 'dotted' (3 px dots, 3 px gap) | 'dashed' (8 px dash, 5 px gap)
    """
    if style == "solid":
        d.line([p0, p1], fill=color, width=width)
        return
    if style == "dotted":
        seg, gap = dot_len, gap_len
    else:  # dashed
        seg, gap = dash_len, dash_gap
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = (dx*dx + dy*dy) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    step = seg + gap
    n = int(length // step) + 1
    for i in range(n):
        s = i * step
        e = min(s + seg, length)
        if e <= s:
            break
        sa = (x0 + ux*s, y0 + uy*s)
        ea = (x0 + ux*e, y0 + uy*e)
        d.line([sa, ea], fill=color, width=width)


def overlay_crosshair(img_rgb, sn_xy, color=CROSSHAIR_COLOR, style="solid",
                      arms_override=None):
    """Return img_rgb (H, W, 3 uint8) with 4-case partial crosshair drawn at sn_xy.

    Two arms only — pointing AWAY from the host galaxy.
    `style` ∈ {'solid', 'dashed', 'dotted'}.
    `arms_override`: optional list (e.g. ['right','up']) to bypass the auto
    quadrant logic — needed when the cutout is recentered on the SN and the
    SN→host direction is known externally.
    """
    h, w = img_rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    sx, sy = sn_xy
    if arms_override is not None:
        arms = arms_override
    else:
        dx = sx - cx
        dy = sy - cy
        arms = quadrant_arms(dx, dy)

    inner = max(2, int(ARM_INNER * min(h, w)))
    outer = max(inner + 4, int(ARM_OUTER * min(h, w)))
    thickness = max(1, min(h, w) // 240)

    pil = Image.fromarray(img_rgb).convert("RGB")
    d = ImageDraw.Draw(pil)
    rgb = (int(color[0]), int(color[1]), int(color[2]))

    if 'right' in arms:
        _draw_segment(d, (sx + inner, sy), (sx + outer, sy), rgb, thickness, style)
    if 'left' in arms:
        _draw_segment(d, (sx - inner, sy), (sx - outer, sy), rgb, thickness, style)
    if 'up' in arms:
        _draw_segment(d, (sx, sy - inner), (sx, sy - outer), rgb, thickness, style)
    if 'down' in arms:
        _draw_segment(d, (sx, sy + inner), (sx, sy + outer), rgb, thickness, style)
    return np.asarray(pil)


def _load_font(size):
    """Try to load a real serif font; fall back to PIL default."""
    for path in [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def repaint_labels(img_rgb, source_id, header, mag_label, mag, z=None,
                   font_pt=LABEL_FONT_PT, color=(255, 255, 255)):
    """Repaint the 4 corner labels uniformly, hiding any baked-in labels.

    Layout (all same font size):
      upper-left  line 1: ID=<source_id>
      upper-left  line 2: <header>            (e.g. "JWST F115/F150/F277")
      lower-left:         <mag_label>=<mag>   (e.g. "F115W=21.83")
      lower-right:        z=<z>               (only if z provided)

    Paints a small black wash over the existing upper-left text first so the
    old label can't bleed through.  Image is assumed to be uint8 RGB.
    """
    pil = Image.fromarray(img_rgb).convert("RGB")
    d = ImageDraw.Draw(pil)
    font = _load_font(font_pt)
    h, w = img_rgb.shape[:2]

    def _stamp(x, baseline_y, txt, anchor="ls"):
        try:
            d.text((x, baseline_y), txt, font=font, fill=color, anchor=anchor)
        except TypeError:
            d.text((x, baseline_y - font_pt), txt, font=font, fill=color)

    # Wash out the existing upper-left labels.  Sized to fit "JWST F115/F150/F277"
    # at the current font size with comfortable padding.
    line_step_est = int(font_pt * 1.25)
    wash_h = int(LABEL_PAD_PX + 2 * line_step_est + LABEL_PAD_PX * 0.5)
    wash_w = int(0.60 * w)
    d.rectangle([(0, 0), (wash_w, wash_h)], fill=(0, 0, 0))

    line_step = int(font_pt * 1.25)

    # Upper-left: two lines (ID, header)
    # Use top-left anchor so we can stack predictably.
    try:
        d.text((LABEL_PAD_PX, LABEL_PAD_PX),
               f"ID={source_id}", font=font, fill=color, anchor="lt")
        d.text((LABEL_PAD_PX, LABEL_PAD_PX + line_step),
               header, font=font, fill=color, anchor="lt")
    except TypeError:
        d.text((LABEL_PAD_PX, LABEL_PAD_PX),
               f"ID={source_id}", font=font, fill=color)
        d.text((LABEL_PAD_PX, LABEL_PAD_PX + line_step),
               header, font=font, fill=color)

    # Lower-left: magnitude
    mag_txt = (f"{mag_label}={mag:.2f}"
               if (mag is not None and np.isfinite(mag))
               else f"{mag_label}=?")
    _stamp(LABEL_PAD_PX, h - LABEL_PAD_PX, mag_txt, anchor="ls")

    # Lower-right: redshift (right-aligned baseline)
    if z is not None and np.isfinite(z):
        _stamp(w - LABEL_PAD_PX, h - LABEL_PAD_PX, f"z={z:.3f}", anchor="rs")

    return np.asarray(pil)


def main():
    seq = TARGET_SEQ
    tag = f"{LIST_NAME}_{seq:04d}"
    files = {
        "hst":      HST_PNG    / f"sn_{tag}_hst.png",
        "euvis":    EUCLID_PNG / f"sn_{tag}_euclid_vis.png",
        "eunisp":   EUCLID_PNG / f"sn_{tag}_euclid_nisp.png",
        "jwst1":    JWST_PNG   / f"sn_{tag}_jwst1.png",
        "jwst2":    JWST_PNG   / f"sn_{tag}_jwst2.png",
    }
    for k, p in files.items():
        if not p.exists():
            sys.exit(f"missing {k}: {p}")
        print(f"  {k}: {p.name}")

    hst_g   = load_gray(files["hst"])
    euvis_g = load_gray(files["euvis"])
    jw1     = load_color(files["jwst1"])
    jw2     = load_color(files["jwst2"])
    eunisp  = load_color(files["eunisp"])

    # Detect SN in JWST native resolution (jwst1 array shape is the canonical one).
    sn = find_sn_position(hst_g, euvis_g, jw1, jw2)
    if sn is None:
        print("\nNo SN detected. Aborting.")
        sys.exit(2)

    sx, sy = sn
    h, w = jw1.shape[:2]
    if RECENTERED:
        # SN is at center -- use original SN-to-host direction for arm quadrant.
        # In the new cutout the host is at offset (-orig_dx, -orig_dy) from
        # center, so we still want arms pointing in +orig direction.
        arms = quadrant_arms(ORIG_SN_DX_PIL, ORIG_SN_DY_PIL)
        sx, sy = w // 2, h // 2     # snap crosshair to exact center
        print(f"\nRECENTERED cutout: snap crosshair to center ({sx},{sy}); "
              f"using ORIG offset ({ORIG_SN_DX_PIL:+d},{ORIG_SN_DY_PIL:+d}) -> arms={arms}")
    else:
        dx = sx - w/2.0
        dy = sy - h/2.0
        arms = quadrant_arms(dx, dy)
        print(f"\nSN @ ({sx},{sy}) in {w}x{h}; dx={dx:+.1f} dy={dy:+.1f}; arms={arms}")

    # Overlay crosshair on all 5 panels.  For HST/EuclidVIS (smaller native
    # res) we resize-up to the JWST grid so the crosshair lands in the same
    # physical place.
    def gray_to_rgb(g):
        u = (g * 255).clip(0, 255).astype(np.uint8)
        return np.stack([u, u, u], axis=-1)

    panels = {
        "jwst1":  (jw1 * 255).clip(0, 255).astype(np.uint8),
        "jwst2":  (jw2 * 255).clip(0, 255).astype(np.uint8),
        "hst":    gray_to_rgb(resize_to(hst_g,   w, h)),
        "euvis":  gray_to_rgb(resize_to(euvis_g, w, h)),
        "eunisp": (resize_to(eunisp, w, h) * 255).clip(0, 255).astype(np.uint8),
    }
    arms_override = arms if RECENTERED else None
    for name, img in panels.items():
        is_detection = (name in DETECTION_PANELS)
        style = "solid" if is_detection else "dotted"
        out = overlay_crosshair(img, (sx, sy), style=style,
                                arms_override=arms_override)

        info = PANEL_LABELS.get(name, dict(header=name, mag_label=name, mag=float("nan")))
        z = PANEL_Z.get(name)
        out = repaint_labels(out, SOURCE_ID,
                             info["header"], info["mag_label"], info["mag"],
                             z=z)

        out_path = OUT_DIR / f"crosshair_{tag}_{name}.png"
        Image.fromarray(out).save(str(out_path))
        kind = "SOLID (detection)" if is_detection else "DOTTED (no detection)"
        z_str = f" z={z:.3f}" if z is not None else ""
        print(f"  wrote {out_path}  [{kind}, {info['header']!r}, "
              f"{info['mag_label']}={info['mag']:.2f}{z_str}]")

    print(f"\nDone. Check {OUT_DIR}/")


if __name__ == "__main__":
    main()
