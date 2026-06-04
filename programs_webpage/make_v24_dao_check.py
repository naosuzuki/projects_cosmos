"""v24 DAO-circle diagnostic plots.

Two corrections from v23:
  ID=468896  SN = previous cyan + 0.2"E  (band: F115W)
  ID=63924   SN = host + 0.4"E + 1.1"N   (band: NIR-Y)

For each correction the closest DAO 3sigma candidate to the new hint
becomes the new SN position.

All 11 SN: render single-panel cutout in their detection band with
  red    = host position
  cyan   = selected SN
  yellow = all DAO 3sigma candidates inside a 3.0" search radius
  white  = 1" scale bar bottom-left
"""
import warnings
warnings.filterwarnings("ignore")

import csv, math, sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import recut_3_hst as R   # reuse path helpers + caches

# ─── inputs ──────────────────────────────────────────────────────────
SRC_CSV  = Path("/tmp/sn_lookup_11.csv")
OUT_CSV  = Path("/tmp/sn_lookup_12.csv")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/v24")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection band per telescope (which image the DAO_check is drawn on)
# Also doubles as the band used to update the SN position for 468896/63924.
DETECT = {
    "HST":    ("F814W", 0.134, lambda src: R._hst_path_fn(src),         "hst"),
    "JWST":   ("F115W", 0.057, lambda src: R._jwst_band_path_fn("f115w")(src), "jwst"),
    "EUCLID": ("NIR-Y", 0.524, lambda src: R._eu_band_path_fn("NIR-Y")(src),   "euclid"),
}

CUTOUT_ARCSEC = 6.0       # plot box size
SEARCH_ARCSEC = 3.0       # DAO search radius around hint
DETECT_SIGMA  = 3.0
CIRCLE_ARCSEC = 0.30      # marker circle radius

# ─── helpers ─────────────────────────────────────────────────────────
def open_image(sci_path):
    h = R._open_cached(sci_path)
    sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    wcs = R._wcs_cached(sci_path, sci_hdu)
    return sci_hdu, wcs

def pix_scale_arcsec(wcs):
    return float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0

def run_dao(sci_path, hint_ra, hint_dec, fwhm_as, search_as=SEARCH_ARCSEC,
            sigma=DETECT_SIGMA):
    """Run DAO on a sub-box; return list of dicts {ra, dec, peak, sharp, rnd1, sep_as}.

    sep_as is the angular distance from (hint_ra, hint_dec) in arcsec.
    """
    sci_hdu, wcs = open_image(sci_path)
    ps = pix_scale_arcsec(wcs)
    fwhm_px = max(1.5, fwhm_as / ps)
    box_as  = max(search_as * 2.0 + 1.0, 5.0)
    half    = int(np.ceil(box_as / ps)) + 1
    sx, sy = wcs.all_world2pix(hint_ra, hint_dec, 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci_hdu.data.shape[-2:]
    x0 = max(0, int(sx) - half); x1 = min(nx, int(sx) + half + 1)
    y0 = max(0, int(sy) - half); y1 = min(ny, int(sy) + half + 1)
    sub = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)
    if sub.size == 0: return []
    finite = np.isfinite(sub) & (sub != 0)
    if not finite.any(): return []
    _, bg_med, bg_std = sigma_clipped_stats(sub[finite], sigma=3.0, maxiters=3)
    res = DAOStarFinder(fwhm=fwhm_px, threshold=sigma*bg_std)(sub - bg_med)
    if res is None or len(res) == 0: return []
    out = []
    hpx, hpy = sx - x0, sy - y0
    for s in res:
        dx = float(s["xcentroid"]) - hpx
        dy = float(s["ycentroid"]) - hpy
        r_as = math.hypot(dx, dy) * ps
        if r_as > search_as:
            continue
        ra, dec = wcs.all_pix2world(float(s["xcentroid"]) + x0,
                                    float(s["ycentroid"]) + y0, 0)
        out.append(dict(ra=float(ra), dec=float(dec),
                        peak=float(s["peak"]),
                        sharp=float(s["sharpness"]),
                        rnd1=float(s["roundness1"]),
                        sep_as=r_as))
    out.sort(key=lambda d: d["sep_as"])
    return out


def closest_to(cands, ra, dec):
    """Return the DAO candidate closest to (ra, dec); or None."""
    if not cands: return None
    c0 = SkyCoord(ra*u.deg, dec*u.deg)
    best = None; best_d = 1e9
    for c in cands:
        d = c0.separation(SkyCoord(c["ra"]*u.deg, c["dec"]*u.deg)).arcsec
        if d < best_d:
            best, best_d = c, d
    return best, best_d


def draw_dao_check(sn, png_path):
    """Render the diagnostic PNG for one SN."""
    tel = sn["telescope"].upper()
    if tel not in DETECT:
        # fall back to JWST for unknown tag
        tel = "JWST"
    band_label, fwhm_as, path_fn, kind = DETECT[tel]
    sci_path = path_fn(sn)
    sci_hdu, wcs = open_image(sci_path)
    ps = pix_scale_arcsec(wcs)

    centre = SkyCoord(sn["sn_ra"]*u.deg, sn["sn_dec"]*u.deg)
    size_pix = int(np.ceil(CUTOUT_ARCSEC / ps))
    cut = Cutout2D(sci_hdu.data, centre, size=(size_pix, size_pix), wcs=wcs,
                   mode="partial", fill_value=np.nan)
    data = cut.data.astype(np.float64)
    cwcs = cut.wcs
    ny, nx = data.shape

    # display: sqrt stretch on positive part
    pos = np.where(np.isfinite(data) & (data > 0), data, 1e-6)
    img = np.sqrt(pos)
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    vmax = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half])) * 0.9
    if not np.isfinite(vmax) or vmax <= 0: vmax = 1.0

    # find all DAO candidates around the SN position (visualisation only)
    cands = run_dao(sci_path, sn["sn_ra"], sn["sn_dec"], fwhm_as,
                    search_as=SEARCH_ARCSEC, sigma=DETECT_SIGMA)

    fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111)
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=vmax)

    rad_px = CIRCLE_ARCSEC / ps
    # all DAO candidates - yellow
    for c in cands:
        px, py = cwcs.all_world2pix(c["ra"], c["dec"], 0)
        ax.add_patch(plt.Circle((float(px), float(py)), rad_px,
                                 edgecolor="#cc0", facecolor="none", lw=1.2))
    # host - red
    hx, hy = cwcs.all_world2pix(sn["host_ra"], sn["host_dec"], 0)
    ax.add_patch(plt.Circle((float(hx), float(hy)), rad_px,
                             edgecolor="red", facecolor="none", lw=2.0))
    # SN selected - cyan
    sx, sy = cwcs.all_world2pix(sn["sn_ra"], sn["sn_dec"], 0)
    ax.add_patch(plt.Circle((float(sx), float(sy)), rad_px,
                             edgecolor="#0cc", facecolor="none", lw=2.0))

    # 1" scale bar bottom-left
    bar_px = 1.0 / ps
    ax.plot([nx*0.06, nx*0.06 + bar_px], [ny*0.06, ny*0.06],
            color="white", lw=2.5)
    ax.text(nx*0.06 + bar_px/2, ny*0.06 + ny*0.015, '1"',
            color="white", ha="center", va="bottom", fontsize=9)

    sep_host = SkyCoord(sn["sn_ra"]*u.deg, sn["sn_dec"]*u.deg).separation(
        SkyCoord(sn["host_ra"]*u.deg, sn["host_dec"]*u.deg)).arcsec
    ax.text(nx*0.04, ny*0.96,
            f"ID {sn['id']}  {tel}/{band_label}  sep_host={sep_host:.2f}\"  DAO={len(cands)}",
            color="yellow", fontsize=9, ha="left", va="top",
            family="monospace")

    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off"); fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)
    return dict(sep_host=sep_host, n_cands=len(cands), band=band_label)


# ─── load + apply two corrections ────────────────────────────────────
rows = []
with SRC_CSV.open() as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        r["id"]      = int(r["id"])
        r["sn_ra"]   = float(r["sn_ra"]);   r["sn_dec"]   = float(r["sn_dec"])
        r["host_ra"] = float(r["host_ra"]); r["host_dec"] = float(r["host_dec"])
        rows.append(r)

# 468896 — previous cyan + 0.2"E (in F115W)
r468 = next(r for r in rows if r["id"] == 468896)
old_ra468, old_dec468 = r468["sn_ra"], r468["sn_dec"]
hint_ra_468  = old_ra468 + 0.2 / 3600.0 / math.cos(math.radians(old_dec468))
hint_dec_468 = old_dec468
band_label, fwhm_as, path_fn, kind = DETECT["JWST"]
cands468 = run_dao(path_fn(r468), hint_ra_468, hint_dec_468, fwhm_as,
                    search_as=1.0, sigma=DETECT_SIGMA)
print(f"ID 468896  hint=({hint_ra_468:.6f},{hint_dec_468:.6f})  "
      f"{len(cands468)} DAO 3σ within 1.0\"")
if cands468:
    best, d = closest_to(cands468, hint_ra_468, hint_dec_468)
    print(f"  closest: RA={best['ra']:.6f} Dec={best['dec']:.6f}  "
          f"d_hint={d:.2f}\" peak={best['peak']:.3f} "
          f"sharp={best['sharp']:.2f} rnd1={best['rnd1']:+.2f}")
    r468["sn_ra"]  = best["ra"]
    r468["sn_dec"] = best["dec"]
else:
    print("  NO DAO 3σ candidate near hint — keeping cyan untouched")

# 63924 — host + 0.4"E + 1.1"N (in NIR-Y)
r639 = next(r for r in rows if r["id"] == 63924)
hra, hdec = r639["host_ra"], r639["host_dec"]
hint_ra_639  = hra  + 0.4 / 3600.0 / math.cos(math.radians(hdec))
hint_dec_639 = hdec + 1.1 / 3600.0
band_label, fwhm_as, path_fn, kind = DETECT["EUCLID"]
cands639 = run_dao(path_fn(r639), hint_ra_639, hint_dec_639, fwhm_as,
                    search_as=1.0, sigma=DETECT_SIGMA)
print(f"ID 63924   hint=({hint_ra_639:.6f},{hint_dec_639:.6f})  "
      f"{len(cands639)} DAO 3σ within 1.0\"")
if cands639:
    best, d = closest_to(cands639, hint_ra_639, hint_dec_639)
    print(f"  closest: RA={best['ra']:.6f} Dec={best['dec']:.6f}  "
          f"d_hint={d:.2f}\" peak={best['peak']:.3f} "
          f"sharp={best['sharp']:.2f} rnd1={best['rnd1']:+.2f}")
    r639["sn_ra"]  = best["ra"]
    r639["sn_dec"] = best["dec"]
else:
    print("  NO DAO 3σ candidate near hint — keeping host position")

# ─── persist updated CSV ─────────────────────────────────────────────
with OUT_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id","telescope","sn_ra","sn_dec",
                                       "host_ra","host_dec","hst","jwst","euclid"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in w.fieldnames})
print(f"wrote {OUT_CSV}")

# ─── render all 11 PNGs ─────────────────────────────────────────────
print("\n=== render DAO check PNGs ===")
summary = []
for r in rows:
    png = OUT_DIR / f"sn_{r['id']}_DAO_check.png"
    info = draw_dao_check(r, png)
    sep_host = info["sep_host"]
    print(f"  {r['id']:6d} ({info['band']:>5}) sep_host={sep_host:.2f}\"  "
          f"DAO={info['n_cands']:2d}  -> {png.name}")
    summary.append((r["id"], info["band"], sep_host, info["n_cands"]))

# ─── index.html ─────────────────────────────────────────────────────
html = ["<!doctype html><html><head><title>v24 — DAO location check (corrections)</title>",
        "<style>body{font-family:serif;background:#fafafa}",
        ".wrap{max-width:1500px;margin:0 auto;padding:20px}",
        ".back{font-size:13px;color:#666}",
        ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}",
        ".cell img{width:100%;display:block;border:1px solid #ddd}",
        ".cell .fn{font-family:monospace;font-size:10px;color:#999;text-align:center;margin-top:3px}",
        "</style></head><body><div class='wrap'>",
        "<p class='back'><a href='../index.html'>&larr; _crosshair_test</a> | "
        "<a href='../../index.html'>all decks</a></p>",
        "<h1>v24 — DAO location check (two more user corrections)</h1>",
        "<p>Corrections from v23: <b>468896</b> SN = previous cyan + 0.2&Prime;E (F115W); "
        "<b>63924</b> SN = host + 0.4&Prime;E + 1.1&Prime;N (NIR-Y).</p>",
        "<p><b style='color:red'>host=red</b>, "
        "<b style='color:#0cc'>SN=cyan</b>, "
        "<b style='color:#cc0'>yellow=DAO 3&sigma; candidates</b>.</p>",
        "<div class='grid'>"]
for r in sorted(rows, key=lambda x: x["id"]):
    png = f"sn_{r['id']}_DAO_check.png"
    html.append(f"<div class='cell'><img src='{png}'><div class='fn'>{png}</div></div>")
html += ["</div></div></body></html>"]
(OUT_DIR / "index.html").write_text("\n".join(html))
print(f"wrote {OUT_DIR/'index.html'}")

print("\n=== summary ===")
for cid, b, s, n in summary:
    print(f"  {cid:6d} ({b:>5}) sep_host={s:.2f}\"  DAO_cands={n}")
print("DONE")
