"""v26 DAO-circle diagnostic plots (after 63924 correction).

Source CSV: /tmp/sn_lookup_13.csv (63924 SN updated to the compact knot
1.05"E + 0.23"N of the v24 cyan; user-confirmed).

Optimisations vs v24:
  * Single DAO scan per source (drops the correction pre-scan).
  * The "selected-SN" DAO parameters are taken from the same scan that
    draws the yellow circles — no second FITS open, no second DAO call.
  * Yellow circles drawn via a PatchCollection (one matplotlib artist
    instead of N individual Circles); helps HST panels with 100-300
    candidates.

For each SN we report the DAO parameters of the candidate CLOSEST to the
catalog SN position (= what cyan is drawn on):
    peak    DAO peak intensity after background subtraction
    sharp   DAOPHOT sharpness  (compactness; ~0.4-0.9 for PSF-like)
    rnd1    DAOPHOT roundness1 (asymmetry along axes; |rnd1| < 0.5 ideal)
    rnd2    DAOPHOT roundness2 (symmetric Gaussian fit residual)
    flux    DAO flux (integrated above background)
    mag     -2.5 log10(flux) instrument magnitude
    sep_sn  arcsec from catalog SN position to nearest DAO
"""
import warnings; warnings.filterwarnings("ignore")
import csv, math, sys, time
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
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).parent))
import recut_3_hst as R   # reuse path helpers + caches

# ─── inputs ──────────────────────────────────────────────────────────
SRC_CSV  = Path("/tmp/sn_lookup_13.csv")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/v26")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Detection band per telescope (which image the DAO_check is drawn on)
DETECT = {
    "HST":    ("F814W", 0.134, lambda src: R._hst_path_fn(src),                   "hst"),
    "JWST":   ("F115W", 0.057, lambda src: R._jwst_band_path_fn("f115w")(src),    "jwst"),
    "EUCLID": ("NIR-Y", 0.524, lambda src: R._eu_band_path_fn("NIR-Y")(src),      "euclid"),
}

CUTOUT_ARCSEC  = 6.0       # plot box
SEARCH_ARCSEC  = 3.0       # DAO search radius around SN position
DETECT_SIGMA   = 3.0
CIRCLE_ARCSEC  = 0.30      # marker circle radius


# ─── helpers ────────────────────────────────────────────────────────
def open_image(sci_path):
    h = R._open_cached(sci_path)
    sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    return sci_hdu, R._wcs_cached(sci_path, sci_hdu)

def pix_scale_arcsec(wcs):
    return float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0


def scan_dao(sci_path, sn_ra, sn_dec, fwhm_as,
             search_as=SEARCH_ARCSEC, sigma=DETECT_SIGMA):
    """Run DAOStarFinder once around the SN position.

    Returns:
        cands  list of dicts {ra, dec, peak, sharp, rnd1, rnd2, flux, mag, sep_sn}
               sorted by sep_sn (distance from catalog SN position).
        meta   dict {sci_hdu, wcs, sub, x0, y0, ps, ny, nx, hpx, hpy} for plotting.
    """
    sci_hdu, wcs = open_image(sci_path)
    ps = pix_scale_arcsec(wcs)
    fwhm_px = max(1.5, fwhm_as / ps)
    box_as  = max(search_as * 2.0 + 1.0, 5.0)
    half    = int(np.ceil(box_as / ps)) + 1
    sx, sy = wcs.all_world2pix(sn_ra, sn_dec, 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci_hdu.data.shape[-2:]
    x0 = max(0, int(sx) - half); x1 = min(nx, int(sx) + half + 1)
    y0 = max(0, int(sy) - half); y1 = min(ny, int(sy) + half + 1)
    sub = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)
    meta = dict(sci_hdu=sci_hdu, wcs=wcs, sub=sub, x0=x0, y0=y0,
                ps=ps, ny=ny, nx=nx, hpx=sx-x0, hpy=sy-y0,
                fwhm_px=fwhm_px)
    if sub.size == 0:
        return [], meta
    finite = np.isfinite(sub) & (sub != 0)
    if not finite.any():
        return [], meta
    _, bg_med, bg_std = sigma_clipped_stats(sub[finite], sigma=3.0, maxiters=3)
    res = DAOStarFinder(fwhm=fwhm_px, threshold=sigma*bg_std)(sub - bg_med)
    if res is None or len(res) == 0:
        return [], meta
    out = []
    for s in res:
        dx = float(s["xcentroid"]) - meta["hpx"]
        dy = float(s["ycentroid"]) - meta["hpy"]
        sep_sn = math.hypot(dx, dy) * ps
        if sep_sn > search_as:
            continue
        ra, dec = wcs.all_pix2world(float(s["xcentroid"]) + x0,
                                    float(s["ycentroid"]) + y0, 0)
        out.append(dict(ra=float(ra), dec=float(dec),
                        peak=float(s["peak"]),
                        sharp=float(s["sharpness"]),
                        rnd1=float(s["roundness1"]),
                        rnd2=float(s["roundness2"]),
                        flux=float(s["flux"]),
                        mag=float(s["mag"]),
                        sep_sn=sep_sn))
    out.sort(key=lambda d: d["sep_sn"])
    return out, meta


def draw_dao_check(sn, cands, meta, band_label, png_path):
    """Render the per-SN diagnostic PNG using already-scanned candidates."""
    sci_hdu, wcs, ps = meta["sci_hdu"], meta["wcs"], meta["ps"]

    centre   = SkyCoord(sn["sn_ra"]*u.deg, sn["sn_dec"]*u.deg)
    size_pix = int(np.ceil(CUTOUT_ARCSEC / ps))
    cut = Cutout2D(sci_hdu.data, centre, size=(size_pix, size_pix), wcs=wcs,
                   mode="partial", fill_value=np.nan)
    data = cut.data.astype(np.float64)
    cwcs = cut.wcs
    ny, nx = data.shape

    pos = np.where(np.isfinite(data) & (data > 0), data, 1e-6)
    img = np.sqrt(pos)
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    vmax = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half])) * 0.9
    if not np.isfinite(vmax) or vmax <= 0: vmax = 1.0

    fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111)
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    rad_px = CIRCLE_ARCSEC / ps

    # ---- single PatchCollection for all yellow DAO candidates ----
    if cands:
        ras  = np.array([c["ra"]  for c in cands])
        decs = np.array([c["dec"] for c in cands])
        pxs, pys = cwcs.all_world2pix(ras, decs, 0)
        circs = [Circle((float(px), float(py)), rad_px) for px, py in zip(pxs, pys)]
        ax.add_collection(PatchCollection(circs, edgecolor="#cc0",
                                          facecolor="none", linewidth=1.2))
    # host (red)
    hx, hy = cwcs.all_world2pix(sn["host_ra"], sn["host_dec"], 0)
    ax.add_patch(Circle((float(hx), float(hy)), rad_px,
                         edgecolor="red", facecolor="none", lw=2.0))
    # SN selected (cyan)
    sx, sy = cwcs.all_world2pix(sn["sn_ra"], sn["sn_dec"], 0)
    ax.add_patch(Circle((float(sx), float(sy)), rad_px,
                         edgecolor="#0cc", facecolor="none", lw=2.0))

    # 1" scale bar
    bar_px = 1.0 / ps
    ax.plot([nx*0.06, nx*0.06 + bar_px], [ny*0.06, ny*0.06],
            color="white", lw=2.5)
    ax.text(nx*0.06 + bar_px/2, ny*0.06 + ny*0.015, '1"',
            color="white", ha="center", va="bottom", fontsize=9)

    sep_host = SkyCoord(sn["sn_ra"]*u.deg, sn["sn_dec"]*u.deg).separation(
        SkyCoord(sn["host_ra"]*u.deg, sn["host_dec"]*u.deg)).arcsec
    ax.text(nx*0.04, ny*0.96,
            f"ID {sn['id']}  {sn['telescope']}/{band_label}  "
            f"sep_host={sep_host:.2f}\"  DAO={len(cands)}",
            color="yellow", fontsize=9, ha="left", va="top",
            family="monospace")

    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)
    return sep_host


# ─── load ───────────────────────────────────────────────────────────
rows = []
with SRC_CSV.open() as f:
    for r in csv.DictReader(f):
        r["id"]      = int(r["id"])
        r["sn_ra"]   = float(r["sn_ra"]);   r["sn_dec"]   = float(r["sn_dec"])
        r["host_ra"] = float(r["host_ra"]); r["host_dec"] = float(r["host_dec"])
        rows.append(r)
print(f"loaded {len(rows)} sources from {SRC_CSV}")


# ─── render all 11 PNGs ────────────────────────────────────────────
t_total = time.time()
summary = []
print("\n=== per-source DAO scan + render ===")
print(f"{'ID':>7} {'BAND':>6} {'t_dao':>7} {'t_png':>7} "
      f"{'nDAO':>5} {'sep_host':>9} {'sep_sn':>7} "
      f"{'peak':>9} {'sharp':>6} {'rnd1':>7} {'rnd2':>7} "
      f"{'flux':>10} {'mag':>7}")
print("-" * 120)

for r in rows:
    tel = r["telescope"].upper()
    band_label, fwhm_as, path_fn, kind = DETECT[tel]
    sci_path = path_fn(r)

    t0 = time.time()
    cands, meta = scan_dao(sci_path, r["sn_ra"], r["sn_dec"], fwhm_as)
    t_dao = time.time() - t0

    t0 = time.time()
    png = OUT_DIR / f"sn_{r['id']}_DAO_check.png"
    sep_host = draw_dao_check(r, cands, meta, band_label, png)
    t_png = time.time() - t0

    # Selected-SN DAO row = closest candidate to (sn_ra, sn_dec)
    real = cands[0] if cands else None
    sep_sn = real["sep_sn"] if real else float("nan")
    peak   = real["peak"]   if real else float("nan")
    sharp  = real["sharp"]  if real else float("nan")
    rnd1   = real["rnd1"]   if real else float("nan")
    rnd2   = real["rnd2"]   if real else float("nan")
    flux   = real["flux"]   if real else float("nan")
    mag    = real["mag"]    if real else float("nan")

    print(f"{r['id']:>7d} {band_label:>6} {t_dao:>6.2f}s {t_png:>6.2f}s "
          f"{len(cands):>5d} {sep_host:>8.2f}\" {sep_sn:>6.2f}\" "
          f"{peak:>9.3f} {sharp:>+6.2f} {rnd1:>+7.2f} {rnd2:>+7.2f} "
          f"{flux:>10.2f} {mag:>+7.2f}")

    summary.append(dict(id=r["id"], band=band_label,
                        sep_host=sep_host, sep_sn=sep_sn,
                        peak=peak, sharp=sharp, rnd1=rnd1, rnd2=rnd2,
                        flux=flux, mag=mag, n_cands=len(cands),
                        t_dao=t_dao, t_png=t_png))

t_total = time.time() - t_total
print(f"\nTotal wall: {t_total:.2f} s")


# ─── write csv of all DAO parameters ───────────────────────────────
csv_out = OUT_DIR / "dao_params.csv"
with csv_out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id","band","n_cands","sep_host","sep_sn",
                                       "peak","sharp","rnd1","rnd2","flux","mag",
                                       "t_dao","t_png"])
    w.writeheader()
    for s in summary:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in s.items()})
print(f"wrote {csv_out}")


# ─── index.html ────────────────────────────────────────────────────
html = ["<!doctype html><html><head><title>v26 — DAO location check (after 63924 fix)</title>",
        "<style>body{font-family:serif;background:#fafafa}",
        ".wrap{max-width:1500px;margin:0 auto;padding:20px}",
        ".back{font-size:13px;color:#666}",
        ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}",
        ".cell img{width:100%;display:block;border:1px solid #ddd}",
        ".cell .fn{font-family:monospace;font-size:10px;color:#999;text-align:center;margin-top:3px}",
        "table{border-collapse:collapse;font-family:monospace;font-size:11px;margin-top:18px}",
        "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
        "th{background:#eee}",
        "</style></head><body><div class='wrap'>",
        "<p class='back'><a href='../index.html'>&larr; _crosshair_test</a> | "
        "<a href='../../index.html'>all decks</a></p>",
        "<h1>v26 &mdash; DAO location check (after 63924 SN fix)</h1>",
        "<p>Correction from v24: <b>63924</b> SN = previous cyan + 1.2&Prime;E + 0.1&Prime;N "
        "(snapped to closest DAO 3&sigma;, peak 7.06, sharp 0.45, rnd1 +0.12).</p>",
        "<p><b style='color:red'>host=red</b>, "
        "<b style='color:#0cc'>SN=cyan</b>, "
        "<b style='color:#cc0'>yellow=DAO 3&sigma; candidates</b>.</p>",
        "<div class='grid'>"]
for r in sorted(rows, key=lambda x: x["id"]):
    png = f"sn_{r['id']}_DAO_check.png"
    html.append(f"<div class='cell'><img src='{png}'><div class='fn'>{png}</div></div>")
html.append("</div>")
html.append("<h2>Selected-SN DAO parameters (closest candidate to catalog position)</h2>")
html.append("<table><thead><tr>"
            "<th>ID</th><th>band</th><th>nDAO</th>"
            "<th>sep_host</th><th>sep_sn</th>"
            "<th>peak</th><th>sharp</th><th>rnd1</th><th>rnd2</th>"
            "<th>flux</th><th>mag</th></tr></thead><tbody>")
for s in sorted(summary, key=lambda x: x["id"]):
    html.append(
        f"<tr><td>{s['id']}</td><td>{s['band']}</td><td>{s['n_cands']}</td>"
        f"<td>{s['sep_host']:.2f}&Prime;</td><td>{s['sep_sn']:.2f}&Prime;</td>"
        f"<td>{s['peak']:.3f}</td><td>{s['sharp']:+.2f}</td>"
        f"<td>{s['rnd1']:+.2f}</td><td>{s['rnd2']:+.2f}</td>"
        f"<td>{s['flux']:.2f}</td><td>{s['mag']:+.2f}</td></tr>")
html.append("</tbody></table>")
html.append("</div></body></html>")
(OUT_DIR / "index.html").write_text("\n".join(html))
print(f"wrote {OUT_DIR/'index.html'}")

# top-level index link
top = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/index.html")
s = top.read_text()
if "v26/index.html" not in s and "</div></body></html>" in s:
    line = ("<div class='ver'><a href='v26/index.html'>v26</a> &mdash; "
            "DAO circles (63924 SN moved to compact knot 1.2&Prime;E) &nbsp; "
            "<span class='meta'>11 PNGs</span></div>\n</div></body></html>")
    s = s.replace("</div></body></html>", line)
    top.write_text(s)
    print("added v26 to top-level index")

print("DONE")
