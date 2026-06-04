"""v27 multi-band DAO diagnostic.

For each SN, scan DAO in every band of the source's own telescope:
  HST     -> F814W                                   (1 band)
  JWST    -> F115W, F150W, F277W, F444W              (4 bands)
  EUCLID  -> NIR-Y, NIR-J, NIR-H                     (3 bands; skip VIS, different epoch)

For each (source, band) pair, take the DAO candidate closest to the
catalog SN position (within 1.0") and record:
    peak      DAO peak intensity (background-subtracted)
    bg_std    local 3-sigma-clipped background standard deviation
    peak_snr  peak / bg_std  (the per-band S/N proxy used to PICK the band)
    sharp     DAOPHOT sharpness
    rnd1      DAOPHOT roundness1
    rnd2      DAOPHOT roundness2
    flux      DAO integrated flux
    mag       -2.5 log10(flux)
    sep_sn    arcsec between catalog SN position and the closest DAO

Best-S/N band per source = argmax(peak_snr).  The cutout PNG is rendered
in that band so the cyan circle is on the SN as it appears in its
strongest detection channel.

Outputs:
  v27/sn_<id>_DAO_check.png      one per SN, rendered in best-S/N band
  v27/dao_params_best.csv        one row per SN, best-band parameters
  v27/dao_params_allbands.csv    one row per (SN, band) pair
  v27/index.html                 image grid + best-band table + full table
"""
import warnings; warnings.filterwarnings("ignore")
import csv, math, sys, time
from pathlib import Path
import numpy as np
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
import recut_3_hst as R

SRC_CSV  = Path("/tmp/sn_lookup_13.csv")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/v27")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── per-telescope band lists (PSF FWHM measured 2026-05-26) ────────
TELESCOPE_BANDS = {
    "HST":    [("F814W", 0.134, lambda s: R._hst_path_fn(s))],
    "JWST":   [("F115W", 0.057, lambda s: R._jwst_band_path_fn("f115w")(s)),
               ("F150W", 0.057, lambda s: R._jwst_band_path_fn("f150w")(s)),
               ("F277W", 0.130, lambda s: R._jwst_band_path_fn("f277w")(s)),
               ("F444W", 0.160, lambda s: R._jwst_band_path_fn("f444w")(s))],
    "EUCLID": [("NIR-Y", 0.524, lambda s: R._eu_band_path_fn("NIR-Y")(s)),
               ("NIR-J", 0.537, lambda s: R._eu_band_path_fn("NIR-J")(s)),
               ("NIR-H", 0.567, lambda s: R._eu_band_path_fn("NIR-H")(s))],
}

CUTOUT_ARCSEC  = 6.0
SEARCH_ARCSEC  = 3.0       # DAO box around SN position for the panel
NEAREST_AS     = 1.0       # tolerance for "closest DAO to SN" per band
SIGMA_DETECT   = 3.0
CIRCLE_ARCSEC  = 0.30


# ─── helpers ────────────────────────────────────────────────────────
def open_image(sci_path):
    h = R._open_cached(sci_path)
    sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    return sci_hdu, R._wcs_cached(sci_path, sci_hdu)


def pix_scale(wcs):
    return float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0


def scan_dao(sci_path, sn_ra, sn_dec, fwhm_as,
             search_as=SEARCH_ARCSEC, sigma=SIGMA_DETECT):
    """Run DAOStarFinder once around the SN position.

    Returns (cands, meta).  cands sorted by sep_sn.  meta carries
    everything needed to draw a panel from this same call.
    """
    sci_hdu, wcs = open_image(sci_path)
    ps = pix_scale(wcs)
    fwhm_px = max(1.5, fwhm_as / ps)
    box_as  = max(search_as * 2.0 + 1.0, 5.0)
    half    = int(np.ceil(box_as / ps)) + 1
    sx, sy = wcs.all_world2pix(sn_ra, sn_dec, 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci_hdu.data.shape[-2:]
    x0 = max(0, int(sx) - half); x1 = min(nx, int(sx) + half + 1)
    y0 = max(0, int(sy) - half); y1 = min(ny, int(sy) + half + 1)
    sub = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)
    meta = dict(sci_hdu=sci_hdu, wcs=wcs, ps=ps,
                bg_med=np.nan, bg_std=np.nan,
                ny=ny, nx=nx, hpx=sx-x0, hpy=sy-y0)
    if sub.size == 0:           return [], meta
    finite = np.isfinite(sub) & (sub != 0)
    if not finite.any():        return [], meta
    _, bg_med, bg_std = sigma_clipped_stats(sub[finite], sigma=3.0, maxiters=3)
    meta["bg_med"] = bg_med; meta["bg_std"] = bg_std
    res = DAOStarFinder(fwhm=fwhm_px, threshold=sigma*bg_std)(sub - bg_med)
    if res is None or len(res) == 0:  return [], meta
    out = []
    for s in res:
        dx = float(s["xcentroid"]) - meta["hpx"]
        dy = float(s["ycentroid"]) - meta["hpy"]
        sep_sn = math.hypot(dx, dy) * ps
        if sep_sn > search_as: continue
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


def draw_panel(sn, cands, meta, band_label, png_path):
    sci_hdu, wcs, ps = meta["sci_hdu"], meta["wcs"], meta["ps"]
    centre   = SkyCoord(sn["sn_ra"]*u.deg, sn["sn_dec"]*u.deg)
    size_pix = int(np.ceil(CUTOUT_ARCSEC / ps))
    cut = Cutout2D(sci_hdu.data, centre, size=(size_pix, size_pix), wcs=wcs,
                   mode="partial", fill_value=np.nan)
    data = cut.data.astype(np.float64); cwcs = cut.wcs
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

    if cands:
        ras  = np.array([c["ra"]  for c in cands])
        decs = np.array([c["dec"] for c in cands])
        pxs, pys = cwcs.all_world2pix(ras, decs, 0)
        circs = [Circle((float(px), float(py)), rad_px) for px, py in zip(pxs, pys)]
        ax.add_collection(PatchCollection(circs, edgecolor="#cc0",
                                          facecolor="none", linewidth=1.2))
    hx, hy = cwcs.all_world2pix(sn["host_ra"], sn["host_dec"], 0)
    ax.add_patch(Circle((float(hx), float(hy)), rad_px,
                         edgecolor="red", facecolor="none", lw=2.0))
    sx, sy = cwcs.all_world2pix(sn["sn_ra"], sn["sn_dec"], 0)
    ax.add_patch(Circle((float(sx), float(sy)), rad_px,
                         edgecolor="#0cc", facecolor="none", lw=2.0))

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


# ─── load ───────────────────────────────────────────────────────────
rows = []
with SRC_CSV.open() as f:
    for r in csv.DictReader(f):
        r["id"]      = int(r["id"])
        r["sn_ra"]   = float(r["sn_ra"]);   r["sn_dec"]   = float(r["sn_dec"])
        r["host_ra"] = float(r["host_ra"]); r["host_dec"] = float(r["host_dec"])
        rows.append(r)
print(f"loaded {len(rows)} sources from {SRC_CSV}")

# ─── multi-band scan ────────────────────────────────────────────────
t0_all = time.time()
allband_rows = []     # one row per (source, band)
best_rows    = []     # one row per source (best-S/N band)

for r in rows:
    tel = r["telescope"].upper()
    bands = TELESCOPE_BANDS.get(tel)
    if bands is None:
        print(f"  {r['id']}: unknown telescope {tel}; skip")
        continue

    per_band = []
    for band, fwhm_as, path_fn in bands:
        sci_path = path_fn(r)
        t0 = time.time()
        cands, meta = scan_dao(sci_path, r["sn_ra"], r["sn_dec"], fwhm_as)
        t_dao = time.time() - t0

        nearest = next((c for c in cands if c["sep_sn"] <= NEAREST_AS), None)
        bg_std  = meta["bg_std"]
        if nearest is not None and bg_std > 0:
            peak_snr = nearest["peak"] / bg_std
        else:
            peak_snr = float("nan")

        rec = dict(id=r["id"], telescope=tel, band=band,
                   n_cands=len(cands), bg_std=float(bg_std),
                   sep_sn=(nearest["sep_sn"] if nearest else float("nan")),
                   peak=(nearest["peak"] if nearest else float("nan")),
                   peak_snr=peak_snr,
                   sharp=(nearest["sharp"] if nearest else float("nan")),
                   rnd1=(nearest["rnd1"] if nearest else float("nan")),
                   rnd2=(nearest["rnd2"] if nearest else float("nan")),
                   flux=(nearest["flux"] if nearest else float("nan")),
                   mag=(nearest["mag"] if nearest else float("nan")),
                   t_dao=t_dao,
                   _cands=cands, _meta=meta)
        per_band.append(rec)
        allband_rows.append(rec)

    # pick best-S/N band that actually has a candidate within NEAREST_AS
    detected = [b for b in per_band if np.isfinite(b["peak_snr"])]
    if not detected:
        # no DAO match within NEAREST_AS in any band; fall back to highest bg_std band
        best = max(per_band, key=lambda b: b["bg_std"])
    else:
        best = max(detected, key=lambda b: b["peak_snr"])
    sep_host = SkyCoord(r["sn_ra"]*u.deg, r["sn_dec"]*u.deg).separation(
        SkyCoord(r["host_ra"]*u.deg, r["host_dec"]*u.deg)).arcsec
    best_rows.append(dict(id=r["id"], telescope=tel, band=best["band"],
                          sep_host=sep_host,
                          sep_sn=best["sep_sn"],
                          peak=best["peak"], peak_snr=best["peak_snr"],
                          sharp=best["sharp"], rnd1=best["rnd1"],
                          rnd2=best["rnd2"], flux=best["flux"], mag=best["mag"],
                          _cands=best["_cands"], _meta=best["_meta"]))

t_scan = time.time() - t0_all

# ─── render PNGs in best-S/N band ───────────────────────────────────
t0 = time.time()
for r in rows:
    b = next(x for x in best_rows if x["id"] == r["id"])
    png = OUT_DIR / f"sn_{r['id']}_DAO_check.png"
    draw_panel(r, b["_cands"], b["_meta"], b["band"], png)
t_png = time.time() - t0

# ─── print best-band table ──────────────────────────────────────────
print("\n=== BEST-S/N BAND per source (DAO parameters) ===")
print(f"{'ID':>7} {'TEL':>6} {'BAND':>5} {'sep_host':>9} {'sep_sn':>7} "
      f"{'peak':>9} {'bgstd':>9} {'peak_snr':>9} "
      f"{'sharp':>6} {'rnd1':>7} {'rnd2':>7} {'flux':>10} {'mag':>7}")
print("-" * 130)
for s in sorted(best_rows, key=lambda x: x["id"]):
    print(f"{s['id']:>7d} {s['telescope']:>6} {s['band']:>5} "
          f"{s['sep_host']:>8.2f}\" {s['sep_sn']:>6.2f}\" "
          f"{s['peak']:>9.3f} {s['_meta']['bg_std']:>9.4f} "
          f"{s['peak_snr']:>9.2f} "
          f"{s['sharp']:>+6.2f} {s['rnd1']:>+7.2f} {s['rnd2']:>+7.2f} "
          f"{s['flux']:>10.2f} {s['mag']:>+7.2f}")
print(f"\nscans={t_scan:.2f}s  renders={t_png:.2f}s  total={t_scan+t_png:.2f}s")

# ─── full multi-band table (one row per (SN, band)) ─────────────────
print("\n=== FULL MULTI-BAND breakdown ===")
print(f"{'ID':>7} {'BAND':>5} {'sep_sn':>7} {'peak':>9} {'bg_std':>9} "
      f"{'peak_snr':>9} {'sharp':>6} {'rnd1':>7} {'rnd2':>7} "
      f"{'flux':>10} {'mag':>7}  BEST?")
print("-" * 120)
best_pairs = {(b["id"], b["band"]) for b in best_rows}
for s in allband_rows:
    flag = "  <==" if (s["id"], s["band"]) in best_pairs else ""
    peak_str  = "    nan  " if not np.isfinite(s["peak"])     else f"{s['peak']:>9.3f}"
    bgstd_str = "    nan  " if not np.isfinite(s["bg_std"])   else f"{s['bg_std']:>9.4f}"
    snr_str   = "    nan  " if not np.isfinite(s["peak_snr"]) else f"{s['peak_snr']:>9.2f}"
    sharp_str = "   nan"   if not np.isfinite(s["sharp"])    else f"{s['sharp']:>+6.2f}"
    rnd1_str  = "    nan" if not np.isfinite(s["rnd1"])     else f"{s['rnd1']:>+7.2f}"
    rnd2_str  = "    nan" if not np.isfinite(s["rnd2"])     else f"{s['rnd2']:>+7.2f}"
    flux_str  = "       nan" if not np.isfinite(s["flux"])  else f"{s['flux']:>10.2f}"
    mag_str   = "    nan"   if not np.isfinite(s["mag"])     else f"{s['mag']:>+7.2f}"
    sep_str   = "    nan" if not np.isfinite(s["sep_sn"])  else f"{s['sep_sn']:>6.2f}\""
    print(f"{s['id']:>7d} {s['band']:>5} {sep_str} {peak_str} {bgstd_str} "
          f"{snr_str} {sharp_str} {rnd1_str} {rnd2_str} "
          f"{flux_str} {mag_str}{flag}")

# ─── CSV outputs ────────────────────────────────────────────────────
out_best = OUT_DIR / "dao_params_best.csv"
with out_best.open("w", newline="") as f:
    cols = ["id","telescope","band","sep_host","sep_sn","peak","bg_std",
            "peak_snr","sharp","rnd1","rnd2","flux","mag"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for s in sorted(best_rows, key=lambda x: x["id"]):
        row = {k: s.get(k, "") for k in cols}
        row["bg_std"] = round(s["_meta"]["bg_std"], 5)
        for k in ("sep_host","sep_sn","peak","peak_snr","sharp","rnd1","rnd2","flux","mag"):
            row[k] = round(float(s[k]), 4) if np.isfinite(s[k]) else ""
        w.writerow(row)
print(f"\nwrote {out_best}")

out_all = OUT_DIR / "dao_params_allbands.csv"
with out_all.open("w", newline="") as f:
    cols = ["id","telescope","band","n_cands","sep_sn","peak","bg_std",
            "peak_snr","sharp","rnd1","rnd2","flux","mag"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for s in allband_rows:
        row = {k: s.get(k, "") for k in cols}
        for k in ("sep_sn","peak","bg_std","peak_snr","sharp","rnd1","rnd2","flux","mag"):
            v = s[k]
            row[k] = round(float(v), 5) if np.isfinite(v) else ""
        w.writerow(row)
print(f"wrote {out_all}")

# ─── index.html ────────────────────────────────────────────────────
html = ["<!doctype html><html><head><title>v27 — multi-band DAO, best-S/N band per source</title>",
        "<style>body{font-family:serif;background:#fafafa}",
        ".wrap{max-width:1500px;margin:0 auto;padding:20px}",
        ".back{font-size:13px;color:#666}",
        ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}",
        ".cell img{width:100%;display:block;border:1px solid #ddd}",
        ".cell .fn{font-family:monospace;font-size:10px;color:#999;text-align:center;margin-top:3px}",
        "table{border-collapse:collapse;font-family:monospace;font-size:11px;margin-top:18px}",
        "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
        "th{background:#eee} td.best{background:#fff5b0}",
        "</style></head><body><div class='wrap'>",
        "<p class='back'><a href='../index.html'>&larr; _crosshair_test</a> | "
        "<a href='../../index.html'>all decks</a></p>",
        "<h1>v27 &mdash; multi-band DAO (best-S/N band per source)</h1>",
        "<p>DAO scan in every band of each source's own telescope; cyan circle and panel "
        "are drawn in the band with the highest <b>peak/bg_std</b>.</p>",
        "<p><b style='color:red'>host=red</b>, "
        "<b style='color:#0cc'>SN=cyan</b>, "
        "<b style='color:#cc0'>yellow=DAO 3&sigma; candidates</b>.</p>",
        "<div class='grid'>"]
for r in sorted(rows, key=lambda x: x["id"]):
    png = f"sn_{r['id']}_DAO_check.png"
    b = next(x for x in best_rows if x["id"] == r["id"])
    cap = f"{r['id']} &middot; best={b['band']} &middot; SNR={b['peak_snr']:.1f}"
    html.append(f"<div class='cell'><img src='{png}'>"
                f"<div class='fn'>{cap}</div></div>")
html.append("</div>")

html.append("<h2>Best-S/N band per source</h2>")
html.append("<table><thead><tr>"
            "<th>ID</th><th>tel</th><th>band</th>"
            "<th>sep_host</th><th>sep_sn</th>"
            "<th>peak</th><th>bg_std</th><th>peak/bg_std</th>"
            "<th>sharp</th><th>rnd1</th><th>rnd2</th>"
            "<th>flux</th><th>mag</th></tr></thead><tbody>")
for s in sorted(best_rows, key=lambda x: x["id"]):
    bg = s["_meta"]["bg_std"]
    html.append(f"<tr><td>{s['id']}</td><td>{s['telescope']}</td>"
                f"<td>{s['band']}</td>"
                f"<td>{s['sep_host']:.2f}&Prime;</td>"
                f"<td>{s['sep_sn']:.2f}&Prime;</td>"
                f"<td>{s['peak']:.3f}</td><td>{bg:.4f}</td>"
                f"<td>{s['peak_snr']:.2f}</td>"
                f"<td>{s['sharp']:+.2f}</td>"
                f"<td>{s['rnd1']:+.2f}</td>"
                f"<td>{s['rnd2']:+.2f}</td>"
                f"<td>{s['flux']:.2f}</td>"
                f"<td>{s['mag']:+.2f}</td></tr>")
html.append("</tbody></table>")

html.append("<h2>Full multi-band breakdown</h2>")
html.append("<table><thead><tr>"
            "<th>ID</th><th>band</th><th>sep_sn</th>"
            "<th>peak</th><th>bg_std</th><th>peak/bg_std</th>"
            "<th>sharp</th><th>rnd1</th><th>rnd2</th>"
            "<th>flux</th><th>mag</th></tr></thead><tbody>")
def _fmt(v, fmt):
    return ("&mdash;" if (not np.isfinite(v)) else (fmt % v))
for s in allband_rows:
    is_best = (s["id"], s["band"]) in best_pairs
    klass = " class='best'" if is_best else ""
    html.append(f"<tr><td{klass}>{s['id']}</td>"
                f"<td{klass}>{s['band']}</td>"
                f"<td{klass}>{_fmt(s['sep_sn'], '%.2f&Prime;')}</td>"
                f"<td{klass}>{_fmt(s['peak'], '%.3f')}</td>"
                f"<td{klass}>{_fmt(s['bg_std'], '%.4f')}</td>"
                f"<td{klass}>{_fmt(s['peak_snr'], '%.2f')}</td>"
                f"<td{klass}>{_fmt(s['sharp'], '%+.2f')}</td>"
                f"<td{klass}>{_fmt(s['rnd1'], '%+.2f')}</td>"
                f"<td{klass}>{_fmt(s['rnd2'], '%+.2f')}</td>"
                f"<td{klass}>{_fmt(s['flux'], '%.2f')}</td>"
                f"<td{klass}>{_fmt(s['mag'], '%+.2f')}</td></tr>")
html.append("</tbody></table>")
html.append("</div></body></html>")
(OUT_DIR / "index.html").write_text("\n".join(html))
print(f"wrote {OUT_DIR/'index.html'}")

# top-level link
top = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/index.html")
s = top.read_text()
if "v27/index.html" not in s and "</div></body></html>" in s:
    line = ("<div class='ver'><a href='v27/index.html'>v27</a> &mdash; "
            "multi-band DAO, best-S/N band per source &nbsp; "
            "<span class='meta'>11 PNGs</span></div>\n</div></body></html>")
    s = s.replace("</div></body></html>", line)
    top.write_text(s)
    print("added v27 to top-level index")

print("DONE")
