"""v29 multi-band DAO with bug fix.

Bug found in v27: best-band picker used `peak/bg_std` of the
DAO candidate closest to the catalog SN position, regardless of how far
that candidate actually was.  For ID=246188 in F277W, the closest DAO
was on the HOST GALAXY at sep_sn=0.33" (5x the F115W PSF), so the picker
compared "host peak in F277W" against "SN peak in F115W" and picked
F277W — wrong band.

Two-condition fix:
  PRIMARY PICKER:  snr_aper  =  PSF-matched aperture S/N at the catalog
                   SN position.  This is measured AT the SN, so a host
                   detected nearby in a redder band cannot hijack it.
  ELIGIBILITY:     a band is eligible as "best" only if DAO found a
                   candidate within sep_sn <= max(0.10", 1.5*FWHM_band).
                   Otherwise the band probably picked up the host or a
                   neighbour and we don't trust its morphology.
  FALLBACK:        if NO band passes the eligibility gate, pick the
                   highest snr_aper anyway (so a non-detection still
                   gets a band assignment).

Aperture sizes are PSF-matched (= SNR-optimal):
    aper_r   = 1.0 * FWHM
    ring_in  = 2.0 * FWHM
    ring_out = 3.5 * FWHM
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
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/v29")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-telescope band lists with PSF FWHM (arcsec) and path function.
TELESCOPE_BANDS = {
    "HST":    [("F814W", 0.134, lambda s: R._hst_path_fn(s),                "hst")],
    "JWST":   [("F115W", 0.057, lambda s: R._jwst_band_path_fn("f115w")(s), "jwst"),
               ("F150W", 0.057, lambda s: R._jwst_band_path_fn("f150w")(s), "jwst"),
               ("F277W", 0.130, lambda s: R._jwst_band_path_fn("f277w")(s), "jwst"),
               ("F444W", 0.160, lambda s: R._jwst_band_path_fn("f444w")(s), "jwst")],
    "EUCLID": [("NIR-Y", 0.524, lambda s: R._eu_band_path_fn("NIR-Y")(s),   "euclid"),
               ("NIR-J", 0.537, lambda s: R._eu_band_path_fn("NIR-J")(s),   "euclid"),
               ("NIR-H", 0.567, lambda s: R._eu_band_path_fn("NIR-H")(s),   "euclid")],
}

CUTOUT_ARCSEC = 6.0       # plot box
SEARCH_ARCSEC = 3.0       # DAO box around SN position
SIGMA_DETECT  = 3.0
CIRCLE_ARCSEC = 0.30


def open_image(sci_path):
    h = R._open_cached(sci_path)
    sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    return sci_hdu, R._wcs_cached(sci_path, sci_hdu)


def pix_scale(wcs):
    return float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0


def measure_band(sn, sci_path, fwhm_as, kind):
    """One pass through a band: DAO scan + aperture phot at catalog SN.

    Returns:
        cands     list of DAO dicts within SEARCH_ARCSEC of catalog SN
        meta      dict for plotting (sci_hdu, wcs, ps, ...)
        result    band-level result dict:
            sep_sn       arcsec from catalog SN to closest DAO (nan if none)
            peak,sharp,rnd1,rnd2,flux,mag   from that closest DAO (or nan)
            peak_snr     peak / bg_std       (single-pixel SNR proxy)
            pix_sn       pixel value at catalog SN (background-subtracted)
            pix_sn_snr   pix_sn / bg_std     (direct per-pixel SNR at catalog SN)
            snr_aper     PSF-matched aperture SNR at catalog SN  (PRIMARY metric)
            mag_aper     PSF-matched aperture AB mag
            bg_std       local sigma-clipped background std
            eligible     True if sep_sn <= max(0.10", 1.5*fwhm) AND has a DAO
    """
    sci_hdu, wcs = open_image(sci_path)
    ps = pix_scale(wcs)
    fwhm_px = max(1.5, fwhm_as / ps)
    box_as = max(SEARCH_ARCSEC * 2.0 + 1.0, 5.0)
    half = int(np.ceil(box_as / ps)) + 1
    sx, sy = wcs.all_world2pix(sn["sn_ra"], sn["sn_dec"], 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci_hdu.data.shape[-2:]
    x0 = max(0, int(sx) - half); x1 = min(nx, int(sx) + half + 1)
    y0 = max(0, int(sy) - half); y1 = min(ny, int(sy) + half + 1)
    sub_raw = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)
    meta = dict(sci_hdu=sci_hdu, wcs=wcs, ps=ps,
                ny=ny, nx=nx, x0=x0, y0=y0, hpx=sx-x0, hpy=sy-y0)
    res_blank = dict(sep_sn=float("nan"), peak=float("nan"),
                     sharp=float("nan"), rnd1=float("nan"), rnd2=float("nan"),
                     flux=float("nan"), mag=float("nan"),
                     peak_snr=float("nan"),
                     pix_sn=float("nan"), pix_sn_snr=float("nan"),
                     snr_aper=float("nan"), mag_aper=float("nan"),
                     bg_std=float("nan"), eligible=False)
    if sub_raw.size == 0: return [], meta, res_blank
    finite = np.isfinite(sub_raw) & (sub_raw != 0)
    if not finite.any(): return [], meta, res_blank
    _, bg_med, bg_std = sigma_clipped_stats(sub_raw[finite], sigma=3.0, maxiters=3)

    # ----- DAO scan -----
    sub = sub_raw - bg_med
    res = DAOStarFinder(fwhm=fwhm_px, threshold=SIGMA_DETECT*bg_std)(sub)
    cands = []
    if res is not None and len(res) > 0:
        for s in res:
            dx = float(s["xcentroid"]) - meta["hpx"]
            dy = float(s["ycentroid"]) - meta["hpy"]
            sep = math.hypot(dx, dy) * ps
            if sep > SEARCH_ARCSEC: continue
            ra, dec = wcs.all_pix2world(float(s["xcentroid"]) + x0,
                                        float(s["ycentroid"]) + y0, 0)
            cands.append(dict(ra=float(ra), dec=float(dec),
                              peak=float(s["peak"]),
                              sharp=float(s["sharpness"]),
                              rnd1=float(s["roundness1"]),
                              rnd2=float(s["roundness2"]),
                              flux=float(s["flux"]),
                              mag=float(s["mag"]),
                              sep_sn=sep))
    cands.sort(key=lambda d: d["sep_sn"])

    # ----- pixel value at catalog SN -----
    sxi, syi = int(round(meta["hpx"])), int(round(meta["hpy"]))
    if 0 <= sxi < sub.shape[1] and 0 <= syi < sub.shape[0]:
        pix_sn = float(sub[syi, sxi])
    else:
        pix_sn = float("nan")

    # ----- PSF-matched aperture photometry at catalog SN -----
    aper_r   = 1.0 * fwhm_as
    ring_in  = 2.0 * fwhm_as
    ring_out = 3.5 * fwhm_as
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - meta["hpx"])**2 + (yy - meta["hpy"])**2) * ps
    in_aper = rr <= aper_r
    in_ring = (rr >= ring_in) & (rr <= ring_out)
    snr_aper = float("nan"); mag_aper = float("nan")
    if in_aper.any() and in_ring.any():
        local_bg = float(np.nanmedian(sub[in_ring]))   # already bg-sub but ring may have residual
        n_aper = int(in_aper.sum())
        flux_aper = float(np.nansum(sub[in_aper])) - local_bg * n_aper
        sig_ring  = float(np.nanstd(sub[in_ring]))
        sig_aper  = sig_ring * math.sqrt(n_aper)
        snr_aper  = flux_aper / (sig_aper + 1e-30)
        # AB mag from header
        hdr = sci_hdu.header
        bunit = (hdr.get("BUNIT") or "").strip()
        if "MJy/sr" in bunit or kind == "jwst":
            pix_sr = hdr.get("PIXAR_SR") or ((ps / 206265.0) ** 2)
            f_jy = flux_aper * float(pix_sr) * 1.0e6
            mag_aper = (-2.5 * np.log10(f_jy / 3631.0)) if f_jy > 0 else float("nan")
        elif kind == "hst":
            zp = hdr.get("ABMAG_ZP") or hdr.get("ABMAGZP") or hdr.get("MAGZP") or 25.937
            mag_aper = (float(zp) - 2.5 * np.log10(flux_aper)) if flux_aper > 0 else float("nan")
        else:  # euclid
            zp = hdr.get("ZP") or hdr.get("MAGZP") or hdr.get("PHOTZP") or 23.9
            mag_aper = (float(zp) - 2.5 * np.log10(flux_aper)) if flux_aper > 0 else float("nan")

    # ----- best DAO row + eligibility -----
    if cands:
        c = cands[0]
        sep_sn = c["sep_sn"]
    else:
        c = None; sep_sn = float("nan")
    tol = max(0.10, 1.5 * fwhm_as)
    eligible = (c is not None) and (sep_sn <= tol)

    result = dict(
        sep_sn=sep_sn,
        peak=(c["peak"] if c else float("nan")),
        sharp=(c["sharp"] if c else float("nan")),
        rnd1=(c["rnd1"] if c else float("nan")),
        rnd2=(c["rnd2"] if c else float("nan")),
        flux=(c["flux"] if c else float("nan")),
        mag=(c["mag"]  if c else float("nan")),
        peak_snr=(c["peak"]/bg_std if (c and bg_std>0) else float("nan")),
        pix_sn=pix_sn,
        pix_sn_snr=(pix_sn/bg_std if bg_std > 0 else float("nan")),
        snr_aper=snr_aper, mag_aper=mag_aper,
        bg_std=float(bg_std), tol=tol, eligible=eligible,
    )
    return cands, meta, result


def draw_panel(sn, cands, meta, band_label, snr_aper, png_path):
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
            f"sep_host={sep_host:.2f}\"  SNR_aper={snr_aper:.1f}  DAO={len(cands)}",
            color="yellow", fontsize=9, ha="left", va="top",
            family="monospace")
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


# ─── load sources ──────────────────────────────────────────────────
rows = []
with SRC_CSV.open() as f:
    for r in csv.DictReader(f):
        r["id"]      = int(r["id"])
        r["sn_ra"]   = float(r["sn_ra"]);   r["sn_dec"]   = float(r["sn_dec"])
        r["host_ra"] = float(r["host_ra"]); r["host_dec"] = float(r["host_dec"])
        rows.append(r)
print(f"loaded {len(rows)} sources from {SRC_CSV}")

# ─── multi-band scan + measure ─────────────────────────────────────
t0_all = time.time()
allband = []   # one rec per (source, band)
best_choice = {}   # id -> dict for best-band record + cands + meta

for r in rows:
    tel = r["telescope"].upper()
    bands = TELESCOPE_BANDS.get(tel)
    if bands is None:
        print(f"  {r['id']}: unknown telescope {tel}; skip")
        continue
    per_band = []
    for band, fwhm_as, path_fn, kind in bands:
        sci_path = path_fn(r)
        t0 = time.time()
        cands, meta, rec = measure_band(r, sci_path, fwhm_as, kind)
        t_meas = time.time() - t0
        rec.update(id=r["id"], telescope=tel, band=band, fwhm=fwhm_as,
                   t_meas=t_meas, n_cands=len(cands),
                   _cands=cands, _meta=meta)
        per_band.append(rec)
        allband.append(rec)

    # --- best-band picker (BUG-FIXED) ---
    eligible = [b for b in per_band if b["eligible"] and np.isfinite(b["snr_aper"])]
    pool = eligible if eligible else [b for b in per_band if np.isfinite(b["snr_aper"])]
    if pool:
        best = max(pool, key=lambda b: b["snr_aper"])
    else:
        best = per_band[0]
    best_choice[r["id"]] = best

t_scan = time.time() - t0_all

# ─── render PNGs in best band per source ───────────────────────────
t0 = time.time()
for r in rows:
    b = best_choice[r["id"]]
    png = OUT_DIR / f"sn_{r['id']}_DAO_check.png"
    draw_panel(r, b["_cands"], b["_meta"], b["band"], b["snr_aper"], png)
t_png = time.time() - t0

# ─── print best-band table ─────────────────────────────────────────
print("\n=== BEST-S/N BAND per source (BUG-FIXED picker) ===")
print(f"{'ID':>7} {'TEL':>6} {'BAND':>5} {'sep_sn':>7} {'pix_snr':>8} "
      f"{'snr_aper':>9} {'mag_aper':>9} {'peak/bg':>9} "
      f"{'sharp':>6} {'rnd1':>7} {'rnd2':>7} elig?")
print("-" * 110)
for r in sorted(rows, key=lambda x: x["id"]):
    b = best_choice[r["id"]]
    print(f"{r['id']:>7d} {b['telescope']:>6} {b['band']:>5} "
          f"{b['sep_sn']:>6.2f}\" {b['pix_sn_snr']:>+7.1f}σ "
          f"{b['snr_aper']:>+8.2f}σ {b['mag_aper']:>+9.2f} {b['peak_snr']:>9.2f} "
          f"{b['sharp']:>+6.2f} {b['rnd1']:>+7.2f} {b['rnd2']:>+7.2f}  {b['eligible']}")

# ─── print full multi-band breakdown ───────────────────────────────
print("\n=== FULL MULTI-BAND breakdown ===")
print(f"{'ID':>7} {'BAND':>5} {'sep_sn':>7} {'tol':>6} "
      f"{'pix_snr':>8} {'snr_aper':>9} {'mag_aper':>9} "
      f"{'peak/bg':>9} {'sharp':>6} {'rnd1':>7} {'rnd2':>7}  BEST?")
print("-" * 120)
best_pairs = {(b["id"], b["band"]) for b in best_choice.values()}
for s in allband:
    flag = "  <==" if (s["id"], s["band"]) in best_pairs else ""
    el = "" if s["eligible"] else "  (sep>tol)"
    def f(v, fmt, w=8):  # noqa
        return ("     nan" if not np.isfinite(v) else (fmt % v))
    print(f"{s['id']:>7d} {s['band']:>5} {f(s['sep_sn'], '%5.2f\"')} {f(s['tol'], '%5.2f\"')} "
          f"{f(s['pix_sn_snr'], '%+7.1fσ')} {f(s['snr_aper'], '%+8.2fσ')} "
          f"{f(s['mag_aper'], '%+9.2f')} {f(s['peak_snr'], '%9.2f')} "
          f"{f(s['sharp'], '%+6.2f')} {f(s['rnd1'], '%+7.2f')} {f(s['rnd2'], '%+7.2f')}"
          f"{flag}{el}")

print(f"\nscans={t_scan:.2f}s  renders={t_png:.2f}s  total={t_scan+t_png:.2f}s")

# ─── csv outputs ───────────────────────────────────────────────────
cols_best = ["id","telescope","band","fwhm","sep_sn","tol","pix_sn_snr",
             "snr_aper","mag_aper","peak_snr","sharp","rnd1","rnd2",
             "flux","mag","bg_std","eligible","n_cands"]
with (OUT_DIR/"dao_params_best.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols_best)
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["id"]):
        b = best_choice[r["id"]]
        out = {k: b.get(k, "") for k in cols_best}
        for k in ("fwhm","sep_sn","tol","pix_sn_snr","snr_aper","mag_aper",
                  "peak_snr","sharp","rnd1","rnd2","flux","mag","bg_std"):
            v = b[k]
            out[k] = (round(float(v), 4) if np.isfinite(v) else "")
        w.writerow(out)

with (OUT_DIR/"dao_params_allbands.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols_best)
    w.writeheader()
    for s in allband:
        out = {k: s.get(k, "") for k in cols_best}
        for k in ("fwhm","sep_sn","tol","pix_sn_snr","snr_aper","mag_aper",
                  "peak_snr","sharp","rnd1","rnd2","flux","mag","bg_std"):
            v = s[k]
            out[k] = (round(float(v), 4) if np.isfinite(v) else "")
        w.writerow(out)

# ─── index.html ────────────────────────────────────────────────────
def fmt(v, p="%.2f"):
    return "&mdash;" if not np.isfinite(v) else (p % v)

html = ["<!doctype html><html><head><title>v29 — multi-band DAO (bug-fixed picker)</title>",
        "<style>body{font-family:serif;background:#fafafa}",
        ".wrap{max-width:1500px;margin:0 auto;padding:20px}",
        ".back{font-size:13px;color:#666}",
        ".grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}",
        ".cell img{width:100%;display:block;border:1px solid #ddd}",
        ".cell .fn{font-family:monospace;font-size:10px;color:#999;text-align:center;margin-top:3px}",
        "table{border-collapse:collapse;font-family:monospace;font-size:11px;margin-top:18px}",
        "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}",
        "th{background:#eee} td.best{background:#fff5b0}",
        "td.bad{background:#fdd}",
        "</style></head><body><div class='wrap'>",
        "<p class='back'><a href='../index.html'>&larr; _crosshair_test</a> | "
        "<a href='../../index.html'>all decks</a></p>",
        "<h1>v29 &mdash; multi-band DAO (best-band picker bug-fixed)</h1>",
        "<p>Picker = highest <b>snr_aper</b> at the catalog SN position among bands where "
        "DAO has a candidate within <b>sep_sn &le; max(0.10&Prime;, 1.5&times;FWHM)</b>.  "
        "snr_aper is measured AT the catalog SN coordinate using a PSF-matched aperture "
        "(<i>r</i>=1&times;FWHM, ring 2&ndash;3.5&times;FWHM), so a host detected in a redder "
        "band cannot hijack the choice.</p>",
        "<p><b style='color:red'>host=red</b>, "
        "<b style='color:#0cc'>SN=cyan</b>, "
        "<b style='color:#cc0'>yellow=DAO 3&sigma; candidates</b>.</p>",
        "<div class='grid'>"]
for r in sorted(rows, key=lambda x: x["id"]):
    b = best_choice[r["id"]]
    png = f"sn_{r['id']}_DAO_check.png"
    cap = f"{r['id']} &middot; best={b['band']} &middot; snr_aper={b['snr_aper']:.1f}"
    html.append(f"<div class='cell'><img src='{png}'><div class='fn'>{cap}</div></div>")
html.append("</div>")

# best-band table
html.append("<h2>Best-S/N band per source</h2>")
html.append("<table><thead><tr>"
            "<th>ID</th><th>tel</th><th>band</th>"
            "<th>sep_sn</th><th>pix_snr</th>"
            "<th>snr_aper</th><th>mag_aper</th>"
            "<th>peak/bg_std</th>"
            "<th>sharp</th><th>rnd1</th><th>rnd2</th>"
            "</tr></thead><tbody>")
for r in sorted(rows, key=lambda x: x["id"]):
    b = best_choice[r["id"]]
    html.append(f"<tr><td>{r['id']}</td><td>{b['telescope']}</td>"
                f"<td>{b['band']}</td>"
                f"<td>{fmt(b['sep_sn'])}&Prime;</td>"
                f"<td>{fmt(b['pix_sn_snr'], '%+.1f')}&sigma;</td>"
                f"<td>{fmt(b['snr_aper'], '%+.2f')}&sigma;</td>"
                f"<td>{fmt(b['mag_aper'], '%+.2f')}</td>"
                f"<td>{fmt(b['peak_snr'])}</td>"
                f"<td>{fmt(b['sharp'], '%+.2f')}</td>"
                f"<td>{fmt(b['rnd1'], '%+.2f')}</td>"
                f"<td>{fmt(b['rnd2'], '%+.2f')}</td></tr>")
html.append("</tbody></table>")

# multi-band table
html.append("<h2>Full multi-band breakdown (one row per (SN, band))</h2>")
html.append("<table><thead><tr>"
            "<th>ID</th><th>band</th>"
            "<th>sep_sn</th><th>tol</th>"
            "<th>pix_snr</th><th>snr_aper</th><th>mag_aper</th>"
            "<th>peak/bg_std</th>"
            "<th>sharp</th><th>rnd1</th><th>rnd2</th>"
            "<th>eligible?</th></tr></thead><tbody>")
for s in allband:
    is_best = (s["id"], s["band"]) in best_pairs
    klass = " class='best'" if is_best else ""
    bad   = " class='bad'"  if (not s["eligible"]) else ""
    elig_txt = "yes" if s["eligible"] else "no"
    html.append(
        f"<tr>"
        f"<td{klass or bad}>{s['id']}</td>"
        f"<td{klass or bad}>{s['band']}</td>"
        f"<td{klass or bad}>{fmt(s['sep_sn'])}&Prime;</td>"
        f"<td{klass or bad}>{fmt(s['tol'])}&Prime;</td>"
        f"<td{klass or bad}>{fmt(s['pix_sn_snr'], '%+.1f')}&sigma;</td>"
        f"<td{klass or bad}>{fmt(s['snr_aper'], '%+.2f')}&sigma;</td>"
        f"<td{klass or bad}>{fmt(s['mag_aper'], '%+.2f')}</td>"
        f"<td{klass or bad}>{fmt(s['peak_snr'])}</td>"
        f"<td{klass or bad}>{fmt(s['sharp'], '%+.2f')}</td>"
        f"<td{klass or bad}>{fmt(s['rnd1'], '%+.2f')}</td>"
        f"<td{klass or bad}>{fmt(s['rnd2'], '%+.2f')}</td>"
        f"<td{klass or bad}>{elig_txt}</td></tr>")
html.append("</tbody></table>")
html.append("</div></body></html>")
(OUT_DIR / "index.html").write_text("\n".join(html))
print(f"\nwrote {OUT_DIR/'index.html'}")
print(f"wrote {OUT_DIR/'dao_params_best.csv'}")
print(f"wrote {OUT_DIR/'dao_params_allbands.csv'}")

# top-level link
top = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/index.html")
s = top.read_text()
if "v29/index.html" not in s and "</div></body></html>" in s:
    line = ("<div class='ver'><a href='v29/index.html'>v29</a> &mdash; "
            "multi-band DAO, BUG-FIXED best-band picker (snr_aper + sep_sn gate) &nbsp; "
            "<span class='meta'>11 PNGs</span></div>\n</div></body></html>")
    s = s.replace("</div></body></html>", line)
    top.write_text(s)
    print("added v29 to top-level index")

print("DONE")
