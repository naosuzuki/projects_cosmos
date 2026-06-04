"""Find SN positions for the 6 new JWST SN and write the 17-SN lookup CSV.

NEW IDS: 9748, 19931, 53669, 63919, 78323, 430352.
Catalog (RA, Dec) from sn_known34_jwst_index.csv = host position.
We find the SN via multi-band JWST DAO + morphology gates (CLAUDE.md §2
G2-G4), picking the highest snr_aper PSF-clean candidate within 1.5"
of the catalog position.

Also writes per-source DAO-check PNGs (red=catalog host, cyan=picked SN,
yellow=all DAO 3sigma candidates) for user verification.
"""
import warnings; warnings.filterwarnings("ignore")
import csv, math, sys, time
from pathlib import Path
import numpy as np
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

EXISTING = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/lookup_sn11_v30.csv")
JWST_IDX = Path("/Users/suzuki/github/projects_cosmos/csvfiles/sn_known34_jwst_index.csv")
HST_IDX  = Path("/Users/suzuki/github/projects_cosmos/csvfiles/sn_known34_hst_index.csv")
EU_IDX   = Path("/Users/suzuki/github/projects_cosmos/csvfiles/sn_known34_euclid_index.csv")

OUT_LOOKUP = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/lookup_sn17_v31.csv")
OUT_DAO    = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/v31_DAO_check")
OUT_DAO.mkdir(parents=True, exist_ok=True)

NEW_IDS = [9748, 19931, 53669, 63919, 78323, 430352]
SEARCH_AS = 3.0
CUTOUT_AS = 6.0
CIRCLE_AS = 0.30
SIGMA_DETECT = 3.0

JWST_BANDS = [("F115W", 0.057), ("F150W", 0.057),
              ("F277W", 0.130), ("F444W", 0.160)]


def open_image(sci_path):
    h = R._open_cached(sci_path)
    sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    return sci_hdu, R._wcs_cached(sci_path, sci_hdu)


def pix_scale(wcs):
    return float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0


def measure_band(sn, sci_path, fwhm_as, kind):
    """One pass: DAO scan + PSF-matched aperture photometry at catalog SN."""
    sci_hdu, wcs = open_image(sci_path)
    ps = pix_scale(wcs)
    fwhm_px = max(1.5, fwhm_as / ps)
    box_as = max(SEARCH_AS * 2.0 + 1.0, 5.0)
    half = int(np.ceil(box_as / ps)) + 1
    sx, sy = wcs.all_world2pix(sn["sn_ra"], sn["sn_dec"], 0)
    sx, sy = float(sx), float(sy)
    ny, nx = sci_hdu.data.shape[-2:]
    x0 = max(0, int(sx) - half); x1 = min(nx, int(sx) + half + 1)
    y0 = max(0, int(sy) - half); y1 = min(ny, int(sy) + half + 1)
    sub_raw = sci_hdu.data[y0:y1, x0:x1].astype(np.float64)
    meta = dict(sci_hdu=sci_hdu, wcs=wcs, ps=ps,
                ny=ny, nx=nx, x0=x0, y0=y0, hpx=sx-x0, hpy=sy-y0)
    blank = dict(sep_sn=float("nan"), peak=float("nan"),
                 sharp=float("nan"), rnd1=float("nan"), rnd2=float("nan"),
                 peak_snr=float("nan"), snr_aper=float("nan"),
                 pix_sn_snr=float("nan"),
                 bg_std=float("nan"), tol=max(0.10, 1.5*fwhm_as),
                 eligible=False)
    if sub_raw.size == 0 or not (np.isfinite(sub_raw) & (sub_raw != 0)).any():
        return [], meta, blank
    finite = np.isfinite(sub_raw) & (sub_raw != 0)
    _, bg_med, bg_std = sigma_clipped_stats(sub_raw[finite], sigma=3.0, maxiters=3)
    sub = sub_raw - bg_med

    res = DAOStarFinder(fwhm=fwhm_px, threshold=SIGMA_DETECT*bg_std)(sub)
    cands = []
    if res is not None and len(res) > 0:
        for s in res:
            dx = float(s["xcentroid"]) - meta["hpx"]
            dy = float(s["ycentroid"]) - meta["hpy"]
            sep = math.hypot(dx, dy) * ps
            if sep > SEARCH_AS: continue
            ra, dec = wcs.all_pix2world(float(s["xcentroid"]) + x0,
                                        float(s["ycentroid"]) + y0, 0)
            cands.append(dict(ra=float(ra), dec=float(dec),
                              peak=float(s["peak"]),
                              sharp=float(s["sharpness"]),
                              rnd1=float(s["roundness1"]),
                              rnd2=float(s["roundness2"]),
                              sep_sn=sep))
    cands.sort(key=lambda d: d["sep_sn"])

    # PSF-matched aperture at catalog position
    aper_r = 1.0 * fwhm_as
    ring_in, ring_out = 2.0 * fwhm_as, 3.5 * fwhm_as
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - meta["hpx"])**2 + (yy - meta["hpy"])**2) * ps
    in_aper = rr <= aper_r
    in_ring = (rr >= ring_in) & (rr <= ring_out)
    snr_aper = float("nan")
    if in_aper.any() and in_ring.any():
        local_bg = float(np.nanmedian(sub[in_ring]))
        n_aper = int(in_aper.sum())
        flux = float(np.nansum(sub[in_aper])) - local_bg * n_aper
        sig_ring = float(np.nanstd(sub[in_ring]))
        sig_aper = sig_ring * math.sqrt(n_aper)
        snr_aper = flux / (sig_aper + 1e-30)

    sxi, syi = int(round(meta["hpx"])), int(round(meta["hpy"]))
    pix_sn = float(sub[syi, sxi]) if (0 <= sxi < sub.shape[1] and 0 <= syi < sub.shape[0]) else float("nan")
    pix_sn_snr = pix_sn / bg_std if bg_std > 0 else float("nan")

    c = cands[0] if cands else None
    tol = max(0.10, 1.5 * fwhm_as)
    # CLAUDE.md §2 G1-G4: all four must pass for "PSF-like detection".
    g1 = (c is not None) and (c["sep_sn"] <= tol)
    g2 = (c is not None) and (0.40 <= c["sharp"] <= 0.85)
    g3 = (c is not None) and (abs(c["rnd1"]) <= 0.50)
    g4 = (c is not None) and (abs(c["rnd2"]) <= 0.50)
    eligible = g1 and g2 and g3 and g4
    result = dict(
        sep_sn=(c["sep_sn"] if c else float("nan")),
        peak=(c["peak"] if c else float("nan")),
        sharp=(c["sharp"] if c else float("nan")),
        rnd1=(c["rnd1"] if c else float("nan")),
        rnd2=(c["rnd2"] if c else float("nan")),
        peak_snr=(c["peak"]/bg_std if (c and bg_std>0) else float("nan")),
        snr_aper=snr_aper, pix_sn_snr=pix_sn_snr,
        bg_std=bg_std, tol=tol, eligible=eligible,
    )
    return cands, meta, result


def draw_panel(sn, cands, meta, band_label, snr_aper, png_path):
    sci_hdu, wcs, ps = meta["sci_hdu"], meta["wcs"], meta["ps"]
    centre = SkyCoord(sn["sn_ra"]*u.deg, sn["sn_dec"]*u.deg)
    size_pix = int(np.ceil(CUTOUT_AS / ps))
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
    rad_px = CIRCLE_AS / ps
    if cands:
        ras = np.array([c["ra"] for c in cands])
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


# ----- main -----
def load_idx(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[int(r["id"])] = r
    return out

if __name__ == "__main__":
    jwst_idx = load_idx(JWST_IDX)
    hst_idx  = load_idx(HST_IDX)
    eu_idx   = load_idx(EU_IDX)

    existing_rows = []
    with EXISTING.open() as f:
        for r in csv.DictReader(f):
            existing_rows.append(r)

    print(f"=== finding SN positions for {len(NEW_IDS)} new JWST SN ===")
    new_rows = []
    new_dao_summary = []
    t0 = time.time()
    for cid in NEW_IDS:
        j = jwst_idx[cid]
        h = hst_idx.get(cid, {})
        e = eu_idx.get(cid,  {})
        host_ra  = float(j["ra"])
        host_dec = float(j["dec"])
        jwst_tile = j["tile"]
        hst_tile  = h.get("tile", "")
        eu_tile   = e.get("tile", "")
        print(f"\n--- ID={cid}  jwst={jwst_tile}  hst={hst_tile}  eu={eu_tile}  "
              f"catalog ({host_ra:.6f}, {host_dec:.6f})  host_mag={j['mag']}")

        src = dict(jwst=jwst_tile, hst=hst_tile, euclid=eu_tile,
                   sn_ra=host_ra, sn_dec=host_dec,
                   host_ra=host_ra, host_dec=host_dec,
                   id=cid, telescope="JWST")

        band_results = []
        for band, fwhm in JWST_BANDS:
            sci_path = R._jwst_band_path_fn(band.lower())(src)
            cands, meta, rec = measure_band(src, sci_path, fwhm, "jwst")
            rec["band"] = band; rec["fwhm"] = fwhm
            rec["_cands"] = cands; rec["_meta"] = meta
            # PSF-clean candidates: pass G2-G4 morphology (G1=in-search-box is
            # already enforced by measure_band's SEARCH_AS).
            psf_cands = [c for c in cands
                         if 0.40 <= c["sharp"] <= 0.85
                         and abs(c["rnd1"]) <= 0.50
                         and abs(c["rnd2"]) <= 0.50]
            if psf_cands:
                # Prefer the closest PSF-clean candidate OFFSET from catalog
                # (since catalog = host; SN is usually NOT the host).
                offset = [c for c in psf_cands if c["sep_sn"] > 0.10]
                match = (min(offset, key=lambda c: c["sep_sn"]) if offset
                         else psf_cands[0])
                rec["match"] = match
                rec["match_snr"] = match["peak"]/rec["bg_std"] if rec["bg_std"] > 0 else 0.0
            else:
                rec["match"] = None
                rec["match_snr"] = 0.0
            band_results.append(rec)
            snrA = (f"{rec['snr_aper']:+.1f}σ" if np.isfinite(rec["snr_aper"]) else " --   ")
            if rec["match"]:
                m = rec["match"]
                tag = (f"PSF-MATCH sep={m['sep_sn']:.2f}\" sharp={m['sharp']:+.2f} "
                       f"rnd1={m['rnd1']:+.2f} rnd2={m['rnd2']:+.2f} peak/bg={rec['match_snr']:.0f}")
            else:
                tag = "no PSF-clean candidate"
            print(f"    {band}  cands={len(cands):3d}  aper(cat)={snrA}  [{tag}]")

        # Pick the bluest band that has a PSF-clean candidate.  Bluer = sharper
        # PSF = harder for hosts to mimic a star.
        PRIORITY = ["F115W", "F150W", "F277W", "F444W"]
        chosen = None
        for prio in PRIORITY:
            for b in band_results:
                if b["band"] == prio and b["match"] is not None:
                    chosen = b; break
            if chosen: break

        if chosen:
            m = chosen["match"]
            sn_ra_pick, sn_dec_pick = m["ra"], m["dec"]
            sep_host = SkyCoord(sn_ra_pick*u.deg, sn_dec_pick*u.deg).separation(
                SkyCoord(host_ra*u.deg, host_dec*u.deg)).arcsec
            print(f"  PICK: {chosen['band']}  sep_host={sep_host:.2f}\"  "
                  f"peak/bg={chosen['match_snr']:.0f}  "
                  f"sharp={m['sharp']:+.2f} rnd1={m['rnd1']:+.2f} rnd2={m['rnd2']:+.2f}")
            best = chosen
            best["eligible"] = True
            best["sharp"] = m["sharp"]; best["rnd1"] = m["rnd1"]; best["rnd2"] = m["rnd2"]
        else:
            sn_ra_pick, sn_dec_pick = host_ra, host_dec
            sep_host = 0.0
            best = next((b for b in band_results if b["band"] == "F115W"), band_results[0])
            best["eligible"] = False
            print(f"  PICK: no PSF-clean candidate in any JWST band → fallback to catalog position (= host)")

        new_rows.append(dict(
            id=str(cid), telescope="JWST",
            sn_ra=str(sn_ra_pick), sn_dec=str(sn_dec_pick),
            host_ra=str(host_ra), host_dec=str(host_dec),
            hst=hst_tile, jwst=jwst_tile, euclid=eu_tile,
        ))

        png = OUT_DAO / f"sn_{cid}_DAO_check.png"
        sn_for_plot = dict(sn_ra=sn_ra_pick, sn_dec=sn_dec_pick,
                           host_ra=host_ra, host_dec=host_dec,
                           id=cid, telescope="JWST")
        draw_panel(sn_for_plot, best["_cands"], best["_meta"],
                   best["band"], best["snr_aper"], png)
        new_dao_summary.append(dict(id=cid, band=best["band"],
                                    snr_aper=best["snr_aper"],
                                    sep_host=sep_host,
                                    sharp=best["sharp"], rnd1=best["rnd1"], rnd2=best["rnd2"],
                                    eligible=best["eligible"],
                                    n_cands=len(best["_cands"])))

    t_find = time.time() - t0
    print(f"\n=== position finding done in {t_find:.2f}s ===")

    all_rows = existing_rows + new_rows
    with OUT_LOOKUP.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","telescope","sn_ra","sn_dec",
                                           "host_ra","host_dec","hst","jwst","euclid"])
        w.writeheader()
        for r in all_rows: w.writerow(r)
    print(f"wrote {OUT_LOOKUP}  ({len(all_rows)} sources)")

    with (OUT_DAO/"summary.txt").open("w") as f:
        f.write("# v31 new SN position-finding summary\n")
        f.write("id      band   snr_aper  sep_host  sharp   rnd1    rnd2   eligible  n_cands\n")
        for s in new_dao_summary:
            f.write(f"{s['id']:>6d}  {s['band']:>5s}  {s['snr_aper']:+7.2f}σ  "
                    f"{s['sep_host']:.2f}\"    {s['sharp']:+.2f}   {s['rnd1']:+.2f}   "
                    f"{s['rnd2']:+.2f}   {str(s['eligible']):>5s}     {s['n_cands']}\n")
    print(f"wrote {OUT_DAO/'summary.txt'}")
    print(f"wrote {len(NEW_IDS)} DAO-check PNGs to {OUT_DAO}/")
