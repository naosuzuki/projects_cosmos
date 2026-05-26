"""Fast 5-source re-cutout: 2 JWST SN + 3 HST SN.

For each source we know the exact tiles (from sn_known34_*_index.csv) so
we skip the full tile-indexing step the production scripts do.  Just open
the specific tiles, cut, render PNG (no labels, JWST N-up rotated).
"""
import warnings
warnings.filterwarnings("ignore")

import glob
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.visualization import make_rgb, ManualInterval
from scipy.ndimage import rotate as ndi_rotate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
SOURCES = [
    # MEASURED SN positions from HST F814W centroid (2026-05-26):
    # Centroid on 1.0"-1.5" search box around user's refined eyeball hint.
    # All centroids landed within 0.05-0.18" of the hint, confirming the
    # eyeball was a reasonable starting point; these are the data-driven
    # final positions.
    #
    # NB: seq=1/2/3 (not 3/4/5) so the output PNG naming
    # sn_known34c_000{1,2,3}_*.png aligns with the polish step's
    # row-index-as-seq convention when called with a 3-row CSV.
    # The v05 PNGs at seq 0003/0004/0005 are preserved in v05/.
    dict(id=130972, seq=1, sn_ra=150.279695, sn_dec=2.041101,
         hst="052", jwst="A4",  euclid="101542818",
         host_ra=150.279917, host_dec=2.041285,
         telescope="HST"),
    dict(id=371996, seq=2, sn_ra=150.282844, sn_dec=1.932259,
         hst="040", jwst="A10", euclid="101542818",
         host_ra=150.282612, host_dec=1.932054,
         telescope="HST"),
    dict(id=471959, seq=3, sn_ra=150.278904, sn_dec=2.430962,
         hst="076", jwst="B3",  euclid="101545698",
         host_ra=150.278761, host_dec=2.430970,
         telescope="HST"),
    # Euclid NISP-only SN; NISP-H centroid 0.04" from catalog → SN essentially
    # at host position (transient sits on top of host in NISP).
    dict(id=63924,  seq=4, sn_ra=149.968179, sn_dec=2.233132,
         hst="066", jwst="A2",  euclid="101544256",
         host_ra=149.968173, host_dec=2.233122,
         telescope="EUCLID"),
]

# Aperture-photometry constants (must match measure_sn_5.py)
APER_ARCSEC      = 0.20
HOST_RING_IN_AS  = 0.25
HOST_RING_OUT_AS = 0.40

# Band catalogue: (label, kind, sci_path_template_func, err_path_template_func)
def _hst_paths(tile):
    return (f"{HST_DIR}/acs_I_030mas_{tile}_sci.fits",
            f"{HST_DIR}/acs_I_030mas_{tile}_wht.fits")
def _eu_path(tile, band):
    matches = sorted(glob.glob(f"{EU_DIR}/EUC_MER_BGSUB-MOSAIC-{band}_TILE{tile}-*.fits"))
    return (matches[-1] if matches else None, None)
def _jwst_paths(tile, band):
    return (resolve_jwst_path(tile, band), None)

BANDS_PHOT = [   # (label, kind, fn-returning(sci_path, err_path))
    ("F814W", "hst",    lambda src: _hst_paths(src["hst"])),
    ("VIS",   "euclid", lambda src: _eu_path(src["euclid"], "VIS")),
    ("Y",     "euclid", lambda src: _eu_path(src["euclid"], "NIR-Y")),
    ("J",     "euclid", lambda src: _eu_path(src["euclid"], "NIR-J")),
    ("H",     "euclid", lambda src: _eu_path(src["euclid"], "NIR-H")),
    ("F115W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f115w")),
    ("F150W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f150w")),
    ("F277W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f277w")),
    ("F444W", "jwst",   lambda src: _jwst_paths(src["jwst"], "f444w")),
]
CUTOUT_SIZE = 6.0 * u.arcsec
ID_TAG = "known34c"

OUT_HST    = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png")
OUT_EUCLID = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png")
OUT_JWST   = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png")

HST_DIR  = "/Volumes/exdisk1/data/HST/COSMOS_v2.0"
JWST_DIR = "/Volumes/exdisk1/data/JWST/COSMOS_v0.8"
EU_DIR   = "/Volumes/exdisk1/data/Euclid/COSMOS_DR1"


# ============================================================
def _draw_crosshair(ax, ny, nx, color="lime"):
    """Tiny '+' centered on the cutout (= SN position by construction)."""
    cy, cx = ny / 2 - 0.5, nx / 2 - 0.5
    gap = max(3, min(ny, nx) * 0.025)        # gap between arms
    arm = max(6, min(ny, nx) * 0.06)         # arm length
    lw  = max(1.0, min(ny, nx) / 200.0)
    ax.plot([cx, cx], [cy + gap, cy + gap + arm], color=color, lw=lw)
    ax.plot([cx, cx], [cy - gap, cy - gap - arm], color=color, lw=lw)
    ax.plot([cx + gap, cx + gap + arm], [cy, cy], color=color, lw=lw)
    ax.plot([cx - gap, cx - gap - arm], [cy, cy], color=color, lw=lw)


def render_gray(data, png_path, crosshair=True):
    minimum = 0.0001
    bw = np.where(data > minimum, data, minimum); bw = np.sqrt(bw)
    ny, nx = bw.shape
    cy, cx = ny // 2, nx // 2
    half = max(5, min(cy, cx) // 4)
    val = float(np.nanmax(bw[cy-half:cy+half, cx-half:cx+half]))
    maximum = 1.0 if val < 1.0 else val * 0.85
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
    ax.imshow(bw, origin="lower", cmap="gray", vmin=minimum, vmax=maximum)
    if crosshair: _draw_crosshair(ax, ny, nx)
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off"); fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)

def render_rgb_jwst(b_data, g_data, r_data, png_path, crosshair=True):
    minimum = 1e-4
    b = np.sqrt(np.where(b_data > minimum, b_data, minimum))
    g = np.sqrt(np.where(g_data > minimum, g_data, minimum))
    r = np.sqrt(np.where(r_data > minimum, r_data, minimum))
    maximum = 0.001
    for img in (b, g, r):
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        half = max(5, min(cy, cx) // 4)
        val = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half]))
        if val > maximum: maximum = val
    maximum = max(1.0, maximum * 0.85)
    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
    ax.imshow(rgb, origin="lower")
    ny, nx = b.shape
    if crosshair: _draw_crosshair(ax, ny, nx, color="lime")
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)

def render_rgb_nisp(b_data, g_data, r_data, png_path, crosshair=True):
    minimum = 1e-5
    b = np.sqrt(np.where(b_data > minimum, b_data, minimum))
    g = np.sqrt(np.where(g_data > minimum, g_data, minimum))
    r = np.sqrt(np.where(r_data > minimum, r_data, minimum))
    maximum = 0.001
    for img in (b, g, r):
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        half = max(5, min(cy, cx) // 4)
        val = float(np.nanmax(img[cy-half:cy+half, cx-half:cx+half]))
        if val > maximum: maximum = val
    maximum = max(1.0, maximum * 1.2)
    rgb = make_rgb(r, g, b, interval=ManualInterval(vmin=minimum, vmax=maximum))
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111)
    ax.imshow(rgb, origin="lower")
    ny, nx = b.shape
    if crosshair: _draw_crosshair(ax, ny, nx, color="lime")
    ax.set_xlim(-0.5, nx-0.5); ax.set_ylim(-0.5, ny-0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close(fig)


# ============================================================
def compute_roll_angle(wcs, ra, dec):
    x0, y0 = wcs.all_world2pix(ra, dec, 0)
    xn, yn = wcs.all_world2pix(ra, dec + 1.0/3600.0, 0)
    return float(np.degrees(np.arctan2(float(xn - x0), float(yn - y0))))


def resolve_jwst_path(jwst_tile, band):
    """Resolve JWST tile file with safe fallbacks.

    Preferred order:
      1. Uncompressed v1.0 i2d (.fits)
      2. v0.8 sci.fits (no ERR HDU but always uncompressed and intact)
      3. v1.0 i2d.fits.gz (LAST resort; gzipped, slow, and may be partial)

    We skip .gz when a v0.8 sci.fits is available because the .gz can be
    partial mid-download and astropy will crash on the corrupt block.
    """
    base = f"{JWST_DIR}/mosaic_nircam_{band}_COSMOS-Web_30mas_{jwst_tile}_v1.0_i2d.fits"
    if Path(base).exists():
        return base
    v08 = f"{JWST_DIR}/scidir/mosaic_nircam_{band}_COSMOS-Web_30mas_{jwst_tile}_v0_8_sci.fits"
    if Path(v08).exists():
        return v08
    return base + ".gz"


def resolve_euclid_path(eu_tile, band):
    matches = sorted(glob.glob(f"{EU_DIR}/EUC_MER_BGSUB-MOSAIC-{band}_TILE{eu_tile}-*.fits"))
    return matches[-1] if matches else None


_FITS_CACHE = {}   # path -> open HDUList (kept open for the script's lifetime)
_WCS_CACHE  = {}   # path -> WCS (avoids re-parsing the header per call)


def _open_cached(path):
    if path not in _FITS_CACHE:
        _FITS_CACHE[path] = fits.open(path, memmap=True)
    return _FITS_CACHE[path]


def _wcs_cached(path, hdu):
    if path not in _WCS_CACHE:
        _WCS_CACHE[path] = WCS(hdu.header)
    return _WCS_CACHE[path]


def aper_photometry(sci_path, err_path, kind, sn_ra, sn_dec):
    """Aperture photometry at (sn_ra, sn_dec).
    Returns (mag_AB, snr).  Uses the same constants as measure_sn_5.py.
    Re-uses _FITS_CACHE — OS page cache makes this near-free if cut()
    has already opened the file.
    """
    if sci_path is None:
        return None, 0.0
    try:
        h = _open_cached(sci_path)
        hdu_names = [x.name for x in h]
        if kind == "jwst" and "SCI" in hdu_names:
            sci_hdu = h["SCI"]
        else:
            sci_hdu = h[0]
        hdr = sci_hdu.header
        wcs = _wcs_cached(sci_path, sci_hdu)
    except Exception as e:
        print(f"      (open failed {sci_path}: {type(e).__name__}: {e})", flush=True)
        return None, 0.0
    try:
        pix_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    except Exception:
        pix_scale = 0.03
    aper_px  = APER_ARCSEC      / pix_scale
    ring_in  = HOST_RING_IN_AS  / pix_scale
    ring_out = HOST_RING_OUT_AS / pix_scale
    half = int(np.ceil(max(ring_out, 2.0 / pix_scale)) + 2)

    sx, sy = wcs.all_world2pix(sn_ra, sn_dec, 0)
    sx, sy = float(sx), float(sy)
    # Get shape from header so we never materialise the full image.
    ny = int(hdr.get("NAXIS2", 0)); nx = int(hdr.get("NAXIS1", 0))
    if ny == 0 or nx == 0:
        # Fallback: read shape from the memmap (cheap, no copy)
        ny, nx = sci_hdu.data.shape[-2:]
    x0 = max(0, int(sx - half)); x1 = min(nx, int(sx + half + 1))
    y0 = max(0, int(sy - half)); y1 = min(ny, int(sy + half + 1))
    # SLICE FIRST (memory-mapped read of just the small box),
    # THEN astype.  Same fix as cut() — avoids reading 25 GB JWST tiles.
    sub_raw = sci_hdu.data[y0:y1, x0:x1]
    if sub_raw.size == 0: return None, 0.0
    sub = sub_raw.astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - sx)**2 + (yy - sy)**2)
    in_aper = rr <= aper_px; in_ring = (rr >= ring_in) & (rr <= ring_out)
    n_aper = int(in_aper.sum())
    if n_aper == 0 or in_ring.sum() == 0: return None, 0.0
    host_sb = float(np.nanmedian(sub[in_ring]))
    flux = float(np.nansum(sub[in_aper])) - host_sb * n_aper
    sigma_ring = float(np.nanstd(sub[in_ring]))
    sigma_aper = sigma_ring * np.sqrt(n_aper)
    snr = flux / (sigma_aper + 1e-30)
    bunit = (hdr.get("BUNIT") or "").strip()
    if "MJy/sr" in bunit or kind == "jwst":
        pix_sr = hdr.get("PIXAR_SR") or ((pix_scale / 206265.0) ** 2)
        f_jy = flux * float(pix_sr) * 1.0e6
        mag = (-2.5 * np.log10(f_jy / 3631.0)) if f_jy > 0 else None
    elif kind == "hst":
        zp = hdr.get("ABMAG_ZP") or hdr.get("ABMAGZP") or hdr.get("MAGZP") or 25.937
        mag = (float(zp) - 2.5 * np.log10(flux)) if flux > 0 else None
    else:
        zp = hdr.get("ZP") or hdr.get("MAGZP") or hdr.get("PHOTZP") or 23.9
        mag = (float(zp) - 2.5 * np.log10(flux)) if flux > 0 else None
    return mag, float(snr)


def cut(path, pos, north_up_jwst=False):
    """Cut 6'' window with two efficiency tricks:
      - Slice from memory-mapped data FIRST, then astype on the small slice
        (avoids reading the full 500MB-30GB tile through .astype).
      - Cache open HDUList per path (shared Euclid tiles between sources).
    If north_up_jwst, rotate by -roll so North is up after cropping.
    """
    h = _open_cached(path)
    sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
    wcs = _wcs_cached(path, sci_hdu)
    # copy=True forces Cutout2D to materialise the ~80KB small region only.
    c = Cutout2D(sci_hdu.data, pos, size=CUTOUT_SIZE, wcs=wcs, copy=True)
    # Cast the small cutout to float32 (cheap):
    c.data = np.asarray(c.data, dtype=np.float32)
    if north_up_jwst:
        angle = compute_roll_angle(wcs, pos.ra.deg, pos.dec.deg)
        data = ndi_rotate(c.data, -angle, reshape=False, order=3,
                          mode="constant", cval=0.0)
        c.data = data
    return c


def main():
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-crosshair", action="store_true",
                    help="skip the simple '+' crosshair (for downstream polish)")
    ap.add_argument("--measure", action="store_true",
                    help="also do aperture photometry per band and write CSV")
    ap.add_argument("--csv", default="/tmp/sn_3.csv",
                    help="output CSV path when --measure is set")
    ap.add_argument("--input-csv", default=None,
                    help="lookup-table CSV with columns: id, telescope, sn_ra, "
                         "sn_dec, host_ra, host_dec, hst, jwst, euclid. "
                         "Replaces the hardcoded SOURCES list when provided.")
    args = ap.parse_args()
    crosshair = not args.no_crosshair

    # Lookup-table mode: read sources from CSV (for mass production)
    if args.input_csv:
        with open(args.input_csv) as fh:
            rdr = csv.DictReader(fh)
            sources = []
            for i, r in enumerate(rdr, 1):
                sources.append(dict(
                    id=int(r["id"]), seq=i,
                    sn_ra=float(r["sn_ra"]), sn_dec=float(r["sn_dec"]),
                    host_ra=float(r["host_ra"]), host_dec=float(r["host_dec"]),
                    hst=r["hst"], jwst=r["jwst"], euclid=r["euclid"],
                    telescope=r.get("telescope", "JWST").strip().upper(),
                ))
        print(f"Loaded {len(sources)} sources from {args.input_csv}", flush=True)
    else:
        sources = SOURCES

    phot_rows = [] if args.measure else None
    for src in sources:
        pos = SkyCoord(src["sn_ra"], src["sn_dec"], unit="deg", frame="icrs")
        seqn = f"{src['seq']:04d}"
        print(f"\n==== ID={src['id']} seq={seqn} ====", flush=True)
        print(f"  SN @ RA={src['sn_ra']:.6f} Dec={src['sn_dec']:.6f}", flush=True)

        # HST
        try:
            hst_path = f"{HST_DIR}/acs_I_030mas_{src['hst']}_sci.fits"
            if not Path(hst_path).exists():
                hst_path += ".gz"
            print(f"  HST F814W ({Path(hst_path).name}) ...", flush=True)
            c = cut(hst_path, pos)
            render_gray(c.data, OUT_HST / f"sn_{ID_TAG}_{seqn}_hst.png",
                        crosshair=crosshair)
        except Exception as e:
            print(f"    HST FAILED: {e}", flush=True)

        # Euclid VIS
        try:
            p = resolve_euclid_path(src["euclid"], "VIS")
            print(f"  Euclid VIS ({Path(p).name}) ...", flush=True)
            c = cut(p, pos)
            render_gray(c.data, OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_vis.png",
                        crosshair=crosshair)
        except Exception as e:
            print(f"    Euclid VIS FAILED: {e}", flush=True)

        # Euclid NISP Y/J/H
        try:
            cy_d = cut(resolve_euclid_path(src["euclid"], "NIR-Y"), pos).data
            cj_d = cut(resolve_euclid_path(src["euclid"], "NIR-J"), pos).data
            ch_d = cut(resolve_euclid_path(src["euclid"], "NIR-H"), pos).data
            print(f"  Euclid NISP Y/J/H ...", flush=True)
            render_rgb_nisp(cy_d, cj_d, ch_d,
                            OUT_EUCLID / f"sn_{ID_TAG}_{seqn}_euclid_nisp.png",
                            crosshair=crosshair)
        except Exception as e:
            print(f"    Euclid NISP FAILED: {e}", flush=True)

        # JWST jwst1 (F115/F150/F277) N-up
        try:
            c115 = cut(resolve_jwst_path(src["jwst"], "f115w"), pos, north_up_jwst=True).data
            c150 = cut(resolve_jwst_path(src["jwst"], "f150w"), pos, north_up_jwst=True).data
            c277 = cut(resolve_jwst_path(src["jwst"], "f277w"), pos, north_up_jwst=True).data
            print(f"  JWST F115/F150/F277 (N-up) ...", flush=True)
            render_rgb_jwst(c115, c150, c277,
                            OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst1.png",
                            crosshair=crosshair)
            c444 = cut(resolve_jwst_path(src["jwst"], "f444w"), pos, north_up_jwst=True).data
            print(f"  JWST F150/F277/F444 (N-up) ...", flush=True)
            render_rgb_jwst(c150, c277, c444,
                            OUT_JWST / f"sn_{ID_TAG}_{seqn}_jwst2.png",
                            crosshair=crosshair)
        except Exception as e:
            print(f"    JWST FAILED: {e}", flush=True)

        # Aperture photometry (--measure).  Re-uses the same FITS_CACHE so
        # the per-band file is already open from the cut() calls above.
        if args.measure:
            row = dict(id=src["id"],
                       telescope=src.get("telescope", "HST"),
                       host_ra=src.get("host_ra", src["sn_ra"]),
                       host_dec=src.get("host_dec", src["sn_dec"]),
                       sn_ra=src["sn_ra"], sn_dec=src["sn_dec"])
            best_band = "F814W"; best_snr = -999.0
            for label, kind, paths_fn in BANDS_PHOT:
                sci_p, err_p = paths_fn(src)
                mag, snr = aper_photometry(sci_p, err_p, kind,
                                           src["sn_ra"], src["sn_dec"])
                row[f"mag_{label}"] = -1 if mag is None else round(mag, 2)
                row[f"snr_{label}"] = round(snr, 2)
                if snr > best_snr:
                    best_snr = snr; best_band = label
            row["best_band"] = best_band
            row["best_snr"]  = round(best_snr, 2)
            row["combined_sigma"] = round(best_snr, 2)  # for HST SN, dominated by F814W
            phot_rows.append(row)
            print(f"  photometry: best={best_band}@{best_snr:.1f}σ", flush=True)

    # Write CSV with the same column order as /tmp/sn_5.csv
    if args.measure and phot_rows:
        cols = ["id","telescope","host_ra","host_dec","sn_ra","sn_dec",
                "best_band","best_snr","combined_sigma",
                "mag_F814W","mag_VIS","mag_Y","mag_J","mag_H",
                "mag_F115W","mag_F150W","mag_F277W","mag_F444W",
                "snr_F814W","snr_VIS","snr_Y","snr_J","snr_H",
                "snr_F115W","snr_F150W","snr_F277W","snr_F444W"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in phot_rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"\nWrote {args.csv}  ({len(phot_rows)} rows)", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
