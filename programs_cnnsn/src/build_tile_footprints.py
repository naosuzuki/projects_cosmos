"""Build a TILE FOOTPRINT table: tile → sky bounding box (from WCS HEADERS only,
zero pixel I/O). This is the scalable coordinate→tile lookup.

Why: extracting cutouts for arbitrary positions (training SNe, future millions
of candidates) must NOT open every FITS to test each point. With this table,
any (RA,Dec)→tile is an in-memory bbox test; we then open each FITS exactly
once, only for tiles that actually contain a requested point.

Reads ONLY the FITS header (WCS + NAXIS) per file — milliseconds each, no
pixel data touched. One file per (survey, band, tile).

Output: csvfiles_sn/tile_footprints.parquet with columns:
  survey, band, tile, path, naxis1, naxis2,
  ra_min, ra_max, dec_min, dec_max,        # axis-aligned sky bbox (deg)
  crval1, crval2, cd11, cd12, cd21, cd22, crpix1, crpix2   # enough to rebuild WCS

Usage: python build_tile_footprints.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, glob, time, re
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R

OUT = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tile_footprints.parquet")
HST_DIR = R.HST_DIR
JWST_DIR = "/Volumes/exdisk1/data/JWST/COSMOS_v0.8"
EUCLID_DIR = "/Volumes/exdisk1/data/Euclid/COSMOS_DR1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def footprint_from_header(path):
    """Open header only; return (naxis1,naxis2, sky bbox, wcs params) or None."""
    try:
        with fits.open(path, memmap=True) as h:
            hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
            hdr = hdu.header
            # NAXIS from header (no data read)
            n1 = int(hdr.get("NAXIS1") or 0); n2 = int(hdr.get("NAXIS2") or 0)
            if n1 == 0 or n2 == 0:
                return None
            wcs = WCS(hdr)
            # four corners (0-indexed pixel)
            corners = np.array([[0,0],[n1-1,0],[0,n2-1],[n1-1,n2-1]], float)
            sky = wcs.all_pix2world(corners[:,0], corners[:,1], 0)
            ra = np.asarray(sky[0], float); dec = np.asarray(sky[1], float)
            cd = wcs.wcs.cd if wcs.wcs.has_cd() else None
            if cd is None:
                # build from PC*CDELT
                cdelt = wcs.wcs.cdelt; pc = wcs.wcs.get_pc()
                cd = np.array([[pc[0,0]*cdelt[0], pc[0,1]*cdelt[0]],
                               [pc[1,0]*cdelt[1], pc[1,1]*cdelt[1]]])
            crval = wcs.wcs.crval; crpix = wcs.wcs.crpix
            return dict(naxis1=n1, naxis2=n2,
                        ra_min=float(ra.min()), ra_max=float(ra.max()),
                        dec_min=float(dec.min()), dec_max=float(dec.max()),
                        crval1=float(crval[0]), crval2=float(crval[1]),
                        cd11=float(cd[0,0]), cd12=float(cd[0,1]),
                        cd21=float(cd[1,0]), cd22=float(cd[1,1]),
                        crpix1=float(crpix[0]), crpix2=float(crpix[1]))
    except Exception as e:
        log(f"  header read failed {Path(path).name}: {e}")
        return None


def enumerate_files():
    """Yield (survey, band, tile, path) for every mosaic, parsing tile from name."""
    out = []
    # HST: acs_I_030mas_<tile>_sci.fits
    for p in sorted(glob.glob(f"{HST_DIR}/acs_I_030mas_*_sci.fits")):
        m = re.search(r"acs_I_030mas_(.+?)_sci\.fits$", os.path.basename(p))
        if m: out.append(("hst","F814W",m.group(1),p))
    # JWST: mosaic_nircam_<band>_COSMOS-Web_30mas_<tile>_v1.0_i2d.fits
    for p in sorted(glob.glob(f"{JWST_DIR}/mosaic_nircam_*_30mas_*_i2d.fits")):
        m = re.search(r"mosaic_nircam_(f\d+w)_COSMOS-Web_30mas_(.+?)_v[\d.]+_i2d\.fits$",
                      os.path.basename(p))
        if m: out.append(("jwst", m.group(1).upper(), m.group(2), p))
    # Euclid: EUC_MER_BGSUB-MOSAIC-<BAND>_TILE<tile>-...fits
    for p in sorted(glob.glob(f"{EUCLID_DIR}/EUC_MER_BGSUB-MOSAIC-*_TILE*.fits")):
        m = re.search(r"BGSUB-MOSAIC-(VIS|NIR-[YJH])_TILE(\d+)-", os.path.basename(p))
        if m: out.append(("euclid", m.group(1), m.group(2), p))
    return out


def main():
    t0 = time.time()
    log("=== build tile footprints (header-only, no pixel I/O) ===")
    files = enumerate_files()
    from collections import Counter
    log(f"found {len(files)} mosaics: {dict(Counter(s for s,_,_,_ in files))}")
    rows = []
    for i,(survey,band,tile,path) in enumerate(files,1):
        fp = footprint_from_header(path)
        if fp is None: continue
        fp.update(survey=survey, band=band, tile=tile, path=path)
        rows.append(fp)
        if i % 40 == 0 or i == len(files):
            log(f"  {i}/{len(files)} headers read ({time.time()-t0:.1f}s)")
    cols = ["survey","band","tile","path","naxis1","naxis2",
            "ra_min","ra_max","dec_min","dec_max",
            "crval1","crval2","cd11","cd12","cd21","cd22","crpix1","crpix2"]
    tbl = pa.table({c: [r[c] for r in rows] for c in cols})
    pq.write_table(tbl, OUT, compression="zstd")
    log(f"wrote {OUT}: {len(rows)} tile-band footprints in {time.time()-t0:.1f}s")
    # quick coverage summary
    for s in ("hst","jwst","euclid"):
        sr=[r for r in rows if r["survey"]==s]
        if sr:
            log(f"  {s}: {len(sr)} files, RA [{min(r['ra_min'] for r in sr):.3f},"
                f"{max(r['ra_max'] for r in sr):.3f}] Dec [{min(r['dec_min'] for r in sr):.3f},"
                f"{max(r['dec_max'] for r in sr):.3f}]")


if __name__ == "__main__":
    main()
