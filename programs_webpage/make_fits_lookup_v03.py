"""Step 2 of SN hunt: per-source FITS tile lookup.

For each SN candidate row, determine which HST, JWST, and Euclid tile
contains its RA/Dec. Tile IDs only — full paths are reconstructable via
recut_3_hst.resolve_*_path() at use time (those resolvers already know
the A10 F115W exception and Euclid superseded handling).

Input:  csvfiles_sn/sn_candidates_v03.parquet   (1,117,776 rows)
Output: csvfiles_sn/fits_lookup_v03.parquet     (same rows)
        csvfiles_sn/fits_lookup_v03_summary.txt

Output columns:
  primary_id (str)
  primary_source (str)
  ra (float64)       best of cat_ra_<mission>, priority hst>jwst>vis>nisp
  dec (float64)
  tile_hst (str)     e.g. "052"   (null if outside HST coverage)
  tile_jwst (str)    e.g. "A4"    (null if outside JWST coverage)
  tile_euclid (str)  e.g. "101542818"
  in_hst (bool)
  in_jwst (bool)
  in_euclid (bool)

Method per survey:
  1. Read every tile FITS header (~ms each, ~5 sec total) → WCS + NAXIS.
  2. For each source, transform RA/Dec to pixel coords using each tile's
     WCS. The tile owns the source iff 0 ≤ x ≤ NAXIS1-1 and same for y.
     This is exact (no rect-on-sphere approximation, handles RA wrap) and
     vectorisable.
  3. On overlap (rare), pick tile whose center is closest to the source.

Per-survey assumptions:
  HST   — one mosaic per field; F814W only.
  JWST  — 4 NIRCam filters per tile are co-spatial (COSMOS-Web pipeline),
          use F115W footprints. Tile IDs A1..B10. For A10 F115W only,
          fall back to scidir/*_v0_8_sci.fits per master CLAUDE.md.
  Euclid— VIS + NIR-Y/J/H per tile are co-spatial (MER), use VIS
          footprints. 60 unique tiles (May-2025 superseded files
          excluded — they live in superseded_may2025/).
"""
import sys, time, glob
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from astropy.io import fits
from astropy.wcs import WCS
import warnings
warnings.filterwarnings("ignore")

IN_PATH  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/sn_candidates_v03.parquet")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
OUT_PARQ = OUT_DIR / "fits_lookup_v03.parquet"
OUT_LOG  = OUT_DIR / "fits_lookup_v03_summary.txt"

HST_DIR  = Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0")
JWST_DIR = Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8")
EU_DIR   = Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1")


def survey_tiles(survey_name, pattern_dir, glob_pat, tile_re, fallback=None):
    """Build a footprint table: [(tile_id, wcs, n1, n2, center_ra, center_dec), ...]
    `tile_re`: compiled regex with one group capturing the tile id."""
    out = []
    paths = sorted(pattern_dir.glob(glob_pat))
    # Drop macOS resource forks
    paths = [p for p in paths if not p.name.startswith("._")]
    seen = {}
    for p in paths:
        m = tile_re.search(p.name)
        if not m: continue
        tile = m.group(1)
        if tile in seen: continue   # one path per tile
        seen[tile] = p
    # Apply fallback (e.g. A10 F115W → scidir/v0_8_sci)
    if fallback:
        for tile, alt_path in fallback.items():
            if tile not in seen and alt_path.exists():
                seen[tile] = alt_path
    for tile, p in seen.items():
        try:
            with fits.open(p, memmap=True) as h:
                sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
                hdr = sci_hdu.header
                wcs = WCS(hdr)
                n1 = int(hdr["NAXIS1"]); n2 = int(hdr["NAXIS2"])
            # Tile center in RA/Dec
            cra, cdec = wcs.all_pix2world([(n1-1)/2.0], [(n2-1)/2.0], 0)
            out.append(dict(tile=tile, wcs=wcs, n1=n1, n2=n2,
                            center_ra=float(cra[0]), center_dec=float(cdec[0]),
                            path=str(p)))
        except Exception as e:
            print(f"  [{survey_name}] header read failed for {p.name}: {e}", flush=True)
    print(f"  {survey_name}: {len(out)} tiles indexed", flush=True)
    return out


def assign_tiles(ra, dec, tiles, survey_name):
    """Vectorised tile assignment.
    Returns:
       tile_ids (object array, dtype=str/None)
       in_cov  (bool array)
    """
    n = len(ra)
    tile_ids = np.full(n, None, dtype=object)
    in_cov   = np.zeros(n, dtype=bool)
    # Distance of each source from each tile center (only used as tiebreaker)
    # Sources sometimes fall into multiple overlapping tiles; pick the one
    # whose center is closest (in degrees, no cosine correction needed for
    # tiebreaker purposes).
    best_dist = np.full(n, np.inf)
    finite = np.isfinite(ra) & np.isfinite(dec)
    if not finite.any():
        return tile_ids, in_cov
    ra_f  = ra[finite]
    dec_f = dec[finite]
    idx   = np.where(finite)[0]
    for ti, tile in enumerate(tiles):
        wcs = tile["wcs"]
        try:
            x, y = wcs.all_world2pix(ra_f, dec_f, 0)
        except Exception:
            continue
        inside = (x >= 0) & (x <= tile["n1"] - 1) & (y >= 0) & (y <= tile["n2"] - 1)
        if not inside.any():
            continue
        # candidates for assignment
        cidx = idx[inside]
        # distance to tile center (deg^2; cheap)
        dra  = ra[cidx]  - tile["center_ra"]
        ddec = dec[cidx] - tile["center_dec"]
        # rough — fine for tiebreaker
        d2 = dra*dra + ddec*ddec
        update = d2 < best_dist[cidx]
        if update.any():
            cidx_u = cidx[update]
            tile_ids[cidx_u] = tile["tile"]
            best_dist[cidx_u] = d2[update]
            in_cov[cidx_u]    = True
    return tile_ids, in_cov


def best_radec_per_row(t):
    """Pick best (ra, dec) per row: priority hst > jwst > vis > nisp."""
    cols_r = ["cat_ra_hst","cat_ra_jwst","cat_ra_vis","cat_ra_nisp"]
    cols_d = ["cat_dec_hst","cat_dec_jwst","cat_dec_vis","cat_dec_nisp"]
    ra_arr  = np.column_stack([t[c].to_numpy(zero_copy_only=False) for c in cols_r])
    dec_arr = np.column_stack([t[c].to_numpy(zero_copy_only=False) for c in cols_d])
    out_ra  = np.full(ra_arr.shape[0], np.nan)
    out_dec = np.full(ra_arr.shape[0], np.nan)
    for j in range(ra_arr.shape[1]):
        m = np.isnan(out_ra) & np.isfinite(ra_arr[:, j])
        out_ra[m]  = ra_arr[m, j]
        out_dec[m] = dec_arr[m, j]
    return out_ra, out_dec


def main():
    t0 = time.time()
    print("Reading SN candidate catalog ...", flush=True)
    table = pq.read_table(IN_PATH, columns=[
        "primary_id", "primary_source",
        "cat_ra_hst", "cat_dec_hst",
        "cat_ra_jwst", "cat_dec_jwst",
        "cat_ra_vis", "cat_dec_vis",
        "cat_ra_nisp", "cat_dec_nisp",
    ])
    n = table.num_rows
    print(f"  {n:,} rows", flush=True)

    ra, dec = best_radec_per_row(table)
    finite = np.isfinite(ra) & np.isfinite(dec)
    print(f"  finite (ra,dec): {finite.sum():,} ({100*finite.sum()/n:.2f}%)", flush=True)

    # --- Build tile footprints ------------------------------------------
    print("\nBuilding HST footprints ...", flush=True)
    hst_re = re.compile(r"acs_I_030mas_(\d+)_sci\.fits$")
    hst_tiles = survey_tiles(
        "HST",
        HST_DIR, "acs_I_030mas_*_sci.fits", hst_re)

    print("Building JWST footprints (F115W; A10 falls back to scidir/v0_8) ...", flush=True)
    jwst_re = re.compile(r"30mas_([AB]\d+)_v[01]")
    a10_fb = {"A10": JWST_DIR / "scidir" /
              "mosaic_nircam_f115w_COSMOS-Web_30mas_A10_v0_8_sci.fits"}
    jwst_tiles = survey_tiles(
        "JWST",
        JWST_DIR, "mosaic_nircam_f115w_COSMOS-Web_30mas_*_v1.0_i2d.fits",
        jwst_re, fallback=a10_fb)

    print("Building Euclid footprints (VIS; main dir only) ...", flush=True)
    eu_re = re.compile(r"TILE(\d+)-")
    eu_tiles = survey_tiles(
        "Euclid",
        EU_DIR, "EUC_MER_BGSUB-MOSAIC-VIS_TILE*.fits", eu_re)

    print(f"\nTile footprint build: {time.time()-t0:.1f}s", flush=True)

    # --- Assign tiles to sources ----------------------------------------
    t1 = time.time()
    print("Assigning HST tiles ...", flush=True)
    tile_hst, in_hst = assign_tiles(ra, dec, hst_tiles, "HST")
    print(f"  {in_hst.sum():,} sources have HST coverage ({100*in_hst.sum()/n:.1f}%)", flush=True)

    print("Assigning JWST tiles ...", flush=True)
    tile_jwst, in_jwst = assign_tiles(ra, dec, jwst_tiles, "JWST")
    print(f"  {in_jwst.sum():,} sources have JWST coverage ({100*in_jwst.sum()/n:.1f}%)", flush=True)

    print("Assigning Euclid tiles ...", flush=True)
    tile_eu, in_eu = assign_tiles(ra, dec, eu_tiles, "Euclid")
    print(f"  {in_eu.sum():,} sources have Euclid coverage ({100*in_eu.sum()/n:.1f}%)", flush=True)

    print(f"\nTile assignment: {time.time()-t1:.1f}s", flush=True)

    # --- Build output table ---------------------------------------------
    print("\nWriting output ...", flush=True)
    out = pa.table({
        "primary_id":     table["primary_id"],
        "primary_source": table["primary_source"],
        "ra":             pa.array(ra),
        "dec":            pa.array(dec),
        "tile_hst":       pa.array(tile_hst.tolist(), type=pa.string()),
        "tile_jwst":      pa.array(tile_jwst.tolist(), type=pa.string()),
        "tile_euclid":    pa.array(tile_eu.tolist(),   type=pa.string()),
        "in_hst":         pa.array(in_hst,  type=pa.bool_()),
        "in_jwst":        pa.array(in_jwst, type=pa.bool_()),
        "in_euclid":      pa.array(in_eu,   type=pa.bool_()),
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, OUT_PARQ, compression="zstd")
    size_mb = OUT_PARQ.stat().st_size / 1e6
    print(f"  wrote {OUT_PARQ}  ({size_mb:.1f} MB)", flush=True)

    # --- Coverage cross-tabs --------------------------------------------
    n_all  = int((in_hst & in_jwst & in_eu).sum())
    n_h_j  = int((in_hst & in_jwst & ~in_eu).sum())
    n_h_e  = int((in_hst & ~in_jwst & in_eu).sum())
    n_j_e  = int((~in_hst & in_jwst & in_eu).sum())
    n_h    = int((in_hst & ~in_jwst & ~in_eu).sum())
    n_j    = int((~in_hst & in_jwst & ~in_eu).sum())
    n_e    = int((~in_hst & ~in_jwst & in_eu).sum())
    n_none = int((~in_hst & ~in_jwst & ~in_eu).sum())

    log = []
    log.append("# SN candidate FITS lookup v01 — summary")
    log.append(f"# Built: {time.strftime('%Y-%m-%d %H:%M:%S')}  ({time.time()-t0:.1f} sec total)")
    log.append(f"# Source: {IN_PATH}")
    log.append(f"# Output: {OUT_PARQ}")
    log.append("")
    log.append("## Tile footprint counts (per survey)")
    log.append(f"  HST    fields:           {len(hst_tiles):>4}")
    log.append(f"  JWST   tiles  (F115W):   {len(jwst_tiles):>4}")
    log.append(f"  Euclid tiles  (VIS):     {len(eu_tiles):>4}")
    log.append("")
    log.append("## Per-survey coverage (any tile assigned)")
    log.append(f"  in_hst    True:  {int(in_hst.sum()):>10,}  ({100*in_hst.sum()/n:.2f}%)")
    log.append(f"  in_jwst   True:  {int(in_jwst.sum()):>10,}  ({100*in_jwst.sum()/n:.2f}%)")
    log.append(f"  in_euclid True:  {int(in_eu.sum()):>10,}  ({100*in_eu.sum()/n:.2f}%)")
    log.append("")
    log.append("## Cross-coverage breakdown")
    log.append(f"  in_hst ∩ in_jwst ∩ in_euclid:      {n_all:>10,}")
    log.append(f"  in_hst ∩ in_jwst (no Euclid):      {n_h_j:>10,}")
    log.append(f"  in_hst ∩ in_euclid (no JWST):      {n_h_e:>10,}")
    log.append(f"  in_jwst ∩ in_euclid (no HST):      {n_j_e:>10,}")
    log.append(f"  HST only:                          {n_h:>10,}")
    log.append(f"  JWST only:                         {n_j:>10,}")
    log.append(f"  Euclid only:                       {n_e:>10,}")
    log.append(f"  None covered (probably outside any tile or no RA/Dec): {n_none:>10,}")
    log.append("")
    log.append("## File size")
    log.append(f"  {size_mb:.1f} MB (zstd parquet)")
    OUT_LOG.write_text("\n".join(log) + "\n")
    print("\n--- summary ---")
    print("\n".join(log))


if __name__ == "__main__":
    main()
