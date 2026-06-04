#!/usr/bin/env python
"""
12_step1_3way_common_area.py — Compute the 3-way HST ∩ JWST ∩ Euclid-VIS
common area and persist its EDGES as a sky-coordinate polygon so
downstream steps never need to reopen pixel masks.

This is the canonical v04 footprint:
    Euclid NISP is intentionally dropped (its Y-band coverage is the
    bottleneck — see Step 1 measured 0.27 deg² 4-way vs the much
    larger 3-way).

Inputs (all from Step 1):
  csvfiles_star/v04/coverage/<mission>_<filter>_<tile>.fits.gz
      Per-tile native-resolution coverage masks (boolean uint8) with
      WCS in their FITS header.  330 files total.

Approach (one-shot, never reopens pixels after this run):

  1. Define a global 1″ COSMOS WCS (same as Step 1).
  2. Reproject + OR per-tile masks within each mission to build:
       HST_global  = union of HST F814W tiles
       VIS_global  = union of Euclid VIS tiles
       JWST_global = INTERSECTION across F115/F150/F277/F444W
                     (each filter's per-tile union first, then AND)
  3. Cache HST_global, VIS_global, JWST_global as FITS so any future
     pretty-plot script does not have to reproject 90 tiles again.
  4. The "edge of real data exists" means:
       3way = HST_global AND VIS_global AND JWST_global
     (each pixel is in the 3-way iff that 1″ patch has real observed
     pixels in HST, in all four JWST NIRCam bands, AND in Euclid VIS.)
  5. Extract the OUTER boundary + any internal hole boundaries with
     skimage.measure.find_contours at level=0.5.
  6. Convert pixel coords (y, x) → sky (RA, Dec) via the WCS, store
     as a list of closed polygons.
  7. Save the polygons three ways for convenience:
       common_area_3way_polygons.parquet  — flat table: one row per
                                            vertex; columns
                                            polygon_id, vertex_order,
                                            ra_deg, dec_deg
       common_area_3way.geojson           — standard GeoJSON
                                            MultiPolygon (RA→x, Dec→y)
                                            for QGIS / web maps
       common_area_3way.wkt               — Shapely-compatible WKT
                                            for fast point-in-polygon
                                            tests in Python
  8. Save the 3-way mask as FITS too (small, gzipped).

After this script, downstream queries look like:

    from shapely import wkt
    poly = wkt.loads(open('csvfiles_star/v04/common_area_3way.wkt').read())
    inside = poly.contains_xy(ra, dec)          # vectorised, no FITS

This is the "do not go back to pixels" promise.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon

V04   = Path('/Users/suzuki/github/projects_cosmos/csvfiles_star/v04')
COV   = V04 / 'coverage'
GCOV  = V04 / 'coverage_global'   # cached per-mission masks on the global WCS
GCOV.mkdir(exist_ok=True)

# Global 1″ COSMOS WCS (same as Step 1)
GLOBAL_WCS_CENTER = (150.1, 2.2)
GLOBAL_WCS_SIZE_DEG = 1.0
GLOBAL_PIXEL_SCALE_ARCSEC = 1.0


def make_global_wcs() -> WCS:
    n = int(2 * GLOBAL_WCS_SIZE_DEG * 3600 / GLOBAL_PIXEL_SCALE_ARCSEC) + 1
    w = WCS(naxis=2)
    w.wcs.crpix = [(n+1)/2, (n+1)/2]
    w.wcs.crval = list(GLOBAL_WCS_CENTER)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cdelt = [-GLOBAL_PIXEL_SCALE_ARCSEC/3600,
                    GLOBAL_PIXEL_SCALE_ARCSEC/3600]
    w.array_shape = (n, n)
    return w


def union_mission_filter(prefix: str, gwcs: WCS, gshape: tuple) -> np.ndarray:
    """OR all per-tile masks matching <prefix>*.fits.gz onto the global grid."""
    out = np.zeros(gshape, dtype=bool)
    files = sorted(COV.glob(f'{prefix}*.fits.gz'))
    print(f'  union {prefix}* ({len(files)} tiles) ...', flush=True)
    for f in files:
        with fits.open(f, memmap=True) as h:
            m = np.asarray(h[0].data, dtype=np.float32)
            w = WCS(h[0].header)
        rp, _ = reproject_interp((m, w), gwcs,
                                 shape_out=gshape,
                                 order='nearest-neighbor')
        out |= (np.nan_to_num(rp, nan=0) > 0.5)
    return out


def write_mask_fits(mask: np.ndarray, wcs: WCS, dest: Path, comment: str):
    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8), header=wcs.to_header())
    hdu.header['BUNIT'] = 'coverage'
    hdu.header['COMMENT'] = comment
    hdu.writeto(dest, overwrite=True)


def load_or_build_mission_mask(name: str, builder, gwcs: WCS, gshape: tuple,
                                comment: str) -> np.ndarray:
    """Read cached per-mission mask if present, else build + cache."""
    dest = GCOV / f'{name}_1arcsec.fits.gz'
    if dest.exists():
        with fits.open(dest) as h:
            return np.asarray(h[0].data, dtype=bool)
    print(f'building {name} ...')
    mask = builder()
    write_mask_fits(mask, gwcs, dest, comment)
    print(f'  saved {dest.name}  ({mask.sum()/1e6:.2f} Mpx covered)')
    return mask


def pixel_contours_to_sky(mask: np.ndarray, wcs: WCS, min_area_pix: int = 25
                          ) -> list[Polygon]:
    """Find the bool mask's boundary polygons and convert to sky-coord."""
    contours = find_contours(mask.astype(np.float32), level=0.5)
    polys = []
    for c in contours:
        # c has shape (N, 2) of (row, col) = (y, x)
        ys, xs = c[:, 0], c[:, 1]
        if len(c) < 4:
            continue
        # Convert pixels to sky in ICRS
        sky = wcs.pixel_to_world_values(xs, ys)
        ra, dec = sky[0], sky[1]
        # Close ring if needed
        if (ra[0] != ra[-1]) or (dec[0] != dec[-1]):
            ra  = np.append(ra,  ra[0])
            dec = np.append(dec, dec[0])
        poly = Polygon(list(zip(ra, dec)))
        if not poly.is_valid:
            # buffer(0) sometimes fixes self-intersections from contour ties
            poly = poly.buffer(0)
        # Drop tiny noise loops (pixel area in the bool mask < min_area_pix)
        area_px = abs(Polygon(list(zip(xs, ys))).area)
        if area_px < min_area_pix:
            continue
        polys.append(poly)
    return polys


def main():
    print('Building global 1″ WCS ...')
    gwcs = make_global_wcs()
    gshape = gwcs.array_shape
    print(f'  grid {gshape[1]}x{gshape[0]} px at 1.0″')

    # ── Per-mission global masks (cached) ──────────────────────────────
    HST_mask = load_or_build_mission_mask(
        'HST_F814W',
        lambda: union_mission_filter('HST_F814W_', gwcs, gshape),
        gwcs, gshape,
        '1 arcsec global HST F814W union of all 10 ACS tiles')
    VIS_mask = load_or_build_mission_mask(
        'Euclid_VIS',
        lambda: union_mission_filter('Euclid_VIS_', gwcs, gshape),
        gwcs, gshape,
        '1 arcsec global Euclid VIS union of all MER tiles')
    # JWST = intersection of all 4 NIRCam bands
    jwst_bands = []
    for flt in ('F115W', 'F150W', 'F277W', 'F444W'):
        m = load_or_build_mission_mask(
            f'JWST_{flt}',
            lambda flt=flt: union_mission_filter(f'JWST_{flt}_', gwcs, gshape),
            gwcs, gshape,
            f'1 arcsec global JWST NIRCam {flt} union')
        jwst_bands.append(m)
    JWST_mask = jwst_bands[0].copy()
    for m in jwst_bands[1:]:
        JWST_mask &= m
    write_mask_fits(JWST_mask, gwcs, GCOV / 'JWST_4band_intersect_1arcsec.fits.gz',
                    '1 arcsec global JWST 4-band intersection F115/F150/F277/F444')

    # ── 3-way intersection: HST ∩ JWST(4-band) ∩ VIS ───────────────────
    three_way = HST_mask & JWST_mask & VIS_mask
    print()
    print(f'  HST union     : {HST_mask.sum()/1e6:6.2f} Mpx')
    print(f'  VIS union     : {VIS_mask.sum()/1e6:6.2f} Mpx')
    print(f'  JWST 4-band ∩ : {JWST_mask.sum()/1e6:6.2f} Mpx')
    print(f'  3-way HST∩JWST∩VIS: {three_way.sum()/1e6:6.2f} Mpx '
          f'≈ {three_way.sum()/3600**2:.4f} deg²')

    three_way_path = V04 / 'common_area_3way_1arcsec.fits.gz'
    write_mask_fits(three_way, gwcs, three_way_path,
                    '3-way HST INTERSECT JWST INTERSECT EuclidVIS common area mask')
    print(f'[save] {three_way_path.name}')

    # ── EDGE EXTRACTION (the persistent "edges" record) ────────────────
    print('\nExtracting boundary polygons of the 3-way common area ...')
    polys = pixel_contours_to_sky(three_way, gwcs, min_area_pix=25)
    print(f'  {len(polys)} polygons after filtering tiny noise loops')

    if not polys:
        raise RuntimeError('no polygons extracted — empty 3-way mask?')

    # The "outer" footprint is conventionally the union (MultiPolygon).
    mp = MultiPolygon(polys) if len(polys) > 1 else Polygon(polys[0].exterior)

    # 1) Parquet — flat vertex table
    rows = []
    for poly_id, poly in enumerate(polys):
        if not isinstance(poly, Polygon):
            continue
        xy = list(poly.exterior.coords)
        for v_order, (ra, dec) in enumerate(xy):
            rows.append({'polygon_id': poly_id,
                         'vertex_order': v_order,
                         'ra_deg':  float(ra),
                         'dec_deg': float(dec)})
    df = pd.DataFrame(rows)
    parquet_path = V04 / 'common_area_3way_polygons.parquet'
    df.to_parquet(parquet_path, index=False)
    print(f'[save] {parquet_path.name}: {len(df):,} vertices in {df.polygon_id.nunique()} polygons')

    # 2) GeoJSON — for visual tools (QGIS, web maps, leaflet)
    features = []
    for poly_id, poly in enumerate(polys):
        if not isinstance(poly, Polygon):
            continue
        xy = list(poly.exterior.coords)
        coords = [[float(x), float(y)] for x, y in xy]
        features.append({'type': 'Feature',
                         'properties': {'polygon_id': poly_id,
                                        'area_deg2': float(poly.area)},
                         'geometry':  {'type': 'Polygon',
                                       'coordinates': [coords]}})
    geojson = {'type': 'FeatureCollection',
               'name': 'v04 3-way common area HST ∩ JWST ∩ Euclid VIS',
               'crs':  {'type': 'name',
                        'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}},
               'features': features}
    geojson_path = V04 / 'common_area_3way.geojson'
    geojson_path.write_text(json.dumps(geojson, indent=2))
    print(f'[save] {geojson_path.name}')

    # 3) WKT — for Python downstream (shapely.wkt.loads → point-in-polygon)
    wkt_path = V04 / 'common_area_3way.wkt'
    wkt_path.write_text(mp.wkt)
    print(f'[save] {wkt_path.name}  '
          f'({len(mp.wkt)/1024:.1f} KB,  use shapely.wkt.loads to query)')

    # ── Summary ────────────────────────────────────────────────────────
    print('\n=== 3-way common-area summary ===')
    print(f'  area (mask cells × 1″²) : {three_way.sum()/3600**2:.4f} deg²')
    print(f'  outer polygons          : {len(polys)}')
    print(f'  polygon vertex count    : {len(df):,}')
    bx = mp.bounds
    print(f'  RA  range               : {bx[0]:.4f} -> {bx[2]:.4f}  '
          f'({(bx[2]-bx[0])*60:.2f} arcmin)')
    print(f'  Dec range               : {bx[1]:.4f} -> {bx[3]:.4f}  '
          f'({(bx[3]-bx[1])*60:.2f} arcmin)')


if __name__ == '__main__':
    main()
