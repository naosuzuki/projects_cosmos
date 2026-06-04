# csvfiles_star/v04/ — v04 cross-mission stellar catalog outputs

This directory holds outputs of the **v04 cross-mission stellar catalog
rebuild**.  Each pipeline step in `programs_star/v04/` (10_, 20_, …)
writes its derived products here.

The bulky intermediate parquet tables (`cat_matched_*.parquet`,
`cross_*.parquet`, `refined_*.parquet` …) are **gitignored** — see the
top-level `.gitignore`.  Only small artefacts that are part of the
canonical record stay tracked.

---

## Step 1 — 4-way common-area mask + tile lookup

**Script:** `programs_star/v04/10_step1_common_area.py`
**Decision log:** `~/github/papers/26_jwsteuclidhst_note/ms.tex`
Appendix B §B.1 (pipeline plan) and §B.2 (locked Q1–Q5 + S1–S9).

### `tile_lookup.parquet` (+ `.csv` sidecar)

One row per (mission, filter, tile) record, **330 rows × 18 columns**:

| Column | Description |
| --- | --- |
| `mission`             | `HST` / `JWST` / `Euclid` |
| `filter`              | `F814W`, `F115W`, `F150W`, `F277W`, `F444W`, `VIS`, `NIR-Y`, `NIR-J`, `NIR-H` |
| `tile_id`             | Mission-native tile identifier (`B1`, `A10`, `TILE101538497`, …) |
| `path`                | Absolute path of the source FITS on `/Volumes/exdisk1` |
| `file_format`         | `multi_extension` (JWST main) / `separate_sci_wht_err` (HST + JWST A10) / `single_hdu` (Euclid + JWST A2/B4/B6 scidir fallback) |
| `naxis1`, `naxis2`    | Image dimensions in pixels |
| `pixscale_mas`        | Native pixel scale in milli-arcsec |
| `ra_min`, `ra_max`    | Tile bounding box in RA (deg, ICRS) |
| `dec_min`, `dec_max`  | Tile bounding box in Dec (deg, ICRS) |
| `corners_ra`, `corners_dec` | 4-corner sky polygon for precise containment |
| `coverage_mask_path`  | Absolute path to the per-tile coverage mask in `coverage/` |
| `n_pix_total`         | Total pixel count of the mosaic |
| `n_pix_observed`      | Pixels passing `sci ≠ NaN AND wht > 0 AND err > 0 AND err ≠ NaN` |
| `n_pix_after_erosion` | Final count after 10-px erosion (drizzle-rim trim) |

Downstream steps consume this lookup table to find which mission/tile
contains any (RA, Dec) without re-opening any FITS file.

### `coverage/` — per-tile native-resolution masks

**330 files** named `<mission>_<filter>_<tile_id>.fits.gz`.  Each is a
`uint8` array on the source mosaic's own WCS.  Value `1` = pixel was
observed by that mission/filter AND lies ≥ 10 native pixels from any
coverage edge.  Native pixel scales are:

| Mission | Filter(s) | Pixel scale |
| --- | --- | --- |
| HST/ACS    | F814W                          | 30 mas |
| JWST/NIRCam| F115W / F150W / F277W / F444W  | 30 mas |
| Euclid/MER | VIS, NIR-Y, NIR-J, NIR-H       | 100 mas |

Storage: each gzipped mask is ~500 KB–1 MB.  All 330 together ≈ 200 MB.

### `common_area_4way_1arcsec.fits.gz` — global QC mask

A **single global** mask covering the COSMOS field at a coarser
resolution.  It marks where **all four missions overlap simultaneously**
(HST ∩ JWST-all-bands ∩ Euclid-VIS ∩ Euclid-NISP-all-bands), reprojected
onto one synthetic COSMOS-centred WCS.

#### Why two flavours of mask?

| Mask | Pixel scale | Size on disk | Purpose |
| --- | --- | --- | --- |
| Per-tile native (`coverage/…`) | 30 mas (HST/JWST) or 100 mas (Euclid) | ~200 MB total | Source-level work: "is *this exact pixel* of *this tile* observed?" |
| Global 1 arcsec (`common_area_4way_1arcsec.fits.gz`) | **1000 mas (1″)** | ~5 MB | Whole-field browse / area calculation / QC plotting |

#### What "1 arcsec" actually means

The global mask is downsampled to a **1 arcsec pixel scale**, i.e.
one global pixel covers a 1″ × 1″ patch of sky.  Compared to the native
HST/JWST 30 mas:

- 1″ = 1000 mas, so one global pixel ≈ **33 × 33 = 1 089 native
  pixels** collapsed into a single bit.
- Global grid is **7 201 × 7 201 pixels** ≈ 2° × 2° centred on
  (RA, Dec) = (150.1°, +2.2°).  Total size ≈ 5 MB compressed.
- 1″ is still **fine enough to resolve real coverage features** —
  chip gaps, dither stripes, partial-tile rims are all larger than 1″,
  so they remain visible.
- 1″ also matches **Gaia DR3's astrometric scale**, so for footprint
  / star-catalog QC it is the right resolution to think at.

#### When to use which

| Question | Mask |
| --- | --- |
| "What's the total 4-way overlap area in deg²?" | global 1″ |
| "Plot a heat-map of the common region" | global 1″ |
| "Is the star at (RA, Dec) inside the common area?" | global 1″ (fast, sub-arcsec accuracy is not needed) |
| "Is *pixel (x, y)* of HST tile B5 observed?" | per-tile native |
| "Reproject one mission's coverage onto another's pixel grid" | per-tile native (input to the reproject step) |

### `_step1_full.log` (gitignored)

Per-run stdout/stderr from a Step 1 invocation.  Not tracked.

---

## Reference

Locked design decisions for v04 (including JWST F115W as the geometric
reference frame, HST_2005 only, edge criterion, 10-px erosion, scidir
fallback for the 4 corrupted F115W tiles, and the 1 arcsec global grid)
are recorded in the paper:
`~/github/papers/26_jwsteuclidhst_note/ms.tex` Appendix B.

Step 1 reads imaging from the locations documented in Appendix A of the
same paper:

- HST 2005 ACS F814W: `/Volumes/exdisk1/data/HST/COSMOS_ACS2005/`
- JWST NIRCam F115/F150/F277/F444W: `/Volumes/exdisk1/data/JWST/COSMOS_v0.8/`
  (with F115W A2/B4/B6/A10 sourced from the `scidir/` subdirectory)
- Euclid VIS + NISP-Y/J/H: `/Volumes/exdisk1/data/Euclid/COSMOS_DR1/`

Last reviewed: 2026-06-04.
