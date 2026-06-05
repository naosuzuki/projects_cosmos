# /Users/suzuki/github/projects_cosmos/configs

Common SExtractor + PSFEx configuration files for the v04 cross-mission
photometry pipeline (Step 3a).  Kept here (not under
`programs_star/v04/step3/`) because they are not step- or
version-specific — they depend only on the (telescope, filter) combination.

---

## File index

### Distribution defaults (copied from SExtractor 2.28.2)
| File | Purpose |
| --- | --- |
| `default.nnw` | Neural-network weights for `CLASS_STAR` |
| `default.conv` | 3 × 3 all-ground convolution kernel |

### Project-wide
| File | Purpose |
| --- | --- |
| `default.param` | Output catalog column list (one config for all bands) |
| `default.sex`   | Generic SExtractor template — superseded by per-instrument files; kept for reference |
| `default.psfex` | PSFEx config (PIXEL_AUTO basis, FWHM-selected sample, 95-percentile envelope) |

### Per-instrument SExtractor configs
| Telescope / instrument | Filter | File |
| --- | --- | --- |
| JWST NIRCam            | F115W  | `jwst_nircam_f115w.sex` |
| JWST NIRCam            | F150W  | (TODO) |
| JWST NIRCam            | F277W  | (TODO) |
| JWST NIRCam            | F444W  | (TODO) |
| HST ACS                | F814W  | (TODO) |
| Euclid VIS             | VIS    | (TODO) |
| Euclid NISP            | Y      | (TODO) |
| Euclid NISP            | J      | (TODO) |
| Euclid NISP            | H      | (TODO) |
| Subaru HSC             | g      | (TODO) |
| Subaru HSC             | r      | (TODO) |
| Subaru HSC             | i      | (TODO) |
| Subaru HSC             | z      | (TODO) |
| Subaru HSC             | Y      | (TODO) |
| SDSS 2.5m              | u,g,r,i,z | (TODO) |
| PS1 1.8m               | g,r,i,z,y | (TODO) |
| DESI Legacy / DECam    | g,r,i,z | (TODO) |
| GALEX                  | FUV, NUV | (TODO) |
| unWISE                 | W1, W2 | (TODO) |

Each per-instrument config encodes the right `PIXEL_SCALE`, `SEEING_FWHM`,
`MAG_ZEROPOINT` (or a placeholder overridden at runtime from FITS header),
`GAIN`, `SATUR_LEVEL`, `DETECT_MINAREA`, `BACK_SIZE`, and `PHOT_APERTURES`
for the (telescope, filter) combination.

---

## Usage from Python

```python
from pathlib import Path
import subprocess

CONFIGS = Path('/Users/suzuki/github/projects_cosmos/configs')

cmd = ['sex', image_path,
       '-c',                str(CONFIGS / 'jwst_nircam_f115w.sex'),
       '-CATALOG_NAME',     str(out_cat),
       '-PARAMETERS_NAME',  str(CONFIGS / 'default.param'),
       '-FILTER_NAME',      str(CONFIGS / 'default.conv'),
       '-STARNNW_NAME',     str(CONFIGS / 'default.nnw'),
       '-WEIGHT_IMAGE',     str(weight_path),
       '-MAG_ZEROPOINT',    f'{zp_ab:.4f}',     # from FITS header
       ]
subprocess.run(cmd, check=True)
```

Last reviewed: 2026-06-05
