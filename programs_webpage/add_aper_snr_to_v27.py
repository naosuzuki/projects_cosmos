"""Append PSF-matched aperture-photometry S/N to the v27 best-band table.

Aperture sizes are scaled to each band's PSF FWHM (measured 2026-05-26):
    aper_r   = 1.0 * FWHM         (SNR-optimal for a Gaussian source)
    ring_in  = 2.0 * FWHM
    ring_out = 3.5 * FWHM

This makes per-band SNR fair across the very different PSFs of HST 30mas,
JWST 30mas, and Euclid 100mas — the fixed 0.2" aperture used by
recut_3_hst.py's measure step is 3-4x too big for JWST F115W and ~3x too
small for Euclid NIR-Y.

We piggyback on recut_3_hst.aper_photometry by temporarily overriding its
module-level aperture constants.

Outputs (overwrites existing v27 artefacts):
  v27/dao_params_best.csv    +columns: aper_r, snr_aper, mag_aper
  v27/index.html             best-band table gains the snr_aper column
"""
import warnings; warnings.filterwarnings("ignore")
import csv, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import recut_3_hst as R

V27 = Path("/Users/suzuki/github/projects_cosmos/htmls/_crosshair_test/v27")
BEST_CSV = V27 / "dao_params_best.csv"
LOOKUP   = Path("/tmp/sn_lookup_13.csv")

# Empirical PSF FWHM (arcsec) per band — from recut_3_hst.py comments.
PSF_FWHM = {
    "F814W": 0.134,
    "F115W": 0.057, "F150W": 0.057, "F277W": 0.130, "F444W": 0.160,
    "NIR-Y": 0.524, "NIR-J": 0.537, "NIR-H": 0.567,
    "VIS":   0.194,
}

# Map band -> (kind, path_fn)
def hst_path(s):  return R._hst_path_fn(s)
def jwst_path(b): return lambda s: R._jwst_band_path_fn(b)(s)
def eu_path(b):   return lambda s: R._eu_band_path_fn(b)(s)

BAND_INFO = {
    "F814W": ("hst",    hst_path),
    "F115W": ("jwst",   jwst_path("f115w")),
    "F150W": ("jwst",   jwst_path("f150w")),
    "F277W": ("jwst",   jwst_path("f277w")),
    "F444W": ("jwst",   jwst_path("f444w")),
    "NIR-Y": ("euclid", eu_path("NIR-Y")),
    "NIR-J": ("euclid", eu_path("NIR-J")),
    "NIR-H": ("euclid", eu_path("NIR-H")),
    "VIS":   ("euclid", eu_path("VIS")),
}


def psf_aper_phot(sn, band):
    """Run aper_photometry with PSF-matched aperture & ring."""
    fwhm = PSF_FWHM[band]
    aper_r   = 1.0 * fwhm
    ring_in  = 2.0 * fwhm
    ring_out = 3.5 * fwhm
    kind, path_fn = BAND_INFO[band]
    sci_path = path_fn(sn)
    # temporary override of recut_3_hst constants
    o_a, o_i, o_o = R.APER_ARCSEC, R.HOST_RING_IN_AS, R.HOST_RING_OUT_AS
    R.APER_ARCSEC      = aper_r
    R.HOST_RING_IN_AS  = ring_in
    R.HOST_RING_OUT_AS = ring_out
    try:
        mag, snr = R.aper_photometry(sci_path, None, kind, sn["sn_ra"], sn["sn_dec"])
    finally:
        R.APER_ARCSEC, R.HOST_RING_IN_AS, R.HOST_RING_OUT_AS = o_a, o_i, o_o
    return aper_r, ring_in, ring_out, mag, snr


# ─── load source coords ─────────────────────────────────────────────
srcs = {}
with LOOKUP.open() as f:
    for r in csv.DictReader(f):
        srcs[int(r["id"])] = dict(
            id=int(r["id"]),
            sn_ra=float(r["sn_ra"]), sn_dec=float(r["sn_dec"]),
            host_ra=float(r["host_ra"]), host_dec=float(r["host_dec"]),
            telescope=r["telescope"],
            hst=r["hst"], jwst=r["jwst"], euclid=r["euclid"],
        )

# ─── load v27 best-band table ───────────────────────────────────────
rows = []
with BEST_CSV.open() as f:
    for r in csv.DictReader(f):
        r["id"] = int(r["id"])
        rows.append(r)

# ─── compute aperture S/N per row ───────────────────────────────────
t0 = time.time()
print(f"{'ID':>7} {'TEL':>6} {'BAND':>5} {'fwhm':>6} {'aper_r':>7} "
      f"{'snr_aper':>9} {'mag_aper':>9}  vs  {'peak_snr':>9}")
print("-" * 90)
for r in rows:
    sn = srcs[r["id"]]
    band = r["band"]
    fwhm = PSF_FWHM[band]
    aper_r, ring_in, ring_out, mag_ap, snr_ap = psf_aper_phot(sn, band)
    r["aper_r"]   = round(aper_r, 4)
    r["snr_aper"] = round(snr_ap, 2)
    r["mag_aper"] = (round(mag_ap, 2) if mag_ap is not None else "")
    print(f"{r['id']:>7} {r['telescope']:>6} {band:>5} "
          f"{fwhm:>6.3f} {aper_r:>6.3f}\" "
          f"{snr_ap:>9.2f} {(mag_ap if mag_ap is not None else float('nan')):>+9.2f}  vs  "
          f"{float(r['peak_snr']):>9.2f}")
t_total = time.time() - t0
print(f"\naperture photometry: {t_total:.2f}s for {len(rows)} sources")

# ─── rewrite dao_params_best.csv with extra columns ─────────────────
cols = ["id","telescope","band","sep_host","sep_sn",
        "peak","bg_std","peak_snr",
        "aper_r","snr_aper","mag_aper",
        "sharp","rnd1","rnd2","flux","mag"]
with BEST_CSV.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(rows, key=lambda x: x["id"]):
        w.writerow({k: r.get(k, "") for k in cols})
print(f"wrote {BEST_CSV}")

# ─── splice snr_aper column into v27/index.html best-band table ─────
idx = V27 / "index.html"
html = idx.read_text()

# add a header cell after <th>peak/bg_std</th>
hdr_old = "<th>peak/bg_std</th>"
hdr_new = "<th>peak/bg_std</th><th>snr_aper</th><th>mag_aper</th>"
if hdr_new not in html:
    html = html.replace(hdr_old, hdr_new, 1)

# add snr_aper and mag_aper cells in the best-band table only.
# The best-band rows look like  ...<td>SNR</td><td>+0.45</td><td>+0.12</td>...
# We splice after peak/bg_std value cell.  Each row is identifiable by
# beginning with <tr><td>{id}</td><td>{tel}</td>.
def _maybe(v, fmt):
    try: return fmt % float(v)
    except (ValueError, TypeError): return "&mdash;"

for r in sorted(rows, key=lambda x: x["id"]):
    needle = f"<tr><td>{r['id']}</td><td>{r['telescope']}</td>"
    pos = html.find(needle)
    if pos < 0: continue
    # find the 8th <td...> after this (peak/bg_std is the 8th column starting from <th>peak/bg_std</th>)
    # Simpler: find the closing </tr> for this row and inject before <td>+0.xx</td><td>+0.yy</td><td>+0.zz</td>...
    # Concretely: after <td>{peak_snr}</td>, insert two <td> cells.
    # We'll search for the peak_snr text and inject right after its </td>.
    peak_snr_str = f"<td>{float(r['peak_snr']):.2f}</td>"
    seg_start = html.find(peak_snr_str, pos)
    if seg_start < 0: continue
    cut = seg_start + len(peak_snr_str)
    inject = (f"<td>{_maybe(r['snr_aper'], '%.2f')}</td>"
              f"<td>{_maybe(r['mag_aper'], '%.2f')}</td>")
    html = html[:cut] + inject + html[cut:]

# Also update the descriptive prose at the top
notes = ("<p><b>aper_r</b>=1&times;FWHM (SNR-optimal); "
         "ring 2&ndash;3.5&times;FWHM; <b>snr_aper</b>=flux/(&sigma;_ring&radic;n).</p>")
if "snr_aper</b>=flux" not in html:
    html = html.replace("<p>DAO scan in every band", notes + "<p>DAO scan in every band", 1)

idx.write_text(html)
print(f"updated {idx}")
print("DONE")
