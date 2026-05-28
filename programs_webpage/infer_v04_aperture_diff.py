"""v04 (revised): Aperture-level difference imaging — fast.

Instead of pixel-level reproject + subtract, compute the per-source
aperture-flux "diff": expected science flux given HST F814W observation,
versus measured science flux. The difference is the transient component.

Mathematically:
  Let K(band) = expected ratio of science_flux_band / HST_F814W_flux
                for a typical galaxy at this magnitude.
  diff_band   = measured_flux_band  −  K(band) · HST_F814W_flux
  diff_snr    = diff / σ(flux estimated from per-band photometry uncertainty)

K(band) is estimated EMPIRICALLY from the per-mag-bin median ratio
across all sources (in the absence of transients, K reflects galaxy
colours). Outliers from this relation are transient candidates.

A real modern SN: science flux ≫ K · HST flux → strong positive residual.
A normal galaxy:   science flux ≈ K · HST flux → residual ~ 0.

Input: csvfiles_sn/_partial/infer_v05_<survey>.parquet  (per-band mag/snr/sharp...)
Output: csvfiles_sn/_partial/infer_v04_aper_diff.parquet (per-source per-band residuals)
        csvfiles_sn/tbl_sn_candidates_v04.csv via consolidate_v04_aperture.py

Runtime: ~minutes (pure arithmetic on existing tables, no FITS reads).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR = CSV_DIR / "_partial"
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"
OUT_PARQ = PART_DIR / "infer_v04_aper_diff.parquet"

# All science bands (modern-epoch); HST F814W is the template
SCI_BANDS = ["F115W","F150W","F277W","F444W","VIS","Y","J","H"]


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_band(survey, band, N):
    """Load mag & snr arrays for one band from infer_v05 partial."""
    cp = PART_DIR / f"infer_v05_{survey}.parquet"
    if not cp.exists(): return None, None
    t = pq.read_table(cp)
    idxs = t["idx"].to_numpy(zero_copy_only=False).astype(np.int64)
    mag = np.full(N, -1.0, dtype=np.float32)
    snr = np.full(N,  0.0, dtype=np.float32)
    mag_col = f"mag_{band}"; snr_col = f"snr_{band}"
    if mag_col not in t.column_names: return None, None
    mag[idxs] = t[mag_col].to_numpy(zero_copy_only=False).astype(np.float32)
    snr[idxs] = t[snr_col].to_numpy(zero_copy_only=False).astype(np.float32)
    return mag, snr


def mag_to_flux(mag):
    """AB mag -> linear flux (arbitrary units, consistent across bands per source)."""
    out = np.full_like(mag, np.nan, dtype=np.float32)
    ok = np.isfinite(mag) & (mag > 0)
    out[ok] = 10**(-0.4 * mag[ok])
    return out


def main():
    t0 = time.time()
    log("=== v04 aperture-level diff (fast) ===")
    lk = pq.read_table(LOOK)
    pid = np.array(lk["primary_id"].to_pylist(), dtype=object)
    ra  = lk["ra"].to_numpy(zero_copy_only=False); dec = lk["dec"].to_numpy(zero_copy_only=False)
    th  = np.array(lk["tile_hst"].to_pylist(), dtype=object)
    tj  = np.array(lk["tile_jwst"].to_pylist(), dtype=object)
    te  = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(lk["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(lk["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    N = len(pid)
    log(f"  {N:,} candidates")

    # Load HST F814W mag/snr (template band)
    hst_mag, hst_snr = load_band("hst", "F814W", N)
    if hst_mag is None: raise RuntimeError("no HST partial")
    hst_flux = mag_to_flux(hst_mag)
    log(f"  HST F814W loaded: {int((hst_mag > 0).sum()):,} sources with finite mag")

    # Per-band processing
    band_survey = {"F115W":"jwst","F150W":"jwst","F277W":"jwst","F444W":"jwst",
                   "VIS":"vis","Y":"nisp","J":"nisp","H":"nisp"}
    diff_flux  = {}; diff_snr = {}; diff_mag = {}; K_emp = {}
    for b in SCI_BANDS:
        sv = band_survey[b]
        mag, snr = load_band(sv, b, N)
        if mag is None:
            log(f"  [{b}] no partial — skip"); continue
        sci_flux = mag_to_flux(mag)
        # Empirical K(b) per HST-mag bin: median(sci_flux / hst_flux) among
        # sources with both finite + non-zero. Use bins of 0.5 mag in HST.
        ok = (hst_flux > 0) & np.isfinite(sci_flux) & (sci_flux > 0)
        if ok.sum() < 100:
            log(f"  [{b}] too few overlap"); continue
        ratio = sci_flux[ok] / hst_flux[ok]
        # bin by hst_mag
        bin_edges = np.arange(18.0, 28.0, 0.5)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        K_bin = np.zeros(len(bin_centers), dtype=np.float64)
        hbins = np.clip(np.digitize(hst_mag[ok], bin_edges) - 1, 0, len(bin_centers)-1)
        for i in range(len(bin_centers)):
            sel = ratio[hbins == i]
            if len(sel) >= 10:
                K_bin[i] = float(np.median(sel))
            else:
                K_bin[i] = float(np.median(ratio)) if len(ratio) else 1.0
        # Look up K per source
        K = np.zeros(N, dtype=np.float64)
        idx = np.clip(np.digitize(hst_mag, bin_edges) - 1, 0, len(bin_centers)-1)
        for i in range(len(bin_centers)):
            K[idx == i] = K_bin[i]
        # Expected science flux given HST flux
        expected = K * hst_flux
        residual = sci_flux - expected   # positive = excess flux = transient candidate
        # SNR of residual: use aperture-photometry sigma (from sci snr column)
        # sci_flux corresponds to snr=sci_snr at some mag; we need sigma
        # σ(flux) = flux / snr  (when snr > 0)
        sigma = np.where((snr > 0) & np.isfinite(sci_flux), sci_flux / snr, np.inf)
        # Account for HST uncertainty contribution (assume small relative to host blending)
        diff_snr_b = np.where(np.isfinite(residual) & (sigma > 0), residual / sigma, 0.0)
        # Diff mag (only positive residuals)
        diff_mag_b = np.where(residual > 0, -2.5*np.log10(residual), np.nan)
        # NB diff_mag is in inconsistent units — but its relative variations
        # are informative; we keep it for the CSV.
        diff_flux[b] = residual.astype(np.float32)
        diff_snr[b]  = diff_snr_b.astype(np.float32)
        diff_mag[b]  = diff_mag_b.astype(np.float32)
        K_emp[b]     = K.astype(np.float32)
        positive = (residual > 0) & (diff_snr_b >= 5.0)
        log(f"  [{b}] residual+ >5σ: {int(positive.sum()):,}  median K={float(np.median(K_bin)):.3f}")

    # Save per-band residuals
    log("Writing parquet ...")
    arrs = {"idx": np.arange(N, dtype=np.int64)}
    for b in SCI_BANDS:
        if b not in diff_flux: continue
        arrs[f"diff_snr_{b}"] = diff_snr[b]
        arrs[f"diff_mag_{b}"] = diff_mag[b]
        arrs[f"residual_{b}"] = diff_flux[b]
        arrs[f"K_{b}"]        = K_emp[b]
    table = pa.table(arrs)
    pq.write_table(table, OUT_PARQ, compression="zstd")
    log(f"Wrote {OUT_PARQ}  ({OUT_PARQ.stat().st_size/1e6:.1f} MB)")
    log(f"=== done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
