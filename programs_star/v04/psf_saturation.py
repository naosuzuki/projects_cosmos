"""psf_saturation.py — shared saturation detection for the single-HDU PSF
builders (HST ACS, Euclid VIS) and their QA plotter.

Two physical saturation signatures, handled together:

  (a) PEAK PLATEAU  (non-masking instruments, e.g. HST ACS drizzled mosaics)
      Once a star saturates, its core peak is pinned at the full-well level,
      so the peak-vs-magnitude relation goes FLAT at the bright end.  We
      detect that plateau from the point-source sample and reject anything
      sitting on it.

  (b) MASKED CORE   (JWST i2d, Euclid VIS MER — saturated cores set to 0)
      Already handled by the masked-core (zero-pixel) test in the builder;
      here we only use it to locate the saturation onset MAGNITUDE for the
      diagnostic line.

`detect_saturation` returns
    sat_peak_level : float  — reject point sources with core peak >= this
                              (np.inf if no plateau, i.e. masking instrument)
    onset_mag      : float|None — magnitude brighter than which sources
                              saturate (for the vertical diagnostic line)
"""
from __future__ import annotations
import numpy as np


def core_peak(sci, xx, yy, half=2, idx=None):
    """Per-source peak = max over a (2*half+1)^2 core box of SCI.

    If `idx` is given, only those rows are computed (others left 0) — use it
    to avoid scanning hundreds of thousands of faint detections.
    """
    ny, nx = sci.shape
    peak = np.zeros(len(xx))
    rows = range(len(xx)) if idx is None else idx
    for k in rows:
        xi = int(round(xx[k])); yi = int(round(yy[k]))
        if half <= xi < nx-half and half <= yi < ny-half:
            peak[k] = float(np.nanmax(sci[yi-half:yi+half+1, xi-half:xi+half+1]))
    return peak


def detect_saturation(mag, peak, masked_core, is_point):
    """Return (sat_peak_level, onset_mag); see module docstring."""
    mag = np.asarray(mag, float); peak = np.asarray(peak, float)
    masked_core = np.asarray(masked_core); is_point = np.asarray(is_point, bool)
    pm = is_point & np.isfinite(mag)
    sat_peak_level = np.inf
    onset_mag = None

    # ── (a) peak plateau on the UN-masked point sources ──
    sel = pm & np.isfinite(peak) & (peak > 0) & (masked_core == 0)
    if sel.sum() >= 12:
        ms, ps = mag[sel], peak[sel]
        edges = np.arange(np.floor(ms.min()), np.ceil(ms.max()) + 0.5, 0.5)
        med = np.array([
            np.median(ps[(ms >= a) & (ms < a + 0.5)])
            if ((ms >= a) & (ms < a + 0.5)).sum() >= 2 else np.nan
            for a in edges])
        if np.isfinite(med).sum() >= 3:
            P = np.nanmax(med)
            iP = int(np.nanargmax(med))
            # trend ~1.5 mag fainter than the plateau peak (unsaturated)
            faint_bins = med[edges >= edges[iP] + 1.5]
            faint = np.nanmedian(faint_bins) if np.isfinite(faint_bins).any() else np.nan
            near = np.isfinite(med) & (med >= 0.8 * P)
            # genuine plateau: >=2 bright bins flat near P AND P >> faint trend
            if near.sum() >= 2 and np.isfinite(faint) and faint > 0 and P >= 3 * faint:
                sat_peak_level = 0.85 * P
                onset_mag = float(edges[near].max() + 0.5)   # faint edge of plateau

    # ── (b) masking instrument (cores zeroed): saturated bright stars have a
    # masked core instead of a peak plateau.  Report an onset ONLY when there
    # is a genuine BRIGHT masked population — i.e. the masked-core point
    # sources are concentrated at least ~1 mag brighter than the clean
    # point-source bulk.  This avoids a spurious "saturation" line on images
    # where saturation does NOT occur (the masked-core points are then just
    # faint sources with a stray zero pixel from a CR / bad column / edge).
    if onset_mag is None:
        masked_core = np.asarray(masked_core)
        msat  = pm & (masked_core > 0)
        clean = pm & (masked_core == 0)
        if msat.sum() >= 3 and clean.sum() >= 10:
            if np.median(mag[msat]) <= np.median(mag[clean]) - 1.0:
                onset_mag = float(np.percentile(mag[msat], 90))  # faint edge of saturated pop

    return sat_peak_level, onset_mag
