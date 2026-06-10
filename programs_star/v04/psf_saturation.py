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


def detect_saturation_turnover(mag, peak, is_point, dev_dex=0.12, slope=-0.4):
    """Saturation onset for SOFT-saturating coadds (DECam LS DR10).

    DECam brick coadds interpolate/suppress saturated cores, so peak-vs-mag
    never pins to a flat plateau (the HST signature) — it BENDS: the growth
    rate drops below the photometric slope (-0.4 dex/mag) and may even
    decline.  Detect the bend: onset = the faintest 0.5-mag bin whose median
    peak falls > `dev_dex` BELOW the fixed-slope extrapolation anchored on
    the 3 bins immediately fainter, with the deficit PERSISTING brighter
    (median deficit of all brighter bins also > dev_dex).

    Returns (sat_peak_level, onset_mag); level = predicted unsaturated peak
    at the onset (diagnostic line), np.inf if no turnover found.
    """
    mag = np.asarray(mag, float); peak = np.asarray(peak, float)
    is_point = np.asarray(is_point, bool)
    sel = is_point & np.isfinite(mag) & np.isfinite(peak) & (peak > 0)
    if sel.sum() < 20:
        return np.inf, None
    ms, ps = mag[sel], np.log10(peak[sel])

    # 1. robust log-linear reference fit on the clearly-UNSATURATED range:
    #    the brightest 1.5 mag of point sources are suspect, the faint tail is
    #    noisy → fit between [p5+1.5, p5+5] (p5 = 5th-percentile magnitude),
    #    iteratively clipped.  Empirical slope (≈-0.45..-0.55 in MAG_AUTO
    #    space, steeper than the textbook -0.4).
    p5 = float(np.percentile(ms, 5))
    fit = (ms > p5 + 1.5) & (ms < p5 + 5.0)
    if fit.sum() < 15:
        return np.inf, None
    kp = fit.copy()
    b = (slope, 0.0)
    for _ in range(5):
        if kp.sum() < 10:
            break
        b = np.polyfit(ms[kp], ps[kp], 1)
        r = ps - np.polyval(b, ms)
        s = 1.4826 * float(np.median(np.abs(r[kp] - np.median(r[kp]))))
        if s <= 0:
            break
        kp = fit & (np.abs(r) < 3.0 * s)
    if not (-0.65 <= b[0] <= -0.30):                 # sanity on the slope
        return np.inf, None

    # 2. PER-STAR deficits brighter than the fit range (no binning — the
    #    saturated bright end is sparse, 1-3 stars per 0.5 mag; bin medians
    #    discard exactly the evidence).  Walk stars FAINT→BRIGHT: onset = the
    #    faintest star such that >=70% of all stars brighter than it
    #    (minimum 3) sit > dev_dex BELOW the reference line.
    deficit = np.polyval(b, ms) - ps                 # >0 = below the line
    br = ms <= p5 + 1.5
    if br.sum() < 3:
        return np.inf, None
    order = np.argsort(ms[br])[::-1]                 # faint → bright
    dm, dd = ms[br][order], deficit[br][order]
    onset, level = None, np.inf
    for j in range(len(dm)):
        nb = len(dm) - j                             # stars at or brighter
        if nb < 3:
            break
        frac = float(np.mean(dd[j:] > dev_dex))
        if frac >= 0.70:
            onset = float(dm[j] + 0.25)              # just faintward of star j
            level = float(10.0 ** np.polyval(b, dm[j]))
            break
    return level, onset


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
