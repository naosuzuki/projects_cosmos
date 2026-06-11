#!/usr/bin/env python
"""
67_step3b_classifier.py — Step 3b: Gaia-PM-anchored star/galaxy
classifier (ms.tex §B.1 verbatim).

Per (mission, tile), operating on 53_'s phot_<tile>.fits:

  1. Match the spine to Gaia DR3 (wide COSMOS CSV, epoch 2016) within
     --gaia-radius (default 1.0", generous for the 2016→2024 PM drift
     of high-PM training stars).
  2. TRAINING SET per band: Gaia sources with significant proper motion
     (PM ≥ 5σ AND |PM| ≥ 5 mas/yr), --g-min < G < --g-max, NOT saturated
     in that band, finite stats — the ground-truth stellar locus.
     NOTE: ms.tex locks 14<G<18, but in the deep space imaging that
     window is itself SATURATED (HST F814W onset 19.0 mag → only 4
     usable training stars on tile A4).  Default --g-max=21: the
     5σ/5 mas/yr PM cut already enforces astrometric purity, which was
     the point of the bright window.  (ms.tex to be amended.)
  3. For each band, fit the per-magnitude 95th-percentile envelope of
     the training distribution in each of the four PSF-shape statistics
       CHI2_PSF       (one-sided upper envelope)
       DAO_SHARP      (two-sided, 2.5–97.5 percentiles)
       DAO_RND1/RND2  (two-sided)
     over MAG_PSF bins (≥ --min-bin training stars per bin, edges held
     flat beyond the training range).  A source votes "star" in a band
     when ALL its finite stats lie inside the band's envelopes.
  4. Aggregate (ms.tex):  is_star = gaia_pm_significant
       OR (≥ --min-votes votes among UNSATURATED bands).
     --min-votes defaults to 3; single-band missions (HST) can only
     flag Gaia-PM stars until the cross-mission union (Step 5).
     No QSO/AGN logic here (deferred to Step 6+); galaxies stay in the
     catalog as the null sample.

Output:
  <mission>_chi2/<tile>/phot/star_<tile>.fits      per-source flags+votes
  <mission>_chi2/<tile>/phot/star_<tile>.meta.json counts + envelope QA
  <mission>_chi2/<tile>/phot/star_envelopes_<tile>.png
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    'chi2_detect', HERE / '51_step3a_chi2_detect.py')
chi2_detect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chi2_detect)
MISSIONS = chi2_detect.MISSIONS
WORK = chi2_detect.WORK

GAIA_CSV = Path('/Volumes/exdisk1/data/catalog/Gaia/COSMOS/'
                'gaia_dr3_cosmos_wide.csv')

# (statistic suffix, two_sided)
STATS = [('CHI2_PSF', False), ('DAO_SHARP', True),
         ('DAO_RND1', True), ('DAO_RND2', True)]


def attach_gaia(t: Table, radius_arcsec: float) -> dict:
    """Add gaia_source_id / gaia_g / gaia_pm / gaia_pm_sig /
    gaia_pm_significant columns to t (in place).  Returns QA counts."""
    import pandas as pd
    from scipy.spatial import cKDTree
    g = pd.read_csv(GAIA_CSV, usecols=['source_id', 'ra', 'dec',
                                       'pmra', 'pmra_error',
                                       'pmdec', 'pmdec_error',
                                       'phot_g_mean_mag'])
    ra = np.asarray(t['ALPHA_J2000'], dtype=float)
    dec = np.asarray(t['DELTA_J2000'], dtype=float)
    cosd = np.cos(np.deg2rad(np.median(dec)))
    tree = cKDTree(np.column_stack([g['ra'].to_numpy() * cosd,
                                    g['dec'].to_numpy()]))
    d, j = tree.query(np.column_stack([ra * cosd, dec]), k=1)
    ok = d * 3600.0 <= radius_arcsec

    n = len(t)
    sid = np.zeros(n, dtype=np.int64)
    gmag = np.full(n, np.nan)
    pm = np.full(n, np.nan)
    pmsig = np.full(n, np.nan)
    jj = j[ok]
    sid[ok] = g['source_id'].to_numpy()[jj]
    gmag[ok] = g['phot_g_mean_mag'].to_numpy()[jj]
    pmra = g['pmra'].to_numpy()[jj]
    pmdec = g['pmdec'].to_numpy()[jj]
    epmra = g['pmra_error'].to_numpy()[jj]
    epmdec = g['pmdec_error'].to_numpy()[jj]
    pmtot = np.hypot(pmra, pmdec)
    # PM significance: |PM| / projected error along the PM direction
    with np.errstate(divide='ignore', invalid='ignore'):
        epm = np.sqrt((pmra * epmra)**2 + (pmdec * epmdec)**2) / pmtot
        sig = pmtot / epm
    pm[ok] = pmtot
    pmsig[ok] = sig
    t['gaia_source_id'] = sid
    t['gaia_g'] = gmag
    t['gaia_pm'] = pm
    t['gaia_pm_sig'] = pmsig
    t['gaia_pm_significant'] = (np.nan_to_num(pm) >= 5.0) & \
                               (np.nan_to_num(pmsig) >= 5.0)
    return dict(n_gaia_matched=int(ok.sum()),
                n_gaia_pm_significant=int(t['gaia_pm_significant'].sum()))


def fit_envelope(mag: np.ndarray, val: np.ndarray, two_sided: bool,
                 min_bin: int):
    """Running-percentile envelope vs magnitude.  Returns dict with bin
    centers and lo/hi arrays (lo=None for one-sided), or None if too few
    stars."""
    ok = np.isfinite(mag) & np.isfinite(val)
    mag, val = mag[ok], val[ok]
    if len(mag) < min_bin:
        return None
    order = np.argsort(mag)
    mag, val = mag[order], val[order]
    # adaptive bins: ~min_bin stars per bin, at least 3 bins
    nbin = max(3, min(12, len(mag) // min_bin))
    edges = np.quantile(mag, np.linspace(0, 1, nbin + 1))
    ctr, lo, hi = [], [], []
    for i in range(nbin):
        m = (mag >= edges[i]) & (mag <= edges[i + 1])
        if m.sum() < max(5, min_bin // 2):
            continue
        ctr.append(float(np.median(mag[m])))
        if two_sided:
            l, h = np.percentile(val[m], [2.5, 97.5])
            lo.append(float(l)); hi.append(float(h))
        else:
            hi.append(float(np.percentile(val[m], 95)))
    if len(ctr) < 2:
        return None
    return dict(ctr=np.array(ctr),
                lo=np.array(lo) if two_sided else None,
                hi=np.array(hi), n_train=int(len(mag)))


def eval_envelope(env: dict, mag: np.ndarray, val: np.ndarray
                  ) -> np.ndarray:
    """True where val is inside the envelope (flat extrapolation)."""
    hi = np.interp(mag, env['ctr'], env['hi'])
    inside = val <= hi
    if env['lo'] is not None:
        lo = np.interp(mag, env['ctr'], env['lo'])
        inside &= val >= lo
    return inside


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mission', required=True, choices=sorted(MISSIONS))
    p.add_argument('--tile', required=True)
    p.add_argument('--gaia-radius', type=float, default=1.0)
    p.add_argument('--g-min', type=float, default=14.0)
    p.add_argument('--g-max', type=float, default=21.0)
    p.add_argument('--min-votes', type=int, default=3)
    p.add_argument('--min-bin', type=int, default=15,
                   help='min training stars per envelope bin')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    spec = MISSIONS[args.mission]
    phot_dir = WORK / f'{args.mission}_chi2' / args.tile / 'phot'
    phot_path = phot_dir / f'phot_{args.tile}.fits'
    out_path = phot_dir / f'star_{args.tile}.fits'
    meta_path = phot_dir / f'star_{args.tile}.meta.json'
    if out_path.exists() and meta_path.exists() and not args.force:
        print(f'{args.mission}/{args.tile}: 3b already done — --force')
        return
    if not phot_path.exists():
        sys.exit(f'need 53_ first: {phot_path} missing')

    t = Table.read(phot_path)
    print(f'{args.mission}/{args.tile}: {len(t):,} sources')
    t0 = time.time()
    qa: dict = dict(mission=args.mission, tile=args.tile, n_sources=len(t),
                    min_votes=args.min_votes,
                    g_window=[args.g_min, args.g_max], bands={})
    qa.update(attach_gaia(t, args.gaia_radius))
    print(f"  gaia matched {qa['n_gaia_matched']:,}  "
          f"PM-significant {qa['n_gaia_pm_significant']:,}")

    bands = [b for b in spec['bands']
             if f'{b}_MAG_PSF' in t.colnames]
    votes = np.zeros(len(t), dtype=np.int16)
    n_votable = np.zeros(len(t), dtype=np.int16)
    envelopes = {}
    for b in bands:
        mag = np.asarray(t[f'{b}_MAG_PSF'], dtype=float)
        mag[mag <= -99] = np.nan
        sat = np.asarray(t[f'{b}_IS_SATURATED'], dtype=bool)
        train = (np.asarray(t['gaia_pm_significant']) &
                 (np.asarray(t['gaia_g']) > args.g_min) &
                 (np.asarray(t['gaia_g']) < args.g_max) & ~sat)
        binfo = dict(n_train=int(train.sum()))
        stat_envs = {}
        inside_all = np.ones(len(t), dtype=bool)
        usable = np.isfinite(mag)
        n_stats_used = 0
        for stat, two in STATS:
            col = f'{b}_{stat}'
            if col not in t.colnames:
                continue
            val = np.asarray(t[col], dtype=float)
            env = fit_envelope(mag[train], val[train], two, args.min_bin)
            if env is None:
                binfo[f'{stat}_envelope'] = 'too_few_training_stars'
                continue
            n_stats_used += 1
            stat_envs[stat] = env
            finite = np.isfinite(val)
            inside = np.zeros(len(t), dtype=bool)
            inside[finite] = eval_envelope(env, mag[finite], val[finite])
            # sources with a non-finite stat don't fail the band on it,
            # but must pass at least one finite stat (tracked via usable)
            inside_all &= inside | ~finite
            usable &= True
        band_vote = inside_all & usable & np.isfinite(mag) & \
            (n_stats_used > 0)
        t[f'is_star_psf_{b}'] = band_vote
        votable = ~sat & np.isfinite(mag) & (n_stats_used > 0)
        votes += (band_vote & votable).astype(np.int16)
        n_votable += votable.astype(np.int16)
        envelopes[b] = stat_envs
        binfo.update(n_stats_used=n_stats_used,
                     n_vote=int((band_vote & votable).sum()))
        qa['bands'][b] = binfo
        print(f'  {b}: train {binfo["n_train"]:>4}  stats {n_stats_used}  '
              f'votes {binfo["n_vote"]:,}')

    t['star_votes'] = votes
    t['star_votable_bands'] = n_votable
    t['is_star'] = np.asarray(t['gaia_pm_significant']) | \
        (votes >= args.min_votes)
    qa.update(n_is_star=int(t['is_star'].sum()),
              n_by_votes=int((votes >= args.min_votes).sum()),
              total_s=round(time.time() - t0, 1),
              created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                        time.gmtime()))

    keep = (['SOURCE_ID', 'NUMBER', 'DETECT_MODE', 'ALPHA_J2000',
             'DELTA_J2000', 'X_IMAGE', 'Y_IMAGE', 'gaia_source_id',
             'gaia_g', 'gaia_pm', 'gaia_pm_sig', 'gaia_pm_significant',
             'star_votes', 'star_votable_bands', 'is_star'] +
            [f'is_star_psf_{b}' for b in bands] +
            [c for b in bands for c in
             (f'{b}_MAG_PSF', f'{b}_IS_SATURATED') if c in t.colnames])
    t[keep].write(out_path, overwrite=True)
    meta_path.write_text(json.dumps(qa, indent=2))

    # QA plot: envelopes over the training + full population
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        nb = max(len(bands), 1)
        fig, axes = plt.subplots(nb, len(STATS),
                                 figsize=(4*len(STATS), 3.2*nb),
                                 squeeze=False)
        for i, b in enumerate(bands):
            mag = np.asarray(t[f'{b}_MAG_PSF'], dtype=float)
            for k, (stat, two) in enumerate(STATS):
                ax = axes[i][k]
                col = f'{b}_{stat}'
                if col not in t.colnames or stat not in envelopes.get(b, {}):
                    ax.set_axis_off()
                    continue
                val = np.asarray(t[col], dtype=float)
                f = np.isfinite(mag) & np.isfinite(val)
                ax.plot(mag[f], val[f], '.', ms=1, alpha=0.15, color='gray')
                tr = f & np.asarray(t['gaia_pm_significant']) & \
                    (np.asarray(t['gaia_g']) > args.g_min) & \
                    (np.asarray(t['gaia_g']) < args.g_max)
                ax.plot(mag[tr], val[tr], '.', ms=4, color='tab:blue')
                env = envelopes[b][stat]
                ax.plot(env['ctr'], env['hi'], '-', color='tab:red')
                if env['lo'] is not None:
                    ax.plot(env['ctr'], env['lo'], '-', color='tab:red')
                lo_y = (np.nanpercentile(val[f], 1) if f.sum() else 0)
                hi_y = (np.nanpercentile(val[f], 99) if f.sum() else 1)
                ax.set_ylim(lo_y, hi_y)
                ax.set_xlim(14, 30)
                ax.set_title(f'{b} {stat}', fontsize=9)
        fig.tight_layout()
        fig.savefig(phot_dir / f'star_envelopes_{args.tile}.png', dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f'  [warn] QA plot failed: {e}')

    print(f'\n=== 3b {args.mission}/{args.tile} ===')
    print(f"  is_star {qa['n_is_star']:,}  "
          f"(by votes {qa['n_by_votes']:,}, "
          f"by Gaia PM {qa['n_gaia_pm_significant']:,})")
    print(f'  [save] {out_path}')


if __name__ == '__main__':
    main()
