#!/usr/bin/env python
"""
60_run_mass_production.py — unified, resumable, TIMED driver for the Step-3a
empirical-PSF mass production across all three missions.

Efficiency design (see programs_star/CLAUDE.md "Step 3a"):
  * No redundant copies: the 54_* builders read the source DIRECTLY —
    JWST i2d SCI=ext1 / WHT=ext4 (scidir _sci.fits weightless fallback for
    A10), HST drz, Euclid MER — so each huge frame is read once by SExtractor.
  * Parallelism is ACROSS tiles (SExtractor + PSFEx are each single-threaded;
    there is no GPU path).  Concurrency is tuned PER MISSION for the single
    external HDD — a handful of big JWST/HST reads saturate the spindle, while
    the small Euclid frames tolerate more.
  * Idempotent / resumable: any (mission, tile, band) whose psf meta already
    exists is skipped, so a crash or Ctrl-C just resumes.
  * Every tile's wall-time (build + plots) is appended to a git-tracked CSV
    (programs_star/csv_timing/mass_timing.csv) as it finishes — live progress.

Usage:
  ./60_run_mass_production.py                       # all missions, default jobs
  ./60_run_mass_production.py --missions euclid     # one mission
  ./60_run_mass_production.py --missions jwst,hst --jobs 4
  ./60_run_mass_production.py --force               # reprocess even if done
  ./60_run_mass_production.py --no-site             # skip site rebuild at end
"""
from __future__ import annotations
import argparse, csv, json, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY   = '/opt/miniconda3/bin/python'
WORK = Path('/Volumes/exdisk1/data/photometry_v04')
EUC  = Path('/Volumes/exdisk1/data/Euclid/COSMOS_DR1')
HSC  = Path('/Volumes/exdisk1/data/HSC/COSMOS/s23b')
TIMING_DIR = Path('/Users/suzuki/github/projects_cosmos/programs_star/csv_timing')
TIMING_DIR.mkdir(parents=True, exist_ok=True)
TIMING_CSV = TIMING_DIR / 'mass_timing.csv'
TIMING_COLS = ['utc', 'mission', 'instrument', 'tile', 'band', 'status',
               'build_s', 'plot_s', 'total_s', 'n_psf_stars', 'psf_accepted',
               'psf_chi2', 'saturated', 'sat_onset_mag']

AB_TILES   = [f'{r}{n}' for r in 'AB' for n in range(1, 11)]   # A1..A10,B1..B10
JWST_BANDS = ['f115w', 'f150w', 'f277w', 'f444w']
HSC_BANDS  = ['g', 'r', 'i', 'z', 'y']                         # tract-9813 grizy

# per-mission concurrency tuned to the single HDD (override with --jobs).
# The copy-based builders are I/O-heavy (read i2d SCI+WHT, write sci_/wht_,
# SExtractor re-reads), so the big missions stay modest; small Euclid frames
# tolerate more.  HSC deepCoadd patches are 4200² (modest) but each build
# writes a ~70 MB cleaned-image copy, so keep it middling.
DEFAULT_JOBS = {'jwst': 3, 'hst': 3, 'euclid': 6, 'euclid_nisp': 6, 'hsc': 4,
                'lsdr10': 4}

# big per-tile intermediates to delete once the model + plots + meta exist,
# so a 160-tile run doesn't pile up ~1 TB on the scratch disk.  Kept: the .psf
# model, stars catalog, outcat, meta JSON, and the 6 QA PNGs.  (image_*.fits is
# the HSC single-HDU cleaned-image copy SExtractor reads; the calexp is read
# from its original path at plot time, so the copy is disposable.)
CLEAN_GLOBS = ('sci_*.fits', 'wht_*.fits', 'pass1_*.fits', 'image_*.fits',
               'samp*.fits', 'resi*.fits', 'proto*.fits', 'snap*.fits')

_lock = threading.Lock()


def euclid_tiles():
    return sorted({p.name.split('TILE')[1].split('-')[0]
                   for p in EUC.glob('EUC_MER_BGSUB-MOSAIC-VIS_TILE*.fits')})


def nisp_tiles():
    # NISP shares the MER tiling with VIS; coverage is patchier (only ~20/60
    # tiles are well-covered), but the builder handles sparse tiles gracefully.
    return sorted({p.name.split('TILE')[1].split('-')[0]
                   for p in EUC.glob('EUC_MER_BGSUB-MOSAIC-NIR-Y_TILE*.fits')})


LS_COADD = Path('/Volumes/exdisk1/data/DESI_Legacy/COSMOS/dr10/south/coadd')


def lsdr10_avail():
    """Set of (brick, band) LS DR10 image coadds present on disk."""
    avail = set()
    for f in LS_COADD.glob('*/*/legacysurvey-*-image-*.fits.fz'):
        m = re.search(r'legacysurvey-(\w+)-image-([griz])\.fits\.fz', f.name)
        if m:
            avail.add((m.group(1), m.group(2)))
    return avail


def hsc_avail():
    """Set of (patch, band) tract-9813 deepCoadd_calexp present on disk, so we
    only fan out over real data (resumable; no work items for missing tiles)."""
    avail = set()
    for f in HSC.glob('s23b_deep2/9813/*/*/deepCoadd_calexp_9813_*_*.fits'):
        m = re.search(r'deepCoadd_calexp_9813_(\d+)_([grizy])_', f.name)
        if m:
            avail.add((m.group(1), m.group(2)))
    return avail


def build_worklist(missions):
    """Return a flat list of work items: dict(mission, instrument, tile, band,
    suffix, meta, build_cmd, plot_cmd)."""
    items = []
    if 'jwst' in missions:
        for t in AB_TILES:
            for b in JWST_BANDS:
                inst = f'jwst_nircam_{b}'
                meta = WORK / inst / t / 'psf' / f'psf_{b}.meta.json'
                items.append(dict(
                    mission='jwst', instrument=inst, tile=t, band=b, suffix=b,
                    meta=meta,
                    build_cmd=[PY, str(HERE / '54_step3a_build_psf_model.py'),
                               '--instrument', inst, '--tile', t, '--filter', b],
                    plot_cmd=[PY, str(HERE / '56_make_psf_qa_plots.py'),
                              '--instrument', inst, '--tile', t, '--filter', b]))
    if 'hst' in missions:
        for t in AB_TILES:
            inst = 'hst_acs_f814w'
            meta = WORK / inst / t / 'psf' / f'psf_{t}.meta.json'
            items.append(dict(
                mission='hst', instrument=inst, tile=t, band='f814w', suffix=t,
                meta=meta,
                build_cmd=[PY, str(HERE / '54_step3a_build_psf_model_hst.py'),
                           '--instrument', inst, '--tile', t],
                plot_cmd=[PY, str(HERE / '56_make_psf_qa_plots_single.py'),
                          '--instrument', inst, '--tile', t]))
    if 'euclid' in missions:
        for t in euclid_tiles():
            inst = 'euclid_vis'
            meta = WORK / inst / t / 'psf' / f'psf_{t}.meta.json'
            items.append(dict(
                mission='euclid', instrument=inst, tile=t, band='vis', suffix=t,
                meta=meta,
                build_cmd=[PY, str(HERE / '54_step3a_build_psf_model_euclid.py'),
                           '--instrument', inst, '--tile', t],
                plot_cmd=[PY, str(HERE / '56_make_psf_qa_plots_single.py'),
                          '--instrument', inst, '--tile', t]))
    if 'euclid_nisp' in missions:
        for t in nisp_tiles():
            for b in ('y', 'j', 'h'):                     # NISP Y/J/H
                inst = f'euclid_nisp_{b}'
                meta = WORK / inst / t / 'psf' / f'psf_{t}.meta.json'
                items.append(dict(
                    mission='euclid_nisp', instrument=inst, tile=t, band=b, suffix=t,
                    meta=meta,
                    build_cmd=[PY, str(HERE / '54_step3a_build_psf_model_euclid.py'),
                               '--band', b, '--tile', t],
                    plot_cmd=[PY, str(HERE / '56_make_psf_qa_plots_single.py'),
                              '--instrument', inst, '--tile', t]))
    if 'hsc' in missions:
        # tract-9813 deepCoadd patches 0-80 × grizy; build only what's on disk.
        for patch, b in sorted(hsc_avail(), key=lambda x: (int(x[0]), x[1])):
            inst = f'hsc_{b}'
            meta = WORK / inst / patch / 'psf' / f'psf_{patch}.meta.json'
            items.append(dict(
                mission='hsc', instrument=inst, tile=patch, band=b, suffix=patch,
                meta=meta,
                build_cmd=[PY, str(HERE / '54_step3a_build_psf_model_hsc.py'),
                           '--patch', patch, '--filter', b],
                plot_cmd=[PY, str(HERE / '56_make_psf_qa_plots_single.py'),
                          '--instrument', inst, '--tile', patch]))
    if 'lsdr10' in missions:
        # LS DR10 south bricks (Euclid-VIS scope, 92) × griz; only what's on disk.
        for brick, b in sorted(lsdr10_avail()):
            inst = f'lsdr10_{b}'
            meta = WORK / inst / brick / 'psf' / f'psf_{brick}.meta.json'
            items.append(dict(
                mission='lsdr10', instrument=inst, tile=brick, band=b, suffix=brick,
                meta=meta,
                build_cmd=[PY, str(HERE / '54_step3a_build_psf_model_lsdr10.py'),
                           '--brick', brick, '--filter', b],
                plot_cmd=[PY, str(HERE / '56_make_psf_qa_plots_single.py'),
                          '--instrument', inst, '--tile', brick]))
    return items


def append_timing(row):
    with _lock:
        new = not TIMING_CSV.exists()
        with open(TIMING_CSV, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=TIMING_COLS)
            if new:
                w.writeheader()
            w.writerow(row)


def run_item(it, force, clean):
    tag = f"{it['mission']}/{it['tile']}/{it['band']}"
    if it['meta'].exists() and not force:
        return ('skip', tag, 0)
    t0 = time.time()
    r1 = subprocess.run(it['build_cmd'], capture_output=True, text=True)
    t1 = time.time()
    status = 'ok'
    if r1.returncode != 0 or not it['meta'].exists():
        status = 'build_fail'
    else:
        r2 = subprocess.run(it['plot_cmd'], capture_output=True, text=True)
        if r2.returncode != 0:
            status = 'plot_fail'
    t2 = time.time()
    # free disk: drop the big intermediates once the model + plots succeeded
    if clean and status == 'ok':
        d = it['meta'].parent
        for pat in CLEAN_GLOBS:
            for f in d.glob(pat):
                try: f.unlink()
                except OSError: pass
    # pull summary numbers from the meta (if produced)
    m = {}
    if it['meta'].exists():
        try:
            m = json.loads(it['meta'].read_text())
        except Exception:
            pass
    append_timing({
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'mission': it['mission'], 'instrument': it['instrument'],
        'tile': it['tile'], 'band': it['band'], 'status': status,
        'build_s': round(t1 - t0, 1), 'plot_s': round(t2 - t1, 1),
        'total_s': round(t2 - t0, 1),
        'n_psf_stars': m.get('n_psf_stars', ''),
        'psf_accepted': m.get('psf_accepted', ''),
        'psf_chi2': (round(m['psf_chi2'], 3) if 'psf_chi2' in m else ''),
        'saturated': ('yes' if m.get('sat_onset_mag') is not None else
                      ('no' if 'sat_onset_mag' in m else '')),
        'sat_onset_mag': m.get('sat_onset_mag', ''),
    })
    return (status, tag, round(t2 - t0, 1))


def run_mission(mission, items, jobs, force, clean):
    todo = [it for it in items if force or not it['meta'].exists()]
    done = len(items) - len(todo)
    print(f'\n=== {mission.upper()}: {len(items)} items '
          f'({done} already done, {len(todo)} to run, {jobs}-way) ===', flush=True)
    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_item, it, force, clean): it for it in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            status, tag, secs = fut.result()
            if status == 'ok':
                ok += 1
            elif status.endswith('fail'):
                fail += 1
            mark = 'OK ' if status == 'ok' else status.upper()
            print(f'  [{i}/{len(todo)}] {mark:10s} {tag:24s} {secs:6.0f}s', flush=True)
    print(f'=== {mission.upper()} done: {ok} ok, {fail} fail, '
          f'wall {time.time()-t0:.0f}s ===', flush=True)
    return ok, fail


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--missions', default='jwst,hst,euclid',
                   help='comma list of jwst,hst,euclid,euclid_nisp,hsc,lsdr10')
    p.add_argument('--jobs', type=int, default=None,
                   help='override per-mission concurrency for all missions')
    p.add_argument('--force', action='store_true', help='reprocess even if done')
    p.add_argument('--no-site', action='store_true',
                   help='skip QA-site rebuild + saturation collect at end')
    p.add_argument('--keep-intermediates', action='store_true',
                   help='do NOT delete sci_/wht_/pass1/checkimages after each tile')
    p.add_argument('--exclude', default='',
                   help='comma list of tile:band to skip, e.g. A2:f115w,B4:f115w')
    p.add_argument('--bands', default='',
                   help='comma list of bands to keep (e.g. f115w,f150w); others dropped')
    p.add_argument('--tiles', default='',
                   help='comma list of tile ids to keep (e.g. 101541375 or A4); others dropped')
    return p.parse_args()


def main():
    args = parse_args()
    missions = [m.strip() for m in args.missions.split(',') if m.strip()]
    all_items = build_worklist(missions)
    excl = {e.strip() for e in args.exclude.split(',') if e.strip()}
    if excl:
        all_items = [it for it in all_items
                     if f"{it['tile']}:{it['band']}" not in excl]
        print(f'excluding: {sorted(excl)}')
    keepb = {b.strip() for b in args.bands.split(',') if b.strip()}
    if keepb:
        all_items = [it for it in all_items if it['band'] in keepb]
        print(f'bands kept: {sorted(keepb)}')
    keept = {t.strip() for t in args.tiles.split(',') if t.strip()}
    if keept:
        all_items = [it for it in all_items if it['tile'] in keept]
        print(f'tiles kept: {sorted(keept)}')
    print(f'mass production: missions={missions}  total items={len(all_items)}')
    print(f'timing CSV → {TIMING_CSV}')
    grand_ok = grand_fail = 0
    g0 = time.time()
    clean = not args.keep_intermediates
    for mission in missions:                       # mission-by-mission
        items = [it for it in all_items if it['mission'] == mission]
        jobs = args.jobs or DEFAULT_JOBS[mission]
        ok, fail = run_mission(mission, items, jobs, args.force, clean)
        grand_ok += ok; grand_fail += fail
    print(f'\n######## GRAND TOTAL: {grand_ok} ok, {grand_fail} fail, '
          f'wall {time.time()-g0:.0f}s ########')
    if not args.no_site:
        print('\n── rebuild QA site + collect saturation ──', flush=True)
        subprocess.run([PY, str(HERE / '55_make_psf_qa_site.py'), '--clean'])
        subprocess.run([PY, str(HERE / 'collect_saturation_levels.py')])


if __name__ == '__main__':
    main()
