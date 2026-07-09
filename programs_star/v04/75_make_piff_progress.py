#!/usr/bin/env python
"""
75_make_piff_progress.py — live progress feed for the Piff-panel batch (73_/74_).

Writes htmls/piff_progress.json every ~8 s for the dashboard page.  It is
SSD-only after a ONE-TIME My Book glob at start-up, so it does NOT compete
with the running driver for the USB disk.

Counting (per instrument dir, e.g. ps1_g, jwst_nircam_f115w):
  present0 = # psf_residuals_piff.png that already existed when THIS generator
             started         (one-time My Book glob)
  ok0      = # OK lines in loop.log at that same instant   (SSD)
  base     = present0 - ok0  = panels present at the DRIVER's launch
  worklist = # tiles queued this run  (/tmp/piff/work.txt, SSD)
  total    = base + worklist  (constant = every tile that will get a panel;
             this counts one line PER TILE, so provisional .psf files and the
             excluded euclid/hsc trees never leak in)
  done     = base + ok_run    (ok_run = live OK count from loop.log, SSD)

Run once in the background:
  nohup ./75_make_piff_progress.py >/tmp/piff/progress.log 2>&1 & disown
"""
from __future__ import annotations
import glob, json, re, subprocess, time
from collections import defaultdict
from pathlib import Path

WORK    = Path('/Volumes/exdisk1/data/photometry_v04')
OUT     = Path('/Users/suzuki/github/projects_cosmos/htmls/piff_progress.json')
LOG     = Path('/tmp/piff/loop.log')
WORKTXT = Path('/tmp/piff/work.txt')

HSC_ROOT = Path('/Users/suzuki/data/psf_v01')
GROUP_ORDER = ['HST', 'JWST', 'unWISE', 'HSC', 'PS1', 'LS DR10', 'SDSS']
GROUP_COLOR = {'HST': '#f2a13d', 'JWST': '#33c9b7', 'unWISE': '#b07cf0',
               'HSC': '#e879b8', 'PS1': '#4b9bff', 'LS DR10': '#5ec96b',
               'SDSS': '#ff6b6b'}
BAND_ORDER = {b: i for i, b in enumerate(
    ['F814W', 'F115W', 'F150W', 'F277W', 'F444W', 'W1', 'W2',
     'g', 'r', 'i', 'z', 'y', 'u'])}


def label(inst: str):
    if inst.startswith('hst'):            return 'HST', 'F814W'
    if inst.startswith('jwst_nircam_'):   return 'JWST', inst.split('_')[-1].upper()
    if inst.startswith('unwise_'):        return 'unWISE', inst.split('_')[-1].upper()
    if inst.startswith('hsc_'):           return 'HSC', inst.replace('hsc_', '').rstrip('2')
    if inst.startswith('ps1_'):           return 'PS1', inst.split('_')[-1]
    if inst.startswith('lsdr10_'):        return 'LS DR10', inst.split('_')[-1]
    if inst.startswith('sdss_'):          return 'SDSS', inst.split('_')[-1]
    return inst, ''


def ok_counts():
    """Per-instrument OK count + last activity from the SSD loop.log."""
    ok = defaultdict(int)
    fails = 0
    last = None
    if LOG.exists():
        for ln in LOG.read_text().splitlines():
            m = re.match(r'(OK|FAIL|PART)\s+\d+/\d+\s+(\S+)\s+(\S+)', ln)
            if not m:
                continue
            if m.group(1) == 'OK':
                ok[m.group(2)] += 1
            elif m.group(1) == 'FAIL':
                fails += 1
            last = (m.group(2), m.group(3))
    return ok, fails, last


def worklist_counts():
    c = defaultdict(int)
    if WORKTXT.exists():
        for ln in WORKTXT.read_text().splitlines():
            p = ln.split()
            if p:
                c[p[0]] += 1
    return c


def driver_alive():
    try:
        r = subprocess.run(['pgrep', '-f', '74_run_piff_all'],
                           capture_output=True, text=True, timeout=5)
        return bool(r.stdout.strip())
    except Exception:
        return False


def snapshot_base():
    """One-time: panels present at the driver's launch, per instrument."""
    present = defaultdict(int)
    for base_root in (WORK, HSC_ROOT):   # HSC panels live in the psf_v01 tree
        for p in base_root.glob('*/*/psf/psf_residuals_piff.png'):
            present[p.parts[-4]] += 1    # .../<inst>/<tile>/psf/<png>
    ok0, _, _ = ok_counts()
    insts = set(present) | set(ok0)
    return {i: max(0, present.get(i, 0) - ok0.get(i, 0)) for i in insts}


def main():
    base = snapshot_base()
    cyc = 0
    while True:
        cyc += 1
        # Re-baseline every ~160 s: `base` is a snapshot of panels-present at
        # the driver's launch, so a driver RELAUNCH (which resets loop.log)
        # would otherwise leave already-finished bands mis-counted until the
        # generator is restarted.  Re-snapshotting makes done == present again
        # (one light My Book metadata glob every ~20 cycles).
        if cyc % 20 == 0:
            base = snapshot_base()
        wl = worklist_counts()
        ok, fails, last = ok_counts()

        insts = set(base) | set(wl)
        bands = []
        for inst in insts:
            total = base.get(inst, 0) + wl.get(inst, 0)
            if total == 0:
                continue                 # euclid/hsc or non-served → skip
            grp, bnd = label(inst)
            done = max(0, min(base.get(inst, 0) + ok.get(inst, 0), total))
            bands.append({'inst': inst, 'group': grp, 'band': bnd,
                          'done': done, 'total': total,
                          'pct': round(100 * done / total, 1) if total else 0.0,
                          'color': GROUP_COLOR.get(grp, '#888')})
        bands.sort(key=lambda x: (GROUP_ORDER.index(x['group'])
                                  if x['group'] in GROUP_ORDER else 99,
                                  BAND_ORDER.get(x['band'], 99), x['band']))

        cur = None
        if last and driver_alive():
            g, b = label(last[0])
            cur = {'group': g, 'band': b, 'tile': last[1], 'inst': last[0],
                   'color': GROUP_COLOR.get(g, '#888')}

        od = sum(x['done'] for x in bands)
        ot = sum(x['total'] for x in bands)
        data = {
            'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_epoch': int(time.time()),
            'overall': {'done': od, 'total': ot,
                        'pct': round(100 * od / ot, 1) if ot else 0.0},
            'current': cur, 'driver_up': driver_alive(), 'fails': fails,
            'group_order': GROUP_ORDER, 'group_color': GROUP_COLOR,
            'bands': bands,
        }
        tmp = OUT.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(data))
        tmp.replace(OUT)                 # atomic swap
        time.sleep(8)


if __name__ == '__main__':
    main()
