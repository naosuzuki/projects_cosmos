#!/usr/bin/env python
"""
57_export_psf_qa_site.py — build a SELF-CONTAINED, deployable copy of a PSF-QA
static site with REAL (quantized) PNGs instead of symlinks to the external disk.

The live site under html/psf_qa/ symlinks every plot PNG to
/Volumes/exdisk1/data/photometry_v04/... — fast for local iteration, but it
breaks the moment exdisk1 is unplugged and cannot be cloned / served elsewhere.
This exporter mirrors the site tree into a standalone directory (its own git
repo): .html/.css are copied verbatim; every .png is dereferenced and
re-encoded as a 256-colour palette PNG.  matplotlib QA plots use few distinct
colours, so the palette PNG is visually indistinguishable from the original
yet ~1/5 the bytes (measured: 21–23%).  Same .png filenames → zero HTML edits.

Result: ~1.2 GB for the full 12-survey site, deployable by `git clone` + any
static webserver, no external disk required.

Usage:
  ./57_export_psf_qa_site.py                          # html/psf_qa -> ~/github/cosmos_psf_qa_site/psf_qa
  ./57_export_psf_qa_site.py --src html/diag_qa
  ./57_export_psf_qa_site.py --out /path/to/site/psf_qa --jobs 6 --colors 256
  ./57_export_psf_qa_site.py --match unWISE           # only PNGs whose path contains 'unWISE' (test)
  ./57_export_psf_qa_site.py --force                  # re-encode even if up to date
"""
from __future__ import annotations
import argparse, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path('/Users/suzuki/github/projects_cosmos')
DEFAULT_REPO = Path.home() / 'github' / 'cosmos_psf_qa_site'

_COLORS = 256   # per-worker global, set by initializer


def _init(colors):
    global _COLORS
    _COLORS = colors


def _convert_one(job):
    """Dereference a (possibly symlinked) PNG and write a palette-quantized copy.
    Returns (dst_bytes, src_bytes) on success, or (-1, message) on failure."""
    src, dst = job
    try:
        from PIL import Image
        src_bytes = os.path.getsize(src)            # follows symlink → real bytes
        im = Image.open(src)
        # flatten any alpha onto white (matplotlib PNGs are opaque, but be safe)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im.convert('RGBA')).convert('RGB')
        else:
            im = im.convert('RGB')
        q = im.quantize(colors=_COLORS, method=Image.FASTOCTREE)
        tmp = str(dst) + '.tmp'
        q.save(tmp, format='PNG', optimize=True)
        os.replace(tmp, dst)
        return (os.path.getsize(dst), src_bytes)
    except Exception as e:                          # noqa: BLE001
        return (-1, f'{src}: {e}')


def _needs_update(src, dst, force):
    if force or not dst.exists():
        return True
    try:
        return os.path.getmtime(src) > os.path.getmtime(dst)   # src follows symlink
    except OSError:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(PROJECT / 'html' / 'psf_qa'),
                    help='source site dir (default html/psf_qa)')
    ap.add_argument('--out', default=None,
                    help='destination dir (default ~/github/cosmos_psf_qa_site/<src-name>)')
    ap.add_argument('--colors', type=int, default=256)
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--match', default=None,
                    help='only convert PNGs whose relative path contains this substring')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    src = Path(a.src).resolve()
    if not src.is_dir():
        sys.exit(f'source not found: {src}')
    out = Path(a.out).resolve() if a.out else (DEFAULT_REPO / src.name)
    out.mkdir(parents=True, exist_ok=True)
    print(f'src  : {src}')
    print(f'out  : {out}')
    print(f'opts : colors={a.colors} jobs={a.jobs} '
          f'match={a.match or "(all)"} force={a.force}')

    # 1. walk the tree: copy .html/.css verbatim, queue .png for quantization
    png_jobs, copied, skipped = [], 0, 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (out / rel).mkdir(parents=True, exist_ok=True)
        for fn in files:
            s, d = Path(root) / fn, out / rel / fn
            if fn.lower().endswith('.png'):
                if a.match and a.match not in str(rel / fn):
                    continue
                if _needs_update(s, d, a.force):
                    png_jobs.append((str(s), str(d)))
                else:
                    skipped += 1
            else:
                shutil.copy2(s, d)
                copied += 1
    print(f'verbatim copied (html/css/…): {copied}')
    print(f'PNG to (re)encode: {len(png_jobs)}   (already up to date: {skipped})')

    if not png_jobs:
        print('nothing to encode; site is current.')
        _summarize(out)
        return

    # 2. quantize in a modest process pool (single HDD — don't oversubscribe I/O)
    t0 = time.time()
    src_tot = dst_tot = done = 0
    fails = []
    step = max(1, len(png_jobs) // 20)
    with Pool(a.jobs, initializer=_init, initargs=(a.colors,)) as pool:
        for dbytes, sbytes in pool.imap_unordered(_convert_one, png_jobs, chunksize=8):
            done += 1
            if dbytes < 0:
                fails.append(sbytes)
            else:
                dst_tot += dbytes
                src_tot += sbytes
            if done % step == 0 or done == len(png_jobs):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(png_jobs) - done) / rate if rate else 0
                print(f'  {done:5d}/{len(png_jobs)}  '
                      f'{dst_tot/1e6:7.1f} MB written  '
                      f'{rate:4.0f}/s  ETA {eta:4.0f}s', flush=True)
    if src_tot:
        print(f'encoded {done - len(fails)} PNGs: '
              f'{src_tot/1e9:.2f} GB → {dst_tot/1e9:.2f} GB '
              f'({dst_tot/src_tot*100:.0f}% of original) in {time.time()-t0:.0f}s')
    if fails:
        print(f'!! {len(fails)} failures:')
        for m in fails[:10]:
            print('   ', m)
    _summarize(out)


def _summarize(out):
    tot = n = 0
    for r, _d, fs in os.walk(out):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(r, f)); n += 1
            except OSError:
                pass
    print(f'=== deployable site at {out}')
    print(f'    {n} files, {tot/1e9:.2f} GB total')


if __name__ == '__main__':
    main()
