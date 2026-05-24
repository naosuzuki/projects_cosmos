"""Build paginated SN-candidate eyeball-validation HTML pages.

For each candidate list (#5 Euclid-only, #6 JWST-only, #4 ACS-only) it
joins the three index CSVs (hst, euclid, jwst) on the 'seq' column and
emits index<NNNN>.html paginated at 100 candidates per page, with rows:

    [JWST1 RGB] [JWST2 RGB] [HST] [Euclid VIS] [Euclid NISP]

Output dir: /Users/suzuki/github/projects_cosmos/htmls/
Symlinks created on first run:
    htmls/jwst/   -> /Volumes/exdisk1/data/JWST/COSMOS_v0.8_png/
    htmls/hst/    -> /Volumes/exdisk1/data/HST/COSMOS_v2.0_png/
    htmls/euclid/ -> /Volumes/exdisk1/data/Euclid/COSMOS_DR1_png/

Usage:
  python 04_create_sn_web.py
  python 04_create_sn_web.py --list jwst_only
  python 04_create_sn_web.py --all
"""
import argparse
import csv
import math
import os
import sys
from pathlib import Path

# ============================================================
HTMLS_DIR = Path("/Users/suzuki/github/projects_cosmos/htmls")
CSV_DIR   = Path("/Users/suzuki/github/projects_cosmos/csvfiles")

LINK_TARGETS = {
    "jwst":   Path("/Volumes/exdisk1/data/JWST/COSMOS_v0.8_png"),
    "hst":    Path("/Volumes/exdisk1/data/HST/COSMOS_v2.0_png"),
    "euclid": Path("/Volumes/exdisk1/data/Euclid/COSMOS_DR1_png"),
}

LISTS = ["euclid_only", "jwst_only", "acs_only"]
PER_PAGE = 100
# ============================================================


def ensure_symlinks():
    HTMLS_DIR.mkdir(parents=True, exist_ok=True)
    for name, target in LINK_TARGETS.items():
        link = HTMLS_DIR / name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(target)
        print(f"  symlink {link} -> {target}")


def join_indices(list_name):
    """Inner-join the three per-survey CSVs on `seq`."""
    paths = {
        "hst":    CSV_DIR / f"sn_{list_name}_hst_index.csv",
        "euclid": CSV_DIR / f"sn_{list_name}_euclid_index.csv",
        "jwst":   CSV_DIR / f"sn_{list_name}_jwst_index.csv",
    }
    tables = {}
    for survey, p in paths.items():
        if not p.exists():
            print(f"  WARN: missing {p}")
            tables[survey] = {}
            continue
        with open(p) as fh:
            rd = csv.DictReader(fh)
            tables[survey] = {row["seq"]: row for row in rd}

    # union of seq keys (sorted numeric)
    seqs = set()
    for t in tables.values():
        seqs |= set(t.keys())
    seqs = sorted(seqs, key=lambda s: int(s))

    rows = []
    for seq in seqs:
        row = dict(seq=seq)
        for survey in ("hst", "euclid", "jwst"):
            if seq in tables[survey]:
                src = tables[survey][seq]
                for k, v in src.items():
                    if k == "seq":
                        continue
                    row[f"{survey}_{k}"] = v
        rows.append(row)
    return rows


CSS = """
<style>
body { font-family: serif; margin: 0; padding: 0; background: #fafafa; }
.header { position: sticky; top: 0; background: white;
          border-bottom: 1px solid #ddd; padding: 8px; z-index: 10; }
.pagination { display: inline-block; }
.pagination a {
  color: black; float: left; padding: 6px 12px; text-decoration: none;
  transition: background-color .2s; border: 1px solid #ccc; margin: 1px;
}
.pagination a.active { background-color: #4CAF50; color: white; }
.pagination a:hover:not(.active) { background-color: #ddd; }
.row { padding: 4px 8px; border-bottom: 1px solid #eee; }
.row h3 { margin: 6px 0 4px 0; font-size: 14px; color: #222; }
.imgs img { width: 18%; margin: 0 0.5%; vertical-align: top; }
.imgs span { display: inline-block; width: 18%; margin: 0 0.5%;
             text-align: center; vertical-align: top;
             padding: 60px 0; color: #888; font-style: italic;
             background: #f0f0f0; }
</style>
"""


def write_pagination(fh, list_name, n_pages, current):
    fh.write("<div class='pagination'>\n")
    fh.write(f"<a href='{list_name}_index0001.html'>&laquo;</a>\n")
    for n in range(1, n_pages + 1):
        cls = " class='active'" if n == current else ""
        fh.write(f"<a href='{list_name}_index{n:04d}.html'{cls}>{n}</a>\n")
    fh.write(f"<a href='{list_name}_index{n_pages:04d}.html'>&raquo;</a>\n")
    fh.write("</div>\n")


def write_row(fh, row):
    seq = row["seq"]
    # Pull header info from whichever CSV has it (Euclid usually most complete)
    id_     = row.get("euclid_id") or row.get("hst_id") or row.get("jwst_id", "")
    ra      = row.get("euclid_ra")  or row.get("hst_ra")  or row.get("jwst_ra", "")
    dec     = row.get("euclid_dec") or row.get("hst_dec") or row.get("jwst_dec", "")
    mag_eu  = row.get("euclid_mag", "")
    mag_hst = row.get("hst_mag", "")
    mag_jw  = row.get("jwst_mag", "")

    fh.write("<div class='row'>\n")
    fh.write(f"  <h3>#{seq}  ID={id_}  RA={ra}  Dec={dec}  "
             f"VIS={mag_eu}  F814W={mag_hst}  F150W={mag_jw}</h3>\n")
    fh.write("  <div class='imgs'>\n")
    panel_specs = [
        ("jwst",   row.get("jwst_jwst1_png", "")),
        ("jwst",   row.get("jwst_jwst2_png", "")),
        ("hst",    row.get("hst_hstpng", "")),
        ("euclid", row.get("euclid_euclid_vis_png", "")),
        ("euclid", row.get("euclid_euclid_nisp_png", "")),
    ]
    captions = ["JWST F115/F150/F277", "JWST F150/F277/F444",
                "HST F814W", "Euclid VIS", "Euclid NISP YJH"]
    for (subdir, png), cap in zip(panel_specs, captions):
        if png:
            fh.write(f"    <img src='./{subdir}/{png}' title='{cap}'>\n")
        else:
            fh.write(f"    <span>{cap}<br>(no image)</span>\n")
    fh.write("  </div>\n")
    fh.write("</div>\n")


def write_page(rows, list_name, page_idx, n_pages):
    out = HTMLS_DIR / f"{list_name}_index{page_idx:04d}.html"
    with open(out, "w") as fh:
        fh.write("<!doctype html>\n<html>\n<head>\n")
        fh.write(f"<title>SN candidates: {list_name} (page {page_idx}/{n_pages})</title>\n")
        fh.write(CSS)
        fh.write("</head>\n<body>\n")
        fh.write("<div class='header'>\n")
        fh.write(f"<h2 style='margin:4px 0;'>SN candidates: <code>{list_name}</code> "
                 f"&mdash; page {page_idx} / {n_pages}</h2>\n")
        write_pagination(fh, list_name, n_pages, page_idx)
        fh.write("</div>\n")
        for r in rows:
            write_row(fh, r)
        fh.write("<div class='header' style='border-top:1px solid #ddd; "
                 "border-bottom:none;'>\n")
        write_pagination(fh, list_name, n_pages, page_idx)
        fh.write("</div>\n")
        fh.write("</body>\n</html>\n")
    return out


def write_landing(decks):
    out = HTMLS_DIR / "index.html"
    with open(out, "w") as fh:
        fh.write("<!doctype html>\n<html>\n<head>\n")
        fh.write("<title>SN candidate decks</title>\n")
        fh.write(CSS)
        fh.write("</head>\n<body>\n<div style='padding:20px;'>\n")
        fh.write("<h1>SN candidate eyeball-validation decks</h1>\n")
        fh.write("<ul>\n")
        for ln, n in decks:
            fh.write(f"  <li><a href='{ln}_index0001.html'>{ln}</a> "
                     f"&mdash; {n} candidates</li>\n")
        fh.write("</ul>\n</div>\n</body>\n</html>\n")
    return out


def build_deck(list_name):
    rows = join_indices(list_name)
    if not rows:
        print(f"  [{list_name}] no rows, skipping")
        return 0
    n_pages = max(1, math.ceil(len(rows) / PER_PAGE))
    print(f"  [{list_name}] {len(rows)} candidates -> {n_pages} pages")
    for p in range(1, n_pages + 1):
        chunk = rows[(p-1)*PER_PAGE : p*PER_PAGE]
        write_page(chunk, list_name, p, n_pages)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", choices=LISTS, default=None,
                    help="Which deck to build; default = the one with CSVs.")
    ap.add_argument("--all", action="store_true",
                    help="Build all three decks.")
    args = ap.parse_args()

    ensure_symlinks()

    if args.all:
        targets = LISTS
    elif args.list:
        targets = [args.list]
    else:
        # auto-detect: build any deck whose CSV exists
        targets = [ln for ln in LISTS
                   if (CSV_DIR / f"sn_{ln}_euclid_index.csv").exists()
                   or (CSV_DIR / f"sn_{ln}_jwst_index.csv").exists()
                   or (CSV_DIR / f"sn_{ln}_hst_index.csv").exists()]
    print(f"Decks to build: {targets}")

    decks = []
    for ln in targets:
        n = build_deck(ln)
        if n > 0:
            decks.append((ln, n))

    if decks:
        landing = write_landing(decks)
        print(f"\nWrote landing -> {landing}")


if __name__ == "__main__":
    main()
