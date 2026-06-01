"""MASTER SUPERNOVA gallery — the living, growing set of validated real SNe.

Currently 52 (17 user-known + 35 validated from the DeCoursey 65-catalog).
This is the curated truth set that feeds v06 CNN training. As more SNe are
validated they are appended to csvfiles_sn/master_sn_list.csv, recompiled to
master_sn_gallery.csv, and re-rendered here.

Each panel set: JWST blue/red, HST F814W, Euclid VIS, Euclid NISP, labeled
with name, source (known17/cat65), redshift, and discovery filter.

Output: htmls/sn_search/master_sn/index.html (+ page PNGs), static (no refresh).
"""
import sys, csv, re
from pathlib import Path
from importlib import import_module

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
mod = import_module("make_v03_polished_web")

mod.CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/master_sn_gallery.csv")
mod.OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/master_sn")
mod.TMP_DIR  = mod.OUT_DIR / "_raw"
mod.DEFAULT_TOP = 200
mod.N_WORKERS = 3

_cat = {r["id"]: r for r in csv.DictReader(open(mod.CAND_CSV))}

_orig = mod.build_html
def _build(top, top_n, n_total, status):
    html = _orig(top, top_n, n_total, status)
    html = re.sub(r"<meta http-equiv=['\"]refresh['\"][^>]*>", "", html)  # static page
    html = html.replace("sn_search v03", "MASTER SN list")
    html = html.replace("v03 &mdash; live top", "MASTER SUPERNOVA LIST &mdash;")
    n_k = sum(1 for r in _cat.values() if r.get("source")=="known17")
    n_c = sum(1 for r in _cat.values() if r.get("source")=="cat65")
    banner = (f"<div class='note' style='background:#e0f5e0;border-color:#3a3'>"
              f"<b>Master validated supernova set — {len(_cat)} SNe</b> "
              f"({n_k} user-known + {n_c} validated from DeCoursey 65-catalog). "
              f"This curated truth set feeds v06 CNN training. Growing as more "
              f"are validated.</div>")
    html = html.replace("<div class='note'>", banner + "<div class='note'>", 1)
    return html
mod.build_html = _build

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--once", "--top", "200"]
    mod.main()
    # enrich headers with source + z + filter
    p = mod.OUT_DIR / "index.html"
    h = p.read_text()
    def repl(m):
        name = m.group(1); r = _cat.get(name, {})
        src = r.get("source",""); z = r.get("z",""); filt = r.get("disc_filter","")
        zlab = f" &middot; z={z}" if z else ""
        flab = f" &middot; {filt}" if filt else ""
        tag = ("known17" if src=="known17" else "validated")
        color = "#06c" if src=="known17" else "#3a7"
        return (f"<h3><b>{name}</b> "
                f"<span style='color:{color}'>[{tag}]{zlab}{flab}</span> &nbsp;<span")
    h = re.sub(r"<h3>([^ <]+) &nbsp; <span", repl, h)
    p.write_text(h)
    print(f"enriched master gallery headers → {p}")
