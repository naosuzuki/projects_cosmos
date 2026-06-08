"""Render the 25 v07-MISSED SNe (LOO P<0.5) for diagnostic inspection.

Goal: see WHY the CNN misses them — wrong position/tile? genuinely near
noise floor? host-dominated? — before spending more compute. JWST panels
are North-up (per CLAUDE.md §5.4). Header shows the LOO P score so we can
tell borderline (0.35-0.5) from hard-zero (<0.15) cases at a glance.

Output: htmls/sn_search/v07_misses/index.html (static).
"""
import sys, csv, re
from pathlib import Path
from importlib import import_module

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_cnnsn/src")
mod = import_module("make_v03_polished_web")

mod.CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/v07_misses_gallery.csv")
mod.OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v07_misses")
mod.TMP_DIR  = mod.OUT_DIR / "_raw"
mod.DEFAULT_TOP = 30
mod.N_WORKERS = 3

_cat = {r["id"]: r for r in csv.DictReader(open(mod.CAND_CSV))}

_orig = mod.build_html
def _build(top, top_n, n_total, status):
    html = _orig(top, top_n, n_total, status)
    html = re.sub(r"<meta http-equiv=['\"]refresh['\"][^>]*>", "", html)
    html = html.replace("sn_search v03", "v07 MISSES")
    html = html.replace("v03 &mdash; live top", "v07 LOO MISSES (P<0.5) &mdash;")
    banner = ("<div class='note' style='background:#fee;border-color:#d00'>"
              "<b>Diagnostic: the 25 SNe v07 leave-one-out MISSED (P&lt;0.5).</b> "
              "Hardest (lowest P) first. Looking for WHY: wrong position, near noise "
              "floor, or host-dominated. Header shows LOO P + z + filter.</div>")
    html = html.replace("<div class='note'>", banner + "<div class='note'>", 1)
    return html
mod.build_html = _build

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--once", "--top", "30"]
    mod.main()
    # enrich headers with P + z + filter
    p = mod.OUT_DIR / "index.html"
    h = p.read_text()
    def repl(m):
        name=m.group(1); r=_cat.get(name,{})
        P=r.get("composite_confidence",""); z=r.get("z",""); filt=r.get("disc_filter","")
        return (f"<h3><b>{name}</b> <span style='color:#d00'>P={P}</span>"
                f"<span style='color:#666'> &middot; z={z} &middot; {filt}</span> &nbsp;<span")
    h=re.sub(r"<h3>([^ <]+) &nbsp; <span", repl, h)
    p.write_text(h)
    print(f"enriched v07-misses gallery → {p}")
