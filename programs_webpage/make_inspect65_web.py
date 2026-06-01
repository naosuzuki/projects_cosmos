"""Inspection gallery for the 65-SN DeCoursey COSMOS-Web catalog (sn_ori.txt).

Renders HST + JWST(blue/red) + Euclid VIS + Euclid NISP panels for each
catalog SN (55 of 65 fall in the full-coverage footprint), labeled with the
SN name, redshift, and discovery filter, so the user can visually confirm
which are genuinely real supernovae. The confirmed set becomes a VALIDATED
positive set for v06 training (legitimately expanding beyond the 17).

Reuses make_v03_polished_web's cutout extraction + panel rendering.
Output: htmls/sn_search/catalog65_inspect/index.html + page PNGs.
"""
import sys
from pathlib import Path
from importlib import import_module

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
mod = import_module("make_v03_polished_web")

mod.CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/sn_catalog65_inspect.csv")
mod.OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/catalog65_inspect")
mod.TMP_DIR  = mod.OUT_DIR / "_raw"
mod.DEFAULT_TOP = 65
mod.N_WORKERS = 3   # external-disk friendly

import csv as _csv
_cat = {r["id"]: r for r in _csv.DictReader(open(mod.CAND_CSV))}

_orig = mod.build_html
def _build(top, top_n, n_total, status):
    html = _orig(top, top_n, n_total, status)
    html = html.replace("sn_search v03", "catalog65 inspect")
    html = html.replace("v03 &mdash; live top", "DeCoursey 65-SN catalog &mdash; inspect")
    html = html.replace(
        "<div class='note'>",
        "<div class='note' style='background:#e8f0ff;border-color:#36c'>"
        "<b>Inspection gallery — mark the genuinely real SNe.</b> Each row is a "
        "catalog SN from sn_ori.txt with its redshift and discovery filter. "
        "Confirmed-real ones become validated positives for v06 training.</div>"
        "<div class='note'>")
    return html
mod.build_html = _build

# Inject z + filter into each candidate's header by patching the source label
_orig_render = mod.render_candidate_pngs  # keep (panels unchanged)

if __name__ == "__main__":
    # one-shot render of all 65 (55 renderable)
    sys.argv = [sys.argv[0], "--once", "--top", "65"]
    mod.main()
