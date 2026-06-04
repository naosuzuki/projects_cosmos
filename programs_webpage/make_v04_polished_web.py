"""Polished image gallery for v04 difference-imaging SN candidates.

Thin wrapper around make_v03_polished_web: reuses the cutout-extraction and
PNG-rendering logic, pointed at the v04 candidate CSV / output dir.

v04 is the difference-imaging list (HST F814W template subtracted from each
modern-epoch band) — it correctly isolates real transients (validated on
known SNe at SNR 16-90), so the top entries here should be genuine SN
candidates, not host-galaxy false positives.

Usage:
    python make_v04_polished_web.py --once --top 200
    python make_v04_polished_web.py --top 200        # loop, refresh every 30s
"""
import sys
from pathlib import Path
from importlib import import_module

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
mod = import_module("make_v03_polished_web")

mod.CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v04.csv")
mod.OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v04/polished_top200")
mod.TMP_DIR  = mod.OUT_DIR / "_raw"
mod.DEFAULT_TOP = 200

_orig_build_html = mod.build_html
def _build_html_v04(top_cands, top_n, n_total, run_status):
    html = _orig_build_html(top_cands, top_n, n_total, run_status)
    html = html.replace("sn_search v03", "sn_search v04")
    html = html.replace("v03 &mdash; live top", "v04 (difference imaging) &mdash; top")
    html = html.replace(
        "surviving v03 rules",
        "surviving v04 rules (HST-template difference imaging, &ge;2-band positive residual ≥5σ)")
    html = html.replace(
        "<div class='note'>",
        "<div class='note' style='background:#e0f5e0;border-color:#3a3'>"
        "<b>v04 = difference imaging.</b> HST F814W (2005-2008) subtracted from each "
        "modern-epoch band; a positive residual at the catalog position = transient "
        "flux that was NOT there in the HST epoch. Validated on known SNe (SNR 16-90). "
        "Top entries should be real transients.</div>"
        "<div class='note'>")
    return html
mod.build_html = _build_html_v04

if __name__ == "__main__":
    mod.main()
