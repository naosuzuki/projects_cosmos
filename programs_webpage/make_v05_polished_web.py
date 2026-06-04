"""Polished image gallery for v05 SN candidates.

Thin wrapper around make_v03_polished_web: reuses all the cutout-extraction
and PNG-rendering logic, but points the input CSV / output dir at v05.

NOTE: v05 candidates are dominated by host-galaxy false positives (only
1/17 known SNe recovered, because the single-epoch CNN fires on the
persistent host). This gallery is for completeness / comparison; v04
(difference imaging) is the list that actually isolates real transients.

Usage:
    python make_v05_polished_web.py --once --top 200
    python make_v05_polished_web.py --top 200        # loop, refresh every 30s
"""
import sys
from pathlib import Path
from importlib import import_module

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
mod = import_module("make_v03_polished_web")

# Override the module-level paths so all its functions write to v05
mod.CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v05.csv")
mod.OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v05/polished_top200")
mod.TMP_DIR  = mod.OUT_DIR / "_raw"
mod.DEFAULT_TOP = 200

# Patch the page header so it says v05 (the builder reads these at call time)
_orig_build_html = mod.build_html
def _build_html_v05(top_cands, top_n, n_total, run_status):
    html = _orig_build_html(top_cands, top_n, n_total, run_status)
    html = html.replace("sn_search v03", "sn_search v05")
    html = html.replace("v03 &mdash; live top", "v05 (synthetic-injection CNN) &mdash; top")
    html = html.replace(
        "surviving v03 rules",
        "surviving v05 rules (CNN P-thresholds @ 80% LOO recall + 1-of-3 telescope)")
    # Honest caveat banner
    html = html.replace(
        "<div class='note'>",
        "<div class='note' style='background:#fee;border-color:#d00'>"
        "<b>Caveat:</b> v05 recovered only 1/17 known SNe in-sky &mdash; the CNN "
        "fires on persistent host galaxies, so most real SNe trip the 1-of-3 rule "
        "and the top entries are dominated by host-galaxy false positives. See v04 "
        "(difference imaging) for the list that isolates real transients.</div>"
        "<div class='note'>")
    return html
mod.build_html = _build_html_v05

if __name__ == "__main__":
    mod.main()
