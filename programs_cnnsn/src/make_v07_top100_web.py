"""Rank v07-scored sources by P and render the TOP-N candidate gallery for
purity inspection. JWST panels North-up. Header shows rank + P.

Reads:  csvfiles_sn/sn_candidates_v07_scored.parquet  (from infer_v07.py)
Writes: csvfiles_sn/v07_top100_gallery.csv + htmls/sn_search/v07_top100/index.html
"""
import sys, csv, re
from pathlib import Path
from importlib import import_module
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_cnnsn/src")
CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
SCORED  = CSV_DIR / "sn_candidates_v07_scored.parquet"
GAL_CSV = CSV_DIR / "v07_top100_gallery.csv"
TOPN = 100

# build the render CSV from the top-N scored sources
def build_csv():
    t = pq.read_table(SCORED).to_pandas()
    t = t.sort_values("P", ascending=False).head(TOPN).reset_index(drop=True)
    cols = ["id","primary_id","sn_ra","sn_dec","hst_tile","jwst_tile","euclid_tile",
            "telescope","best_band","best_snr","composite_confidence","z","disc_filter","source"]
    with GAL_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for i, r in t.iterrows():
            w.writerow(dict(id=str(r["primary_id"]), primary_id=str(r["primary_id"]),
                sn_ra=f"{r['ra']:.7f}", sn_dec=f"{r['dec']:.7f}",
                hst_tile=str(r["tile_hst"]), jwst_tile=str(r["tile_jwst"]), euclid_tile="",
                telescope="JWST", best_band="F115W", best_snr="0",
                composite_confidence=f"{r['P']:.4f}", z="", disc_filter="", source="v07cand"))
    print(f"top-{TOPN} render CSV written; P range {t['P'].min():.3f}–{t['P'].max():.3f}")
    return GAL_CSV

if __name__ == "__main__":
    cand_csv = build_csv()
    mod = import_module("make_v03_polished_web")
    mod.CAND_CSV = cand_csv
    mod.OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v07_top100")
    mod.TMP_DIR  = mod.OUT_DIR / "_raw"
    mod.DEFAULT_TOP = TOPN
    mod.N_WORKERS = 3
    _cat = {r["id"]: r for r in csv.DictReader(open(cand_csv))}
    _orig = mod.build_html
    def _build(top, top_n, n_total, status):
        h = _orig(top, top_n, n_total, status)
        h = re.sub(r"<meta http-equiv=['\"]refresh['\"][^>]*>", "", h)
        h = h.replace("sn_search v03", "v07 TOP-100")
        h = h.replace("v03 &mdash; live top", "v07 TOP-100 candidates &mdash;")
        banner = ("<div class='note' style='background:#eef;border-color:#33a'>"
                  "<b>v07 top-100 by CNN P (ranked).</b> Purity check: how many of "
                  "the highest-confidence picks are real transients? JWST N-up. "
                  "LOO completeness of this model is 57% (32/56).</div>")
        h = h.replace("<div class='note'>", banner + "<div class='note'>", 1)
        return h
    mod.build_html = _build
    sys.argv = [sys.argv[0], "--once", "--top", str(TOPN)]
    mod.main()
    # enrich headers with rank + P
    p = mod.OUT_DIR / "index.html"
    h = p.read_text()
    rank = {}
    for i, r in enumerate(csv.DictReader(open(cand_csv)), 1): rank[r["id"]] = (i, r["composite_confidence"])
    def repl(m):
        name = m.group(1); rk, P = rank.get(name, ("?","?"))
        return (f"<h3><b>#{rk}</b> {name} <span style='color:#33a'>P={P}</span> &nbsp;<span")
    h = re.sub(r"<h3>([^ <]+) &nbsp; <span", repl, h)
    p.write_text(h)
    print(f"v07 top-100 gallery → {p}")
