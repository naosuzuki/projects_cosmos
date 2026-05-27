"""Visual inspection page for top v01 SN candidates.

For each of the top N candidates by composite_confidence, render a single
PNG showing 9 thumbnails (HST F814W; JWST F115/150/277/444W;
Euclid VIS/Y/J/H), with a cyan crosshair at the candidate position.

Writes:
  htmls/sn_search/v01/diagnostic/cand_*.png
  htmls/sn_search/v01/diagnostic/index.html
"""
import warnings; warnings.filterwarnings("ignore")
import sys, csv, time, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import recut_3_hst as R
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.nddata import Cutout2D

CAND_CSV = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn/tbl_sn_candidates_v01.csv")
OUT_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v01/diagnostic")
TOP_N    = 5
CUT_AS   = 6.0   # 6" cutout

BANDS = [
    ("HST F814W",  "hst",    "F814W"),
    ("JWST F115W", "jwst",   "f115w"),
    ("JWST F150W", "jwst",   "f150w"),
    ("JWST F277W", "jwst",   "f277w"),
    ("JWST F444W", "jwst",   "f444w"),
    ("Euclid VIS", "euclid", "VIS"),
    ("Euclid Y",   "euclid", "NIR-Y"),
    ("Euclid J",   "euclid", "NIR-J"),
    ("Euclid H",   "euclid", "NIR-H"),
]


def resolve_path(survey, band, hst_tile, jwst_tile, eu_tile):
    if survey == "hst":
        return f"{R.HST_DIR}/acs_I_030mas_{hst_tile}_sci.fits" if hst_tile else None
    if survey == "jwst":
        return R.resolve_jwst_path(jwst_tile, band) if jwst_tile else None
    return R.resolve_euclid_path(eu_tile, band) if eu_tile else None


def render_candidate(cand, out_png):
    cid = cand["id"]
    ra = float(cand["sn_ra"]); dec = float(cand["sn_dec"])
    hst_tile = cand.get("hst_tile") or None
    jwst_tile = cand.get("jwst_tile") or None
    eu_tile = cand.get("euclid_tile") or None

    fig, axes = plt.subplots(1, 9, figsize=(22, 3.0))
    for ax, (label, survey, band) in zip(axes, BANDS):
        path = resolve_path(survey, band, hst_tile, jwst_tile, eu_tile)
        if not path or not Path(path).exists():
            ax.text(0.5, 0.5, "no coverage", ha="center", va="center",
                    color="grey", fontsize=8)
            ax.set_title(label, fontsize=8); ax.axis("off"); continue
        try:
            with fits.open(path, memmap=True) as h:
                sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
                wcs = WCS(sci_hdu.header)
                ps = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
                pos = SkyCoord(ra*u.deg, dec*u.deg)
                n_pix = max(8, int(np.ceil(CUT_AS / ps)))
                c = Cutout2D(sci_hdu.data, pos, size=(n_pix, n_pix), wcs=wcs,
                             mode="partial", fill_value=np.nan)
                data = c.data.astype(np.float64)
            pos_data = np.where(np.isfinite(data) & (data > 0), data, 1e-6)
            img = np.sqrt(pos_data)
            vmin = float(np.nanpercentile(img, 5))
            vmax = float(np.nanpercentile(img, 99.5))
            if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0
            ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            ny, nx = img.shape
            cy, cx = ny // 2, nx // 2
            # crosshair (cyan, with a gap at center so the SN is visible)
            gap = max(2, int(0.2 / ps))
            arm = max(5, int(0.6 / ps))
            ax.plot([cx-arm-gap, cx-gap], [cy, cy], color="#0ff", lw=1.4)
            ax.plot([cx+gap, cx+arm+gap], [cy, cy], color="#0ff", lw=1.4)
            ax.plot([cx, cx], [cy-arm-gap, cy-gap], color="#0ff", lw=1.4)
            ax.plot([cx, cx], [cy+gap, cy+arm+gap], color="#0ff", lw=1.4)
            # 1" scale bar
            bar_px = 1.0 / ps
            ax.plot([nx*0.06, nx*0.06 + bar_px], [ny*0.10, ny*0.10],
                    color="white", lw=1.5)
            ax.text(nx*0.06 + bar_px/2, ny*0.14, '1"',
                    color="white", ha="center", va="bottom", fontsize=7)
            ax.set_title(label, fontsize=8)
            ax.axis("off")
        except Exception as e:
            ax.text(0.5, 0.5, f"err: {type(e).__name__}", ha="center", va="center",
                    color="red", fontsize=8)
            ax.set_title(label, fontsize=8); ax.axis("off")
    title = (f"{cid}  ·  primary_id={cand['primary_id']}  ·  "
             f"detected={cand['telescope']}  ·  RA={ra:.6f} Dec={dec:.6f}  ·  "
             f"comp_conf={cand['composite_confidence']}  ·  best_band={cand['best_band']} (SNR={cand['best_snr']})")
    fig.suptitle(title, fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = []
    with CAND_CSV.open() as f:
        cands = list(csv.DictReader(f))
    print(f"Loaded {len(cands)} candidates from {CAND_CSV.name}")
    # Already sorted by composite_confidence desc
    top = cands[:TOP_N]
    t0 = time.time()
    for cand in top:
        out_png = OUT_DIR / f"{cand['id']}.png"
        print(f"  rendering {cand['id']} ...", flush=True)
        render_candidate(cand, out_png)
    print(f"Rendered {len(top)} candidates in {time.time()-t0:.1f}s")

    # HTML
    html = ["<!doctype html><html><head><title>v01 top SN candidates — visual inspection</title>",
            "<style>body{font-family:serif;background:#fafafa;color:#222}",
            ".wrap{max-width:1700px;margin:0 auto;padding:20px}",
            "img{width:100%;display:block;border:1px solid #ccc;margin-bottom:18px}",
            "h1{color:#333} h2{margin-top:30px}",
            ".meta{font-family:monospace;font-size:12px;background:#eee;padding:6px 10px;border-radius:3px}",
            ".bug{background:#fee;border:1px solid #d00;padding:10px 14px;margin:14px 0}",
            "</style></head><body><div class='wrap'>",
            f"<h1>v01 — top {TOP_N} SN candidates &mdash; visual inspection</h1>",
            f"<p style='font-size:13px;color:#666'>Generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f" &mdash; from <code>tbl_sn_candidates_v01.csv</code> ({len(cands):,} candidates total).</p>",
            "<div class='bug'><b>Note on v01:</b> this run had a labeling bug "
            "(all 17 known SNe were marked positive in every survey, instead of "
            "only their discovery survey). Top candidates are likely dominated "
            "by host-galaxy false positives. v02 with corrected per-survey "
            "labels is pending.</div>",
            "<p>Each row: <b>HST F814W</b> | <b>JWST F115W / F150W / F277W / F444W</b> | "
            "<b>Euclid VIS / NIR-Y / NIR-J / NIR-H</b>. "
            "<span style='color:#0aa'>Cyan crosshair</span> = candidate SN position. "
            "1″ scale bar at bottom-left of each panel.</p>"]
    for cand in cands[:TOP_N]:
        cid = cand["id"]
        html.append(f"<h2>{cid}</h2>")
        html.append(f"<div class='meta'>")
        html.append(f"primary_id = {cand['primary_id']}<br>")
        html.append(f"telescope (detected by CNN) = <b>{cand['telescope']}</b><br>")
        html.append(f"RA = {cand['sn_ra']}, Dec = {cand['sn_dec']}<br>")
        html.append(f"hst_tile = {cand['hst_tile'] or '—'}, "
                    f"jwst_tile = {cand['jwst_tile'] or '—'}, "
                    f"euclid_tile = {cand['euclid_tile'] or '—'}<br>")
        html.append(f"best_band = {cand['best_band']} (SNR={cand['best_snr']}); "
                    f"composite_confidence = {cand['composite_confidence']}<br>")
        html.append(f"per-survey CNN P: hst={cand['cnn_conf_hst']}, "
                    f"jwst={cand['cnn_conf_jwst']}, vis={cand['cnn_conf_vis']}, "
                    f"nisp={cand['cnn_conf_nisp']}")
        html.append("</div>")
        html.append(f"<img src='{cid}.png'>")
    html.append("</div></body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(html))
    print(f"Wrote {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
