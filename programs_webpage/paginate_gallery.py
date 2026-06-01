"""Rebuild a polished-gallery as PAGINATED HTML (light, lazy-loaded).

A single page with 200 candidates x 5 PNGs (~155 MB) overwhelms the browser
(it renders top-down and appears to stall after a few). This splits the
gallery into pages of PER_PAGE candidates, each ~PER_PAGE*5 images, with a
page navigator and loading="lazy" so only on-screen images load.

Reuses the already-rendered polished_<key>_<panel>.png — no re-rendering.

Usage:
    python paginate_gallery.py --version v05 [--per-page 25] [--top 200]
    python paginate_gallery.py --version v04 [--per-page 25] [--top 200]
"""
import sys, csv, re, time
from pathlib import Path

CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
HTML_BASE = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search")

PANELS = ["jwst1", "jwst2", "hst", "euvis", "eunisp"]
PANEL_LABEL = {
    "jwst1":  "JWST F115/F150/F277", "jwst2": "JWST F150/F277/F444",
    "hst": "HST F814W", "euvis": "Euclid VIS", "eunisp": "Euclid NISP YJH",
}

NOTE = {
    "v05": ("#fee", "#d00",
            "<b>v05 = synthetic-injection CNN.</b> Only 1/17 known SNe recovered "
            "in-sky; the CNN fires on persistent host galaxies, so most top entries "
            "are host-galaxy false positives. See v04 (difference imaging) for the "
            "list that isolates real transients."),
    "v04": ("#e0f5e0", "#3a3",
            "<b>v04 = difference imaging.</b> HST F814W (2005-2008) subtracted from "
            "each modern-epoch band; a positive residual = transient flux absent in "
            "the HST epoch. Validated on known SNe (residual SNR 16-90)."),
    "v03": ("#fee", "#d00",
            "<b>v03 = first principled CNN pipeline.</b> Per-survey CNN with corrected "
            "labels (discovery-telescope only). Only 1/17 known SNe recovered in-sky; "
            "the CNN learns 'compact source on a host' rather than the transient "
            "signature. See v04 (difference imaging) for the validated list."),
}


def safe_pid(pid):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(pid))


def main():
    args = sys.argv[1:]
    version = "v05"
    per_page = 25
    top = 200
    if "--version" in args: version = args[args.index("--version")+1]
    if "--per-page" in args: per_page = int(args[args.index("--per-page")+1])
    if "--top" in args: top = int(args[args.index("--top")+1])

    cand_csv = CSV_DIR / f"tbl_sn_candidates_{version}.csv"
    # Auto-detect existing gallery dir (v05/v04 use polished_top200, v03 uses polished_top100)
    for cand_dir in ("polished_top200", "polished_top100"):
        if (HTML_BASE / version / cand_dir).exists():
            out_dir = HTML_BASE / version / cand_dir
            break
    else:
        out_dir = HTML_BASE / version / "polished_top200"
    out_dir.mkdir(parents=True, exist_ok=True)

    with cand_csv.open() as f:
        cands = list(csv.DictReader(f))[:top]
    # keep only candidates whose PNGs exist
    have = []
    for c in cands:
        key = safe_pid(c["primary_id"])
        if all((out_dir / f"polished_{key}_{p}.png").exists() for p in PANELS):
            have.append((key, c))
    n = len(have)
    n_pages = max(1, (n + per_page - 1) // per_page)
    print(f"{version}: {n} candidates with PNGs -> {n_pages} pages of {per_page}", flush=True)

    bg, bc, note_html = NOTE.get(version, ("#fff5b0", "#d4a017", ""))

    css = (
        "<style>body{font-family:serif;background:#fafafa;margin:0}"
        ".wrap{max-width:1500px;margin:0 auto;padding:20px}"
        ".src{margin-bottom:24px;padding:12px;background:#fff;border:1px solid #ddd}"
        ".src h3{margin:0 0 8px 0;font-size:14px;color:#222}"
        ".panels{display:flex;flex-wrap:wrap;gap:6px}"
        ".panel{text-align:center;font-size:11px;color:#666}"
        ".panel img{width:240px;height:240px;display:block;border:1px solid #eee;background:#111}"
        ".panel .lab{font-weight:bold;color:#333;margin-top:3px}"
        ".note{padding:10px 14px;margin:12px 0;border:1px solid " + bc + ";background:" + bg + "}"
        ".nav{position:sticky;top:0;background:#fafafaee;padding:10px 0;border-bottom:1px solid #ccc;"
        "font-family:monospace;font-size:13px;z-index:10}"
        ".nav a{margin:0 4px;text-decoration:none;color:#06c}"
        ".nav a.cur{font-weight:bold;color:#000;text-decoration:underline}"
        "h1{color:#333;font-size:20px}</style>"
    )

    def nav_bar(cur):
        links = []
        for p in range(1, n_pages + 1):
            fn = "index.html" if p == 1 else f"page{p}.html"
            cls = " class='cur'" if p == cur else ""
            links.append(f"<a href='{fn}'{cls}>{p}</a>")
        return "<div class='nav'><b>Pages:</b> " + " ".join(links) + "</div>"

    for page in range(1, n_pages + 1):
        chunk = have[(page-1)*per_page : page*per_page]
        html = ["<!doctype html><html><head>",
                f"<title>SN search {version} — gallery p{page}/{n_pages}</title>",
                css, "</head><body><div class='wrap'>",
                f"<h1>SN search {version} — polished gallery "
                f"(page {page}/{n_pages}, {n} candidates total)</h1>",
                f"<p style='color:#666;font-size:12px'>Generated {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"&middot; ranked #{(page-1)*per_page+1}–{(page-1)*per_page+len(chunk)} by composite confidence</p>",
                f"<div class='note'>{note_html}</div>",
                nav_bar(page)]
        for rank_in_page, (key, c) in enumerate(chunk):
            rank = (page-1)*per_page + rank_in_page + 1
            tel = c.get("telescope", "?"); bb = c.get("best_band", "?")
            bs = c.get("best_snr", "?"); cc = c.get("composite_confidence", "?")
            html.append(f"<div class='src'><h3>#{rank} &nbsp; {c.get('id','')} &nbsp; "
                        f"<span style='font-weight:normal;color:#666'>{tel} &middot; "
                        f"best {bb}@{bs}&sigma; &middot; comp_conf={cc} &middot; "
                        f"primary_id={c['primary_id']} &middot; "
                        f"RA={c.get('sn_ra','')} Dec={c.get('sn_dec','')}</span></h3>"
                        f"<div class='panels'>")
            for p in PANELS:
                html.append(f"<div class='panel'>"
                            f"<img loading='lazy' src='polished_{key}_{p}.png'>"
                            f"<div class='lab'>{PANEL_LABEL[p]}</div></div>")
            html.append("</div></div>")
        html.append(nav_bar(page))
        html.append("</div></body></html>")
        fn = "index.html" if page == 1 else f"page{page}.html"
        (out_dir / fn).write_text("\n".join(html))
    print(f"wrote {n_pages} pages to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
