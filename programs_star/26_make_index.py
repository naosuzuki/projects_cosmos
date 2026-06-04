#!/usr/bin/env python
"""
26_make_index.py — assemble htmls/star_v01/index.html linking to all v01
artefacts (footprints, DAO catalog summary, PM map, orphans page).

Pure metadata: reads only summary CSVs and existing artefact paths.
"""
from pathlib import Path
import pandas as pd

ROOT = Path('/Users/suzuki/github/projects_cosmos')
OUT  = ROOT / 'csvfiles_star'
HTML = ROOT / 'htmls' / 'star_v01'
HTML.mkdir(parents=True, exist_ok=True)


def safe_read_csv(p):
    try: return pd.read_csv(p)
    except Exception: return None


def main():
    parts = []
    parts.append('<h2>1. Footprints (Step 1)</h2>')
    foot = safe_read_csv(OUT / 'footprints' / 'tile_polygons.csv')
    if foot is not None:
        bn = foot.groupby('band').size().to_dict()
        parts.append('<ul>' + ''.join(f'<li>{b}: {n} tiles</li>' for b, n in bn.items()) + '</ul>')

    cov_files = sorted((OUT / 'inputs_with_coverage').glob('star_*_v01.csv'))
    if cov_files:
        parts.append('<p>Per-star coverage flags (cov_&lt;BAND&gt;, n_cov, has_all_9) in:</p><ul>')
        for f in cov_files:
            parts.append(f'<li><code>{f.relative_to(ROOT)}</code></li>')
        parts.append('</ul>')

    parts.append('<h2>2. DAO detections (Step 2)</h2>')
    summ = safe_read_csv(OUT / 'dao' / 'detection_summary.csv')
    if summ is not None:
        by = summ.groupby('band').agg(n_tiles=('tile', 'count'),
                                       n_src=('n_src', 'sum'),
                                       t_min=('t_sec', lambda v: v.sum()/60)).reset_index()
        parts.append('<table><thead><tr><th>band</th><th>tiles</th><th>n DAO srcs</th><th>wall t (min)</th></tr></thead><tbody>')
        for r in by.itertuples():
            parts.append(f'<tr><td>{r.band}</td><td>{r.n_tiles}</td><td>{r.n_src:,}</td><td>{r.t_min:.1f}</td></tr>')
        parts.append('</tbody></table>')

    parts.append('<h2>3. Star classification (Step 3)</h2>')
    cls = safe_read_csv(OUT / 'classify_summary.csv')
    if cls is not None:
        parts.append('<table><thead><tr><th>band</th><th>n DAO</th><th>n good star</th><th>n saturated</th></tr></thead><tbody>')
        for r in cls.itertuples():
            parts.append(f'<tr><td>{r.band}</td><td>{r.n_dao:,}</td><td>{r.n_good:,}</td><td>{r.n_sat:,}</td></tr>')
        parts.append('</tbody></table>')

    parts.append('<h2>4. Cross-matched pairs (Step 4)</h2><ul>')
    for p in ['pairs_HST_Euclid.parquet','pairs_JWST_Euclid.parquet','pairs_HST_JWST.parquet']:
        f = OUT / p
        if f.exists():
            try:
                n = len(pd.read_parquet(f))
                parts.append(f'<li>{p}: <b>{n:,}</b> pairs</li>')
            except Exception:
                parts.append(f'<li>{p}: present (unreadable)</li>')
    parts.append('</ul>')

    parts.append('<h2>5. Proper motion (Step 5)</h2>')
    if (HTML / 'pm_quiver.png').exists():
        parts.append('<p><img src="pm_quiver.png" style="max-width:900px;border:1px solid #ccc"></p>')
    if (HTML / 'pm_hist.png').exists():
        parts.append('<p><img src="pm_hist.png" style="max-width:700px;border:1px solid #ccc"></p>')
    pm = safe_read_csv(OUT / 'proper_motion_v01.csv')
    if pm is not None:
        n3 = (pm['n_epochs'] == 3).sum()
        parts.append(f'<p>Total PM rows: {len(pm):,}  &nbsp; 3-epoch: {n3:,}</p>')

    parts.append('<h2>6. Orphan candidates (Step 6)</h2>')
    if (HTML / 'orphans.html').exists():
        parts.append('<p><a href="orphans.html">→ orphans.html</a></p>')

    # Unified master catalog (post-step)
    master_pq = OUT / 'star_master_v01.parquet'
    if master_pq.exists():
        parts.append('<h2>7. Unified master catalog (27_unified_catalog)</h2>')
        try:
            mdf = pd.read_parquet(master_pq, columns=['n_epochs','n_bands_detected','pm_tot_mas_yr'])
            parts.append(
                '<ul>'
                f'<li>Total rows: <b>{len(mdf):,}</b></li>'
                f'<li>3-epoch PM (HST + JWST + Euclid): <b>{(mdf["n_epochs"]==3).sum():,}</b></li>'
                f'<li>2-epoch PM: <b>{(mdf["n_epochs"]==2).sum():,}</b></li>'
                f'<li>Median |μ| (3-epoch, |μ|&lt;200 mas/yr): '
                f'<b>{mdf.loc[(mdf["n_epochs"]==3) & (mdf["pm_tot_mas_yr"]<200), "pm_tot_mas_yr"].median():.2f}</b> mas/yr</li>'
                f'<li>Median n_bands_detected: <b>{mdf["n_bands_detected"].median():.1f}</b></li>'
                '</ul>'
                '<p>Files: <code>csvfiles_star/star_master_v01.parquet</code> '
                '(full) &nbsp; / &nbsp; <code>star_master_v01.csv</code> (21-col lite view)</p>'
            )
        except Exception as e:
            parts.append(f'<p>(could not summarise: {e})</p>')

    # Gaia DR3 validation (step 9)
    gaia_pq = OUT / 'gaia_match_v01.parquet'
    if gaia_pq.exists():
        parts.append('<h2>8. Gaia DR3 validation (29_gaia_validation)</h2>')
        try:
            gj = pd.read_parquet(gaia_pq, columns=['gaia_phot_g_mean_mag','n_epochs',
                                                    'pm_ra_mas_yr','pm_dec_mas_yr',
                                                    'gaia_pmra','gaia_pmdec'])
            sub3 = gj[gj['n_epochs'] == 3].dropna(subset=['gaia_pmra'])
            parts.append(
                '<ul>'
                f'<li>Matched stars: <b>{len(gj):,}</b>  (of ~17k Gaia DR3 G&lt;21.5 in COSMOS box)</li>'
                f'<li>3-epoch comparison: <b>{len(sub3):,}</b></li>'
            )
            if len(sub3):
                dra = (sub3["pm_ra_mas_yr"] - sub3["gaia_pmra"])
                ddc = (sub3["pm_dec_mas_yr"] - sub3["gaia_pmdec"])
                parts.append(f'<li>median Δμ<sub>α*</sub> = {dra.median():.2f}, '
                             f'σ = {dra.std():.2f} mas/yr</li>')
                parts.append(f'<li>median Δμ<sub>δ</sub>  = {ddc.median():.2f}, '
                             f'σ = {ddc.std():.2f} mas/yr</li>')
            parts.append('</ul>')
            for png in ['gaia_completeness.png','gaia_pm_compare.png','gaia_pos_compare.png']:
                if (HTML / png).exists():
                    parts.append(f'<p><img src="{png}" style="max-width:900px;border:1px solid #ccc"></p>')
        except Exception as e:
            parts.append(f'<p>(could not summarise Gaia: {e})</p>')

    # Clean stars (step 10)
    clean_pq = OUT / 'clean_stars_v01.parquet'
    if clean_pq.exists():
        parts.append('<h2>9. Clean stellar subset (30_clean_stars)</h2>')
        try:
            cj = pd.read_parquet(clean_pq, columns=['pm_tot_mas_yr','n_epochs','n_bands_detected'])
            parts.append(f'<p>Tight cuts (sharp 0.5-0.75, ≥4 bands, σ_µ &lt; 5, 2+ epochs): '
                         f'<b>{len(cj):,}</b> stars.  '
                         f'Median |µ| = {cj["pm_tot_mas_yr"].median():.2f} mas/yr.</p>'
                         '<p>Files: <code>csvfiles_star/clean_stars_v01.parquet</code> + .csv</p>')
            if (HTML / 'clean_pm_quiver.png').exists():
                parts.append('<p><img src="clean_pm_quiver.png" style="max-width:900px;border:1px solid #ccc"></p>')
            if (HTML / 'clean_pm_hist.png').exists():
                parts.append('<p><img src="clean_pm_hist.png" style="max-width:700px;border:1px solid #ccc"></p>')
        except Exception as e:
            parts.append(f'<p>(could not summarise clean: {e})</p>')

    # Orphan thumbnails (step 11)
    if (HTML / 'orphans_thumbs.html').exists():
        parts.append('<h2>10. Orphan thumbnails (31_orphan_thumbs)</h2>')
        parts.append('<p>Top 40 per detection mission, multi-band cutouts. '
                     '<a href="orphans_thumbs.html">→ orphans_thumbs.html</a></p>')

    # Refined orphans (step 8)
    refined_pq = OUT / 'orphans_refined.parquet'
    if refined_pq.exists():
        try:
            rdf = pd.read_parquet(refined_pq, columns=['real_orphan'])
            parts.append(
                f'<h2>11. Refined orphans (28_pixel_coverage_refine)</h2>'
                f'<p>After pixel-mask refinement: real_orphan count drops to '
                f'<b>{int(rdf["real_orphan"].sum()):,}</b>.  '
                f'File: <code>csvfiles_star/orphans_refined.parquet</code></p>'
            )
        except Exception:
            pass

    parts.append('<h2>Caveats / known v01 limitations</h2>'
                 '<ul>'
                 '<li><b>Footprints are bbox-only.</b> Real JWST chip gaps and rotated-frame corners are not subtracted, so some "orphans" are actually outside actual pixel coverage. Refine with <code>28_pixel_coverage_refine.py</code> (deferred).</li>'
                 '<li><b>Step 3 morphology cuts (G2-G4 from master CLAUDE.md) are loose</b> (sharp 0.4-0.85). Many compact galaxies are accepted as "good stars". To tighten, raise sharplo or restrict to sharp 0.5-0.75.</li>'
                 '<li><b>Saturation detector is conservative.</b> Few stars are flagged saturated because we use the brightest non-saturated peak as the threshold reference, which can be unreliable.</li>'
                 '<li><b>HST F814W has 11.5M DAO raw detections</b> (vs 1.2M for F115W) because of low background and many faint galaxies. Step 3 reduces to 730k good "stars".</li>'
                 '<li><b>Orphan count (1.3M) is an upper bound.</b> Tightening morphology cuts and using pixel-level masks will bring it down.</li>'
                 '</ul>')

    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>COSMOS star catalog v01 — index</title>'
        '<style>body{font-family:sans-serif;margin:20px;max-width:1100px} '
        'table{border-collapse:collapse;margin:8px 0} '
        'td,th{padding:3px 8px;border:1px solid #ccc;font-size:13px}</style>'
        '</head><body><h1>COSMOS star catalog v01</h1>'
        '<p>Build: 2026-05-26.  Output of programs_star/ pipeline 20–25.</p>'
        + ''.join(parts) +
        '</body></html>'
    )
    (HTML / 'index.html').write_text(html)
    print(f'Wrote {HTML/"index.html"}')


if __name__ == '__main__':
    main()
