"""v07 inference: score all HST+JWST-coverage sources with the saved v07
deployment ensemble, rank by P, write a candidate table.

Tile-major IO: group sources by JWST tile; for each, open the 4 JWST band
mosaics + the matching HST tile once, extract 5-band σ-units stacks, run the
K-model ensemble (averaged P), checkpoint per tile. Footprint lookup resolves
authoritative paths (no glob). 14-ch input via train_v07.to_input.

Output: csvfiles_sn/sn_candidates_v07_scored.parquet  (primary_id, ra, dec,
        tiles, P, per-band snr-ish peak), then rank → top-N for the gallery.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import torch

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
import train_v07 as T          # reuse sigma_units, to_input, cutout_at, SNCNN, CHANNELS
from tile_lookup import TileResolver
from astropy.io import fits
from astropy.wcs import WCS

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
LOOK     = CSV_DIR / "fits_lookup_v03.parquet"
MODEL    = CSV_DIR / "cnn_v07.pt"
OUT_PARQ = CSV_DIR / "sn_candidates_v07_scored.parquet"
BATCH    = 1024
CUT = T.CUT


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    log("=== v07 inference ===")
    device = T.best_device(); log(f"device={device}")
    bundle = torch.load(MODEL, map_location="cpu", weights_only=False)
    states = bundle["ensemble_states"]; in_ch = bundle["in_ch"]
    log(f"loaded {len(states)}-model ensemble (LOO {bundle.get('loo_rec05','?')}/{bundle.get('loo_n','?')})")
    models = []
    for st in states:
        m = T.SNCNN(in_ch=in_ch).to(device); m.load_state_dict(st); m.eval(); models.append(m)

    lk = pq.read_table(LOOK).to_pandas()
    full = lk[(lk["in_hst"]) & (lk["in_jwst"])].reset_index(drop=True)
    resolver = TileResolver()

    # group by JWST tile (the limiting survey); each group shares 4 JWST bands
    by_jtile = defaultdict(list)
    for i, r in full.iterrows():
        by_jtile[str(r["tile_jwst"])].append(
            (str(r["primary_id"]), float(r["ra"]), float(r["dec"]), str(r["tile_hst"])))

    # ALSO add the 56 master-SN positions explicitly (id prefixed "KNOWN_"), so
    # all known SNe are scored even when offset from the nearest catalog row
    # (3 of 56 — 468896/19931/39020 — sit 2.4-4.6" from any 495K row). Resolve
    # each SN's JWST+HST tile by footprint geometry. This makes the scored set
    # a guaranteed superset of all 56, enabling complete purity/recovery checks.
    import csv as _csv
    n_known = 0
    for r in _csv.DictReader(open(CSV_DIR / "lookup_master56.csv")):
        sra, sdec = float(r["sn_ra"]), float(r["sn_dec"])
        jt, _ = resolver.resolve("jwst", "F115W", sra, sdec)
        ht, _ = resolver.resolve("hst", "F814W", sra, sdec)
        if jt is None or ht is None:
            log(f"  [warn] known SN {r['id']} not in JWST+HST footprint, skip"); continue
        by_jtile[jt].append((f"KNOWN_{r['id']}", sra, sdec, ht)); n_known += 1
    log(f"scoring {len(full):,} catalog sources + {n_known} known-SN positions")
    log(f"{len(by_jtile)} JWST tiles")

    jbands = ["F115W","F150W","F277W","F444W"]
    acc = {"primary_id": [], "ra": [], "dec": [], "P": [],
           "tile_hst": [], "tile_jwst": []}

    def open_sci(path):
        h = fits.open(path, memmap=True)
        sci = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
        return h, sci, WCS(sci.header)

    sorted_tiles = sorted(by_jtile.items())
    for ti, (jtile, srcs) in enumerate(sorted_tiles, 1):
        # resolve the 4 JWST band paths for this tile (one representative coord)
        ra0, dec0 = srcs[0][1], srcs[0][2]
        jpaths = {}
        for b in jbands:
            _, p = resolver.resolve("jwst", b, ra0, dec0)
            jpaths[b] = p
        if any(jpaths[b] is None for b in jbands):
            log(f"  tile {jtile}: missing JWST band path, skip {len(srcs)} srcs"); continue
        # open the 4 JWST mosaics once
        jhandles = {}
        ok = True
        for b in jbands:
            try:
                jhandles[b] = open_sci(jpaths[b])
            except Exception as e:
                log(f"  open fail {b} {jtile}: {e}"); ok = False; break
        if not ok:
            for hh,_,_ in jhandles.values():
                try: hh.close()
                except: pass
            continue

        # group this tile's sources by HST tile (HST opened per HST-tile)
        by_htile = defaultdict(list)
        for pid, ra, dec, htile in srcs:
            by_htile[htile].append((pid, ra, dec))

        tile_stacks = []; tile_meta = []
        for htile, hsrcs in by_htile.items():
            _, hp = resolver.resolve("hst", "F814W", hsrcs[0][1], hsrcs[0][2])
            if hp is None: continue
            try:
                hh, hsci, hwcs = open_sci(hp)
            except Exception:
                continue
            for pid, ra, dec in hsrcs:
                stk = np.zeros((5, CUT, CUT), dtype=np.float32); good = False
                c = T.cutout_at(hsci.data, hwcs, ra, dec)
                if c is not None and np.any(np.isfinite(c)):
                    stk[0] = T.sigma_units(c); good = True
                for bi, b in enumerate(jbands, start=1):
                    _, jsci, jwcs = jhandles[b]
                    cj = T.cutout_at(jsci.data, jwcs, ra, dec)
                    if cj is not None and np.any(np.isfinite(cj)):
                        stk[bi] = T.sigma_units(cj); good = True
                if good:
                    tile_stacks.append(stk); tile_meta.append((pid, ra, dec, htile))
            try: hh.close()
            except: pass
        for hh,_,_ in jhandles.values():
            try: hh.close()
            except: pass

        # batched ensemble inference for this tile
        if tile_stacks:
            X = np.stack(tile_stacks)
            Xi = T.to_input(X)
            with torch.no_grad():
                Pacc = np.zeros(len(Xi), dtype=np.float64)
                for b0 in range(0, len(Xi), BATCH):
                    xb = torch.tensor(Xi[b0:b0+BATCH], dtype=torch.float32, device=device)
                    psum = np.zeros(xb.shape[0], dtype=np.float64)
                    for m in models:
                        psum += torch.sigmoid(m(xb)).cpu().numpy()
                    Pacc[b0:b0+BATCH] = psum/len(models)
            for (pid, ra, dec, htile), P in zip(tile_meta, Pacc):
                acc["primary_id"].append(pid); acc["ra"].append(ra); acc["dec"].append(dec)
                acc["P"].append(float(P)); acc["tile_hst"].append(htile); acc["tile_jwst"].append(jtile)
        log(f"  tile [{ti}/{len(sorted_tiles)}] {jtile}: {len(tile_stacks)} scored "
            f"(total {len(acc['P']):,})  elapsed={time.time()-t0:.0f}s")
        # checkpoint each tile
        pq.write_table(pa.table(acc), OUT_PARQ, compression="zstd")

    log(f"=== done: scored {len(acc['P']):,} in {time.time()-t0:.0f}s → {OUT_PARQ} ===")
    P = np.array(acc["P"])
    if len(P):
        for thr in (0.9,0.8,0.7,0.5):
            log(f"  P>={thr}: {int((P>=thr).sum()):,}")


if __name__ == "__main__":
    main()
