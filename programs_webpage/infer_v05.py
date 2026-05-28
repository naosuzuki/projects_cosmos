"""Step 6 v05: inference using v05 (synthetic-injection-trained) CNNs.

Re-uses v03's survey_worker / cutout_and_aper / update_outputs infrastructure
but with the v05 model bundle. Writes per-survey partials to
csvfiles_sn/_partial/infer_v05_<survey>.parquet, and consolidates to
csvfiles_sn/tbl_sn_candidates_v05.csv with the 1-of-3 telescope rule.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, os, json
from pathlib import Path
from multiprocessing import Pool, Process, Event, set_start_method
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
# Re-use v03's worker + helpers, but redirect partial path to _v05_
from infer_sn_v03 import (cutout_and_aper, atomic_write_parquet, FWHM_AS,
                          HST_BANDS, JWST_BANDS, VIS_BANDS, NISP_BANDS,
                          CUT_PIX, BATCH_CNN, SURVEY_BANDS, CSV_BANDS,
                          update_outputs as _v03_update_outputs)
import recut_3_hst as R

CSV_DIR   = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
PART_DIR  = CSV_DIR / "_partial"
HTML_DIR  = Path("/Users/suzuki/github/projects_cosmos/htmls/sn_search/v05")
LOOK      = CSV_DIR / "fits_lookup_v03.parquet"
MODELS    = CSV_DIR / "cnn_models_v05.pt"
OUT_CSV   = CSV_DIR / "tbl_sn_candidates_v05.csv"

WATCH_INTERVAL = 30


def log(msg, prefix="V05"):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix}: {msg}", flush=True)


def survey_worker_v05(args):
    """Same as v03's survey_worker but writes to _v05_ partial parquet."""
    import warnings; warnings.filterwarnings("ignore")
    import sys, os, time
    for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
    import recut_3_hst as R_local
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    import pyarrow as pa
    import torch
    from cnn_models_v03 import SmallCNN
    from collections import defaultdict
    from infer_sn_v03 import cutout_and_aper, atomic_write_parquet, FWHM_AS

    survey = args["survey"]; in_ch = args["in_ch"]
    sources = args["sources"]
    bands = SURVEY_BANDS[survey]
    kind  = {"hst":"hst","jwst":"jwst","vis":"euclid","nisp":"euclid"}[survey]
    band_specs = {"hst":HST_BANDS,"jwst":JWST_BANDS,"vis":VIS_BANDS,"nisp":NISP_BANDS}[survey]
    checkpoint_path = PART_DIR / f"infer_v05_{survey}.parquet"
    PART_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = SmallCNN(in_ch=in_ch).to(device)
    model.load_state_dict(args["model_state"])
    model.set_norm(torch.tensor(args["norm_mean"], device=device),
                   torch.tensor(args["norm_std"],  device=device))
    model.eval()

    by_tile = defaultdict(list)
    for (idx, ra, dec, tile) in sources:
        by_tile[tile].append((idx, ra, dec))

    N_total = len(sources)
    acc = {"idx": [], "P": []}
    for b in bands:
        acc[f"mag_{b}"]   = []
        acc[f"snr_{b}"]   = []
        acc[f"sharp_{b}"] = []
        acc[f"rnd1_{b}"]  = []
        acc[f"rnd2_{b}"]  = []
        acc[f"sep_{b}"]   = []
        acc[f"nanfrac_{b}"] = []

    t0 = time.time()
    sorted_tiles = sorted(by_tile.items())
    for ti, (tile, tile_sources) in enumerate(sorted_tiles):
        hdus = {}; wcss = {}; hdrs = {}
        for blabel, R_band in band_specs:
            if survey == "hst":
                path = f"{R_local.HST_DIR}/acs_I_030mas_{tile}_sci.fits"
            elif survey == "jwst":
                path = R_local.resolve_jwst_path(tile, R_band.lower())
            else:
                path = R_local.resolve_euclid_path(tile, R_band)
            if not path or not os.path.exists(path):
                hdus[blabel] = None; continue
            try:
                h = fits.open(path, memmap=True)
                sci_hdu = h["SCI"] if "SCI" in [x.name for x in h] else h[0]
                hdus[blabel] = sci_hdu
                wcss[blabel] = WCS(sci_hdu.header)
                hdrs[blabel] = sci_hdu.header
            except Exception as e:
                print(f"[{survey}] open failed {Path(path).name}: {e}", flush=True)
                hdus[blabel] = None
        ref_wcs = next((w for w in wcss.values() if w is not None), None)
        if ref_wcs is None:
            for gidx, _, _ in tile_sources:
                acc["idx"].append(gidx); acc["P"].append(float("nan"))
                for b in bands:
                    acc[f"mag_{b}"].append(-1.0); acc[f"snr_{b}"].append(0.0)
                    acc[f"sharp_{b}"].append(float("nan"))
                    acc[f"rnd1_{b}"].append(float("nan"))
                    acc[f"rnd2_{b}"].append(float("nan"))
                    acc[f"sep_{b}"].append(float("nan"))
                    acc[f"nanfrac_{b}"].append(1.0)
            continue
        try:
            ras  = np.array([s[1] for s in tile_sources])
            decs = np.array([s[2] for s in tile_sources])
            _, ys = ref_wcs.all_world2pix(ras, decs, 0)
            order = np.argsort(np.where(np.isfinite(ys), ys, 0.0))
        except Exception:
            order = np.arange(len(tile_sources))
        ordered = [tile_sources[i] for i in order]
        for bstart in range(0, len(ordered), BATCH_CNN):
            batch = ordered[bstart:bstart + BATCH_CNN]
            cutouts = np.full((len(batch), in_ch, CUT_PIX, CUT_PIX), np.nan, dtype=np.float32)
            mag_b   = {b: [] for b in bands}; snr_b = {b: [] for b in bands}
            sharp_b = {b: [] for b in bands}
            rnd1_b  = {b: [] for b in bands}; rnd2_b = {b: [] for b in bands}
            sep_b   = {b: [] for b in bands}; nf_b = {b: [] for b in bands}
            for k, (gidx, ra_v, dec_v) in enumerate(batch):
                for ci, (blabel, _) in enumerate(band_specs):
                    if hdus[blabel] is None:
                        mag_b[blabel].append(-1.0); snr_b[blabel].append(0.0)
                        sharp_b[blabel].append(float("nan")); rnd1_b[blabel].append(float("nan"))
                        rnd2_b[blabel].append(float("nan")); sep_b[blabel].append(float("nan"))
                        nf_b[blabel].append(1.0); continue
                    c, mag, snr, sharp, rnd1, rnd2, sep_sn, nan_frac = cutout_and_aper(
                        hdus[blabel].data, hdrs[blabel], wcss[blabel],
                        ra_v, dec_v, FWHM_AS[blabel], kind=kind)
                    cutouts[k, ci] = c
                    mag_b[blabel].append(mag); snr_b[blabel].append(snr)
                    sharp_b[blabel].append(sharp); rnd1_b[blabel].append(rnd1)
                    rnd2_b[blabel].append(rnd2); sep_b[blabel].append(sep_sn)
                    nf_b[blabel].append(nan_frac)
            with torch.no_grad():
                xb = torch.tensor(cutouts, dtype=torch.float32, device=device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
            for k, (gidx, _, _) in enumerate(batch):
                acc["idx"].append(gidx)
                acc["P"].append(float(p[k]))
                for b in bands:
                    acc[f"mag_{b}"].append(mag_b[b][k])
                    acc[f"snr_{b}"].append(snr_b[b][k])
                    acc[f"sharp_{b}"].append(sharp_b[b][k])
                    acc[f"rnd1_{b}"].append(rnd1_b[b][k])
                    acc[f"rnd2_{b}"].append(rnd2_b[b][k])
                    acc[f"sep_{b}"].append(sep_b[b][k])
                    acc[f"nanfrac_{b}"].append(nf_b[b][k])
        for blabel in list(hdus):
            try:
                if hdus[blabel] is not None:
                    hdus[blabel].fileinfo()["file"].close()
            except Exception: pass
        table = pa.table({k: np.array(v) for k, v in acc.items()})
        try: atomic_write_parquet(table, checkpoint_path)
        except Exception as e: print(f"[{survey}] checkpoint write failed: {e}", flush=True)
        print(f"[{survey}-v05] tile [{ti+1}/{len(sorted_tiles)}] {tile} acc={len(acc['idx']):,}/"
              f"{N_total:,} elapsed={time.time()-t0:.1f}s", flush=True)
    table = pa.table({k: np.array(v) for k, v in acc.items()})
    atomic_write_parquet(table, checkpoint_path)
    print(f"[{survey}-v05] DONE elapsed={time.time()-t0:.1f}s wrote {checkpoint_path.name}", flush=True)
    return dict(survey=survey, elapsed=time.time()-t0, n=len(acc["idx"]))


def main():
    t_start = time.time()
    log("=== v05 inference (uses synthetic-injection-trained CNNs) ===")
    import torch
    bundle = torch.load(MODELS, map_location="cpu", weights_only=False)
    thresholds = bundle["thresholds"]
    log(f"Thresholds (v05): {thresholds}")

    lk = pq.read_table(LOOK)
    pid = np.array(lk["primary_id"].to_pylist(), dtype=object)
    ra  = lk["ra"].to_numpy(zero_copy_only=False)
    dec = lk["dec"].to_numpy(zero_copy_only=False)
    th  = np.array(lk["tile_hst"].to_pylist(), dtype=object)
    tj  = np.array(lk["tile_jwst"].to_pylist(), dtype=object)
    te  = np.array(lk["tile_euclid"].to_pylist(), dtype=object)
    in_h = np.array(lk["in_hst"].to_pylist(), dtype=bool)
    in_j = np.array(lk["in_jwst"].to_pylist(), dtype=bool)
    in_e = np.array(lk["in_euclid"].to_pylist(), dtype=bool)
    N = len(pid)
    log(f"{N:,} candidates")

    PART_DIR.mkdir(parents=True, exist_ok=True)
    for sv in ("hst","jwst","vis","nisp"):
        cp = PART_DIR / f"infer_v05_{sv}.parquet"
        if cp.exists(): cp.unlink()

    survey_specs = [
        ("hst",  1, in_h, th),
        ("jwst", 4, in_j, tj),
        ("vis",  1, in_e, te),
        ("nisp", 3, in_e, te),
    ]
    jobs = []
    for survey, in_ch, in_mask, tiles in survey_specs:
        if survey not in bundle["models"]:
            log(f"  [warn] no v05 model for {survey}, skip"); continue
        idxs = np.where(in_mask)[0]
        sources = [(int(i), float(ra[i]), float(dec[i]), str(tiles[i])) for i in idxs if tiles[i]]
        log(f"  {survey}: {len(sources):,} sources")
        jobs.append(dict(
            survey=survey, in_ch=in_ch,
            model_state={k: v.detach().cpu() for k, v in bundle["models"][survey].items()},
            norm_mean=bundle["norm_mean"][survey], norm_std=bundle["norm_std"][survey],
            sources=sources,
        ))

    try: set_start_method("spawn", force=True)
    except RuntimeError: pass
    log(f"Dispatching {len(jobs)} survey workers ...")
    t_par = time.time()
    with Pool(processes=len(jobs)) as pool:
        results = pool.map(survey_worker_v05, jobs)
    log(f"All v05 workers done in {time.time()-t_par:.1f}s")
    for r in results:
        log(f"  {r['survey']} elapsed={r['elapsed']:.1f}s n={r['n']:,}")
    log(f"=== v05 inference total {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
