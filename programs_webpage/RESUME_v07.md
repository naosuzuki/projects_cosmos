# v07 CNN training — RESUME NOTE (2026-06-01)

Goal: leave-one-out recovery ≥70% (40/56) on the MASTER SN list, then run
inference + build candidate gallery for user visual inspection.

## Where things are (ALL durable — in git repo, NOT /tmp, NOT only-cache)
- Trainer: `programs_webpage/train_v07.py`  (committed)
- Tile lookup: `programs_webpage/{build_tile_footprints.py,tile_lookup.py}` + `csvfiles_sn/tile_footprints.parquet` (committed)
- 56-SN truth set: `csvfiles_sn/master_sn_list.csv` + adapter `csvfiles_sn/lookup_master56.csv` (committed)
- AT-RISK (rebuildable): σ-units cutout cache `/Volumes/My Book/data/cosmos_v06/_v07_cutouts_sigma.npz`.
  If the external disk doesn't remount, the trainer just re-extracts (~13 min, one IO pass). Not critical.

## How to resume
```
cd /Users/suzuki/github/projects_cosmos/programs_webpage
python train_v07.py            # fast 4-fold CV (config iteration), ~9-14 min
V07_FULL_LOO=1 python train_v07.py   # full 56-fold LOO (confirming run)
```
Each run prints `[HH:MM:SS]` per line and a final `LOO RESULT: N/56 ... (P>=0.5)`
plus a threshold sweep (P>=0.3..0.8). Report also at `csvfiles_sn/train_v07_report.txt`.

## Scoreboard (honest held-out CV on 56 master SNe)
| iter | change | P≥0.5 | P≥0.3 |
|------|--------|-------|-------|
| 1 | baseline 10ch (killed early) | ~27% | — |
| 2 | wider + on-host injection | 22/56 (39%) | — |
| 3 | + 4 diff channels (JWST−HST) | **26/56 (46%)** | **40/56 (71%)** ← best so far |
| 4 | +26 epochs +pos_weight=2 | 16/56 (29%) OVERFIT ✗ → reverted | 31/56 |
| 5 | avg+max global pooling | (pending) | (pending) |

## KEY DIAGNOSIS (the crux for next steps)
iter-3 DETECTS 40/56 (71%) but they cluster at P=0.3–0.5 → it's a
**separation/calibration gap, not a detection gap**. The remaining hard
population = faint **on-host cross-band JWST** SNe.
- pos_weight + more epochs made it WORSE (overfit) — do NOT repeat.
- iter-5 lever: avg+max pooling (preserve the centered compact peak the
  avg-pool was washing out). If it helps, next try: focal loss, per-fold
  threshold calibration on a synthetic val set, or a small Siamese HST↔JWST
  branch. Do NOT just lower the threshold to claim 70% — that's gaming.

## Config knobs in train_v07.py
- `NFOLD`: 4 (fast) / set env `V07_FULL_LOO=1` → 56 (true LOO)
- `K_ENS=5` ensemble; `epochs=18` (iter-3 sweet spot; 26 overfit)
- `to_input()` = 14ch: 5 σ-units asinh + 5 per-band compactness + 4 (JWST−HST) diff
- injection: amp ×U(0.35,1.5), 60% on-host (±3px), 40% spread (±8px)

## STATE AT 2026-06-01 10:50 (user left mid-run)
- Best config = iter-5 (avg+max pool, K=5, 18ep, 14ch). LOO = **32/56 (57%) @P≥0.5**,
  38-40/56 @P≥0.3. iter-6 (K=9) = 31/56 (ensemble saturated). pos_weight/more-epochs OVERFIT.
- A FINAL train+save run was launched 10:46 (pid was 44799). It does 4-fold LOO
  (~9min) then trains+saves a K=5 ensemble to `csvfiles_sn/cnn_v07.pt`. May have
  finished after the user left — CHECK: `ls -la csvfiles_sn/cnn_v07.pt`.
- Diagnostic gallery of the 25 LOO-misses: `htmls/sn_search/v07_misses/index.html`.
  Misses span z=0.01-3.55 (median 0.62); several LOW-z hard-zeros (adzu/aeaj/aead)
  are suspicious → inspect whether SN is on host core or position is off.

## NEXT ACTION after resume (in order)
1. `ls -la csvfiles_sn/cnn_v07.pt` — if present, the deployment ensemble is ready.
   If not, run `python train_v07.py` (cache warm → ~12 min) to produce it.
2. **Run inference for the user's TOP-100 request:**
   `python infer_v07.py`  → scores 495,532 HST+JWST sources, ~60-90 min,
   writes `csvfiles_sn/sn_candidates_v07_scored.parquet` (checkpoints per tile).
3. Rank by P, render top-100 gallery (adapt make_master_sn_web / make_v07_misses_web
   pointing at the scored parquet's top 100). Show user — they want to validate
   the best candidates and "learn from there".
4. Separately: user was about to inspect the v07_misses gallery (esp. low-z
   hard-zeros) to decide the next recovery lever (focal loss vs position-fix).
