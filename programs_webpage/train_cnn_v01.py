"""Step 5 v01: train per-survey CNNs + LOO eval on the 17 known SNe.

Inputs:
  csvfiles_sn/training_set_v01.npz  (built by make_training_set_v01.py)

Outputs:
  csvfiles_sn/cnn_models_v01.pt          (state_dicts per survey)
  csvfiles_sn/training_summary_v01.txt   (LOO recall, AUC, suggested thresholds)
  csvfiles_sn/cnn_thresholds_v01.json    (per-survey P-thresholds @ FPR=0.001)

Method:
  - For each survey (hst/jwst/vis/nisp), prepare X (N,C,64,64) and y (N,).
  - Per-channel normalisation: median + 1.4826*MAD from training set.
  - 17-fold leave-one-out across the known SNe: in each fold, hold out all
    cutouts whose id matches the held-out SN id (including augmented copies)
    AND all whose group=="known_pos" with that id. Negatives stay in train.
  - Train SmallCNN for N_EPOCHS on remaining; record probability of held-out.
  - After all folds, refit on the full training set for the final model.
  - Pick threshold at FPR=0.001 on the negatives (CV-style):
       sort negative scores desc; take score at 99.9th percentile.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
from cnn_models_v01 import SmallCNN, make_models, best_device

CSV_DIR = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
NPZ_PATH = CSV_DIR / "training_set_v01.npz"
OUT_PT  = CSV_DIR / "cnn_models_v01.pt"
OUT_TXT = CSV_DIR / "training_summary_v01.txt"
OUT_JSON = CSV_DIR / "cnn_thresholds_v01.json"
STATUS_LOG = CSV_DIR / "run_v01_status.log"

EPOCHS = 30
BATCH = 256          # MPS prefers larger batches
LR = 1e-3
TARGET_FPR = 1e-3


def log(msg):
    """Print only — orchestrator handles file logging."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def per_channel_stats(X, frac_finite_min=0.5):
    """Per-channel median + MAD scaled to sigma. Robust to NaN."""
    C = X.shape[1]
    med = np.zeros(C, dtype=np.float32)
    sig = np.ones(C, dtype=np.float32)
    for c in range(C):
        v = X[:, c]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        m = np.median(v)
        mad = np.median(np.abs(v - m))
        med[c] = float(m)
        sig[c] = float(1.4826 * mad) if mad > 0 else 1.0
    return med, sig


def train_one(model, X_train, y_train, device, epochs=EPOCHS, batch=BATCH, lr=LR):
    """Train SmallCNN on (X_train, y_train). Returns trained model."""
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    ds = TensorDataset(X_t, y_t)
    n_pos = int(y_t.sum().item()); n_neg = len(y_t) - n_pos
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        # quick: don't log every epoch
    return model


def predict(model, X, device, batch=256):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            p = torch.sigmoid(model(xb)).cpu().numpy()
            out[i:i+batch] = p
    return out


def loo_eval_survey(name, X, y, groups, ids, augs, device):
    """Leave-one-SN-out evaluation.

    Held-out fold per known SN id: all rows whose (group in {known_pos,aug_pos}
    and id == held_id) are excluded from training; the held-out cutouts'
    predictions are recorded.

    Returns dict with per-held-id prediction (label, score, group).
    """
    known_ids = sorted(set(int(i) for g, i in zip(groups, ids) if g == "known_pos"))
    log(f"  [{name}] LOO over {len(known_ids)} known SN ids")
    if not known_ids:
        return dict(fold_results=[], all_neg_scores=np.zeros(0),
                    held_pos_scores=np.zeros(0), held_pos_labels=np.zeros(0),
                    held_pos_groups=np.zeros(0, dtype=object))
    held_pos_scores = []   # P(SN) for held-out positives (known + aug)
    held_pos_labels = []   # always 1
    held_pos_meta   = []   # (sn_id, group)
    all_neg_scores  = []   # final model's negative scores for threshold tuning
    fold_summaries  = []
    in_ch = X.shape[1]
    for held_id in known_ids:
        # mask: rows to HOLD OUT (= positive of this SN + its augments).
        hold = np.array([
            ((groups[i] in ("known_pos", "aug_pos")) and int(ids[i]) == held_id)
            for i in range(len(groups))
        ])
        train_idx = np.where(~hold)[0]
        held_idx  = np.where(hold)[0]
        if len(held_idx) == 0:
            continue
        # train
        m = SmallCNN(in_ch=in_ch).to(device)
        med, sig = per_channel_stats(X[train_idx])
        m.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
        train_one(m, X[train_idx], y[train_idx], device)
        # predict on held-out positives only
        p = predict(m, X[held_idx], device)
        for ii, pp in zip(held_idx, p):
            held_pos_scores.append(float(pp))
            held_pos_labels.append(int(y[ii]))
            held_pos_meta.append((int(ids[ii]), groups[ii]))
        fold_summaries.append(dict(held_id=held_id,
                                   n_held=int(len(held_idx)),
                                   p_mean=float(p.mean()),
                                   p_known_pos=float(p[
                                       np.array([groups[i] == "known_pos" for i in held_idx])
                                   ].max()) if any(groups[i] == "known_pos" for i in held_idx) else float("nan")))
    # final model on full train set for threshold + production inference
    log(f"  [{name}] training final model on full set ({len(X):,} samples)")
    final = SmallCNN(in_ch=in_ch).to(device)
    med, sig = per_channel_stats(X)
    final.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
    train_one(final, X, y, device)
    # score negatives with the final model
    neg_idx = np.where(y == 0)[0]
    if len(neg_idx):
        all_neg_scores = predict(final, X[neg_idx], device)
    else:
        all_neg_scores = np.zeros(0)
    return dict(
        fold_results=fold_summaries,
        held_pos_scores=np.array(held_pos_scores, dtype=np.float32),
        held_pos_labels=np.array(held_pos_labels, dtype=np.int8),
        held_pos_meta=held_pos_meta,
        all_neg_scores=np.array(all_neg_scores, dtype=np.float32),
        final_model=final,
        norm_mean=med, norm_std=sig,
    )


def threshold_at_fpr(neg_scores, target_fpr=TARGET_FPR):
    if len(neg_scores) == 0:
        return 0.5
    return float(np.quantile(neg_scores, 1.0 - target_fpr))


def main():
    log("=== Step 5 v01: train per-survey CNNs ===")
    t_start = time.time()
    log(f"Loading {NPZ_PATH} ...")
    z = np.load(NPZ_PATH, allow_pickle=True)
    device = best_device()
    log(f"Device: {device}")

    summary = []
    results = {}
    thresholds = {}
    state_dicts = {}

    for survey in ("hst", "jwst", "vis", "nisp"):
        X = z[f"X_{survey}"]
        y = z[f"y_{survey}"]
        if len(X) == 0:
            log(f"  [{survey}] no data, skipping")
            continue
        groups = z[f"{survey}_group"]
        ids    = z[f"{survey}_id"]
        augs   = z[f"{survey}_aug"]
        log(f"\n--- training survey: {survey}  (X={X.shape}, y={len(y)}, pos={int(y.sum())}) ---")
        t0 = time.time()
        res = loo_eval_survey(survey, X, y, groups, ids, augs, device)
        elapsed = time.time() - t0
        # recall: how many of the known_pos folds recovered the SN as P>thresh?
        thr = threshold_at_fpr(res["all_neg_scores"], TARGET_FPR)
        thresholds[survey] = thr
        known_recovered = 0
        known_held_ids = set()
        for s, lab, (sid, grp) in zip(res["held_pos_scores"], res["held_pos_labels"], res["held_pos_meta"]):
            if grp == "known_pos":
                known_held_ids.add(sid)
                if s >= thr:
                    known_recovered += 1
        n_known = len(known_held_ids)
        recall = known_recovered / n_known if n_known else 0.0
        log(f"  [{survey}] threshold@FPR={TARGET_FPR:.0e}: P>={thr:.4f}  "
            f"known recall: {known_recovered}/{n_known} ({100*recall:.1f}%)  "
            f"time={elapsed:.1f}s")
        results[survey] = res
        state_dicts[survey] = {k: v.detach().cpu() for k, v in res["final_model"].state_dict().items()}
        # remove the model object (not picklable through state dict only)
        summary.append((survey, n_known, known_recovered, recall, thr,
                        int(len(res["held_pos_scores"])), float(res["all_neg_scores"].mean() if len(res["all_neg_scores"]) else 0)))

    # ----- save model weights + norms -----
    out = {
        "models": state_dicts,
        "norm_mean": {s: results[s]["norm_mean"] for s in results},
        "norm_std":  {s: results[s]["norm_std"]  for s in results},
        "thresholds": thresholds,
        "target_fpr": TARGET_FPR,
        "epochs": EPOCHS,
        "batch":  BATCH,
        "lr":     LR,
    }
    torch.save(out, OUT_PT)
    log(f"\nSaved final models → {OUT_PT}")
    OUT_JSON.write_text(json.dumps(thresholds, indent=2))
    log(f"Saved thresholds → {OUT_JSON}")

    lines = []
    lines.append("# Step 5 v01: per-survey CNN training summary")
    lines.append(f"# Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"# Device: {device}")
    lines.append(f"# Epochs: {EPOCHS}  Batch: {BATCH}  LR: {LR}  TargetFPR: {TARGET_FPR}")
    lines.append("")
    lines.append(f"{'survey':>6} {'known':>6} {'recovered':>10} {'recall':>8} {'P_thr':>8} {'n_held':>8} {'mean_neg':>10}")
    for s, n_kn, n_rec, rec, thr, n_hp, neg_mean in summary:
        lines.append(f"{s:>6} {n_kn:>6} {n_rec:>10} {100*rec:>7.1f}% {thr:>8.4f} {n_hp:>8} {neg_mean:>10.4f}")
    lines.append("")
    lines.append("Per-known-SN per-survey LOO scores:")
    for survey, res in results.items():
        lines.append(f"\n=== {survey} ===")
        # collect per-known-SN best score
        by_id = {}
        for s, lab, (sid, grp) in zip(res["held_pos_scores"], res["held_pos_labels"], res["held_pos_meta"]):
            if grp == "known_pos":
                by_id[sid] = max(by_id.get(sid, 0.0), s)
        thr = thresholds.get(survey, 0.5)
        for sid in sorted(by_id):
            tag = "RECOVERED" if by_id[sid] >= thr else "MISSED   "
            lines.append(f"  SN {sid:>7d}  P={by_id[sid]:.4f}  thr={thr:.4f}  {tag}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    log(f"Saved summary → {OUT_TXT}")
    log(f"=== Step 5 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
