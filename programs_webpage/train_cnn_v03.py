"""Step 5 v03: train per-survey CNNs with CORRECT labels.

Inputs:
  csvfiles_sn/training_set_v03.npz (built by make_training_set_v03.py)

Outputs:
  csvfiles_sn/cnn_models_v03.pt          (state_dicts + norms + thresholds)
  csvfiles_sn/training_summary_v03.txt   (per-survey recall + LOO table)
  csvfiles_sn/cnn_thresholds_v03.json    (P-thresholds at FPR=10⁻³)

Per-survey labels were FIXED upstream in make_training_set_v03 by using each
SN's discovery telescope (per-survey, NISP for the Euclid SN):
  hst:  3 known SNe + augmentations
  jwst: 13 known SNe + augmentations
  vis:  SKIPPED — 0 positives, no CNN trained for VIS
  nisp: 1 known SN + augmentations

Threshold tuning: FPR=0.001 on the negatives in each survey.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
from cnn_models_v03 import SmallCNN, best_device

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
NPZ_PATH = CSV_DIR / "training_set_v03.npz"
OUT_PT   = CSV_DIR / "cnn_models_v03.pt"
OUT_TXT  = CSV_DIR / "training_summary_v03.txt"
OUT_JSON = CSV_DIR / "cnn_thresholds_v03.json"

EPOCHS = 30
BATCH = 256
LR = 1e-3
TARGET_FPR = 1e-3


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def per_channel_stats(X):
    C = X.shape[1]
    med = np.zeros(C, dtype=np.float32); sig = np.ones(C, dtype=np.float32)
    for c in range(C):
        v = X[:, c]; v = v[np.isfinite(v)]
        if v.size == 0: continue
        m = np.median(v)
        mad = np.median(np.abs(v - m))
        med[c] = float(m); sig[c] = float(1.4826 * mad) if mad > 0 else 1.0
    return med, sig


def train_one(model, X, y, device, epochs=EPOCHS, batch=BATCH, lr=LR):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    ds = TensorDataset(X_t, y_t)
    n_pos = int(y_t.sum().item()); n_neg = len(y_t) - n_pos
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); l = crit(model(xb), yb); l.backward(); opt.step()
    return model


def predict(model, X, device, batch=512):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            out[i:i+batch] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def threshold_at_fpr(neg_scores, target_fpr=TARGET_FPR):
    return float(np.quantile(neg_scores, 1.0 - target_fpr)) if len(neg_scores) else 0.5


def main():
    log("=== Step 5 v03: per-survey CNN training (corrected labels) ===")
    t0 = time.time()
    z = np.load(NPZ_PATH, allow_pickle=True)
    device = best_device()
    log(f"Device: {device}")

    summary = []
    state_dicts = {}; norm_means = {}; norm_stds = {}; thresholds = {}
    per_sn_scores = {}

    for survey in ("hst", "jwst", "nisp"):
        if f"X_{survey}" not in z.files:
            log(f"  [{survey}] no data, skipping"); continue
        X = z[f"X_{survey}"]; y = z[f"y_{survey}"]
        groups = z[f"{survey}_group"]; ids = z[f"{survey}_id"]
        n_pos = int(y.sum()); n_total = len(y)
        log(f"\n--- {survey}: X={X.shape}  pos={n_pos}  neg={n_total-n_pos} ---")
        in_ch = X.shape[1]

        # Per-known-SN LOO eval
        known_ids = sorted({int(i) for g, i in zip(groups, ids) if g == "known"})
        log(f"  [{survey}] LOO over {len(known_ids)} known SN ids: {known_ids}")
        held_scores = {}
        if len(known_ids) >= 2:
            for held_id in known_ids:
                hold = np.array([(g in ("known","aug") and int(i)==held_id)
                                  for g, i in zip(groups, ids)])
                train_idx = np.where(~hold)[0]
                held_idx  = np.where(hold)[0]
                m = SmallCNN(in_ch=in_ch).to(device)
                med, sig = per_channel_stats(X[train_idx])
                m.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
                train_one(m, X[train_idx], y[train_idx], device)
                p = predict(m, X[held_idx], device)
                held_scores[held_id] = float(p.max())
        else:
            # too few SNe for LOO; train on full set, no held-out
            log(f"  [{survey}] only {len(known_ids)} known SN — skip LOO, train+report in-sample")

        # Train final model on full set
        final = SmallCNN(in_ch=in_ch).to(device)
        med, sig = per_channel_stats(X)
        final.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
        train_one(final, X, y, device)
        neg_idx = np.where(y == 0)[0]
        neg_scores = predict(final, X[neg_idx], device) if len(neg_idx) else np.zeros(0)
        thr = threshold_at_fpr(neg_scores, TARGET_FPR)
        thresholds[survey] = thr
        # For sub-2-SN surveys, score known positives in-sample with final model
        if len(known_ids) < 2:
            for known_id in known_ids:
                pos_mask = np.array([(g == "known" and int(i) == known_id)
                                     for g, i in zip(groups, ids)])
                if pos_mask.any():
                    p = predict(final, X[pos_mask], device)
                    held_scores[known_id] = float(p.max())
        recovered = sum(1 for v in held_scores.values() if v >= thr)
        recall = recovered / max(1, len(known_ids))
        log(f"  [{survey}] threshold@FPR={TARGET_FPR:.0e}: P>={thr:.4f}  "
            f"recall {recovered}/{len(known_ids)} ({100*recall:.1f}%)")
        per_sn_scores[survey] = held_scores
        state_dicts[survey] = {k: v.detach().cpu() for k, v in final.state_dict().items()}
        norm_means[survey] = med; norm_stds[survey] = sig
        summary.append((survey, len(known_ids), recovered, recall, thr, n_pos, n_total))

    bundle = {
        "models": state_dicts,
        "norm_mean": norm_means,
        "norm_std":  norm_stds,
        "thresholds": thresholds,
        "target_fpr": TARGET_FPR, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
    }
    torch.save(bundle, OUT_PT)
    OUT_JSON.write_text(json.dumps(thresholds, indent=2))
    log(f"\nSaved {OUT_PT} + {OUT_JSON}")

    lines = ["# Step 5 v03 training summary", f"# Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"# Device: {device}  Epochs: {EPOCHS}  Batch: {BATCH}",
             "", f"{'survey':>6} {'known':>6} {'rec':>6} {'recall':>8} {'P_thr':>8} {'pos':>6} {'total':>7}"]
    for s, kn, rec, recl, thr, npos, ntot in summary:
        lines.append(f"{s:>6} {kn:>6} {rec:>6} {100*recl:>7.1f}% {thr:>8.4f} {npos:>6} {ntot:>7}")
    lines.append("")
    for survey, scores in per_sn_scores.items():
        lines.append(f"\n=== {survey} per-SN LOO scores ===")
        thr = thresholds.get(survey, 0.5)
        for sid in sorted(scores):
            tag = "RECOVERED" if scores[sid] >= thr else "MISSED"
            lines.append(f"  SN {sid:>7d}  P={scores[sid]:.4f}  thr={thr:.4f}  {tag}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    log(f"Saved {OUT_TXT}")
    log(f"=== Step 5 v03 done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
