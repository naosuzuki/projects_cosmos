"""Step 5 v05: train per-survey CNNs with recall-based thresholds.

v05 fix vs v03: instead of FPR=1e-3 threshold (which the small per-survey
training set made overly strict and unreliable), use the 20th-percentile
score of the REAL-SN positives as the threshold. This ensures 80% recall
on the labeled SNe by construction; false-positive rate is whatever falls
out (typically a few percent — acceptable since downstream 1-of-N rule +
morphology gates filter further).

Inputs: csvfiles_sn/training_set_v05.npz
Outputs: csvfiles_sn/cnn_models_v05.pt
         csvfiles_sn/training_summary_v05.txt
         csvfiles_sn/cnn_thresholds_v05.json
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
NPZ_PATH = CSV_DIR / "training_set_v05.npz"
OUT_PT   = CSV_DIR / "cnn_models_v05.pt"
OUT_TXT  = CSV_DIR / "training_summary_v05.txt"
OUT_JSON = CSV_DIR / "cnn_thresholds_v05.json"

EPOCHS = 30
BATCH = 256
LR = 1e-3
TARGET_RECALL = 0.80   # use 20th-percentile of positive scores as threshold

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def per_channel_stats(X):
    C = X.shape[1]
    med = np.zeros(C, dtype=np.float32); sig = np.ones(C, dtype=np.float32)
    for c in range(C):
        v = X[:, c]; v = v[np.isfinite(v)]
        if v.size == 0: continue
        m = np.median(v); mad = np.median(np.abs(v - m))
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

def main():
    log("=== Step 5 v05: synthetic-injection training (recall-based thr) ===")
    t0 = time.time()
    z = np.load(NPZ_PATH, allow_pickle=True)
    device = best_device(); log(f"Device: {device}")

    summary = []
    state_dicts = {}; norm_means = {}; norm_stds = {}; thresholds = {}
    per_sn_scores = {}

    for survey in ("hst","jwst","vis","nisp"):
        if f"X_{survey}" not in z.files:
            log(f"  [{survey}] absent, skip"); continue
        X = z[f"X_{survey}"]; y = z[f"y_{survey}"]
        groups = z[f"{survey}_group"]; ids = z[f"{survey}_id"]
        in_ch = X.shape[1]
        n_pos = int(y.sum()); n_total = len(y)
        n_real = int((groups == "known").sum())
        n_inj  = int((groups == "inj").sum())
        log(f"\n--- {survey}: X={X.shape}  real={n_real}  inj={n_inj}  neg={n_total-n_pos} ---")

        # LOO over real SNe only
        real_sids = sorted({str(i) for g,i in zip(groups, ids) if g == "known"})
        held_scores = {}
        if len(real_sids) >= 2:
            log(f"  [{survey}] LOO over {len(real_sids)} real SN ids")
            for held in real_sids:
                hold = np.array([(g in ("known","known_rot") and i == held)
                                 for g, i in zip(groups, ids)])
                tr = np.where(~hold)[0]; he = np.where(hold)[0]
                m = SmallCNN(in_ch=in_ch).to(device)
                med, sig = per_channel_stats(X[tr])
                m.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
                train_one(m, X[tr], y[tr], device)
                p = predict(m, X[he], device)
                held_scores[held] = float(p.max())
        # Train final on full set
        final = SmallCNN(in_ch=in_ch).to(device)
        med, sig = per_channel_stats(X)
        final.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
        train_one(final, X, y, device)
        # Threshold: 20th percentile of REAL-SN scores (target 80% recall on labeled)
        # If too few real SNe, fall back to a combined positive percentile
        pos_idx_real = np.where((y == 1) & ((groups == "known") | (groups == "known_rot")))[0]
        if len(pos_idx_real) >= 4:
            pos_scores = predict(final, X[pos_idx_real], device)
            thr = float(np.quantile(pos_scores, 1.0 - TARGET_RECALL))
        else:
            # fallback to all positives
            pos_idx_all = np.where(y == 1)[0]
            pos_scores = predict(final, X[pos_idx_all], device)
            thr = float(np.quantile(pos_scores, 1.0 - TARGET_RECALL))
        # Clamp to [0.1, 0.95] — sanity bounds
        thr = max(0.1, min(0.95, thr))
        # Score the real positives + measure recovery
        recovered = sum(1 for sid in real_sids if held_scores.get(sid, predict(final, X[np.where((groups=="known") & (ids==sid))[0]], device).max() if (groups=="known").any() else 0) >= thr) if held_scores else 0
        # Actually fill held_scores for any missing IDs with in-sample max
        for sid in real_sids:
            if sid not in held_scores:
                idx_sid = np.where(((groups=="known") | (groups=="known_rot")) & (ids==sid))[0]
                if len(idx_sid):
                    held_scores[sid] = float(predict(final, X[idx_sid], device).max())
        recovered = sum(1 for v in held_scores.values() if v >= thr)
        recall = recovered / max(1, len(real_sids))
        log(f"  [{survey}] threshold={thr:.4f}  recall {recovered}/{len(real_sids)} ({100*recall:.1f}%)")
        per_sn_scores[survey] = held_scores
        thresholds[survey] = thr
        state_dicts[survey] = {k: v.detach().cpu() for k, v in final.state_dict().items()}
        norm_means[survey] = med; norm_stds[survey] = sig
        summary.append((survey, len(real_sids), recovered, recall, thr, n_pos, n_total))

    bundle = {"models": state_dicts, "norm_mean": norm_means, "norm_std": norm_stds,
              "thresholds": thresholds, "target_recall": TARGET_RECALL,
              "epochs": EPOCHS, "batch": BATCH, "lr": LR}
    torch.save(bundle, OUT_PT)
    OUT_JSON.write_text(json.dumps(thresholds, indent=2))
    log(f"\nSaved {OUT_PT} + {OUT_JSON}")

    lines = ["# Step 5 v05 training summary (synthetic injection + recall thr)",
             f"# Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"# Device: {device}  Epochs: {EPOCHS}  Batch: {BATCH}  TARGET_RECALL={TARGET_RECALL}",
             "",
             f"{'survey':>6} {'known':>6} {'rec':>6} {'recall':>8} {'P_thr':>8} {'pos':>6} {'total':>7}"]
    for s, kn, rec, recl, thr, np_, nt in summary:
        lines.append(f"{s:>6} {kn:>6} {rec:>6} {100*recl:>7.1f}% {thr:>8.4f} {np_:>6} {nt:>7}")
    lines.append("")
    for survey, scores in per_sn_scores.items():
        lines.append(f"\n=== {survey} per-SN scores ===")
        thr = thresholds.get(survey, 0.5)
        for sid in sorted(scores):
            tag = "RECOVERED" if scores[sid] >= thr else "MISSED"
            lines.append(f"  SN {sid:>9s}  P={scores[sid]:.4f}  thr={thr:.4f}  {tag}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    log(f"Saved {OUT_TXT}")
    log(f"=== Step 5 v05 done in {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
