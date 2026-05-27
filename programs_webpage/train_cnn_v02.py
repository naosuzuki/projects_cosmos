"""Step 5 v02: per-survey CNNs with CORRECT labels.

Fix vs v01: each known SN is positive ONLY in the survey of discovery
(telescope column of lookup_sn17_v32.csv). All other (sn_id, survey)
pairs revert to label=0 because the cutout in that survey is just the
host galaxy.

Discovery-telescope → survey mapping:
   HST    →  hst
   JWST   →  jwst
   EUCLID →  both vis AND nisp (Euclid covers VIS + NISP simultaneously)

LOO behaviour:
  - hst:   3 known positives → 3 folds
  - jwst:  13 known positives → 13 folds
  - vis:   1 known positive  → degenerate; skip LOO, train + report in-sample
  - nisp:  1 known positive  → same as vis

Inputs:
  csvfiles_sn/training_set_v01.npz          (reused — cutouts unchanged)
  csvfiles_sn/lookup_sn17_v32.csv           (telescope lookup)

Outputs:
  csvfiles_sn/cnn_models_v02.pt
  csvfiles_sn/training_summary_v02.txt
  csvfiles_sn/cnn_thresholds_v02.json
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "/Users/suzuki/github/projects_cosmos/programs_webpage")
from cnn_models_v01 import SmallCNN, best_device

CSV_DIR  = Path("/Users/suzuki/github/projects_cosmos/csvfiles_sn")
NPZ_PATH = CSV_DIR / "training_set_v01.npz"
LOOKUP   = CSV_DIR / "lookup_sn17_v32.csv"
OUT_PT   = CSV_DIR / "cnn_models_v02.pt"
OUT_TXT  = CSV_DIR / "training_summary_v02.txt"
OUT_JSON = CSV_DIR / "cnn_thresholds_v02.json"

EPOCHS = 30
BATCH  = 256
LR     = 1e-3
TARGET_FPR = 1e-3

# Survey -> set of telescope-string values whose SNe should label POSITIVE here
SURVEY_TEL = {
    "hst":  {"HST"},
    "jwst": {"JWST"},
    "vis":  {"EUCLID"},
    "nisp": {"EUCLID"},
}


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


def train_one(model, X_train, y_train, device, epochs=EPOCHS, batch=BATCH, lr=LR):
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


def relabel(y_orig, groups, ids, survey, tel_by_id):
    """Return y with positives kept only where SN telescope ∈ SURVEY_TEL[survey]."""
    y = y_orig.copy().astype(np.int8)
    matching = SURVEY_TEL[survey]
    for i in range(len(y)):
        if y[i] == 1 and groups[i] in (b"known_pos", b"aug_pos", "known_pos", "aug_pos"):
            sid = ids[i]
            try: sid = int(sid)
            except: pass
            tel = tel_by_id.get(int(sid), "")
            if tel not in matching:
                y[i] = 0
    return y


def main():
    t_start = time.time()
    log("=== Step 5 v02: per-survey CNNs with corrected labels ===")
    # load telescope mapping
    tel_by_id = {}
    with LOOKUP.open() as f:
        for r in csv.DictReader(f):
            tel_by_id[int(r["id"])] = r["telescope"]
    log(f"Loaded telescope mapping for {len(tel_by_id)} known SNe")
    pos_counts = {}
    for sid, tel in tel_by_id.items():
        pos_counts[tel] = pos_counts.get(tel, 0) + 1
    log(f"Discovery telescope mix: {pos_counts}")

    log(f"Loading {NPZ_PATH} ...")
    z = np.load(NPZ_PATH, allow_pickle=True)
    device = best_device()
    log(f"Device: {device}")

    summary = []
    results = {}
    thresholds = {}
    state_dicts = {}
    norm_means = {}; norm_stds = {}

    for survey in ("hst", "jwst", "vis", "nisp"):
        X = z[f"X_{survey}"]
        y_orig = z[f"y_{survey}"]
        if len(X) == 0:
            log(f"  [{survey}] no data, skip"); continue
        groups = z[f"{survey}_group"]
        ids    = z[f"{survey}_id"]
        # decode bytes if needed
        try: groups = np.array([g.decode() if isinstance(g, bytes) else g for g in groups])
        except: pass

        y = relabel(y_orig, groups, ids, survey, tel_by_id)
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        # known-pos SN ids that survived (matching this survey's discovery telescope)
        matching = SURVEY_TEL[survey]
        relevant_ids = sorted({int(sid) for sid, tel in tel_by_id.items() if tel in matching})
        log(f"\n--- {survey}: X={X.shape}  pos={n_pos} (was {int(y_orig.sum())})  "
            f"relevant_SN_ids={relevant_ids}")

        in_ch = X.shape[1]
        # LOO over relevant_ids
        held_pos_scores = []; held_pos_meta = []
        fold_summary = []
        skip_loo = (len(relevant_ids) < 2)
        if skip_loo:
            log(f"  [{survey}] only {len(relevant_ids)} positive SN id → skip LOO; train+eval-in-sample")
        for held_id in relevant_ids:
            if skip_loo: continue
            hold = np.array([
                (groups[i] in ("known_pos","aug_pos") and int(ids[i]) == held_id)
                for i in range(len(groups))
            ])
            train_idx = np.where(~hold)[0]
            held_idx  = np.where(hold)[0]
            if len(held_idx) == 0: continue
            m = SmallCNN(in_ch=in_ch).to(device)
            med, sig = per_channel_stats(X[train_idx])
            m.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
            train_one(m, X[train_idx], y[train_idx], device)
            p = predict(m, X[held_idx], device)
            for ii, pp in zip(held_idx, p):
                held_pos_scores.append(float(pp))
                held_pos_meta.append((int(ids[ii]), groups[ii]))
            fold_summary.append(dict(held_id=held_id, n_held=int(len(held_idx)),
                                     p_max=float(p.max()), p_mean=float(p.mean())))
            log(f"  [{survey}] LOO held={held_id:>7d}  n={len(held_idx)}  "
                f"P_max={p.max():.4f}  P_mean={p.mean():.4f}")

        # final model on full set
        log(f"  [{survey}] training final model (full set, {len(X):,} samples)")
        final = SmallCNN(in_ch=in_ch).to(device)
        med, sig = per_channel_stats(X)
        final.set_norm(torch.tensor(med, device=device), torch.tensor(sig, device=device))
        train_one(final, X, y, device)
        # negative scores for threshold
        neg_idx = np.where(y == 0)[0]
        neg_scores = predict(final, X[neg_idx], device) if len(neg_idx) else np.zeros(0)
        # positive scores from full final model (for skip_loo path) — gives in-sample recall
        pos_idx = np.where(y == 1)[0]
        if skip_loo and len(pos_idx) > 0:
            p_full = predict(final, X[pos_idx], device)
            for k, ii in enumerate(pos_idx):
                if groups[ii] == "known_pos":
                    held_pos_scores.append(float(p_full[k]))
                    held_pos_meta.append((int(ids[ii]), groups[ii]))

        thr = threshold_at_fpr(neg_scores, TARGET_FPR)
        thresholds[survey] = thr

        # collect recall over known_pos SNe
        by_id = {}
        for s, (sid, grp) in zip(held_pos_scores, held_pos_meta):
            if grp == "known_pos":
                by_id[sid] = max(by_id.get(sid, 0.0), s)
        # IMPORTANT: only count recall over relevant_ids (this survey's discovery telescope).
        rec_count = sum(1 for sid in relevant_ids if by_id.get(sid, 0.0) >= thr)
        total = len(relevant_ids)
        recall = rec_count / total if total else 0.0
        log(f"  [{survey}] threshold@FPR={TARGET_FPR:.0e}: P>={thr:.4f}  "
            f"recall over {survey}-discovery SNe: {rec_count}/{total} ({100*recall:.1f}%)")

        results[survey] = dict(
            held_pos_scores=held_pos_scores, held_pos_meta=held_pos_meta,
            relevant_ids=relevant_ids, neg_scores=neg_scores,
            by_id=by_id, fold_summary=fold_summary, skip_loo=skip_loo,
        )
        state_dicts[survey] = {k: v.detach().cpu() for k, v in final.state_dict().items()}
        norm_means[survey] = med; norm_stds[survey] = sig
        summary.append((survey, total, rec_count, recall, thr, n_pos, float(neg_scores.mean() if len(neg_scores) else 0)))

    # save
    out = {
        "models": state_dicts,
        "norm_mean": norm_means, "norm_std": norm_stds,
        "thresholds": thresholds,
        "target_fpr": TARGET_FPR, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
    }
    torch.save(out, OUT_PT)
    OUT_JSON.write_text(json.dumps(thresholds, indent=2))
    log(f"\nSaved {OUT_PT}")
    log(f"Saved {OUT_JSON}")

    lines = ["# Step 5 v02 (corrected labels) — training summary",
             f"# Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}  Device: mps",
             f"# Epochs: {EPOCHS}  Batch: {BATCH}  LR: {LR}  TargetFPR: {TARGET_FPR}",
             "",
             f"{'survey':>6} {'discoveries':>11} {'recovered':>10} {'recall':>8} {'P_thr':>8} {'n_pos':>8} {'mean_neg':>10}"]
    for s, total, recv, recall, thr, n_pos, neg_mean in summary:
        lines.append(f"{s:>6} {total:>11} {recv:>10} {100*recall:>7.1f}% {thr:>8.4f} {n_pos:>8} {neg_mean:>10.4f}")
    lines += ["", "Per-known-SN per-survey LOO scores (only discovery-survey rows shown):"]
    for survey, res in results.items():
        lines.append(f"\n=== {survey} (skip_LOO={res['skip_loo']}) ===")
        thr = thresholds.get(survey, 0.5)
        for sid in sorted(res["by_id"]):
            tag = "RECOVERED" if res["by_id"][sid] >= thr else "MISSED   "
            note = "  [discovery]" if sid in res["relevant_ids"] else ""
            lines.append(f"  SN {sid:>7d}  P={res['by_id'][sid]:.4f}  thr={thr:.4f}  {tag}{note}")

    # Overall recall on 17 known SNe (summing across surveys, but each SN counts once via its discovery survey)
    total_recovered = sum(rec for (_, _, rec, _, _, _, _) in summary)
    total_known = sum(total for (_, total, _, _, _, _, _) in summary)
    lines.append("")
    lines.append(f"Overall recall (each known SN counted in its discovery survey): "
                 f"{total_recovered}/{total_known}")
    OUT_TXT.write_text("\n".join(lines) + "\n")
    log(f"Saved {OUT_TXT}")
    log(f"=== Step 5 v02 done in {time.time()-t_start:.1f}s ===")


if __name__ == "__main__":
    main()
