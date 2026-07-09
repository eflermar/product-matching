"""
Retrain XGBoost pair classifier on medium-similarity pairs (0.70 <= combined_sim < 0.88).

KEY FIXES vs. previous version (which over-merged at inference):

  (1) Negative ratio: 3 → 99 to match the empirical inference positive rate
      of 1.07% in s ∈ [0.70, 0.88).  Previous training used 25% positives,
      a 23× prior mismatch that biased every borderline pair toward "positive".

  (2) Drop brand-OR-dept-matched-only mining.  In this dataset most items have
      brand_id = dept_id = 0, so the filter discarded most candidate negatives.
      The model never saw what an "ordinary" negative in this similarity range
      looks like and over-predicted on them at inference.  Pairs with
      s ∈ [0.70, 0.88) are inherently hard (high cosine, different products) —
      no metadata filter is needed.

  (3) Threshold sweep extended to 0.99 and reports F1 at high thresholds, since
      the calibrated model should only be trusted at very high confidence in
      the 1%-positive regime.

Usage:
  python retrain_pair_medmatch.py
  # produces: glami_xgb_medmatch.json

Then evaluate:
  python phase2_pipeline.py --validate --use_model --medmatch --sweep
"""
import os
import numpy as np
import pandas as pd
import faiss
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from tqdm import tqdm

from features import (EmbeddingLookup, compute_pair_features_pca,
                      ALL_PAIR_FEATURE_NAMES)

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_H5      = 'glami_embeddings_train_ft.h5'
TRAIN_CSV     = 'data/items_train.csv'
MODEL_OUT     = 'glami_xgb_medmatch.json'

SIM_LOW       = 0.70
SIM_HIGH      = 0.88
TOP_K         = 100
NLIST         = 500
NPROBE        = 25
CHUNK_SZ      = 10_000

# ── PRIOR-MATCHED SAMPLING ────────────────────────────────────────────────────
# Inference positive rate in [0.70, 0.88) is ~1.07% (from diagnose_recall.py).
# Match it: 1 pos : 99 neg ≡ 1.0% positive rate.
MAX_POS       = 60_000
MAX_NEG_RATIO = 99           # was 3
# If mining 6M negs is too slow / memory-heavy, halve both:
#   MAX_POS = 30_000;  MAX_NEG_RATIO = 99   (3M negatives, same prior)

PCA_DIM       = 32
PCA_FIT_N     = 50_000

XGB_PARAMS = dict(
    n_estimators          = 1500,
    learning_rate         = 0.05,
    max_depth             = 7,
    min_child_weight      = 5,
    subsample             = 0.8,
    colsample_bytree      = 0.7,
    eval_metric           = 'logloss',
    early_stopping_rounds = 30,
    device                = 'cpu',
    random_state          = 42,
    # scale_pos_weight = 1: training data already encodes the ~1% prior, so
    # the model outputs calibrated probabilities and the threshold has direct
    # meaning at inference.
    scale_pos_weight      = 1,
)


# ── PCA ───────────────────────────────────────────────────────────────────────

def fit_apply_pca(lookup):
    n   = len(lookup.text_emb)
    idx = np.random.choice(n, min(PCA_FIT_N, n), replace=False)
    print(f"Fitting PCA on {len(idx):,} items (dim {PCA_DIM})...")
    text_pca = PCA(n_components=PCA_DIM)
    img_pca  = PCA(n_components=PCA_DIM)
    text_pca.fit(lookup.text_emb[idx].astype(np.float32))
    img_pca.fit(lookup.image_emb[idx].astype(np.float32))
    print("Applying PCA...")
    text_pca_emb = text_pca.transform(lookup.text_emb.astype(np.float32)).astype(np.float32)
    img_pca_emb  = img_pca.transform(lookup.image_emb.astype(np.float32)).astype(np.float32)
    return text_pca_emb, img_pca_emb


# ── Pair mining ───────────────────────────────────────────────────────────────

def mine_pairs(df_train, lookup):
    """
    Mine same-label and different-label pairs in s ∈ [SIM_LOW, SIM_HIGH).

    Negatives are NOT pre-filtered by metadata.  Candidates in this similarity
    range are inherently hard (high cosine, different products), and matching
    the inference distribution is more important than artificial hardness.
    """
    item_to_label = dict(zip(df_train['itemId'].astype(str), df_train['label']))
    idx_to_label  = {}
    for iid, lbl in item_to_label.items():
        ix = lookup.id_to_idx.get(iid)
        if ix is not None:
            idx_to_label[ix] = lbl

    n        = len(lookup.text_emb)
    text_emb = lookup.text_emb.astype(np.float32)
    img_emb  = lookup.image_emb.astype(np.float32)

    print(f"Building FAISS indices over {n:,} items...")
    def _build(emb):
        q   = faiss.IndexFlatIP(emb.shape[1])
        idx = faiss.IndexIVFFlat(q, emb.shape[1], NLIST, faiss.METRIC_INNER_PRODUCT)
        idx.train(emb)
        idx.add(emb)
        idx.nprobe = NPROBE
        return idx

    text_index = _build(text_emb)
    img_index  = _build(img_emb)

    all_indices = np.array(list(idx_to_label.keys()), dtype=np.int64)
    np.random.shuffle(all_indices)

    max_neg  = MAX_POS * MAX_NEG_RATIO
    pos_seen, neg_seen = set(), set()
    positives, negatives = [], []

    pbar = tqdm(range(0, len(all_indices), CHUNK_SZ),
                desc=f"Mining (target: {MAX_POS:,} pos, {max_neg:,} neg)")
    for s in pbar:
        e         = min(s + CHUNK_SZ, len(all_indices))
        batch_idx = all_indices[s:e]

        # Union of text and image neighbors
        nbr_sets = {}
        for index, emb in [(text_index, text_emb), (img_index, img_emb)]:
            _, nbrs_batch = index.search(emb[batch_idx], TOP_K + 1)
            for i, anchor_ix in enumerate(batch_idx):
                if anchor_ix not in nbr_sets:
                    nbr_sets[anchor_ix] = set()
                nbr_sets[anchor_ix].update(int(x) for x in nbrs_batch[i])

        for anchor_ix, nbrs in nbr_sets.items():
            a_lbl = idx_to_label.get(int(anchor_ix))
            if a_lbl is None:
                continue
            for nbr in nbrs:
                if nbr < 0 or nbr >= n or nbr == anchor_ix:
                    continue
                t_sim    = float(np.dot(text_emb[anchor_ix], text_emb[nbr]))
                v_sim    = float(np.dot(img_emb[anchor_ix], img_emb[nbr]))
                combined = (t_sim + v_sim) * 0.5
                if combined < SIM_LOW or combined >= SIM_HIGH:
                    continue

                pair = (min(int(anchor_ix), nbr), max(int(anchor_ix), nbr))
                same = (idx_to_label.get(nbr) == a_lbl)

                if same:
                    if pair not in pos_seen and len(positives) < MAX_POS:
                        pos_seen.add(pair)
                        positives.append(pair)
                else:
                    # No metadata filter — accept all different-label pairs in range.
                    if pair not in neg_seen and len(negatives) < max_neg:
                        neg_seen.add(pair)
                        negatives.append(pair)

        pbar.set_postfix(pos=len(positives), neg=len(negatives))
        if len(positives) >= MAX_POS and len(negatives) >= max_neg:
            print(f"  Caps reached at chunk {s}–{e}, stopping early.")
            break

    print(f"  Mined {len(positives):,} positives, {len(negatives):,} negatives  "
          f"(pos_rate = {len(positives) / max(len(positives) + len(negatives), 1):.4f})")
    return positives, negatives


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading training data...")
    df_train           = pd.read_csv(TRAIN_CSV)
    df_train['itemId'] = df_train['itemId'].astype(str)
    print(f"  {len(df_train):,} items, {df_train['label'].nunique():,} labels")

    lookup = EmbeddingLookup(TRAIN_H5)

    text_pca_emb, img_pca_emb = fit_apply_pca(lookup)

    positives, negatives = mine_pairs(df_train, lookup)

    np.random.shuffle(positives)
    np.random.shuffle(negatives)
    # Don't truncate positives — but cap negatives at MAX_POS * MAX_NEG_RATIO
    # in case mining went past it.
    negatives = negatives[:len(positives) * MAX_NEG_RATIO]

    all_pairs  = positives + negatives
    all_labels = [1] * len(positives) + [0] * len(negatives)
    order      = np.random.permutation(len(all_pairs))
    all_pairs  = [all_pairs[i] for i in order]
    all_labels = [all_labels[i] for i in order]

    ia = np.array([p[0] for p in all_pairs], dtype=np.int64)
    ib = np.array([p[1] for p in all_pairs], dtype=np.int64)
    y  = np.array(all_labels, dtype=np.int32)

    print(f"\nDataset: {len(y):,} pairs  pos_rate={y.mean():.4f}  "
          f"(target: {1.0 / (1.0 + MAX_NEG_RATIO):.4f}, matches inference prior)")
    print("Computing 84 pair features...")
    X = compute_pair_features_pca(lookup, ia, ib, text_pca_emb, img_pca_emb)
    print(f"  X shape: {X.shape}  (memory: {X.nbytes / 1e9:.2f} GB)")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_tr):,}  Val: {len(X_va):,}")

    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=50)

    proba = model.predict_proba(X_va)[:, 1]
    auc   = roc_auc_score(y_va, proba)
    print(f"\nVal AUC: {auc:.4f}")

    # ── Threshold sweep, focused on HIGH thresholds ──────────────────────────
    # With a 1% prior, low thresholds will produce many false positives.
    # We expect the operating point to land around 0.7-0.97.
    print("\nThreshold sweep (val):")
    print(f"  {'thr':>6}{'F1':>10}{'precision':>12}{'recall':>10}"
          f"{'pred_pos':>12}{'true_pos':>12}")
    best_t, best_f1 = 0.5, 0.0
    for t in [0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]:
        pred = (proba >= t).astype(int)
        tp   = int(((pred == 1) & (y_va == 1)).sum())
        fp   = int(((pred == 1) & (y_va == 0)).sum())
        fn   = int(((pred == 0) & (y_va == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        print(f"  {t:>6.2f}{f1:>10.4f}{prec:>12.4f}{rec:>10.4f}"
              f"{pred.sum():>12,}{tp:>12,}")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\nBest val F1={best_f1:.4f} at t={best_t:.2f}")
    print(classification_report(y_va, (proba >= best_t).astype(int), zero_division=0))

    model.save_model(MODEL_OUT)
    print(f"\nSaved → {MODEL_OUT}")
    print(f"\nNext step: validate end-to-end at multiple gating thresholds, e.g.")
    print(f"  python phase2_pipeline.py --validate --use_model --medmatch --sweep")
    print(f"and pick the threshold that maximises Phase-2 F1, not val F1.")


if __name__ == '__main__':
    main()