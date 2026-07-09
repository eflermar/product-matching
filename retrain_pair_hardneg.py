"""
Retrain XGBoost pair classifier on hard negative pairs (combined_sim >= SIM_THRESHOLD).

The V5/V6 models were trained on broadly sampled pairs — they're miscalibrated for the
high-sim range seen at inference (all FAISS candidates have combined_sim >= 0.88).

This model trains on EXACTLY that distribution:
  Positives: same-label pairs where combined_sim >= SIM_THRESHOLD
  Negatives: different-label pairs where combined_sim >= SIM_THRESHOLD

Brand/dept/price features become the decisive signal since cosine sim is already high
for both classes. Two items from same brand + same dept + similar price at sim=0.92
are almost certainly the same product — the model learns this explicitly.

Usage:
  python retrain_pair_hardneg.py
  # produces: glami_xgb_hardneg.json

Then in phase2_pipeline.py set:
  PAIR_MODEL_PATH = 'glami_xgb_hardneg.json'
And run:
  python phase2_pipeline.py --phase1 --use_model --sweep
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
MODEL_OUT     = 'glami_xgb_hardneg.json'

SIM_THRESHOLD = 0.88    # must match phase2_pipeline.py inference pre-filter
TOP_K         = 100     # FAISS search width
NLIST         = 500
NPROBE        = 25
CHUNK_SZ      = 10_000

MAX_POS       = 80_000  # cap positive pairs
MAX_NEG_RATIO = 3       # negatives = MAX_POS * MAX_NEG_RATIO

PCA_DIM       = 32
PCA_FIT_N     = 50_000  # subsample for PCA fitting — avoids RAM spike on 900k items

XGB_PARAMS = dict(
    n_estimators        = 1500,
    learning_rate       = 0.05,
    max_depth           = 7,
    min_child_weight    = 5,
    subsample           = 0.8,
    colsample_bytree    = 0.7,
    eval_metric         = 'logloss',
    early_stopping_rounds = 30,
    device              = 'cpu',
    random_state        = 42,
)


# ── PCA ───────────────────────────────────────────────────────────────────────

def fit_apply_pca(lookup):
    n = len(lookup.text_emb)
    idx = np.random.choice(n, min(PCA_FIT_N, n), replace=False)
    print(f"Fitting PCA on {len(idx):,} items (dim {PCA_DIM})...")
    text_pca = PCA(n_components=PCA_DIM)
    img_pca  = PCA(n_components=PCA_DIM)
    text_pca.fit(lookup.text_emb[idx].astype(np.float32))
    img_pca.fit(lookup.image_emb[idx].astype(np.float32))
    print("Applying PCA to all items...")
    text_pca_emb = text_pca.transform(lookup.text_emb.astype(np.float32)).astype(np.float32)
    img_pca_emb  = img_pca.transform(lookup.image_emb.astype(np.float32)).astype(np.float32)
    return text_pca_emb, img_pca_emb


# ── Pair mining ───────────────────────────────────────────────────────────────

def mine_pairs(df_train, lookup):
    item_to_label = dict(zip(df_train['itemId'].astype(str), df_train['label']))
    idx_to_label  = {}
    for iid, lbl in item_to_label.items():
        ix = lookup.id_to_idx.get(iid)
        if ix is not None:
            idx_to_label[ix] = lbl

    n        = len(lookup.text_emb)
    text_emb = lookup.text_emb.astype(np.float32)
    img_emb  = lookup.image_emb.astype(np.float32)

    print(f"Building FAISS IVF index over {n:,} items...")
    quantizer = faiss.IndexFlatIP(text_emb.shape[1])
    index     = faiss.IndexIVFFlat(quantizer, text_emb.shape[1], NLIST,
                                   faiss.METRIC_INNER_PRODUCT)
    index.train(text_emb)
    index.add(text_emb)
    index.nprobe = NPROBE

    all_indices = np.array(list(idx_to_label.keys()), dtype=np.int64)
    np.random.shuffle(all_indices)

    max_neg  = MAX_POS * MAX_NEG_RATIO
    pos_seen = set()
    neg_seen = set()
    positives, negatives = [], []

    pbar = tqdm(range(0, len(all_indices), CHUNK_SZ), desc="Mining pairs")
    for s in pbar:
        e         = min(s + CHUNK_SZ, len(all_indices))
        batch_idx = all_indices[s:e]
        _, nbrs   = index.search(text_emb[batch_idx], TOP_K + 1)

        for anchor_ix, row in zip(batch_idx, nbrs):
            a_lbl = idx_to_label.get(int(anchor_ix))
            if a_lbl is None:
                continue
            for nbr in row:
                nbr = int(nbr)
                if nbr < 0 or nbr >= n or nbr == anchor_ix:
                    continue
                t_sim = float(np.dot(text_emb[anchor_ix], text_emb[nbr]))
                v_sim = float(np.dot(img_emb[anchor_ix], img_emb[nbr]))
                if (t_sim + v_sim) * 0.5 < SIM_THRESHOLD:
                    continue
                pair = (min(int(anchor_ix), nbr), max(int(anchor_ix), nbr))
                same = (idx_to_label.get(nbr) == a_lbl)
                if same and pair not in pos_seen and len(positives) < MAX_POS:
                    pos_seen.add(pair)
                    positives.append(pair)
                elif not same and pair not in neg_seen and len(negatives) < max_neg:
                    neg_seen.add(pair)
                    negatives.append(pair)

        pbar.set_postfix(pos=len(positives), neg=len(negatives))
        if len(positives) >= MAX_POS and len(negatives) >= max_neg:
            print(f"  Caps reached at chunk {s}–{e}, stopping early.")
            break

    print(f"  Mined {len(positives):,} positives, {len(negatives):,} negatives")
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
    negatives = negatives[:len(positives) * MAX_NEG_RATIO]

    all_pairs  = positives + negatives
    all_labels = [1] * len(positives) + [0] * len(negatives)
    order      = np.random.permutation(len(all_pairs))
    all_pairs  = [all_pairs[i] for i in order]
    all_labels = [all_labels[i] for i in order]

    ia = np.array([p[0] for p in all_pairs], dtype=np.int64)
    ib = np.array([p[1] for p in all_pairs], dtype=np.int64)
    y  = np.array(all_labels, dtype=np.int32)

    print(f"\nDataset: {len(y):,} pairs  pos_rate={y.mean():.3f}")
    print("Computing 84 pair features...")
    X = compute_pair_features_pca(lookup, ia, ib, text_pca_emb, img_pca_emb)
    print(f"  X shape: {X.shape}")

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

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.96, 0.05):
        f1 = f1_score(y_va, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"Val F1={best_f1:.4f} at t={best_t:.2f}")
    print(classification_report(y_va, (proba >= best_t).astype(int), zero_division=0))

    model.save_model(MODEL_OUT)
    print(f"\nSaved → {MODEL_OUT}")
    print("\nIn phase2_pipeline.py set:")
    print(f"  PAIR_MODEL_PATH = '{MODEL_OUT}'")
    print("Then run:")
    print("  python phase2_pipeline.py --phase1 --use_model --sweep")


if __name__ == '__main__':
    main()
