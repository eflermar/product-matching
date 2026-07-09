"""
Retrain pair model (V6) with FAISS-mined hard negatives.

The original V5 model was trained on random + same-dept negatives, which are
much easier than the nearest-neighbour pairs seen during Phase 2 inference.
This causes it to saturate at ~1.0 for almost all FAISS candidates.

Fix: mine negatives the same way Phase 2 does — take top-K FAISS neighbours
of each anchor that have a *different* label.  The retrained model (V6) will
be properly calibrated for the Phase 2 candidate distribution.

Usage:
  python retrain_pair_faiss.py
  # produces glami_xgb_pairwise_ft_v6.json

Then in phase2_pipeline.py, set PAIR_MODEL_PATH = 'glami_xgb_pairwise_ft_v6.json'
and run:
  python phase2_pipeline.py --phase1 --use_model --sweep
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import faiss
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from features import EmbeddingLookup, compute_pair_features_pca, ALL_PAIR_FEATURE_NAMES

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_H5       = 'glami_embeddings_train_ft.h5'
TRAIN_CSV      = 'data/items_train.csv'
PAIR_MODEL_OUT = 'glami_xgb_pairwise_ft_v6.json'

PCA_DIM            = 32
MAX_POS_PER_LABEL  = 3       # positive pairs to sample per label
TOP_K_NEG          = 50      # FAISS neighbours to search for hard negatives
NLIST              = 500
NPROBE             = 25


# ── PCA — identical to pipeline_ft_v5 ────────────────────────────────────────

def fit_pca(lookup, n_components=PCA_DIM, sample_size=100_000):
    print(f"Fitting PCA (dim={n_components}) on {sample_size} samples...")
    n   = len(lookup.text_emb)
    idx = np.random.choice(n, min(sample_size, n), replace=False)
    text_pca = PCA(n_components=n_components, random_state=42)
    img_pca  = PCA(n_components=n_components, random_state=42)
    text_pca.fit(lookup.text_emb[idx].astype(np.float32))
    img_pca.fit(lookup.image_emb[idx].astype(np.float32))
    print(f"  Text PCA: {sum(text_pca.explained_variance_ratio_) * 100:.1f}% variance")
    print(f"  Image PCA: {sum(img_pca.explained_variance_ratio_) * 100:.1f}% variance")
    return text_pca, img_pca


def apply_pca(embeddings, pca_model, chunk_size=100_000):
    n   = len(embeddings)
    out = np.empty((n, pca_model.n_components), dtype=np.float32)
    for s in range(0, n, chunk_size):
        e        = min(s + chunk_size, n)
        out[s:e] = pca_model.transform(embeddings[s:e].astype(np.float32))
    return out


# ── FAISS index on training items ─────────────────────────────────────────────

def build_ivf_index(emb, nlist=NLIST, nprobe=NPROBE):
    d         = emb.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index     = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    index.nprobe = nprobe
    return index


# ── Pair mining ───────────────────────────────────────────────────────────────

def mine_pairs(df_train, lookup):
    """
    Returns (pair_indices, pair_labels) arrays.

    Positives  : same-label pairs (up to MAX_POS_PER_LABEL per label)
    Hard negatives: FAISS top-K text neighbours with a different label,
                    equal count to positives (1:1 ratio)
    """
    item_to_label = dict(zip(df_train['itemId'].astype(str), df_train['label']))
    n             = len(lookup.text_emb)

    # ── Positive pairs ────────────────────────────────────────────────────────
    print("Mining positive pairs...")
    grouped   = df_train.groupby('label')['itemId'].apply(list)
    positives = []
    for label, items in grouped.items():
        items = [str(x) for x in items]
        idxs  = lookup.resolve_indices(items)
        valid = [ix for ix in idxs if ix >= 0]
        if len(valid) < 2:
            continue
        combos = list(combinations(valid, 2))
        if len(combos) > MAX_POS_PER_LABEL:
            chosen = np.random.choice(len(combos), MAX_POS_PER_LABEL, replace=False)
            combos = [combos[i] for i in chosen]
        positives.extend((a, b) for a, b in combos)
    print(f"  Positive pairs: {len(positives):,}")

    # Build label lookup by embedding index (covers all items in lookup)
    idx_to_label = {}
    for iid, lbl in item_to_label.items():
        ix = lookup.id_to_idx.get(iid)
        if ix is not None:
            idx_to_label[ix] = lbl

    # ── FAISS hard negatives ──────────────────────────────────────────────────
    print("Building FAISS index on training embeddings...")
    text_emb   = lookup.text_emb.astype(np.float32)
    text_index = build_ivf_index(text_emb)
    print("  Index built.")

    print(f"Mining FAISS hard negatives (top_k={TOP_K_NEG})...")
    hard_negs  = []
    neg_seen   = set()
    n_needed   = len(positives)
    batch_size = 5_000
    anchor_indices = list(idx_to_label.keys())
    np.random.shuffle(anchor_indices)

    for batch_start in tqdm(range(0, len(anchor_indices), batch_size),
                            desc="Hard neg mining"):
        if len(hard_negs) >= n_needed:
            break
        batch_end   = min(batch_start + batch_size, len(anchor_indices))
        batch_idx   = anchor_indices[batch_start:batch_end]
        batch_emb   = text_emb[batch_idx]
        _, neighbors = text_index.search(batch_emb, TOP_K_NEG + 1)

        for anchor_ix, nbrs in zip(batch_idx, neighbors):
            anchor_lbl = idx_to_label.get(anchor_ix)
            if anchor_lbl is None:
                continue
            for nbr in nbrs:
                if nbr < 0 or nbr >= n or nbr == anchor_ix:
                    continue
                nbr_lbl = idx_to_label.get(nbr)
                if nbr_lbl is None or nbr_lbl == anchor_lbl:
                    continue
                pair = (min(anchor_ix, nbr), max(anchor_ix, nbr))
                if pair not in neg_seen:
                    neg_seen.add(pair)
                    hard_negs.append(pair)
                    break   # one hard neg per anchor

    hard_negs = hard_negs[:n_needed]
    print(f"  Hard negative pairs: {len(hard_negs):,}")

    # ── Assemble ──────────────────────────────────────────────────────────────
    all_pairs  = positives + hard_negs
    all_labels = [1] * len(positives) + [0] * len(hard_negs)
    order      = np.random.permutation(len(all_pairs))
    pair_arr   = np.array(all_pairs, dtype=np.int32)[order]
    label_arr  = np.array(all_labels, dtype=np.int32)[order]
    return pair_arr, label_arr


# ── Model training ────────────────────────────────────────────────────────────

def train_pair_model(lookup, pair_arr, label_arr, text_pca_emb, img_pca_emb):
    print("\nComputing 84 pair features...")
    X = compute_pair_features_pca(
        lookup, pair_arr[:, 0], pair_arr[:, 1], text_pca_emb, img_pca_emb
    )
    y = label_arr

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  "
          f"Pos rate: {y_tr.mean():.3f}")

    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        scale_pos_weight=n_neg / n_pos,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=42,
        tree_method='hist',
        device='cuda',
    )

    print("Training XGBoost V6 (FAISS hard negatives)...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    # Find best threshold on validation set
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.3, 0.95, 0.05):
        f1 = f1_score(y_val, (val_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\nVal F1 = {best_f1:.4f} at threshold {best_t:.2f}")
    print(classification_report(y_val, (val_proba >= best_t).astype(int)))

    model.save_model(PAIR_MODEL_OUT)
    print(f"Model saved → {PAIR_MODEL_OUT}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading training data...")
    df_train          = pd.read_csv(TRAIN_CSV)
    df_train['itemId'] = df_train['itemId'].astype(str)
    print(f"  {len(df_train):,} items, {df_train['label'].nunique():,} distinct labels")

    lookup = EmbeddingLookup(TRAIN_H5)

    # PCA — must match V5 / phase2_pipeline.py
    text_pca, img_pca = fit_pca(lookup)
    text_pca_emb = apply_pca(lookup.text_emb, text_pca)
    img_pca_emb  = apply_pca(lookup.image_emb, img_pca)

    # Mine pairs
    pair_arr, label_arr = mine_pairs(df_train, lookup)

    # Train
    train_pair_model(lookup, pair_arr, label_arr, text_pca_emb, img_pca_emb)

    print("\nNext steps:")
    print("  1. In phase2_pipeline.py set:  PAIR_MODEL_PATH = 'glami_xgb_pairwise_ft_v6.json'")
    print("  2. python phase2_pipeline.py --phase1 --use_model --sweep")


if __name__ == '__main__':
    main()
