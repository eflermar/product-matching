"""
Ensemble: combine V2 and V5 group model predictions.

Loads both pipelines' trained models, computes group probabilities independently,
then averages with configurable weights. Sweeps thresholds to find the best combo.

Best result: w_v2=0.3, w_v5=0.7, threshold=0.64 → 0.9937
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import h5py
from sklearn.decomposition import PCA

from features import EmbeddingLookup, compute_group_features

PCA_DIM = 32

# ── Load models ──────────────────────────────────────────────────────────────

print("Loading models...")
pair_model_v2 = xgb.XGBClassifier()
pair_model_v2.load_model('glami_xgb_pairwise_ft.json')

pair_model_v5 = xgb.XGBClassifier()
pair_model_v5.load_model('glami_xgb_pairwise_ft_v5.json')

group_model_v2 = xgb.XGBClassifier()
group_model_v2.load_model('glami_group_meta_ft_v2.json')

group_model_v5 = xgb.XGBClassifier()
group_model_v5.load_model('glami_group_meta_ft_v5.json')


# ── Fit PCA (must match pipeline_ft_v5 exactly) ─────────────────────────────

print("Loading training embeddings for PCA fitting...")
with h5py.File('glami_embeddings_train_ft.h5', 'r') as f:
    train_text = f['text_emb'][:].astype(np.float16)
    train_img = f['image_emb'][:].astype(np.float16)

np.random.seed(42)
sample_idx = np.random.choice(len(train_text), min(100000, len(train_text)), replace=False)

text_pca = PCA(n_components=PCA_DIM, random_state=42)
img_pca = PCA(n_components=PCA_DIM, random_state=42)
text_pca.fit(train_text[sample_idx].astype(np.float32))
img_pca.fit(train_img[sample_idx].astype(np.float32))
print(f"  Text PCA: {sum(text_pca.explained_variance_ratio_) * 100:.1f}% var")
print(f"  Image PCA: {sum(img_pca.explained_variance_ratio_) * 100:.1f}% var")
del train_text, train_img


# ── Load phase_1 embeddings & apply PCA ──────────────────────────────────────

phase1 = EmbeddingLookup('glami_embeddings_phase_1_ft.h5')

print("Applying PCA to phase_1 embeddings...")
CHUNK = 100000
text_pca_emb = np.empty((len(phase1.text_emb), PCA_DIM), dtype=np.float32)
img_pca_emb = np.empty((len(phase1.image_emb), PCA_DIM), dtype=np.float32)
for start in range(0, len(phase1.text_emb), CHUNK):
    end = min(start + CHUNK, len(phase1.text_emb))
    text_pca_emb[start:end] = text_pca.transform(phase1.text_emb[start:end].astype(np.float32))
    img_pca_emb[start:end] = img_pca.transform(phase1.image_emb[start:end].astype(np.float32))
print("  Done.")


# ── Load task groups ─────────────────────────────────────────────────────────

df_task = pd.read_csv('data/task_1.csv')
groups = [[str(row[f'item{i}']) for i in range(1, 6)] for _, row in df_task.iterrows()]
print(f"\nTask groups: {len(groups)}")


# ── Get predictions from both pipelines ──────────────────────────────────────

print("\n--- V2 predictions (20-feature pairs) ---")
X_v2 = compute_group_features(groups, pair_model_v2, phase1)
proba_v2 = group_model_v2.predict_proba(X_v2)[:, 1]
print(f"  Mean: {proba_v2.mean():.4f}, Pos @0.64: {np.sum(proba_v2 >= 0.64)}")

print("\n--- V5 predictions (84-feature PCA pairs) ---")
X_v5 = compute_group_features(groups, pair_model_v5, phase1,
                               text_pca_emb=text_pca_emb, img_pca_emb=img_pca_emb)
proba_v5 = group_model_v5.predict_proba(X_v5)[:, 1]
print(f"  Mean: {proba_v5.mean():.4f}, Pos @0.64: {np.sum(proba_v5 >= 0.64)}")


# ── Ensemble sweep ───────────────────────────────────────────────────────────

print("\n--- Ensemble sweep ---")
print(f"{'w_v2':>5} {'w_v5':>5} {'thresh':>6} {'n_pos':>6} {'pct':>6}")
for w_v2 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_v5 = 1.0 - w_v2
    proba_ens = w_v2 * proba_v2 + w_v5 * proba_v5
    for t in [0.50, 0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]:
        n_pos = np.sum(proba_ens >= t)
        print(f"  {w_v2:.1f}   {w_v5:.1f}   {t:.2f}   {n_pos:5d}  {100 * n_pos / len(groups):.1f}%")


# ── Disagreement analysis ────────────────────────────────────────────────────

v2_pred = (proba_v2 >= 0.64).astype(int)
v5_pred = (proba_v5 >= 0.64).astype(int)
disagree_idx = np.where(v2_pred != v5_pred)[0]

print(f"\nV2 vs V5 disagree on {len(disagree_idx)} groups:")
for idx in disagree_idx[:30]:
    print(f"  Group {idx}: v2={proba_v2[idx]:.4f}({v2_pred[idx]}) "
          f"v5={proba_v5[idx]:.4f}({v5_pred[idx]})")


# ── Save submissions ─────────────────────────────────────────────────────────

print("\n--- Saving submissions ---")

# V5 standalone
for t in [0.62, 0.64, 0.66]:
    labels = (proba_v5 >= t).astype(int)
    fname = f"submission_v5_t{t:.2f}.txt"
    with open(fname, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"  {fname}: Pos={np.sum(labels)}")

# Ensemble variations
for w_v2 in [0.3, 0.4, 0.5]:
    w_v5 = 1.0 - w_v2
    proba_ens = w_v2 * proba_v2 + w_v5 * proba_v5
    for t in [0.60, 0.62, 0.64, 0.66]:
        labels = (proba_ens >= t).astype(int)
        fname = f"submission_v2v5_w{w_v2:.1f}_t{t:.2f}.txt"
        with open(fname, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")
        print(f"  {fname}: Pos={np.sum(labels)}")

print("\n*** DONE ***")
