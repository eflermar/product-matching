"""
Pipeline V5: PCA-enriched pair features for finer-grained matching.

Extends V2 by adding 64 PCA features to each pair (32 text + 32 image),
giving XGBoost per-dimension information instead of just scalar cosine sim.

Two-stage approach:
  1. Pair model: 84 features (20 base + 64 PCA diffs)
  2. Group model: 25 group features aggregated from pair probabilities

Produces: glami_xgb_pairwise_ft_v5.json, glami_group_meta_ft_v5.json, submission_ft_v5.txt
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

from features import (
    EmbeddingLookup, compute_pair_features_pca, compute_group_features,
    ALL_PAIR_FEATURE_NAMES, GROUP_FEATURE_NAMES, N_PCA_PAIR_FEATURES,
)

np.random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_H5 = 'glami_embeddings_train_ft.h5'
PHASE1_H5 = 'glami_embeddings_phase_1_ft.h5'
PAIR_MODEL_OUT = 'glami_xgb_pairwise_ft_v5.json'
GROUP_MODEL_OUT = 'glami_group_meta_ft_v5.json'
SUBMISSION_OUT = 'submission_ft_v5.txt'

PCA_DIM = 32


# ── PCA ──────────────────────────────────────────────────────────────────────

def fit_pca(lookup, n_components=PCA_DIM, sample_size=100000):
    """Fit PCA on a random sample of training embeddings."""
    print(f"\nFitting PCA (dim={n_components}) on {sample_size} samples...")
    n = len(lookup.text_emb)
    idx = np.random.choice(n, min(sample_size, n), replace=False)

    text_pca = PCA(n_components=n_components, random_state=42)
    img_pca = PCA(n_components=n_components, random_state=42)

    text_pca.fit(lookup.text_emb[idx].astype(np.float32))
    img_pca.fit(lookup.image_emb[idx].astype(np.float32))

    print(f"  Text PCA: {sum(text_pca.explained_variance_ratio_) * 100:.1f}% variance")
    print(f"  Image PCA: {sum(img_pca.explained_variance_ratio_) * 100:.1f}% variance")
    return text_pca, img_pca


def apply_pca(embeddings, pca_model, chunk_size=100000):
    """Transform full embedding array through PCA in memory-safe chunks."""
    n = len(embeddings)
    result = np.empty((n, pca_model.n_components), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        result[start:end] = pca_model.transform(embeddings[start:end].astype(np.float32))
    return result


# ── Data mining (same strategy as V2) ────────────────────────────────────────

def mine_pair_data(df_train):
    """Create balanced pair training data."""
    item_to_label = dict(zip(df_train['itemId'], df_train['label']))
    grouped = df_train.groupby('label')['itemId'].apply(list)

    print("\nMining positive pairs...")
    positives = []
    for label, items in grouped.items():
        if len(items) < 2:
            continue
        combos = list(combinations(items, 2))
        if len(combos) > 3:
            chosen = np.random.choice(len(combos), 3, replace=False)
            combos = [combos[i] for i in chosen]
        positives.extend((a, b, 1) for a, b in combos)
    print(f"  Positive: {len(positives)}")

    print("Mining hard negatives (same dept)...")
    dept_to_items = {}
    for _, row in df_train.iterrows():
        if pd.notna(row['departmentIds']):
            dept_to_items.setdefault(str(row['departmentIds']), []).append(str(row['itemId']))

    hard_negs = []
    num_hard = len(positives) // 2
    dept_keys = list(dept_to_items.keys())
    np.random.shuffle(dept_keys)
    for dept in dept_keys:
        items = dept_to_items[dept]
        if len(items) < 2:
            continue
        np.random.shuffle(items)
        for i in range(0, len(items) - 1, 2):
            a, b = items[i], items[i + 1]
            if item_to_label.get(a) != item_to_label.get(b):
                hard_negs.append((a, b, 0))
            if len(hard_negs) >= num_hard:
                break
        if len(hard_negs) >= num_hard:
            break
    print(f"  Hard neg: {len(hard_negs)}")

    print("Mining random negatives...")
    all_ids = df_train['itemId'].values
    rand_negs = []
    num_rand = len(positives) // 2
    while len(rand_negs) < num_rand:
        i, j = np.random.randint(0, len(all_ids), 2)
        a, b = str(all_ids[i]), str(all_ids[j])
        if a != b and item_to_label.get(a) != item_to_label.get(b):
            rand_negs.append((a, b, 0))
    print(f"  Random neg: {len(rand_negs)}")

    all_pairs = positives + hard_negs + rand_negs
    np.random.shuffle(all_pairs)
    print(f"  Total: {len(all_pairs)}")
    return all_pairs


def mine_group_data(df_train, num_pos=15000, num_hard_neg=15000, num_rand_neg=30000):
    """Create group training data (identical to V2)."""
    grouped = df_train.groupby('label')['itemId'].apply(list)
    all_labels = list(grouped.keys())

    labels_by_min_size = {
        2: {l: items for l, items in grouped.items() if len(items) >= 2},
        3: {l: items for l, items in grouped.items() if len(items) >= 3},
        4: {l: items for l, items in grouped.items() if len(items) >= 4},
    }

    dept_to_labels = {}
    for _, row in df_train.iterrows():
        if pd.notna(row['departmentIds']):
            dept_to_labels.setdefault(str(row['departmentIds']), set()).add(row['label'])
    dept_to_labels = {d: list(ls) for d, ls in dept_to_labels.items() if len(ls) >= 5}

    groups, labels = [], []

    distribution = [(2, int(num_pos * 0.5)), (3, int(num_pos * 0.3)),
                    (4, num_pos - int(num_pos * 0.5) - int(num_pos * 0.3))]

    for n_same, count in distribution:
        pool = labels_by_min_size.get(n_same, labels_by_min_size[2])
        pool_keys = list(pool.keys())
        for i in range(count):
            label = pool_keys[i % len(pool_keys)]
            same_items = list(np.random.choice(pool[label], n_same, replace=False))
            others = []
            for _ in range(200):
                rand_label = all_labels[np.random.randint(len(all_labels))]
                if rand_label != label:
                    rand_items = grouped[rand_label]
                    others.append(str(rand_items[np.random.randint(len(rand_items))]))
                if len(others) >= 5 - n_same:
                    break
            group = [str(x) for x in same_items] + others[:5 - n_same]
            while len(group) < 5:
                group.append(group[-1])
            np.random.shuffle(group)
            groups.append(group)
            labels.append(1)

    dept_keys = list(dept_to_labels.keys())
    hard_count = 0
    for _ in range(num_hard_neg * 3):
        if hard_count >= num_hard_neg:
            break
        dept = dept_keys[np.random.randint(len(dept_keys))]
        if len(dept_to_labels[dept]) < 5:
            continue
        chosen = np.random.choice(dept_to_labels[dept], 5, replace=False)
        group = [str(grouped[l][np.random.randint(len(grouped[l]))]) for l in chosen]
        groups.append(group)
        labels.append(0)
        hard_count += 1

    for _ in range(num_rand_neg):
        chosen_idx = np.random.choice(len(all_labels), 5, replace=False)
        group = [str(grouped[all_labels[c]][np.random.randint(len(grouped[all_labels[c]]))]) for c in chosen_idx]
        groups.append(group)
        labels.append(0)

    labels = np.array(labels)
    print(f"Groups: {len(groups)} (Pos: {np.sum(labels == 1)}, Neg: {np.sum(labels == 0)})")
    return groups, labels


# ── Training ─────────────────────────────────────────────────────────────────

def train_pair_model(lookup, text_pca_emb, img_pca_emb):
    print("\n" + "=" * 60)
    print("PART 1: TRAIN PAIR-LEVEL XGBOOST (84 PCA features)")
    print("=" * 60)

    df_train = pd.read_csv('data/items_train.csv', dtype={'itemId': str})
    all_pairs = mine_pair_data(df_train)

    items_a = [p[0] for p in all_pairs]
    items_b = [p[1] for p in all_pairs]
    y_all = np.array([p[2] for p in all_pairs], dtype=np.int32)

    idx_a = lookup.resolve_indices(items_a)
    idx_b = lookup.resolve_indices(items_b)
    valid = (idx_a >= 0) & (idx_b >= 0)
    idx_a, idx_b, y_all = idx_a[valid], idx_b[valid], y_all[valid]
    print(f"  Valid pairs: {len(y_all)}")

    X = compute_pair_features_pca(lookup, idx_a, idx_b, text_pca_emb, img_pca_emb)
    print(f"  Features: {X.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    print(f"Train: {len(y_train)} (Pos: {pos_count}, Neg: {neg_count})")

    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=8,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.6,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=neg_count / pos_count, eval_metric='logloss',
        early_stopping_rounds=50, random_state=42, tree_method='hist',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    print("\n--- Pair Validation ---")
    print(classification_report(y_val, model.predict(X_val)))

    importances = sorted(zip(ALL_PAIR_FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("Top-20 Feature Importances:")
    for name, imp in importances[:20]:
        print(f"  {name:25s} {imp:.4f}")

    model.save_model(PAIR_MODEL_OUT)
    print(f"Saved: {PAIR_MODEL_OUT}")
    return model


def train_group_model(pair_model, lookup, text_pca_emb, img_pca_emb):
    print("\n" + "=" * 60)
    print("PART 2: TRAIN GROUP-LEVEL META-MODEL")
    print("=" * 60)

    df_train = pd.read_csv('data/items_train.csv', dtype={'itemId': str})
    groups, y_groups = mine_group_data(df_train)

    print("Extracting group features...")
    X_groups = compute_group_features(
        groups, pair_model, lookup,
        text_pca_emb=text_pca_emb, img_pca_emb=img_pca_emb,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_groups, y_groups, test_size=0.2, random_state=42, stratify=y_groups
    )
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    print(f"Train: {len(y_train)} (Pos: {pos_count}, Neg: {neg_count})")

    model = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg_count / pos_count, eval_metric='logloss',
        early_stopping_rounds=50, random_state=42, tree_method='hist',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    print("\n--- Group Validation ---")
    print(classification_report(y_val, model.predict(X_val)))

    y_proba = model.predict_proba(X_val)[:, 1]
    best_acc, best_threshold = 0, 0.5
    for t in np.arange(0.10, 0.96, 0.02):
        acc = accuracy_score(y_val, (y_proba >= t).astype(int))
        f1 = f1_score(y_val, (y_proba >= t).astype(int))
        if acc > best_acc:
            best_acc, best_threshold = acc, t
        print(f"  t={t:.2f}  Acc={acc:.4f}  F1={f1:.4f}")

    print(f"\n*** BEST THRESHOLD: {best_threshold:.2f} (Acc={best_acc:.4f}) ***")

    model.save_model(GROUP_MODEL_OUT)
    print(f"Saved: {GROUP_MODEL_OUT}")

    importances = sorted(zip(GROUP_FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\nGroup Feature Importances:")
    for name, imp in importances:
        print(f"  {name:25s} {imp:.4f}")

    return model, best_threshold


def predict(pair_model, group_model, threshold, phase1_lookup,
            text_pca_emb, img_pca_emb):
    print("\n" + "=" * 60)
    print("PART 3: PREDICT")
    print("=" * 60)

    df_task = pd.read_csv('data/task_1.csv')
    groups = [[str(row[f'item{i}']) for i in range(1, 6)] for _, row in df_task.iterrows()]
    print(f"Task groups: {len(groups)}")

    print("Extracting group features...")
    X_task = compute_group_features(
        groups, pair_model, phase1_lookup,
        text_pca_emb=text_pca_emb, img_pca_emb=img_pca_emb,
    )
    proba = group_model.predict_proba(X_task)[:, 1]
    labels = (proba >= threshold).astype(int)

    print(f"Threshold: {threshold}")
    print(f"Positive: {np.sum(labels)}/{len(labels)} ({100 * np.mean(labels):.1f}%)")

    with open(SUBMISSION_OUT, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Saved: {SUBMISSION_OUT}")
    return proba


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_lookup = EmbeddingLookup(TRAIN_H5)
    text_pca, img_pca = fit_pca(train_lookup)

    print("Applying PCA to training embeddings...")
    train_text_pca = apply_pca(train_lookup.text_emb, text_pca)
    train_img_pca = apply_pca(train_lookup.image_emb, img_pca)

    pair_model = train_pair_model(train_lookup, train_text_pca, train_img_pca)
    group_model, best_threshold = train_group_model(
        pair_model, train_lookup, train_text_pca, train_img_pca
    )

    phase1_lookup = EmbeddingLookup(PHASE1_H5)
    print("Applying PCA to phase1 embeddings...")
    phase1_text_pca = apply_pca(phase1_lookup.text_emb, text_pca)
    phase1_img_pca = apply_pca(phase1_lookup.image_emb, img_pca)

    predict(pair_model, group_model, best_threshold,
            phase1_lookup, phase1_text_pca, phase1_img_pca)

    print("\n*** ALL DONE ***")
