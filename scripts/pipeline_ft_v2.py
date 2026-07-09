"""
Pipeline V2: Train pair + group XGBoost models on fine-tuned embeddings.

Two-stage approach:
  1. Pair model: classify whether two items share a product label (20 features)
  2. Group model: classify whether a group of 5 items contains any matching pair

Training data strategy:
  - Pairs: 50% hard negatives (same department) + 50% random negatives
  - Groups: diverse positives (2/3/4 same-label items) + hard & random negatives

Produces: glami_xgb_pairwise_ft.json, glami_group_meta_ft_v2.json, submission_ft_v2.txt
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score

from features import (
    EmbeddingLookup, compute_pair_features, compute_group_features,
    PAIR_FEATURE_NAMES, GROUP_FEATURE_NAMES,
)

np.random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_H5 = 'glami_embeddings_train_ft.h5'
PHASE1_H5 = 'glami_embeddings_phase_1_ft.h5'
PAIR_MODEL_OUT = 'glami_xgb_pairwise_ft.json'
GROUP_MODEL_OUT = 'glami_group_meta_ft_v2.json'
SUBMISSION_OUT = 'submission_ft_v2.txt'


# ── Data mining helpers ──────────────────────────────────────────────────────

def mine_pair_data(df_train):
    """Create balanced pair training data: positives + hard negatives + random negatives."""
    item_to_label = dict(zip(df_train['itemId'], df_train['label']))
    grouped = df_train.groupby('label')['itemId'].apply(list)

    # Positive pairs: up to 3 per label
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

    # Hard negatives: same department, different label
    print("Mining hard negatives (same dept)...")
    dept_to_items = {}
    for _, row in df_train.iterrows():
        dept = row['departmentIds']
        if pd.notna(dept):
            dept_to_items.setdefault(str(dept), []).append(str(row['itemId']))

    hard_negs = []
    num_hard = len(positives) // 2
    for items in dept_to_items.values():
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

    # Random negatives
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
    """Create group training data with diverse positives and hard negatives."""
    grouped = df_train.groupby('label')['itemId'].apply(list)
    all_labels = list(grouped.keys())

    labels_by_min_size = {
        2: {l: items for l, items in grouped.items() if len(items) >= 2},
        3: {l: items for l, items in grouped.items() if len(items) >= 3},
        4: {l: items for l, items in grouped.items() if len(items) >= 4},
    }

    # Department → labels for hard negatives
    dept_to_labels = {}
    for _, row in df_train.iterrows():
        if pd.notna(row['departmentIds']):
            dept_to_labels.setdefault(str(row['departmentIds']), set()).add(row['label'])
    dept_to_labels = {d: list(ls) for d, ls in dept_to_labels.items() if len(ls) >= 5}

    groups, labels = [], []

    # ── Positive groups: mix of 2/3/4 same-label items ──
    distribution = [(2, int(num_pos * 0.5)), (3, int(num_pos * 0.3)), (4, num_pos - int(num_pos * 0.5) - int(num_pos * 0.3))]

    for n_same, count in distribution:
        pool = labels_by_min_size.get(n_same, labels_by_min_size[2])
        pool_keys = list(pool.keys())
        for i in range(count):
            label = pool_keys[i % len(pool_keys)]
            same_items = list(np.random.choice(pool[label], n_same, replace=False))

            # Fill remaining slots with random items from other labels
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

    # ── Hard negative groups: 5 items from same department, all different labels ──
    dept_keys = list(dept_to_labels.keys())
    hard_count = 0
    for _ in range(num_hard_neg * 3):
        if hard_count >= num_hard_neg:
            break
        dept = dept_keys[np.random.randint(len(dept_keys))]
        dept_labels = dept_to_labels[dept]
        if len(dept_labels) < 5:
            continue
        chosen = np.random.choice(dept_labels, 5, replace=False)
        group = [str(grouped[l][np.random.randint(len(grouped[l]))]) for l in chosen]
        groups.append(group)
        labels.append(0)
        hard_count += 1

    # ── Random negative groups: 5 items from 5 different labels ──
    for _ in range(num_rand_neg):
        chosen_idx = np.random.choice(len(all_labels), 5, replace=False)
        group = [str(grouped[all_labels[c]][np.random.randint(len(grouped[all_labels[c]]))]) for c in chosen_idx]
        groups.append(group)
        labels.append(0)

    labels = np.array(labels)
    n_pos, n_neg = np.sum(labels == 1), np.sum(labels == 0)
    print(f"Groups: {len(groups)} (Pos: {n_pos}, Neg: {n_neg})")
    return groups, labels


# ── Training ─────────────────────────────────────────────────────────────────

def train_pair_model(lookup):
    print("\n" + "=" * 60)
    print("PART 1: TRAIN PAIR-LEVEL XGBOOST")
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

    X = compute_pair_features(lookup, idx_a, idx_b)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_all, test_size=0.15, random_state=42, stratify=y_all
    )
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    print(f"Train: {len(y_train)} (Pos: {pos_count}, Neg: {neg_count})")

    model = xgb.XGBClassifier(
        n_estimators=1500, learning_rate=0.05, max_depth=8,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=neg_count / pos_count, eval_metric='logloss',
        early_stopping_rounds=50, random_state=42, tree_method='hist',
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    print("\n--- Pair Validation ---")
    print(classification_report(y_val, model.predict(X_val)))

    importances = sorted(zip(PAIR_FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("Feature Importances:")
    for name, imp in importances:
        print(f"  {name:25s} {imp:.4f}")

    model.save_model(PAIR_MODEL_OUT)
    print(f"Saved: {PAIR_MODEL_OUT}")
    return model


def train_group_model(pair_model, lookup):
    print("\n" + "=" * 60)
    print("PART 2: TRAIN GROUP-LEVEL META-MODEL")
    print("=" * 60)

    df_train = pd.read_csv('data/items_train.csv', dtype={'itemId': str})
    groups, y_groups = mine_group_data(df_train)

    print("Extracting group features...")
    X_groups = compute_group_features(groups, pair_model, lookup)

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


def predict(pair_model, group_model, threshold):
    print("\n" + "=" * 60)
    print("PART 3: PREDICT")
    print("=" * 60)

    lookup = EmbeddingLookup(PHASE1_H5)
    df_task = pd.read_csv('data/task_1.csv')
    groups = [[str(row[f'item{i}']) for i in range(1, 6)] for _, row in df_task.iterrows()]
    print(f"Task groups: {len(groups)}")

    print("Extracting group features...")
    X_task = compute_group_features(groups, pair_model, lookup)
    proba = group_model.predict_proba(X_task)[:, 1]
    labels = (proba >= threshold).astype(int)

    print(f"Threshold: {threshold}")
    print(f"Positive: {np.sum(labels)}/{len(labels)} ({100 * np.mean(labels):.1f}%)")

    with open(SUBMISSION_OUT, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Saved: {SUBMISSION_OUT}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    lookup = EmbeddingLookup(TRAIN_H5)
    pair_model = train_pair_model(lookup)
    group_model, best_threshold = train_group_model(pair_model, lookup)
    predict(pair_model, group_model, best_threshold)
    print("\n*** ALL DONE ***")
