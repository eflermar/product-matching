"""
Shared feature extraction for the product-matching pipeline.

Provides:
  - EmbeddingLookup: loads HDF5 embeddings into RAM
  - compute_pair_features: 20 base features for an item pair
  - compute_pair_features_pca: 84 features (20 base + 64 PCA diffs)
  - compute_group_features: 25 group-level features from pair probabilities
"""
import h5py
import numpy as np

# ── Feature name lists (for XGBoost & diagnostics) ──────────────────────────

PAIR_FEATURE_NAMES = [
    'text_cosine_sim', 'img_cosine_sim',
    'text_l2_dist', 'img_l2_dist',
    'text_diff_mean', 'text_diff_std', 'text_diff_max',
    'img_diff_mean', 'img_diff_std', 'img_diff_max',
    'text_prod_mean', 'img_prod_mean',
    'combined_sim', 'sim_diff',
    'price_ratio', 'price_diff',
    'dept_match', 'color_match', 'brand_match', 'total_matches',
]

PCA_FEATURE_NAMES = (
    [f'text_pca_diff_{i}' for i in range(32)] +
    [f'img_pca_diff_{i}' for i in range(32)]
)

ALL_PAIR_FEATURE_NAMES = PAIR_FEATURE_NAMES + PCA_FEATURE_NAMES

GROUP_FEATURE_NAMES = [
    'prob_max', 'prob_2nd', 'prob_3rd',
    'prob_mean', 'prob_std', 'prob_median',
    'prob_gap_1_2',
    'count_above_50', 'count_above_70', 'count_above_90',
    'text_sim_max', 'text_sim_mean', 'text_sim_std',
    'price_ratio_mean', 'price_ratio_max',
    'dept_match_count', 'color_match_count', 'brand_match_count',
    'total_meta_matches', 'n_unique_depts', 'n_unique_colors',
    'prob_x_text_max', 'prob_x_img_max', 'prob_x_price_max', 'prob_x_meta',
]

N_BASE_PAIR_FEATURES = 20
N_PCA_PAIR_FEATURES = 84
N_GROUP_FEATURES = 25


# ── Embedding storage ────────────────────────────────────────────────────────

class EmbeddingLookup:
    """Loads an HDF5 embedding file fully into RAM for fast indexed access."""

    def __init__(self, h5_path):
        print(f"Loading {h5_path} into RAM...")
        with h5py.File(h5_path, 'r') as f:
            raw_ids = f['item_ids'][:]
            self.id_to_idx = {uid.decode('utf-8'): i for i, uid in enumerate(raw_ids)}
            self.text_emb = f['text_emb'][:].astype(np.float16)
            self.image_emb = f['image_emb'][:].astype(np.float16)
            self.price = f['price'][:]
            self.dept_id = f['dept_id'][:]
            self.color_id = f['color_id'][:]
            self.brand_id = f['brand_id'][:]
        print(f"  Loaded {len(self.id_to_idx)} items.")

    def resolve_indices(self, item_ids):
        """Map string item IDs → integer indices. Returns -1 for missing items."""
        return np.array([self.id_to_idx.get(str(x), -1) for x in item_ids])


# ── Pair-level features ──────────────────────────────────────────────────────

def _compute_base_pair_block(lookup, idx_a, idx_b):
    """
    Compute the 20 base pair features for a block of index pairs.
    Returns (features, text_sim, img_sim, price_ratio, dept_match, color_match, brand_match).
    The extra returns are reused by group feature extraction.
    """
    text_a = lookup.text_emb[idx_a].astype(np.float32)
    text_b = lookup.text_emb[idx_b].astype(np.float32)
    img_a = lookup.image_emb[idx_a].astype(np.float32)
    img_b = lookup.image_emb[idx_b].astype(np.float32)

    # Similarity & distance
    text_sim = np.sum(text_a * text_b, axis=1)
    img_sim = np.sum(img_a * img_b, axis=1)
    text_diff = text_a - text_b
    img_diff = img_a - img_b
    text_l2 = np.sqrt(np.sum(text_diff ** 2, axis=1))
    img_l2 = np.sqrt(np.sum(img_diff ** 2, axis=1))
    text_abs_diff = np.abs(text_diff)
    img_abs_diff = np.abs(img_diff)

    # Metadata
    price_a = lookup.price[idx_a].flatten()
    price_b = lookup.price[idx_b].flatten()
    dept_a = lookup.dept_id[idx_a].flatten()
    dept_b = lookup.dept_id[idx_b].flatten()
    color_a = lookup.color_id[idx_a].flatten()
    color_b = lookup.color_id[idx_b].flatten()
    brand_a = lookup.brand_id[idx_a].flatten()
    brand_b = lookup.brand_id[idx_b].flatten()

    price_ratio = np.minimum(price_a, price_b) / (np.maximum(price_a, price_b) + 1e-5)
    dept_match = ((dept_a == dept_b) & (dept_a != 0)).astype(np.float32)
    color_match = ((color_a == color_b) & (color_a != 0)).astype(np.float32)
    brand_match = ((brand_a == brand_b) & (brand_a != 0)).astype(np.float32)

    features = np.column_stack([
        text_sim, img_sim, text_l2, img_l2,
        np.mean(text_abs_diff, axis=1), np.std(text_abs_diff, axis=1), np.max(text_abs_diff, axis=1),
        np.mean(img_abs_diff, axis=1), np.std(img_abs_diff, axis=1), np.max(img_abs_diff, axis=1),
        np.mean(text_a * text_b, axis=1), np.mean(img_a * img_b, axis=1),
        (text_sim + img_sim) / 2.0, np.abs(text_sim - img_sim),
        price_ratio, np.abs(price_a - price_b),
        dept_match, color_match, brand_match,
        dept_match + color_match + brand_match,
    ])

    return features, text_sim, img_sim, price_ratio, dept_match, color_match, brand_match


def compute_pair_features(lookup, idx_a, idx_b, chunk_size=50000):
    """Compute 20 base pair features in chunks. Returns (N, 20) float32 array."""
    n = len(idx_a)
    result = np.empty((n, N_BASE_PAIR_FEATURES), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block_a, block_b = idx_a[start:end], idx_b[start:end]
        features, *_ = _compute_base_pair_block(lookup, block_a, block_b)
        result[start:end] = features

    return result


def compute_pair_features_pca(lookup, idx_a, idx_b, text_pca_emb, img_pca_emb,
                               chunk_size=50000):
    """Compute 84 pair features (20 base + 32 text PCA diffs + 32 img PCA diffs)."""
    n = len(idx_a)
    result = np.empty((n, N_PCA_PAIR_FEATURES), dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block_a, block_b = idx_a[start:end], idx_b[start:end]

        base_features, *_ = _compute_base_pair_block(lookup, block_a, block_b)

        text_pca_diff = np.abs(text_pca_emb[block_a] - text_pca_emb[block_b])
        img_pca_diff = np.abs(img_pca_emb[block_a] - img_pca_emb[block_b])

        result[start:end] = np.column_stack([base_features, text_pca_diff, img_pca_diff])

    return result


# ── Group-level features ─────────────────────────────────────────────────────

def _aggregate_group(pair_probs, text_sims, img_sims, price_ratios,
                     dept_matches, color_matches, brand_matches, group_meta):
    """Compute 25 group features from one group's pair-level statistics."""
    sorted_probs = np.sort(pair_probs)[::-1]

    prob_max = sorted_probs[0]
    prob_2nd = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    prob_3rd = sorted_probs[2] if len(sorted_probs) > 2 else 0.0

    dept_count = float(np.sum(dept_matches))
    color_count = float(np.sum(color_matches))
    brand_count = float(np.sum(brand_matches))
    meta_total = dept_count + color_count + brand_count

    n_unique_depts = len({d for d, c in group_meta if d != 0})
    n_unique_colors = len({c for d, c in group_meta if c != 0})

    return [
        prob_max, prob_2nd, prob_3rd,
        float(np.mean(pair_probs)),
        float(np.std(pair_probs)),
        float(np.median(pair_probs)),
        prob_max - prob_2nd,
        float(np.sum(pair_probs >= 0.5)),
        float(np.sum(pair_probs >= 0.7)),
        float(np.sum(pair_probs >= 0.9)),
        float(np.max(text_sims)),
        float(np.mean(text_sims)),
        float(np.std(text_sims)),
        float(np.mean(price_ratios)),
        float(np.max(price_ratios)),
        dept_count, color_count, brand_count, meta_total,
        n_unique_depts, n_unique_colors,
        prob_max * float(np.max(text_sims)),
        prob_max * float(np.max(img_sims)),
        prob_max * float(np.max(price_ratios)),
        prob_max * meta_total,
    ]


def compute_group_features(groups, pair_model, lookup, chunk_size=2000,
                           text_pca_emb=None, img_pca_emb=None):
    """
    Compute 25 group features for a list of groups.

    Each group is a list of 5 item IDs. All C(5,2)=10 pairs are scored by the
    pair model, then aggregated into group-level statistics.

    If text_pca_emb/img_pca_emb are provided, uses 84-feature PCA pairs;
    otherwise uses 20-feature base pairs.
    """
    use_pca = (text_pca_emb is not None and img_pca_emb is not None)
    result = np.zeros((len(groups), N_GROUP_FEATURES), dtype=np.float32)

    for chunk_start in range(0, len(groups), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(groups))
        chunk = groups[chunk_start:chunk_end]

        # Collect all pairs across the chunk
        group_metadata = []
        pair_group_ids, pair_idx_a, pair_idx_b = [], [], []

        for group_i, group in enumerate(chunk):
            item_indices = [lookup.id_to_idx.get(str(x)) for x in group]
            meta = [
                (int(lookup.dept_id[ix][0]), int(lookup.color_id[ix][0]))
                for ix in item_indices if ix is not None
            ]
            group_metadata.append(meta)

            for i in range(5):
                for j in range(i + 1, 5):
                    if item_indices[i] is not None and item_indices[j] is not None:
                        pair_group_ids.append(group_i)
                        pair_idx_a.append(item_indices[i])
                        pair_idx_b.append(item_indices[j])

        if not pair_idx_a:
            continue

        group_ids = np.array(pair_group_ids)
        idx_a = np.array(pair_idx_a)
        idx_b = np.array(pair_idx_b)

        # Compute pair features + raw similarities for group aggregation
        (base_feats, text_sims, img_sims, price_ratios,
         dept_matches, color_matches, brand_matches) = _compute_base_pair_block(
            lookup, idx_a, idx_b
        )

        if use_pca:
            text_pca_diff = np.abs(text_pca_emb[idx_a] - text_pca_emb[idx_b])
            img_pca_diff = np.abs(img_pca_emb[idx_a] - img_pca_emb[idx_b])
            pair_feats = np.column_stack([base_feats, text_pca_diff, img_pca_diff])
        else:
            pair_feats = base_feats

        pair_probs = pair_model.predict_proba(pair_feats.astype(np.float32))[:, 1]

        # Aggregate per group
        for group_i in range(len(chunk)):
            mask = (group_ids == group_i)
            if not np.any(mask):
                continue

            result[chunk_start + group_i] = _aggregate_group(
                pair_probs[mask],
                text_sims[mask], img_sims[mask], price_ratios[mask],
                dept_matches[mask], color_matches[mask], brand_matches[mask],
                group_metadata[group_i],
            )

        if chunk_end % 5000 < chunk_size or chunk_end == len(groups):
            print(f"  {chunk_end}/{len(groups)}")

    return result
