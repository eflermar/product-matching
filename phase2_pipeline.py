"""
Phase 2 Pipeline: End-to-end product grouping.

Workflow:
  1. Load fine-tuned embeddings for phase_2 (or phase_1 for leaderboard testing).
  2. FAISS ANN: top-K text + image neighbors per item → candidate pairs (cached).
  3. Score candidate pairs (cached per scoring mode).
  4. Optional: zero out cross-department pairs (--dept_filter).
  5. Connected components on edges above threshold → product groups.
  6. Split any component > 100 items via MST binary search.
  7. Write submission CSV: one group per line, comma-separated item_ids.

Scoring modes (mutually exclusive):
  default              weighted average: text_weight * text_sim + (1-text_weight) * img_sim
  --text_weight 0.7    favour text over image (useful for cross-geo matching)
  --and_sim            min(text_sim, img_sim) — both modalities must agree
  --use_model          XGBoost V6 (only useful after retraining on FAISS negatives)

Usage:
  python phase2_pipeline.py --phase1 --sweep                          # baseline rawsim sweep
  python phase2_pipeline.py --phase1 --text_weight 0.7 --dept_filter --sweep
  python phase2_pipeline.py --phase1 --text_weight 0.8 --dept_filter --sweep
  python phase2_pipeline.py --phase1 --and_sim --dept_filter --sweep
  python phase2_pipeline.py --phase1 --top_k 100 --dept_filter --sweep   # overnight
  python phase2_pipeline.py --threshold 0.95                          # phase_2 final
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import faiss
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import networkx as nx
from tqdm import tqdm

from features import EmbeddingLookup, compute_pair_features_pca

np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
EMB_H5 = {
    'ft': {
        'train':   'glami_embeddings_train_ft.h5',
        'phase_1': 'glami_embeddings_phase_1_ft.h5',
        'phase_2': 'glami_embeddings_phase_2_ft.h5',
    },
    'ft_v2': {
        'train':   'glami_embeddings_train_ft_v2.h5',
        'phase_1': 'glami_embeddings_phase_1_ft_v2.h5',
        'phase_2': 'glami_embeddings_phase_2_ft_v2.h5',
    },
    'e5dino': {
        'train':   'glami_embeddings_train_e5dino.h5',
        'phase_1': 'glami_embeddings_phase_1_e5dino.h5',
        'phase_2': 'glami_embeddings_phase_2_e5dino.h5',
    },
    'v2': {
        'train':   'glami_embeddings_train_v2.h5',
        'phase_1': 'glami_embeddings_phase_1_v2.h5',
        'phase_2': 'glami_embeddings_phase_2_v2.h5',
    },
}
TRAIN_H5_BY_EMB = {k: v['train'] for k, v in EMB_H5.items()}  # PCA source per emb set
PAIR_MODEL_PATH = 'glami_xgb_hardneg.json'   # retrained on sim>=0.88 pairs
CROSSENCODER_PATH   = 'finetuned_crossencoder_v2'
CROSSENCODER_PRESIM = 0.88   # raw_sim pre-filter before cross-encoder inference
DESC_CHARS          = 150
ITEM_CSVS = {
    'phase_1': 'data/items_phase_1.csv',
    'phase_2': 'data/items_phase_2.csv',
}
PCA_DIM         = 32
TOP_K           = 50
MAX_GROUP_SIZE  = 100


# ── PCA ───────────────────────────────────────────────────────────────────────

def fit_pca(train_lookup, n_components=PCA_DIM, sample_size=100_000):
    print(f"Fitting PCA (dim={n_components}) on {sample_size} training samples...")
    n   = len(train_lookup.text_emb)
    idx = np.random.choice(n, min(sample_size, n), replace=False)
    text_pca = PCA(n_components=n_components, random_state=42)
    img_pca  = PCA(n_components=n_components, random_state=42)
    text_pca.fit(train_lookup.text_emb[idx].astype(np.float32))
    img_pca.fit(train_lookup.image_emb[idx].astype(np.float32))
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


# ── FAISS candidate generation ────────────────────────────────────────────────

def _build_ivf_index(emb, nlist=400, nprobe=20):
    d         = emb.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index     = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    index.nprobe = nprobe
    return index


def get_candidate_pairs(lookup, top_k=TOP_K, batch_size=20_000, nlist=400, nprobe=20):
    """
    IVFFlat ANN in text and image space.
    Returns unique (i, j) index pairs with i < j as int32 array.
    """
    n        = len(lookup.text_emb)
    text_emb = lookup.text_emb.astype(np.float32)
    img_emb  = lookup.image_emb.astype(np.float32)

    faiss.omp_set_num_threads(os.cpu_count() or 8)

    print("Building IVF indices (training quantiser)...")
    text_index = _build_ivf_index(text_emb, nlist, nprobe)
    img_index  = _build_ivf_index(img_emb,  nlist, nprobe)
    print(f"FAISS IVF indices built ({n} items, {text_emb.shape[1]} dims, "
          f"nlist={nlist}, nprobe={nprobe})")

    pairs = set()
    for start in tqdm(range(0, n, batch_size), desc="FAISS ANN"):
        end = min(start + batch_size, n)
        for index, emb in [(text_index, text_emb), (img_index, img_emb)]:
            _, neighbors = index.search(emb[start:end], top_k + 1)
            for local_i, nbrs in enumerate(neighbors):
                item_i = start + local_i
                for j in nbrs:
                    if 0 <= j < n and j != item_i:
                        pairs.add((min(item_i, j), max(item_i, j)))

    arr = np.array(sorted(pairs), dtype=np.int32)
    print(f"Candidate pairs: {len(arr):,}")
    return arr


def get_anchor_pairs(phase_lookup, train_h5_path, df_anchor,
                     sim_threshold=0.92, top_k=10, max_per_label=30,
                     batch_size=10_000):
    """
    Find phase-item pairs linked via shared high-sim training anchors.

    For each phase item, searches top-k training neighbors. If combined_sim >= sim_threshold,
    records (phase_item → training_label). Phase items sharing a label get linked with
    score = min(sim_a, sim_b).  Recovers cross-geo pairs that are far apart in embedding
    space but each individually close to a training instance of the same product.

    Returns (pair_indices, pair_scores) as (N,2) int32 and (N,) float32.
    """
    print(f"Loading anchor embeddings from {train_h5_path}...")
    anchor = EmbeddingLookup(train_h5_path)

    iid_to_label = dict(zip(df_anchor['itemId'].astype(str), df_anchor['label']))
    idx_to_label = {anchor.id_to_idx[iid]: lbl
                    for iid, lbl in iid_to_label.items()
                    if iid in anchor.id_to_idx}
    print(f"  {len(idx_to_label):,} anchor items with labels")

    anchor_text = anchor.text_emb.astype(np.float32)
    anchor_img  = anchor.image_emb.astype(np.float32)
    n_anchor    = len(anchor_text)
    nlist       = min(500, n_anchor // 10)

    print(f"Building anchor FAISS index ({n_anchor:,} items, nlist={nlist})...")
    q     = faiss.IndexFlatIP(anchor_text.shape[1])
    index = faiss.IndexIVFFlat(q, anchor_text.shape[1], nlist,
                               faiss.METRIC_INNER_PRODUCT)
    index.train(anchor_text)
    index.add(anchor_text)
    index.nprobe = 25

    phase_text = phase_lookup.text_emb.astype(np.float32)
    phase_img  = phase_lookup.image_emb.astype(np.float32)
    n_phase    = len(phase_text)

    # phase_idx → {label: best combined_sim}
    item_anchors = defaultdict(dict)

    for start in tqdm(range(0, n_phase, batch_size), desc="Anchor search"):
        end = min(start + batch_size, n_phase)
        t_sims, nbrs = index.search(phase_text[start:end], top_k)
        for local_i, (srow, nrow) in enumerate(zip(t_sims, nbrs)):
            pidx = start + local_i
            for t_sim, aidx in zip(srow, nrow):
                if aidx < 0:
                    continue
                lbl = idx_to_label.get(int(aidx))
                if lbl is None:
                    continue
                v_sim    = float(np.dot(phase_img[pidx], anchor_img[int(aidx)]))
                combined = (float(t_sim) + v_sim) * 0.5
                if combined >= sim_threshold:
                    if combined > item_anchors[pidx].get(lbl, 0.0):
                        item_anchors[pidx][lbl] = combined

    del anchor
    print(f"  Phase items with anchors: {len(item_anchors):,}")

    label_to_items = defaultdict(list)
    for pidx, adict in item_anchors.items():
        for lbl, sim in adict.items():
            label_to_items[lbl].append((pidx, sim))

    pair_dict = {}
    for lbl, items in label_to_items.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: -x[1])
        items = items[:max_per_label]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, sa = items[i]
                b, sb = items[j]
                key   = (min(a, b), max(a, b))
                score = min(sa, sb)
                if key not in pair_dict or score > pair_dict[key]:
                    pair_dict[key] = score

    if not pair_dict:
        print("  No anchor pairs found.")
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)

    keys   = np.array(list(pair_dict.keys()), dtype=np.int32)
    scores = np.array(list(pair_dict.values()), dtype=np.float32)
    print(f"  Anchor pairs: {len(keys):,}  "
          f"score p50={np.median(scores):.3f}  p10={np.percentile(scores,10):.3f}")
    return keys, scores


# ── Pair scoring ──────────────────────────────────────────────────────────────

def score_pairs_weighted(lookup, pair_indices, text_weight=0.5, chunk_size=500_000):
    """
    Score pairs by text_weight * text_cosine_sim + (1-text_weight) * img_cosine_sim.

    text_weight=0.5 → plain average (rawsim baseline, F1=0.8796 at t=0.95).
    text_weight=0.7 → favours text, better for cross-geo matching where images
                      vary by angle/background but text titles are consistent.
    """
    img_weight = 1.0 - text_weight
    n      = len(pair_indices)
    scores = np.empty(n, dtype=np.float32)
    desc   = f"Scoring (text={text_weight:.1f}/img={img_weight:.1f})"
    for s in tqdm(range(0, n, chunk_size), desc=desc):
        e     = min(s + chunk_size, n)
        ia    = pair_indices[s:e, 0]
        ib    = pair_indices[s:e, 1]
        t_sim = np.sum(lookup.text_emb[ia].astype(np.float32) *
                       lookup.text_emb[ib].astype(np.float32), axis=1)
        v_sim = np.sum(lookup.image_emb[ia].astype(np.float32) *
                       lookup.image_emb[ib].astype(np.float32), axis=1)
        scores[s:e] = text_weight * t_sim + img_weight * v_sim
    pcts = np.percentile(scores, [50, 75, 90, 95, 99, 99.5])
    print(f"Score stats: mean={scores.mean():.3f}")
    print(f"  Percentiles — p50:{pcts[0]:.3f}  p75:{pcts[1]:.3f}  "
          f"p90:{pcts[2]:.3f}  p95:{pcts[3]:.3f}  p99:{pcts[4]:.3f}  p99.5:{pcts[5]:.3f}")
    print(f"  Above threshold — "
          f">0.80:{(scores>0.80).sum():,}  >0.85:{(scores>0.85).sum():,}  "
          f">0.90:{(scores>0.90).sum():,}  >0.93:{(scores>0.93).sum():,}  "
          f">0.95:{(scores>0.95).sum():,}  >0.97:{(scores>0.97).sum():,}")
    return scores


def score_pairs_and_sim(lookup, pair_indices, chunk_size=500_000):
    """
    Score pairs by min(text_cosine_sim, img_cosine_sim).
    Requires both modalities to agree — cuts false matches where only one is similar.
    """
    n      = len(pair_indices)
    scores = np.empty(n, dtype=np.float32)
    for s in tqdm(range(0, n, chunk_size), desc="Scoring (and-sim)"):
        e     = min(s + chunk_size, n)
        ia    = pair_indices[s:e, 0]
        ib    = pair_indices[s:e, 1]
        t_sim = np.sum(lookup.text_emb[ia].astype(np.float32) *
                       lookup.text_emb[ib].astype(np.float32), axis=1)
        v_sim = np.sum(lookup.image_emb[ia].astype(np.float32) *
                       lookup.image_emb[ib].astype(np.float32), axis=1)
        scores[s:e] = np.minimum(t_sim, v_sim)
    pcts = np.percentile(scores, [50, 75, 90, 95, 99, 99.5])
    print(f"Score stats (and-sim): mean={scores.mean():.3f}")
    print(f"  Percentiles — p50:{pcts[0]:.3f}  p75:{pcts[1]:.3f}  "
          f"p90:{pcts[2]:.3f}  p95:{pcts[3]:.3f}  p99:{pcts[4]:.3f}  p99.5:{pcts[5]:.3f}")
    return scores


def apply_borderline_boost(lookup, pair_indices, pair_scores,
                           score_low=0.15, score_high=0.30,
                           chunk_size=500_000):
    """
    Promote borderline edges (score in [score_low, score_high)) to score_high
    if they have brand_match AND dept_match AND color_match.

    Targets cross-geo same-product pairs where XGBoost score is suppressed by
    text_sim differences (different language titles) but metadata confirms same product.
    """
    scores   = pair_scores.copy()
    n_boosted = 0

    for s in tqdm(range(0, len(pair_indices), chunk_size), desc="Borderline boost"):
        e  = min(s + chunk_size, len(pair_indices))
        ia = pair_indices[s:e, 0]
        ib = pair_indices[s:e, 1]

        brand_a = lookup.brand_id[ia].flatten()
        brand_b = lookup.brand_id[ib].flatten()
        dept_a  = lookup.dept_id[ia].flatten()
        dept_b  = lookup.dept_id[ib].flatten()
        color_a = lookup.color_id[ia].flatten()
        color_b = lookup.color_id[ib].flatten()

        brand_match = (brand_a == brand_b) & (brand_a != 0)
        dept_match  = (dept_a  == dept_b)  & (dept_a  != 0)
        color_match = (color_a == color_b) & (color_a != 0)

        borderline = (scores[s:e] >= score_low) & (scores[s:e] < score_high)
        boost_mask = borderline & brand_match & dept_match & color_match
        scores[s:e][boost_mask] = score_high
        n_boosted += int(boost_mask.sum())

    print(f"  Borderline boost: {n_boosted:,} pairs promoted to {score_high}")
    return scores


def apply_rule_override(lookup, pair_indices, pair_scores,
                        img_threshold=0.82, price_threshold=0.88,
                        sim_low=0.70, sim_high=0.88, rule_score=0.85,
                        chunk_size=500_000):
    """
    Rule-based override for cross-geo pairs the hardneg model misses.

    Pairs with combined_sim in [sim_low, sim_high) that satisfy ALL of:
      brand_match, dept_match, price_ratio >= price_threshold, img_sim >= img_threshold
    get their score set to rule_score (above t=0.30 threshold).

    Only overrides pairs currently scored < 0.30 (doesn't downgrade good scores).
    """
    scores      = pair_scores.copy()
    n_overridden = 0

    for s in tqdm(range(0, len(pair_indices), chunk_size), desc="Rule override"):
        e  = min(s + chunk_size, len(pair_indices))
        ia = pair_indices[s:e, 0]
        ib = pair_indices[s:e, 1]

        t_sim = np.sum(lookup.text_emb[ia].astype(np.float32) *
                       lookup.text_emb[ib].astype(np.float32), axis=1)
        v_sim = np.sum(lookup.image_emb[ia].astype(np.float32) *
                       lookup.image_emb[ib].astype(np.float32), axis=1)
        combined = (t_sim + v_sim) * 0.5

        price_a = lookup.price[ia].flatten()
        price_b = lookup.price[ib].flatten()
        brand_a = lookup.brand_id[ia].flatten()
        brand_b = lookup.brand_id[ib].flatten()
        dept_a  = lookup.dept_id[ia].flatten()
        dept_b  = lookup.dept_id[ib].flatten()

        price_ratio = np.minimum(price_a, price_b) / (np.maximum(price_a, price_b) + 1e-5)
        brand_match = (brand_a == brand_b) & (brand_a != 0)
        dept_match  = (dept_a  == dept_b)  & (dept_a  != 0)

        rule_mask    = ((combined >= sim_low) & (combined < sim_high) &
                        brand_match & dept_match &
                        (price_ratio >= price_threshold) & (v_sim >= img_threshold))
        override_mask = rule_mask & (scores[s:e] < 0.30)
        scores[s:e][override_mask] = rule_score
        n_overridden += int(override_mask.sum())

    print(f"  Rule override: {n_overridden:,} pairs promoted to score {rule_score}")
    return scores


def score_pairs_model(lookup, pair_indices, pair_model, text_pca_emb, img_pca_emb,
                      chunk_size=200_000):
    """XGBoost pair classifier (84 features). Use only after retraining on FAISS negatives."""
    n     = len(pair_indices)
    proba = np.empty(n, dtype=np.float32)
    for s in tqdm(range(0, n, chunk_size), desc="Scoring pairs (XGB)"):
        e          = min(s + chunk_size, n)
        X          = compute_pair_features_pca(
            lookup, pair_indices[s:e, 0], pair_indices[s:e, 1],
            text_pca_emb, img_pca_emb,
        )
        proba[s:e] = pair_model.predict_proba(X.astype(np.float32))[:, 1]
    pcts = np.percentile(proba, [50, 75, 90, 95, 99, 99.5])
    print(f"Score stats (XGB): mean={proba.mean():.3f}")
    print(f"  Percentiles — p50:{pcts[0]:.3f}  p75:{pcts[1]:.3f}  "
          f"p90:{pcts[2]:.3f}  p95:{pcts[3]:.3f}  p99:{pcts[4]:.3f}  p99.5:{pcts[5]:.3f}")
    return proba


def score_pairs_medmatch(lookup, pair_indices, pair_model, medmatch_model,
                         text_pca_emb, img_pca_emb,
                         sim_low=0.70, sim_high=0.88,
                         med_gate=0.7,
                         chunk_size=200_000):
    """
    Two-model scoring with strict gating on the medium range.

    For combined_sim >= sim_high (0.88): use hardneg model directly.
    For sim_low <= combined_sim < sim_high (0.70-0.88):
      use medmatch model, but ONLY when its prob >= med_gate.  Below the gate,
      the score is 0 so the pair won't survive the Phase-2 clustering threshold.

    The med_gate is essential: medmatch operates at a ~1% positive prior, so its
    low-confidence predictions are dominated by false positives.  Empirically,
    only the top few percent of medmatch predictions are precise enough to
    promote into the cluster graph without over-merging.
    """
    n      = len(pair_indices)
    proba  = np.empty(n, dtype=np.float32)
    n_med_promoted = 0

    for s in tqdm(range(0, n, chunk_size), desc="Scoring pairs (XGB+medmatch)"):
        e    = min(s + chunk_size, n)
        ia   = pair_indices[s:e, 0]
        ib   = pair_indices[s:e, 1]
        t_sim = np.sum(lookup.text_emb[ia].astype(np.float32) *
                       lookup.text_emb[ib].astype(np.float32), axis=1)
        v_sim = np.sum(lookup.image_emb[ia].astype(np.float32) *
                       lookup.image_emb[ib].astype(np.float32), axis=1)
        combined = (t_sim + v_sim) * 0.5

        X = compute_pair_features_pca(lookup, ia, ib, text_pca_emb, img_pca_emb)
        X = X.astype(np.float32)

        high_mask = combined >= sim_high
        med_mask  = (combined >= sim_low) & ~high_mask

        result = np.zeros(len(ia), dtype=np.float32)
        if high_mask.any():
            p_hard = pair_model.predict_proba(X[high_mask])[:, 1]
            result[high_mask] = p_hard
        if med_mask.any():
            p_med = medmatch_model.predict_proba(X[med_mask])[:, 1]
            # Strict gate: only emit medmatch scores at high confidence.
            gated = np.where(p_med >= med_gate, p_med, 0.0)
            result[med_mask] = gated
            n_med_promoted += int((gated > 0).sum())
        proba[s:e] = result

    pcts = np.percentile(proba, [50, 75, 90, 95, 99])
    print(f"Score stats (XGB+medmatch, gate={med_gate}): mean={proba.mean():.3f}")
    print(f"  p50:{pcts[0]:.3f}  p75:{pcts[1]:.3f}  p90:{pcts[2]:.3f}  "
          f"p95:{pcts[3]:.3f}  p99:{pcts[4]:.3f}")
    print(f"  Medmatch promoted (p_med >= {med_gate}): {n_med_promoted:,} pairs")
    return proba


def score_pairs_crossencoder(lookup, pair_indices, idx_to_text,
                             model_path=CROSSENCODER_PATH,
                             presim_threshold=CROSSENCODER_PRESIM,
                             batch_size=128, max_length=256):
    """
    Re-score FAISS candidate pairs with a cross-encoder.

    Pre-filters to pairs where raw_sim >= presim_threshold (~0.80) so the
    cross-encoder only runs on ~1-3M pairs instead of 13M.  Pairs below the
    pre-filter keep a score of 0.0.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading cross-encoder from {model_path}  (device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    n               = len(pair_indices)
    ia, ib          = pair_indices[:, 0], pair_indices[:, 1]
    prefilter_mask  = np.zeros(n, dtype=bool)
    presim_chunk    = 500_000
    for s in tqdm(range(0, n, presim_chunk), desc="Pre-filter raw_sim"):
        e = min(s + presim_chunk, n)
        t_sim = np.sum(lookup.text_emb[ia[s:e]].astype(np.float32) *
                       lookup.text_emb[ib[s:e]].astype(np.float32), axis=1)
        v_sim = np.sum(lookup.image_emb[ia[s:e]].astype(np.float32) *
                       lookup.image_emb[ib[s:e]].astype(np.float32), axis=1)
        prefilter_mask[s:e] = (t_sim + v_sim) * 0.5 >= presim_threshold
    candidate_idxs = np.where(prefilter_mask)[0]
    print(f"Pre-filter (raw_sim >= {presim_threshold}): "
          f"{len(candidate_idxs):,} / {n:,} pairs ({prefilter_mask.mean()*100:.1f}%)")

    scores = np.zeros(n, dtype=np.float32)
    if len(candidate_idxs) == 0:
        return scores

    all_proba = []
    for start in tqdm(range(0, len(candidate_idxs), batch_size), desc="Cross-encoder"):
        end      = min(start + batch_size, len(candidate_idxs))
        pi_batch = candidate_idxs[start:end]
        texts_a  = [idx_to_text.get(int(pair_indices[pi, 0]), '') for pi in pi_batch]
        texts_b  = [idx_to_text.get(int(pair_indices[pi, 1]), '') for pi in pi_batch]
        enc = tokenizer(texts_a, texts_b, padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(input_ids=enc['input_ids'].to(device),
                           attention_mask=enc['attention_mask'].to(device)
                           ).logits.squeeze(-1)
            all_proba.append(torch.sigmoid(logits.float()).cpu().numpy())

    proba              = np.concatenate(all_proba).astype(np.float32)
    scores[candidate_idxs] = proba
    pcts = np.percentile(proba, [50, 75, 90, 95, 99])
    print(f"Cross-encoder scores (n={len(candidate_idxs):,}): mean={proba.mean():.3f}")
    print(f"  Percentiles — p50:{pcts[0]:.3f}  p75:{pcts[1]:.3f}  "
          f"p90:{pcts[2]:.3f}  p95:{pcts[3]:.3f}  p99:{pcts[4]:.3f}")
    above = [(t, (proba >= t).sum()) for t in [0.30, 0.40, 0.50, 0.60, 0.70]]
    print("  Above threshold — " +
          "  ".join(f">{t}:{cnt:,}" for t, cnt in above))
    return scores


# ── Department hard filter ────────────────────────────────────────────────────

def apply_dept_filter(lookup, pair_indices, pair_scores):
    """
    Zero out scores for pairs where both items have a known (non-zero) department
    that differs. Items from different departments cannot be the same product.
    Applied after cache load so no recompute needed.
    """
    ia     = pair_indices[:, 0]
    ib     = pair_indices[:, 1]
    dept_a = lookup.dept_id[ia].flatten()
    dept_b = lookup.dept_id[ib].flatten()
    mismatch = (dept_a != 0) & (dept_b != 0) & (dept_a != dept_b)
    filtered = mismatch.sum()
    out = pair_scores.copy()
    out[mismatch] = 0.0
    print(f"Dept filter: zeroed {filtered:,} cross-dept pairs "
          f"({filtered / len(pair_scores) * 100:.1f}%)")
    return out


# ── Graph clustering ──────────────────────────────────────────────────────────

def _split_oversized(G, nodes, max_size):
    """
    Split an oversized component using MST-based edge cutting with binary search.
    O(n log(n/max_size)) instead of O(n²).
    """
    sub          = G.subgraph(nodes).copy()
    mst          = nx.minimum_spanning_tree(sub, weight='weight')
    edges_sorted = sorted(mst.edges(data='weight'), key=lambda e: e[2])
    n            = len(nodes)

    def _build_and_check(k):
        g = nx.Graph()
        g.add_nodes_from(mst.nodes())
        for u, v, _ in edges_sorted[k:]:
            g.add_edge(u, v)
        return all(len(c) <= max_size for c in nx.connected_components(g))

    lo, hi = max(0, (n - 1) // max_size), len(edges_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if _build_and_check(mid):
            hi = mid
        else:
            lo = mid + 1

    g = nx.Graph()
    g.add_nodes_from(mst.nodes())
    for u, v, _ in edges_sorted[lo:]:
        g.add_edge(u, v)
    return [list(c) for c in nx.connected_components(g)]


def cluster_items_louvain(item_ids, pair_indices, pair_proba, threshold,
                          max_size=MAX_GROUP_SIZE):
    """
    Louvain community detection instead of connected components.
    Avoids transitivity errors (A~B, B~C doesn't force A~C into same group).
    """
    mask    = pair_proba >= threshold
    print(f"  Edges above threshold {threshold:.3f}: {mask.sum():,}")

    if mask.sum() == 0:
        return [[iid] for iid in item_ids]

    G = nx.Graph()
    G.add_nodes_from(range(len(item_ids)))
    for (a, b), w in zip(pair_indices[mask], pair_proba[mask]):
        G.add_edge(int(a), int(b), weight=float(w))

    communities = nx.community.louvain_communities(G, weight='weight', seed=42)
    print(f"  Louvain communities: {len(communities):,}")

    normal, oversized = [], []
    for comm in communities:
        nodes = list(comm)
        (oversized if len(nodes) > max_size else normal).append(nodes)

    groups = list(normal)
    if oversized:
        print(f"  Splitting {len(oversized)} oversized communities...")
        all_os  = {node for comp in oversized for node in comp}
        os_mask = np.isin(pair_indices[mask, 0], list(all_os)) | \
                  np.isin(pair_indices[mask, 1], list(all_os))
        sub_pairs  = pair_indices[mask][os_mask]
        sub_scores = pair_proba[mask][os_mask]
        G_os = nx.Graph()
        for (u, v), w in zip(sub_pairs, sub_scores):
            G_os.add_edge(int(u), int(v), weight=float(w))
        for comp in oversized:
            groups.extend(_split_oversized(G_os, comp, max_size))

    return [[item_ids[i] for i in grp] for grp in groups]


def cluster_items(item_ids, pair_indices, pair_proba, threshold,
                  max_size=MAX_GROUP_SIZE):
    n    = len(item_ids)
    mask = pair_proba >= threshold
    print(f"  Edges above threshold {threshold:.3f}: {mask.sum():,}")

    if mask.sum() == 0:
        print("  No edges — all items are singletons")
        return [[iid] for iid in item_ids]

    rows    = pair_indices[mask, 0].astype(np.int32)
    cols    = pair_indices[mask, 1].astype(np.int32)
    weights = pair_proba[mask]

    adj = csr_matrix(
        (np.concatenate([weights, weights]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n),
    )

    n_comps, labels = connected_components(adj, directed=False)
    print(f"  Connected components: {n_comps:,}")

    comp_map = defaultdict(list)
    for i, lbl in enumerate(labels):
        comp_map[lbl].append(i)

    normal, oversized = [], []
    for items in comp_map.values():
        (oversized if len(items) > max_size else normal).append(items)

    groups = list(normal)

    if oversized:
        print(f"  Splitting {len(oversized)} oversized components...")
        all_os  = {node for comp in oversized for node in comp}
        os_mask = np.isin(rows, list(all_os)) | np.isin(cols, list(all_os))
        G_os    = nx.Graph()
        for u, v, w in zip(rows[os_mask], cols[os_mask], weights[os_mask]):
            G_os.add_edge(int(u), int(v), weight=float(w))
        for comp in oversized:
            groups.extend(_split_oversized(G_os, comp, max_size))

    return [[item_ids[i] for i in grp] for grp in groups]


# ── Validation helpers ───────────────────────────────────────────────────────

class SubsetLookup:
    """Re-indexed EmbeddingLookup containing only a subset of items."""
    def __init__(self, lookup, item_ids):
        indices = np.array([lookup.id_to_idx[iid] for iid in item_ids
                            if iid in lookup.id_to_idx], dtype=np.int64)
        valid   = [iid for iid in item_ids if iid in lookup.id_to_idx]
        self.id_to_idx = {iid: i for i, iid in enumerate(valid)}
        self.text_emb  = lookup.text_emb[indices]
        self.image_emb = lookup.image_emb[indices]
        self.price     = lookup.price[indices]
        self.dept_id   = lookup.dept_id[indices]
        self.color_id  = lookup.color_id[indices]
        self.brand_id  = lookup.brand_id[indices]


def pairwise_f1(clusters, true_labels):
    """Compute pairwise F1 against ground-truth labels."""
    pred_pairs = set()
    for cluster in clusters:
        items = [str(x) for x in cluster]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pred_pairs.add((min(items[i], items[j]), max(items[i], items[j])))

    true_pairs  = set()
    lbl_to_items = defaultdict(list)
    for iid, lbl in true_labels.items():
        lbl_to_items[lbl].append(str(iid))
    for items in lbl_to_items.values():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                true_pairs.add((min(items[i], items[j]), max(items[i], items[j])))

    tp   = len(pred_pairs & true_pairs)
    fp   = len(pred_pairs - true_pairs)
    fn   = len(true_pairs - pred_pairs)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, prec, rec, tp, fp, fn


# ── Submission writer ─────────────────────────────────────────────────────────

def write_submission(groups, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for grp in groups:
            f.write(','.join(str(x) for x in grp) + '\n')
    sizes        = [len(g) for g in groups]
    n_singletons = sum(1 for s in sizes if s == 1)
    print(f"  {len(groups):,} groups | {n_singletons:,} singletons | "
          f"max={max(sizes)} | mean={np.mean(sizes):.2f} → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 clustering pipeline")
    parser.add_argument('--phase1', action='store_true',
                        help='Use phase_1 embeddings for leaderboard testing')
    parser.add_argument('--validate', action='store_true',
                        help='Evaluate on held-out 20%% of training labels with local pairwise F1')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Edge score threshold for single run (default: 0.95)')
    parser.add_argument('--top_k', type=int, default=TOP_K,
                        help=f'ANN neighbors per item (default: {TOP_K})')
    parser.add_argument('--sweep', action='store_true',
                        help='Write submissions for multiple thresholds')
    parser.add_argument('--text_weight', type=float, default=0.5,
                        help='Weight for text similarity, 0.0–1.0 (default 0.5 = equal). '
                             'Try 0.7 or 0.8 for cross-geo text-heavy matching.')
    parser.add_argument('--and_sim', action='store_true',
                        help='Score by min(text_sim, img_sim) — both must agree')
    parser.add_argument('--dept_filter', action='store_true',
                        help='Zero out pairs with known mismatching departments')
    parser.add_argument('--use_model', action='store_true',
                        help='Score pairs with XGBoost V6 (experimental)')
    parser.add_argument('--emb', default='ft',
                        choices=['ft', 'ft_v2', 'e5dino', 'v2'],
                        help='Embedding set: ft (default), e5dino (pretrained large), '
                             'v2 (fine-tuned mE5+DINOv2)')
    parser.add_argument('--crossencoder', action='store_true',
                        help='Re-score pairs with cross-encoder (requires finetuned_crossencoder/). '
                             'Pre-filters by raw_sim >= 0.80 then runs XLM-RoBERTa cross-encoder.')
    parser.add_argument('--louvain', action='store_true',
                        help='Use Louvain community detection instead of connected components')
    parser.add_argument('--ce_filter', action='store_true',
                        help='Use cached cross-encoder scores to zero out raw_sim edges where '
                             'CE score < 0.30 (removes false positives without rerunning inference)')
    parser.add_argument('--boost', action='store_true',
                        help='Promote borderline edges (score 0.15-0.30) with brand+dept+color '
                             'match to 0.30. Targets cross-geo pairs suppressed by text differences.')
    parser.add_argument('--rules', action='store_true',
                        help='Rule override: promote medium-sim pairs (0.70-0.88) with '
                             'brand+dept+price+image match to score 0.85. No retraining needed.')
    parser.add_argument('--medmatch', action='store_true',
                        help='Apply medmatch model (glami_xgb_medmatch.json) to medium-sim pairs '
                             '(0.70<=combined_sim<0.88) in addition to hardneg model. '
                             'Requires --use_model.')
    parser.add_argument('--anchor', action='store_true',
                        help='Add training-anchor pairs: phase items bridged via shared training label')
    parser.add_argument('--anchor_sim', type=float, default=0.92,
                        help='Min combined_sim for phase→training anchor link (default: 0.92)')
    args = parser.parse_args()

    true_labels   = None   # only set in --validate mode
    val_label_set = set()  # used by --anchor to exclude val labels from anchor pool

    if args.validate:
        print("Loading training labels for validation split...")
        df_tr = pd.read_csv('data/items_train.csv', dtype={'itemId': str})
        # Stable 20% label hold-out: labels whose integer value mod 5 == 0
        val_label_set = {int(l) for l in df_tr['label'].unique() if int(l) % 5 == 0}
        val_df        = df_tr[df_tr['label'].isin(val_label_set)]
        true_labels   = dict(zip(val_df['itemId'], val_df['label'].astype(int)))
        print(f"  Val: {len(val_label_set):,} labels, {len(val_df):,} items")
        h5_path = EMB_H5[args.emb]['train']
        tag     = f"val_{args.emb}"
    else:
        split   = 'phase_1' if args.phase1 else 'phase_2'
        h5_path = EMB_H5[args.emb][split]
        tag     = f"{'phase1' if args.phase1 else 'phase2'}_{args.emb}"

    if not os.path.exists(h5_path):
        print(f"ERROR: embeddings not found at '{h5_path}'")
        if not args.validate and not args.phase1:
            print("Run: python extract_embeddings_v2.py")
        return

    t0 = time.time()

    full_lookup = EmbeddingLookup(h5_path)

    if args.validate:
        lookup = SubsetLookup(full_lookup, list(true_labels.keys()))
        # keep only items that exist in the H5
        true_labels = {iid: lbl for iid, lbl in true_labels.items()
                       if iid in lookup.id_to_idx}
        del full_lookup
        print(f"  {len(lookup.id_to_idx):,} val items in H5")
    else:
        lookup = full_lookup

    n        = len(lookup.id_to_idx)
    item_ids = [''] * n
    for uid, idx in lookup.id_to_idx.items():
        item_ids[idx] = uid
    print(f"Loaded {n:,} items from {h5_path}")

    # PCA + model only needed for --use_model; text lookup for --crossencoder
    text_pca_emb = img_pca_emb = pair_model = medmatch_model = idx_to_text = None
    if args.use_model:
        print("\n[1/4] PCA")
        train_lookup = EmbeddingLookup(TRAIN_H5_BY_EMB[args.emb])
        text_pca, img_pca = fit_pca(train_lookup)
        del train_lookup
        text_pca_emb = apply_pca(lookup.text_emb, text_pca)
        img_pca_emb  = apply_pca(lookup.image_emb, img_pca)
        pair_model = xgb.XGBClassifier(device='cpu')
        pair_model.load_model(PAIR_MODEL_PATH)
        print(f"Pair model loaded from {PAIR_MODEL_PATH}")
        if args.medmatch:
            medmatch_path = 'glami_xgb_medmatch.json'
            if not os.path.exists(medmatch_path):
                print(f"ERROR: medmatch model not found at {medmatch_path}. "
                      f"Run: python retrain_pair_medmatch.py")
                return
            medmatch_model = xgb.XGBClassifier(device='cpu')
            medmatch_model.load_model(medmatch_path)
            print(f"Medmatch model loaded from {medmatch_path}")
    elif args.crossencoder:
        print("\n[1/4] Building item text lookup for cross-encoder...")
        df           = pd.read_csv(ITEM_CSVS[split])
        df['itemId'] = df['itemId'].astype(str)
        id_to_row    = df.set_index('itemId')
        idx_to_id    = {v: k for k, v in lookup.id_to_idx.items()}
        idx_to_text  = {}
        for idx, iid in idx_to_id.items():
            if iid not in id_to_row.index:
                continue
            row   = id_to_row.loc[iid]
            title = str(row.get('title', '') or '')
            desc  = str(row.get('description', '') or '')[:DESC_CHARS]
            idx_to_text[idx] = f"{title} {desc}".strip()
        print(f"  {len(idx_to_text):,} items with text")
    else:
        print("\n[1/4] Skipping PCA/model")

    # ── Determine score mode (used for cache filename) ────────────────────────
    if args.crossencoder:
        score_mode = 'crossencoder'
    elif args.use_model:
        score_mode = 'model_med' if args.medmatch else 'model'
    elif args.and_sim:
        score_mode = 'andsim'
    else:
        tw = round(args.text_weight, 2)
        score_mode = 'rawsim' if tw == 0.5 else f'textsim{int(tw * 100)}'

    pairs_cache = f'cache_{tag}_k{args.top_k}_pairs.npz'
    score_cache = f'cache_{tag}_k{args.top_k}_{score_mode}.npz'
    # backward compat: old rawsim cache stored pairs+scores together
    legacy_rawsim = f'cache_{tag}_k{args.top_k}_rawsim.npz'

    # ── Load or compute pairs + scores ───────────────────────────────────────
    if os.path.exists(score_cache):
        print(f"\n[2-3/4] Loading cached pairs+scores from {score_cache}")
        cache        = np.load(score_cache)
        pair_indices = cache['pair_indices']
        pair_scores  = cache['pair_scores']
        print(f"  {len(pair_indices):,} pairs loaded")
    else:
        # Load pair indices from whichever cache exists
        if os.path.exists(pairs_cache):
            print(f"\n[2/4] Loading cached pairs from {pairs_cache}")
            pair_indices = np.load(pairs_cache)['pair_indices']
            print(f"  {len(pair_indices):,} pairs loaded")
        elif os.path.exists(legacy_rawsim):
            print(f"\n[2/4] Loading cached pairs from {legacy_rawsim}")
            pair_indices = np.load(legacy_rawsim)['pair_indices']
            print(f"  {len(pair_indices):,} pairs loaded")
        else:
            print(f"\n[2/4] Candidate generation (top_k={args.top_k})")
            pair_indices = get_candidate_pairs(lookup, top_k=args.top_k)
            np.savez_compressed(pairs_cache, pair_indices=pair_indices)
            print(f"  Pairs cached → {pairs_cache}")

        print("\n[3/4] Pair scoring")
        if args.crossencoder:
            pair_scores = score_pairs_crossencoder(lookup, pair_indices, idx_to_text)
        elif args.use_model and args.medmatch:
            pair_scores = score_pairs_medmatch(lookup, pair_indices, pair_model,
                                               medmatch_model, text_pca_emb, img_pca_emb)
        elif args.use_model:
            pair_scores = score_pairs_model(lookup, pair_indices, pair_model,
                                            text_pca_emb, img_pca_emb)
        elif args.and_sim:
            pair_scores = score_pairs_and_sim(lookup, pair_indices)
        else:
            pair_scores = score_pairs_weighted(lookup, pair_indices, args.text_weight)

        np.savez_compressed(score_cache, pair_indices=pair_indices, pair_scores=pair_scores)
        print(f"  Cached → {score_cache}")

    # ── Department hard filter (post-cache, no recompute) ────────────────────
    if args.dept_filter:
        pair_scores = apply_dept_filter(lookup, pair_indices, pair_scores)

    # ── Borderline boost: metadata-confirmed cross-geo pairs ─────────────────
    if args.boost:
        pair_scores = apply_borderline_boost(lookup, pair_indices, pair_scores)

    # ── Rule-based override for cross-geo medium-sim pairs ───────────────────
    if args.rules:
        pair_scores = apply_rule_override(lookup, pair_indices, pair_scores)

    # ── Training-anchor bridge pairs ─────────────────────────────────────────
    if args.anchor:
        df_tr_full = pd.read_csv('data/items_train.csv', dtype={'itemId': str})
        if args.validate and val_label_set:
            df_anchor_pool = df_tr_full[~df_tr_full['label'].isin(val_label_set)]
            print(f"Anchor pool: {len(df_anchor_pool):,} items (val labels excluded to avoid leakage)")
            print("  NOTE: validate+anchor will produce ~0 pairs since val labels have no "
                  "training anchors. Run --phase1 --anchor for real evaluation.")
        else:
            df_anchor_pool = df_tr_full

        anchor_cache = (f'cache_{tag}_k{args.top_k}'
                        f'_anchors_sim{int(args.anchor_sim * 100)}.npz')
        if os.path.exists(anchor_cache):
            print(f"Loading cached anchor pairs from {anchor_cache}")
            ac         = np.load(anchor_cache)
            anc_pairs  = ac['pair_indices']
            anc_scores = ac['pair_scores']
            print(f"  {len(anc_pairs):,} anchor pairs loaded")
        else:
            anc_pairs, anc_scores = get_anchor_pairs(
                lookup, TRAIN_H5_BY_EMB[args.emb], df_anchor_pool,
                sim_threshold=args.anchor_sim,
            )
            np.savez_compressed(anchor_cache, pair_indices=anc_pairs, pair_scores=anc_scores)
            print(f"  Cached → {anchor_cache}")

        if len(anc_pairs) > 0:
            # Filter anchor pairs: require brand AND dept to match (when both known)
            ia, ib = anc_pairs[:, 0].astype(np.int64), anc_pairs[:, 1].astype(np.int64)
            brand_a = lookup.brand_id[ia].flatten()
            brand_b = lookup.brand_id[ib].flatten()
            dept_a  = lookup.dept_id[ia].flatten()
            dept_b  = lookup.dept_id[ib].flatten()
            brand_ok = ((brand_a == 0) | (brand_b == 0) | (brand_a == brand_b))
            dept_ok  = ((dept_a  == 0) | (dept_b  == 0) | (dept_a  == dept_b))
            meta_mask = brand_ok & dept_ok
            anc_pairs  = anc_pairs[meta_mask]
            anc_scores = anc_scores[meta_mask]
            print(f"  After brand+dept filter: {meta_mask.sum():,} / {len(meta_mask):,} anchor pairs kept")

            print(f"Merging {len(anc_pairs):,} anchor pairs into {len(pair_indices):,} FAISS pairs")
            all_pairs  = np.vstack([pair_indices, anc_pairs])
            all_scores = np.concatenate([pair_scores, anc_scores])
            df_p       = pd.DataFrame({'a': all_pairs[:, 0].astype(np.int64),
                                       'b': all_pairs[:, 1].astype(np.int64),
                                       's': all_scores})
            df_p       = df_p.groupby(['a', 'b'], sort=False)['s'].max().reset_index()
            pair_indices = df_p[['a', 'b']].values.astype(np.int32)
            pair_scores  = df_p['s'].values.astype(np.float32)
            print(f"  Combined: {len(pair_indices):,} pairs")

    # ── Cross-encoder false-positive filter (post-cache) ─────────────────────
    if args.ce_filter and not args.crossencoder:
        ce_cache_path = f'cache_{tag}_k{args.top_k}_crossencoder.npz'
        if os.path.exists(ce_cache_path):
            ce_scores = np.load(ce_cache_path)['pair_scores']
            rejected  = (ce_scores > 0) & (ce_scores < 0.30)
            pair_scores = pair_scores.copy()
            pair_scores[rejected] = 0.0
            print(f"CE filter: zeroed {rejected.sum():,} pairs that CE rejected "
                  f"({rejected.sum() / len(pair_scores) * 100:.1f}%)")
        else:
            print(f"WARNING: CE cache not found at {ce_cache_path}, skipping --ce_filter")

    # ── Sweep thresholds ──────────────────────────────────────────────────────
    if args.crossencoder:
        sweep_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    elif args.use_model:
        sweep_thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    else:
        sweep_thresholds = [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.97]

    dept_tag = '_dept' if args.dept_filter else ''
    ce_tag   = '_cef'  if args.ce_filter  else ''
    lv_tag   = '_lv'   if args.louvain    else ''
    thresholds = sweep_thresholds if args.sweep else [args.threshold]

    cluster_fn = cluster_items_louvain if args.louvain else cluster_items

    print("\n[4/4] Clustering + evaluation")
    for t in thresholds:
        t      = round(t, 3)
        groups = cluster_fn(item_ids, pair_indices, pair_scores, threshold=t)
        if true_labels:
            f1, prec, rec, tp, fp, fn = pairwise_f1(groups, true_labels)
            print(f"  *** t={t}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}"
                  f"  TP={tp:,} FP={fp:,} FN={fn:,} ***")
        else:
            outpath = (f'submissions/submission_{tag}_{score_mode}'
                       f'{dept_tag}{ce_tag}{lv_tag}_t{t}.csv')
            write_submission(groups, outpath)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()