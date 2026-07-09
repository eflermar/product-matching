"""
Diagnostic: where is F1 being lost on the Phase-2 validation split?

Computes:
  [1] FAISS recall  — fraction of true-positive pairs that survive candidate generation.
  [2] Recall after combined_sim >= {0.88, 0.80, 0.70, 0.60} filter.
  [3] Positive rate per similarity bin — tells you how class-balanced each region is,
      and whether widening the XGBoost training cut would help or hurt.

Run AFTER any --validate invocation that built the pair cache, e.g.:
    python phase2_pipeline.py --validate --threshold 0.95

Usage:
    python diagnose_recall.py
    python diagnose_recall.py --emb ft --top_k 50
"""
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from features import EmbeddingLookup
from phase2_pipeline import SubsetLookup, EMB_H5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb',   default='ft', choices=list(EMB_H5.keys()))
    ap.add_argument('--top_k', type=int, default=50)
    args = ap.parse_args()

    tag           = f'val_{args.emb}'
    pairs_cache   = f'cache_{tag}_k{args.top_k}_pairs.npz'
    rawsim_cache  = f'cache_{tag}_k{args.top_k}_rawsim.npz'

    # ── Reproduce the validation split exactly as phase2_pipeline.main() does ──
    print("Loading training labels for validation split...")
    df_tr         = pd.read_csv('data/items_train.csv', dtype={'itemId': str})
    val_label_set = {int(l) for l in df_tr['label'].unique() if int(l) % 5 == 0}
    val_df        = df_tr[df_tr['label'].isin(val_label_set)]
    true_labels   = dict(zip(val_df['itemId'], val_df['label'].astype(int)))
    print(f"  Val: {len(val_label_set):,} labels, {len(val_df):,} items")

    # ── Load embeddings (val subset only, same indexing as pipeline) ──────────
    h5_path     = EMB_H5[args.emb]['train']
    full_lookup = EmbeddingLookup(h5_path)
    lookup      = SubsetLookup(full_lookup, list(true_labels.keys()))
    true_labels = {iid: lbl for iid, lbl in true_labels.items()
                   if iid in lookup.id_to_idx}
    del full_lookup
    print(f"  {len(lookup.id_to_idx):,} val items in H5")

    # ── Locate the pair cache (try both possible filenames) ───────────────────
    cache_path = None
    for p in [pairs_cache, rawsim_cache]:
        if os.path.exists(p):
            cache_path = p
            break
    if cache_path is None:
        print(f"\nERROR: no pair cache found. Looked for:")
        print(f"  {pairs_cache}")
        print(f"  {rawsim_cache}")
        print(f"\nRun first:  python phase2_pipeline.py --validate --threshold 0.95")
        raise SystemExit(1)
    print(f"Loading candidates from {cache_path}")
    pair_indices = np.load(cache_path)['pair_indices']
    print(f"  {len(pair_indices):,} candidate pairs")

    # ── Build the ground-truth positive-pair set in val-index space ───────────
    lbl_to_idx = defaultdict(list)
    for iid, lbl in true_labels.items():
        lbl_to_idx[lbl].append(lookup.id_to_idx[iid])
    true_pairs = set()
    for items in lbl_to_idx.values():
        items.sort()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                true_pairs.add((items[i], items[j]))
    n_true = len(true_pairs)
    print(f"True positive pairs (val): {n_true:,}")
    if n_true == 0:
        print("No positive pairs in val split — check label hold-out logic.")
        raise SystemExit(1)

    # ── [1] FAISS recall ─────────────────────────────────────────────────────
    cand_set = set(map(tuple, pair_indices.tolist()))
    hit      = len(true_pairs & cand_set)
    print(f"\n[1] FAISS recall: {hit / n_true:.4f}  ({hit:,} / {n_true:,})")
    print(f"    Lost at retrieval: {n_true - hit:,} pairs "
          f"({(n_true - hit) / n_true * 100:.2f}%)")

    # ── Compute combined_sim for every candidate ──────────────────────────────
    ia, ib = pair_indices[:, 0], pair_indices[:, 1]
    s      = np.empty(len(pair_indices), dtype=np.float32)
    chunk  = 500_000
    for k in range(0, len(pair_indices), chunk):
        e    = min(k + chunk, len(pair_indices))
        tsim = np.sum(lookup.text_emb[ia[k:e]].astype(np.float32) *
                      lookup.text_emb[ib[k:e]].astype(np.float32), axis=1)
        vsim = np.sum(lookup.image_emb[ia[k:e]].astype(np.float32) *
                      lookup.image_emb[ib[k:e]].astype(np.float32), axis=1)
        s[k:e] = 0.5 * tsim + 0.5 * vsim

    # ── [2] Recall after similarity floor ────────────────────────────────────
    print("\n[2] Recall after combined_sim >= threshold:")
    pair_tuples = [tuple(p) for p in pair_indices.tolist()]
    for thr in [0.88, 0.80, 0.70, 0.60]:
        mask = s >= thr
        kept = {pair_tuples[i] for i in np.where(mask)[0]}
        hit2 = len(true_pairs & kept)
        print(f"    s >= {thr}: recall = {hit2 / n_true:.4f}  "
              f"({hit2:,} / {n_true:,} TP kept; {mask.sum():,} candidates total)")

    # ── [3] Positive rate per similarity bin ─────────────────────────────────
    print("\n[3] Positive rate by combined-sim bin (where the model has to discriminate):")
    print(f"    {'bin':<18}{'n_cand':>12}{'n_TP':>10}{'pos_rate':>12}")
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.88, 0.90, 0.92, 0.95, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        bin_mask = (s >= lo) & (s < hi)
        if not bin_mask.any():
            continue
        bin_pairs = {pair_tuples[i] for i in np.where(bin_mask)[0]}
        bin_tp    = len(bin_pairs & true_pairs)
        rate      = bin_tp / max(len(bin_pairs), 1)
        print(f"    [{lo:.2f}, {hi:.2f}){'':<6}"
              f"{bin_mask.sum():>12,}{bin_tp:>10,}{rate * 100:>11.2f}%")

    # ── Summary line you can paste back to me ────────────────────────────────
    r88 = len(true_pairs & {pair_tuples[i] for i in np.where(s >= 0.88)[0]}) / n_true
    r70 = len(true_pairs & {pair_tuples[i] for i in np.where(s >= 0.70)[0]}) / n_true
    print(f"\nSummary:  FAISS_recall={hit / n_true:.4f}  "
          f"recall_at_s>=0.88={r88:.4f}  recall_at_s>=0.70={r70:.4f}")


if __name__ == '__main__':
    main()
