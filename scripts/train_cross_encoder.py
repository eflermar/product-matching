"""
Train a cross-encoder for product pair re-ranking.

Architecture:
  Input:  [CLS] title_A description_A [SEP] title_B description_B [SEP]
  Model:  finetuned_text_model (XLM-RoBERTa-base) + linear classification head
  Output: P(same product)

The cross-encoder reads both items simultaneously so it can catch fine-grained
differences (colour words, model numbers, sizes) that cosine similarity misses.

Training data: same FAISS hard-negative pairs as V6, but text input instead of
84 hand-crafted features — the model sees the raw titles and descriptions.

Usage:
  python train_cross_encoder.py
  # produces: finetuned_crossencoder/

Then:
  python phase2_pipeline.py --phase1 --crossencoder --sweep
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from itertools import combinations
import faiss
from tqdm import tqdm

from features import EmbeddingLookup

np.random.seed(42)
torch.manual_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_H5          = 'glami_embeddings_train_ft.h5'
TRAIN_CSV         = 'data/items_train.csv'
BASE_MODEL        = 'finetuned_text_model'   # start from domain-adapted weights
MODEL_OUT         = 'finetuned_crossencoder_v2'

MAX_LENGTH        = 256    # ~128 tokens per item
BATCH_SIZE        = 16
GRAD_ACCUM        = 4      # effective batch size = 64
LR                = 2e-5
EPOCHS            = 2
WARMUP_RATIO      = 0.06

MAX_POS_PER_LABEL = 3
TOP_K_NEG         = 100    # search wider to find high-sim negatives
NLIST             = 500
NPROBE            = 25
DESC_CHARS        = 150    # how many description chars to include
NEG_MIN_SIM       = 0.88   # only accept negatives with raw_sim >= this


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairTextDataset(Dataset):
    def __init__(self, pairs, labels, idx_to_text, tokenizer):
        self.pairs       = pairs
        self.labels      = labels
        self.idx_to_text = idx_to_text
        self.tokenizer   = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b   = self.pairs[idx]
        text_a = self.idx_to_text.get(int(a), '')
        text_b = self.idx_to_text.get(int(b), '')
        enc    = self.tokenizer(
            text_a, text_b,
            padding='max_length', truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Pair mining (same logic as retrain_pair_faiss.py) ────────────────────────

def mine_pairs(df_train, lookup):
    item_to_label = dict(zip(df_train['itemId'].astype(str), df_train['label']))
    n             = len(lookup.text_emb)

    print("Mining positive pairs...")
    grouped   = df_train.groupby('label')['itemId'].apply(list)
    positives = []
    for label, items in grouped.items():
        items = [str(x) for x in items]
        idxs  = [lookup.id_to_idx[i] for i in items if i in lookup.id_to_idx]
        if len(idxs) < 2:
            continue
        combos = list(combinations(idxs, 2))
        if len(combos) > MAX_POS_PER_LABEL:
            chosen = np.random.choice(len(combos), MAX_POS_PER_LABEL, replace=False)
            combos = [combos[i] for i in chosen]
        positives.extend(combos)
    print(f"  Positive pairs: {len(positives):,}")

    idx_to_label = {}
    for iid, lbl in item_to_label.items():
        ix = lookup.id_to_idx.get(iid)
        if ix is not None:
            idx_to_label[ix] = lbl

    print("Building FAISS index on training embeddings...")
    text_emb   = lookup.text_emb.astype(np.float32)
    quantizer  = faiss.IndexFlatIP(text_emb.shape[1])
    index      = faiss.IndexIVFFlat(quantizer, text_emb.shape[1], NLIST,
                                    faiss.METRIC_INNER_PRODUCT)
    index.train(text_emb)
    index.add(text_emb)
    index.nprobe = NPROBE

    print(f"Mining FAISS hard negatives (top_k={TOP_K_NEG})...")
    hard_negs      = []
    neg_seen       = set()
    n_needed       = len(positives)
    anchor_indices = list(idx_to_label.keys())
    np.random.shuffle(anchor_indices)

    for batch_start in tqdm(range(0, len(anchor_indices), 5_000), desc="Hard neg mining"):
        if len(hard_negs) >= n_needed:
            break
        batch_end = min(batch_start + 5_000, len(anchor_indices))
        batch_idx = anchor_indices[batch_start:batch_end]
        _, neighbors = index.search(text_emb[batch_idx], TOP_K_NEG + 1)

        for anchor_ix, nbrs in zip(batch_idx, neighbors):
            anchor_lbl = idx_to_label.get(anchor_ix)
            if anchor_lbl is None:
                continue
            for nbr in nbrs:
                if nbr < 0 or nbr >= n or nbr == anchor_ix:
                    continue
                if idx_to_label.get(nbr) == anchor_lbl:
                    continue
                # Only accept negatives that are hard in the inference range
                sim = float(np.dot(text_emb[anchor_ix], text_emb[nbr]))
                if sim < NEG_MIN_SIM:
                    continue
                pair = (min(anchor_ix, nbr), max(anchor_ix, nbr))
                if pair not in neg_seen:
                    neg_seen.add(pair)
                    hard_negs.append(pair)
                    break

    hard_negs = hard_negs[:n_needed]
    print(f"  Hard negative pairs: {len(hard_negs):,}")

    all_pairs  = positives + hard_negs
    all_labels = [1] * len(positives) + [0] * len(hard_negs)
    order      = np.random.permutation(len(all_pairs))
    return [all_pairs[i] for i in order], [all_labels[i] for i in order]


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, loader, val_loader, device, n_steps):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps * WARMUP_RATIO),
        num_training_steps=n_steps,
    )
    loss_fn   = nn.BCEWithLogitsLoss()

    model.train()
    optimizer.zero_grad()
    step = accum_loss = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"  Train")):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
            loss   = loss_fn(logits, label) / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

        # Validation at end of each epoch
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Val"):
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                out  = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                val_logits.append(out.cpu())
                val_labels.append(batch['label'])

        val_logits = torch.cat(val_logits).numpy()
        val_labels = torch.cat(val_labels).numpy().astype(int)
        val_proba  = torch.sigmoid(torch.tensor(val_logits)).numpy()

        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.3, 0.8, 0.05):
            f1 = f1_score(val_labels, (val_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        print(f"  Val F1={best_f1:.4f} at t={best_t:.2f}")
        print(classification_report(val_labels, (val_proba >= best_t).astype(int),
                                    zero_division=0))
        model.train()

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading training data...")
    df_train           = pd.read_csv(TRAIN_CSV)
    df_train['itemId'] = df_train['itemId'].astype(str)
    print(f"  {len(df_train):,} items, {df_train['label'].nunique():,} labels")

    lookup = EmbeddingLookup(TRAIN_H5)

    # Build idx → text lookup
    print("Building item text lookup...")
    idx_to_id   = {v: k for k, v in lookup.id_to_idx.items()}
    id_to_row   = df_train.set_index('itemId')
    idx_to_text = {}
    for idx, iid in idx_to_id.items():
        if iid not in id_to_row.index:
            continue
        row   = id_to_row.loc[iid]
        title = str(row.get('title', '') or '')
        desc  = str(row.get('description', '') or '')[:DESC_CHARS]
        idx_to_text[idx] = f"{title} {desc}".strip()
    print(f"  {len(idx_to_text):,} items with text")

    # Mine pairs
    pairs, labels = mine_pairs(df_train, lookup)

    # Train / val split
    tr_pairs, va_pairs, tr_labels, va_labels = train_test_split(
        pairs, labels, test_size=0.1, random_state=42, stratify=labels,
    )
    print(f"\nTrain: {len(tr_pairs):,}  Val: {len(va_pairs):,}  "
          f"Pos rate: {np.mean(tr_labels):.3f}")

    # Load model + tokenizer
    print(f"\nLoading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1, ignore_mismatched_sizes=True,
    ).to(device)

    tr_dataset = PairTextDataset(tr_pairs, tr_labels, idx_to_text, tokenizer)
    va_dataset = PairTextDataset(va_pairs, va_labels, idx_to_text, tokenizer)
    tr_loader  = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True)
    va_loader  = DataLoader(va_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    n_steps = (len(tr_loader) // GRAD_ACCUM) * EPOCHS
    print(f"Training: {n_steps:,} optimizer steps over {EPOCHS} epochs")

    model = train(model, tr_loader, va_loader, device, n_steps)

    os.makedirs(MODEL_OUT, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    print(f"\nSaved → {MODEL_OUT}/")
    print("\nNext:")
    print("  python phase2_pipeline.py --phase1 --crossencoder --sweep")


if __name__ == '__main__':
    main()
