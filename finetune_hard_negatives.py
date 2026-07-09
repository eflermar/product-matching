"""
Continue fine-tuning the text encoder with FAISS hard negatives.

The original contrastive training used random in-batch negatives, which are
easy to separate. This script mines pairs where raw_sim >= NEG_MIN_SIM but
labels differ — exactly the regime where phase_2 clustering fails — and
trains the model to push those apart.

InfoNCE loss: for each anchor, positive = same-label item, negatives = all
other positives in batch + all hard negatives in batch.  Hard negatives are
2x more informative than random negatives because the model must actually
read the text to distinguish them.

Produces: finetuned_text_model_v2/

Usage:
  python finetune_hard_negatives.py
  python extract_embeddings_v2.py --text_model finetuned_text_model_v2
  python phase2_pipeline.py --phase1 --emb ft_v2 --sweep
"""
import os
import math
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from features import EmbeddingLookup

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_H5   = 'glami_embeddings_train_ft.h5'
TRAIN_CSV  = 'data/items_train.csv'
BASE_MODEL = 'finetuned_text_model'
MODEL_OUT  = 'finetuned_text_model_v2'

LR          = 2e-6    # very small — fine-tuning, not from scratch
BATCH_SIZE  = 32
EPOCHS      = 1
TEMPERATURE = 0.05
MAX_LEN     = 128
DESC_CHARS  = 150
EMBED_DIM   = 256     # must match existing projection head

NEG_MIN_SIM       = 0.88   # hard negatives must be at least this similar
NEG_MIN_SIM_FLOOR = 0.82   # fall back to this if not enough pairs found
TOP_K_NEG         = 100
NLIST             = 500
NPROBE            = 25
MAX_POS_PER_LABEL = 3
MIN_TRIPLETS      = 5_000  # abort if fewer than this found


# ── Model (same architecture as finetune_contrastive.py) ─────────────────────

class TextEncoder(nn.Module):
    def __init__(self, model_path, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path)
        self.proj     = nn.Sequential(
            nn.Linear(768, 768), nn.GELU(), nn.Linear(768, embed_dim),
        )
        proj_path = os.path.join(model_path, 'proj_head.pt')
        if os.path.exists(proj_path):
            self.proj.load_state_dict(
                torch.load(proj_path, map_location='cpu', weights_only=True)
            )
            print(f"  Loaded projection head from {proj_path}")

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(self.proj(cls), p=2, dim=1)


# ── Loss ──────────────────────────────────────────────────────────────────────

def info_nce_with_hard_neg(a_emb, p_emb, n_emb, temperature=TEMPERATURE):
    """
    InfoNCE where hard negatives are EXPLICITLY in the denominator.

    For each anchor i the positive is p_emb[i].
    Negatives: all p_emb[j≠i]  (random in-batch)
             + all n_emb[j]     (FAISS hard negatives from different labels)

    The second term is the key improvement: the model must learn to push apart
    pairs that look very similar (raw_sim >= 0.88) but are different products.
    """
    B        = a_emb.shape[0]
    all_keys = torch.cat([p_emb, n_emb], dim=0)   # (2B, D)
    logits   = a_emb @ all_keys.T / temperature    # (B, 2B)
    labels   = torch.arange(B, device=a_emb.device)
    return F.cross_entropy(logits, labels)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TripletTextDataset(Dataset):
    def __init__(self, triplets, idx_to_text, tokenizer):
        self.triplets    = triplets
        self.idx_to_text = idx_to_text
        self.tokenizer   = tokenizer

    def __len__(self):
        return len(self.triplets)

    def _enc(self, text):
        t = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=MAX_LEN, return_tensors='pt',
        )
        return t['input_ids'].squeeze(0), t['attention_mask'].squeeze(0)

    def __getitem__(self, idx):
        a_ix, p_ix, n_ix = self.triplets[idx]
        a_ids, a_mask = self._enc(self.idx_to_text.get(a_ix, ''))
        p_ids, p_mask = self._enc(self.idx_to_text.get(p_ix, ''))
        n_ids, n_mask = self._enc(self.idx_to_text.get(n_ix, ''))
        return (a_ids, a_mask, p_ids, p_mask, n_ids, n_mask)


# ── Hard negative mining ──────────────────────────────────────────────────────

def mine_triplets(df_train, lookup, neg_min_sim):
    item_to_label = dict(zip(df_train['itemId'].astype(str), df_train['label']))
    idx_to_label  = {}
    for iid, lbl in item_to_label.items():
        ix = lookup.id_to_idx.get(iid)
        if ix is not None:
            idx_to_label[ix] = lbl

    # Positive pairs
    grouped   = df_train.groupby('label')['itemId'].apply(list)
    positives = []
    for label, items in grouped.items():
        items = [str(x) for x in items]
        idxs  = [lookup.id_to_idx[i] for i in items if i in lookup.id_to_idx]
        if len(idxs) < 2:
            continue
        combos = list(combinations(idxs, 2))
        if len(combos) > MAX_POS_PER_LABEL:
            sel    = np.random.choice(len(combos), MAX_POS_PER_LABEL, replace=False)
            combos = [combos[i] for i in sel]
        positives.extend(combos)
    print(f"  Positive pairs: {len(positives):,}")

    # FAISS index
    text_emb  = lookup.text_emb.astype(np.float32)
    n         = len(text_emb)
    quantizer = faiss.IndexFlatIP(text_emb.shape[1])
    index     = faiss.IndexIVFFlat(quantizer, text_emb.shape[1], NLIST,
                                   faiss.METRIC_INNER_PRODUCT)
    index.train(text_emb)
    index.add(text_emb)
    index.nprobe = NPROBE

    # Mine one hard negative per unique anchor
    unique_anchors = list({a for a, _ in positives})
    np.random.shuffle(unique_anchors)
    anchor_to_neg  = {}

    for batch_start in tqdm(range(0, len(unique_anchors), 5_000), desc="Hard neg mining"):
        batch_end = min(batch_start + 5_000, len(unique_anchors))
        batch_idx = unique_anchors[batch_start:batch_end]
        _, nbrs   = index.search(text_emb[batch_idx], TOP_K_NEG + 1)

        for anchor_ix, row in zip(batch_idx, nbrs):
            if anchor_ix in anchor_to_neg:
                continue
            anchor_lbl = idx_to_label.get(anchor_ix)
            if anchor_lbl is None:
                continue
            for nbr in row:
                if nbr < 0 or nbr >= n or nbr == anchor_ix:
                    continue
                if idx_to_label.get(nbr) == anchor_lbl:
                    continue
                if float(np.dot(text_emb[anchor_ix], text_emb[nbr])) >= neg_min_sim:
                    anchor_to_neg[anchor_ix] = nbr
                    break

    triplets = [
        (a, p, anchor_to_neg[a])
        for a, p in positives
        if a in anchor_to_neg
    ]
    coverage = len(triplets) / max(len(positives), 1) * 100
    print(f"  Triplets: {len(triplets):,}  (coverage {coverage:.1f}% of positives)")
    return triplets


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading training data...")
    df_train           = pd.read_csv(TRAIN_CSV)
    df_train['itemId'] = df_train['itemId'].astype(str)
    print(f"  {len(df_train):,} items, {df_train['label'].nunique():,} labels")

    lookup = EmbeddingLookup(TRAIN_H5)

    # Build idx → text
    idx_to_id   = {v: k for k, v in lookup.id_to_idx.items()}
    id_to_row   = df_train.set_index('itemId')
    idx_to_text = {}
    for idx, iid in idx_to_id.items():
        if iid not in id_to_row.index:
            continue
        row              = id_to_row.loc[iid]
        title            = str(row.get('title', '') or '')
        desc             = str(row.get('description', '') or '')[:DESC_CHARS]
        idx_to_text[idx] = f"{title} {desc}".strip()
    print(f"  {len(idx_to_text):,} items with text")

    # Mine hard negatives — fall back to lower threshold if needed
    for neg_sim in [NEG_MIN_SIM, NEG_MIN_SIM_FLOOR]:
        print(f"\nMining hard negative triplets (neg_min_sim={neg_sim})...")
        triplets = mine_triplets(df_train, lookup, neg_sim)
        if len(triplets) >= MIN_TRIPLETS:
            break
        print(f"  Only {len(triplets)} triplets — trying lower threshold...")

    if len(triplets) < MIN_TRIPLETS:
        print(f"ERROR: only {len(triplets)} triplets found (need {MIN_TRIPLETS}). "
              f"Lower NEG_MIN_SIM_FLOOR further.")
        return

    # Load model + tokenizer
    print(f"\nLoading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = TextEncoder(BASE_MODEL).to(device)

    dataset = TripletTextDataset(triplets, idx_to_text, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)

    n_steps    = len(loader) * EPOCHS
    warmup     = max(1, int(n_steps * 0.05))
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler  = get_linear_schedule_with_warmup(optimizer, warmup, n_steps)
    scaler     = torch.amp.GradScaler()

    print(f"\nTraining: {len(triplets):,} triplets, {n_steps:,} steps, LR={LR}")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar       = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for a_ids, a_mask, p_ids, p_mask, n_ids, n_mask in pbar:
            a_ids, a_mask = a_ids.to(device), a_mask.to(device)
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)
            n_ids, n_mask = n_ids.to(device), n_mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                a_emb = model(a_ids, a_mask)
                p_emb = model(p_ids, p_mask)
                n_emb = model(n_ids, n_mask)
                loss  = info_nce_with_hard_neg(a_emb, p_emb, n_emb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        print(f"Epoch {epoch+1} avg loss: {total_loss / len(loader):.4f}")

    # Save backbone + tokenizer + projection head
    os.makedirs(MODEL_OUT, exist_ok=True)
    model.backbone.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    torch.save(model.proj.state_dict(), f'{MODEL_OUT}/proj_head.pt')
    print(f"\nSaved → {MODEL_OUT}/")
    print("\nNext:")
    print(f"  python extract_embeddings_v2.py --text_model {MODEL_OUT}")
    print(f"  python phase2_pipeline.py --phase1 --emb ft_v2 --sweep")


if __name__ == '__main__':
    train()
