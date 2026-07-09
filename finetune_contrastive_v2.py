"""
Fine-tune mE5-large (text) + DINOv2-large (image) with ANCE hard negatives.

Architecture:
  Text:  intfloat/multilingual-e5-large  (1024-dim avg pool + "query: " prefix)
  Image: facebook/dinov2-large           (1024-dim CLS token)
  Both:  2-layer projection head → 512-dim for contrastive loss
         Backbone embeddings (1024-dim) used at inference (not the projection)

Training strategy (ANCE):
  Phase 1 — Warm-up (NUM_WARMUP_EPOCHS epochs):
    InfoNCE with random in-batch negatives.
    Domain-adapts both models to GLAMI fashion products.
  Phase 2 — Hard negative refinement (NUM_HARD_EPOCHS epochs):
    Extract backbone embeddings from training set with current weights.
    FAISS: for each item, find nearest neighbours with sim >= NEG_MIN_SIM
           that have a DIFFERENT label.  These are the hard negatives the
           cosine-similarity threshold keeps confusing.
    InfoNCE with hard negatives explicitly in the denominator.

Both models are trained sequentially to stay within 12 GB VRAM.
Gradient checkpointing is enabled for both large models.

Output:
  finetuned_text_model_v2/   (mE5-large backbone + tokenizer)
  finetuned_image_model_v2/  (DINOv2-large backbone.pt)

Next steps:
  python extract_embeddings_e5dino.py \\
      --text_model  finetuned_text_model_v2 \\
      --image_model finetuned_image_model_v2 \\
      --suffix v2
  python phase2_pipeline.py --phase1 --emb v2 --sweep
"""
import os
import math
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_CSV   = 'data/items_train.csv'
IMG_DIR     = '/mnt/c/Users/lordr/Desktop/adm/fit_dataset_images'

TEXT_MODEL_NAME  = 'intfloat/multilingual-e5-large'
IMAGE_MODEL_NAME = 'facebook/dinov2-large'
TEXT_OUT         = 'finetuned_text_model_v2'
IMAGE_OUT        = 'finetuned_image_model_v2'

EMBED_DIM          = 512    # projection head output (backbone dim is 1024)
TEMPERATURE        = 0.05
MAX_TEXT_LEN       = 128    # 192→128: attention is O(n²), big speedup
DESC_CHARS         = 200
PAIRS_PER_LABEL    = 1
MAX_POS_PER_LABEL  = 3      # for hard neg phase
MAX_WARMUP_PAIRS   = 25_000  # cap warmup dataset — domain adapt, not full training

TEXT_BATCH         = 32     # was 16 — if OOM, change back to 16
TEXT_ACCUM         = 4      # effective batch = 128
TEXT_LR_BACKBONE   = 1e-5
TEXT_LR_HEAD       = 1e-4
NUM_WARMUP_EPOCHS  = 1      # was 2
NUM_HARD_EPOCHS    = 1

IMAGE_BATCH        = 32     # was 16 — if OOM, change back to 16
IMAGE_ACCUM        = 4
IMAGE_LR_BACKBONE  = 5e-6
IMAGE_LR_HEAD      = 5e-5
NUM_IMAGE_WARMUP   = 1      # was 3
NUM_IMAGE_HARD     = 1      # was 2

NEG_MIN_SIM        = 0.88
NEG_MIN_SIM_FLOOR  = 0.82
TOP_K_NEG          = 100
NLIST              = 500
NPROBE             = 25
NUM_WORKERS        = 4


# ── Encoders ──────────────────────────────────────────────────────────────────

def _avg_pool(last_hidden_state, attention_mask):
    hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return hidden.sum(1) / attention_mask.sum(1, keepdim=True).float()


class TextEncoderV2(nn.Module):
    def __init__(self, model_name=TEXT_MODEL_NAME, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size          # 1024 for large
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pool = _avg_pool(out.last_hidden_state, attention_mask)
        return F.normalize(self.proj(pool), p=2, dim=1)

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pool = _avg_pool(out.last_hidden_state, attention_mask)
        return F.normalize(pool, p=2, dim=1)              # raw backbone for FAISS


class ImageEncoderV2(nn.Module):
    def __init__(self, model_name=IMAGE_MODEL_NAME, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size          # 1024 for large
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, embed_dim),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]               # CLS token
        return F.normalize(self.proj(cls), p=2, dim=1)

    @torch.no_grad()
    def encode(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=1)


# ── Losses ────────────────────────────────────────────────────────────────────

def info_nce(a, p, temp=TEMPERATURE):
    """Standard symmetric InfoNCE with in-batch negatives."""
    logits = a @ p.T / temp
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def info_nce_hard(a, p, n, temp=TEMPERATURE):
    """
    InfoNCE with FAISS hard negatives explicitly in the denominator.
    For each anchor i:  positive = p[i],  negatives = {p[j≠i]} ∪ {n[j∀j]}
    """
    B    = a.shape[0]
    keys = torch.cat([p, n], dim=0)       # (2B, D)
    sim  = a @ keys.T / temp              # (B, 2B)
    lbl  = torch.arange(B, device=a.device)
    return F.cross_entropy(sim, lbl)


# ── Datasets ──────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    """Warm-up: random (anchor, positive) pairs sharing a label."""
    def __init__(self, csv_file, mode='text'):
        self.mode = mode
        df = pd.read_csv(csv_file, dtype={'itemId': str})
        grouped = df.groupby('label').filter(lambda x: len(x) >= 2)
        self.label_to_items = (
            grouped.groupby('label')
            .apply(lambda x: x.to_dict('records'), include_groups=False)
            .to_dict()
        )
        self.labels = list(self.label_to_items.keys())

        if mode == 'text':
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        else:
            self.tf_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return min(len(self.labels) * PAIRS_PER_LABEL, MAX_WARMUP_PAIRS)

    def _text(self, item):
        t = str(item.get('title', '') or '')
        d = str(item.get('description', '') or '')[:DESC_CHARS]
        enc = self.tokenizer(
            f"query: {t} {d}".strip(),
            padding='max_length', truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors='pt',
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def _image(self, item):
        path = os.path.join(IMG_DIR, f"{item['itemId']}.jpg")
        try:
            return self.tf_train(Image.open(path).convert('RGB'))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        label = self.labels[idx % len(self.labels)]
        items = self.label_to_items[label]
        chosen = np.random.choice(len(items), 2, replace=len(items) < 2)
        a, p   = items[chosen[0]], items[chosen[1]]
        if self.mode == 'text':
            a_ids, a_msk = self._text(a)
            p_ids, p_msk = self._text(p)
            return a_ids, a_msk, p_ids, p_msk
        else:
            return self._image(a), self._image(p)


class SingleItemTextDataset(Dataset):
    """Ordered single-item dataset for embedding extraction — matches id_to_idx order."""
    def __init__(self, df):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        t   = str(row.get('title', '') or '')
        d   = str(row.get('description', '') or '')[:DESC_CHARS]
        enc = self.tokenizer(
            f"query: {t} {d}".strip(),
            padding='max_length', truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors='pt',
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)


class TripletDataset(Dataset):
    """Hard negative phase: (anchor, positive, hard_neg) triplets."""
    def __init__(self, triplets, idx_to_item, mode='text'):
        self.triplets    = triplets
        self.idx_to_item = idx_to_item
        self.mode        = mode
        if mode == 'text':
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        else:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.triplets)

    def _text(self, idx):
        item = self.idx_to_item.get(idx, {})
        t    = str(item.get('title', '') or '')
        d    = str(item.get('description', '') or '')[:DESC_CHARS]
        enc  = self.tokenizer(
            f"query: {t} {d}".strip(),
            padding='max_length', truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors='pt',
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def _image(self, idx):
        item = self.idx_to_item.get(idx, {})
        path = os.path.join(IMG_DIR, f"{item.get('itemId', '_')}.jpg")
        try:
            return self.tf(Image.open(path).convert('RGB'))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        a_ix, p_ix, n_ix = self.triplets[idx]
        if self.mode == 'text':
            a_ids, a_msk = self._text(a_ix)
            p_ids, p_msk = self._text(p_ix)
            n_ids, n_msk = self._text(n_ix)
            return a_ids, a_msk, p_ids, p_msk, n_ids, n_msk
        else:
            return self._image(a_ix), self._image(p_ix), self._image(n_ix)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_optimizer(model, lr_backbone, lr_head):
    return AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.proj.parameters(),     'lr': lr_head},
    ], weight_decay=0.01)


def make_scheduler(optimizer, n_steps, warmup_frac=0.06):
    warmup = max(1, int(n_steps * warmup_frac))
    return get_linear_schedule_with_warmup(optimizer, warmup, n_steps)


@torch.no_grad()
def extract_embeddings(model, loader, device, mode='text'):
    """Extract backbone embeddings (not projected) for FAISS mining."""
    model.eval()
    parts = []
    for batch in tqdm(loader, desc="Extracting embeddings"):
        if mode == 'text':
            ids, msk = batch[0].to(device), batch[1].to(device)
            parts.append(model.encode(ids, msk).cpu().numpy())
        else:
            pv = batch[0].to(device)
            parts.append(model.encode(pv).cpu().numpy())
    model.train()
    return np.concatenate(parts, axis=0).astype(np.float32)


def mine_hard_triplets_from_lookup(embeddings, df_train, id_to_idx, min_sim):
    item_to_label = dict(zip(df_train['itemId'].astype(str), df_train['label']))
    idx_to_label  = {
        ix: item_to_label[iid]
        for iid, ix in id_to_idx.items()
        if iid in item_to_label
    }

    grouped   = df_train.groupby('label')['itemId'].apply(list)
    positives = []
    for label, items in grouped.items():
        idxs = [id_to_idx[str(i)] for i in items if str(i) in id_to_idx]
        if len(idxs) < 2:
            continue
        combos = list(combinations(idxs, 2))
        if len(combos) > MAX_POS_PER_LABEL:
            sel    = np.random.choice(len(combos), MAX_POS_PER_LABEL, replace=False)
            combos = [combos[i] for i in sel]
        positives.extend(combos)

    emb       = embeddings.astype(np.float32)
    n         = len(emb)
    quantizer = faiss.IndexFlatIP(emb.shape[1])
    index     = faiss.IndexIVFFlat(quantizer, emb.shape[1], NLIST,
                                   faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    index.nprobe = NPROBE

    unique_anchors = list({a for a, _ in positives})
    np.random.shuffle(unique_anchors)
    anchor_to_neg  = {}

    for s in tqdm(range(0, len(unique_anchors), 5_000), desc="Hard neg mining"):
        e        = min(s + 5_000, len(unique_anchors))
        batch    = unique_anchors[s:e]
        _, nbrs  = index.search(emb[batch], TOP_K_NEG + 1)
        for anchor_ix, row in zip(batch, nbrs):
            if anchor_ix in anchor_to_neg:
                continue
            a_lbl = idx_to_label.get(anchor_ix)
            if a_lbl is None:
                continue
            for nbr in row:
                if nbr < 0 or nbr >= n or nbr == anchor_ix:
                    continue
                if idx_to_label.get(nbr) == a_lbl:
                    continue
                if float(np.dot(emb[anchor_ix], emb[nbr])) >= min_sim:
                    anchor_to_neg[anchor_ix] = nbr
                    break

    triplets = [
        (a, p, anchor_to_neg[a])
        for a, p in positives if a in anchor_to_neg
    ]
    coverage = len(triplets) / max(len(positives), 1) * 100
    print(f"  Triplets: {len(triplets):,}  coverage {coverage:.1f}%  "
          f"(min_sim={min_sim})")
    return triplets


def build_idx_to_item(df_train, id_to_idx):
    """idx → row dict for dataset lookups."""
    id_to_row   = df_train.set_index('itemId').to_dict('index')
    idx_to_item = {}
    for iid, ix in id_to_idx.items():
        if iid in id_to_row:
            row = id_to_row[iid].copy()
            row['itemId'] = iid
            idx_to_item[ix] = row
    return idx_to_item


# ── Text training ─────────────────────────────────────────────────────────────

def train_text(df_train, id_to_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nTEXT ENCODER (mE5-large + hard negatives)\n{'='*60}")
    print(f"Device: {device}")

    model = TextEncoderV2().to(device)
    print(f"  Backbone hidden size: {model.backbone.config.hidden_size}")

    # ── Phase 1: warm-up ─────────────────────────────────────────────────────
    print(f"\nPhase 1: {NUM_WARMUP_EPOCHS} warm-up epochs (random in-batch negatives)")
    dataset   = PairDataset(TRAIN_CSV, mode='text')
    loader    = DataLoader(dataset, batch_size=TEXT_BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    n_steps   = (len(loader) // TEXT_ACCUM) * NUM_WARMUP_EPOCHS
    optimizer = make_optimizer(model, TEXT_LR_BACKBONE, TEXT_LR_HEAD)
    scheduler = make_scheduler(optimizer, n_steps)
    scaler    = torch.amp.GradScaler()

    model.train()
    for epoch in range(NUM_WARMUP_EPOCHS):
        total = 0.0
        optimizer.zero_grad()
        for step, (a_ids, a_msk, p_ids, p_msk) in enumerate(
                tqdm(loader, desc=f"  Text warm-up epoch {epoch+1}")):
            a_ids, a_msk = a_ids.to(device), a_msk.to(device)
            p_ids, p_msk = p_ids.to(device), p_msk.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = info_nce(model(a_ids, a_msk), model(p_ids, p_msk)) / TEXT_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % TEXT_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            total += loss.item() * TEXT_ACCUM
        print(f"  Epoch {epoch+1} avg loss: {total/len(loader):.4f}")

    # ── Extract embeddings for FAISS mining ──────────────────────────────────
    print("\nExtracting train text embeddings for hard negative mining...")
    # Use ordered single-item dataset so embeddings[id_to_idx[item_id]] is correct
    extract_dataset = SingleItemTextDataset(df_train)
    extract_loader  = DataLoader(extract_dataset, batch_size=TEXT_BATCH * 2,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)
    embeddings = extract_embeddings(model, extract_loader, device, mode='text')
    print(f"  Shape: {embeddings.shape}")

    # ── Mine hard negatives ───────────────────────────────────────────────────
    for neg_sim in [NEG_MIN_SIM, NEG_MIN_SIM_FLOOR]:
        triplets = mine_hard_triplets_from_lookup(embeddings, df_train, id_to_idx, neg_sim)
        if len(triplets) >= 5_000:
            break
        print(f"  Too few at {neg_sim}, trying {NEG_MIN_SIM_FLOOR}...")
    if len(triplets) < 1_000:
        print("WARNING: very few hard negative triplets found — skipping hard neg phase")
    else:
        # ── Phase 2: hard negative training ──────────────────────────────────
        print(f"\nPhase 2: {NUM_HARD_EPOCHS} hard negative epoch(s)  "
              f"({len(triplets):,} triplets)")
        idx_to_item  = build_idx_to_item(df_train, id_to_idx)
        hard_dataset = TripletDataset(triplets, idx_to_item, mode='text')
        hard_loader  = DataLoader(hard_dataset, batch_size=TEXT_BATCH, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        n_steps2    = (len(hard_loader) // TEXT_ACCUM) * NUM_HARD_EPOCHS
        optimizer2  = make_optimizer(model, TEXT_LR_BACKBONE / 2, TEXT_LR_HEAD / 2)
        scheduler2  = make_scheduler(optimizer2, n_steps2)
        scaler2     = torch.amp.GradScaler()

        model.train()
        for epoch in range(NUM_HARD_EPOCHS):
            total = 0.0
            optimizer2.zero_grad()
            for step, batch in enumerate(
                    tqdm(hard_loader, desc=f"  Text hard-neg epoch {epoch+1}")):
                a_ids, a_msk, p_ids, p_msk, n_ids, n_msk = [b.to(device) for b in batch]
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    a_e = model(a_ids, a_msk)
                    p_e = model(p_ids, p_msk)
                    n_e = model(n_ids, n_msk)
                    loss = info_nce_hard(a_e, p_e, n_e) / TEXT_ACCUM
                scaler2.scale(loss).backward()
                if (step + 1) % TEXT_ACCUM == 0:
                    scaler2.unscale_(optimizer2)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler2.step(optimizer2)
                    scaler2.update()
                    optimizer2.zero_grad()
                    scheduler2.step()
                total += loss.item() * TEXT_ACCUM
            print(f"  Hard epoch {epoch+1} avg loss: {total/len(hard_loader):.4f}")

    os.makedirs(TEXT_OUT, exist_ok=True)
    model.backbone.save_pretrained(TEXT_OUT)
    dataset.tokenizer.save_pretrained(TEXT_OUT)
    torch.save(model.proj.state_dict(), f'{TEXT_OUT}/proj_head.pt')
    print(f"\nText model saved → {TEXT_OUT}/")


# ── Image training ────────────────────────────────────────────────────────────

def train_image(df_train, id_to_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nIMAGE ENCODER (DINOv2-large + hard negatives)\n{'='*60}")

    model = ImageEncoderV2().to(device)

    # ── Phase 1: warm-up ─────────────────────────────────────────────────────
    print(f"\nPhase 1: {NUM_IMAGE_WARMUP} warm-up epochs (random in-batch negatives)")
    dataset   = PairDataset(TRAIN_CSV, mode='image')
    loader    = DataLoader(dataset, batch_size=IMAGE_BATCH, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    n_steps   = (len(loader) // IMAGE_ACCUM) * NUM_IMAGE_WARMUP
    optimizer = make_optimizer(model, IMAGE_LR_BACKBONE, IMAGE_LR_HEAD)
    scheduler = make_scheduler(optimizer, n_steps)
    scaler    = torch.amp.GradScaler()

    model.train()
    for epoch in range(NUM_IMAGE_WARMUP):
        total = 0.0
        optimizer.zero_grad()
        for step, (a_img, p_img) in enumerate(
                tqdm(loader, desc=f"  Image warm-up epoch {epoch+1}")):
            a_img, p_img = a_img.to(device), p_img.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = info_nce(model(a_img), model(p_img)) / IMAGE_ACCUM
            scaler.scale(loss).backward()
            if (step + 1) % IMAGE_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            total += loss.item() * IMAGE_ACCUM
        print(f"  Epoch {epoch+1} avg loss: {total/len(loader):.4f}")

    # ── Extract image embeddings for FAISS mining ─────────────────────────────
    print("\nExtracting train image embeddings for hard negative mining...")
    tf_eval = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class EvalImageDataset(Dataset):
        def __init__(self, df, img_dir, tf):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.tf = tf
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            path = os.path.join(self.img_dir, f"{row['itemId']}.jpg")
            try:   return self.tf(Image.open(path).convert('RGB')),
            except: return torch.zeros(3, 224, 224),

    eval_dataset = EvalImageDataset(
        df_train[df_train['itemId'].astype(str).isin(id_to_idx)].copy(),
        IMG_DIR, tf_eval
    )
    eval_loader  = DataLoader(eval_dataset, batch_size=IMAGE_BATCH * 2, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    img_emb      = extract_embeddings(model, eval_loader, device, mode='image')
    print(f"  Shape: {img_emb.shape}")

    # ── Mine image hard negatives ─────────────────────────────────────────────
    img_neg_sim = 0.80   # images don't need as high a threshold as text
    for neg_sim in [img_neg_sim, 0.70]:
        triplets = mine_hard_triplets_from_lookup(img_emb, df_train, id_to_idx, neg_sim)
        if len(triplets) >= 5_000:
            break
        print(f"  Too few at {neg_sim}, trying 0.70...")
    if len(triplets) < 1_000:
        print("WARNING: very few image hard negatives — skipping hard neg phase")
    else:
        print(f"\nPhase 2: {NUM_IMAGE_HARD} hard negative epoch(s)  "
              f"({len(triplets):,} triplets)")
        idx_to_item  = build_idx_to_item(df_train, id_to_idx)
        hard_dataset = TripletDataset(triplets, idx_to_item, mode='image')
        hard_loader  = DataLoader(hard_dataset, batch_size=IMAGE_BATCH, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        n_steps2   = (len(hard_loader) // IMAGE_ACCUM) * NUM_IMAGE_HARD
        optimizer2 = make_optimizer(model, IMAGE_LR_BACKBONE / 2, IMAGE_LR_HEAD / 2)
        scheduler2 = make_scheduler(optimizer2, n_steps2)
        scaler2    = torch.amp.GradScaler()

        model.train()
        for epoch in range(NUM_IMAGE_HARD):
            total = 0.0
            optimizer2.zero_grad()
            for step, (a_img, p_img, n_img) in enumerate(
                    tqdm(hard_loader, desc=f"  Image hard-neg epoch {epoch+1}")):
                a_img, p_img, n_img = a_img.to(device), p_img.to(device), n_img.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss = info_nce_hard(model(a_img), model(p_img), model(n_img)) / IMAGE_ACCUM
                scaler2.scale(loss).backward()
                if (step + 1) % IMAGE_ACCUM == 0:
                    scaler2.unscale_(optimizer2)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler2.step(optimizer2)
                    scaler2.update()
                    optimizer2.zero_grad()
                    scheduler2.step()
                total += loss.item() * IMAGE_ACCUM
            print(f"  Hard epoch {epoch+1} avg loss: {total/len(hard_loader):.4f}")

    os.makedirs(IMAGE_OUT, exist_ok=True)
    torch.save(model.backbone.state_dict(), f'{IMAGE_OUT}/backbone.pt')
    print(f"\nImage model saved → {IMAGE_OUT}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading training data...")
    df_train           = pd.read_csv(TRAIN_CSV)
    df_train['itemId'] = df_train['itemId'].astype(str)
    print(f"  {len(df_train):,} items, {df_train['label'].nunique():,} labels")

    # Build id → embedding index (position in DataFrame order)
    id_to_idx = {row['itemId']: i for i, row in df_train.iterrows()}

    train_text(df_train, id_to_idx)
    train_image(df_train, id_to_idx)

    print("\n" + "="*60)
    print("Training complete. Next steps:")
    print(f"  python extract_embeddings_e5dino.py \\")
    print(f"      --text_model  {TEXT_OUT} \\")
    print(f"      --image_model {IMAGE_OUT} \\")
    print(f"      --suffix v2")
    print(f"  python phase2_pipeline.py --phase1 --emb v2 --sweep")


if __name__ == '__main__':
    main()
