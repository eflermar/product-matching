"""
Fine-tune XLM-RoBERTa-base and ConvNeXt-Tiny with contrastive learning.
Same-label items → embeddings close together, different-label → far apart.

Uses MultipleNegativesRankingLoss-style approach:
  - Each batch has anchor-positive pairs (same label)
  - All other items in the batch serve as negatives
  - This is efficient: batch_size=64 gives 63 negatives per anchor for free

Requires: torch, transformers, torchvision, PIL, pandas, tqdm
GPU: 4070 (12GB VRAM) — batch_size=48 is safe for both models

Output:
  - finetuned_text_model/   (XLM-RoBERTa with projection head)
  - finetuned_image_model/  (ConvNeXt-Tiny with projection head)
"""
import os
import re
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# CONFIG
# =============================================================================
TRAIN_CSV = 'data/items_train.csv'
IMG_DIR = '/mnt/c/Users/lordr/Desktop/adm/fit_dataset_images'
MAPPINGS_FILE = 'categorical_mappings.json'

TEXT_LR = 2e-5
IMAGE_LR = 5e-5
BATCH_SIZE = 48        # Safe for 12GB VRAM with both models
NUM_EPOCHS_TEXT = 3
NUM_EPOCHS_IMAGE = 5
EMBED_DIM = 256        # Projection head output dimension
TEMPERATURE = 0.05     # InfoNCE temperature
MAX_TEXT_LEN = 128     # Full title + description
NUM_WORKERS = 4
ACCUMULATION_STEPS = 4 # Effective batch = 48*4 = 192 (more in-batch negatives)
PAIRS_PER_LABEL = 1    # Keep at 1, longer text already gives more signal
WARMUP_RATIO = 0.1     # Linear warmup for first 10% of optimizer steps

# =============================================================================
# MODELS
# =============================================================================
class TextEncoder(nn.Module):
    """XLM-RoBERTa with a projection head for contrastive learning."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained('xlm-roberta-base')
        self.proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, embed_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # CLS token
        return F.normalize(self.proj(cls), p=2, dim=1)

    def get_backbone_embedding(self, input_ids, attention_mask):
        """For extraction: return L2-normalized 768-dim CLS (no projection)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return F.normalize(cls, p=2, dim=1)


class ImageEncoder(nn.Module):
    """ConvNeXt-Tiny with a projection head for contrastive learning."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = models.convnext_tiny(weights='DEFAULT')
        # Remove classifier head but keep LayerNorm + Flatten
        self.backbone.classifier[2] = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, embed_dim),
        )

    def forward(self, images):
        feats = self.backbone(images).flatten(1)
        return F.normalize(self.proj(feats), p=2, dim=1)

    def get_backbone_embedding(self, images):
        """For extraction: return L2-normalized 768-dim features (no projection)."""
        feats = self.backbone(images).flatten(1)
        return F.normalize(feats, p=2, dim=1)


# =============================================================================
# DATASET — yields anchor-positive pairs
# =============================================================================
class ContrastivePairDataset(Dataset):
    """
    Each __getitem__ returns an (anchor, positive) pair of items sharing a label.
    The dataloader batches these, giving us B anchor-positive pairs per batch.
    All other items in the batch act as negatives (in-batch negatives).
    """
    def __init__(self, csv_file, img_dir, mode='text'):
        self.mode = mode
        self.img_dir = img_dir

        df = pd.read_csv(csv_file, dtype={'itemId': str})
        # Build label → items mapping (only labels with 2+ items)
        grouped = df.groupby('label').filter(lambda x: len(x) >= 2)
        self.label_to_items = grouped.groupby('label').apply(
            lambda x: x.to_dict('records'), include_groups=False
        ).to_dict()
        self.labels = list(self.label_to_items.keys())

        if mode == 'text':
            self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        else:
            self.image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        # Multiple pairs per label per epoch for more training signal
        return len(self.labels) * PAIRS_PER_LABEL

    def _get_text(self, item):
        title = str(item.get('title', '') or '')
        desc = str(item.get('description', '') or '')
        text = f"{title} {desc}".strip()
        tokens = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors='pt'
        )
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

    def _get_image(self, item):
        img_path = os.path.join(self.img_dir, f"{item['itemId']}.jpg")
        try:
            img = Image.open(img_path).convert('RGB')
            return self.image_transforms(img)
        except Exception:
            return torch.zeros((3, 224, 224))

    def __getitem__(self, idx):
        label = self.labels[idx % len(self.labels)]
        items = self.label_to_items[label]
        # Pick 2 random items with this label
        chosen = np.random.choice(len(items), 2, replace=len(items) < 2)
        anchor_item = items[chosen[0]]
        pos_item = items[chosen[1]]

        if self.mode == 'text':
            a_ids, a_mask = self._get_text(anchor_item)
            p_ids, p_mask = self._get_text(pos_item)
            return {
                'a_input_ids': a_ids, 'a_attention_mask': a_mask,
                'p_input_ids': p_ids, 'p_attention_mask': p_mask,
            }
        else:
            return {
                'a_image': self._get_image(anchor_item),
                'p_image': self._get_image(pos_item),
            }


# =============================================================================
# INFO-NCE LOSS (symmetric)
# =============================================================================
def info_nce_loss(anchor_emb, positive_emb, temperature=0.05):
    """
    Symmetric InfoNCE: each anchor's positive is the corresponding positive,
    all other items in the batch are negatives.
    """
    # anchor_emb: (B, D), positive_emb: (B, D), both L2-normalized
    logits = anchor_emb @ positive_emb.T / temperature  # (B, B)
    labels = torch.arange(len(anchor_emb), device=anchor_emb.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_p) / 2


# =============================================================================
# TRAINING LOOPS
# =============================================================================
def train_text_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = ContrastivePairDataset(TRAIN_CSV, IMG_DIR, mode='text')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model = TextEncoder(embed_dim=EMBED_DIM).to(device)

    # Different LR for backbone vs projection head
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': TEXT_LR},
        {'params': model.proj.parameters(), 'lr': TEXT_LR * 10},
    ], weight_decay=0.01)

    scaler = torch.amp.GradScaler()
    # total_steps = number of OPTIMIZER steps (not batches)
    optimizer_steps = (len(loader) // ACCUMULATION_STEPS) * NUM_EPOCHS_TEXT
    warmup_steps = int(optimizer_steps * WARMUP_RATIO)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*60}")
    print(f"TRAINING TEXT ENCODER")
    print(f"Labels: {len(dataset.labels)}, Steps/epoch: {len(loader)}")
    print(f"Epochs: {NUM_EPOCHS_TEXT}, Batch: {BATCH_SIZE}, Accum: {ACCUMULATION_STEPS}")
    print(f"Optimizer steps: {optimizer_steps}, Warmup: {warmup_steps}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(NUM_EPOCHS_TEXT):
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Text Epoch {epoch+1}/{NUM_EPOCHS_TEXT}")
        for step, batch in enumerate(pbar):
            a_ids = batch['a_input_ids'].to(device)
            a_mask = batch['a_attention_mask'].to(device)
            p_ids = batch['p_input_ids'].to(device)
            p_mask = batch['p_attention_mask'].to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                a_emb = model(a_ids, a_mask)
                p_emb = model(p_ids, p_mask)
                loss = info_nce_loss(a_emb, p_emb, TEMPERATURE) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix(loss=f"{loss.item()*ACCUMULATION_STEPS:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

    # Save
    os.makedirs('finetuned_text_model', exist_ok=True)
    model.backbone.save_pretrained('finetuned_text_model')
    dataset.tokenizer.save_pretrained('finetuned_text_model')
    torch.save(model.proj.state_dict(), 'finetuned_text_model/proj_head.pt')
    print("Text model saved to finetuned_text_model/")
    return model


def train_image_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = ContrastivePairDataset(TRAIN_CSV, IMG_DIR, mode='image')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model = ImageEncoder(embed_dim=EMBED_DIM).to(device)

    # Freeze early layers (0-3), fine-tune later layers (4-7) + projection
    for name, param in model.backbone.named_parameters():
        if not any(f'features.{i}' in name for i in [4, 5, 6, 7]):
            param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.backbone.named_parameters()
                    if p.requires_grad and ('features.4' in n or 'features.5' in n)],
         'lr': IMAGE_LR * 0.1},
        {'params': [p for n, p in model.backbone.named_parameters()
                    if p.requires_grad and ('features.6' in n or 'features.7' in n)],
         'lr': IMAGE_LR},
        {'params': model.proj.parameters(), 'lr': IMAGE_LR * 2},
    ], weight_decay=0.01)

    scaler = torch.amp.GradScaler()
    # total_steps = number of OPTIMIZER steps (not batches)
    optimizer_steps = (len(loader) // ACCUMULATION_STEPS) * NUM_EPOCHS_IMAGE
    warmup_steps = int(optimizer_steps * WARMUP_RATIO)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(optimizer_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*60}")
    print(f"TRAINING IMAGE ENCODER")
    print(f"Labels: {len(dataset.labels)}, Steps/epoch: {len(loader)}")
    print(f"Epochs: {NUM_EPOCHS_IMAGE}, Batch: {BATCH_SIZE}, Accum: {ACCUMULATION_STEPS}")
    print(f"Optimizer steps: {optimizer_steps}, Warmup: {warmup_steps}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(NUM_EPOCHS_IMAGE):
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Image Epoch {epoch+1}/{NUM_EPOCHS_IMAGE}")
        for step, batch in enumerate(pbar):
            a_img = batch['a_image'].to(device)
            p_img = batch['p_image'].to(device)

            # No autocast for ConvNeXt (produces NaN with float16)
            a_emb = model(a_img.float())
            p_emb = model(p_img.float())
            loss = info_nce_loss(a_emb, p_emb, TEMPERATURE) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix(loss=f"{loss.item()*ACCUMULATION_STEPS:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")

    # Save
    os.makedirs('finetuned_image_model', exist_ok=True)
    torch.save(model.backbone.state_dict(), 'finetuned_image_model/backbone.pt')
    torch.save(model.proj.state_dict(), 'finetuned_image_model/proj_head.pt')
    print("Image model saved to finetuned_image_model/")
    return model


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("CONTRASTIVE FINE-TUNING")
    print("=" * 60)

    print("\n--- Phase 1: Text Model ---")
    train_text_model()

    print("\n--- Phase 2: Image Model ---")
    train_image_model()

    print("\nDone! Now run extract_embeddings_v2.py to re-extract embeddings.")
