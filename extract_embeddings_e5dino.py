"""
Extract embeddings using mE5-large + DINOv2-large.
Works with both the pretrained HuggingFace weights and fine-tuned checkpoints.

Usage (pretrained — e5dino suffix):
  python extract_embeddings_e5dino.py

Usage (fine-tuned — v2 suffix):
  python extract_embeddings_e5dino.py \\
      --text_model  finetuned_text_model_v2 \\
      --image_model finetuned_image_model_v2 \\
      --suffix v2

Produces (pretrained):
  glami_embeddings_train_e5dino.h5
  glami_embeddings_phase_1_e5dino.h5
  glami_embeddings_phase_2_e5dino.h5

Produces (fine-tuned, --suffix v2):
  glami_embeddings_train_v2.h5
  glami_embeddings_phase_1_v2.h5
  glami_embeddings_phase_2_v2.h5
"""
import os
import re
import json
import h5py
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_TEXT_MODEL  = 'intfloat/multilingual-e5-large'
DEFAULT_IMAGE_MODEL = 'facebook/dinov2-large'
EMBEDDING_DIM       = 1024
MAX_TEXT_LEN        = 192
IMG_DIR             = '/mnt/c/Users/lordr/Desktop/adm/fit_dataset_images'


# ── Dataset ───────────────────────────────────────────────────────────────────

class GlamiItemDataset(Dataset):
    def __init__(self, csv_file, img_dir, mappings_file,
                 text_model_name=DEFAULT_TEXT_MODEL):
        self.data    = pd.read_csv(csv_file)
        self.img_dir = img_dir

        with open(mappings_file, 'r') as f:
            self.mappings = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Standard ImageNet transforms — correct for DINOv2
        self.image_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.geo_to_eur = {
            'gr': 1.0, 'sk': 1.0, 'si': 1.0, 'hr': 1.0,
            'lt': 1.0, 'lv': 1.0, 'it': 1.0, 'ee': 1.0,
            'bg': 0.51, 'cz': 0.04, 'ro': 0.20, 'pl': 0.23, 'hu': 0.0025,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row     = self.data.iloc[idx]
        item_id = str(row['itemId'])

        title = str(row.get('title', '') or '')
        desc  = str(row.get('description', '') or '')
        # "query: " prefix activates multilingual-e5's retrieval mode
        text_inputs = self.tokenizer(
            f"query: {title} {desc}",
            padding='max_length', truncation=True,
            max_length=MAX_TEXT_LEN, return_tensors='pt',
        )

        img_path = os.path.join(self.img_dir, f"{item_id}.jpg")
        try:
            image_tensor = self.image_transforms(Image.open(img_path).convert('RGB'))
        except Exception:
            image_tensor = torch.zeros((3, 224, 224))

        price            = float(row.get('price', 0.0))
        geo              = str(row.get('geo', '')).lower()
        normalized_price = price * self.geo_to_eur.get(geo, 1.0)

        def get_mapped_id(raw_val, map_dict):
            numbers = re.findall(r'\d+', str(raw_val))
            if numbers and str(numbers[0]) in map_dict:
                return map_dict[str(numbers[0])]
            return 0

        return {
            'item_id':        item_id,
            'input_ids':      text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'image':          image_tensor,
            'price':          torch.tensor([normalized_price], dtype=torch.float32),
            'dept_id':        torch.tensor([get_mapped_id(row.get('departmentIds', ''),
                                                          self.mappings['departments'])],
                                           dtype=torch.int32),
            'color_id':       torch.tensor([get_mapped_id(row.get('colorTagIdsString', ''),
                                                          self.mappings['colors'])],
                                           dtype=torch.int32),
            'brand_id':       torch.tensor([get_mapped_id(row.get('brandEditionTagId', ''),
                                                          self.mappings['brands'])],
                                           dtype=torch.int32),
        }


# ── Extraction ────────────────────────────────────────────────────────────────

def _avg_pool(last_hidden_state, attention_mask):
    """Average-pool token embeddings, ignoring padding."""
    hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()


def extract_and_save(csv_path, output_h5, text_model, image_model, device,
                     text_model_name=DEFAULT_TEXT_MODEL, batch_size=32):
    dataset    = GlamiItemDataset(csv_path, IMG_DIR, 'categorical_mappings.json',
                                  text_model_name=text_model_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    total = len(dataset)
    print(f"  {total:,} items  →  {output_h5}")

    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('item_ids',  shape=(total,),               dtype='S20')
        f.create_dataset('text_emb',  shape=(total, EMBEDDING_DIM), dtype='float32')
        f.create_dataset('image_emb', shape=(total, EMBEDDING_DIM), dtype='float32')
        f.create_dataset('price',     shape=(total, 1),             dtype='float32')
        f.create_dataset('dept_id',   shape=(total, 1),             dtype='int32')
        f.create_dataset('color_id',  shape=(total, 1),             dtype='int32')
        f.create_dataset('brand_id',  shape=(total, 1),             dtype='int32')

        ptr = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=os.path.basename(output_h5)):
                bs = len(batch['item_id'])

                # ── Text: multilingual-e5-large (average pool) ────────────────
                ids   = batch['input_ids'].to(device)
                mask  = batch['attention_mask'].to(device)
                t_out = text_model(input_ids=ids, attention_mask=mask)
                t_feat = _avg_pool(t_out.last_hidden_state.float(), mask)
                t_feat = F.normalize(t_feat, p=2, dim=1)

                # ── Image: DINOv2-large (CLS token) ──────────────────────────
                pv     = batch['image'].to(device)
                v_out  = image_model(pixel_values=pv)
                v_feat = v_out.last_hidden_state[:, 0, :].float()  # CLS
                v_feat = F.normalize(v_feat, p=2, dim=1)

                f['item_ids'][ptr:ptr+bs]  = [uid.encode('utf8') for uid in batch['item_id']]
                f['text_emb'][ptr:ptr+bs]  = t_feat.cpu().numpy()
                f['image_emb'][ptr:ptr+bs] = v_feat.cpu().numpy()
                f['price'][ptr:ptr+bs]     = batch['price'].numpy()
                f['dept_id'][ptr:ptr+bs]   = batch['dept_id'].numpy()
                f['color_id'][ptr:ptr+bs]  = batch['color_id'].numpy()
                f['brand_id'][ptr:ptr+bs]  = batch['brand_id'].numpy()

                ptr += bs

    print(f"  Saved {ptr:,} items to {output_h5}")


# ── Main ──────────────────────────────────────────────────────────────────────

BASE_SPLITS = {
    'train':   'data/items_train.csv',
    'phase_1': 'data/items_phase_1.csv',
    'phase_2': 'data/items_phase_2.csv',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'phase_1', 'phase_2'],
                        choices=list(BASE_SPLITS.keys()),
                        help='Which splits to extract (default: all three)')
    parser.add_argument('--text_model', default=DEFAULT_TEXT_MODEL,
                        help='Text model: HF name or path to saved backbone dir '
                             '(default: intfloat/multilingual-e5-large)')
    parser.add_argument('--image_model', default=DEFAULT_IMAGE_MODEL,
                        help='Image model: HF name or dir with backbone.pt '
                             '(default: facebook/dinov2-large)')
    parser.add_argument('--suffix', default=None,
                        help='Output file suffix, e.g. "v2" → '
                             'glami_embeddings_train_v2.h5 '
                             '(default: "e5dino" for pretrained, required otherwise)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Reduce to 16 if CUDA OOM (default: 32)')
    args = parser.parse_args()

    # Determine output suffix
    if args.suffix is None:
        if args.text_model == DEFAULT_TEXT_MODEL and args.image_model == DEFAULT_IMAGE_MODEL:
            suffix = 'e5dino'
        else:
            parser.error('--suffix is required when using non-default models')
    else:
        suffix = args.suffix

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Text  model : {args.text_model}")
    print(f"Image model : {args.image_model}")
    print(f"Suffix      : {suffix}")

    # Load text model
    print(f"\nLoading text model from {args.text_model}...")
    text_model = AutoModel.from_pretrained(args.text_model,
                                           torch_dtype=torch.float16).to(device)
    text_model.eval()

    # Load image model — handle fine-tuned backbone.pt separately
    backbone_pt = os.path.join(args.image_model, 'backbone.pt')
    if os.path.isfile(backbone_pt):
        print(f"Loading DINOv2-large architecture then fine-tuned weights "
              f"from {backbone_pt}...")
        image_model = AutoModel.from_pretrained(DEFAULT_IMAGE_MODEL,
                                                torch_dtype=torch.float16).to(device)
        state = torch.load(backbone_pt, map_location=device, weights_only=True)
        image_model.load_state_dict(state)
    else:
        print(f"Loading image model from {args.image_model}...")
        image_model = AutoModel.from_pretrained(args.image_model,
                                                torch_dtype=torch.float16).to(device)
    image_model.eval()

    for split in args.splits:
        csv_path  = BASE_SPLITS[split]
        output_h5 = f'glami_embeddings_{split}_{suffix}.h5'
        print(f"\n{'='*60}\n{split.upper()}\n{'='*60}")
        extract_and_save(csv_path, output_h5, text_model, image_model, device,
                         text_model_name=args.text_model,
                         batch_size=args.batch_size)

    print("\nAll done. Now run:")
    print(f"  python phase2_pipeline.py --phase1 --emb {suffix} --sweep")
