"""
Extract embeddings using FINE-TUNED text + image models.
Uses the backbone (768-dim) not the projection head — the backbone
learns better general representations, the projection head is just
for the contrastive loss geometry.

Produces: glami_embeddings_train_ft.h5 and glami_embeddings_phase_1_ft.h5
"""
import os
import re
import json
import h5py
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# DATASET (same as original extract_embeddings.py)
# =============================================================================
class GlamiItemDataset(Dataset):
    def __init__(self, csv_file, img_dir, mappings_file):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

        with open(mappings_file, 'r') as f:
            self.mappings = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained('finetuned_text_model')

        self.image_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.geo_to_eur = {
            'gr': 1.0, 'sk': 1.0, 'si': 1.0, 'hr': 1.0,
            'lt': 1.0, 'lv': 1.0, 'it': 1.0, 'ee': 1.0,
            'bg': 0.51, 'cz': 0.04, 'ro': 0.20, 'pl': 0.23, 'hu': 0.0025
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        item_id = str(row['itemId'])

        title = str(row.get('title', '') or '')
        description = str(row.get('description', '') or '')
        text_inputs = self.tokenizer(
            f"{title} {description}",
            padding='max_length', truncation=True, max_length=128, return_tensors='pt'
        )

        img_path = os.path.join(self.img_dir, f"{item_id}.jpg")
        try:
            image_tensor = self.image_transforms(Image.open(img_path).convert('RGB'))
        except Exception:
            image_tensor = torch.zeros((3, 224, 224))

        price = float(row.get('price', 0.0))
        geo = str(row.get('geo', '')).lower()
        normalized_price = price * self.geo_to_eur.get(geo, 1.0)

        def get_mapped_id(raw_val, map_dict):
            numbers = re.findall(r'\d+', str(raw_val))
            if numbers and str(numbers[0]) in map_dict:
                return map_dict[str(numbers[0])]
            return 0

        dept_id = get_mapped_id(row.get('departmentIds', ''), self.mappings['departments'])
        color_id = get_mapped_id(row.get('colorTagIdsString', ''), self.mappings['colors'])
        brand_id = get_mapped_id(row.get('brandEditionTagId', ''), self.mappings['brands'])

        return {
            'item_id': item_id,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'image': image_tensor,
            'price': torch.tensor([normalized_price], dtype=torch.float32),
            'dept_id': torch.tensor([dept_id], dtype=torch.int32),
            'color_id': torch.tensor([color_id], dtype=torch.int32),
            'brand_id': torch.tensor([brand_id], dtype=torch.int32)
        }


# =============================================================================
# EXTRACTION
# =============================================================================
def extract_and_save(csv_path, img_path, output_h5, text_model_path='finetuned_text_model'):
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = GlamiItemDataset(csv_path, img_path, 'categorical_mappings.json')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Load fine-tuned text model (backbone only)
    print(f"Loading fine-tuned text model from {text_model_path}...")
    text_model = AutoModel.from_pretrained(text_model_path).to(device)
    text_model.eval()

    # Load fine-tuned image model
    print("Loading fine-tuned image model...")
    vision_model = models.convnext_tiny(weights='DEFAULT')
    vision_model.classifier[2] = torch.nn.Identity()
    vision_model.load_state_dict(torch.load('finetuned_image_model/backbone.pt',
                                             map_location=device, weights_only=True))
    vision_model = vision_model.to(device)
    vision_model.eval()

    total_items = len(dataset)
    embedding_dim = 768

    print(f"Extracting {total_items} items → {output_h5}")
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('item_ids', shape=(total_items,), dtype='S20')
        f.create_dataset('text_emb', shape=(total_items, embedding_dim), dtype='float32')
        f.create_dataset('image_emb', shape=(total_items, embedding_dim), dtype='float32')
        f.create_dataset('price', shape=(total_items, 1), dtype='float32')
        f.create_dataset('dept_id', shape=(total_items, 1), dtype='int32')
        f.create_dataset('color_id', shape=(total_items, 1), dtype='int32')
        f.create_dataset('brand_id', shape=(total_items, 1), dtype='int32')

        ptr = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting → {output_h5}"):
                bs = len(batch['item_id'])

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)

                # Text: use autocast (works fine)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    text_out = text_model(input_ids=input_ids, attention_mask=attention_mask)
                    text_features = text_out.last_hidden_state[:, 0, :]

                # Image: full float32 (ConvNeXt NaN issue)
                image_features = vision_model(images.float()).flatten(1)

                # L2 normalize
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
                image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

                f['item_ids'][ptr:ptr+bs] = [uid.encode('utf8') for uid in batch['item_id']]
                f['text_emb'][ptr:ptr+bs] = text_features.float().cpu().numpy()
                f['image_emb'][ptr:ptr+bs] = image_features.cpu().numpy()
                f['price'][ptr:ptr+bs] = batch['price'].numpy()
                f['dept_id'][ptr:ptr+bs] = batch['dept_id'].numpy()
                f['color_id'][ptr:ptr+bs] = batch['color_id'].numpy()
                f['brand_id'][ptr:ptr+bs] = batch['brand_id'].numpy()

                ptr += bs

    print(f"Done! Saved {ptr} items to {output_h5}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model', default='finetuned_text_model',
                        help='Path to text model backbone (default: finetuned_text_model)')
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'phase_1', 'phase_2'],
                        choices=['train', 'phase_1', 'phase_2'])
    args = parser.parse_args()

    suffix = '_v2' if args.text_model != 'finetuned_text_model' else ''

    img_path = '/mnt/c/Users/lordr/Desktop/adm/fit_dataset_images'
    split_map = {
        'train':   ('data/items_train.csv',   f'glami_embeddings_train_ft{suffix}.h5'),
        'phase_1': ('data/items_phase_1.csv', f'glami_embeddings_phase_1_ft{suffix}.h5'),
        'phase_2': ('data/items_phase_2.csv', f'glami_embeddings_phase_2_ft{suffix}.h5'),
    }

    for split in args.splits:
        csv_path, output_h5 = split_map[split]
        print("\n" + "=" * 60)
        print(f"EXTRACTING {split.upper()} (text_model={args.text_model})")
        print("=" * 60)
        extract_and_save(csv_path, img_path, output_h5,
                         text_model_path=args.text_model)

    print("\nAll done! Now run phase2_pipeline.py")
