import torch
from typing import Dict
from torch.utils.data import Dataset
from transformers import ViTImageProcessor

def process_example_for_ViT(example: Dataset, processor: ViTImageProcessor) -> Dict[str, torch.Tensor]:
    inputs = processor(example['image'], return_tensors='pt')
    inputs['labels'] = example['label']
    return inputs

def transform_data_for_ViT(example_batch: Dataset, processor: ViTImageProcessor) -> Dict[str, torch.Tensor]:
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

def collate_fn_for_ViT(batch):
    return {'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])}
