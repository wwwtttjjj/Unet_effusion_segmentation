from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2
def transformer_img():
    transform = albumentations.Compose([
        albumentations.CLAHE(clip_limit=5.0, tile_grid_size=(8, 8)),
        albumentations.ColorJitter(brightness=0.5),
        ])
    return transform

