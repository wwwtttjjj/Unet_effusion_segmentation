import albumentations
from albumentations.pytorch import ToTensorV2

height = 125
width = 300


def transformer_img():
    transform = albumentations.Compose([
        albumentations.CLAHE(clip_limit=5.0, tile_grid_size=(8, 8)),
        # albumentations.GaussianBlur(),
        # albumentations.Rotate([-35, -35], p=0.5),
        # albumentations.HorizontalFlip(p=0.5),
        # albumentations.VerticalFlip(p=0.5),
        # albumentations.Affine(shear=(-15, 15), translate_percent=0.1),
        # albumentations.ColorJitter(brightness=0.5),
        albumentations.Resize(height=height, width=width),
        ToTensorV2(),
    ])
    return transform


def val_form():
    transform = albumentations.Compose([
        albumentations.Resize(height=height, width=width),
        ToTensorV2(),
    ])
    return transform
