"""Dataset and dataloaders for PlantVillage (folder per class)."""
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.config import (
    RAW_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_SPLIT,
    RANDOM_SEED,
)


class SubsetWithTransform(Dataset):
    """Subset that applies a transform to each item."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def find_plantvillage_root():
    """Find folder with class subfolders (e.g. Potato___Late_blight)."""
    for candidate in [
        RAW_DIR / "PlantVillage",
        RAW_DIR / "plantvillage-dataset",
        RAW_DIR,
    ]:
        if not candidate.exists():
            continue
        for d in candidate.iterdir():
            if d.is_dir() and "___" in d.name:
                return candidate
        if (candidate / "train").exists():
            return candidate / "train"
        if (candidate / "Plant_leave_diseases_dataset_with_augmentation").exists():
            return candidate / "Plant_leave_diseases_dataset_with_augmentation" / "train"
    return None


def get_dataloaders(data_root=None):
    if data_root is None:
        data_root = find_plantvillage_root()
    if data_root is None:
        raise FileNotFoundError(
            "PlantVillage not found. Download to data/raw/ and ensure structure: "
            "data/raw/<...>/ClassName/img1.jpg ..."
        )
    data_root = Path(data_root)

    full_ds = ImageFolder(str(data_root), transform=None)
    n = len(full_ds)
    indices = list(range(n))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)
    n_val = int(n * VAL_SPLIT)
    n_train = n - n_val
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = SubsetWithTransform(full_ds, train_idx, get_transforms(True))
    val_ds = SubsetWithTransform(full_ds, val_idx, get_transforms(False))

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, val_loader, full_ds.classes
