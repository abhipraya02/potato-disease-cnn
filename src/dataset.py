import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import IMG_SIZE, ROOT, SPLITS_DIR

# Standard ImageNet colour statistics (used by pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size=IMG_SIZE, augment=True):
    """Image preprocessing. Augmentation is only used for training."""
    if augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PotatoDataset(Dataset):
    """Loads images listed in a split CSV (path, class_name, label)."""

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(ROOT / row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])


def get_loaders(img_size=IMG_SIZE, batch_size=32, augment=True, num_workers=0):
    """Build train/val/test DataLoaders from the saved split CSVs."""
    train_ds = PotatoDataset(SPLITS_DIR / "train.csv", get_transforms(img_size, augment))
    val_ds = PotatoDataset(SPLITS_DIR / "val.csv", get_transforms(img_size, augment=False))
    test_ds = PotatoDataset(SPLITS_DIR / "test.csv", get_transforms(img_size, augment=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader