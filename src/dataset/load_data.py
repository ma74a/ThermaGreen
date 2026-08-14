from torch.utils.data import DataLoader
from typing import Tuple

from dataset.dataset import LoveDADataset
from utils.config import Config

def load_data() -> Tuple[DataLoader, DataLoader]:
    train_dataset = LoveDADataset(
        data_root=Config.DATA_ROOT,
        split="Train",
        transforms=Config.TRANSFORMS_DICT
    )
    val_dataset = LoveDADataset(
        data_root=Config.DATA_ROOT,
        split="Val",
        transforms=Config.TRANSFORMS_DICT
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    return train_loader, val_loader