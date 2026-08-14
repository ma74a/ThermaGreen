import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple

from utils.config import Config

class LoveDADataset(Dataset):
    def __init__(self, data_root: str, split: str="Train", transforms: T=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms

        self.split_dir = self.data_root / self.split
        self.data = []

        for domain in ["Rural", "Urban"]:
            image_dir = self.split_dir / domain / "images_png"
            masks_dir = self.split_dir / domain / "masks_png"
            for img_path in image_dir.glob("*.png"):
                mask_path = masks_dir / img_path.name

                if mask_path.exists():
                    self.data.append((img_path, mask_path))

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.data[index]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transforms:
            img = Config.TRANSFORMS_DICT["img"](img)
            mask = Config.TRANSFORMS_DICT["mask"](mask)

        return img, torch.tensor(np.array(mask), dtype=torch.long)