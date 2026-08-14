import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from utils.config import Config

class LoveDADataset(Dataset):
    def __init__(self, data_root: str, split: str="Train", transforms: T=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms

        self.split_dir = self.data_root / self.split / self.split
        self.data = []

        for domain in ["Rural", "Urban"]:
            image_dir = self.split_dir / domain / "images_png"
            masks_dir = self.split_dir / domain / "masks_png"

            for img_path in image_dir.glob("*.png"):
                mask_path = masks_dir / img_path.name

                if mask_path.exists():
                    self.data.append(
                        (img_path, mask_path)
                    )

        print(f"{split}: {len(self.data)} samples")

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


# if __name__ == "__main__":
#     obj = LoveDADataset(Config.DATA_ROOT, transforms=Config.TRANSFORMS_DICT)
#     print(len(obj))
#     img, mask = obj[0]
#     print(img.shape)
#     print(mask.shape)
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Image")
#     plt.imshow(img.permute(1, 2, 0))
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.title("Mask")
#     plt.imshow(mask.cpu().numpy(), cmap="jet")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()