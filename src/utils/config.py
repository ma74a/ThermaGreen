import torchvision.transforms as T
import torch

class Config:
    DATA_ROOT="/home/etman/etman/ThermaGreen/data"

    IMG_SIZE=1024


    # Hyperparameter
    EPOCHS=20
    LR=0.001
    BATCH_SIZE=8

    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    # T.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    ])

    mask_transforms = T.Compose([
        T.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=T.InterpolationMode.NEAREST
        )
    ])

    TRANSFORMS_DICT = {
        "img": image_transforms,
        "mask": mask_transforms
    }
