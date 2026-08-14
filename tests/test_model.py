import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms as T
from matplotlib.patches import Patch


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "/home/etman/etman/ThermaGreen/checkpoints/best_model.pth"
IMAGE_PATH = "/home/etman/etman/ThermaGreen/tests/imgs/5167.png"

IMG_SIZE = 512
NUM_CLASSES = 8


# ============================================================
# Classes
# ============================================================

CLASSES = [
    "No-data",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agriculture"
]


# ============================================================
# Model
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_connections, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels+skip_connections, out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Padding in case the input dimensions are not perfectly divisible by 2
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            
            # Concatenate along the channel dimension
            x = torch.cat([skip, x], dim=1)

        return self.conv(x)
    


class ResNetUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNetUNet, self).__init__()
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.encoder0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.pool = model.maxpool
        self.encoder1 = model.layer1
        self.encoder2 = model.layer2
        self.encoder3 = model.layer3
        self.encoder4 = model.layer4

        # Decoder
        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128,  64,  64)
        self.up1 = DecoderBlock( 64,  64,  64)
        self.up0 = DecoderBlock(64, 0, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3) # Bottleneck layer

        d = self.up4(e4, skip=e3)
        d = self.up3(d, skip=e2)
        d = self.up2(d, skip=e1)
        d = self.up1(d, skip=e0)
        d = self.up0(d)

        return self.final_conv(d)


# ============================================================
# Load model
# ============================================================

print("Device:", DEVICE)

model = ResNetUNet(
    num_classes=NUM_CLASSES
).to(DEVICE)


print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# If you saved only model.state_dict()
model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")


# ============================================================
# Image preprocessing
# ============================================================

img_transforms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])


# ============================================================
# Load image
# ============================================================

print("Loading image...")

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

input_tensor = img_transforms(
    original_image
)


# Add batch dimension

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(DEVICE)

print(
    "Input shape:",
    input_tensor.shape
)


# ============================================================
# Inference
# ============================================================

print("Running inference...")

with torch.no_grad():

    outputs = model(
        input_tensor
    )


print(
    "Output shape:",
    outputs.shape
)


# ============================================================
# Get predicted class
# ============================================================

prediction = outputs.argmax(
    dim=1
)


# Remove batch dimension

prediction = prediction.squeeze(0)

prediction = prediction.cpu()


print(
    "Prediction shape:",
    prediction.shape
)


# ============================================================
# Print classes found in image
# ============================================================

unique_classes = torch.unique(
    prediction
)

print("\nClasses detected:")

for class_id in unique_classes:

    class_id = class_id.item()

    print(
        f"{class_id}: {CLASSES[class_id]}"
    )


# ============================================================
# Color palette
# ============================================================

colors = [

    [0, 0, 0],          # No-data

    [255, 255, 255],    # Background

    [255, 0, 0],        # Building

    [128, 128, 128],    # Road

    [0, 0, 255],        # Water

    [210, 180, 140],    # Barren

    [0, 128, 0],        # Forest

    [255, 255, 0]       # Agriculture
]


colors = torch.tensor(
    colors,
    dtype=torch.uint8
)


# Convert class IDs → RGB

colored_mask = colors[
    prediction
].numpy()


# ============================================================
# Visualization
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 7)
)


# Original image

axes[0].imshow(
    original_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )
)

axes[0].set_title(
    "Input Satellite Image"
)

axes[0].axis("off")


# Segmentation

axes[1].imshow(
    colored_mask
)

axes[1].set_title(
    "Predicted Segmentation"
)

axes[1].axis("off")


# ============================================================
# Legend
# ============================================================

legend = []

for i, class_name in enumerate(CLASSES):

    legend.append(
        Patch(
            facecolor=[
                c / 255
                for c in colors[i].tolist()
            ],
            label=f"{i}: {class_name}"
        )
    )


axes[1].legend(
    handles=legend,
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)


plt.tight_layout()

plt.show()