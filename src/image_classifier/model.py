from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from image_classifier.transforms import symbol_transform

HEAD_PATH = Path("models/head.pt")


@lru_cache(maxsize=1)
def load_backbone():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval()
    return backbone


@lru_cache(maxsize=1)
def load_head():
    ckpt = torch.load(HEAD_PATH, map_location="cpu")
    head = nn.Linear(ckpt["mean"].shape[1], len(ckpt["classes"]))
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    return head, ckpt["classes"], ckpt["mean"], ckpt["std"]


def predict(image: Image.Image) -> dict:
    backbone = load_backbone()
    head, classes, mean, std = load_head()

    width, height = image.size
    batch = symbol_transform(image).unsqueeze(0)

    with torch.no_grad():
        features = backbone(batch).flatten(1)
        sizes = torch.tensor(
            [[width / 100.0, height / 100.0, width / height]],
            dtype=torch.float32,
        )
        x = (torch.cat([features, sizes], dim=1) - mean) / std
        probs = head(x).softmax(dim=1)[0]

    score, index = probs.max(dim=0)
    return {"label": classes[index], "confidence": round(score.item(), 4)}