from functools import lru_cache

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


@lru_cache(maxsize=1)
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    return model, weights


def predict(image: Image.Image) -> dict:
    model, weights = load_model()
    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(batch)

    probs = logits.softmax(dim=1)[0]
    score, index = probs.max(dim=0)
    label = weights.meta["categories"][index]

    return {"label": label, "confidence": round(score.item(), 4)}