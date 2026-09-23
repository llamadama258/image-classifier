import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from image_classifier.model import load_backbone
from image_classifier.transforms import symbol_transform

BATCH_SIZE = 64


class SymbolDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        tensor = symbol_transform(Image.open(record["path"]))
        size = torch.tensor(
            [
                record["width"] / 100.0,
                record["height"] / 100.0,
                record["width"] / record["height"],
            ],
            dtype=torch.float32,
        )
        return tensor, size, record["label"]


def extract(records, backbone, device):
    loader = DataLoader(SymbolDataset(records), batch_size=BATCH_SIZE, num_workers=4)
    feats, sizes, labels = [], [], []

    with torch.no_grad():
        for images, size_batch, label_batch in loader:
            feats.append(backbone(images.to(device)).flatten(1).cpu())
            sizes.append(size_batch)
            labels.append(label_batch)

    return torch.cat(feats), torch.cat(sizes), torch.cat(labels)


def main():
    index = json.loads(Path("data/index.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    backbone = load_backbone().to(device)

    data = {"classes": index["classes"]}
    for split in ["train", "test"]:
        features, sizes, labels = extract(index[split], backbone, device)
        data[split] = {"features": features, "sizes": sizes, "labels": labels}
        print(f"{split}: features {tuple(features.shape)}  labels {tuple(labels.shape)}")

    torch.save(data, Path("data/features.pt"))
    print("\nwrote data/features.pt")


if __name__ == "__main__":
    main()