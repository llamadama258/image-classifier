from pathlib import Path

import torch
from torch import nn

EPOCHS = 400
LR = 1e-3
SEED = 1


def load_split(data, split, device):
    d = data[split]
    x = torch.cat([d["features"], d["sizes"]], dim=1).to(device)
    return x, d["labels"].to(device)


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load("data/features.pt")
    classes = data["classes"]

    train_x, train_y = load_split(data, "train", device)
    test_x, test_y = load_split(data, "test", device)

    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True) + 1e-6
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    head = nn.Linear(train_x.shape[1], len(classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        head.train()
        optimizer.zero_grad()
        loss = loss_fn(head(train_x), train_y)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 50 == 0:
            head.eval()
            with torch.no_grad():
                train_acc = (head(train_x).argmax(1) == train_y).float().mean()
                test_acc = (head(test_x).argmax(1) == test_y).float().mean()
            print(
                f"epoch {epoch:4d}  loss {loss.item():.3f}  "
                f"train {train_acc:.3f}  test {test_acc:.3f}"
            )

    Path("models").mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "classes": classes,
            "mean": mean.cpu(),
            "std": std.cpu(),
        },
        "models/head.pt",
    )
    print("\nwrote models/head.pt")


if __name__ == "__main__":
    main()