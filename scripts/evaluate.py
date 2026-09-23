import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn


def main():
    data = torch.load("data/features.pt")
    ckpt = torch.load("models/head.pt")
    classes = ckpt["classes"]

    x = torch.cat([data["test"]["features"], data["test"]["sizes"]], dim=1)
    x = (x - ckpt["mean"]) / ckpt["std"]
    y = data["test"]["labels"]

    head = nn.Linear(x.shape[1], len(classes))
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    with torch.no_grad():
        pred = head(x).argmax(1)

    print(classification_report(y, pred, target_names=classes, digits=3, zero_division=0))

    cm = confusion_matrix(y, pred)
    errors = [
        (cm[i, j], classes[i], classes[j])
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]

    print("Top confusions (true -> predicted):")
    for count, true_name, pred_name in sorted(errors, reverse=True)[:15]:
        print(f"  {count:3d}  {true_name} -> {pred_name}")


if __name__ == "__main__":
    main()