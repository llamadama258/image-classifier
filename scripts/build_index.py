import json
import random
from pathlib import Path

from PIL import Image

ROOTS = [Path("data/rebelo1"), Path("data/rebelo2")]
MIN_PER_CLASS = 50
DROP = {"Other"}
SEED = 1
TEST_FRACTION = 0.2

paths_by_class: dict[str, list[Path]] = {}
for root in ROOTS:
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        paths_by_class.setdefault(class_dir.name, []).extend(
            sorted(class_dir.glob("*.png"))
        )

keep = sorted(
    name
    for name, paths in paths_by_class.items()
    if len(paths) >= MIN_PER_CLASS and name not in DROP
)

records = []
for label_index, name in enumerate(keep):
    for path in paths_by_class[name]:
        width, height = Image.open(path).size
        records.append(
            {
                "path": str(path),
                "label": label_index,
                "class_name": name,
                "width": width,
                "height": height,
            }
        )

random.Random(SEED).shuffle(records)
cut = int(len(records) * (1 - TEST_FRACTION))
train, test = records[:cut], records[cut:]

Path("data/index.json").write_text(
    json.dumps({"classes": keep, "train": train, "test": test})
)

print(f"kept {len(keep)} classes, dropped {len(paths_by_class) - len(keep)}")
print(f"{len(train)} train, {len(test)} test\n")

for name in keep:
    n_train = sum(1 for r in train if r["class_name"] == name)
    n_test = sum(1 for r in test if r["class_name"] == name)
    print(f"  {name:26s} train {n_train:5d}   test {n_test:4d}")