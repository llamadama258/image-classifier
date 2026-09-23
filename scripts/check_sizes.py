from pathlib import Path

from PIL import Image

root = Path("data/rebelo1")

for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
    sizes = [Image.open(p).size for p in sorted(class_dir.glob("*.png"))[:200]]
    ws = [w for w, h in sizes]
    hs = [h for w, h in sizes]
    rs = [w / h for w, h in sizes]
    print(
        f"{class_dir.name:26s} "
        f"w {min(ws):4d}-{max(ws):4d}  "
        f"h {min(hs):4d}-{max(hs):4d}  "
        f"w/h {min(rs):5.2f}-{max(rs):5.2f}"
    )