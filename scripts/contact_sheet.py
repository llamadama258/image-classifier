from pathlib import Path

from PIL import Image

from image_classifier.transforms import pad_to_square

root = Path("data/rebelo1")
out = Path("scratch")
out.mkdir(exist_ok=True)

classes = sorted(d for d in root.iterdir() if d.is_dir())
cell, cols = 64, 10

sheet = Image.new("RGB", (cols * cell, len(classes) * cell), "white")

for row, class_dir in enumerate(classes):
    print(f"row {row:2d}: {class_dir.name}")
    for col, path in enumerate(sorted(class_dir.glob("*.png"))[:cols]):
        img = pad_to_square(Image.open(path)).convert("RGB").resize((cell, cell))
        sheet.paste(img, (col * cell, row * cell))

sheet.save(out / "contact_sheet_padded.png")
print("\nwrote scratch/contact_sheet_padded.png")