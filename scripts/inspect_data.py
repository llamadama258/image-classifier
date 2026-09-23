from collections import Counter
from pathlib import Path

for root in [Path("data/rebelo1"), Path("data/rebelo2")]:
    counts = Counter(p.parent.name for p in root.rglob("*.png"))
    print(f"\n{root} — {len(counts)} classes, {sum(counts.values())} images")
    for name, n in counts.most_common():
        print(f"  {n:6d}  {name}")