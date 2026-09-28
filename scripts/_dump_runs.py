"""Print a one-line summary of every run's test metrics."""
from __future__ import annotations

import json
from pathlib import Path

rows = []
for d in sorted(Path("runs").iterdir()):
    if not d.is_dir():
        continue
    m = d / "metrics.json"
    if not m.exists():
        rows.append((d.name, "--", "--", "--", "--", "--"))
        continue
    j = json.loads(m.read_text(encoding="utf-8"))
    rows.append(
        (
            d.name,
            f"{100 * j.get('test_accuracy', 0):.2f}",
            f"{100 * j.get('test_macro_f1', 0):.2f}",
            str(j.get("pooling")),
            str(j.get("leak_free")),
            str(j.get("seed")),
        )
    )

w = max(len(r[0]) for r in rows)
print(f"{'tag'.ljust(w)}  acc    f1     pool   leak   seed")
for r in rows:
    print(f"{r[0].ljust(w)}  {r[1]:<6} {r[2]:<6} {r[3]:<6} {r[4]:<6} {r[5]}")