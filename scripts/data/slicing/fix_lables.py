#!/usr/bin/env python3
"""
Relabel YOLO annotations from class id 1 → 0 and patch data.yml
Usage:  python fix_labels.py
"""
import pathlib, yaml

root = pathlib.Path("data/merged_sliced")

# 1) Rewrite all .txt files
for txt in root.rglob("*.txt"):
    with txt.open() as f:
        lines = [l.strip().split() for l in f if l.strip()]
    for line in lines:
        if line[0] == "1":
            line[0] = "0"
    txt.write_text("\n".join(" ".join(line) for line in lines) + "\n")

# 2) Patch data.yml
yaml_file = root / "data.yml"
data = yaml.safe_load(yaml_file.read_text())
if "1" in data.get("names", {}):
    data["names"]["0"] = data["names"].pop("1")
yaml_file.write_text(yaml.dump(data, sort_keys=False))

print("Done — all labels are now 0-indexed.")
