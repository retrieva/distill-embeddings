import json
from pathlib import Path

from datasets import load_from_disk

data_dir = Path("data/gte-eng")
summary_dict = {}
for data_path in data_dir.iterdir():
    try:
        dataset = load_from_disk(data_path)
        summary_dict[data_path.name] = {"len": len(dataset), "keys": dataset.column_names, "samples": dataset[:5]}
    except Exception as e:
        print(f"Failed to load {data_path}: {e}")
with open(data_dir / "dataset_summary.json", "w") as f:
    json.dump(summary_dict, f, indent=4)
