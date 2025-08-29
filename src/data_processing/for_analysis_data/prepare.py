from pathlib import Path

from datasets import load_dataset

data = load_dataset("graelo/wikipedia", "20230601.en", split="train", trust_remote_code=True)
data = data.shuffle(seed=42).select(range(10000))
data = data.map(lambda x: {"text": x["title"].strip() + " " + x["text"].strip()})
data = data.remove_columns([c for c in data.column_names if c != "text"])

Path("data/anly-wiki/en").mkdir(parents=True, exist_ok=True)
data.save_to_disk("data/anly-wiki/en")
