import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk

base_dataset = load_from_disk("data/gte_plus-eng/triplet/Qwen_Qwen3-Embedding-4B_encoded/1000000")
subsets = list(set(base_dataset["subset"]))
raw_data_dir = "data/gte_plus-eng/triplet/raw"
output_dir = "data/gte_plus-eng/triplet/Qwen_Qwen3-Embedding-4B_encoded/1000000/with_neg"

Path(output_dir).mkdir(parents=True, exist_ok=True)


def concat_negs_from_dict(subset: str, subset_data: Dataset):
    original_data = pd.read_json(f"{raw_data_dir}/{subset}.jsonl", orient="records", lines=True)
    base_data = subset_data.to_pandas()
    concat_data = pd.merge(base_data, original_data, left_on=["anc", "pos"], right_on=["query", "pos"], how="left")
    return Dataset.from_pandas(concat_data)


def concat_negs_from_list(subset: str, subset_data: Dataset):
    original_data = pd.read_json(f"{raw_data_dir}/{subset}.jsonl", orient="records", lines=True, typ="series")
    original_data = original_data.apply(lambda x: tuple(x) if isinstance(x, list) else x)
    base_data = subset_data.to_pandas()
    concat_data = pd.merge(
        base_data, original_data.to_frame(name="set"), left_on=["anc", "pos"], right_on="set", how="left"
    )
    concat_data = concat_data.drop(columns=["set"])
    return Dataset.from_pandas(concat_data)


final_dataset = []
summary = {}
for subset in subsets:
    subset_data = base_dataset.filter(lambda example, subset=subset: example["subset"] == subset)
    with open(f"{raw_data_dir}/{subset}.jsonl", encoding="utf-8") as f:
        first_line = json.loads(f.readline().strip())
    if isinstance(first_line, dict):
        data_keys = set(first_line.keys())
        if data_keys == {"set"} or data_keys == {"query", "pos"}:
            print(f"{subset}: no negs")
            summary[subset] = {"min": 0, "max": 0, "mean": 0}
            concat_data = subset_data.add_column("neg", [""] * len(subset_data))
        elif data_keys == {"query", "pos", "neg"}:
            concat_data = concat_negs_from_dict(subset, subset_data)
            neg_lens = [len(neg) for neg in concat_data["neg"] if isinstance(neg, list)]
            summary[subset] = {"min": min(neg_lens), "max": max(neg_lens), "mean": sum(neg_lens) / len(neg_lens)}
            print(
                f"{subset}: has negs, min: {summary[subset]['min']}, max: {summary[subset]['max']}, mean: {summary[subset]['mean']}"
            )
        else:
            raise ValueError(f"Unexpected data keys: {data_keys}")
    elif isinstance(first_line, list | tuple):
        if len(first_line) <= 2:
            print(f"{subset}: no negs")
            summary[subset] = {"min": 0, "max": 0, "mean": 0}
            concat_data = subset_data.add_column("neg", [""] * len(subset_data))
        else:
            print(f"{subset}: has negs")
            concat_data = concat_negs_from_list(subset, subset_data)
            neg_lens = [len(neg) for neg in concat_data["neg"] if isinstance(neg, list)]
            summary[subset] = {"min": min(neg_lens), "max": max(neg_lens), "mean": sum(neg_lens) / len(neg_lens)}
            print(
                f"{subset}: has negs, min: {summary[subset]['min']}, max: {summary[subset]['max']}, mean: {summary[subset]['mean']}"
            )
    else:
        raise ValueError(f"Unexpected data format: {type(first_line)}")
    final_dataset.append(concat_data)
final_dataset = concatenate_datasets(final_dataset)
final_dataset.save_to_disk(output_dir)
print(f"Saved to {output_dir}")
with open(f"{output_dir}/neg_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)
"""
Pairs: ["text1", "text2"] - This is a positive pair that should be close in vector space.
Triplets: ["anchor", "positive", "negative"] - This is a triplet: The positive text should be close to the anchor, while the negative text should be distant to the anchor.
Sets: {"set": ["text1", "text2", ...]} A set of texts describing the same thing, e.g. different paraphrases of the same question, different captions for the same image. Any combination of the elements is considered as a positive pair.
Query-Pairs: {"query": "text", "pos": ["text1", "text2", ...]} A query together with a set of positive texts. Can be formed to a pair ["query", "positive"] by randomly selecting a text from pos.
Query-Triplets: {"query": "text", "pos": ["text1", "text2", ...], "neg": ["text1", "text2", ...]} A query together with a set of positive texts and negative texts. Can be formed to a triplet ["query", "positive", "negative"] by randomly selecting a text from pos and neg.
"""
