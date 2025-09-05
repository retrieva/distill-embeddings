from collections import Counter

from datasets import load_from_disk

base_dataset = load_from_disk("data/gte_plus-eng/triplet/Qwen_Qwen3-Embedding-4B_encoded/1000000")


def show_subset_counts(ds):
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        print("DatasetDict 内訳")
        for name, sub in ds.items():
            print(f"[{name}] 件数: {len(sub)}")
            if "subset" in sub.column_names:
                cnt = Counter(sub["subset"])
                print("  subset 列の内訳:")
                for k, v in cnt.most_common():
                    print(f"    {k}: {v}")
    else:
        print(f"単一 Dataset 件数: {len(ds)}")
        if "subset" in ds.column_names:
            cnt = Counter(ds["subset"])
            print("subset 列 内訳:")
            for k, v in cnt.most_common():
                print(f"  {k}: {v}")
        else:
            print("subset 列は存在しません。列一覧:", ds.column_names)


show_subset_counts(base_dataset)
