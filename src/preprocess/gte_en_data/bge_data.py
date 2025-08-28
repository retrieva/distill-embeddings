"""
NQ, Trivia, Hotpot, SQuAD, NLI for SimCSEはbgeの訓練データとして、まとめて公開されていたので利用する。
"""
import pandas as pd
from datasets import Dataset
from pathlib import Path

raw_data_dir = Path("/home/chihiro_yano/work/distill-embeddings/data/gte-en/raw")

for data_name in raw_data_dir.iterdir():
    data_df = []
    for data_path in data_name.iterdir():
        data = pd.read_json(data_path, lines=True, orient="records")
        data_df.append(data)
    data_df = pd.concat(data_df, ignore_index=True)
    data_df["source"] = data_name.name
    data_df["pos"] = data_df["pos"].apply(lambda x: x[0] if type(x) == list else x)
    data_df = data_df[["source", "query", "pos"]].rename(columns={"query": "anc"}).sample(frac=1).reset_index(drop=True)
    dataset = Dataset.from_pandas(data_df)
    dataset.save_to_disk(raw_data_dir.parent / data_name.name)