from pathlib import Path

from datasets import load_dataset

dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")


# バッチ処理に対応したフラット化関数
def flatten_passages_batched(examples):
    # 出力用の新しいカラムに対応する空のリストを用意
    new_examples = {"query": [], "query_id": [], "answers": [], "passage_text": [], "is_selected": []}

    # バッチ内の各データ（複数形）をループ
    for i in range(len(examples["query"])):
        # 1つのクエリに対応する10個のpassageをループ
        for passage_text, is_selected in zip(
            examples["passages"][i]["passage_text"], examples["passages"][i]["is_selected"]
        ):
            # 新しいリストに各情報を追加していく
            new_examples["query"].append(examples["query"][i])
            new_examples["query_id"].append(examples["query_id"][i])
            new_examples["answers"].append(examples["answers"][i])
            new_examples["passage_text"].append(passage_text)
            new_examples["is_selected"].append(is_selected)

    return new_examples


# mapメソッドをbatched=Trueで実行
flattened_dataset_batched = dataset.map(
    flatten_passages_batched,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Flattening passages with batching",  # 進捗表示
)
pair_dataset = flattened_dataset_batched.filter(lambda x: x["is_selected"] == 1)
pair_dataset = pair_dataset.rename_columns({"query": "anc", "passage_text": "pos"}).remove_columns([
    "query_id",
    "answers",
    "is_selected",
])
pair_dataset.add_column("source", ["ms-marco" for _ in range(len(pair_dataset))])
print(pair_dataset)
Path("data/gte-en/ms-marco").mkdir(parents=True, exist_ok=True)
pair_dataset.save_to_disk("data/gte-en/ms-marco")
