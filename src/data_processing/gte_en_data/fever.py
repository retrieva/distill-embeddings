from datasets import load_dataset, Dataset
from pathlib import Path

# データセットの読み込みは同じ
dataset:Dataset = load_dataset("mteb/fever", "default", split="train")
queries:Dataset = load_dataset("mteb/fever", "queries", split="queries")
corpus:Dataset = load_dataset("mteb/fever", "corpus", split="corpus")

# --- 修正点 1: IDとテキストの対応辞書を事前に作成 ---
print("Creating dictionaries for queries and corpus...")
# クエリID -> クエリテキスト の辞書を作成
query_id_to_text = {row['_id']: row['text'] for row in queries}

# コーパスID -> コーパステキスト の辞書を作成
# titleとtextをスペースで連結
corpus_id_to_text = {row['_id']: row['title'] + " " + row['text'] for row in corpus}
print("Dictionaries created.")
valid_query_ids = set(query_id_to_text.keys())
valid_corpus_ids = set(corpus_id_to_text.keys())

# 2. 有効なIDペアを持つ行だけを残すようにフィルタリング
filtered_dataset = dataset.filter(
    lambda example: example["query-id"] in valid_query_ids and example["corpus-id"] in valid_corpus_ids,
    num_proc=4
)

# --- 修正点 2: 辞書を使って高速にテキストを取得する関数を定義 ---
def merge_using_dicts(examples):
    # examples["query_id"]のリストを元に、辞書から対応するテキストを順番に取得
    anc_texts = [query_id_to_text[qid] for qid in examples["query-id"]]
    
    # examples["corpus_id"]のリストを元に、辞書から対応するテキストを順番に取得
    pos_texts = [corpus_id_to_text[cid] for cid in examples["corpus-id"]]
    
    return {"anc": anc_texts, "pos": pos_texts}

# --- 修正点 3: 新しい関数を使ってmapを適用 ---
merged_dataset = filtered_dataset.map(
    merge_using_dicts,
    batched=True,
    num_proc=4,
    remove_columns=filtered_dataset.column_names
)

print(merged_dataset)
# 出力例:
# Dataset({
#     features: ['anc', 'pos'],
#     num_rows: 91102
# })
print(merged_dataset[0])
# 出力例:
# {
#  'anc': 'does jimmy carter have brothers and sisters',
#  'pos': 'Jimmy Carter He is the eldest of four children. His siblings were sister Gloria (1926–1990), sister Ruth (1929–1983), and brother Billy (1937–1988).'
# }
merged_dataset.add_column("source", ["fever" for _ in range(len(merged_dataset))])
Path("data/gte-en/fever").mkdir(parents=True, exist_ok=True)
merged_dataset.save_to_disk("data/gte-en/fever")