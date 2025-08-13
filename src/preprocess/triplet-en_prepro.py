import argparse
import numpy as np
import json
import logging
import os
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def stat_text_lengths(texts):
    lengths = [len(text) for text in texts]
    return {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "25th_percentile": float(np.percentile(lengths, 25)),
        "75th_percentile": float(np.percentile(lengths, 75))
    }

def flatten_dataset_batch(batch, indices, subset):
    results = {"text": [], "subset": [], "id": [], "column": [], "len": []}
    
    for i, idx in enumerate(indices):
        for column in ["anc", "pos"]:
            text = batch[column][i] if column in batch else ""
            results["text"].append(text)
            results["subset"].append(subset)
            results["id"].append(idx)
            results["column"].append(column)
            results["len"].append(len(text))
    
    return results

def main(args):
    # Load a dataset from the Hugging Face Hub
    raw_data_dir = f"{args.output_dir}/raw"
    dataset_summary = {}
    
    for path in os.listdir(raw_data_dir):
        if path.endswith(".jsonl"):
            data_name = path.split(".")[0]
            
            # 1. 一度に全ファイルを読み込む代わりに、ストリーミング処理
            # 2. pandas DataFrame の代わりに辞書のリストで処理
            output_data = []
            
            with open(f"{raw_data_dir}/{path}", 'r', encoding='utf-8') as f:
                first_line = json.loads(f.readline().strip())
                f.seek(0)  # ファイルの先頭に戻る
                
                # データ形式を判定
                if isinstance(first_line, dict):
                    data_keys = set(first_line.keys())
                    
                    # 3. バッチ処理でメモリ効率を改善
                    batch_size = 10000
                    batch = []
                    
                    for line in f:
                        item = json.loads(line.strip())
                        batch.append(item)
                        
                        if len(batch) >= batch_size:
                            output_data.extend(process_batch(batch, data_keys))
                            batch = []
                        if len(output_data) > 1_000_000:
                            logger.warning(f"多すぎるので終わり！")
                            break
                    # 残りのバッチを処理
                    if batch:
                        output_data.extend(process_batch(batch, data_keys))
                        
                elif isinstance(first_line, (list, tuple)):
                    for line in f:
                        item = json.loads(line.strip())
                        processed_item = {
                            "anc": item[0], 
                            "pos": item[1], 
                            **({"neg": item[2]} if len(item) > 2 else {})
                        }
                        output_data.append(processed_item)
                else:
                    raise ValueError(f"Unexpected data format: {type(first_line)}")
            
            # 4. pandas DataFrame作成を最後に1回だけ
            output_df = pd.DataFrame(output_data)
            
            dataset_summary[data_name] = {
                "len": len(output_df),
                "keys": list(output_df.columns),  # data.keys() ではなく columns を使用
                "sample": output_df.head(10).to_dict(orient="records")
            }
            
            logger.info(f"Processed {data_name}: {len(output_df)} records")
            if len(output_df) > 1_000_000:
                output_df = output_df.sample(n=1_000_000, random_state=42)
            output_df.to_json(f"{args.output_dir}/{data_name}.jsonl", orient="records", lines=True, force_ascii=False)

    # 5. 一度だけファイルに書き込み
    with open(f"{args.output_dir}/dataset_summary.json", "w", encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
    
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))

def process_batch(batch, data_keys):
    """バッチ処理用の関数"""
    processed_batch = []
    
    if data_keys == {"set"}:
        for item in batch:
            set_items = item["set"]
            for i in range(len(set_items) // 2):
                processed_batch.append({
                    "anc": set_items[i],
                    "pos": set_items[i+1],
                })
    elif data_keys == {"query", "pos"}:
        for item in batch:
            anc = item["query"]
            for pos in item["pos"]:
                processed_batch.append({
                    "anc": anc, 
                    "pos": pos, 
                })
    elif data_keys == {"query", "pos", "neg"}:
        for item in batch:
            anc = item["query"]
            pos_list = item["pos"]
            neg_list = item["neg"]
            for i, pos in enumerate(pos_list):
                processed_batch.append({
                    "anc": anc, 
                    "pos": pos, 
                    "neg": neg_list[min(i, len(neg_list)-1)]
                })
    else:
        raise ValueError(f"Unexpected keys in data: {data_keys}")
    
    return processed_batch

'''
Pairs: ["text1", "text2"] - This is a positive pair that should be close in vector space.
Triplets: ["anchor", "positive", "negative"] - This is a triplet: The positive text should be close to the anchor, while the negative text should be distant to the anchor.
Sets: {"set": ["text1", "text2", ...]} A set of texts describing the same thing, e.g. different paraphrases of the same question, different captions for the same image. Any combination of the elements is considered as a positive pair.
Query-Pairs: {"query": "text", "pos": ["text1", "text2", ...]} A query together with a set of positive texts. Can be formed to a pair ["query", "positive"] by randomly selecting a text from pos.
Query-Triplets: {"query": "text", "pos": ["text1", "text2", ...], "neg": ["text1", "text2", ...]} A query together with a set of positive texts and negative texts. Can be formed to a triplet ["query", "positive", "negative"] by randomly selecting a text from pos and neg.
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print samples from a dataset.")
    parser.add_argument("--output_dir", type=str, default="data/triplet-en", help="Path to save the output directory")
    args = parser.parse_args()
    main(args)