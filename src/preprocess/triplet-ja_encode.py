from datasets import Dataset, load_dataset, get_dataset_config_names,get_dataset_config_info,concatenate_datasets
import argparse
from sentence_transformers import SentenceTransformer
import random
from pathlib import Path
import torch
import numpy as np
import json
import logging
import pickle
import os
from tqdm import tqdm

UNUSED_SUBSET=["wordnet-ja-synonyms",
               "wordnet-ja-same-synsets",
                "word-ja-definitions",
                "ihyoki",
                "jawiki-paragraphs1",
                "jawiki-hyperlinks",
                "wiki-summary-title2text",
                "wiki-summary-text2text",
                "wiki-para-text2text",
                "wiki-para-title2text",
                "wiki-qa",
                ]

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

def save_checkpoint(features, batch_idx, checkpoint_path):
    """チェックポイントを保存"""
    checkpoint_data = {
        'features': features,
        'batch_idx': batch_idx
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Checkpoint saved: batch {batch_idx}, features shape: {np.array(features).shape}")

def load_checkpoint(checkpoint_path):
    """チェックポイントを読み込み"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint loaded: batch {data['batch_idx']}, features shape: {np.array(data['features']).shape}")
        return data['features'], data['batch_idx']
    return [], 0

def encode_with_checkpoint(model, texts, batch_size, checkpoint_path, max_length=None):
    """チェックポイント機能付きエンコーディング"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # チェックポイントから復元
    all_features, start_batch = load_checkpoint(checkpoint_path)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch}/{total_batches}")
    
    for batch_idx in tqdm(range(start_batch, total_batches), desc="Encoding", initial=start_batch, total=total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        batch_features = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            max_length=max_length,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_features.extend(batch_features)

        # 定期的にチェックポイント保存（1000バッチごと）
        if (batch_idx + 1) % 1000 == 0:
            save_checkpoint(all_features, batch_idx + 1, checkpoint_path)
    
    # 完了時にチェックポイントファイルを削除
    if checkpoint_path.exists():
        os.remove(checkpoint_path)
        logger.info("Processing completed. Checkpoint file removed.")
    
    return np.array(all_features)

def reconstruct_dataset(encoded_dataset):
    """
    エンコードされたデータセットを元のanc/pos形式に戻す
    """
    # IDでグループ化
    grouped = {}
    for i, example in enumerate(encoded_dataset):
        key = (example["subset"], example["id"])
        if key not in grouped:
            grouped[key] = {}
        
        column = example["column"]
        grouped[key][column] = {
            "text": example["text"],
            "teacher_features": example["teacher_features"]
        }
    
    # 元の形式に再構築
    reconstructed_data = {
        "anc": [],
        "pos": [], 
        "anc_features": [],
        "pos_features": [],
        "subset": [],
        "id": []
    }
    
    for (subset, example_id), columns in grouped.items():
        if "anc" in columns and "pos" in columns:
            reconstructed_data["anc"].append(columns["anc"]["text"])
            reconstructed_data["pos"].append(columns["pos"]["text"])
            reconstructed_data["anc_features"].append(columns["anc"]["teacher_features"])
            reconstructed_data["pos_features"].append(columns["pos"]["teacher_features"])
            reconstructed_data["subset"].append(subset)
            reconstructed_data["id"].append(example_id)
    
    return Dataset.from_dict(reconstructed_data)

def main(args):
    # 出力パスとチェックポイントパスの設定
    output_path = Path(args.output_dir) / f"{args.teacher_model.replace('/', '_')}_encoded" / (f"{args.sample_size}" if args.sample_size else 'full')
    checkpoint_dir = output_path / "checkpoints"
    
    # Load a dataset from the Hugging Face Hub
    configs = get_dataset_config_names(args.data_name)
    print(f"Available dataset configurations: {configs}")
    subset_to_num_examples = {}
    subset_to_target_num_examples = {}
    for subset in configs:
        if subset in UNUSED_SUBSET:
            continue
        dataset_config = get_dataset_config_info(
            args.data_name,
            subset,
        )
        subset_to_num_examples[subset] = dataset_config.splits["train"].num_examples
    total_num_examples = sum(subset_to_num_examples.values())
    down_sampling_ratio = args.sample_size / total_num_examples if total_num_examples > 0 else 0
    subset_to_target_num_examples = {subset: int(num_examples * down_sampling_ratio) for subset, num_examples in subset_to_num_examples.items()}
    print(subset_to_target_num_examples)
    encode_datasets=[]
    for subset in subset_to_target_num_examples.keys():
        dataset = load_dataset(args.data_name, subset, split='train')
        dataset = dataset.shuffle(seed=42).select(range(subset_to_target_num_examples[subset]))
        encode_dataset = dataset.map(
            flatten_dataset_batch, 
            with_indices=True, 
            remove_columns=dataset.column_names,
            num_proc=4,
            fn_kwargs={"subset": subset},
            batched=True  # バッチ処理で効率化
        )
        logger.info(f"Loaded dataset: {args.data_name} ({subset}) with {len(encode_dataset)} samples")
        encode_datasets.append(encode_dataset)
    encode_dataset = concatenate_datasets(encode_datasets)
    encode_dataset = encode_dataset.sort("len", reverse=True)
    # Print statistics about the distribution of text lengths
    stats = stat_text_lengths(encode_dataset["text"])
    logger.info(f"Text length statistics: {stats}")
    long_texts = encode_dataset.filter(lambda x: len(x["text"]) > args.threshold)
    short_texts = encode_dataset.filter(lambda x: len(x["text"]) <= args.threshold)
    teacher_model = SentenceTransformer(args.teacher_model).bfloat16()
    logger.info(f"Loaded teacher model: {args.teacher_model}")

    with torch.no_grad():
        logger.info("-- Check OOM --")
        teacher_model.encode(
            long_texts["text"][:2],
            show_progress_bar=True, 
            batch_size=args.long_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
        )
        teacher_model.encode(
            short_texts["text"][:32],
            show_progress_bar=True, 
            batch_size=args.short_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
        )
        logger.info("-- Long Encode --")
        # チェックポイント機能付きエンコーディング
        long_teacher_features = encode_with_checkpoint(
            teacher_model,
            long_texts["text"],
            args.long_batch_size,
            checkpoint_dir / "long_checkpoint.pkl",
            args.max_length
        )
        logger.info("-- Short Encode --")
        short_teacher_features = encode_with_checkpoint(
            teacher_model,
            short_texts["text"],
            args.short_batch_size,
            checkpoint_dir / "short_checkpoint.pkl",
            args.max_length
        )
    # add_columnの代わりにfrom_dictを使用
    long_dataset = Dataset.from_dict({
        **long_texts.to_dict(),
        'teacher_features': long_teacher_features
    })
    short_dataset = Dataset.from_dict({
        **short_texts.to_dict(),
        'teacher_features': short_teacher_features
    })
    output_dataset = concatenate_datasets([long_dataset, short_dataset])
    
    # 元の形式に再構築
    reconstructed_dataset = reconstruct_dataset(output_dataset)

    output_path.mkdir(parents=True, exist_ok=True)
    reconstructed_dataset.save_to_disk(output_path)
    json.dump(stats, open(output_path / "stats.json", "w"), indent=4)
    
    # チェックポイントディレクトリをクリーンアップ
    if checkpoint_dir.exists():
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    logger.info(f"Processing completed. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print samples from a dataset.")
    parser.add_argument("--data_name", type=str, default="cl-nagoya/ruri-dataset-v2-pt", help="Name of the HF dataset")
    parser.add_argument("--output_dir", type=str, default="data/triplet-ja", help="Path to save the output directory")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-Embedding-4B", help="Path to the teacher model")
    # parser.add_argument("--sample_size", type=int, default=1_000, help="Number of samples to load")
    parser.add_argument("--sample_size", type=int, default=1_000_000, help="Number of samples to load")
    parser.add_argument("--long_batch_size", type=int, default=1, help="Batch size for processing long texts")
    parser.add_argument("--short_batch_size", type=int, default=32, help="Batch size for processing short texts")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum length for tokenization")
    parser.add_argument("--threshold", type=int, default=4096, help="Threshold for distinguishing long and short texts 文字数なのに注意")
    args = parser.parse_args()
    main(args)

