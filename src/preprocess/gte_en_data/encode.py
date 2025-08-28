from datasets import Dataset,concatenate_datasets, load_from_disk
import argparse
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch
import json
import logging
import pandas as pd

# 共通ユーティリティをインポート
from src.preprocess.encode_utils import (
    stat_text_lengths, 
    flatten_dataset_batch, 
    encode_with_checkpoint,
    encode_with_checkpoint_multigpu,
    setup_multiprocess_pool,
    get_available_gpus,
    process_encoded_dataset,
    save_split_dataset  # 新しい関数を追加
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def main(args):
    # 出力パスとチェックポイントパスの設定
    args.output_dir = Path(args.output_dir)

    with open(args.output_dir / "dataset_summary.json", "r") as f:
        dataset_summary = json.load(f)
    print("use these subsets",dataset_summary.keys())

    subset_to_num_examples = {}
    subset_to_target_num_examples = {}
    for subset, info in dataset_summary.items():
        subset_to_num_examples[subset] = info["len"]
    total_num_examples = min(args.sample_size, sum(subset_to_num_examples.values()))
    down_sampling_ratio = args.sample_size / total_num_examples if total_num_examples > 0 else 0
    subset_to_target_num_examples = {subset: int(num_examples) * min(1.0, down_sampling_ratio) for subset, num_examples in subset_to_num_examples.items()}
    print(subset_to_target_num_examples)
    final_sample_size = int(sum(subset_to_target_num_examples.values()))
    output_path = args.output_dir / f"{args.teacher_model.replace('/', '_')}_encoded" / str(final_sample_size)
    checkpoint_dir = output_path / "checkpoints"

    print(f"Total target sample size: {final_sample_size}")
    exit()


    if args.w_instruction:
        with open(args.output_dir / "instruction.json", "r") as f:
            instruction = json.load(f)
    else:
        instruction = {data_name: "" for data_name in dataset_summary.keys()}

    encode_datasets=[]
    for subset in subset_to_target_num_examples.keys():
        dataset = load_from_disk(f"{args.output_dir}/{subset}")
        dataset = dataset.shuffle(seed=42).select(range(subset_to_target_num_examples[subset]))
        encode_dataset = dataset.map(
            flatten_dataset_batch, 
            with_indices=True, 
            remove_columns=dataset.column_names,
            num_proc=4,
            fn_kwargs={"subset": subset, "instruction": instruction[subset]},
            batched=True  # バッチ処理で効率化
        )
        logger.info(f"Loaded dataset: {subset} with {len(encode_dataset)} samples")
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
    
    # GPU数をチェック
    gpu_count = get_available_gpus()
    logger.info(f"Available GPUs: {gpu_count}")
    
    # マルチGPUが利用可能で、且つ無効化されていない場合はマルチGPUを使用
    use_multigpu = gpu_count > 1 and not args.disable_multigpu
    
    if use_multigpu:
        logger.info("Using multi-GPU encoding")
        pool = setup_multiprocess_pool(teacher_model)
    else:
        logger.info("Using single-GPU encoding")
        pool = None

    with torch.no_grad():
        logger.info("-- Check OOM --")
        teacher_model.encode(
            long_texts["text"][:args.long_batch_size*2],
            show_progress_bar=True, 
            batch_size=args.long_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
        )
        teacher_model.encode(
            short_texts["text"][:args.short_batch_size*2],
            show_progress_bar=True, 
            batch_size=args.short_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
        )
        
        logger.info("-- Long Encode --")
        # マルチGPUまたはシングルGPUでエンコーディング
        if use_multigpu:
            long_teacher_features = encode_with_checkpoint_multigpu(
                teacher_model,
                long_texts["text"],
                args.long_batch_size,
                checkpoint_dir / "long_checkpoint.pkl",
                args.max_length,
                pool
            )
        else:
            long_teacher_features = encode_with_checkpoint(
                teacher_model,
                long_texts["text"],
                args.long_batch_size,
                checkpoint_dir / "long_checkpoint.pkl",
                args.max_length
            )
        
        logger.info("-- Short Encode --")
        if use_multigpu:
            short_teacher_features = encode_with_checkpoint_multigpu(
                teacher_model,
                short_texts["text"],
                args.short_batch_size,
                checkpoint_dir / "short_checkpoint.pkl",
                args.max_length,
                pool
            )
        else:
            short_teacher_features = encode_with_checkpoint(
                teacher_model,
                short_texts["text"],
                args.short_batch_size,
                checkpoint_dir / "short_checkpoint.pkl",
                args.max_length
            )
    
    # マルチプロセスプールをクリーンアップ
    if pool is not None:
        teacher_model.stop_multi_process_pool(pool)
    
    # データサイズの整合性をチェック
    assert len(long_texts) == len(long_teacher_features), f"Long texts size mismatch: {len(long_texts)} vs {len(long_teacher_features)}"
    assert len(short_texts) == len(short_teacher_features), f"Short texts size mismatch: {len(short_texts)} vs {len(short_teacher_features)}"
    
    logger.info(f"Data size validation passed - Long: {len(long_texts)}, Short: {len(short_texts)}")
    
    # 共通関数を使用してデータセット処理
    all_features, reconstructed_dataset = process_encoded_dataset(
        long_texts, short_texts, long_teacher_features, short_teacher_features
    )
    # 分割保存を使用
    save_split_dataset(reconstructed_dataset, all_features, output_path)
    json.dump(stats, open(output_path / "stats.json", "w"), indent=4)

    logger.info(f"Processing completed. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print samples from a dataset.")
    parser.add_argument("--output_dir", type=str, default="data/gte-eng", help="Path to save the output directory")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-Embedding-4B", help="Path to the teacher model")
    parser.add_argument("--sample_size", type=int, default=1_000, help="Number of samples to load")
    parser.add_argument("--long_batch_size", type=int, default=1, help="Batch size for processing long texts")
    parser.add_argument("--short_batch_size", type=int, default=32, help="Batch size for processing short texts")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum length for tokenization")
    parser.add_argument("--threshold", type=int, default=2048, help="Threshold for distinguishing long and short texts 文字数なのに注意")
    parser.add_argument("--w_instruction", action="store_true", help="Include instruction in the input")
    parser.add_argument("--disable_multigpu", action="store_true", help="Disable multi-GPU processing")
    args = parser.parse_args()
    main(args)

