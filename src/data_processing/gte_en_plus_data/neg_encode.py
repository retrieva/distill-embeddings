import argparse
import os
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_from_disk
from sentence_transformers import SentenceTransformer

from src.data_processing.encode_utils import (
    encode_with_checkpoint,
    encode_with_checkpoint_multigpu,
    get_available_gpus,
    setup_multiprocess_pool,
)


def unique_preserve_order(iterable):
    seen = set()
    out = []
    for x in iterable:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/gte_plus-eng/gte/with_neg_noencode/794554")
    parser.add_argument("--output_path", type=str, default="data/gte_plus-eng/gte/with_neg")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=["en_NLI_data", "HotpotQA", "NQ", "Trivia", "SQuAD", "ms-marco"],
        help="load_from_disk(base_path/subsets/<subset>) から読み込み",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--threshold", type=int, default=2048, help="長文/短文の閾値（文字数）")
    parser.add_argument("--long_batch_size", type=int, default=8)
    parser.add_argument("--short_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--disable_multigpu", action="store_true")
    args = parser.parse_args()

    base_path = args.base_path
    output_path = Path(args.output_path)
    checkpoint_dir = output_path / "checkpoints"
    os.makedirs(output_path, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    data_list = []
    for s in args.subsets:
        data_list.append(load_from_disk(f"{base_path}/subsets/{s}"))
    full_dataset = concatenate_datasets(data_list)

    # 列名の自動判定（negs or neg）
    if "negs" in full_dataset.column_names:
        negs_col = "negs"
    elif "neg" in full_dataset.column_names:
        negs_col = "neg"
    else:
        raise KeyError("Dataset does not contain 'negs' or 'neg' column.")

    # ユニークな neg テキストを順序保持で抽出
    all_negs = []
    for negs in full_dataset[negs_col]:
        if isinstance(negs, list):
            all_negs.extend(negs)
    texts = unique_preserve_order(all_negs)

    print(f"  Total unique neg texts to encode: {len(texts)}")

    # 長さで分割
    lengths = [len(t) for t in texts]
    long_indices = [i for i, L in enumerate(lengths) if L > args.threshold]
    short_indices = [i for i, L in enumerate(lengths) if L <= args.threshold]
    long_texts = [texts[i] for i in long_indices]
    short_texts = [texts[i] for i in short_indices]
    print(f"  Long texts: {len(long_texts)} | Short texts: {len(short_texts)} (threshold={args.threshold})")

    # モデル・GPU設定
    model = SentenceTransformer(args.model)
    gpu_count = get_available_gpus()
    use_multigpu = gpu_count > 1 and not args.disable_multigpu
    pool = None
    if use_multigpu:
        print(f"  Using multi-GPU: {gpu_count} GPUs")
        pool = setup_multiprocess_pool(model)
    else:
        print("  Using single GPU/CPU")

    # エンコード（チェックポイント付き）
    if len(long_texts) > 0:
        if use_multigpu:
            long_embs = encode_with_checkpoint_multigpu(
                model,
                long_texts,
                args.long_batch_size,
                checkpoint_dir / "neg_long.pkl",
                max_length=args.max_length,
                pool=pool,
            )
        else:
            long_embs = encode_with_checkpoint(
                model, long_texts, args.long_batch_size, checkpoint_dir / "neg_long.pkl", max_length=args.max_length
            )
    else:
        long_embs = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    if len(short_texts) > 0:
        if use_multigpu:
            short_embs = encode_with_checkpoint_multigpu(
                model,
                short_texts,
                args.short_batch_size,
                checkpoint_dir / "neg_short.pkl",
                max_length=args.max_length,
                pool=pool,
            )
        else:
            short_embs = encode_with_checkpoint(
                model, short_texts, args.short_batch_size, checkpoint_dir / "neg_short.pkl", max_length=args.max_length
            )
    else:
        short_embs = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    if pool is not None:
        model.stop_multi_process_pool(pool)

    # 元の順序に再配置
    emb_dim = (
        (long_embs.shape[1] if long_embs.size else short_embs.shape[1])
        if (long_embs.size or short_embs.size)
        else model.get_sentence_embedding_dimension()
    )
    embs = np.zeros((len(texts), emb_dim), dtype=np.float32)
    if len(long_indices) > 0:
        for dst_i, src_i in enumerate(long_indices):
            embs[src_i] = long_embs[dst_i]
    if len(short_indices) > 0:
        for dst_i, src_i in enumerate(short_indices):
            embs[src_i] = short_embs[dst_i]

    # id 付与
    text_to_id = {text: idx for idx, text in enumerate(texts)}
    neg_id_col = []
    for negs in full_dataset[negs_col]:
        if isinstance(negs, list):
            neg_id_col.append([text_to_id[n] for n in negs])
        else:
            neg_id_col.append([])
    full_dataset = full_dataset.add_column("neg_emb_idx", neg_id_col)

    # 保存
    full_dataset.save_to_disk(output_path.as_posix())
    np.savez_compressed((output_path / "neg_embeddings.npz").as_posix(), embeddings=embs)
    print(f"  Saved dataset to: {output_path}")
    print(f"  Saved embeddings to: {output_path / 'neg_embeddings.npz'}")


if __name__ == "__main__":
    main()
