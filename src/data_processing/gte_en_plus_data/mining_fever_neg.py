import argparse
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

MODEL_NAME = os.environ.get("MINING_MODEL", "BAAI/bge-small-en-v1.5")
BATCH_SIZE = 2048
NEG_K = 10
EXTRA_MARGIN = 40
DEFAULT_OUTPUT_DIR = Path("data/gte-en-plus/gte/raw/fever")
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def load_fever():
    print("Loading FEVER datasets (mteb/fever)...")
    dataset = load_dataset("mteb/fever", "default", split="train")
    queries = load_dataset("mteb/fever", "queries", split="queries")
    corpus = load_dataset("mteb/fever", "corpus", split="corpus")
    print("Loaded.")
    return dataset, queries, corpus


def build_mappings(dataset, queries, corpus):
    query_id_to_text: dict[str, str] = {r["_id"]: r["text"] for r in queries}
    corpus_id_to_text: dict[str, str] = {r["_id"]: (r["title"] + " " + r["text"]).strip() for r in corpus}
    valid_query_ids = set(query_id_to_text.keys())
    valid_corpus_ids = set(corpus_id_to_text.keys())

    # 2. 有効なIDペアを持つ行だけを残すようにフィルタリング
    filtered_dataset = dataset.filter(
        lambda example: example["query-id"] in valid_query_ids and example["corpus-id"] in valid_corpus_ids, num_proc=4
    )
    qid_to_pos_ids: dict[str, set[str]] = defaultdict(set)
    for row in filtered_dataset:
        qid_to_pos_ids[row["query-id"]].add(row["corpus-id"])

    target_query_ids = [qid for qid in qid_to_pos_ids if qid in query_id_to_text]

    query_text_to_union_pos_ids: dict[str, set[str]] = defaultdict(set)
    for qid in target_query_ids:
        q_text = query_id_to_text[qid]
        query_text_to_union_pos_ids[q_text].update(qid_to_pos_ids[qid])

    return (
        query_id_to_text,
        corpus_id_to_text,
        qid_to_pos_ids,
        target_query_ids,
        query_text_to_union_pos_ids,
    )


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()))

    embs = []
    total_batches = math.ceil(len(texts) / batch_size)
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Embedding"):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,  # 外側のtqdmを使う
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embs.append(emb)
    return np.vstack(embs)


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def mine_hard_negatives(
    model: SentenceTransformer,
    query_id_to_text: dict[str, str],
    corpus_id_to_text: dict[str, str],
    qid_to_pos_ids: dict[str, set[str]],
    target_query_ids: list[str],
    corpus_embeddings: np.ndarray,
    index,
    query_text_to_union_pos_ids: dict[str, set[str]],
):
    corpus_ids = list(corpus_id_to_text.keys())
    corpus_id_array = np.array(corpus_ids)

    print("Embedding queries...")
    query_texts = [query_id_to_text[qid] for qid in target_query_ids]
    query_embeddings = embed_texts(model, query_texts, BATCH_SIZE)

    max_union_pos = max(len(query_text_to_union_pos_ids[qt]) for qt in query_texts)
    top_k_search = NEG_K + max_union_pos + EXTRA_MARGIN
    print(f"FAISS search top_k={top_k_search}")
    distances, indices = index.search(query_embeddings, top_k_search)

    records = []
    for row_idx, qid in enumerate(target_query_ids):
        q_text = query_id_to_text[qid]
        pos_ids = qid_to_pos_ids[qid]

        pos_texts = [corpus_id_to_text[cid] for cid in sorted(pos_ids)]

        union_pos_ids = query_text_to_union_pos_ids[q_text]

        candidate_ids = corpus_id_array[indices[row_idx]]

        neg_texts: list[str] = []
        seen: set[str] = set()
        for cid in candidate_ids:
            if cid in union_pos_ids:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            neg_texts.append(corpus_id_to_text[cid])
            if len(neg_texts) >= NEG_K:
                break

        records.append({
            "query": q_text,
            "pos": pos_texts,
            "neg": neg_texts,
            "source": "fever",
        })
    return records


def apply_sampling(
    corpus_id_to_text: dict[str, str],
    qid_to_pos_ids: dict[str, set[str]],
    query_id_to_text: dict[str, str],
    target_query_ids: list[str],
    corpus_sample: int,
    query_sample: int,
    seed: int,
):
    random.seed(seed)

    # 1. 先にクエリをサンプリング
    if query_sample > 0 and query_sample < len(target_query_ids):
        target_query_ids = target_query_ids[:query_sample]
        print(f"[Sampling] queries -> {len(target_query_ids)}")

    # 2. そのクエリが参照する positive corpus を必ず保持
    required_pos_ids: set[str] = set()
    for qid in target_query_ids:
        required_pos_ids.update(qid_to_pos_ids[qid])

    if corpus_sample > 0:
        if len(required_pos_ids) > corpus_sample:
            print(
                f"[Warning] corpus_sample({corpus_sample}) < 必要positive数({len(required_pos_ids)}). "
                "必要分に拡張します。"
            )
            corpus_sample = len(required_pos_ids)

        # 追加で negative 候補を入れる余地
        all_ids_sorted = sorted(corpus_id_to_text.keys())
        remaining_ids = [cid for cid in all_ids_sorted if cid not in required_pos_ids]

        need_extra = corpus_sample - len(required_pos_ids)
        if need_extra > 0:
            if len(remaining_ids) <= need_extra:
                extra_ids = remaining_ids
            else:
                extra_ids = random.sample(remaining_ids, need_extra)
        else:
            extra_ids = []

        keep_ids = required_pos_ids.union(extra_ids)
        corpus_id_to_text = {cid: corpus_id_to_text[cid] for cid in keep_ids}
        print(
            f"[Sampling] corpus -> {len(corpus_id_to_text)} "
            f"(positives kept={len(required_pos_ids)}, added negatives={len(extra_ids)})"
        )

    # 3. positive が消えたクエリを再チェック（基本必ず残るはずだが安全策）
    filtered_target = []
    new_qid_to_pos_ids: dict[str, set[str]] = {}
    for qid in target_query_ids:
        filtered_pos = {cid for cid in qid_to_pos_ids[qid] if cid in corpus_id_to_text}
        if filtered_pos:
            new_qid_to_pos_ids[qid] = filtered_pos
            filtered_target.append(qid)
    target_query_ids = filtered_target
    qid_to_pos_ids = new_qid_to_pos_ids

    # 4. union 再計算
    query_text_to_union_pos_ids: dict[str, set[str]] = defaultdict(set)
    for qid in target_query_ids:
        q_text = query_id_to_text[qid]
        query_text_to_union_pos_ids[q_text].update(qid_to_pos_ids[qid])

    return corpus_id_to_text, qid_to_pos_ids, target_query_ids, query_text_to_union_pos_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-sample", type=int, default=0, help="実験用: コーパス目標サンプル数 (0=全件)")
    parser.add_argument("--query-sample", type=int, default=0, help="実験用: クエリID先頭N (0=全件)")
    parser.add_argument("--seed", type=int, default=42, help="サンプリング乱数シード")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="出力ディレクトリ")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, queries, corpus = load_fever()
    (
        query_id_to_text,
        corpus_id_to_text,
        qid_to_pos_ids,
        target_query_ids,
        query_text_to_union_pos_ids,
    ) = build_mappings(dataset, queries, corpus)
    print(f"#total query ids (with positives): {len(target_query_ids)}")
    print(f"#total corpus docs: {len(corpus_id_to_text)}")
    if args.corpus_sample > 0 or args.query_sample > 0:
        (
            corpus_id_to_text,
            qid_to_pos_ids,
            target_query_ids,
            query_text_to_union_pos_ids,
        ) = apply_sampling(
            corpus_id_to_text,
            qid_to_pos_ids,
            query_id_to_text,
            target_query_ids,
            args.corpus_sample,
            args.query_sample,
            args.seed,
        )

    print(f"#query ids (with positives): {len(target_query_ids)}")
    print(f"#corpus docs: {len(corpus_id_to_text)}")

    if not target_query_ids:
        print("クエリがありません。終了。")
        return

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding corpus...")
    corpus_ids = list(corpus_id_to_text.keys())
    corpus_texts = [corpus_id_to_text[cid] for cid in corpus_ids]
    print(len(corpus_texts), "texts to embed")
    corpus_embeddings = embed_texts(model, corpus_texts, BATCH_SIZE)

    print("Building FAISS index...")
    index = build_faiss_index(corpus_embeddings)

    records = mine_hard_negatives(
        model,
        query_id_to_text,
        corpus_id_to_text,
        qid_to_pos_ids,
        target_query_ids,
        corpus_embeddings,
        index,
        query_text_to_union_pos_ids,
    )

    suffix = []
    if args.corpus_sample > 0:
        suffix.append(f"c{args.corpus_sample}")
    if args.query_sample > 0:
        suffix.append(f"q{args.query_sample}")
    suffix_str = ".".join(suffix) if suffix else "full"
    out_file = output_dir / f"mining_fever_neg.{suffix_str}.jsonl"

    pd.DataFrame(records).to_json(out_file, lines=True, orient="records", force_ascii=False)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
