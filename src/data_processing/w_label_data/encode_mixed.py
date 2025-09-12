import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.data_processing.encode_utils import (
    encode_with_checkpoint,
    get_available_gpus,
    save_split_dataset,
    stat_text_lengths,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_instructions(ds: Dataset, instruction_file: str | None, instruction_text: str | None) -> dict[str, str]:
    subsets = sorted(set(ds["subset"])) if "subset" in ds.column_names else ["mixed"]
    if instruction_file:
        with open(instruction_file, encoding="utf-8") as f:
            inst = json.load(f)
        # fill missing with empty string
        for s in subsets:
            if s not in inst:
                print(f"[WARN] instruction missing for subset={s}, use empty")
                inst[s] = ""
        return inst
    if instruction_text is not None:
        return dict.fromkeys(subsets, instruction_text)
    return dict.fromkeys(subsets, "")


def main(args: argparse.Namespace):
    # Load base dataset (HF datasets saved by build_mixed_triplets.py)
    base_path = Path(args.base_dir)
    ds = load_from_disk(str(base_path))
    logger.info(f"Loaded mixed dataset from {base_path} with {len(ds)} rows; columns={ds.column_names}")

    # Prepare output
    out_path = (
        Path(args.output_dir)
        / f"{args.teacher_model.replace('/', '_')}_encoded"
        / (f"{args.sample_size}" if args.sample_size else "full")
    )
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Instruction per subset
    inst_map = load_instructions(ds, args.instruction_file, args.instruction_text)
    logger.info(f"Instruction map loaded for {len(inst_map)} subsets")

    # Sample if needed to match sample_size (uniform over dataset order)
    if args.sample_size and args.sample_size < len(ds):
        ds = ds.shuffle(seed=42).select(range(args.sample_size))
        logger.info(f"Downsampled to {len(ds)} rows for encoding")

    # Flatten anc/pos/neg into encode records. anc にだけ instruction を付与。
    def flatten_with_neg(batch, indices, subset, instruction: str = ""):
        res = {"text": [], "subset": [], "id": [], "column": [], "slot": [], "label": [], "len": []}
        neg_list = batch.get("neg")
        for i, idx in enumerate(indices):
            # anc
            anc = batch.get("anc", [""])[i] if "anc" in batch else ""
            if instruction:
                anc = f"Instruct:{instruction}\nQuery:{anc}"
            res["text"].append(anc)
            res["subset"].append(subset)
            res["id"].append(idx)
            res["column"].append("anc")
            res["slot"].append(-1)
            res["label"].append(batch.get("label", [""])[i] if "label" in batch else "")
            res["len"].append(len(anc))
            # pos
            pos = batch.get("pos", [""])[i] if "pos" in batch else ""
            res["text"].append(pos)
            res["subset"].append(subset)
            res["id"].append(idx)
            res["column"].append("pos")
            res["slot"].append(-1)
            res["label"].append(batch.get("label", [""])[i] if "label" in batch else "")
            res["len"].append(len(pos))
            # negs
            if neg_list is not None and i < len(neg_list) and neg_list[i] is not None:
                for j, neg_text in enumerate(neg_list[i]):
                    if neg_text is None:
                        continue
                    res["text"].append(neg_text)
                    res["subset"].append(subset)
                    res["id"].append(idx)
                    res["column"].append("neg")
                    res["slot"].append(j)
                    res["label"].append(batch.get("label", [""])[i] if "label" in batch else "")
                    res["len"].append(len(neg_text))
        return res

    to_encode_parts = []
    if "subset" in ds.column_names:
        # 高速スプリット：subsetごとにfilterせず、一度だけindexを集計してselect
        from collections import defaultdict

        ds = ds.flatten_indices()  # ensure efficient select
        subs = ds["subset"]
        buckets = defaultdict(list)
        for i, s in enumerate(subs):
            buckets[s].append(i)
        for subset in sorted(buckets.keys()):
            sub_ds = ds.select(buckets[subset])
            enc_part = sub_ds.map(
                flatten_with_neg,
                with_indices=True,
                remove_columns=sub_ds.column_names,
                num_proc=args.num_proc,
                fn_kwargs={"subset": subset, "instruction": inst_map.get(subset, "")},
                batched=True,
                desc=f"flatten:{subset}",
            )
            to_encode_parts.append(enc_part)
    else:
        enc_part = ds.map(
            flatten_with_neg,
            with_indices=True,
            remove_columns=ds.column_names,
            num_proc=args.num_proc,
            fn_kwargs={"subset": "mixed", "instruction": inst_map.get("mixed", "")},
            batched=True,
            desc="flatten:mixed",
        )
        to_encode_parts.append(enc_part)

    enc_ds = concatenate_datasets(to_encode_parts) if len(to_encode_parts) > 1 else to_encode_parts[0]
    # Sort to encode long first (reduce fragmentation)
    # 'len' は flatten 時に作っているが、無いケースに備えて安全に追加
    if "len" not in enc_ds.column_names:
        enc_ds = enc_ds.add_column("len", [len(t) for t in enc_ds["text"]])
    enc_ds = enc_ds.sort("len", reverse=True)

    # Stats + split long/medium/short
    stats = stat_text_lengths(enc_ds["text"])
    logger.info(f"Text length stats: {stats}")

    # First split by character threshold for super-long texts
    long_texts = enc_ds.filter(lambda x: x["len"] > args.threshold)
    non_long = enc_ds.filter(lambda x: x["len"] <= args.threshold)

    # Medium bucket: texts whose tokenized length fits within medium_threshold (default: 128)
    # Use teacher tokenizer to measure token length without truncation
    logger.info("Tokenizing to compute token lengths for medium split...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=True)

    def add_tok_len(batch):
        toks = tokenizer(batch["text"], add_special_tokens=True, truncation=False)
        return {"tok_len": [len(ids) for ids in toks["input_ids"]]}

    non_long = non_long.map(add_tok_len, batched=True, batch_size=1024, desc="compute tok_len")
    medium_texts = non_long.filter(lambda x: x["tok_len"] <= args.medium_threshold)
    short_texts = non_long.filter(lambda x: x["tok_len"] > args.medium_threshold)
    # Drop helper column to keep schemas aligned for concatenation
    medium_texts = medium_texts.remove_columns([c for c in ["tok_len"] if c in medium_texts.column_names])
    short_texts = short_texts.remove_columns([c for c in ["tok_len"] if c in short_texts.column_names])

    logger.info(f"Long={len(long_texts)}, Medium={len(medium_texts)}, Short={len(short_texts)}")

    # Load teacher model (single GPU)
    model = SentenceTransformer(args.teacher_model).bfloat16()
    logger.info(f"Loaded model: {args.teacher_model}")
    logger.info(f"GPUs available: {get_available_gpus()}")

    # OOM check
    with torch.no_grad():
        logger.info("-- OOM Check --")
        if len(long_texts) > 0:
            model.encode(
                long_texts["text"][: args.long_batch_size * 2],
                show_progress_bar=True,
                batch_size=args.long_batch_size,
                max_length=args.max_length,
                normalize_embeddings=True,
            )
        if len(medium_texts) > 0:
            model.encode(
                medium_texts["text"][: args.medium_batch_size * 2],
                show_progress_bar=True,
                batch_size=args.medium_batch_size,
                max_length=args.max_length,
                normalize_embeddings=True,
            )
        if len(short_texts) > 0:
            model.encode(
                short_texts["text"][: args.short_batch_size * 2],
                show_progress_bar=True,
                batch_size=args.short_batch_size,
                max_length=args.max_length,
                normalize_embeddings=True,
            )

        # Encode with checkpoints (single GPU)
        logger.info("-- Long Encode --")
        long_feats = (
            encode_with_checkpoint(
                model,
                long_texts["text"],
                args.long_batch_size,
                checkpoint_dir / "long.pkl",
                args.max_length,
            )
            if len(long_texts) > 0
            else torch.empty((0, model.get_sentence_embedding_dimension()))
        )

        logger.info("-- Medium Encode --")
        medium_feats = (
            encode_with_checkpoint(
                model,
                medium_texts["text"],
                args.medium_batch_size,
                checkpoint_dir / "medium.pkl",
                args.max_length,
            )
            if len(medium_texts) > 0
            else torch.empty((0, model.get_sentence_embedding_dimension()))
        )

        logger.info("-- Short Encode --")
        short_feats = (
            encode_with_checkpoint(
                model,
                short_texts["text"],
                args.short_batch_size,
                checkpoint_dir / "short.pkl",
                args.max_length,
            )
            if len(short_texts) > 0
            else torch.empty((0, model.get_sentence_embedding_dimension()))
        )

    # Validate sizes
    assert len(long_texts) == len(long_feats), f"long size mismatch {len(long_texts)} vs {len(long_feats)}"
    assert len(medium_texts) == len(medium_feats), f"medium size mismatch {len(medium_texts)} vs {len(medium_feats)}"
    assert len(short_texts) == len(short_feats), f"short size mismatch {len(short_texts)} vs {len(short_feats)}"

    # Reconstruct back to anc/pos/neg grouped by (subset,id)
    import numpy as np

    if isinstance(long_feats, torch.Tensor):
        long_feats = long_feats.cpu().numpy()
    if isinstance(short_feats, torch.Tensor):
        short_feats = short_feats.cpu().numpy()
    parts_texts = []
    parts_feats = []
    if len(long_texts) > 0:
        parts_texts.append(long_texts)
        parts_feats.append(long_feats)
    if len(medium_texts) > 0:
        parts_texts.append(medium_texts)
        parts_feats.append(medium_feats)
    if len(short_texts) > 0:
        parts_texts.append(short_texts)
        parts_feats.append(short_feats)

    all_texts = concatenate_datasets(parts_texts) if len(parts_texts) > 1 else parts_texts[0]
    all_features = np.vstack(parts_feats) if len(parts_feats) > 1 else parts_feats[0]
    assert len(all_texts) == len(all_features), f"Combined data size mismatch: {len(all_texts)} vs {len(all_features)}"
    all_texts = all_texts.add_column(name="emb_idx", column=[i for i in range(len(all_texts))])

    grouped = {}
    for ex in all_texts:
        key = (ex["subset"], ex["id"])
        rec = grouped.setdefault(key, {"neg": {}})
        col = ex["column"]
        if col == "anc":
            rec["anc"] = {"text": ex["text"], "emb_idx": ex["emb_idx"], "label": ex.get("label", "")}
        elif col == "pos":
            rec["pos"] = {"text": ex["text"], "emb_idx": ex["emb_idx"], "label": ex.get("label", "")}
        elif col == "neg":
            slot = ex.get("slot", 0) or 0
            rec["neg"][int(slot)] = {"text": ex["text"], "emb_idx": ex["emb_idx"]}

    out = {
        "anc": [],
        "pos": [],
        "neg": [],
        "label": [],
        "anc_emb_idx": [],
        "pos_emb_idx": [],
        "neg_emb_idx": [],
        "subset": [],
        "id": [],
        "neg_num": [],
    }
    incomplete = 0
    for (subset, _id), rec in grouped.items():
        if "anc" not in rec or "pos" not in rec:
            incomplete += 1
            continue
        out["anc"].append(rec["anc"]["text"])
        out["pos"].append(rec["pos"]["text"])
        out["anc_emb_idx"].append(rec["anc"]["emb_idx"])
        out["pos_emb_idx"].append(rec["pos"]["emb_idx"])
        # label は anc か pos のどちらかから
        out["label"].append(str(rec["anc"].get("label", "") or rec["pos"].get("label", "")))
        # neg は slot 順
        if rec["neg"]:
            neg_slots = sorted(rec["neg"].keys())
            neg_texts = [rec["neg"][j]["text"] for j in neg_slots]
            neg_idx = [rec["neg"][j]["emb_idx"] for j in neg_slots]
        else:
            neg_texts, neg_idx = [], []
        out["neg"].append(neg_texts)
        out["neg_emb_idx"].append(neg_idx)
        out["neg_num"].append(len(neg_texts))
        out["subset"].append(subset)
        out["id"].append(_id)
    if incomplete:
        logger.warning(f"Found {incomplete} incomplete items (missing anc or pos)")
    recon_ds = Dataset.from_dict(out)

    # Save
    save_split_dataset(recon_ds, all_features, out_path)
    with open(out_path / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved encoded dataset to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Encode mixed triplets (anc/pos) with teacher model (single GPU)")
    p.add_argument("--base-dir", default="data/w_label_data/mixed_triplets_hf", help="HF dataset path to load")
    p.add_argument("--output-dir", default="data/w_label_data/mixed_encoded", help="Output base directory")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--sample-size", type=int, default=None, help="Optional cap of rows to encode")
    p.add_argument("--threshold", type=int, default=2048, help="Split boundary by char length")
    p.add_argument("--long-batch-size", type=int, default=1)
    p.add_argument("--medium-batch-size", type=int, default=None, help="Batch size for medium texts (<=128 tokens)")
    p.add_argument("--short-batch-size", type=int, default=32)
    # Max seq length used during encoding for all buckets
    p.add_argument("--max-length", type=int, default=None, help="Maximum token length used in encode()")
    # Medium threshold controls only the split; keep deprecated alias for compat
    p.add_argument("--medium-threshold", type=int, default=128, help="Token threshold for medium bucket split")
    p.add_argument("--medium-max-length", type=int, default=None, help="[Deprecated] Use --medium-threshold instead")
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument(
        "--instruction-file", default="data/w_label_data/instruction.json", help="JSON mapping: subset -> instruction"
    )
    p.add_argument("--instruction-text", default=None, help="Single instruction string for all subsets")
    args = p.parse_args()
    # Normalization for deprecated alias
    if args.medium_max_length is not None:
        args.medium_threshold = args.medium_max_length
    # Default medium batch size to short if not provided
    if args.medium_batch_size is None:
        args.medium_batch_size = args.short_batch_size
    main(args)
