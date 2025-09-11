from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset, load_from_disk

UNUSED_SUBSET_DEFAULT = {
    # 元の triplet-eng/encode.py の除外リストに準拠（一部を残す）
    "altlex",
    "WikiAnswers",
    "S2ORC_citations_titles",
    "specter_train_triples",
    # "yahoo_answers_title_answer",  # は差し替えるので除外しない（固定側）
    # yahoo_answers_question_answer は今回使わないので除外に入れておく
    "yahoo_answers_question_answer",
    "yahoo_answers_title_question",
    "cnn_dailymail_splitted",
    "S2ORC_citations_abstracts",
    "msmarco-triples",
    "quora_duplicates",
    "quora_duplicates_triplets",
    "PAQ_pairs",
    "flickr30k_captions",
    # "agnews",
    "AllNLI",
    "NQ-train_pairs",
    "squad_pairs",
    "TriviaQA_pairs",
}


def to_triplet_shape_from_pairs(ds: Dataset, label_value: str = "") -> Dataset:
    def _map(ex):
        return {
            "anc": str(ex.get("anc", "") or ""),
            "pos": str(ex.get("pos", "") or ""),
            "neg": [],
            "label": str(label_value),
        }

    return ds.map(_map, remove_columns=ds.column_names, desc="to_triplet_shape")


def ensure_triplet_schema(ds: Dataset) -> Dataset:
    """
    anc,pos は string、neg は List[string]、label は string（null は空文字）に統一。
    """
    cols = set(ds.column_names)

    def _fix(ex):
        anc = str(ex.get("anc", "") or "")
        pos = str(ex.get("pos", "") or "")
        lab = ex.get("label", "")
        lab = "" if lab is None else str(lab)
        subset = str(ex.get("subset", "") or "")
        neg = ex.get("neg")
        if not isinstance(neg, list):
            neg = [] if neg is None else [str(neg)]
        else:
            neg = [str(x) for x in neg]
        return {"anc": anc, "pos": pos, "neg": neg, "label": lab, "subset": subset}

    # remove unknown columns to avoid concat feature conflicts
    remove_cols = [c for c in ds.column_names if c not in ("anc", "pos", "neg", "label", "subset")]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)
    ds = ds.map(_fix, desc="ensure_schema")
    # 明示的に features をキャストして型不一致を防ぐ
    target_features = Features({
        "anc": Value("string"),
        "pos": Value("string"),
        "neg": Sequence(feature=Value("string")),
        "label": Value("string"),
        "subset": Value("string"),
    })
    try:
        ds = ds.cast(target_features)
    except Exception as e:
        print(f"[schema] cast warning: {e}")
    return ds


def sample_dataset(ds: Dataset, n: int, seed: int) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def compute_plan(
    triplet_base_dir: str,
    replace_agnews_dir: str,
    replace_yahoo_title_answer_dir: str,
    total_target: int,
    exclude_subsets: set[str] | None,
) -> dict:
    """
    配分プランを算出して JSON として返す。
    - agnews / yahoo_answers_title_answer も含め、全サブセットで合計 total_target になるよう比例配分
    - agnews/yahoo は置換ソース（HF datasets）の実長を available として使用
    - 他 subset は triplet-eng の dataset_summary.json の len を available として使用
    """
    triplet_dir = Path(triplet_base_dir)
    with open(triplet_dir / "dataset_summary.json") as f:
        summary: dict[str, dict] = json.load(f)

    exclude = set(exclude_subsets or set())
    # 置換サブセット
    replaced = {
        "agnews": {"source": replace_agnews_dir, "source_type": "hf"},
        "yahoo_answers_title_answer": {"source": replace_yahoo_title_answer_dir, "source_type": "hf"},
    }

    allocs: dict[str, dict] = {}
    total_available = 0
    # 置換側の available
    for name, meta in replaced.items():
        if name in exclude:
            continue
        try:
            n = len(load_from_disk(meta["source"]))
        except Exception:
            n = 0
        n = min(n, 1_000_000)
        allocs[name] = {"len": n, "source": meta["source"], "source_type": meta["source_type"]}
        total_available += n

    # JSONL 側の available
    for subset, info in summary.items():
        if subset in exclude or subset in replaced:
            continue
        L = min(int(info.get("len", 0)), 1_000_000)
        if L <= 0:
            continue
        allocs[subset] = {"len": L, "source": str(triplet_dir / f"{subset}.jsonl"), "source_type": "jsonl"}
        total_available += L

    ratio = (total_target / total_available) if total_available > 0 else 0.0
    for subset, meta in allocs.items():
        meta["alloc"] = int(meta["len"] * ratio)

    plan = {
        "total_target": total_target,
        "total_available": total_available,
        "allocs": allocs,
        "excluded": sorted(list(exclude)),
        "summary": {
            "planned_total": sum(m["alloc"] for m in allocs.values()),
            "num_subsets": len(allocs),
        },
    }
    return plan


def build_triplet_mix_from_plan(plan: dict) -> tuple[Dataset, dict]:
    pieces: list[Dataset] = []
    counts: dict[str, int] = {}

    for subset, meta in plan["allocs"].items():
        alloc = int(meta.get("alloc", 0))
        if alloc <= 0:
            continue
        source = meta["source"]
        stype = meta.get("source_type", "jsonl")
        if stype == "hf":
            ds_src = load_from_disk(source)
            print(f"[mix] subset '{subset}' (hf): path={source}, avail={len(ds_src)}, alloc={alloc}")
            ds_part = sample_dataset(ds_src, alloc, seed=42)
            ds_part = ds_part.map(lambda ex: {"subset": subset})
            ds_part = ensure_triplet_schema(ds_part)
        else:
            print(f"[mix] subset '{subset}' (jsonl): source={source}, alloc={alloc}")
            ds_raw = load_dataset("json", data_files=source, split="train")
            ds_raw = sample_dataset(ds_raw, alloc, seed=42)
            ds_part = to_triplet_shape_from_pairs(ds_raw, label_value="")
            ds_part = ds_part.map(lambda ex: {"subset": subset})
            ds_part = ensure_triplet_schema(ds_part)
        pieces.append(ds_part)
        counts[subset] = len(ds_part)

    mixed = concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
    return mixed, counts


def build_with_gte(triplet_mix: Dataset, gte_subsets_dir: str, fever_dir: str | None = None) -> tuple[Dataset, dict]:
    parts: list[Dataset] = [triplet_mix]
    counts: dict[str, int] = {}

    # GTE 各 subset
    for name in sorted(os.listdir(gte_subsets_dir)):
        p = Path(gte_subsets_dir) / name
        if not p.is_dir():
            continue
        try:
            ds = load_from_disk(str(p))
        except Exception:
            continue
        print(f"[gte] subset '{name}': path={p}, len={len(ds)}")

        def _map(ex):
            return {
                "anc": ex.get("anc", ""),
                "pos": ex.get("pos", ""),
                "neg": ex.get("neg", []) or [],
                "label": "",
            }

        ds2 = ds.map(
            _map, remove_columns=[c for c in ds.column_names if c not in ("anc", "pos", "neg")], desc=f"gte_{name}"
        )
        ds2 = ds2.map(lambda ex: {"label": "", "subset": name})
        ds2 = ensure_triplet_schema(ds2)
        parts.append(ds2)
        counts[name] = len(ds2)

    # FEVER
    if fever_dir:
        try:
            ds_fever = load_from_disk(fever_dir)
            cols = set(ds_fever.column_names)
            print(f"[gte] fever: path={fever_dir}, len={len(ds_fever)}")
            if {"anc", "pos"}.issubset(cols):

                def _map_fv(ex):
                    return {
                        "anc": ex.get("anc", ""),
                        "pos": ex.get("pos", ""),
                        "neg": ex.get("neg", []) or [],
                        "label": "",
                    }

                ds_fever2 = ds_fever.map(
                    _map_fv,
                    remove_columns=[c for c in ds_fever.column_names if c not in ("anc", "pos", "neg")],
                    desc="fever",
                )
            else:

                def _map_fb(ex):
                    return {
                        "anc": ex.get("anc", "") or ex.get("query", ""),
                        "pos": ex.get("pos", "") or ex.get("document", ""),
                        "neg": ex.get("neg", []) or [],
                        "label": "",
                    }

                ds_fever2 = ds_fever.map(_map_fb, remove_columns=ds_fever.column_names, desc="fever_fb")
            ds_fever2 = ds_fever2.map(lambda ex: {"label": "", "subset": "fever"})
            ds_fever2 = ensure_triplet_schema(ds_fever2)
            parts.append(ds_fever2)
            counts["fever"] = len(ds_fever2)
        except Exception:
            pass
    full = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return full, counts


def main():
    ap = argparse.ArgumentParser(description="Build mixed triplet dataset with replacements and GTE concat")
    ap.add_argument("--triplet-base-dir", default="data/triplet-eng")
    ap.add_argument("--replace-agnews-dir", default="data/w_label_data/agnews/agnews_triplets_hf")
    ap.add_argument("--replace-yahoo-title-answer-dir", default="data/w_label_data/yahoo_answers/yahoo_triplets_hf")
    ap.add_argument("--total-target", type=int, default=1_000_000)
    ap.add_argument(
        "--gte-subsets-dir",
        default="/home/chihiro_yano/work/distill-embeddings/data/gte_plus-eng/gte/with_neg_noencode/794554/subsets",
    )
    ap.add_argument("--fever-dir", default="/home/chihiro_yano/work/distill-embeddings/data/gte_plus-eng/gte/fever")
    ap.add_argument("--out-dir", default="data/w_label_data/mixed_triplets_hf")
    ap.add_argument("--plan-out", default="data/w_label_data/mixed_plan.json")
    ap.add_argument("--result-summary-out", default="data/w_label_data/mixed_result_summary.json")
    ap.add_argument("--plan-only", action="store_true", help="計画JSONだけ出力して終了（重い処理なし）")
    args = ap.parse_args()

    exclude = set(UNUSED_SUBSET_DEFAULT)
    # 差し替え対象は固定側で扱うので除外セットから外す
    exclude.discard("agnews")
    exclude.discard("yahoo_answers_title_answer")

    print("[plan] computing allocation plan ...")
    plan = compute_plan(
        triplet_base_dir=args.triplet_base_dir,
        replace_agnews_dir=args.replace_agnews_dir,
        replace_yahoo_title_answer_dir=args.replace_yahoo_title_answer_dir,
        total_target=args.total_target,
        exclude_subsets=exclude,
    )

    os.makedirs(os.path.dirname(args.plan_out) or ".", exist_ok=True)
    with open(args.plan_out, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f"[plan] wrote {args.plan_out}")

    print(
        f"[plan] total_available={plan['total_available']} -> target={plan['total_target']} planned_total={plan['summary']['planned_total']}"
    )
    print(f"[plan] subsets in plan: {len(plan['allocs'])}")

    if args.plan_only:
        return

    print("[mix] building triplet mix from plan ...")
    mix, tri_counts = build_triplet_mix_from_plan(plan)
    print(f"[mix] mixed triplet rows: {len(mix)}")

    print("[gte] concatenating GTE subsets and FEVER ...")
    full, gte_counts = build_with_gte(mix, gte_subsets_dir=args.gte_subsets_dir, fever_dir=args.fever_dir)
    print(f"[out] final rows: {len(full)}")

    # サマリJSONを書き出し
    summary = {
        "triplet_counts": tri_counts,
        "triplet_total": sum(tri_counts.values()),
        "gte_counts": gte_counts,
        "gte_total": sum(gte_counts.values()),
        "final_total": len(full),
    }
    os.makedirs(os.path.dirname(args.result_summary_out) or ".", exist_ok=True)
    with open(args.result_summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[summary] wrote {args.result_summary_out}")

    os.makedirs(args.out_dir, exist_ok=True)
    full.save_to_disk(args.out_dir)
    print(f"[done] Saved combined dataset to {args.out_dir} with {len(full)} rows")


if __name__ == "__main__":
    main()
