import argparse
import re
from collections import defaultdict

import mteb
import pandas as pd
import yaml
from mteb.leaderboard.table import create_tables


def _safe_task_name(t) -> str | None:
    meta = getattr(t, "metadata", None)
    if meta is not None and getattr(meta, "name", None):
        return meta.name
    for attr in ("name", "task_name"):
        if getattr(t, attr, None):
            return getattr(t, attr)
    # Fallback: class name
    return t.__class__.__name__ if t is not None else None


def _safe_task_type(t) -> str | None:
    meta = getattr(t, "metadata", None)
    for obj in (meta, t):
        if obj is None:
            continue
        for attr in ("type", "task_type", "category"):
            val = getattr(obj, attr, None)
            if val is None:
                continue
            if isinstance(val, str) and val:
                return val
            # Enum-like value
            v = getattr(val, "value", None)
            if isinstance(v, str) and v:
                return v
    return None


def _strip_markdown_model(val: str) -> str:
    # Convert "[Name](url)" -> "Name"
    if not isinstance(val, str):
        return str(val)
    m = re.match(r"\[([^\]]+)\]\([^\)]+\)$", val)
    return m.group(1) if m else val


def main():
    parser = argparse.ArgumentParser(description="MTEB/JMTEB評価スコアを取得")
    parser.add_argument(
        "--lang", "-l", choices=["jpn", "eng"], default="jpn", help="評価言語を選択 (jpn: 日本語, eng: 英語)"
    )
    parser.add_argument("--benchmark_name", type=str, default="on_eval_tasks", help="Name of the benchmark.")
    parser.add_argument(
        "--wandb_style",
        action="store_true",
        help="wandb_aggスタイルの列構成で出力 (code_name/run_name, macro/micro, avg_カテゴリ, タスク列)",
    )
    parser.add_argument(
        "--task_sort",
        choices=["category_alpha", "alpha", "none"],
        default="category_alpha",
        help="タスク列の並び順 (wandb_aggと揃えるならcategory_alpha)",
    )
    parser.add_argument(
        "--drop_tasks",
        type=str,
        default="",
        help="除外するタスク名をカンマ区切りで指定 (例: Touche2020Retrieval.v3,ToxicConversationsClassification)",
    )
    args = parser.parse_args()

    models = [
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-8B",
    ]

    if args.lang == "eng":
        models.extend([
            "nomic-ai/modernbert-embed-base",
            "nomic-ai/modernbert-embed-base-unsupervised",
            "Alibaba-NLP/gte-modernbert-base",
            "lightonai/modernbert-embed-large-unsupervised",
            "lightonai/modernbert-embed-large",
        ])

    if args.benchmark_name in ["on_eval_tasks", "on_train_end_tasks", "on_train_tasks"]:
        with open("tasks.yaml") as file:
            tasks = yaml.safe_load(file)
            tasks = tasks[args.lang][args.benchmark_name]
    elif args.benchmark_name in ["MTEB(eng, v2)"]:
        benchmark = mteb.get_benchmark(args.benchmark_name)
        tasks = benchmark.tasks
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
    print(f"評価言語: {args.lang}")
    print(f"タスク数: {len(tasks)}")

    scores = mteb.load_results(models=models, tasks=tasks)
    # Convert scores into long format
    scores_long = scores.get_scores(format="long")

    # Convert scores into leaderboard tables
    summary_gr_df, per_task_gr_df = create_tables(scores_long=scores_long)

    # Convert Gradio DataFrames to Pandas
    summary_df = pd.DataFrame(summary_gr_df.value["data"], columns=summary_gr_df.value["headers"])
    per_task_df = pd.DataFrame(per_task_gr_df.value["data"], columns=per_task_gr_df.value["headers"])

    if not args.wandb_style:
        scores_df = pd.concat([summary_df, per_task_df], axis=1)
        output_file = f"output/loaded_{'jmteb' if args.lang == 'jpn' else 'mteb'}_score.csv"
        scores_df.to_csv(output_file, index=False)
        print(f"結果を保存しました: {output_file}")
        return

    # wandb_aggスタイルの整形
    # 1) タスク名の集合をベンチマークから取得し、タスク->カテゴリの対応を構築
    benchmark = mteb.get_benchmark(args.benchmark_name)
    task_names = []
    task_to_cat = {}
    for t in getattr(benchmark, "tasks", []) or []:
        n = _safe_task_name(t)
        ty = _safe_task_type(t)
        if isinstance(n, str) and n:
            task_names.append(n)
            if isinstance(ty, str) and ty:
                task_to_cat[n] = ty

    # 2) per_task_df から必要なタスク列 + Model 列のみ抽出
    # per_task_df には "Model" 列と各タスク列が含まれる
    drop_set = {s.strip() for s in (args.drop_tasks.split(",") if args.drop_tasks else []) if s.strip()}
    keep_cols = [c for c in per_task_df.columns if (c == "Model" or c in set(task_names)) and c not in drop_set]
    pt = per_task_df[keep_cols].copy()
    # Model列のMarkdownリンク表現をプレーン名に
    pt["code_name"] = pt["Model"].map(_strip_markdown_model)
    pt["run_name"] = pt["code_name"]
    pt = pt.drop(columns=["Model"], errors="ignore")

    # 3) 数値化（欠損はNaNのまま）
    for c in task_names:
        if c in pt.columns:
            pt[c] = pd.to_numeric(pt[c], errors="coerce")

    # 4) カテゴリ平均、micro/macro平均を計算
    # カテゴリ別列
    cat_to_cols: dict[str, list[str]] = defaultdict(list)
    for tname in task_names:
        cat = task_to_cat.get(tname)
        if cat is not None and tname in pt.columns:
            cat_to_cols[cat].append(tname)

    # wandb_aggと同じカテゴリ順
    category_order = [
        "Retrieval",
        "Reranking",
        "Classification",
        "PairClassification",
        "STS",
        "Clustering",
        "Summarization",
        "Other",
    ]
    avg_cols: list[str] = []
    for cat in category_order:
        cols = cat_to_cols.get(cat, [])
        if not cols:
            continue
        avg_col = f"avg_{cat}"
        pt[avg_col] = pt[cols].astype(float).mean(axis=1, skipna=True)
        avg_cols.append(avg_col)

    # micro: 全タスクの平均, macro: カテゴリ平均の平均
    existing_task_cols = [c for c in task_names if c in pt.columns]
    pt["micro_ave"] = pt[existing_task_cols].astype(float).mean(axis=1, skipna=True) if existing_task_cols else float("nan")
    pt["macro_ave"] = pt[avg_cols].astype(float).mean(axis=1, skipna=True) if avg_cols else float("nan")

    # 5) 列順を決定（wandb_agg互換）
    # デフォルトはカテゴリ→アルファベット順（カテゴリ順は上の固定順）
    if args.task_sort == "alpha":
        sorted_task_cols = sorted(existing_task_cols)
    elif args.task_sort == "none":
        sorted_task_cols = existing_task_cols
    else:  # category_alpha
        sorted_task_cols = []
        for cat in category_order:
            cols = sorted([c for c in existing_task_cols if task_to_cat.get(c) == cat])
            if cols:
                sorted_task_cols.extend(cols)
        # どのカテゴリにも属さないものがあれば最後にアルファベット順で
        remaining = [c for c in existing_task_cols if c not in set(sorted_task_cols)]
        sorted_task_cols.extend(sorted(remaining))

    left_cols = ["macro_ave", "micro_ave"] + avg_cols
    ordered_cols = ["code_name", "run_name"] + left_cols + [c for c in sorted_task_cols if c not in left_cols]
    # 重複削除
    seen = set()
    ordered_cols = [c for c in ordered_cols if (c not in seen and not seen.add(c))]
    # 存在しない列は落とす
    ordered_cols = [c for c in ordered_cols if c in pt.columns]
    pt = pt[ordered_cols]

    # 6) 保存
    out_dir = "wandb_agg_output"
    out_file = f"{out_dir}/leaderboard_metrics_{args.lang}.csv"
    import os
    os.makedirs(out_dir, exist_ok=True)
    pt.to_csv(out_file, index=False)
    print(f"wandb_aggスタイルで保存しました: {out_file}")


if __name__ == "__main__":
    main()
