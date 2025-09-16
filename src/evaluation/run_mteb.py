import argparse
import re
from collections import defaultdict

import mteb
import pandas as pd
import yaml
from mteb.encoder_interface import PromptType
from mteb.leaderboard.table import create_tables
from mteb.load_results.benchmark_results import ModelResult

from src.utils import load_model


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


def main(args):
    model_name = args.model_name
    model, output_folder = load_model(model_name=model_name, return_output_folder=True)
    if args.benchmark_name in ["on_eval_tasks", "on_train_end_tasks", "on_train_tasks"]:
        with open("tasks.yaml") as file:
            tasks = yaml.safe_load(file)
            tasks = tasks[args.language][args.benchmark_name]
    elif args.benchmark_name in ["MTEB(eng, v2)"]:
        benchmark = mteb.get_benchmark(args.benchmark_name)
        tasks = benchmark.tasks
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")

    # Optionally restrict to a subset of tasks by exact name
    only_tasks_str: str = getattr(args, "only_tasks", "") or ""
    if only_tasks_str.strip():
        only_set = {s.strip() for s in only_tasks_str.split(",") if s.strip()}
        name_to_task = {}
        resolved = []
        for t in tasks:
            n = _safe_task_name(t)
            if isinstance(n, str) and n:
                name_to_task[n] = t
        for name in only_set:
            t = name_to_task.get(name)
            if t is not None:
                resolved.append(t)
        if not resolved:
            raise ValueError(
                f"No tasks matched in --only_tasks. Provided: {sorted(only_set)}.\n"
                f"Available examples include: {sorted(list(name_to_task.keys()))[:10]} ..."
            )
        tasks = resolved

    if args.add_prefix:
        model_prompts = {
            PromptType.query.value: "query: ",
            PromptType.passage.value: "document: ",
        }
        model.prompts = model_prompts
    evaluation = mteb.MTEB(tasks=tasks, task_langs=[args.language])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # MTEBの評価を実行
        scores = evaluation.run(
            model,
            output_folder=output_folder,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            trust_remote_code=True,
            verbosity=1,
            overwrite_results=args.overwrite_results,
        )
    scores_long = ModelResult(
        model_name=model_name,
        model_revision=model.model_card_data.base_model_revision,
        task_results=scores,
    ).get_scores(format="long")

    # Convert scores into leaderboard tables
    summary_gr_df, per_task_gr_df = create_tables(scores_long=scores_long)
    # Convert Gradio DataFrames to Pandas
    summary_df = pd.DataFrame(summary_gr_df.value["data"], columns=summary_gr_df.value["headers"])
    per_task_df = pd.DataFrame(per_task_gr_df.value["data"], columns=per_task_gr_df.value["headers"])

    # Always write the original wide table for compatibility
    scores = pd.concat([summary_df, per_task_df], axis=1)
    scores.to_csv(output_folder / f"{args.language}_{args.benchmark_name}_scores.csv", index=False)

    # Optionally emit wandb_agg-style output with consistent ordering
    if args.wandb_style:
        # Build task -> category mapping from the benchmark definition
        bench = mteb.get_benchmark(args.benchmark_name)
        task_names = []
        task_to_cat = {}
        for t in getattr(bench, "tasks", []) or []:
            n = _safe_task_name(t)
            ty = _safe_task_type(t)
            if isinstance(n, str) and n:
                task_names.append(n)
                if isinstance(ty, str) and ty:
                    task_to_cat[n] = ty

        drop_set = {s.strip() for s in (args.drop_tasks.split(",") if args.drop_tasks else []) if s.strip()}
        keep_cols = [c for c in per_task_df.columns if (c == "Model" or c in set(task_names)) and c not in drop_set]
        pt = per_task_df[keep_cols].copy()
        pt["code_name"] = pt["Model"].map(_strip_markdown_model)
        pt["run_name"] = _strip_markdown_model(args.model_name)
        pt = pt.drop(columns=["Model"], errors="ignore")

        # Numeric conversion for task columns
        for c in task_names:
            if c in pt.columns:
                pt[c] = pd.to_numeric(pt[c], errors="coerce")

        # Category averages
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

        existing_task_cols = [c for c in task_names if c in pt.columns]
        pt["micro_ave"] = pt[existing_task_cols].astype(float).mean(axis=1, skipna=True) if existing_task_cols else float("nan")
        pt["macro_ave"] = pt[avg_cols].astype(float).mean(axis=1, skipna=True) if avg_cols else float("nan")

        # Ordering
        if args.task_sort == "alpha":
            sorted_task_cols = sorted(existing_task_cols)
        elif args.task_sort == "none":
            sorted_task_cols = existing_task_cols
        else:  # category_alpha (default)
            sorted_task_cols = []
            for cat in category_order:
                cols = sorted([c for c in existing_task_cols if task_to_cat.get(c) == cat])
                if cols:
                    sorted_task_cols.extend(cols)
            remaining = [c for c in existing_task_cols if c not in set(sorted_task_cols)]
            sorted_task_cols.extend(sorted(remaining))

        left_cols = ["macro_ave", "micro_ave"] + avg_cols
        ordered_cols = ["code_name", "run_name"] + left_cols + [c for c in sorted_task_cols if c not in left_cols]
        seen = set()
        ordered_cols = [c for c in ordered_cols if (c not in seen and not seen.add(c))]
        ordered_cols = [c for c in ordered_cols if c in pt.columns]
        pt = pt[ordered_cols]

        pt.to_csv(output_folder / f"{args.language}_{args.benchmark_name}_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTEB evaluation on a model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Name of the model to evaluate.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="eng",
        choices=["eng", "jpn"],
        help="Language of the model.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--benchmark_name", type=str, default="on_train_end_tasks", help="Name of the benchmark.")
    # Parse bools from strings like "true"/"false"
    parser.add_argument(
        "--add_prefix",
        type=lambda x: str(x).lower() in ("true", "1", "t", "y", "yes"),
        default=True,
        help="Whether to add prefix to the input.",
    )
    parser.add_argument(
        "--wandb_style",
        action="store_true",
        help="wandb_aggスタイルの列構成で追加出力 (code_name/run_name, macro/micro, avg_カテゴリ, タスク列)",
    )
    parser.add_argument(
        "--overwrite_results",
        type=lambda x: str(x).lower() in ("true", "1", "t", "y", "yes"),
        default=False,
        help="既存のタスクJSONを上書きして再評価する (false ならキャッシュを流用)",
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
    parser.add_argument(
        "--only_tasks",
        type=str,
        default="",
        help="このリストに含まれるタスクのみを実行（カンマ区切り・完全一致）",
    )
    args = parser.parse_args()
    main(args)
