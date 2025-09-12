import os
from collections import defaultdict

import mteb
import pandas as pd
import yaml  # yamlをインポート

import wandb
from src.utils import get_code_name

with open("src/aggregate/agg_config.yaml") as f:
    config = yaml.safe_load(f)

project = config["project"]
filters = config["filters"]
group_code_name = config["group_code_name"]
drop_tasks = set(config.get("drop_tasks", []) or [])

api = wandb.Api()
print(f"Fetching runs with filters: {filters}")
runs = api.runs(project, filters=filters)

all_summaries = []
for run in runs:
    summary = run.summary
    config = run.config
    code_name = get_code_name(config)

    if summary:
        mteb_final_data = {}
        for k, v in summary.items():
            if k.startswith("mteb_final/"):
                clean_key = k.replace("mteb_final/", "")
                mteb_final_data[clean_key] = v

        if mteb_final_data:
            mteb_final_data["run_name"] = run.name
            mteb_final_data["code_name"] = code_name
            all_summaries.append(mteb_final_data)

if all_summaries:
    combined_df = pd.DataFrame(all_summaries)
    output_dir = "wandb_agg_output"
    os.makedirs(output_dir, exist_ok=True)

    # Build task_name -> category map from MTEB benchmark metadata
    overrides: dict[str, str] = config.get("task_category_overrides", {}) if isinstance(config, dict) else {}
    benchmark_name: str = config.get("benchmark_name", "MTEB(eng, v2)")

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

    mteb_task_type_map: dict[str, str] = {}
    try:
        bench = mteb.get_benchmark(benchmark_name)
        for t in getattr(bench, "tasks", []) or []:
            n = _safe_task_name(t)
            ty = _safe_task_type(t)
            if isinstance(n, str) and isinstance(ty, str) and n and ty:
                mteb_task_type_map[n] = ty
    except Exception:
        # If MTEB loading fails, we will fall back to heuristics below
        mteb_task_type_map = {}

    def infer_task_category(task_name: str) -> str:
        # Priority: explicit overrides > MTEB metadata > heuristics
        if task_name in overrides:
            return overrides[task_name]
        if task_name in mteb_task_type_map:
            return mteb_task_type_map[task_name]
        n = task_name.lower()
        # Heuristic fallback (kept as safety net)
        if "reranking" in n:
            return "Reranking"
        if "retrieval" in n:
            return "Retrieval"
        if "clustering" in n:
            return "Clustering"
        if "classification" in n:
            return "Classification"
        if "summarization" in n or "summeval" in n:
            return "Summarization"
        if "sts" in n or "sick" in n or "biosses" in n:
            return "STS"
        if any(
            alias in n
            for alias in {
                "arguana",
                "askubuntudupquestions",
                "fiqa2018",
                "scidocs",
                "treccovid",
                "twitterurlcorpus",
                "hotpotqahardnegatives",
                "feverhardnegatives",
                "climatefeverhardnegatives",
                "touche2020",
            }
        ):
            return "Retrieval"
        if "duplicate" in n or "quora" in n:
            return "PairClassification"
        return "Other"

    # Choose a default category order; can be overridden via config.
    default_category_order = [
        "Retrieval",
        "Reranking",
        "Classification",
        "PairClassification",
        "STS",
        "Clustering",
        "Summarization",
        "Other",
    ]
    category_order: list[str] = config.get("category_order", default_category_order)

    # Decide how to sort task columns; default groups by category then name.
    task_sort_mode: str = config.get("task_sort", "category_alpha")  # options: category_alpha, alpha, none

    # 2段階ソートで左端に code_name / run_name を固定
    df = combined_df.sort_values(by=["code_name", "run_name"]).reset_index(drop=True)

    # Identify task columns (metric columns only)
    meta_cols = {"code_name", "run_name"}
    task_cols = [c for c in df.columns if c not in meta_cols]

    # Optionally restrict to tasks in the specified benchmark
    only_benchmark_tasks: bool = bool(config.get("only_benchmark_tasks", True))
    bench_task_names = set(mteb_task_type_map.keys()) if 'mteb_task_type_map' in locals() else set()
    if only_benchmark_tasks and bench_task_names:
        task_cols = [c for c in task_cols if c in bench_task_names]
        # Trim DataFrame to selected task columns
        df = df[["code_name", "run_name"] + task_cols]
    # Drop explicitly excluded tasks
    if drop_tasks:
        task_cols = [c for c in task_cols if c not in drop_tasks]
        df = df[["code_name", "run_name"] + task_cols]

    # Build category -> columns mapping
    cat_to_cols: dict[str, list[str]] = defaultdict(list)
    task_to_cat: dict[str, str] = {}
    for col in task_cols:
        cat = infer_task_category(col)
        task_to_cat[col] = cat
        cat_to_cols[cat].append(col)

    # Compute per-category averages
    avg_cols: list[str] = []
    for cat in category_order:
        cols = cat_to_cols.get(cat, [])
        if not cols:
            continue
        avg_col = f"avg_{cat}"
        df[avg_col] = df[cols].astype(float).mean(axis=1, skipna=True)
        avg_cols.append(avg_col)

    # Compute macro/micro averages
    # - micro_ave: mean of all task columns
    # - macro_ave: mean of available per-category averages (equal weight by category)
    df["micro_ave"] = df[task_cols].astype(float).mean(axis=1, skipna=True) if task_cols else float("nan")
    df["macro_ave"] = df[avg_cols].astype(float).mean(axis=1, skipna=True) if avg_cols else float("nan")

    # Sort remaining task columns
    if task_sort_mode == "alpha":
        sorted_task_cols = sorted(task_cols)
    elif task_sort_mode == "none":
        sorted_task_cols = task_cols
    else:  # category_alpha
        sorted_task_cols = []
        # Respect category order, alphabetical within each category
        for cat in category_order:
            if cat in cat_to_cols:
                sorted_task_cols.extend(sorted(cat_to_cols[cat]))
        # Include any unseen categories at the end
        remaining = [c for c in task_cols if c not in set(sorted_task_cols)]
        # Place remaining alphabetically to be deterministic
        sorted_task_cols.extend(sorted(remaining))

    # Final column order: meta -> macro/micro -> category averages -> tasks
    left_cols = ["macro_ave", "micro_ave"] + avg_cols
    ordered_cols = ["code_name", "run_name"] + left_cols + [c for c in sorted_task_cols if c not in left_cols]
    # De-duplicate in case of any overlap
    seen = set()
    ordered_cols = [c for c in ordered_cols if (c not in seen and not seen.add(c))]
    df = df[ordered_cols]

    integrated_output_path = f"{output_dir}/{group_code_name}_metrics.csv"
    df.to_csv(integrated_output_path, index=False)
    print(f"Saved integrated CSV: {integrated_output_path}")
    print(f"✅ All CSV files saved to the '{output_dir}' directory.")
else:
    print("指定された条件に一致する実験が見つからなかったか、mteb_finalで始まるサマリーデータが存在しませんでした。")
