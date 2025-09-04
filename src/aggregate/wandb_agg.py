import os

import pandas as pd
import yaml  # yamlをインポート

import wandb
from src.utils import get_code_name

with open("src/aggregate/agg_config.yaml") as f:
    config = yaml.safe_load(f)

project = config["project"]
filters = config["filters"]
group_code_name = config["group_code_name"]

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

    metrics_to_export = [col for col in combined_df.columns if col != "run_name"]

    integrated_output_path = f"{output_dir}/{group_code_name}_metrics.csv"
    # 2段階ソートして両カラムを保持したまま出力（左端に配置）
    sorted_df = combined_df.sort_values(by=["code_name", "run_name"])
    ordered_cols = ["code_name", "run_name"] + [c for c in sorted_df.columns if c not in ["code_name", "run_name"]]
    sorted_df = sorted_df[ordered_cols]
    sorted_df.to_csv(integrated_output_path, index=False)
    print(f"Saved integrated CSV: {integrated_output_path}")
    print(f"✅ All CSV files saved to the '{output_dir}' directory.")
else:
    print("指定された条件に一致する実験が見つからなかったか、mteb_finalで始まるサマリーデータが存在しませんでした。")
