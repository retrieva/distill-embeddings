import os

import pandas as pd
import yaml  # yamlをインポート

import wandb

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
    if summary:
        mteb_final_data = {}
        for k, v in summary.items():
            if k.startswith("mteb_final/"):
                clean_key = k.replace("mteb_final/", "")
                mteb_final_data[clean_key] = v

        if mteb_final_data:
            mteb_final_data["run_name"] = run.name
            all_summaries.append(mteb_final_data)

if all_summaries:
    combined_df = pd.DataFrame(all_summaries)
    output_dir = "wandb_agg_output"
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_export = [col for col in combined_df.columns if col != "run_name"]

    integrated_output_path = f"{output_dir}/{group_code_name}_metrics.csv"
    combined_df.set_index("run_name").to_csv(integrated_output_path)
    print(f"Saved integrated CSV: {integrated_output_path}")
    print(f"✅ All CSV files saved to the '{output_dir}' directory.")
else:
    print("指定された条件に一致する実験が見つからなかったか、mteb_finalで始まるサマリーデータが存在しませんでした。")
