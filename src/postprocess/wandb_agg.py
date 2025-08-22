import wandb
import pandas as pd
import os

# --- 1, 2, 3は変更なし ---
api = wandb.Api()
filters = {
    "$and": [
        {"config.num_epochs": {"$gte": 3}},
        {"config.batch_size": {"$gte": 64}},
        {"tags": "Qwen_Qwen3-Embedding-4B_encoded"},
        {"state": "finished"}
    ]
}
group_code_name = "ep3_bs64_4B"
runs = api.runs("retrieva-research/distillation", filters=filters)

# --- 4. summaryデータの取得と結合 ---
all_summaries = []
for run in runs:
    summary = run.summary
    if summary:
        # mteb_finalで始まるキーのみを抽出し、プレフィックスを削除
        mteb_final_data = {}
        for k, v in summary.items():
            if k.startswith('mteb_final/'):
                # mteb_final/ プレフィックスを削除
                clean_key = k.replace('mteb_final/', '')
                mteb_final_data[clean_key] = v
        
        if mteb_final_data:
            mteb_final_data['run_name'] = run.name
            all_summaries.append(mteb_final_data)

# --- 5. CSVファイルを生成 ---
if all_summaries:
    combined_df = pd.DataFrame(all_summaries)
    output_dir = "wandb_agg_output"
    os.makedirs(output_dir, exist_ok=True)

    # run_nameを除いたメトリクス名を取得
    metrics_to_export = [col for col in combined_df.columns if col != 'run_name']

    # 全メトリクスをまとめた統合CSV作成
    integrated_output_path = f'{output_dir}/{group_code_name}_metrics.csv'
    combined_df.set_index('run_name').to_csv(integrated_output_path)
    print(f"Saved integrated CSV: {integrated_output_path}")

    print(f"✅ All CSV files saved to the '{output_dir}' directory.")

else:
    print("指定された条件に一致する実験が見つからなかったか、mteb_finalで始まるサマリーデータが存在しませんでした。")