import wandb
import pandas as pd
import plotly.express as px
import os

# --- 1, 2, 3は変更なし ---
api = wandb.Api()
filters = {
    "$and": [
        {"config.num_epochs": {"$gte": 2}},
        {"tags": "Qwen_Qwen3-Embedding-4B_encoded"},
        {"tags": "1000"},
        {"state": "finished"}
    ]
}
runs = api.runs("retrieva-research/distillation", filters=filters)

target_keys = [
    'trainer/global_step',
    'StackExchangeClustering.v2',
    'STSBenchmark',
    'SICK-R',
    'HotpotQAHardNegatives',
    'train/loss',
    'run_name'
]
metrics_to_fill = [
    'StackExchangeClustering.v2',
    'STSBenchmark',
    'SICK-R',
    'HotpotQAHardNegatives',
]


# --- 4. データの取得と結合（変更なし） ---
all_histories = []
for run in [runs[1]]:
    history_df = run.history()
    if not history_df.empty:
        # # 評価メトリクスの欠損値を直前の値で埋める (Forward Fill)
        # for metric in metrics_to_fill:
        #     if metric in history_df.columns:
        #         history_df[metric] = history_df[metric].interpolate(method='linear')

        # 実際に存在するキーと、欲しいキーのリストで共通するものだけを抽出
        available_keys = history_df.columns
        keys_to_keep = [key for key in target_keys if key in available_keys]
        
        if keys_to_keep:
            subset_df = history_df[keys_to_keep].copy()
            subset_df['run_name'] = run.name
            all_histories.append(subset_df)
# --- 5. Plotlyでインタラクティブなグラフを作成 ---
if all_histories:
    combined_df = pd.concat(all_histories, ignore_index=True)
    output_dir = "wandb_plotly_charts"
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = [key for key in target_keys if key not in ['trainer/global_step', 'run_name']]

    for metric in metrics_to_plot:
        # DataFrameにそのメトリクスの列が存在するか確認
        if metric not in combined_df.columns:
            print(f"Skipping plot for '{metric}' as it's not in the combined data.")
            continue
        
        print(f"Generating interactive plot for {metric}...")
        
        # Plotly Expressで折れ線グラフを作成
        fig = px.line(
            combined_df,
            x='trainer/global_step',
            y=metric,
            color='run_name',  # run_nameごとに色分け
            title=f'{metric} for All Filtered Runs',
            labels={'trainer/global_step': 'Global Step', metric: metric} # 軸ラベル
        )
        fig.update_xaxes(range=[0, combined_df['trainer/global_step'].max()])

        
        # グラフをHTMLファイルとして保存
        safe_metric_name = metric.replace('/', '_')
        output_path = f'{output_dir}/{safe_metric_name}.html'
        fig.write_html(output_path)
        
        # fig.show() # Jupyter Notebookなどではこの行で直接表示できます

    print(f"✅ All interactive charts saved to the '{output_dir}' directory.")

else:
    print("指定された条件に一致する実験が見つからなかったか、対象のキーを持つ履歴データが存在しませんでした。")