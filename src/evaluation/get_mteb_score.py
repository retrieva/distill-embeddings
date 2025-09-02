import argparse

import mteb
import pandas as pd
import yaml
from mteb.leaderboard.table import create_tables


def main():
    parser = argparse.ArgumentParser(description="MTEB/JMTEB評価スコアを取得")
    parser.add_argument(
        "--lang", "-l", choices=["jpn", "eng"], default="jpn", help="評価言語を選択 (jpn: 日本語, eng: 英語)"
    )
    parser.add_argument("--benchmark_name", type=str, default="on_eval_tasks", help="Name of the benchmark.")
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
    scores = pd.concat([summary_df, per_task_df], axis=1)

    # 出力ファイル名を言語に応じて変更
    output_file = f"output/loaded_{'jmteb' if args.lang == 'jpn' else 'mteb'}_score.csv"
    scores.to_csv(output_file, index=False)
    print(f"結果を保存しました: {output_file}")


if __name__ == "__main__":
    main()
