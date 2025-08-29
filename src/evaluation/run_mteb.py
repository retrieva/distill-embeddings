import argparse

import mteb
import pandas as pd
import yaml
from mteb.leaderboard.table import create_tables
from mteb.load_results.benchmark_results import ModelResult

from src.utils import load_model


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
    evaluation = mteb.MTEB(tasks=tasks, task_langs=[args.language])
    evaluation.tasks[0].calculate_metadata_metrics()
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
        )
    scores_long = ModelResult(
        model_name=model_name,
        model_revision=model.model_card_data.base_model_revision,
        task_results=scores,
    ).get_scores(format="long")

    # Convert scores into leaderboard tables
    summary_gr_df, per_task_gr_df = create_tables(scores_long=scores_long)
    print(per_task_gr_df.value)
    # Convert Gradio DataFrames to Pandas
    summary_df = pd.DataFrame(summary_gr_df.value["data"], columns=summary_gr_df.value["headers"])
    per_task_df = pd.DataFrame(per_task_gr_df.value["data"], columns=per_task_gr_df.value["headers"])
    scores = pd.concat([summary_df, per_task_df], axis=1)
    scores.to_csv(output_folder / f"{args.language}_mteb_scores.csv")


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
    parser.add_argument("--benchmark_name", type=str, default="on_eval_tasks", help="Name of the benchmark.")
    args = parser.parse_args()
    main(args)
