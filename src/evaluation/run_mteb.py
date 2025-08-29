from sentence_transformers import SentenceTransformer
import mteb
import argparse
import torch
from pathlib import Path
import yaml
from mteb.leaderboard.table import create_tables
import pandas as pd
from mteb.load_results.benchmark_results import ModelResult

with open("tasks.yaml", 'r') as file:
    tasks = yaml.safe_load(file)
    tasks = tasks["eng"]["on_eval_tasks"]

def main(args):
    model_name = args.model_name
    if model_name.endswith(".ckpt"):
        checkpoint = torch.load(model_name)
        model_weights = {k.removeprefix("student_model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("student_model.")}
        model = SentenceTransformer(checkpoint["student_model_name"])
        model.load_state_dict(model_weights)
        model.eval().bfloat16()
        output_folder = Path(model_name).parent.parent
    else:
        # 2. SentenceTransformerモデルを直接ロード
        model = SentenceTransformer(model_name)
        output_folder = Path("output") / model_name.replace("/", "_")
    evaluation = mteb.MTEB(tasks=tasks, task_langs=["eng"],)
    evaluation.tasks[0].calculate_metadata_metrics()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # MTEBの評価を実行
        scores = evaluation.run(model, output_folder=output_folder,
                                batch_size=args.batch_size, num_workers=args.num_workers,trust_remote_code=True,verbosity=1)
    scores_long = ModelResult(model_name=model_name,
                              model_revision=model.model_card_data.base_model_revision,
                              task_results=scores).get_scores(format="long")

    # Convert scores into leaderboard tables
    summary_gr_df, per_task_gr_df = create_tables(scores_long=scores_long)
    print(per_task_gr_df.value)
    # Convert Gradio DataFrames to Pandas
    summary_df = pd.DataFrame(
        summary_gr_df.value["data"], columns=summary_gr_df.value["headers"]
    )
    per_task_df = pd.DataFrame(
        per_task_gr_df.value["data"], columns=per_task_gr_df.value["headers"]
    )
    scores = pd.concat([summary_df, per_task_df], axis=1)
    scores.to_csv(output_folder / "mteb_scores.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTEB evaluation on a model.")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base",help="Name of the model to evaluate.")
    parser.add_argument("--batch_size", type=int, default=64,help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4,help="Number of workers for data loading.")
    args = parser.parse_args()
    main(args)