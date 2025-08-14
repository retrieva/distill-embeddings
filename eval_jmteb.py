from sentence_transformers import SentenceTransformer
import mteb
import argparse
import torch
from pathlib import Path
import yaml

with open("tasks.yaml", 'r') as file:
    tasks = yaml.safe_load(file)
    tasks = tasks["jpn"]["on_eval_tasks"]

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
    evaluation = mteb.MTEB(tasks=tasks, task_langs=["jpn"],)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # MTEBの評価を実行
        scores = evaluation.run(model, output_folder=output_folder,
                                batch_size=args.batch_size, num_workers=args.num_workers,trust_remote_code=True,verbosity=0)
    mteb_dict = {score.task_name: score.get_score() for score in scores}
    print("Evaluation scores:")
    for task_name, score in mteb_dict.items():
        print(f"{task_name}: {score}")
    with open(output_folder / "scores.txt", "w") as f:
        for task_name in mteb_dict.keys():
            f.write(f"{task_name}\t")
        f.write(f"\n")
        for score in mteb_dict.values():
            f.write(f"{score}\t")
    print(f"Scores saved to {output_folder / 'scores.txt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTEB evaluation on a model.")
    parser.add_argument("--model_name", type=str, default="cl-nagoya/ruri-v3-pt-30m",help="Name of the model to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8,help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4,help="Number of workers for data loading.")
    args = parser.parse_args()
    main(args)