from sentence_transformers import SentenceTransformer
import mteb
import argparse
import torch
from pathlib import Path

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
        output_folder = "output" + model_name.replace("/", "_")
    evaluation = mteb.MTEB(tasks=["AmazonCounterfactualClassification",
                                # "AmazonReviewsClassification",
                                # "LivedoorNewsClustering.v2",
                                # "MewsC16JaClustering",
                                # "MIRACLReranking",
                                # "NLPJournalAbsIntroRetrieval",
                                # "NLPJournalTitleAbsRetrieval",
                                # "NLPJournalTitleIntroRetrieval",
                                "JSICK",
                                "JSTS"],
                                task_langs=["jpn"],)
    scores = evaluation.run(model, output_folder=output_folder,
                            batch_size=args.batch_size, num_workers=args.num_workers,trust_remote_code=True)
    import pdb; pdb.set_trace()
    print(f"Evaluation results saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTEB evaluation on a model.")
    parser.add_argument("--model_name", type=str, default="cl-nagoya/ruri-v3-pt-30m",help="Name of the model to evaluate.")
    parser.add_argument("--batch_size", type=int, default=8,help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4,help="Number of workers for data loading.")
    args = parser.parse_args()
    main(args)