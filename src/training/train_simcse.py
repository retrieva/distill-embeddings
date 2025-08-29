import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import load_dataset
import torch
import argparse
from pathlib import Path
import mteb


def main(args):
    # モデル読み込み（ckpt対応）
    if args.model_name.endswith(".ckpt"):
        checkpoint = torch.load(args.model_name, map_location="cpu")
        model_weights = {
            k.removeprefix("student_model."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("student_model.")
        }
        model = SentenceTransformer(checkpoint["student_model_name"])
        model.load_state_dict(model_weights, strict=False)
        output_folder = Path(args.model_name).parent.parent / "sup_simcse"
    else:
        model = SentenceTransformer(args.model_name, trust_remote_code=True)
        output_folder = Path("output") / args.model_name.replace("/", "_") / "sup_simcse"

    output_folder.mkdir(parents=True, exist_ok=True)

    print(mteb.model_meta_from_sentence_transformers(model))
    # データ準備（anc-pos ペア）
    dataset = load_dataset("hpprc/emb", "jsnli-triplet", split="train")
    dataset = dataset.map(
        lambda x: {"anchor": x["anc"].strip(), "positive": x["pos"][0].strip(), "negative": x["neg"][0].strip()}
    )
    dataset = dataset.select_columns(["anchor", "positive", "negative"])
    # 新しい Trainer API
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_folder),
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,  # 必要に応じて True
        logging_steps=100,
        save_strategy="epoch",
        report_to=[],  # wandb 無効化
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model),
    )

    trainer.train()

    # 評価
    evaluation = mteb.MTEB(tasks=["AmazonCounterfactualClassification", "JSICK", "JSTS"], task_langs=["jpn-Jpan"])
    model.model_card_data.language = ["jpn-Jpan"]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = evaluation.run(
            model,
            output_folder=output_folder,
            encode_kwargs={"batch_size": args.batch_size},
            num_workers=args.num_workers,
            verbosity=0,
        )

    # 結果保存
    results = {s.task_name: s.get_score() for s in scores}
    print("Results:", results)

    with open(output_folder / "scores.txt", "w") as f:
        f.write("|" + "|".join(results.keys()) + "|\n")
        f.write("|" + "|".join(f"{v:.4f}" for v in results.values()) + "|")

    print(f"✅ Saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="cl-nagoya/ruri-v3-30m")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    main(parser.parse_args())
