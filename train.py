import os
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from src.data import DataModuleForDistill
from src.model import KDForSentEmb
from src.arguments import parse_args
from pathlib import Path

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(42, workers=True)
    # torch.set_float32_matmul_precision("high")
    data_path = Path(args.data_dir) / f"{args.teacher_model.replace('/','_')}_encoded" / args.dataset_name
    code_name = f"e{args.num_epochs}_bs{args.batch_size}_{args.loss_type}"
    args.output_dir = Path(args.output_dir) / args.student_model.replace('/', '_') / args.teacher_model.replace('/', '_') / args.dataset_name / code_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = KDForSentEmb(args)
    model.configure_model()
    data = DataModuleForDistill(
        data_path=data_path,
        student_tokenizer=model.student_model.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )
    # 実装しなきゃ
    # modelcheckpoint = ModelCheckpoint(
    #     dirpath=args.output_dir,
    #     monitor="val_0/rougeL",
    #     mode="max",
    #     save_top_k=1,
    #     save_last=False,
    # )
    trainer = L.Trainer(
        devices="auto",
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        precision="bf16-true",
        num_sanity_val_steps=0,
        # strategy=DeepSpeedStrategy(
        #     stage=2, allgather_bucket_size=5e8, reduce_bucket_size=5e8
        # ),
        # callbacks=[modelcheckpoint],
        logger=WandbLogger(
            name=os.path.basename(args.output_dir),
            project="distillation",
        ),
    )
    if args.validate_first:
        trainer.validate(model, data)
    trainer.fit(model, data)