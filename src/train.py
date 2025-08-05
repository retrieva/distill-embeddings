import os
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DeepSpeedStrategy
from src.data import StreamingSFTDataModule
from src.model import KDForSentEmb
from src.arguments import parse_args

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    data = IIKANZI_DataModule(
        # 後ほど実装
        tokenizer_path=args.teacher_model,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = KDForSentEmb(args)
    # 実装しなきゃ
    # modelcheckpoint = ModelCheckpoint(
    #     dirpath=args.output_dir,
    #     monitor="val_0/rougeL",
    #     mode="max",
    #     save_top_k=1,
    #     save_last=False,
    # )
    trainer = L.Trainer(
        devices="0,1,2,3,4,5,6,7",
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        limit_train_batches=10,
        limit_val_batches=5,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=DeepSpeedStrategy(
            stage=2, allgather_bucket_size=5e8, reduce_bucket_size=5e8
        ),
        callbacks=[modelcheckpoint],
        logger=WandbLogger(
            name=os.path.basename(args.output_dir),
            project="distillation",
        ),
    )
    # if args.validate_first:
    #     trainer.validate(model, data)
    trainer.fit(model, train_dataloader, val_dataloader)