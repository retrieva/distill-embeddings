import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from src.training.arguments import parse_args
from src.training.data import DataModuleForDistill
from src.training.model import KDForSentEmb
from src.utils import get_code_name

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    data_dir = (
        Path(args.data_dir) / f"{args.data_name}-{args.language}" / f"{args.teacher_model.replace('/', '_')}_encoded"
    )
    code_name = get_code_name(args)

    args.output_dir = (
        Path(args.output_dir)
        / args.student_model.replace("/", "_")
        / args.teacher_model.replace("/", "_")
        / args.data_size
        / code_name
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = KDForSentEmb(args)
    model.configure_model()
    data = DataModuleForDistill(
        data_dir=data_dir,
        data_num=args.data_size,
        student_tokenizer=model.student_model.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        add_prefix=args.add_prefix,
    )

    modelcheckpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints", filename="{epoch:02d}", every_n_epochs=1, save_top_k=-1
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        devices="auto",
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        callbacks=[modelcheckpoint, lr_monitor],
        strategy=DeepSpeedStrategy(stage=2, allgather_bucket_size=5e8, reduce_bucket_size=5e8),
        logger=WandbLogger(
            name=os.path.basename(args.output_dir),
            project="distillation",
            group=args.student_model.replace("/", "_"),
            tags=[args.data_size, f"{args.teacher_model.replace('/', '_')}_encoded"],
            save_dir=args.output_dir,
            id=args.your_run_id if args.your_run_id else None,
            resume="must" if args.your_run_id else None,
        ),
    )
    if args.validate_first:
        trainer.validate(model, data)
    trainer.fit(model, data, ckpt_path=args.ckpt_path if args.ckpt_path else None)
