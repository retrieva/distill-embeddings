import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from src.data import DataModuleForDistill
from src.model import KDForSentEmb
from src.arguments import parse_args
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(42, workers=True)
    # torch.set_float32_matmul_precision("high")
    data_path = Path(args.data_dir) / f"triplet-{args.language}" / f"{args.teacher_model.replace('/','_')}_encoded" / args.dataset_name
    use_pos = "_w/pos" if args.use_pos else ""
    code_name = f"e{args.num_epochs}_bs{args.batch_size}_lr{args.lr}_{args.loss_type}{use_pos}"
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
    modelcheckpoint = ModelCheckpoint(
        monitor="val_0/loss",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        devices="auto",
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        callbacks=[modelcheckpoint],
        logger=WandbLogger(
            name=os.path.basename(args.output_dir),
            project="distillation",
            group=args.student_model.replace('/', '_'),
            tags=[args.dataset_name,f"{args.teacher_model.replace('/','_')}_encoded"],
            save_dir=args.output_dir,
        ),
    )
    if args.validate_first:
        trainer.validate(model, data)
    trainer.fit(model, data)