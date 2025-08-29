import os
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from src.data import DataModuleForDistill
from src.model import KDForSentEmb
from src.arguments import parse_args
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    data_dir = Path(args.data_dir) / f"{args.data_name}-{args.language}" / f"{args.teacher_model.replace('/','_')}_encoded"
    use_pos = "_w-pos" if args.use_pos else ""
    # GPU数を取得してグローバルバッチサイズを計算
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    global_batch_size = args.batch_size * num_devices
    code_name = f"{args.data_name}_e{args.num_epochs}_bs{global_batch_size}_{args.scheduler}{args.lr}_{args.loss_type}{use_pos}"
    args.output_dir = Path(args.output_dir) / args.student_model.replace('/', '_') / args.teacher_model.replace('/', '_') / args.dataset_name / code_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = KDForSentEmb(args)
    model.configure_model()
    data = DataModuleForDistill(
        data_dir=data_dir,
        data_num = args.dataset_name,
        student_tokenizer=model.student_model.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length
    )

    modelcheckpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1 
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = L.Trainer(
        devices="auto",
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
        callbacks=[modelcheckpoint,lr_monitor],
        logger=WandbLogger(
            name=os.path.basename(args.output_dir),
            project="distillation",
            group=args.student_model.replace('/', '_'),
            tags=[args.dataset_name,f"{args.teacher_model.replace('/','_')}_encoded"],
            save_dir=args.output_dir,
            id=args.your_run_id if args.your_run_id else None,
            resume="must" if args.your_run_id else None,
        ),
    )
    if args.validate_first:
        trainer.validate(model, data)
    trainer.fit(model, data, ckpt_path=args.ckpt_path if args.ckpt_path else None)