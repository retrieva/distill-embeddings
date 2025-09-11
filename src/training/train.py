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
    # Avoid long first-step kernel compiles (Flash/MemEff SDP) on some setups
    try:
        from torch.backends.cuda import sdp_kernel

        sdp_kernel.enable_flash(False)
        sdp_kernel.enable_mem_efficient(False)
        sdp_kernel.enable_math(True)
    except Exception:
        pass
    # Safer defaults to avoid OpenMP/joblib conflicts during eval/clustering
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()
    world_size = 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # できれば環境変数優先で取得（torchrun が設定）
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1 and torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    args.world_size = world_size
    args.global_batch_size = args.batch_size * args.world_size
    L.seed_everything(42, workers=True)
    # Keep CPU thread usage minimal for stability on HPC environments
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
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
        chunk_parts=args.chunk_parts,
        max_effective_pairs_per_rank=args.max_effective_pairs_per_rank,
    )

    modelcheckpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # Decide strategy: default to plain Lightning on single GPU, DeepSpeed on multi-GPU,
    # unless user explicitly forces DeepSpeed via --strategy deepspeed
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ds = (args.strategy == "deepspeed") or (args.strategy == "auto" and env_world_size > 1)
    strategy = None
    if use_ds:
        deepspeed_config = {
            "zero_optimization": {"stage": int(args.zero_stage)},
            "train_micro_batch_size_per_gpu": args.batch_size,
            "activation_checkpointing": {
                "partition_activations": True,
                "contiguous_memory_optimization": False,
                "cpu_checkpointing": False,
            },
        }
        strategy = DeepSpeedStrategy(config=deepspeed_config)

    trainer = L.Trainer(
        devices="auto",
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        precision=args.precision,
        num_sanity_val_steps=0,
        callbacks=[modelcheckpoint, lr_monitor],
        strategy=strategy or "auto",
        use_distributed_sampler=False,
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
