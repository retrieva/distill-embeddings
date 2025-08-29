import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Experiment Arguments")

    # --- Model Arguments ---
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument(
        "--student_model", type=str, default="nomic-ai/modernbert-embed-base-unsupervised", help="student model path"
    )
    model_args.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-Embedding-4B", help="teacher model path")
    model_args.add_argument("--max_length", type=int, default=4096, help="maximum sequence length")

    # --- Data Arguments ---
    data_args = parser.add_argument_group("Data Arguments")
    data_args.add_argument("--data_dir", type=str, default="data", help="path to data dir")
    data_args.add_argument("--data_size", type=str, required=True, help="data size")
    data_args.add_argument("--data_name", type=str, default="gte", choices=["gte", "triplet"], help="name of the data")
    data_args.add_argument(
        "--language", type=str, default="eng", choices=["eng", "jpn"], help="language for experiment"
    )
    data_args.add_argument("--num_workers", type=int, default=4, help="number of workers for data loader")

    # --- Training Arguments ---
    training_args = parser.add_argument_group("Training Arguments")
    training_args.add_argument("--output_dir", type=str, default="output/result", help="output directory")
    training_args.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    training_args.add_argument("--batch_size", type=int, default=8, help="batch size per device")
    training_args.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    training_args.add_argument(
        "--scheduler", type=str, default="cosine", choices=["constant", "wsd", "cosine"], help="scheduler type"
    )
    training_args.add_argument(
        "--warmup_ratio", type=float, default=0.05, help="warmup ratio for learning rate scheduler"
    )

    # --- Loss & Distillation Arguments ---
    loss_args = parser.add_argument_group("Loss & Distillation Arguments")
    loss_args.add_argument("--loss_type", type=str, required=True, help="loss type")
    # Temperatures
    loss_args.add_argument("--ckd_temp", type=float, default=0.05, help="CKD temperature")
    loss_args.add_argument("--kld_temp", type=float, default=1.0, help="KLD temperature")
    loss_args.add_argument("--cse_temp", type=float, default=0.05, help="InfoCSE temperature")
    # TAID specific
    loss_args.add_argument("--taid_t_start", type=float, default=0.4, help="t_start in TAID")
    loss_args.add_argument("--taid_t_end", type=float, default=1.0, help="t_end in TAID")
    loss_args.add_argument("--taid_alpha", type=float, default=5e-4, help="t learning rate for TAID's adaptive update")
    loss_args.add_argument("--taid_beta", type=float, default=0.99, help="momentum coeff for TAID's adaptive update")
    loss_args.add_argument("--taid_disable_adaptive", action="store_true", help="disable the TAID's adaptive update")
    # Other loss params
    loss_args.add_argument("--ckd_max_queue_len", type=int, default=65536, help="maximum queue length for CKD")
    loss_args.add_argument("--use_pos", action="store_true", help="use positive samples (contrastive)")

    # --- Logging & Evaluation Arguments ---
    log_eval_args = parser.add_argument_group("Logging & Evaluation Arguments")
    log_eval_args.add_argument("--log_every_n_steps", type=int, default=1, help="log every n steps")
    log_eval_args.add_argument("--val_check_interval", type=float, default=1.0)
    log_eval_args.add_argument("--validate_first", action="store_true", help="run validation before training")
    log_eval_args.add_argument("--mteb_eval", action="store_true", help="run MTEB evaluation at the end of training")
    log_eval_args.add_argument("--get_id_iso", action="store_true", help="get intrinsic dimension and iso score")

    # --- Experiment Management Arguments ---
    exp_args = parser.add_argument_group("Experiment Management Arguments")
    exp_args.add_argument("--your_run_id", type=str, default=None, help="your run id")
    exp_args.add_argument("--ckpt_path", type=str, default=None, help="path to the checkpoint file")

    return parser.parse_args()
