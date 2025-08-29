import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # model args
    parser.add_argument(
        "--student_model",
        type=str,
        default="TinyLlama/TinyLlama_v1.1",
        help="student model path",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="teacher model path",
    )

    # training args
    parser.add_argument(
        "--output_dir", type=str, default="logs", help="output directory"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="path to data dir")
    parser.add_argument("--dataset_name", type=str, required=True, help="name of the dataset")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="log every n steps")
    parser.add_argument(
        "--validate_first", action="store_true", help="run validation before training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size per device"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loader"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--ckd_temp", type=float, default=0.05, help="CKD temperature")
    parser.add_argument("--kld_temp", type=float, default=1.0, help="KLD temperature")
    parser.add_argument("--cse_temp", type=float, default=0.05, help="InfoCSE temperature")

    # loss
    parser.add_argument("--loss_type", type=str, required=True, help="loss type")
    parser.add_argument(
        "--distil_ratio",
        type=float,
        default=1.0,
        help="weight of distil loss against cross entropy loss",
    )
    parser.add_argument(
        "--adaptive_kl_threshold",
        type=float,
        default=0.5,
        help="threshold for head. It indicates mu in the paper (https://arxiv.org/abs/2404.02657)",
    )
    parser.add_argument("--skew_beta", type=float, default=0.1, help="skew weight")
    parser.add_argument(
        "--taid_t_start", type=float, default=0.4, help="t_start in TAID"
    )
    parser.add_argument("--taid_t_end", type=float, default=1.0, help="t_end in TAID")
    parser.add_argument(
        "--taid_alpha",
        type=float,
        default=5e-4,
        help="t learning rate for TAID's adaptive update",
    )
    parser.add_argument(
        "--taid_beta",
        type=float,
        default=0.99,
        help="momentum coeff for TAID's adaptive update",
    )
    parser.add_argument(
        "--ckd_max_queue_len",
        type=int,
        default=65536,
        help="maximum queue length for CKD (default: 65536, set to 0 to disable queue)",
    )
    parser.add_argument(
        "--taid_disable_adaptive",
        action="store_true",
        help="disable the TAID's adaptive update",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="warmup ratio for learning rate scheduler",
    )
    parser.add_argument("--scheduler",type=str, default="cosine",
        choices=["constant", "wsd", "cosine"],
        help="scheduler type",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="maximum sequence length",
    )
    parser.add_argument("--mteb_eval",action="store_true",
        help="run MTEB evaluation at the end of training",
    )
    parser.add_argument("--get_id_iso",action="store_true",
        help="get intrinsic dimension and iso score",
    )
    parser.add_argument("--language", type=str, default="eng",
        help="language for experiment",
    )
    parser.add_argument("--use_pos",action="store_true",
        help="use positive samples(contrastive)",
    )
    parser.add_argument("--your_run_id", type=str, default=None,
                        help="your run id")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="path to the checkpoint file")
    parser.add_argument("--data_name", type=str, default="gte",
                        choices=["gte", "triplet"],
                        help="name of the data")
    return parser.parse_args()
