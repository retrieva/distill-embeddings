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
    parser.add_argument(
        "--validate_first", action="store_true", help="run validation before training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size per device"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loader"
    )
    parser.add_argument("--lr", type=float, default=1.0e-4, help="learning rate")

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
        "--taid_disable_adaptive",
        action="store_true",
        help="disable the TAID's adaptive update",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.05,
        help="CKD temperature for scaling the logits",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="maximum sequence length",
    )
    return parser.parse_args()
