from argparse import Namespace
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def load_model(model_name: str | Path, return_output_folder: bool = False):
    if model_name.endswith(".ckpt"):
        checkpoint = torch.load(model_name, weights_only=False)
        model_weights = {
            k.removeprefix("student_model."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("student_model.")
        }
        model = SentenceTransformer(checkpoint["student_model_name"])
        model.load_state_dict(model_weights)
        model.eval().bfloat16()
        code_name = get_code_name(checkpoint["hyper_parameters"])
        output_folder = Path(model_name).parent.parent.parent / code_name
    else:
        model = SentenceTransformer(model_name)
        output_folder = Path("output") / model_name.replace("/", "_")
    if return_output_folder:
        return model, output_folder
    return model


PROMPT_MAP = {
    "none": "",
    "retrieval": "Given a question, retrieve passages that answer the question",
    "sts": "Retrieve semantically similar text",
    "classification": "Given a text, classify its topic",
}


def get_code_name(args: Namespace | dict) -> str:
    if isinstance(args, dict):
        args = Namespace(**args)
    use_pos, add_prefix, use_lora, distill_weight = "", "", "", ""
    try:
        use_pos = "_w-pos" if getattr(args, "use_pos", False) else ""
        add_prefix = "_prefix" if getattr(args, "add_prefix", False) else ""
        distill_weight = f"{str(args.distill_weight)}" if getattr(args, "distill_weight", 1.0) != 1.0 else ""
        use_lora = "_lora" if getattr(args, "use_lora", False) else ""
    except AttributeError:
        pass
    if hasattr(args, "global_batch_size"):
        global_batch_size = args.global_batch_size
    else:
        import os

        try:
            world_size = getattr(
                args,
                "world_size",
                int(os.environ.get("WORLD_SIZE", torch.cuda.device_count() if torch.cuda.is_available() else 1)),
            )
        except Exception:
            world_size = 1
        global_batch_size = args.batch_size * world_size
        # 評価時再現のため設定へ反映（任意）
        args.global_batch_size = global_batch_size
    data_name = args.data_name if "data_name" in args.__dict__ else args.dataset_name
    code_name = (
        f"{data_name}_e{args.num_epochs}_bs{global_batch_size}_"
        f"{args.scheduler}{args.lr}_{args.loss_type}{distill_weight}{use_pos}{add_prefix}{use_lora}"
    )
    return code_name
