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
        output_folder = Path(model_name).parent.parent
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
