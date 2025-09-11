import argparse
from pathlib import Path

from src.evaluation.posthoc_eval_to_wandb import _load_student_from_ds_dir


def _infer_output_base_from_ds_dir(ds_dir: Path) -> Path:
    # Walk up a few levels to find an output dir containing wandb/
    p = ds_dir
    for _ in range(4):
        if (p / "wandb").exists():
            return p
        p = p.parent
    return ds_dir.parent


def main():
    ap = argparse.ArgumentParser(description="Extract SentenceTransformer student weights from a DeepSpeed checkpoint dir")
    ap.add_argument("--ds_dir", type=str, required=True, help="Path to DeepSpeed checkpoint directory (contains shards)")
    ap.add_argument("--output_dir", type=str, required=True, help="Where to save the SentenceTransformer folder")
    ap.add_argument("--student_model", type=str, default=None, help="Base student model if W&B config not available")
    args = ap.parse_args()

    ds_dir = Path(args.ds_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_base = _infer_output_base_from_ds_dir(ds_dir)
    model, _hp = _load_student_from_ds_dir(ds_dir, output_base, args.student_model)
    model.save(str(out_dir))
    print(f"Saved SentenceTransformer weights to: {out_dir}")


if __name__ == "__main__":
    main()

