import argparse
import json
import warnings
from pathlib import Path

import mteb
import torch
from mteb.encoder_interface import PromptType
from sentence_transformers import SentenceTransformer

import wandb


def _infer_output_base_from_ckpt(ckpt_path: Path) -> Path:
    # last.ckpt is saved at <output_dir>/last.ckpt
    # epoch checkpoints are at <output_dir>/checkpoints/<epoch>.ckpt
    if ckpt_path.name == "last.ckpt":
        return ckpt_path.parent
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    # fallback to parent
    return ckpt_path.parent


def _infer_run_id(output_base: Path) -> str | None:
    # Try W&B latest-run metadata
    meta = output_base / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("id")
        except Exception:
            return None
    # Try glob any run directory and pick the newest
    wandb_dir = output_base / "wandb"
    if wandb_dir.exists():
        runs = sorted(wandb_dir.glob("run-*/files/wandb-metadata.json"))
        if runs:
            try:
                return json.loads(runs[-1].read_text()).get("id")
            except Exception:
                return None
    return None


def _load_student_from_ckpt(ckpt_path: Path) -> tuple[SentenceTransformer, dict]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    base_model = ckpt["student_model_name"]
    hp = ckpt.get("hyper_parameters", {})
    model = SentenceTransformer(base_model)
    sd = {k.removeprefix("student_model."): v for k, v in ckpt["state_dict"].items() if k.startswith("student_model.")}
    model.load_state_dict(sd)
    model.eval().bfloat16()
    return model, hp


def run_eval_and_update_wandb(
    ckpt_path: Path,
    run_id: str | None,
    benchmark_name: str,
    language: str | None,
    batch_size: int | None,
    num_workers: int | None,
    add_prefix: bool | None,
    project: str,
):
    model, hp = _load_student_from_ckpt(ckpt_path)

    # Resolve settings from ckpt hyperparameters if not given
    lang = language or hp.get("language", "eng")
    bsz = batch_size or hp.get("batch_size", 64)
    workers = num_workers or hp.get("num_workers", 4)
    use_prefix = add_prefix if add_prefix is not None else hp.get("add_prefix", True)

    if use_prefix:
        model.prompts = {
            PromptType.query.value: "query: ",
            PromptType.passage.value: "document: ",
        }

    output_base = _infer_output_base_from_ckpt(ckpt_path)
    output_folder = output_base / "mteb_eval"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Resolve task list
    if benchmark_name in ["on_eval_tasks", "on_train_end_tasks", "on_train_tasks"]:
        # Keep parity with run_mteb.py behavior
        import yaml

        with open("tasks.yaml") as f:
            tasks = yaml.safe_load(f)[lang][benchmark_name]
        evaluation = mteb.MTEB(tasks=tasks, task_langs=[lang])
    elif benchmark_name in ["MTEB(eng, v2)"]:
        tasks = mteb.get_benchmark(benchmark_name).tasks
        evaluation = mteb.MTEB(tasks=tasks, task_langs=[lang])
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = evaluation.run(
            model,
            output_folder=output_folder,
            num_workers=workers,
            overwrite_results=True,
            verbosity=1,
            encode_kwargs={"batch_size": bsz},
        )

    mteb_dict = {s.task_name: s.get_score() for s in scores}
    final_summary = {f"mteb_final/{k}": v for k, v in mteb_dict.items()}

    # WandB update
    if not run_id:
        run_id = _infer_run_id(output_base)
    if not run_id:
        raise RuntimeError("W&B run id not provided and could not be inferred from output directory.")

    wandb.init(project=project, id=run_id, resume="must")
    wandb.run.summary.update(final_summary)
    print(f"Updated W&B summary for run {run_id}")
    for k, v in sorted(final_summary.items()):
        print(f"  {k}: {v}")


def main():
    p = argparse.ArgumentParser(description="Posthoc MTEB eval and push to W&B summary")
    p.add_argument("--ckpt", type=str, required=True, help="Path to last.ckpt or checkpoints/<epoch>.ckpt")
    p.add_argument("--run_id", type=str, default=None, help="Existing W&B run id (infer if omitted)")
    p.add_argument(
        "--benchmark_name",
        type=str,
        default="MTEB(eng, v2)",
        help="Benchmark alias or name (e.g., MTEB(eng, v2) or on_train_end_tasks)",
    )
    p.add_argument("--language", type=str, default=None, help="Language override (defaults to ckpt hyperparameters)")
    p.add_argument("--batch_size", type=int, default=None, help="Batch size override for encoding")
    p.add_argument("--num_workers", type=int, default=None, help="Dataloader workers for evaluation")
    p.add_argument(
        "--add_prefix", type=lambda x: x.lower() == "true", default=None, help="Force add_prefix True/False"
    )
    p.add_argument("--project", type=str, default="distillation", help="W&B project name")
    args = p.parse_args()

    run_eval_and_update_wandb(
        ckpt_path=Path(args.ckpt),
        run_id=args.run_id,
        benchmark_name=args.benchmark_name,
        language=args.language,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        add_prefix=args.add_prefix,
        project=args.project,
    )


if __name__ == "__main__":
    main()
