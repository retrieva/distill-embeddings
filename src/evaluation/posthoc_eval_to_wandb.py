import argparse
import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import mteb
import torch
import yaml
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


def _read_run_id_from_metadata(path: Path) -> str | None:
    """Read W&B run id from a metadata JSON file with robust fallbacks.

    - First try strict JSON decode and lookup 'id'.
    - If that fails, use a regex to extract "id": "..." even from partially written files.
    """
    try:
        data = json.loads(path.read_text())
        rid = data.get("id") if isinstance(data, dict) else None
        if isinstance(rid, str) and rid:
            return rid
    except Exception:
        pass
    try:
        m = re.search(r'"id"\s*:\s*"([^"]+)"', path.read_text(errors="ignore"))
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _infer_run_id(output_base: Path) -> str | None:
    # Env override (helpful for batch replays or CI)
    env_rid = os.environ.get("WANDB_RUN_ID") or os.environ.get("RUN_ID")
    if env_rid:
        return env_rid

    # Try W&B latest-run metadata
    meta = output_base / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
    if meta.exists():
        rid = _read_run_id_from_metadata(meta)
        if rid:
            return rid
    # Try glob any run directory and pick the newest
    wandb_dir = output_base / "wandb"
    if wandb_dir.exists():
        runs = sorted(wandb_dir.glob("run-*/files/wandb-metadata.json"))
        if runs:
            rid = _read_run_id_from_metadata(runs[-1])
            if rid:
                return rid
    return None


def _find_first_weight_file(root: Path) -> Path | None:
    """Find a plausible weight file under a merged DeepSpeed output.

    Prefer an actual file named 'pytorch_model.bin'. If not present, pick the largest
    '*.bin' or '*.pt' file found recursively. Handle the case where a directory named
    'pytorch_model.bin' is created by some versions.
    """
    if root.is_file():
        return root
    cand = root / "pytorch_model.bin"
    if cand.exists() and cand.is_file():
        return cand
    # zero_to_fp32 sometimes creates a directory named 'pytorch_model.bin'
    if cand.exists() and cand.is_dir():
        nested = cand / "pytorch_model.bin"
        if nested.exists() and nested.is_file():
            return nested
    # Fallback: pick largest bin/pt file
    files = [p for p in root.rglob("*.bin")] + [p for p in root.rglob("*.pt")]
    files = [p for p in files if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[0]


def _merge_deepspeed_shards(ckpt_dir: Path) -> Path:
    """Run zero_to_fp32.py; return path to merged output (file or directory)."""
    zero_script = ckpt_dir / "zero_to_fp32.py"
    if not zero_script.exists():
        raise RuntimeError(f"DeepSpeed checkpoint at {ckpt_dir} missing zero_to_fp32.py")
    import sys as _sys
    tmpdir = tempfile.mkdtemp(prefix="ds_merge_")
    merged_path = Path(tmpdir) / "pytorch_model.bin"
    cmd = [_sys.executable, str(zero_script), str(ckpt_dir), str(merged_path)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"zero_to_fp32 failed: {e}")
    # Depending on DS version, either file or a directory may be created.
    return merged_path if merged_path.exists() else Path(tmpdir)


def _load_wandb_config(output_base: Path) -> dict:
    cfg_path = output_base / "wandb" / "latest-run" / "files" / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        raw = yaml.safe_load(cfg_path.read_text())
        return {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}
    except Exception:
        return {}


def _infer_wandb_entity_project(output_base: Path) -> tuple[str | None, str | None]:
    """Infer W&B entity and project from wandb-metadata.json if available."""
    meta = output_base / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
    if not meta.exists():
        return None, None
    try:
        data = json.loads(meta.read_text())
        ent = data.get("entity")
        proj = data.get("project")
        return ent, proj
    except Exception:
        return None, None


def _extract_student_state_dict(full_sd: dict) -> dict:
    out = {}
    for k, v in full_sd.items():
        if ".student_model." in k:
            tail = k.split(".student_model.", 1)[1]
            out[tail] = v
        elif k.startswith("student_model."):
            out[k[len("student_model.") :]] = v
    return out


def _load_student_from_ds_dir(ckpt_dir: Path, output_base: Path, base_model_override: str | None = None) -> tuple[SentenceTransformer, dict]:
    cfg = _load_wandb_config(output_base)
    base_model = base_model_override or cfg.get("student_model")
    if not base_model:
        raise RuntimeError(
            "Could not infer student_model from W&B config. Provide a Lightning ckpt file or ensure W&B config exists."
        )

    merged_output = _merge_deepspeed_shards(ckpt_dir)
    weight_file = _find_first_weight_file(merged_output)
    if not weight_file:
        raise RuntimeError(f"Merged weights not found under {merged_output}")
    print(f"[posthoc] Using merged weight file: {weight_file}")
    full_sd = torch.load(str(weight_file), map_location="cpu")

    st_sd = _extract_student_state_dict(full_sd)
    if not st_sd:
        # Fallback 1: raw HF keys mapped under SentenceTransformer's first module
        if isinstance(full_sd, dict) and all(isinstance(k, str) for k in full_sd.keys()):
            st_sd = {f"0.auto_model.{k}": v for k, v in full_sd.items()}
    if not st_sd:
        # Fallback 2: keys already look like '0.xxx' or '1.xxx'
        st_sd = {k: v for k, v in full_sd.items() if isinstance(k, str) and (k.startswith("0.") or k.startswith("1."))}
    if not st_sd:
        raise RuntimeError("Could not map merged state_dict to SentenceTransformer keys.")

    model = SentenceTransformer(base_model)
    model.load_state_dict(st_sd, strict=False)
    model.eval().bfloat16()
    hp = {
        "language": cfg.get("language", "eng"),
        "batch_size": cfg.get("batch_size", 64),
        "num_workers": cfg.get("num_workers", 4),
        "add_prefix": cfg.get("add_prefix", True),
    }
    return model, hp


def _load_student_from_ckpt(ckpt_path: Path, base_model_override: str | None = None) -> tuple[SentenceTransformer, dict]:
    if ckpt_path.is_dir():
        output_base = _infer_output_base_from_ckpt(ckpt_path)
        return _load_student_from_ds_dir(ckpt_path, output_base, base_model_override)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    base_model = base_model_override or ckpt["student_model_name"]
    hp = ckpt.get("hyper_parameters", {})
    model = SentenceTransformer(base_model)
    sd = {k.removeprefix("student_model."): v for k, v in ckpt["state_dict"].items() if k.startswith("student_model.")}
    model.load_state_dict(sd)
    model.eval().bfloat16()
    return model, hp


def _collect_cached_mteb_scores(output_folder: Path) -> dict[str, float]:
    """Scan mteb_eval and aggregate main_score per task from existing JSON files.

    Tries to be robust to different MTEB versions by checking several common shapes:
      - {"scores": [{"split": "test", "main_score": ...}, ...]}
      - {"test": {"main_score": ...}}
      - {"main_score": ...}

    For task name, prefers JSON field 'task_name', else falls back to parent directory name.
    """
    results: dict[str, float] = {}

    def extract_main_score(d: dict) -> float | None:
        # Prefer a 'test' split entry
        scores = d.get("scores")
        if isinstance(scores, list):
            for entry in scores:
                if isinstance(entry, dict) and entry.get("split") == "test" and isinstance(entry.get("main_score"), (int, float)):
                    return float(entry["main_score"])
            for entry in scores:
                if isinstance(entry, dict) and isinstance(entry.get("main_score"), (int, float)):
                    return float(entry["main_score"])
        test = d.get("test")
        if isinstance(test, dict) and isinstance(test.get("main_score"), (int, float)):
            return float(test["main_score"])
        if isinstance(d.get("main_score"), (int, float)):
            return float(d["main_score"])
        return None

    for p in output_folder.rglob("*.json"):
        if p.name != "results.json" and not p.name.endswith("results.json"):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        task = data.get("task_name") or p.parent.name
        ms = extract_main_score(data)
        if task and ms is not None:
            # Prefer the latest file seen wins; rglob order is path-sorted, but OK for our use case
            results[task] = ms
    return results


def run_eval_and_update_wandb(
    ckpt_path: Path,
    run_id: str | None,
    benchmark_name: str,
    language: str | None,
    batch_size: int | None,
    num_workers: int | None,
    add_prefix: bool | None,
    project: str,
    entity: str | None = None,
    student_model: str | None = None,
    reuse_cached: bool = False,
    cached_only: bool = False,
    resume_mode: str = "must",
):
    # Load model only if we will evaluate; cached_only mode skips model load
    model = None
    hp: dict = {}
    if not cached_only:
        model, hp = _load_student_from_ckpt(ckpt_path, student_model)

    # Resolve settings from ckpt hyperparameters if not given
    lang = language or hp.get("language", "eng")
    bsz = batch_size or hp.get("batch_size", 64)
    workers = num_workers or hp.get("num_workers", 4)
    use_prefix = add_prefix if add_prefix is not None else hp.get("add_prefix", True)

    if not cached_only and use_prefix:
        model.prompts = {
            PromptType.query.value: "query: ",
            PromptType.passage.value: "document: ",
        }
    # Determine output_base robustly so cached-only works even without a ckpt file
    try:
        ckpt_p = ckpt_path
        if ckpt_p.is_dir() and (ckpt_p / "mteb_eval").exists():
            output_base = ckpt_p
        else:
            output_base = _infer_output_base_from_ckpt(ckpt_p)
    except Exception:
        # Fall back to parent
        output_base = ckpt_path.parent
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

    mteb_dict = _collect_cached_mteb_scores(output_folder)
    scores = []
    if not cached_only:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = evaluation.run(
                model,
                output_folder=output_folder,
                num_workers=workers,
                overwrite_results=not reuse_cached,
                verbosity=1,
                encode_kwargs={"batch_size": bsz},
            )
        # Merge fresh scores on top of cached
        for s in scores:
            try:
                mteb_dict[s.task_name] = s.get_score()
            except Exception:
                pass
    final_summary = {f"mteb_final/{k}": v for k, v in mteb_dict.items()}

    # WandB update
    if not run_id:
        run_id = _infer_run_id(output_base)
    if not run_id:
        meta = output_base / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
        raise RuntimeError(
            "W&B run id not provided and could not be inferred from output directory. "
            f"Tried: {meta} (exists={meta.exists()}), env WANDB_RUN_ID. "
            "Pass --run_id explicitly or set WANDB_RUN_ID."
        )

    # Resolve entity/project from metadata if not provided
    meta_entity, meta_project = _infer_wandb_entity_project(output_base)
    ent = entity or os.environ.get("WANDB_ENTITY") or meta_entity
    proj = project or meta_project or project

    try:
        wandb.init(project=proj, entity=ent, id=run_id, resume=resume_mode, dir=str(output_base))
    except Exception as e:
        # Helpful diagnostics
        raise RuntimeError(
            f"wandb.init failed with resume={resume_mode}, project={proj}, entity={ent}, id={run_id}: {e}. "
            "If this is an existing cloud run, ensure the correct entity/project. "
            "Otherwise set --resume_mode allow to create if missing."
        )
    wandb.run.summary.update(final_summary)
    print(f"Updated W&B summary for run {run_id}")
    for k, v in sorted(final_summary.items()):
        print(f"  {k}: {v}")


def main():
    p = argparse.ArgumentParser(description="Posthoc MTEB eval and push to W&B summary")
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help=(
            "Path to Lightning .ckpt file or DeepSpeed checkpoint directory (e.g., .../last.ckpt). "
            "In --cached_only mode, you may pass the experiment directory containing mteb_eval/ instead."
        ),
    )
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
    p.add_argument("--entity", type=str, default=None, help="W&B entity (user or team)")
    p.add_argument("--student_model", type=str, default=None, help="Override base student model (for DS ckpts without W&B config)")
    p.add_argument("--reuse_cached", action="store_true", help="Reuse cached MTEB results under mteb_eval instead of recomputing")
    p.add_argument("--cached_only", action="store_true", help="Do not run evaluation; only read existing mteb_eval results and push to W&B")
    p.add_argument("--resume_mode", type=str, default="must", choices=["must", "allow"], help="W&B resume behavior")
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
        entity=args.entity,
        student_model=args.student_model,
        reuse_cached=args.reuse_cached,
        cached_only=args.cached_only,
        resume_mode=args.resume_mode,
    )


if __name__ == "__main__":
    main()
