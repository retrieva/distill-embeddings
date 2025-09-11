import argparse
import json
import sys
from pathlib import Path

from .posthoc_eval_to_wandb import run_eval_and_update_wandb


def find_ckpt(exp_dir: Path, epoch: int | None) -> Path | None:
    """Pick last.ckpt or a specific epoch ckpt if requested."""
    if epoch is None:
        last = exp_dir / "last.ckpt"
        if last.exists():
            return last
        ckpt_dir = exp_dir / "checkpoints"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                return ckpts[-1]
        return None
    else:
        # Prefer zero-padded epoch name like 03.ckpt, but try non-padded as fallback
        cand = exp_dir / "checkpoints" / f"{epoch:02d}.ckpt"
        if cand.exists():
            return cand
        cand = exp_dir / "checkpoints" / f"{epoch}.ckpt"
        return cand if cand.exists() else None


def summary_has_mteb(exp_dir: Path, prefix: str = "mteb_final/") -> bool:
    summary = exp_dir / "wandb" / "latest-run" / "files" / "wandb-summary.json"
    if not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text())
    except Exception:
        return False
    return any(k.startswith(prefix) for k in data.keys())


def main():
    p = argparse.ArgumentParser(description="Batch posthoc MTEB eval and push to W&B summary")
    p.add_argument("--root", type=str, default="output/result", help="Root of experiments")
    p.add_argument("--pattern", type=str, default="*/*/*/*", help="Glob under root to locate experiment dirs")
    p.add_argument("--epoch", type=int, default=None, help="Use checkpoints/<epoch>.ckpt instead of last.ckpt")
    p.add_argument("--skip_if_exists", action="store_true", help="Skip runs that already have mteb_final/* in summary")
    # Pass-through to evaluation
    p.add_argument("--benchmark_name", type=str, default="MTEB(eng, v2)")
    p.add_argument("--language", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--add_prefix", type=lambda x: x.lower() == "true", default=None)
    p.add_argument("--project", type=str, default="distillation")
    p.add_argument("--reuse_cached", action="store_true", help="Reuse cached results in mteb_eval instead of recomputing")
    p.add_argument("--cached_only", action="store_true", help="Skip evaluation; only push from existing mteb_eval results")
    args = p.parse_args()

    root = Path(args.root)
    exp_dirs = sorted([p for p in root.glob(args.pattern) if p.is_dir()])
    if not exp_dirs:
        print(f"No experiment directories found under {root} with pattern {args.pattern}")
        return
    processed = 0
    skipped = 0
    for exp in exp_dirs:
        if args.skip_if_exists and summary_has_mteb(exp):
            print(f"[skip exists] {exp}")
            skipped += 1
            continue
        ckpt = find_ckpt(exp, args.epoch)
        if not ckpt:
            if args.cached_only and (exp / "mteb_eval").exists():
                # Proceed without a ckpt by pointing to the experiment dir; will use cached_only path
                ckpt = exp
            else:
                print(f"[skip no ckpt] {exp}")
                skipped += 1
                continue
        try:
            run_eval_and_update_wandb(
                ckpt_path=ckpt,
                run_id=None,
                benchmark_name=args.benchmark_name,
                language=args.language,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                add_prefix=args.add_prefix,
                project=args.project,
                reuse_cached=args.reuse_cached,
                cached_only=args.cached_only,
            )
            processed += 1
        except Exception as e:
            print(f"[error] {exp}: {e}", file=sys.stderr)
    print(f"Done. processed={processed}, skipped={skipped}, total={len(exp_dirs)}")


if __name__ == "__main__":
    main()
