#!/usr/bin/env python3
"""Update W&B runs so config.data_name uses a new value."""

import argparse
import sys
from collections.abc import Iterable

import wandb


def iter_target_runs(
    api: wandb.Api,
    entity: str,
    project: str,
    old_value: str,
    student_model: str,
) -> Iterable[wandb.apis.public.runs.Run]:
    return api.runs(
        f"{entity}/{project}",
        filters={
            "config.data_name": old_value,
            "config.student_model": student_model,
        },
    )


def update_runs(
    entity: str,
    project: str,
    old_value: str,
    new_value: str,
    student_model: str,
    dry_run: bool,
) -> None:
    api = wandb.Api()
    runs = list(iter_target_runs(api, entity, project, old_value, student_model))
    if not runs:
        print("No runs matched the supplied filters.")
        return

    for run in runs:
        current = run.config.get("data_name")
        if current == new_value:
            print(f"[skip] {run.path}: already {new_value}")
            continue

        prefix = "[dry-run]" if dry_run else "[update]"
        print(f"{prefix} {run.path}: {current} -> {new_value}")

        if dry_run:
            continue

        run.config["data_name"] = new_value
        run.update()
        run.load(force=True)  # refresh local cache so repeated runs stay in sync


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Rewrite config.data_name for selected W&B runs.")
    parser.add_argument("--entity", default="retrieva-research")
    parser.add_argument("--project", default="distillation")
    parser.add_argument("--old", default="fineweb", help="current data_name value")
    parser.add_argument("--new", required=True, help="replacement data_name value")
    parser.add_argument(
        "--student-model",
        default="nomic-ai/modernbert-embed-base-unsupervised",
        help="required config.student_model value",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the runs that would change without updating them",
    )
    args = parser.parse_args(argv)

    update_runs(
        entity=args.entity,
        project=args.project,
        old_value=args.old,
        new_value=args.new,
        student_model=args.student_model,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
