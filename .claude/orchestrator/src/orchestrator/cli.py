"""CLI entry point for the orchestrator.

Usage:

    mlflow-reviewer <PR_NUMBER> [--lite] [--no-cache] [--hybrid] [--no-dry-run]

By default, runs in dry-run mode and emits the would-be review JSON to stdout.
The GitHub Actions workflow (Stack 3) flips `--no-dry-run` on to enable
posting in production.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from orchestrator.orchestrator import Config, Mode, run_review


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlflow-reviewer",
        description="Adversarial-checklist code review for an MLflow PR.",
    )
    parser.add_argument("pr_number", type=int, help="Pull request number to review.")
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Run only the reviewer-standalone agent. Cheaper, lower recall.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-run even if the head SHA was already reviewed.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use Opus for the spotter discovery step; Sonnet elsewhere.",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Post comments to GitHub instead of emitting JSON to stdout. Stack 3 wires this in.",
    )
    parser.add_argument(
        "--repo",
        default="mlflow/mlflow",
        help="Repository in <owner>/<name> form. Defaults to mlflow/mlflow.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config(
        pr_number=args.pr_number,
        mode=Mode.LITE if args.lite else Mode.DEFAULT,
        no_cache=args.no_cache,
        hybrid=args.hybrid,
        dry_run=not args.no_dry_run,
        repo=args.repo,
    )
    asyncio.run(run_review(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
