"""
Migrate MLflow FileStore data to a SQLite database.

Usage:
    mlflow migrate-filestore --source ./mlruns --target sqlite:///mlflow.db

Or directly:
    uv run python fs2db/src/migrate.py \
        --source /tmp/fs2db/v3.5.1/ \
        --target sqlite:////tmp/migrated.db
"""

import argparse
from pathlib import Path

from mlflow.store.fs2db import migrate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate MLflow FileStore data to a SQLite database"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Root directory containing mlruns/ FileStore data",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="SQLite URI (e.g. sqlite:////tmp/migrated.db)",
    )
    args = parser.parse_args()

    if not args.target.startswith("sqlite:///"):
        raise SystemExit("--target must be a SQLite URI starting with 'sqlite:///'")

    source = Path(args.source).resolve()
    if not source.is_dir():
        raise SystemExit(f"--source directory does not exist: {source}")

    migrate(source, args.target)


if __name__ == "__main__":
    main()
