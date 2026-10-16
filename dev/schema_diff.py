"""Operator CLI tool for analyzing MLflow database migration safety.

Usage:
    # Check upgrade path from current DB to latest
    python dev/schema_diff.py --db-url "postgresql://..."

    # Check specific revision range (no DB needed)
    python dev/schema_diff.py --from-revision abc123 --to-revision def456

    # JSON output for CI integration
    python dev/schema_diff.py --from-revision abc123 --to-revision def456 --json

Exit codes: 0=all safe, 1=cautious, 2=breaking
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MLflow database migration safety for upgrade paths."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--db-url",
        help="Database URL to check (reads current revision from DB).",
    )
    group.add_argument(
        "--from-revision",
        help="Starting revision (use with --to-revision, no DB connection needed).",
    )
    parser.add_argument(
        "--to-revision",
        help="Target revision. Defaults to latest head if not specified.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        default=False,
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    from mlflow.store.db.utils import (
        _get_latest_schema_revision,
        _get_schema_version,
        create_sqlalchemy_engine_with_retry,
    )
    from mlflow.store.db_migrations.migration_classifier import (
        MigrationSafety,
        classify_range,
    )

    if args.db_url:
        engine = create_sqlalchemy_engine_with_retry(args.db_url)
        from_rev = _get_schema_version(engine)
        engine.dispose()
    else:
        from_rev = args.from_revision

    to_rev = args.to_revision or _get_latest_schema_revision()

    if from_rev == to_rev:
        if args.output_json:
            print(json.dumps({"status": "up_to_date", "current": from_rev}))
        else:
            print(f"Schema is up to date (revision {from_rev}).")
        sys.exit(0)

    analyses = classify_range(from_rev, to_rev)

    worst = MigrationSafety.SAFE
    for a in analyses:
        if a.safety == MigrationSafety.BREAKING:
            worst = MigrationSafety.BREAKING
            break
        if a.safety == MigrationSafety.CAUTIOUS:
            worst = MigrationSafety.CAUTIOUS

    if args.output_json:
        result = {
            "from_revision": from_rev,
            "to_revision": to_rev,
            "overall_safety": worst.value,
            "pending_migrations": [
                {
                    "revision": a.revision,
                    "safety": a.safety.value,
                    "operations": [
                        {"name": op.name, "safety": op.safety.value, "detail": op.detail}
                        for op in a.operations
                    ],
                    "notes": a.notes,
                }
                for a in analyses
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"From revision: {from_rev}")
        print(f"To revision:   {to_rev}")
        print(f"Pending migrations: {len(analyses)}")
        print()

        for a in analyses:
            icon = {"safe": "+", "cautious": "~", "breaking": "!"}[a.safety.value]
            print(f"  [{icon}] {a.revision} ({a.safety.value.upper()})")
            for op in a.operations:
                print(f"      {op.name}: {op.detail}")
            for note in a.notes:
                print(f"      note: {note}")

        print()
        if worst == MigrationSafety.SAFE:
            print("All pending migrations are SAFE. Zero-downtime upgrade is possible.")
        elif worst == MigrationSafety.CAUTIOUS:
            print("Some migrations require CAUTION. Review before upgrading.")
        else:
            print("BREAKING migrations detected. Downtime is required.")

    exit_code = {"safe": 0, "cautious": 1, "breaking": 2}[worst.value]
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
