"""
Script to automatically remove @experimental decorators from functions
that have been experimental for more than 6 months.
"""

import argparse
import ast
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen


@dataclass
class ExperimentalDecorator:
    version: str
    line_number: int
    end_line_number: int
    column: int
    age_days: int
    content: str


def get_tracked_python_files() -> list[Path]:
    """Get all tracked Python files in the repository."""
    result = subprocess.check_output(["git", "ls-files", "*.py"], text=True)
    return [Path(f) for f in result.strip().split("\n") if f]


def get_mlflow_release_dates() -> dict[str, datetime]:
    """Fetch MLflow release dates from PyPI API."""
    with urlopen("https://pypi.org/pypi/mlflow/json") as response:
        data = json.loads(response.read().decode())

    release_dates: dict[str, datetime] = {}
    for version, releases in data["releases"].items():
        if releases:  # Some versions might have empty release lists
            # Get the earliest release date for this version
            upload_times: list[str] = [r["upload_time"] for r in releases if "upload_time" in r]
            if upload_times:
                earliest_time = min(upload_times)
                # Parse ISO format datetime and convert to UTC
                release_date = datetime.fromisoformat(earliest_time.replace("Z", "+00:00"))
                if release_date.tzinfo is None:
                    release_date = release_date.replace(tzinfo=timezone.utc)
                release_dates[version] = release_date

    return release_dates


def find_experimental_decorators(
    file_path: Path, release_dates: dict[str, datetime], now: datetime
) -> list[ExperimentalDecorator]:
    """
    Find all @experimental decorators in a Python file using AST and return their information
    with computed age.
    """
    content = file_path.read_text()
    tree = ast.parse(content)
    decorators: list[ExperimentalDecorator] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue

            if not (isinstance(decorator.func, ast.Name) and decorator.func.id == "experimental"):
                continue

            version = _extract_version_from_ast_decorator(decorator)
            if not version or version not in release_dates:
                continue

            release_date = release_dates[version]
            age_days = (now - release_date).days

            decorator_info = ExperimentalDecorator(
                version=version,
                line_number=decorator.lineno,
                end_line_number=decorator.end_lineno or decorator.lineno,
                column=decorator.col_offset + 1,  # 1-indexed
                age_days=age_days,
                content=ast.unparse(decorator),
            )
            decorators.append(decorator_info)

    return decorators


def _extract_version_from_ast_decorator(decorator: ast.Call) -> str | None:
    """Extract version string from AST decorator node."""
    for keyword in decorator.keywords:
        if keyword.arg == "version" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return None


def remove_decorators_from_file(
    file_path: Path,
    decorators_to_remove: list[ExperimentalDecorator],
    dry_run: bool,
) -> list[ExperimentalDecorator]:
    if not decorators_to_remove:
        return []

    lines = file_path.read_text().splitlines(keepends=True)
    # Create a set of line numbers to remove for quick lookup (handle ranges)
    lines_to_remove: set[int] = set()
    for decorator in decorators_to_remove:
        for line_num in range(decorator.line_number, decorator.end_line_number + 1):
            lines_to_remove.add(line_num)

    new_lines: list[str] = []

    for line_num, line in enumerate(lines, 1):
        if line_num not in lines_to_remove:
            new_lines.append(line)

    if not dry_run:
        file_path.write_text("".join(new_lines))

    return decorators_to_remove


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Remove @experimental decorators older than 6 months"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be removed without making changes"
    )
    parser.add_argument(
        "files", nargs="*", help="Python files to process (defaults to all tracked Python files)"
    )

    args = parser.parse_args()
    release_dates = get_mlflow_release_dates()
    # Calculate cutoff date (6 months ago from now)
    now = datetime.now(timezone.utc)
    cutoff_date = now - timedelta(days=6 * 30)  # Approximate 6 months
    print(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    python_files = [Path(f) for f in args.files] if args.files else get_tracked_python_files()
    for file_path in python_files:
        if not file_path.exists():
            continue

        # First, find all experimental decorators in the file with computed ages
        decorators = find_experimental_decorators(file_path, release_dates, now)
        if not decorators:
            continue

        # Filter to only decorators that should be removed (older than 6 months)
        old_decorators = [d for d in decorators if d.age_days > 6 * 30]  # 6 months approx
        if not old_decorators:
            continue

        # Remove old decorators
        removed = remove_decorators_from_file(file_path, old_decorators, args.dry_run)
        if removed:
            for decorator in removed:
                action = "Would remove" if args.dry_run else "Removed"
                print(
                    f"{file_path}:{decorator.line_number}:{decorator.column}: "
                    f"{action} {decorator.content} (age: {decorator.age_days} days)"
                )


if __name__ == "__main__":
    main()
