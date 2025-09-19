import argparse
import itertools
import json
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from typing_extensions import Self

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import lint_file
from clint.utils import get_repo_root, resolve_paths


@dataclass
class Args:
    files: list[str]
    output_format: Literal["text", "json"]

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(description="Custom linter for mlflow.")
        parser.add_argument(
            "files",
            nargs="*",
            help="Files to lint. If not specified, lints all files in the current directory.",
        )
        parser.add_argument("--output-format", default="text")
        args, _ = parser.parse_known_args()
        return cls(files=args.files, output_format=args.output_format)


def main() -> None:
    config = Config.load()
    args = Args.parse()

    input_paths = [Path(f) for f in args.files]

    resolved_files = resolve_paths(input_paths)

    # Apply exclude filtering
    files: list[Path] = []
    if config.exclude:
        repo_root = get_repo_root()
        cwd = Path.cwd()
        regex = re.compile("|".join(map(re.escape, config.exclude)))
        for f in resolved_files:
            # Convert file path to be relative to repo root for exclude pattern matching
            repo_relative_path = (cwd / f).resolve().relative_to(repo_root)
            if not regex.match(repo_relative_path.as_posix()):
                files.append(f)
    else:
        files = resolved_files

    # Exit early if no files to lint
    if not files:
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pickle `SymbolIndex` to avoid expensive serialization overhead when passing
        # the large index object to multiple worker processes
        index_path = Path(tmp_dir) / "symbol_index.pkl"
        SymbolIndex.build().save(index_path)
        with ProcessPoolExecutor() as pool:
            futures = [pool.submit(lint_file, f, f.read_text(), config, index_path) for f in files]
            violations_iter = itertools.chain.from_iterable(
                f.result() for f in as_completed(futures)
            )
            if violations := list(violations_iter):
                if args.output_format == "json":
                    sys.stdout.write(json.dumps([v.json() for v in violations]))
                elif args.output_format == "text":
                    sys.stderr.write("\n".join(map(str, violations)) + "\n")
                count = len(violations)
                label = "error" if count == 1 else "errors"
                print(f"Found {count} {label}", file=sys.stderr)
                sys.exit(1)
            else:
                print("No errors found!", file=sys.stderr)


if __name__ == "__main__":
    main()
