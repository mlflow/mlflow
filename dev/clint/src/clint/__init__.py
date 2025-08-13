import argparse
import itertools
import json
import os
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


@dataclass
class Args:
    files: list[str]
    output_format: Literal["text", "json"]

    @classmethod
    def parse(cls) -> Self:
        parser = argparse.ArgumentParser(description="Custom linter for mlflow.")
        parser.add_argument("files", nargs="+", help="Files to lint.")
        parser.add_argument("--output-format", default="text")
        args, _ = parser.parse_known_args()
        return cls(files=args.files, output_format=args.output_format)


def main() -> None:
    config = Config.load()
    args = Args.parse()
    files = args.files
    if config.exclude:
        regex = re.compile("|".join(map(re.escape, config.exclude)))
        files = [f for f in files if not regex.match(f) and os.path.exists(f)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pickle `SymbolIndex` to avoid expensive serialization overhead when passing
        # the large index object to multiple worker processes
        index_path = Path(tmp_dir) / "symbol_index.pkl"
        SymbolIndex.build().save(index_path)
        with ProcessPoolExecutor() as pool:
            futures = [pool.submit(lint_file, Path(f), config, index_path) for f in files]
            violations_iter = itertools.chain.from_iterable(
                f.result() for f in as_completed(futures)
            )
            if violations := list(violations_iter):
                if args.output_format == "json":
                    sys.stdout.write(json.dumps([v.json() for v in violations]))
                elif args.output_format == "text":
                    sys.stderr.write("\n".join(map(str, violations)) + "\n")
                sys.exit(1)


if __name__ == "__main__":
    main()
