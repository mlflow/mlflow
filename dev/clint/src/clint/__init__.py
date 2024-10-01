from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from clint.config import Config
from clint.linter import lint_file


@dataclass
class Args:
    files: list[str]
    output_format: Literal["text", "json"]

    @classmethod
    def parse(cls) -> Args:
        parser = argparse.ArgumentParser(description="Custom linter for mlflow.")
        parser.add_argument("files", nargs="+", help="Files to lint.")
        parser.add_argument("--output-format", default="text")
        args, _ = parser.parse_known_args()
        return cls(files=args.files, output_format=args.output_format)


def main():
    config = Config.load()
    EXCLUDE_REGEX = re.compile("|".join(map(re.escape, config.exclude)))
    args = Args.parse()
    with ProcessPoolExecutor() as pool:
        futures = [
            pool.submit(lint_file, Path(f))
            for f in args.files
            if not EXCLUDE_REGEX.match(f) and os.path.exists(f)
        ]
        violations_iter = itertools.chain.from_iterable(f.result() for f in as_completed(futures))
        if violations := list(violations_iter):
            if args.output_format == "json":
                sys.stdout.write(json.dumps([v.json() for v in violations]))
            elif args.output_format == "text":
                sys.stderr.write("\n".join(map(str, violations)) + "\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
