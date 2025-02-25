import os
import re
import subprocess
import sys

RUFF = [sys.executable, "-m", "ruff", "check"]
MESSAGE_REGEX = re.compile(r"^.+:\d+:\d+: ([A-Z0-9]+) (\[\*\] )?.+$")


def transform(stdout: str, is_maintainer: bool) -> str:
    transformed = []
    for line in stdout.splitlines():
        if m := MESSAGE_REGEX.match(line):
            if m.group(2) is not None:
                command = (
                    "`ruff --fix .` or comment `/autoformat`" if is_maintainer else "`ruff --fix .`"
                )
                line = f"{line}. Run {command} to fix this error."
            else:
                line = (
                    f"{line}. See https://docs.astral.sh/ruff/rules/{m.group(1)} for how to fix "
                    f"this error."
                )
        transformed.append(line)
    return "\n".join(transformed)


def main():
    if "NO_FIX" in os.environ:
        with subprocess.Popen(
            [
                *RUFF,
                *sys.argv[1:],
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as prc:
            stdout, stderr = prc.communicate()
            is_maintainer = os.environ.get("IS_MAINTAINER", "false").lower() == "true"
            sys.stdout.write(transform(stdout, is_maintainer))
            sys.stderr.write(stderr)
            sys.exit(prc.returncode)
    else:
        with subprocess.Popen(
            [
                *RUFF,
                "--fix",
                "--exit-non-zero-on-fix",
                *sys.argv[1:],
            ]
        ) as prc:
            prc.communicate()
            sys.exit(prc.returncode)


if __name__ == "__main__":
    main()
