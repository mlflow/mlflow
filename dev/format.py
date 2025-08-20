import os
import re
import subprocess
import sys

RUFF_FORMAT = [sys.executable, "-m", "ruff", "format"]
MESSAGE_REGEX = re.compile(r"^Would reformat: (.+)$")


def transform(stdout: str, is_maintainer: bool) -> str:
    if not stdout:
        return stdout
    transformed = []
    for line in stdout.splitlines():
        if m := MESSAGE_REGEX.match(line):
            path = m.group(1)
            command = (
                "`ruff format .` or comment `/autoformat`" if is_maintainer else "`ruff format .`"
            )
            # As a workaround for https://github.com/orgs/community/discussions/165826,
            # add fake line:column numbers (1:1)
            line = f"{path}:1:1: Unformatted file. Run {command} to format."

        transformed.append(line)
    return "\n".join(transformed) + "\n"


def main():
    if "NO_FIX" in os.environ:
        with subprocess.Popen(
            [
                *RUFF_FORMAT,
                "--check",
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
                *RUFF_FORMAT,
                *sys.argv[1:],
            ]
        ) as prc:
            prc.communicate()
            sys.exit(prc.returncode)


if __name__ == "__main__":
    main()
