import os
import re
import subprocess
import sys

BLACK = [sys.executable, "-m", "black"]
MESSAGE_REGEX = re.compile(r"^reformatted (.+)$")


def transform(stdout: str, is_maintainer: bool) -> str:
    if not stdout:
        return stdout
    transformed = []
    for line in stdout.splitlines():
        if m := MESSAGE_REGEX.match(line):
            path = m.group(1)
            command = (
                "`black .` or comment `@mlflow-automation autoformat`"
                if is_maintainer
                else "`black .`"
            )
            line = f"{path}: Unformatted file. Run {command} to format."

        transformed.append(line)
    return "\n".join(transformed) + "\n"


def main():
    if "GITHUB_ACTIONS" in os.environ:
        with subprocess.Popen(
            [
                *BLACK,
                *sys.argv[1:],
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as prc:
            stdout, stderr = prc.communicate()
            is_maintainer = os.environ.get("IS_MAINTAINER", "false").lower() == "true"
            sys.stdout.write(stdout)
            sys.stderr.write(transform(stderr, is_maintainer))
            sys.exit(prc.returncode)
    else:
        with subprocess.Popen(
            [
                *BLACK,
                *sys.argv[1:],
            ]
        ) as prc:
            prc.communicate()
            sys.exit(prc.returncode)


if __name__ == "__main__":
    main()
