import functools
import json
import os
import re
import subprocess
import sys

RUFF = [sys.executable, "-m", "ruff"]
MESSAGE_REGEX = re.compile(r"^.+:\d+:\d+: ([A-Z0-9]+) (\[\*\] )?.+$")


@functools.lru_cache
def get_rule_name(code: str) -> str:
    out = subprocess.check_output([*RUFF, "rule", "--format", "json", code], text=True).strip()
    return json.loads(out)["name"]


def transform(stdout: str) -> str:
    transformed = []
    for line in stdout.splitlines():
        if m := MESSAGE_REGEX.match(line):
            if m.group(2) is not None:
                line = f"{line}. Fixable via `ruff --fix .`."
            else:
                name = get_rule_name(m.group(1))
                line = f"{line}. See https://beta.ruff.rs/docs/rules/{name} for details."
        transformed.append(line)
    return "\n".join(transformed)


def main():
    cmd = [*RUFF, *sys.argv[1:]]
    if "GITHUB_ACTIONS" in os.environ:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as prc:
            stdout, stderr = prc.communicate()
            sys.stdout.write(transform(stdout))
            sys.stderr.write(stderr)
            sys.exit(prc.returncode)
    else:
        with subprocess.Popen(cmd) as prc:
            prc.communicate()
            sys.exit(prc.returncode)


if __name__ == "__main__":
    main()
