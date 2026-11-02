import base64
import json
import os
import sys
from urllib.parse import quote, quote_plus


def redact(text: str, secrets: list[str]) -> str:
    values = set()
    for secret in secrets:
        if secret:
            values.update([
                secret,
                json.dumps(secret)[1:-1],
                quote(secret, safe=""),
                quote_plus(secret),
                base64.b64encode(secret.encode()).decode(),
                base64.urlsafe_b64encode(secret.encode()).decode(),
            ])
    for value in tuple(values):
        while len(value) <= len(text):
            values.add(value)
            escaped = json.dumps(value)[1:-1]
            if escaped == value:
                break
            value = escaped
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return text


def main() -> None:
    secrets = os.environ.get("SECRETS_TO_REDACT")
    if secrets is None:
        sys.exit("SECRETS_TO_REDACT must be set")
    sys.stdout.write(redact(sys.stdin.read(), secrets.splitlines()))


if __name__ == "__main__":
    main()
