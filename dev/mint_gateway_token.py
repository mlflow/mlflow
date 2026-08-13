# /// script
# dependencies = ["databricks-sdk"]
# [tool.uv]
# # `uv run --no-project` skips project discovery, so pyproject's cooldown would
# # not otherwise cover this dependency.
# exclude-newer = "P7D"
# ///
"""Mint a short-lived Databricks OAuth token from a GitHub Actions OIDC JWT.

`databricks auth token` cannot do this — it reads the U2M token cache and rejects
machine-to-machine auth — so the SDK owns the exchange.

`DATABRICKS_TOKEN_AUDIENCE` must equal the `audiences` entry on the federation
policy; the SDK otherwise defaults it to the workspace token endpoint URL, which
the policy rejects.

Usage:
  DATABRICKS_HOST=... DATABRICKS_CLIENT_ID=... DATABRICKS_TOKEN_AUDIENCE=... \
    DATABRICKS_AUTH_TYPE=github-oidc uv run dev/mint_gateway_token.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

from databricks.sdk.core import Config
from databricks.sdk.oauth import Token


def mint(cfg: Config, attempts: int = 3) -> Token:
    # The SDK's exchange is a bare `requests.post` with no retry adapter, and a
    # refreshing caller runs it many times per session.
    for attempt in range(1, attempts + 1):
        try:
            return cfg.oauth_token()
        except Exception as e:
            if attempt == attempts:
                raise
            delay = 2 ** (attempt - 1)
            print(f"Token exchange failed ({e}); retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def log_diagnostics(expiry: datetime | None) -> None:
    """Log the token's lifetime, which callers size their refresh interval against.

    Stdout is the credential and `apiKeyHelper` swallows stderr, so `MINT_TOKEN_LOG`
    is the only channel that reaches the surrounding log. Best-effort throughout:
    diagnostics must never fail the exchange.
    """
    # The SDK builds `expiry` from a naive local `datetime.now()`, so an aware UTC
    # clock would only match on UTC runners. `datetime.now(None)` is naive local.
    now = datetime.now(expiry.tzinfo) if expiry else None
    lifetime = f"{(expiry - now).total_seconds():.0f}s" if expiry and now else "unknown"
    line = f"{datetime.now(timezone.utc).isoformat()} minted, lifetime {lifetime}"
    print(line, file=sys.stderr)
    if path := os.environ.get("MINT_TOKEN_LOG"):
        try:
            with open(path, "a") as f:
                f.write(f"{line}\n")
        except OSError as e:
            print(f"Could not write {path}: {e}", file=sys.stderr)


def record_token(token: str) -> None:
    """Record the credential for the caller to scrub from its logs.

    Covers only what this script hands out; an agent sharing the environment can
    run its own exchange and obtain a token this never sees.
    """
    if not (path := os.environ.get("MINT_TOKEN_SECRETS")):
        return
    try:
        with open(path, "a") as f:
            f.write(f"{token}\n")
    except OSError as e:
        print(f"Could not write {path}: {e}", file=sys.stderr)


def main() -> None:
    # `ai-gateway` rather than the default `all-apis`: it is all a review needs.
    # Not a control on an attacker, who can request any scope — see review.yml.
    cfg = Config(scopes=["ai-gateway"])
    # This prints a credential on stdout, so ambient auth must never reach it —
    # otherwise running it locally prints the default profile's PAT.
    if cfg.auth_type != "github-oidc":
        sys.exit(
            f"Refusing to print a {cfg.auth_type!r} credential. This script mints a "
            "federated token for CI; set DATABRICKS_AUTH_TYPE=github-oidc."
        )
    # One exchange: the SDK mints a fresh token per call, so asking twice would
    # print one credential and log another's expiry.
    token = mint(cfg)
    if token.token_type != "Bearer" or not token.access_token:
        sys.exit(f"Expected a Bearer credential from the OIDC exchange, got {token.token_type!r}")
    log_diagnostics(token.expiry)
    record_token(token.access_token)
    print(token.access_token)


if __name__ == "__main__":
    main()
