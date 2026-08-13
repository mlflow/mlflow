# /// script
# dependencies = ["databricks-sdk"]
# [tool.uv]
# # Mirrors pyproject.toml's cooldown. `uv run --no-project` skips project
# # discovery, so without this the repo's 7-day guard against freshly-published
# # (possibly compromised) releases would not cover a dependency installed on a
# # job that holds an OIDC identity.
# exclude-newer = "P7D"
# ///
"""Mint a short-lived Databricks OAuth token from a GitHub Actions OIDC JWT.

`review.yml` routes Claude Code through the Databricks AI Gateway rather than the
Anthropic API, so it needs a Databricks bearer token instead of an API key. The
token comes from workload identity federation: GitHub issues an OIDC JWT scoped to
the job, and the service principal's federation policy exchanges it for an OAuth
token. Nothing is stored as a GitHub secret.

The Databricks CLI cannot do this. `databricks auth token` reads the U2M token
cache and explicitly rejects machine-to-machine auth, so the SDK owns the exchange.

`DATABRICKS_TOKEN_AUDIENCE` must equal the `audiences` entry on the federation
policy. Without it the SDK defaults the audience to the workspace token endpoint
URL, which the policy rejects as a mismatch.

Usage:
  DATABRICKS_HOST=... DATABRICKS_CLIENT_ID=... DATABRICKS_TOKEN_AUDIENCE=... \
    DATABRICKS_AUTH_TYPE=github-oidc uv run dev/mint_gateway_token.py

`MINT_TOKEN_LOG` and `MINT_TOKEN_SECRETS` are optional; see the functions below.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

from databricks.sdk.core import Config
from databricks.sdk.oauth import Token


def mint(cfg: Config, attempts: int = 3) -> Token:
    """Exchange the OIDC JWT for a Databricks token, retrying transient failures.

    The SDK posts to the token endpoint with a bare ``requests.post`` and no retry
    adapter. A caller that refreshes every couple of minutes runs this many times
    per session, so a single transient 5xx or network blip would end the session —
    exactly the failure the refresh exists to prevent.
    """
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
    """Record when this token was minted and how long it is valid for.

    Stdout is reserved for the bare credential, and a caller like Claude Code's
    `apiKeyHelper` captures stderr into its own process, so a stderr-only line
    never reaches the surrounding log. ``MINT_TOKEN_LOG`` makes the line count
    (did a refresh fire?) and the lifetime observable to whoever set the caller's
    refresh interval.

    Diagnostics must never fail the exchange, so every step here is best-effort.
    """
    # The SDK builds `expiry` from a naive local `datetime.now()`, so measuring it
    # against an aware UTC clock is only accidentally correct on UTC runners.
    # Match whatever tzinfo it carries — `datetime.now(None)` is naive local — and
    # log the lifetime directly rather than leaving a subtraction to the reader.
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
    """Append the minted credential to ``MINT_TOKEN_SECRETS`` for later scrubbing.

    This covers only the credentials handed out by this script. A caller shares
    its environment with whatever else it runs, so an agent can perform its own
    exchange and obtain a token this file never sees — that residual exposure is
    bounded by the principal's entitlements and the token's short life, not here.
    """
    if not (path := os.environ.get("MINT_TOKEN_SECRETS")):
        return
    try:
        with open(path, "a") as f:
            f.write(f"{token}\n")
    except OSError as e:
        print(f"Could not write {path}: {e}", file=sys.stderr)


def main() -> None:
    # Narrow the credential to gateway inference. The SDK's default `all-apis`
    # would let a leaked token reach any workspace API the principal is entitled
    # to, which is a wider blast radius than the Anthropic API key this replaces;
    # `ai-gateway` is advertised by the workspace's OIDC discovery document and is
    # the only capability a review actually needs.
    cfg = Config(scopes=["ai-gateway"])
    # This script prints a credential on stdout, so it must never be reachable by
    # ambient auth. Without this guard, running it on a developer machine happily
    # resolves the default ~/.databrickscfg profile and prints that PAT instead.
    if cfg.auth_type != "github-oidc":
        sys.exit(
            f"Refusing to print a {cfg.auth_type!r} credential. This script mints a "
            "federated token for CI; set DATABRICKS_AUTH_TYPE=github-oidc."
        )
    # One exchange, and everything derived from it. The SDK builds a fresh
    # credentials source per call, so `authenticate()` and `oauth_token()` each
    # hit the token endpoint and return *different* tokens — calling both would
    # print one credential while logging and scrubbing another.
    token = mint(cfg)
    if token.token_type != "Bearer" or not token.access_token:
        sys.exit(f"Expected a Bearer credential from the OIDC exchange, got {token.token_type!r}")
    log_diagnostics(token.expiry)
    record_token(token.access_token)
    print(token.access_token)


if __name__ == "__main__":
    main()
