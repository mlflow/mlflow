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
from datetime import datetime, timezone

from databricks.sdk.core import Config


def log_diagnostics(cfg: Config) -> None:
    """Record when this token was minted and when it expires.

    Stdout is reserved for the bare credential, and a caller like Claude Code's
    `apiKeyHelper` captures stderr into its own process, so a stderr-only line
    never reaches the surrounding log. Appending to ``MINT_TOKEN_LOG`` keeps both
    facts observable: the line count shows whether a refresh actually fired, and
    the expiry timestamps give the federated token's real lifetime, which any
    caller's refresh interval has to stay under.

    Diagnostics must never fail the exchange, so every step here is best-effort.
    """
    try:
        expiry = str(cfg.oauth_token().expiry)
    except Exception as e:
        expiry = f"unavailable ({e})"
    line = f"{datetime.now(timezone.utc).isoformat()} minted, expires {expiry}"
    print(line, file=sys.stderr)
    if path := os.environ.get("MINT_TOKEN_LOG"):
        try:
            with open(path, "a") as f:
                f.write(f"{line}\n")
        except OSError as e:
            print(f"Could not write {path}: {e}", file=sys.stderr)


def record_token(token: str) -> None:
    """Append the minted credential to ``MINT_TOKEN_SECRETS`` for later scrubbing.

    A CI caller shares its job environment with whatever it runs, so an agent can
    mint a token itself, and a transcript that tees raw tool output will capture
    it. Recording every minted value gives a redaction step something concrete to
    scrub; the alternative is a transcript with no credential scrubbing at all.
    """
    if not (path := os.environ.get("MINT_TOKEN_SECRETS")):
        return
    try:
        with open(path, "a", opener=lambda p, f: os.open(p, f, 0o600)) as f:
            f.write(f"{token}\n")
    except OSError as e:
        print(f"Could not write {path}: {e}", file=sys.stderr)


def main() -> None:
    cfg = Config()
    # This script prints a credential on stdout, so it must never be reachable by
    # ambient auth. Without this guard, running it on a developer machine happily
    # resolves the default ~/.databrickscfg profile and prints that PAT instead.
    if cfg.auth_type != "github-oidc":
        sys.exit(
            f"Refusing to print a {cfg.auth_type!r} credential. This script mints a "
            "federated token for CI; set DATABRICKS_AUTH_TYPE=github-oidc."
        )
    scheme, _, token = cfg.authenticate()["Authorization"].partition(" ")
    if scheme != "Bearer" or not token:
        sys.exit(f"Expected a Bearer credential from the OIDC exchange, got {scheme!r}")
    log_diagnostics(cfg)
    record_token(token)
    print(token)


if __name__ == "__main__":
    main()
