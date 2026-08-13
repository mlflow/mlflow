# /// script
# dependencies = ["databricks-sdk"]
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
  DATABRICKS_AUTH_TYPE=github-oidc uv run dev/mint_gateway_token.py
"""

from __future__ import annotations

import sys

from databricks.sdk.core import Config


def main() -> None:
    cfg = Config()
    scheme, _, token = cfg.authenticate()["Authorization"].partition(" ")
    if scheme != "Bearer" or not token:
        sys.exit(f"Expected a Bearer credential from the OIDC exchange, got {scheme!r}")
    # Diagnostics go to stderr so stdout stays consumable as the bare credential.
    # The federated token's lifetime is far shorter than a U2M token's (~10 min
    # observed, not the ~1h the docs quote for U2M), and the caller's refresh
    # interval has to stay inside it.
    try:
        print(f"Databricks token expires at {cfg.oauth_token().expiry}", file=sys.stderr)
    except Exception as e:  # diagnostics must never fail the exchange
        print(f"Could not read token expiry: {e}", file=sys.stderr)
    print(token)


if __name__ == "__main__":
    main()
