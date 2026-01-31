"""Generate improved PR title using AI."""
# ruff: noqa: T201

import json
import os
import re
import subprocess
import sys
import urllib.request

REPO = os.environ["REPO"]
PR_NUMBER = os.environ["PR_NUMBER"]


def run_gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True)


def get_pr_info() -> tuple[str, str]:
    print("Fetching PR information...", file=sys.stderr)
    output = run_gh("pr", "view", PR_NUMBER, "--repo", REPO, "--json", "title,body")
    data = json.loads(output)
    return data["title"], data.get("body") or ""


def get_pr_diff() -> str:
    """Fetch PR diff, truncated if too long."""
    print("Fetching PR diff...", file=sys.stderr)
    try:
        diff = run_gh("pr", "diff", PR_NUMBER, "--repo", REPO)
    except subprocess.CalledProcessError:
        return ""

    max_length = 50000
    if len(diff) > max_length:
        diff = diff[:max_length] + "\n\n... [diff truncated due to length] ..."
    return diff


def extract_description(body: str) -> str:
    """Extract the 'What changes are proposed' section from PR body."""
    pattern = r"### What changes are proposed in this pull request\?\s*(.+?)(?=###|$)"
    if match := re.search(pattern, body, flags=re.DOTALL | re.IGNORECASE):
        return match.group(1).strip()
    return body


def build_prompt(title: str, body: str, diff: str) -> str:
    body = extract_description(body).strip() or "(No description provided)"
    return f"""\
Rewrite the PR title to be more descriptive and follow the guidelines below.

## Current PR Title
{title}

## PR Description
{body}

## Code Changes (Diff)
```diff
{diff}
```

## Guidelines for a good PR title:
1. Start with a verb in imperative mood (e.g., "Add", "Fix", "Update", "Remove", "Refactor")
2. Be specific about what changed and where
3. Keep it concise
4. Do not include issue numbers in the title (they belong in the PR body)
5. Focus on the "what" and "why", not the "how"
6. Use proper capitalization (capitalize first letter, no period at end)
7. Use backticks for code/file references (e.g., `ClassName`, `function_name`, `module.path`)

Rewrite the PR title following these guidelines."""


def call_anthropic_api(prompt: str) -> str:
    print("Calling Claude API...", file=sys.stderr)

    api_key = os.environ["ANTHROPIC_API_KEY"]
    request_body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "The rewritten PR title.",
                        }
                    },
                    "required": ["title"],
                    "additionalProperties": False,
                },
            }
        },
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(request_body).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    with urllib.request.urlopen(req) as resp:
        response = json.loads(resp.read().decode())

    print("API Response:", file=sys.stderr)
    print(json.dumps(response, indent=2), file=sys.stderr)

    if "error" in response:
        raise RuntimeError(f"Error from Claude API: {response['error']['message']}")

    content = json.loads(response["content"][0]["text"])
    return str(content["title"])


def main() -> None:
    title, body = get_pr_info()
    print(f"Original title: {title}", file=sys.stderr)

    diff = get_pr_diff()
    prompt = build_prompt(title, body, diff)
    new_title = call_anthropic_api(prompt)

    print(new_title)


if __name__ == "__main__":
    main()
