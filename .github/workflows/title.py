"""Generate improved PR or issue title using AI."""
# ruff: noqa: T201

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

# Maximum number of linked issues to fetch from GitHub API
MAX_LINKED_ISSUES = 10


def run_gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True)


def extract_stacked_pr_base_sha(pr_body: str | None, head_ref: str) -> str | None:
    """Extract the base SHA from the stacked PR incremental diff link.

    In stacked PR descriptions, the current PR is marked with bold (double
    asterisks). We find the bold entry matching the branch name and extract
    the base SHA from the files URL pattern /files/<base>..<head>.
    """
    if not pr_body or "Stacked PR" not in pr_body:
        return None

    # Find the line with bold branch name [**<branch>**], then extract /files/<base>..<head>
    marker = f"[**{head_ref}**]"
    for line in pr_body.split("\n"):
        if marker in line:
            if m := re.search(r"/files/(?P<base>[a-f0-9]{7,40})\.\.(?P<head>[a-f0-9]{7,40})", line):
                return m.group("base")

    return None


def get_pr_info(repo: str, pr_number: str) -> tuple[str, str, str, str]:
    """Return (title, body, head_sha, head_ref)."""
    print("Fetching PR information...", file=sys.stderr)
    output = run_gh(
        "pr", "view", pr_number, "--repo", repo, "--json", "title,body,headRefOid,headRefName"
    )
    data = json.loads(output)
    return data["title"], data.get("body") or "", data["headRefOid"], data["headRefName"]


def get_pr_diff(repo: str, pr_number: str, body: str, head_sha: str, head_ref: str) -> str:
    """Fetch PR diff with stacked PR support."""
    print("Fetching PR diff...", file=sys.stderr)

    if base_sha := extract_stacked_pr_base_sha(body, head_ref):
        print(
            f"Detected stacked PR, fetching incremental diff: {base_sha[:7]}..{head_sha[:7]}",
            file=sys.stderr,
        )
        diff = run_gh(
            "api",
            f"repos/{repo}/compare/{base_sha}...{head_sha}",
            "-H",
            "Accept: application/vnd.github.v3.diff",
        )
    else:
        diff = run_gh("pr", "diff", pr_number, "--repo", repo)

    max_length = 50000
    if len(diff) > max_length:
        diff = diff[:max_length] + "\n\n... [diff truncated due to length] ..."
    return diff


def get_closing_issues(repo: str, pr_number: str) -> list[tuple[int, str]]:
    """Fetch linked issues via GitHub GraphQL API.

    Returns a list of (issue_number, issue_title) tuples for issues
    linked via closing keywords (e.g., "Fixes #123").
    """
    if "/" not in repo:
        raise ValueError(f"Invalid repo format: '{repo}'. Expected format: 'owner/name'")

    owner, name = repo.split("/", 1)

    # Validate inputs to prevent GraphQL injection
    # GitHub usernames/org names: alphanumeric, hyphens, max 39 chars
    # Repository names: alphanumeric, hyphens, underscores, dots
    # PR numbers: positive integers
    if not re.match(r"^[a-zA-Z0-9-]{1,39}$", owner):
        raise ValueError(f"Invalid owner format: '{owner}'")
    if not re.match(r"^[a-zA-Z0-9._-]+$", name):
        raise ValueError(f"Invalid repository name format: '{name}'")
    if not pr_number.isdigit():
        raise ValueError(f"Invalid PR number: '{pr_number}'")

    query = f"""
    {{
      repository(owner: "{owner}", name: "{name}") {{
        pullRequest(number: {pr_number}) {{
          closingIssuesReferences(first: {MAX_LINKED_ISSUES}) {{
            nodes {{
              number
              title
            }}
          }}
        }}
      }}
    }}
    """

    print("Fetching closing issues...", file=sys.stderr)
    try:
        output = run_gh("api", "graphql", "-f", f"query={query}")
        data = json.loads(output)
        pr_data = data.get("data", {}).get("repository", {}).get("pullRequest", {})
        nodes = pr_data.get("closingIssuesReferences", {}).get("nodes", [])
        return [(node["number"], node["title"]) for node in nodes]
    except (subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to fetch closing issues: {e}", file=sys.stderr)
        return []


def extract_description(body: str) -> str:
    """Extract the 'What changes are proposed' section from PR body."""
    pattern = r"### What changes are proposed in this pull request\?\s*(.+?)(?=###|$)"
    if match := re.search(pattern, body, flags=re.DOTALL | re.IGNORECASE):
        return match.group(1).strip()
    return body


def build_prompt(
    title: str, body: str, diff: str, linked_issues: list[tuple[int, str]] | None = None
) -> str:
    body = extract_description(body).strip() or "(No description provided)"

    # Build linked issues section with leading/trailing newlines for consistent spacing.
    # When empty, this is "", when populated it's "\n## Linked Issues\n...\n"
    # This ensures one blank line between sections in both cases.
    linked_issues_section = ""
    if linked_issues:
        linked_issues_section = "\n## Linked Issues\n"
        for issue_num, issue_title in linked_issues:
            linked_issues_section += f"- #{issue_num}: {issue_title}\n"
        linked_issues_section += "\n"

    return f"""\
Rewrite the PR title to be more descriptive and follow the guidelines below.

## Current PR Title
{title}

## PR Description
{body}
{linked_issues_section}
## Code Changes (Diff)
```diff
{diff}
```

## Guidelines for a good PR title:
1. Start with a verb in imperative mood (e.g., "Add", "Fix", "Update", "Remove", "Refactor")
2. Be specific about what changed and where
3. Keep it concise (aim for 72 characters or less, 100 characters maximum)
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

    content = json.loads(response["content"][0]["text"])
    return str(content["title"])


def get_issue_info(repo: str, number: str) -> tuple[str, str]:
    """Return (title, body)."""
    print("Fetching issue information...", file=sys.stderr)
    output = run_gh("issue", "view", number, "--repo", repo, "--json", "title,body")
    data = json.loads(output)
    return data["title"], data.get("body") or ""


def build_issue_prompt(title: str, body: str) -> str:
    body = body.strip() or "(No description provided)"
    return f"""\
Rewrite the issue title to be more descriptive and follow the guidelines below.

## Current Issue Title
{title}

## Issue Description
{body}

## Guidelines for a good issue title:
1. Clearly describe the problem or feature request (e.g., "`load_model` fails with `KeyError`
   when model has nested flavors", "Support custom metric types in autologging")
2. Be specific about what the issue is about
3. Keep it concise (aim for 72 characters or less, 100 characters maximum)
4. Do not include issue numbers in the title
5. Focus on the problem or feature request
6. Use proper capitalization (capitalize first letter, no period at end)
7. Use backticks for code/file references (e.g., `ClassName`, `function_name`, `module.path`)

Rewrite the issue title following these guidelines."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate improved PR or issue title using AI")
    parser.add_argument("--repo", required=True, help="Repository (owner/name)")
    parser.add_argument("--number", required=True, help="PR or issue number")
    parser.add_argument("--type", required=True, choices=["pr", "issue"], help="Type: pr or issue")
    args = parser.parse_args()

    if args.type == "pr":
        title, body, head_sha, head_ref = get_pr_info(args.repo, args.number)
        print(f"Original title: {title}", file=sys.stderr)
        diff = get_pr_diff(args.repo, args.number, body, head_sha, head_ref)
        linked_issues = get_closing_issues(args.repo, args.number)
        if linked_issues:
            print(f"Found {len(linked_issues)} linked issue(s)", file=sys.stderr)
        prompt = build_prompt(title, body, diff, linked_issues)
    else:
        title, body = get_issue_info(args.repo, args.number)
        print(f"Original title: {title}", file=sys.stderr)
        prompt = build_issue_prompt(title, body)

    new_title = call_anthropic_api(prompt)
    print(new_title)


if __name__ == "__main__":
    main()
