"""Classify whether a GitHub issue needs visual media (screenshots/recordings)."""
# ruff: noqa: T201

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

VISUAL_MEDIA_PATTERNS = [
    r"!\[",  # Markdown image
    r"<img\s",  # HTML image tag
    r"<video\s",  # HTML video tag
    r"\.png",
    r"\.jpg",
    r"\.jpeg",
    r"\.gif",
    r"\.mp4",
    r"\.webm",
    r"\.webp",
    r"user-attachments",
    r"user-images\.githubusercontent\.com",
]

VISUAL_MEDIA_RE = re.compile("|".join(VISUAL_MEDIA_PATTERNS), re.IGNORECASE)


def run_gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True)


def get_issue(repo: str, issue_number: str) -> dict[str, str]:
    output = run_gh("issue", "view", issue_number, "--repo", repo, "--json", "title,body,labels")
    return json.loads(output)


def has_visual_media(body: str | None) -> bool:
    if not body:
        return False
    return bool(VISUAL_MEDIA_RE.search(body))


PROMPT_TEMPLATE = """\
Classify whether the following GitHub issue describes a problem that is \
visual/UI-related and would benefit from a screenshot or screen recording \
to understand and reproduce.

## Issue Title
{title}

## Issue Body
{body}

## Instructions
- Return {{"needs_visual_media": true}} if the issue describes a visual or \
UI problem (e.g., layout issues, rendering bugs, styling problems, \
UI elements not appearing correctly, visual regressions).
- Return {{"needs_visual_media": false}} if the issue is about backend logic, \
APIs, CLI behavior, documentation, performance metrics, or other non-visual topics.
- When in doubt, return false — only flag issues that clearly describe \
something visual."""


def build_prompt(title: str, body: str) -> str:
    return PROMPT_TEMPLATE.format(title=title, body=body)


def call_anthropic_api(prompt: str) -> dict[str, object]:
    print("Calling Claude API...", file=sys.stderr)

    api_key = os.environ["ANTHROPIC_API_KEY"]
    request_body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "needs_visual_media": {
                            "type": "boolean",
                            "description": (
                                "Whether the issue would benefit from a screenshot or recording."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": ("Brief explanation for the classification."),
                        },
                    },
                    "required": ["needs_visual_media", "reason"],
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

    try:
        with urllib.request.urlopen(req) as resp:
            response = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"API Error {e.code}: {error_body}", file=sys.stderr)
        raise

    print("API Response:", file=sys.stderr)
    print(json.dumps(response, indent=2), file=sys.stderr)

    return json.loads(response["content"][0]["text"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify if an issue needs visual media")
    parser.add_argument("--repo", required=True, help="Repository (owner/name)")
    parser.add_argument("--issue-number", required=True, help="Issue number")
    args = parser.parse_args()

    issue = get_issue(args.repo, args.issue_number)
    title = issue["title"]
    body = issue.get("body") or ""
    labels = [label["name"] for label in issue.get("labels", [])]
    print(f"Issue: {title}", file=sys.stderr)
    print(f"Labels: {labels}", file=sys.stderr)

    # Short-circuit: already has visual media
    if has_visual_media(body):
        print("Issue already contains visual media, skipping.", file=sys.stderr)
        result = {
            "needs_visual_media": False,
            "has_visual_media": True,
            "reason": "already has visual media",
        }
        print(json.dumps(result))
        return

    # Call LLM to classify
    classification = call_anthropic_api(build_prompt(title, body))
    result = {
        "needs_visual_media": classification["needs_visual_media"],
        "has_visual_media": False,
        "reason": classification["reason"],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
