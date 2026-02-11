"""Triage GitHub issues: generate a comment requesting missing info."""
# ruff: noqa: T201

import argparse
import concurrent.futures
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = """\
Triage the following GitHub issue and decide whether to request more information \
from the author.

## Issue Title
{title}

## Issue Body
{body}

## Instructions
Evaluate the issue and return a JSON object with two fields:

- `comment`: A polite comment to post on the issue requesting missing information, \
or null if no comment is needed. The comment should be concise and specific about \
what information would help. It may ask for any combination of:
  - Steps to reproduce the problem (for bug reports without clear repro steps)
  - Environment info such as OS, Python version, or MLflow version (for bug reports)
  - Full traceback (for bug reports that mention an error but don't include one)
  - A screenshot or screen recording (only for issues that would benefit from visual evidence \
to understand and reproduce, e.g., layout issues, rendering bugs, styling problems — \
do not request for backend, API, CLI, docs, or performance issues)
- `reason`: A brief explanation of why you decided to return or not return a comment. \
This is for internal verification only and will not be shown to the user.

Guidelines:
- Only request information that is clearly missing and would help investigate the issue.
- Do not request repro steps if the issue already contains numbered steps, a code snippet, \
or a clear description of how to trigger the bug.
- Do not request environment info if OS, Python version, or MLflow version is already provided.
- Do not request anything for feature requests.
- When in doubt, return null — only return a comment when information is clearly missing."""

MAX_BODY_LENGTH = 10_000


def strip_html_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def build_prompt(title: str, body: str) -> str:
    body = strip_html_comments(body)
    return PROMPT_TEMPLATE.format(
        title=title,
        body=body[:MAX_BODY_LENGTH],
    )


def call_anthropic_api(prompt: str) -> tuple[dict[str, Any], dict[str, int]]:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    request_body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "comment": {
                            "type": ["string", "null"],
                            "description": (
                                "A comment to post requesting missing information, "
                                "or null if no comment is needed."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation for the decision.",
                        },
                    },
                    "required": ["comment", "reason"],
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

    usage = response.get("usage", {})
    classification = json.loads(response["content"][0]["text"])
    return classification, usage


# https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table
HAIKU_INPUT_COST_PER_MTOK = 1.00
HAIKU_OUTPUT_COST_PER_MTOK = 5.00


def compute_cost(usage: dict[str, int]) -> float:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return (
        input_tokens * HAIKU_INPUT_COST_PER_MTOK + output_tokens * HAIKU_OUTPUT_COST_PER_MTOK
    ) / 1_000_000


def triage_issue(title: str, body: str) -> dict[str, Any]:
    # Skip triage for security vulnerability issues
    if "security vulnerability" in title.lower():
        return {
            "comment": None,
            "reason": "Skipped: Issue title contains 'Security Vulnerability'",
        }

    prompt = build_prompt(title, body)
    classification, _ = call_anthropic_api(prompt)
    return {
        "comment": classification["comment"],
        "reason": classification["reason"],
    }


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def parse_dataset(path: Path) -> list[dict[str, str]]:
    text = path.read_text()
    issues = []
    for section in re.split(r"\n---\n", text):
        header_match = re.search(r"^## (.+)$", section, re.MULTILINE)
        title_match = re.search(r"\*\*Title:\*\*\s*(.+)$", section, re.MULTILINE)
        body_match = re.search(r"\*\*Body:\*\*\s*\n(.*)", section, re.DOTALL)
        if header_match and title_match and body_match:
            issues.append(
                {
                    "header": header_match.group(1).strip(),
                    "title": title_match.group(1).strip(),
                    "body": body_match.group(1).strip(),
                }
            )
    return issues


def triage_synthetic(title: str, body: str) -> tuple[dict[str, Any], dict[str, int]]:
    prompt = build_prompt(title, body)
    classification, usage = call_anthropic_api(prompt)
    return {
        "title": title,
        "comment": classification["comment"],
        "reason": classification["reason"],
    }, usage


def run_tests() -> None:
    dataset_path = Path(__file__).parent / "triage.md"
    issues = parse_dataset(dataset_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(triage_synthetic, issue["title"], issue["body"]): issue
            for issue in issues
        }

    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for future in futures:
        issue = futures[future]
        result, usage = future.result()
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        has_comment = result["comment"] is not None
        color = RED if has_comment else GREEN
        print(f"{color}{issue['header']}{RESET}")
        print(f"  reason: {result['reason']}")
        if result["comment"]:
            print(f"  comment: {result['comment'][:200]}")

    cost = compute_cost(total_usage)
    print(
        f"\nTokens: {total_usage['input_tokens']} input, {total_usage['output_tokens']} output"
        f" (${cost:.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage GitHub issues")
    subparsers = parser.add_subparsers(dest="command")

    triage_parser = subparsers.add_parser("triage")
    triage_parser.add_argument("--title", required=True)
    triage_parser.add_argument("--body", default="")

    subparsers.add_parser("test")

    args = parser.parse_args()
    match args.command:
        case "triage":
            result = triage_issue(args.title, args.body)
            print(json.dumps(result))
        case "test":
            run_tests()
        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
