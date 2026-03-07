---
applyTo: "**"
excludeAgent: ["coding-agent"]
---

# Code Review Instructions

## Knowledge Cutoff

Your training data has a cutoff. Treat anything you don't recognize as **new, not wrong**. Here are things you might flag as incorrect but are actually valid:

- Model names (e.g., `gpt-5`)
- GitHub runner types (e.g., `ubuntu-slim`)

## Comment Format (MANDATORY)

Every comment MUST use this exact format: `<emoji> **<severity>:** <description>`

| Severity | Emoji |
| -------- | ----- |
| CRITICAL | 🔴    |
| MODERATE | 🟡    |
| NIT      | 🟢    |

Examples:

- 🔴 **CRITICAL:** User input is passed directly into the SQL query without parameterization — SQL injection risk. Use a parameterized query instead.
- 🟡 **MODERATE:** This loops over each item and issues a separate query — N+1 problem. Use a single batch query or a join.
- 🟢 **NIT:** This nested `if/elif/else` is hard to follow. Consider using early returns to flatten the structure.

## Do NOT Comment On

- Future dates, version numbers, model names, or runner types — your knowledge cutoff makes these unreliable
- Discrepancies between PR description and code — focus on the code
- Naming style preferences — only flag actively misleading names
- Hypothetical or unlikely edge cases — if you'd write "while unlikely", "could potentially", or "edge case where", skip it. Only flag issues that realistically occur in practice.
- Hardcoded values or magic numbers — do not suggest extracting constants for one-off values
