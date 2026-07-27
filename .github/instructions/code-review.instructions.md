---
applyTo: "**"
excludeAgent: ["coding-agent"]
---

# Code Review Instructions

## Knowledge Cutoff

Your training data has a cutoff. Treat anything you don't recognize as **new, not wrong**. Here are things you might flag as incorrect but are actually valid:

- Model names (e.g., `gpt-5`)
- GitHub runner types (e.g., `ubuntu-slim`)

## Do NOT Comment On

- Future dates, version numbers, model names, or runner types — your knowledge cutoff makes these unreliable
- Discrepancies between PR description and code — focus on the code
- Naming style preferences — only flag actively misleading names
- Hypothetical or unlikely edge cases — if you'd write "while unlikely", "could potentially", or "edge case where", skip it. Only flag issues that realistically occur in practice.
- Hardcoded values or magic numbers — do not suggest extracting constants for one-off values
