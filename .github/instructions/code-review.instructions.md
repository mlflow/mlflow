---
applyTo: "**"
excludeAgent: "cloud-agent"
---

# Code Review Instructions

## Knowledge Cutoff

Your training data has a cutoff. Treat anything you don't recognize as **new, not wrong**. Here are things you might flag as incorrect but are actually valid:

- Model names (e.g., `gpt-5`)
- GitHub runner types (e.g., `ubuntu-slim`)

## Verify Rather Than Infer

Before claiming changed code misbehaves, run it. A scratch snippet through `python3` or `node` will settle most questions in seconds; installing a dependency or running a single test is fine too when it's quick.

## Do NOT Comment On

- Future dates, version numbers, model names, or runner types — your knowledge cutoff makes these unreliable
- Naming style preferences — only flag actively misleading names
- Hypothetical or unlikely edge cases — if you'd write "while unlikely", "could potentially", or "edge case where", skip it. Only flag issues that realistically occur in practice.
- Hardcoded values or magic numbers — do not suggest extracting constants for one-off values
