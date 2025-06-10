---
description: >
  Tuning judges
last_update:
  date: 2025-05-18
---

# tuning judges

:::danger UNDER CONSTRUCTION
🔴 This page is under construction and content may be incomplete or subject to change.
:::

how to tune the guideline

- collect some traces
- have them labeled with the same question as your guideline
- create an eval dataset with them
- create the guideline (either as cstom metric or predefined)
- create a new metric that wraps that and checks vs. the expectation
- iterate
