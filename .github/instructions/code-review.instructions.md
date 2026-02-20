---
applyTo: "**"
excludeAgent: ["coding-agent"]
---

# Code Review Instructions

## Comment Format (MANDATORY)

Every comment MUST use this exact format: `<emoji> **<LABEL>:** <description>`

- 🔴 **CRITICAL:** bugs, security vulnerabilities, data loss, breaking changes
- 🟡 **IMPORTANT:** performance problems, race conditions, API contract violations
- 🟢 **SUGGESTION:** readability, minor improvements, alternative approaches

Examples:

- 🔴 **CRITICAL:** User input is passed directly into the SQL query without parameterization — SQL injection risk. Use a parameterized query instead.
- 🟡 **IMPORTANT:** This loops over each item and issues a separate query — N+1 problem. Use a single batch query or a join.
- 🟢 **SUGGESTION:** This nested `if/elif/else` is hard to follow. Consider using early returns to flatten the structure.
