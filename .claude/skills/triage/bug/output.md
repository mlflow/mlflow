# Bug triage output

## Initial triage

Pick exactly one `verdict` from the selected schema's descriptions and set it to that
exact value. Write `comment` using its report template. Keep the verdict unprefixed
in the comment; the workflow adds `triage: ` to the issue label.
When proposing a PR, still describe the suggested fix in full in the issue comment.

## Maintainer follow-up

Write a concise, free-form `comment` addressing the request without repeating the
original report. Do not include a `verdict`.

## Pull request eligibility

Only propose a PR when the bug reproduces at the current checkout and the fix is
verified against the reproduction and relevant tests. Include `pull_request` only
with a verified `fix.patch` that meets the [shared output requirements](../output.md)
and the selected action's eligibility rules.
