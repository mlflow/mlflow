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

When more than one fix would work and every option behaves the same for users, differing
only in how the code is written, implement the one you recommend. List the alternatives and
why you chose it in the PR body and, on initial triage, in the issue comment. Do not propose
a PR when users would notice which option was chosen, for example through a public API, a
default, error semantics, a stored or wire format, or compatibility, unless a maintainer has
chosen: describe the options, recommend one, and leave the decision to them.
