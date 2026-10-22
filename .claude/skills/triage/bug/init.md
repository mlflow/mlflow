# Initial bug triage

Investigate the reported bug using [bug reproduction and verification](reproduce.md).
Explain what reproduces on the current checkout, the root cause, and the suggested fix.

## Automatic fix

Only propose a PR for a simple issue with a straightforward fix that meets the shared
output requirements. If the author offers to implement it (for example, "I'd be happy to
submit a PR"), leave the fix to them. Describe unresolved design options for a maintainer
to choose.

## Response

Read [init.schema.yml](init.schema.yml). Pick exactly one `verdict` from its
descriptions and set it to that exact value. Write `comment` using its report template.
Keep the verdict unprefixed in the comment; the workflow adds `triage: ` to the issue label.
When proposing a PR, still describe the suggested fix in full in the issue comment.
Add a diagram to the Summary as described in [visual aids](../viz-aid.md).

Follow the [shared guidance for issue titles](../output.md#optional-issue-title).
