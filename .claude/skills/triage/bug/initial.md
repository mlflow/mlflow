# Initial bug triage

Reproduce the reported bug, explain the root cause and suggested fix, and assign a verdict.
Follow the investigation and verification steps in [README.md](README.md).

## Automatic fix

Only propose a PR for a simple issue with a straightforward fix that meets the shared
requirements in `README.md`. If the issue author offers to fix it themselves (for example,
"I'd be happy to submit a PR"), leave the fix to them. Describe unresolved design options
for a maintainer to choose.

## Response

Use [payload.schema.yml](payload.schema.yml). Pick exactly one verdict from its label
descriptions and set `label` to that exact value. Write `comment` using the template in
`comment.description`. Prefer runnable commands and code to manual steps, and keep it tight.
When proposing a PR, still describe the suggested fix in full in the issue comment.
A PR in the payload is a proposal for the workflow to publish; do not claim it is already open.
