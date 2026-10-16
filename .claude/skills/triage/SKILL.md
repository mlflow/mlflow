---
name: triage
description: Perform initial bug triage in a sandbox and write a verdict, report, and optional fix proposal for the workflow to publish.
disable-model-invocation: true
argument-hint: "<issue_path> <out_dir>"
arguments: [issue_path, out_dir]
---

# Initial bug triage

Read `$issue_path`: JSON with `title`, `body`, `repository`, and `issue_number`.
This entry point handles initial triage only. If `followup` is non-null, stop without
writing a payload; the workflow must invoke `/triage-followup` for that request.

Read the shared [runtime](../../triage/runtime.md) and [output rules](../../triage/output.md).
Investigate the reported bug using [bug reproduction and verification](../../triage/bug.md).
Explain what reproduces on the current checkout, the root cause, and the suggested fix.

## Automatic fix

Only propose a PR for a simple issue with a straightforward fix that meets the shared
output requirements. If the author offers to implement it (for example, "I'd be happy to
submit a PR"), leave the fix to them. Describe unresolved design options for a maintainer
to choose.

## Response

Read [payload.schema.yml](payload.schema.yml). Pick exactly one verdict from its label
descriptions and set `label` to that exact value. Write `comment` using its report template.
When proposing a PR, still describe the suggested fix in full in the issue comment.

Write `$out_dir/payload.json`, then validate it:

```bash
uvx check-jsonschema@0.37.4 \
  --schemafile .claude/skills/triage/payload.schema.yml "$out_dir/payload.json"
```

Fix validation errors before finishing.
