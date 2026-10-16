---
name: triage
description: Triage bugs or handle authenticated maintainer follow-ups in a sandbox, writing a response and optional verified fix proposal for the workflow to publish.
disable-model-invocation: true
argument-hint: "<initial|followup> <issue_path> <out_dir>"
arguments: [action, issue_path, out_dir]
---

# Triage

Read `$issue_path`: JSON with `title`, `body`, `repository`, `issue_number`, and optional
`followup`. The workflow selects `$action` explicitly:

| Action     | Instructions               | Output schema                              | Input requirement          |
| ---------- | -------------------------- | ------------------------------------------ | -------------------------- |
| `initial`  | [initial.md](initial.md)   | [initial.schema.yml](initial.schema.yml)   | `followup` missing or null |
| `followup` | [followup.md](followup.md) | [followup.schema.yml](followup.schema.yml) | `followup` non-null        |

Stop without writing a payload if the action is unknown or its input requirement is unmet.
Read only the selected action's instructions and schema, plus the shared
[runtime](../../triage/runtime.md) and [output rules](../../triage/output.md).

Write `$out_dir/payload.json`, then validate it against the selected action's schema:

```bash
uvx check-jsonschema@0.37.4 \
  --schemafile ".claude/skills/triage/$action.schema.yml" "$out_dir/payload.json"
```

Fix validation errors before finishing.
