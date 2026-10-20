---
name: triage
description: Triage GitHub issues or handle maintainer follow-ups in a sandbox, writing a response and optional verified implementation proposal for the workflow to publish.
disable-model-invocation: true
argument-hint: "<init|follow-up> <issue_path> <out_dir>"
arguments: [action, issue_path, out_dir]
---

# Triage

Read `$issue_path`: JSON with `title`, `body` (the issue description), `comments` (earlier
comments as `{author, created_at, url, body}`, oldest first; empty for `init`), `repository`,
`issue_number`, and optional `followup`. This skill currently supports bugs. The workflow
selects `$action` explicitly:

| Action      | Instructions                     | Output schema                                    | Input requirement          |
| ----------- | -------------------------------- | ------------------------------------------------ | -------------------------- |
| `init`      | [init.md](bug/init.md)           | [init.schema.yml](bug/init.schema.yml)           | `followup` missing or null |
| `follow-up` | [follow-up.md](bug/follow-up.md) | [follow-up.schema.yml](bug/follow-up.schema.yml) | `followup` non-null        |

Stop without writing a payload if the action is unknown or its input requirement is unmet.
Read only the selected action's instructions and schema, plus the shared
[runtime](runtime.md) and [output rules](output.md).

Write `$out_dir/payload.json`, then validate it against the selected action's schema:

```bash
uvx check-jsonschema@0.37.4 \
  --schemafile ".claude/skills/triage/bug/$action.schema.yml" "$out_dir/payload.json"
```

Fix validation errors before finishing.
