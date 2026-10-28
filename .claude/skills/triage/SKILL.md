---
name: triage
description: Ignore this skill unless invoked through an explicit skill mention. Triage GitHub issues or handle maintainer follow-ups in a sandbox, writing a response and optional verified implementation proposal for the workflow to publish.
disable-model-invocation: true
argument-hint: "<init|follow-up> <issue_path> <out_dir>"
arguments: [action, issue_path, out_dir]
---

# Triage

Read `$issue_path`: JSON with `title`, `body` (the issue description), `comments` (earlier
comments as `{author, created_at, url, body}`, oldest first; empty for `init`), `repository`,
`issue_number`, and optional `followup`. The workflow selects `$action` explicitly and sets
`TRIAGE_TYPE` to `bug` or `feature_request`. Read the matching instructions and schema:

| Type              | Action      | Instructions                                 | Output schema                                                | Input requirement          |
| ----------------- | ----------- | -------------------------------------------- | ------------------------------------------------------------ | -------------------------- |
| `bug`             | `init`      | [init.md](bug/init.md)                       | [init.schema.yml](bug/init.schema.yml)                       | `followup` missing or null |
| `bug`             | `follow-up` | [follow-up.md](bug/follow-up.md)             | [follow-up.schema.yml](bug/follow-up.schema.yml)             | `followup` non-null        |
| `feature_request` | `init`      | [init.md](feature_request/init.md)           | [init.schema.yml](feature_request/init.schema.yml)           | `followup` missing or null |
| `feature_request` | `follow-up` | [follow-up.md](feature_request/follow-up.md) | [follow-up.schema.yml](feature_request/follow-up.schema.yml) | `followup` non-null        |

Stop without writing a payload if the type or action is unknown or its input requirement is unmet.
Read only the selected action's instructions and schema, the selected type's output rules
([bug](bug/output.md) or [feature request](feature_request/output.md)), and the shared
[runtime](runtime.md) and [output rules](output.md).

Write `$out_dir/payload.json`, then validate it against the selected action's schema:

```bash
uvx check-jsonschema@0.37.4 \
  --schemafile ".claude/skills/triage/$TRIAGE_TYPE/$action.schema.yml" "$out_dir/payload.json"
```

Fix validation errors before finishing.
