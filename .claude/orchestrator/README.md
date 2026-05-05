# mlflow-reviewer orchestrator

Comment-triggered code-review bot for the MLflow repository. A maintainer comments
`/review` on a PR and the bot posts inline review comments derived from an adversarial
checklist plus a discovery-mode spotter agent.

## Status

This directory ships incrementally as a stack of PRs.

| Stack | Scope | Status |
|---|---|---|
| 1 | Scaffolding: package layout, agent prompts, README. | landed |
| 2 | Core orchestrator logic + CLI dry-run | landed |
| 3 | GitHub Actions workflow + posting + GitHub App identity | this PR |
| 4 | Helpers-index refresh workflow | future |
| 5 | Activation: flip dry-run default off, smoke test | future |

The workflow is wired up but the dry-run default is still on; Stack 5 flips
`--no-dry-run` to enable posting.

## Architecture (forward-looking)

The orchestrator is invoked from `.github/workflows/review.yml` when a maintainer posts
`/review` on a PR. It runs three model calls in parallel-then-serial:

```
        ┌─ mlflow-reviewer agent (full review)  ─┐
trigger ┤                                          ├─ cluster judge ─ post curated findings
        └─ issue-spotter (discovery)
            └─ mlflow-reviewer agent (opinion mode) ┘
```

Default mode (`/review`) runs both branches and dedupes; `/review --lite` runs only the
top branch (cheap, lower recall).

See `agents/mlflow-reviewer.md` and `agents/issue-spotter.md` for the agent system
prompts.

## Modes

| Command | Architecture | Cost (Sonnet, medium PR) | Recall vs human |
|---|---|---|---|
| `/review` (default) | Discovery + opinion + dedup | ~$0.85 | 40% |
| `/review --lite` | Discovery only (single agent) | ~$0.30 | 16% |
| `/review --no-cache` | Force re-run on already-reviewed head SHA | same as parent | same |
| `/review --hybrid` | Use Opus for the spotter step, Sonnet elsewhere | ~$1.87 | TBD |

Flags compose: `/review --lite --no-cache`, `/review --hybrid`, etc.

## Authorization

Only repository maintainers (`MEMBER` / `OWNER` / `COLLABORATOR` association) can
trigger. Cap of 5 `/review` invocations per PR per day. Monthly spend cap of $750
configured in the Anthropic console.

## Identity

Comments post as `mlflow-reviewer[bot]` (a registered GitHub App). Each comment ends
with a trailer attributing the post to the bot.

## Dedup

Before posting, the orchestrator filters draft findings against existing review threads:

1. Skip if a hard-resolved thread exists at `path:line ±10`.
2. Skip if a soft-resolved thread (open thread where the PR author replied to a non-author
   opener) exists at `path:line ±10`.
3. Skip if `mlflow-reviewer[bot]` already posted at `path:line ±10` (self-dedup).
4. Skip if any open thread exists at `path:line ±10` AND a semantic-match judge confirms
   the existing thread covers the same concern (M1 conservative default).

## Repo knowledge

The orchestrator loads two kinds of knowledge files into the agent prompts:

- `repo_knowledge/helpers_*.md`: auto-generated symbol index, refreshed weekly by
  `.github/workflows/refresh-helpers.yml` (Stack 4).
- `repo_knowledge/*.md` (other): hand-curated architectural notes. No automated
  refresh; updated by maintainers when architecture changes.

The per-reviewer-mimic agents (Phase 5) are not part of M1.

## Local dry-run (post-Stack 2)

```bash
cd .claude/orchestrator
uv pip install -e .
ANTHROPIC_API_KEY=sk-... mlflow-reviewer 22962 --dry-run
```

Emits JSON to stdout instead of posting.

## Production trigger

A maintainer comments `/review` on a PR. The workflow at
`.github/workflows/mlflow-reviewer.yml` picks up the `issue_comment` event, validates
the comment author and per-PR rate limit, then runs the orchestrator with the bot's
GitHub App installation token and the Anthropic API key from repo secrets.

## GitHub App registration (one-time setup)

The bot posts as `mlflow-reviewer[bot]`, a registered GitHub App. To activate:

1. **Register the App.** Go to <https://github.com/settings/apps/new> (or under the
   org settings page) and use the values from `app-manifest.yml` as a reference.
   Important fields:
   - Name: `mlflow-reviewer` (yields login `mlflow-reviewer[bot]`).
   - Permissions: `contents: read`, `pull_requests: write`, `issues: write`,
     `metadata: read`.
   - Webhooks: disabled (the workflow handles the trigger via `issue_comment`).
2. **Generate a private key.** From the App's settings page, Generate a private key.
   Add it to the `mlflow/mlflow` repository secrets as `MLFLOW_REVIEWER_APP_KEY`.
3. **Note the App ID.** Add it as a repository variable named `MLFLOW_REVIEWER_APP_ID`.
4. **Install the App** on `mlflow/mlflow`.
5. **Add the Anthropic API key** as repo secret `ANTHROPIC_API_KEY`.
6. **Configure the monthly Anthropic spend cap** ($750) in the Anthropic console.
7. **Upload a logo** for the App (1MB, square PNG/JPG); shows up next to bot
   comments.
8. **Stack 5 activation.** Once the above is done, the workflow's `--no-dry-run`
   flag in `.github/workflows/mlflow-reviewer.yml` activates posting. Test on a
   closed PR or a fork before pointing at live mlflow.

## Cost model

At ~10 `/review`s per day on Sonnet:

- Default mode: ~$165 / month
- Lite mode: ~$90 / month
- Hybrid (Opus spotter): ~$560 / month

Numbers are approximate; actual spend varies with PR size. The Anthropic monthly cap
acts as a hard ceiling.
