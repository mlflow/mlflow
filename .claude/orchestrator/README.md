# mlflow-reviewer orchestrator

Comment-triggered code-review bot for the MLflow repository. A maintainer comments
`/review` on a PR and the bot posts inline review comments derived from an adversarial
checklist plus a discovery-mode spotter agent.

## Status

This directory ships incrementally as a stack of PRs.

| Stack | Scope | Status |
|---|---|---|
| 1 | Scaffolding: package layout, agent prompts, README. No behavior. | this PR |
| 2 | Core orchestrator logic + CLI dry-run | future |
| 3 | GitHub Actions workflow + posting + GitHub App identity | future |
| 4 | Helpers-index refresh workflow | future |
| 5 | Activation: flip dry-run default off, smoke test | future |

Stack 1 introduces the directory and the agent prompts. Nothing runs in CI yet.

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

## Production trigger (post-Stack 3)

A maintainer comments `/review` on a PR. The workflow at
`.github/workflows/review.yml` picks up the `issue_comment` event, validates the
comment author and rate limits, then runs the orchestrator with the bot's GitHub App
installation token and the Anthropic API key from repo secrets.

## Cost model

At ~10 `/review`s per day on Sonnet:

- Default mode: ~$165 / month
- Lite mode: ~$90 / month
- Hybrid (Opus spotter): ~$560 / month

Numbers are approximate; actual spend varies with PR size. The Anthropic monthly cap
acts as a hard ceiling.
