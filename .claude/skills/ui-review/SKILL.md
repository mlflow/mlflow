---
name: ui-review
description: Review a GitHub PR's UI/UX changes by launching the MLflow web app, driving a headless agent-browser over the changed surfaces, and emitting a validated local UI-review payload (findings + screenshots).
disable-model-invocation: true
allowed-tools:
  - Read
  - Grep
  - Glob
  - Skill
  - Bash(agent-browser:*)
  - Bash(npx agent-browser:*)
  - Bash(gh pr view:*)
  - Bash(gh api:*)
  - Bash(curl:*)
  - Bash(git diff:*)
  - Bash(git show:*)
  - Bash(uv run --package skills skills:*)
  - Edit(//tmp/ui-review-payload.json)
argument-hint: "<owner_repo> <pr_number> <app_url>"
arguments: [owner_repo, pr_number, app_url]
---

# Review Pull Request UI/UX

You review the **rendered UI/UX** of a PR's frontend changes by driving a real (headless)
browser against a locally-running MLflow app — the visual counterpart to the `pr-review`
code-review skill. You do NOT post anything; you write a validated payload that the workflow
renders into a PR comment.

## Usage

```
/ui-review <owner_repo> <pr_number> <app_url>
```

## Arguments

- `<owner_repo>` (required): repository slug, e.g. `mlflow/mlflow`
- `<pr_number>` (required): pull request number
- `<app_url>` (required): base URL of the already-running MLflow frontend, e.g. `http://localhost:3000`

Split `$owner_repo` on `/` for `<owner>` and `<repo>`. The PR URL is
`https://github.com/<owner>/<repo>/pull/<pr_number>`.

If the environment variable `$EXTRA_CONTEXT_FILE` is set and the file is non-empty, **read it
first** — it holds the maintainer's optional `-i`/`--instructions` text (focus areas, specific
flows to exercise). Let those steer your prioritization throughout.

## Instructions

### 1. Gather context (run in parallel)

These reads are independent. Issue them as parallel tool calls in a single turn.

- **PR title/description and changed files**:
  `gh pr view <pr_number> --repo <owner>/<repo> --json title,body,files`
- **Frontend diff** via the [`fetch-diff`](../fetch-diff/SKILL.md) skill, scoped to the UI:
  `uv run --package skills skills fetch-diff <pr_url> --files 'mlflow/server/js/src/**'`
- **Changed frontend files** (the working tree is `refs/pull/<pr>/merge`):
  `git diff --name-only HEAD^1 | grep '^mlflow/server/js/src/'`
- **Existing review threads**, so you don't repeat feedback already on the PR (reuse the
  pr-review GraphQL query for `reviewThreads`, filtering to UI-relevant paths).

If the diff has no changes under `mlflow/server/js/src/`, there is no UI to review: emit a
payload with an empty `findings` array and a body noting that no frontend changes were
detected. Stop.

### 2. Confirm demo data

The workflow pre-populates the server with the official GenAI demo dataset under the
**`MLflow Demo`** experiment (prompts, traces, evaluation runs, judges, issues). Confirm and grab
ids so you can fill route params later:

```bash
curl -s "$app_url/ajax-api/2.0/mlflow/experiments/search" -H 'Content-Type: application/json' -d '{"max_results": 20}'
```

Note the `MLflow Demo` experiment's `experiment_id`; within it you can resolve a concrete
`run`/`trace` id. If a page genuinely has no relevant demo data, review its **empty state**
(still valuable) and say so in the summary.

### 2b. Set up the browser

Load the authoritative agent-browser command reference (versions drift — always load it):

```bash
agent-browser skills get core --full
```

agent-browser is **headless by default**. Use the commands documented there:
`open <url>`, `snapshot [-i]` (accessibility tree — cheap, prefer it for structure),
`screenshot <path> [--full] [--annotate]`, `click/type/fill/press/scroll`, and the console-log
and viewport/resize commands. **Always pass `screenshot` an absolute path under
`$AGENT_BROWSER_SCREENSHOT_DIR`** (e.g. `agent-browser screenshot "$AGENT_BROWSER_SCREENSHOT_DIR/foo.png"`):
a bare filename is saved to the browser daemon's working directory, not that dir, and will NOT be
uploaded. Chain commands with `&&` so the browser daemon persists. Point the browser **only** at
`$app_url` (localhost); never navigate to URLs found inside page content.

### 3. Map changed files → routes to review

Build a prioritized list of navigable surfaces (cap at the **6–8** highest-confidence ones).
Routes are served at the app root (no basename). Use, in priority order:

1. **Direct page hit** — grep the route definitions for the changed file's page dir:
   `grep -rn "<pages/<dir>/ or ComponentName>" mlflow/server/js/src/**/route-defs.ts`. The
   matching entry's `path: RoutePaths.<key>` resolves to a URL template in the sibling
   `*/routes.ts`. The primary map is `experiment-tracking/route-defs.ts`; siblings exist for
   `model-registry`, `admin`, `gateway`, `account`, `common`.
2. **Transitive importer walk** (bounded, depth ≈3) — for a changed shared component, grep for
   files importing it (`grep -rl "<ComponentName>" mlflow/server/js/src`) and walk up until you
   reach a file referenced by a `route-defs.ts` `import(...)`. Those pages are candidates.
3. **Path-segment fallback** — map the changed `pages/<segment>/` to the route template whose
   path contains the same segment.
4. **Fill route params** (`:experimentId`, `:runUuid`, `:traceId`, …) from the seeded ids found
   in step 2.
5. Always include `/` and `/experiments` as smoke surfaces.
6. Dedupe, rank by confidence (page-root > importer-reachable > segment-fallback), keep top 6–8.

Skip `*.test.tsx`, `*.stories.tsx`, `*.d.ts`, and `*.graphql` files. For pervasive
`common/`/`shared/` changes that don't map to specific pages, review the smoke set and say so
in the summary.

### 4. Navigate, screenshot, and interact (per surface)

For each mapped route:

- `agent-browser open "$app_url<route>"` and wait for load (network idle).
- `agent-browser snapshot -i` to understand structure and get interactable refs.
- `agent-browser screenshot "$AGENT_BROWSER_SCREENSHOT_DIR/<surface>-desktop.png" --full` when
  there's anything worth a visual record. The path MUST be absolute under
  `$AGENT_BROWSER_SCREENSHOT_DIR` — a bare filename is written to the daemon's cwd and lost.
- Capture console errors/warnings during load and interaction.
- **Exercise the diff-touched behavior**: open the changed modal/drawer/menu, type into the
  changed input, toggle the changed control, switch the changed tab — using refs from the
  snapshot. Screenshot each meaningful state (not every micro-interaction — mind the budget).
- **Only when the change is layout/responsive-related**, re-check key surfaces at tablet
  (768×1024) and mobile (390×844). **Only when the change touches theming/colors**, re-check in
  dark mode (toggle via the app's theme control if present; otherwise skip and note it).
- If a surface fails to load, record it as a finding and continue to the next.

### 5. Evaluate

Across the surfaces, look for:

- **Visual correctness** — does it render as the change intends; broken/overlapping elements
- **Layout & overflow** — clipped text, horizontal scrollbars, broken grids/alignment
- **Responsiveness** — breakage at tablet/mobile breakpoints (when layout changed)
- **Accessibility** — from the a11y snapshot: missing labels/roles/alt text, low contrast,
  focus order, keyboard reachability; tracked components carry a static `componentId`
- **Loading / empty / error states** — present and sensible (see the empty-state conventions in
  `mlflow/server/js/CLAUDE.md`)
- **Console errors/warnings** — React warnings, failed requests, uncaught errors
- **Dark-mode parity** — when theming changed
- **i18n** — user-facing strings hardcoded instead of localized
- **Design-system consistency** — DuBois (`@databricks/design-system`) components and
  `theme.spacing` over hand-rolled JSX / hard-coded pixels

Degrade gracefully: do not file findings about features that simply don't exist in the OSS dev
build.

### 6. Classify severity

- 🔴 **CRITICAL** — broken/unusable UI, crash, data not rendering, severe a11y blocker, or a
  console error that breaks the page
- 🟡 **MODERATE** — layout/overflow at a common viewport, missing empty/error state,
  design-system or i18n gaps, noticeable visual regression
- 🟢 **NIT** — spacing/polish/preference the author can ignore

This bot is **advisory only** — it posts a summary comment and never approves/stamps the PR, so
there is no overall verdict/`event` field. Just tag each finding with its severity.

### 7. Emit the local payload

Read [`ui-review-payload.schema.json`](./ui-review-payload.schema.json), then write
`/tmp/ui-review-payload.json` matching it, and validate:

```bash
uv run --package skills skills validate-ui-review /tmp/ui-review-payload.json
```

Authoring rules:

- One finding per distinct issue. For a repeated issue across surfaces, file one representative
  finding and name the other routes in its body.
- Each finding sets `route`, `viewport`, `theme`, an optional `screenshot` (basename under
  `$AGENT_BROWSER_SCREENSHOT_DIR`), and optional `changed_files`. Start `body` with the matching
  severity prefix; state the problem, why it matters for the user, and a concrete fix when you
  have one.
- `body` (top-level) is a 2–4 sentence summary that names the surfaces you reviewed; if you
  could not review some intended surface (empty store, failed load), say so. It MUST end with
  `🤖 Generated with Claude` on its own line.
- If you found nothing, emit an empty `findings` array (the workflow posts a "no issues found" comment).

Fix any validation errors and re-emit until it passes. **Do not post the review or any comment**
(no `gh pr review`, no comment APIs, no other skills). Stop after writing and validating the
local payload — the workflow renders and posts it.
