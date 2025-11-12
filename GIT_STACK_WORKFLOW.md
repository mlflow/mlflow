# Git Stack Workflow for MLflow Secrets Feature

**IMPORTANT**: This repo uses `git stack` for stacked PRs. NEVER use regular `git push` or `git commit --amend` without following up with `git stack sync`.

Full documentation: go/stacked-prs

---

## Initial Setup (Already Done)

```bash
# Create first PR branch from master
git checkout master
git stack create secrets/db-migration-tables
```

---

## Making Changes

### Option 1: New commit
```bash
# Make changes
git add <files>
git stack commit -m "commit message"
# This automatically runs: git commit + git stack sync
```

### Option 2: Amend existing commit
```bash
# Make changes
git add <files>
git stack amend
# This automatically runs: git commit --amend + git stack sync
```

### Option 3: Manual workflow (AVOID - easy to forget sync!)
```bash
# Make changes
git add <files>
git commit --amend --no-edit  # OR: git commit -m "message"
git stack sync  # CRITICAL: Must sync after ANY commit operation!
# ⚠️ Better to just use `git stack commit` or `git stack amend` instead
```

---

## Pushing Changes

**ALWAYS use git stack push, NEVER git push!**

```bash
# Push current branch and create/update PR
git stack push origin

# Options:
# --only: Only push current branch
# --skip-ancestor: Skip parents
# --publish: Create all PRs as formal (not draft)
```

---

## Viewing Stack

```bash
# List all stacks
git stack ls

# List only current stack
git stack ls -c
```

---

## Creating Next PR in Stack

```bash
# Create child PR (builds on current branch)
git stack create secrets/entity-layer

# Make changes
git add <files>
git stack commit -m "Add entity layer and store interface"

# Push to create PR
git stack push origin
```

---

## Fixing Issues After Push

### If you forgot to add files (like we just did with schema files)

**WRONG WAY** (what I just did):
```bash
git add tests/db/schemas/*.sql
git commit --amend --no-edit
git push origin --force-with-lease  # ❌ BREAKS STACK METADATA!
```

**RIGHT WAY**:
```bash
# 1. Update schemas first (if needed)
./tests/db/update_schemas.sh

# 2. Stage the new files
git add tests/db/schemas/*.sql

# 3. Amend commit using git stack (does commit + sync)
git stack amend

# 4. Push using git stack (updates PR with correct metadata)
git stack push origin
```

---

## After PR is Merged

```bash
# Update master
git checkout master && git pull databricks master

# Go to next PR in stack
git checkout secrets/entity-layer

# Rebase onto master (removes merged parent)
git stack rebase master

# Remove merged branch from stack
git stack remove secrets/db-migration-tables

# Sync and push updated stack
git stack sync
git stack push origin
```

---

## Handling Merge Conflicts

If `git stack sync` hits merge conflicts:

```bash
# Fix conflicts in editor
git add <resolved-files>

# Continue the sync
git rebase --continue && git stack sync --continue --from <current-branch>
```

---

## Common Mistakes to Avoid

❌ **DON'T**: Use `git push` directly
✅ **DO**: Use `git stack push origin`

❌ **DON'T**: Use `git commit --amend` without `git stack sync`
✅ **DO**: Use `git stack commit --amend` (does both)

❌ **DON'T**: Force push with `git push --force`
✅ **DO**: Use `git stack push origin` (handles force push correctly)

❌ **DON'T**: Create branches with `git checkout -b`
✅ **DO**: Use `git stack create <branch-name>`

---

## Useful Aliases

```bash
alias gs='git stack'
alias gsc='git stack commit'
alias gsp='git stack push origin'
alias gsl='git stack ls -c'
```

---

## Current Stack Structure

```
master
  └── stack/secrets/db-migration-tables (PR #1) - DB Schema
       └── stack/secrets/entity-layer (PR #2) - Entities & Store [TO BE CREATED]
            └── stack/secrets/crypto (PR #3) - Cryptography [TO BE CREATED]
                 └── ... (more PRs)
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Create new PR in stack | `git stack create <branch-name>` |
| Commit changes | `git stack commit -m "message"` |
| Amend commit | `git stack amend` (or `git stack amend -a` to include unstaged) |
| Push to remote | `git stack push origin` |
| View stack | `git stack ls -c` |
| Sync after changes | `git stack sync` |
| Rebase to new parent | `git stack rebase <target-branch>` |
| Remove branch | `git stack remove <branch-name>` |

---

## What I Did Wrong (Example - Don't Do This!)

1. ❌ Used `git commit --amend --no-edit` directly
2. ❌ Used `git push origin stack/secrets/db-migration-tables --force-with-lease`
3. ✅ Should have used: `git stack amend` then `git stack push origin`

**Why this is bad:**
- Bypasses git stack's metadata tracking
- PR diff links in description become outdated/broken
- Stack sync state gets out of sync

**How to fix if you did this:**
```bash
# Re-sync the stack to fix metadata
git stack sync

# Re-push to update PR with correct stack info
git stack push origin
```
