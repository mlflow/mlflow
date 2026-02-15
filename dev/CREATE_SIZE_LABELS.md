# Create Missing size/\* Labels

## Problem

The PR Size Labeling workflow (`.github/workflows/pr-size.yml`) fails when trying to add labels that don't exist in the repository. The workflow uses `gh pr edit --add-label` which exits with code 1 if the label doesn't exist.

Reference: https://github.com/mlflow/mlflow/actions/runs/22018853511/job/63624663902?pr=20816

## Current State

Only two size labels currently exist:

- `size/XS` (no description)
- `size/M` (no description)

Missing labels: `size/S`, `size/L`, `size/XL`

## Solution

Run the provided script with appropriate permissions:

```bash
bash dev/create-size-labels.sh
```

**Prerequisites:**

- GitHub CLI (`gh`) installed and authenticated
- Repository admin or write permissions for `mlflow/mlflow`

## Manual Commands

If you prefer to run commands manually:

```bash
# Create missing labels
gh label create "size/S" --repo mlflow/mlflow --color "ededed" --description "Small PR (10-49 LoC)"
gh label create "size/L" --repo mlflow/mlflow --color "ededed" --description "Large PR (200-499 LoC)"
gh label create "size/XL" --repo mlflow/mlflow --color "ededed" --description "Extra-large PR (500+ LoC)"

# Add descriptions to existing labels (optional, for consistency)
gh label edit "size/XS" --repo mlflow/mlflow --description "Extra-small PR (0-9 LoC)"
gh label edit "size/M" --repo mlflow/mlflow --description "Medium PR (50-199 LoC)"
```

## Label Thresholds

The PR size is calculated based on total lines of code (additions + deletions) excluding generated files:

| Label     | Description    | Lines of Code |
| --------- | -------------- | ------------- |
| `size/XS` | Extra-small PR | 0-9           |
| `size/S`  | Small PR       | 10-49         |
| `size/M`  | Medium PR      | 50-199        |
| `size/L`  | Large PR       | 200-499       |
| `size/XL` | Extra-large PR | 500+          |

## Verification

After creating the labels, verify they exist:

```bash
gh label list --repo mlflow/mlflow --limit 200 | grep "size/"
```

Expected output:

```
size/L      Large PR (200-499 LoC)          #ededed
size/M      Medium PR (50-199 LoC)          #ededed
size/S      Small PR (10-49 LoC)            #ededed
size/XL     Extra-large PR (500+ LoC)       #ededed
size/XS     Extra-small PR (0-9 LoC)        #ededed
```

## Note

The current workflow on master uses `actions/github-script` with `github.rest.issues.addLabels`, which auto-creates labels if they don't exist (with default gray color and no description). Creating labels explicitly ensures consistent styling and meaningful descriptions.
