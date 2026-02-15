#!/usr/bin/env bash
# Script to create missing size/* labels for PR Size Labeling workflow
# Requires: GitHub CLI (gh) and repository admin/write permissions
# Reference: https://github.com/mlflow/mlflow/actions/runs/22018853511/job/63624663902?pr=20816

set -e

REPO="mlflow/mlflow"

echo "Creating missing size/* labels in ${REPO}..."
echo ""

# Create missing labels
echo "Creating size/S label..."
gh label create "size/S" --repo "$REPO" --color "ededed" --description "Small PR (10-49 LoC)" || echo "size/S already exists"

echo "Creating size/L label..."
gh label create "size/L" --repo "$REPO" --color "ededed" --description "Large PR (200-499 LoC)" || echo "size/L already exists"

echo "Creating size/XL label..."
gh label create "size/XL" --repo "$REPO" --color "ededed" --description "Extra-large PR (500+ LoC)" || echo "size/XL already exists"

echo ""
echo "Adding descriptions to existing labels..."

# Add descriptions to existing labels for consistency
echo "Updating size/XS label..."
gh label edit "size/XS" --repo "$REPO" --description "Extra-small PR (0-9 LoC)" || echo "Failed to update size/XS"

echo "Updating size/M label..."
gh label edit "size/M" --repo "$REPO" --description "Medium PR (50-199 LoC)" || echo "Failed to update size/M"

echo ""
echo "Done! Verifying labels..."
gh label list --repo "$REPO" --limit 200 | grep "size/"
