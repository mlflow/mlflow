#!/usr/bin/env bash
# Checks that api_inventory.txt is up to date with the documented APIs.
# This rebuilds the Sphinx docs and verifies the inventory file hasn't changed.

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT/docs/api_reference"

cleanup() {
    rm -rf build
}
trap cleanup EXIT

# Clean before build to ensure fresh doctrees
rm -rf build

echo -e "${CYAN}Building API docs to regenerate inventory...${NC}"
uv run --group docs --extra gateway python gateway_api_docs.py
uv run --group docs --extra gateway sphinx-build -b html -W --keep-going -n -T -d build/doctrees source build/html

if [ -n "$(git status --porcelain api_inventory.txt)" ]; then
    echo ""
    echo -e "${BOLD}${RED}ERROR: api_inventory.txt is outdated.${NC}"
    echo ""
    echo "If adding new APIs, mark them with @experimental if appropriate."
    echo ""
    echo "Diff:"
    git --no-pager diff api_inventory.txt
    exit 1
fi

echo -e "${BOLD}${GREEN}API inventory is up to date.${NC}"
