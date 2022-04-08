#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

echo "You're about to sync web-shared to web-shared-bundle"
echo "NOTE: You're responsible for the regression and compatibilty risk"
read -p "Press enter to continue"

UNIVERSE_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
MLFLOW_DIR="$UNIVERSE_ROOT/mlflow/web/js"
WEBSHARED_DIR="$UNIVERSE_ROOT/js/packages/web-shared"
DEST_DIR="$MLFLOW_DIR/src/__generated__/web-shared-bundle"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"
mkdir -p "$DEST_DIR/dist"
(cd "$WEBSHARED_DIR" && yarn install && yarn build)

for SUB_PACKAGE in hooks metrics mfe-services
do
  cp "$WEBSHARED_DIR/$SUB_PACKAGE.mjs" "$DEST_DIR"
  cp "$WEBSHARED_DIR/$SUB_PACKAGE.d.ts" "$DEST_DIR"
done
cp -r "$WEBSHARED_DIR/dist" "$DEST_DIR/"
rm -rf "$DEST_DIR/dist/cjs"
