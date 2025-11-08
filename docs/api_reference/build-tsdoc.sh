#!/usr/bin/env bash

set -ex

# Function to build TypeDoc for a package
build_tsdoc() {
    local package_path=$1
    local package_name=$2
    local output_path=$3

    echo "Building TypeDoc for $package_name..."

    # Store the absolute path to the analytics file before changing directories
    local analytics_js_path="$(cd "$(dirname "$0")" && pwd)/typedoc-analytics.js"

    pushd "$package_path"

    # Skip npm install since we're using yarn workspaces
    # Dependencies should be installed at the workspace root

    # Generate TypeDoc documentation
    npx typedoc \
        --out "$output_path" \
        --name "$package_name" \
        --readme README.md \
        --tsconfig tsconfig.json \
        --excludePrivate \
        --excludeProtected \
        --excludeExternals \
        --includeVersion \
        --searchInComments \
        --navigation \
        --excludeNotDocumented false \
        --customJs "$analytics_js_path" \
        src/index.ts

    popd
}

# Base paths
TYPESCRIPT_BASE="../../libs/typescript"
DOCS_OUTPUT_BASE="build/html/typescript_api"

# First ensure dependencies are installed at workspace root
echo "Ensuring TypeScript workspace dependencies are installed..."
pushd "$TYPESCRIPT_BASE"
npm install
popd

# Remove existing docs if they exist
rm -rf "$DOCS_OUTPUT_BASE"
# Create output directory
mkdir -p "$DOCS_OUTPUT_BASE"

# Build documentation for mlflow-tracing
build_tsdoc \
    "$TYPESCRIPT_BASE/core" \
    "mlflow-tracing" \
    "$(pwd)/$DOCS_OUTPUT_BASE/mlflow-tracing"

# Build documentation for mlflow-openai
build_tsdoc \
    "$TYPESCRIPT_BASE/integrations/openai" \
    "mlflow-openai" \
    "$(pwd)/$DOCS_OUTPUT_BASE/mlflow-openai"

# Copy the HTML template to create index.html
cp "$(dirname "$0")/tsdoc.index.html.template" "$DOCS_OUTPUT_BASE/index.html"

echo "Copied TypeScript documentation into docs/build/html/typescript_api/"
