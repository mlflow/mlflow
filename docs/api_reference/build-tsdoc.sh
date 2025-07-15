#!/usr/bin/env bash

set -ex

# Function to build TypeDoc for a package
build_tsdoc() {
    local package_path=$1
    local package_name=$2
    local output_path=$3

    echo "Building TypeDoc for $package_name..."

    pushd "$package_path"

    # Skip npm install since we're using yarn workspaces
    # Dependencies should be installed at the workspace root

    # Build the TypeScript code first
    npm run build

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
        src/index.ts

    popd
}

# Base paths
TYPESCRIPT_BASE="../../packages/typescript"
DOCS_OUTPUT_BASE="build/html/typescript_api"

# First ensure yarn dependencies are installed at workspace root
echo "Ensuring TypeScript workspace dependencies are installed..."
pushd "$TYPESCRIPT_BASE"
yarn install
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

# Create an index.html with just the header section and API documentation links
cat > "$DOCS_OUTPUT_BASE/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>MLflow TypeScript SDK Documentation</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f8f9fa;
        }
        .content {
            background-color: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 3rem;
            padding-bottom: 2rem;
            border-bottom: 2px solid #e1e4e8;
        }
        .header h1 {
            font-size: 2.5rem;
            color: #2c3e50;
            margin: 1rem 0;
            font-weight: 600;
        }
        .header .logo {
            margin-bottom: 1rem;
        }
        .header .logo img {
            width: 200px;
            height: auto;
        }
        .description {
            font-size: 1.1rem;
            color: #586069;
            text-align: center;
            margin: 2rem 0;
        }
        .description a {
            color: #0366d6;
            text-decoration: none;
        }
        .description a:hover {
            text-decoration: underline;
        }
        .api-docs-section {
            margin-top: 3rem;
        }
        .api-docs-section h2 {
            font-size: 1.8rem;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        .package-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        .package-item {
            background-color: #f6f8fa;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e1e4e8;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .package-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .package-item h3 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.3rem;
        }
        .package-item p {
            margin: 0.5rem 0;
            color: #586069;
        }
        .package-link {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.5rem 1rem;
            background-color: #0366d6;
            color: white;
            border-radius: 4px;
            font-weight: 500;
            text-decoration: none;
        }
        .package-link:hover {
            background-color: #0256c7;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="content">
        <div class="header">
            <div class="logo">
                <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" alt="MLflow Logo" />
            </div>
            <h1>MLflow TypeScript SDK</h1>
            <div class="description">
                This is the API reference for the MLflow TypeScript SDK, a set of packages for integrating your Node.js applications with <a href="https://mlflow.org">MLflow</a>.
            </div>
        </div>

        <div class="quickstart-section">
            <h2>🚀 New to MLflow?</h2>
            <p>Get started with our comprehensive quickstart guide that walks you through setting up MLflow tracing for your TypeScript applications.</p>
            <a href="https://mlflow.org/docs/latest/genai/tracing/quickstart/typescript-openai" class="quickstart-link">View Quickstart Guide →</a>
        </div>

        <div class="api-docs-section">
            <h2>📚 API Reference</h2>
            <p>Explore the detailed API documentation for each package:</p>
            <div class="package-list">
                <div class="package-item">
                    <h3>mlflow-tracing</h3>
                    <p>Core tracing functionality and manual instrumentation for MLflow TypeScript SDK.</p>
                    <a href="mlflow-tracing/index.html" class="package-link">View API Docs →</a>
                </div>
                <div class="package-item">
                    <h3>mlflow-openai</h3>
                    <p>Auto-instrumentation integration for OpenAI, making it easy to trace OpenAI API calls.</p>
                    <a href="mlflow-openai/index.html" class="package-link">View API Docs →</a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
EOF

echo "Copied TypeScript documentation into docs/build/html/typescript_api/"