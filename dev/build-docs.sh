#!/usr/bin/env bash
# =============================================================================
# Static Site Build Script
#
# This script performs the following tasks:
#   1. Checks that NodeJS (>=18.0) is installed; if not, it prints instructions
#      for installing via nvm.
#   2. Changes directory into the docs folder so that all commands run from there.
#      (Note: This ensures that the DOCS_BASE_URL value is interpreted relative
#      to the docs folder. For example, if running from the project root, the
#      effective path will be: <root>/docs/docs/latest.)
#   3. Installs dependencies via npm.
#   4. (Optional) Builds the API docs:
#         - If --build-api-docs is provided, then the API docs are built.
#         - If --with-r-docs is also provided, the build includes R docs;
#           otherwise, R docs are skipped.
#   5. Converts notebooks to MDX.
#   6. Exports the DOCS_BASE_URL environment variable (default: /docs/latest) so
#      that Docusaurus uses the proper base URL.
#   7. Builds the static site.
#
# Once complete, the script instructs the user to navigate into the docs folder
# and run:
#
#     npm run serve -- --port <your_port_number>
#
# Options:
#   --build-api-docs        Opt in to build the API docs (default: do not build)
#   --with-r-docs           When building API docs, include R documentation 
#                           (default: skip R docs)
#   --docs-base-url URL     Override the default DOCS_BASE_URL (default: /docs/latest)
#   -h, --help              Display this help message and exit
#
# Example:
#   ./dev/build-docs.sh --build-api-docs --with-r-docs --docs-base-url /docs/latest
# =============================================================================

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as an error, and fail on pipeline errors.
set -euo pipefail

# -----------------------------------------------------------------------------
# Define color and style variables for styled output.
# -----------------------------------------------------------------------------
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# -----------------------------------------------------------------------------
# Logging functions for consistent output styling.
# -----------------------------------------------------------------------------
log_info()    { echo -e "${BOLD}${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${BOLD}${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${BOLD}${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${BOLD}${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# Default configuration values.
# -----------------------------------------------------------------------------
BUILD_API_DOCS=false
WITH_R_DOCS=false
# Default DOCS_BASE_URL is set as expected when running from within the docs folder.
# Note: When running from the project root, since we change into docs/,
# the effective reference becomes: <root>/docs/docs/latest.
DOCS_BASE_URL="/docs/latest"

# -----------------------------------------------------------------------------
# Display usage information.
# -----------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --build-api-docs        Opt in to build the API docs (default: do not build)
  --with-r-docs           When building API docs, include R documentation 
                          (default: skip R docs)
  --docs-base-url URL     Override the default DOCS_BASE_URL (default: /docs/latest)
  -h, --help              Display this help and exit

Example:
  $0 --build-api-docs --with-r-docs --docs-base-url /docs/latest
EOF
}

# -----------------------------------------------------------------------------
# Parse command-line arguments manually.
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-api-docs)
      BUILD_API_DOCS=true
      shift
      ;;
    --with-r-docs)
      WITH_R_DOCS=true
      shift
      ;;
    --docs-base-url)
      if [[ -n "${2:-}" ]]; then
          DOCS_BASE_URL="$2"
          shift 2
      else
          log_error "--docs-base-url requires an argument."
          usage
          exit 1
      fi
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# -----------------------------------------------------------------------------
# Function to compare semantic version numbers.
# Returns 0 (true) if version $1 is greater than or equal to version $2.
# -----------------------------------------------------------------------------
version_ge() {
    # Usage: version_ge "18.15.0" "18.0.0"
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

# -----------------------------------------------------------------------------
# Check that NodeJS is installed and meets the version requirement (>= 18.0).
# -----------------------------------------------------------------------------
if ! command -v node >/dev/null 2>&1; then
    log_error "NodeJS is not installed. Please install NodeJS (>= 18.0) from https://nodejs.org/ or via nvm."
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/v//')
REQUIRED_VERSION="18.0.0"
if ! version_ge "$NODE_VERSION" "$REQUIRED_VERSION"; then
    log_error "Detected NodeJS version $NODE_VERSION. Please install NodeJS >= $REQUIRED_VERSION."
    log_info "If you have nvm installed, you can run:"
    echo -e "${BOLD}nvm install node && nvm use node${NC}"
    exit 1
fi
log_success "NodeJS version $NODE_VERSION is valid."

# -----------------------------------------------------------------------------
# Check that npm is installed.
# -----------------------------------------------------------------------------
if ! command -v npm >/dev/null 2>&1; then
    log_error "npm is not installed. Please install npm from https://nodejs.org/."
    exit 1
fi

# -----------------------------------------------------------------------------
# Change directory into the docs folder so that all commands run from there.
# This ensures that DOCS_BASE_URL is interpreted correctly.
# -----------------------------------------------------------------------------
if [ ! -d "docs" ]; then
    log_error "The docs directory was not found. Make sure you're running this script from the project root."
    exit 1
fi

log_info "Changing directory to docs/ ..."
cd docs

# -----------------------------------------------------------------------------
# Install dependencies via npm.
# -----------------------------------------------------------------------------
log_info "Installing dependencies with npm..."
npm install
log_success "Dependencies installed."

# -----------------------------------------------------------------------------
# Optionally build the API documentation.
# This step is opt-in via the --build-api-docs flag.
# If building API docs, the --with-r-docs flag controls whether R docs are included.
# -----------------------------------------------------------------------------
if [ "$BUILD_API_DOCS" = true ]; then
    if [ "$WITH_R_DOCS" = true ]; then
        log_info "Building API docs including R documentation..."
        npm run build-api-docs
    else
        log_info "Building API docs without R documentation..."
        npm run build-api-docs:no-r
    fi
    log_success "API docs built successfully."
else
    log_info "Skipping API docs build phase."
fi

# -----------------------------------------------------------------------------
# Update the API module references for link functionality
# -----------------------------------------------------------------------------
log_info "Updating API module links..."
npm run update-api-modules
log_success "Updated API module links."

# -----------------------------------------------------------------------------
# Convert notebooks to MDX format.
# -----------------------------------------------------------------------------
log_info "Converting notebooks to MDX..."
npm run convert-notebooks
log_success "Notebooks converted to MDX."

# -----------------------------------------------------------------------------
# Export DOCS_BASE_URL and build the static site.
#
# Since we're in the docs folder, exporting DOCS_BASE_URL as "/docs/latest"
# means that when served from the project root, the built site will be available
# at <root>/docs/docs/latest.
# -----------------------------------------------------------------------------
export DOCS_BASE_URL
log_info "DOCS_BASE_URL set to '${DOCS_BASE_URL}'."
log_info "Building static site files with npm..."
npm run build
log_success "Static site built successfully."

# -----------------------------------------------------------------------------
# Final instructions for the user.
# -----------------------------------------------------------------------------
log_info "To run the site locally, please navigate to the 'docs' folder and execute:"
echo -e "${BOLD}npm run serve -- --port <your_port_number>${NC}"
log_info "For example: ${BOLD}npm run serve -- --port 3000${NC}"
log_success "Static site build process completed."
