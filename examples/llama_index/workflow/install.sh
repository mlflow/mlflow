#!/bin/bash

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No color

# Ensure that poetry is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Install poetry if it's not installed
if ! command -v poetry &> /dev/null
then
    echo -e "${YELLOW}Poetry not found, installing...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo -e "${GREEN}Poetry is already installed.${NC}"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo -e "${RED}Docker not found.${NC} ${CYAN}Please install Docker to use Qdrant, or use the '--no-qdrant' flag if you don't need Qdrant.${NC}"
else
    echo -e "${GREEN}Docker is already installed.${NC}"
fi

# Check for --no-qdrant flag
if [[ "$*" == *"--no-qdrant"* ]]; then
    echo -e "${CYAN}Skipping Docker check because --no-qdrant flag was used.${NC}"
fi

# Install Jupyter Notebook
echo -e "${CYAN}Installing Jupyter Notebook...${NC}"
poetry run pip install jupyter

# Install dependencies from pyproject.toml
echo -e "${CYAN}Installing dependencies from pyproject.toml...${NC}"
poetry install

echo -e "${GREEN}All dependencies installed successfully.${NC}"
