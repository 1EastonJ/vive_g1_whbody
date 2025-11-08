#!/bin/bash
# ===================================================
# Setup instructions for Mink_RL / MujocoDeploy (UV version)
# ===================================================

set -e

cd "$(dirname "$0")"

# Step 1: Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    pip install uv
fi

# Step 2: Create virtual environment using uv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment using uv..."
    uv venv .venv
else
    echo "Virtual environment already exists."
fi

# Step 3: Activate the virtual environment
source .venv/bin/activate

# Step 4: Install dependencies
cd MujocoDeploy
echo "Installing dependencies..."
uv pip install -r ../requirements.txt

echo "Setup completed successfully."
