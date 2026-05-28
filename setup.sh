#!/bin/bash
echo "========================================"
echo "Bayesian Optimization Setup"
echo "========================================"
echo

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install uv"
        exit 1
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[1/3] Installing dependencies..."
uv sync
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "[2/3] Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name bayesopt-fluorescence --display-name "Python (BayesOpt)"

echo
echo "[3/3] Done!"
echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To get started:"
echo "  1. Run: uv run jupyter notebook"
echo "  2. Open workshop/workshop_notebook.ipynb"
echo
