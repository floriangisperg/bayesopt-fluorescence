#!/bin/bash
echo "========================================"
echo "Bayesian Optimization Setup"
echo "========================================"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[1/3] Creating conda environment..."
conda env create -f environment.yml --force
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create environment"
    exit 1
fi

echo
echo "[2/3] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate bayesopt-fluorescence

echo
echo "[3/3] Registering Jupyter kernel..."
python -m ipykernel install --user --name bayesopt-fluorescence --display-name "Python (BayesOpt)"

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To get started:"
echo "  1. Run: conda activate bayesopt-fluorescence"
echo "  2. Run: jupyter notebook"
echo "  3. Open workshop_notebook.ipynb"
echo
