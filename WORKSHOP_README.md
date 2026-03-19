# Bayesian Optimization Workshop

Welcome to the hands-on Bayesian Optimization workshop!

## Before You Start

### 1. Install Anaconda (if not already installed)

Download from: https://www.anaconda.com/download

### 2. Create a conda environment

Open Anaconda Prompt (Windows) or Terminal (Mac/Linux) and run:

```bash
conda create -n bayesopt python=3.9
conda activate bayesopt
```

### 3. Install required packages

Navigate to this folder and run:

```bash
cd path/to/bayesopt-fluorescence
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas torch gpytorch botorch matplotlib openpyxl scipy scikit-learn
```

## Running the Workshop

1. Open Anaconda Navigator
2. Launch Jupyter Notebook
3. Navigate to this folder
4. Open `workshop_notebook.ipynb`
5. Follow the instructions in the notebook!

## Quick Start Guide

1. **Edit Step 2** - Define your experiment parameters and objectives
2. **Run Step 3** - Generate your initial experimental design
3. **Do experiments** - Perform the experiments and record results in the Excel file
4. **Run Step 4** - Load your results
5. **Run Step 5** - Train the model
6. **Run Step 6** - Get AI-suggested next experiments
7. **Repeat** steps 3-6 for more iterations

## Need Help?

- Check the "Tips & Troubleshooting" section at the end of the notebook
- Ask the workshop instructor

## Files

- `workshop_notebook.ipynb` - The main workshop notebook
- `workshop_results/` - Your experimental results will be saved here
- `config.py` - Advanced configuration (usually don't need to edit)
