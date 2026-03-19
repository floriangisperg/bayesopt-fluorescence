# Bayesian Optimization Workshop

Welcome to the hands-on Bayesian Optimization workshop!

---

## Installation

**Prerequisite:** Anaconda or Miniconda must be installed.
- Download from: https://docs.conda.io/en/latest/miniconda.html

---

### Option A: Double-click setup (Easiest)

Just double-click the setup file in this folder:

| Windows | Mac/Linux |
|---------|-----------|
| `setup.bat` | `setup.sh` |

Wait for it to finish. Done!

---

### Option B: Use the terminal

Open Anaconda Prompt (Windows) or Terminal (Mac/Linux), navigate to this folder, and run:

```bash
conda env create -f environment.yml
conda activate bayesopt-fluorescence
```

## Getting Started

After installation:

```bash
conda activate bayesopt-fluorescence
jupyter notebook
```

Then open `workshop_notebook.ipynb` and follow the step-by-step guide.

**Prefer the command line?** See [CLI_GUIDE.md](CLI_GUIDE.md) for the terminal-based workflow.

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
