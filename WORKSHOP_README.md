# Workshop Quickstart

This file is the short notebook-first entry point for workshop participants.

For the full project documentation, setup details, configuration notes, and the current constraint description, see [README.md](README.md).

## Installation

Prerequisite: install Anaconda or Miniconda.

### Option A: Double-click setup

| Windows     | Mac/Linux  |
| ----------- | ---------- |
| `setup.bat` | `setup.sh` |

### Option B: Terminal

```bash
conda env create -f environment.yml
conda activate bayesopt-fluorescence
```

## Start the Notebook

```bash
conda activate bayesopt-fluorescence
jupyter notebook
```

Open `workshop_notebook.ipynb` and follow the cells in order.

## Workshop Loop

1. Define the campaign settings in the notebook.
2. Generate the initial experimental plan.
3. Run the experiments and fill in the Excel file.
4. Load the measured results.
5. Train the models.
6. Generate the next AI-suggested batch.
7. Repeat.

## If You Prefer the Terminal

Use [CLI_GUIDE.md](CLI_GUIDE.md) for the script-based workflow.

## Files You Will Use

- `workshop_notebook.ipynb`: guided workflow
- `workshop_results/`: generated workshop outputs
- `config.py`: advanced configuration, including the urea constraint toggle
