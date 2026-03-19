@echo off
echo ========================================
echo Bayesian Optimization Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/3] Creating conda environment...
conda env create -f environment.yml --force
if %errorlevel% neq 0 (
    echo ERROR: Failed to create environment
    pause
    exit /b 1
)

echo.
echo [2/3] Activating environment...
call conda activate bayesopt-fluorescence

echo.
echo [3/3] Registering Jupyter kernel...
python -m ipykernel install --user --name bayesopt-fluorescence --display-name "Python (BayesOpt)"

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To get started:
echo   1. Close this terminal
echo   2. Open a new terminal
echo   3. Run: conda activate bayesopt-fluorescence
echo   4. Run: jupyter notebook
echo   5. Open workshop_notebook.ipynb
echo.
pause
