@echo off
echo ========================================
echo Bayesian Optimization Setup
echo ========================================
echo.

REM Check if uv is available
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install uv
        pause
        exit /b 1
    )
)

echo [1/3] Installing dependencies...
uv sync
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Registering Jupyter kernel...
uv run python -m ipykernel install --user --name bayesopt-fluorescence --display-name "Python (BayesOpt)"

echo.
echo [3/3] Done!
echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To get started:
echo   1. Run: uv run jupyter notebook
echo   2. Open workshop/workshop_notebook.ipynb
echo.
pause
