## Documentation Sync Report

### Executive Summary
The codebase and documentation are highly consistent. Parameter names, default values, and workflow descriptions in `README.md` and `CLI_GUIDE.md` accurately reflect the implementation in the Python scripts. Constraints logic is correctly implemented as described. One minor code-config decoupling issue was identified.

### In Sync
- **CLI Parameters:** All arguments for `generate_initial_design.py`, `train_models.py`, and `run_optimization.py` match `CLI_GUIDE.md`.
- **Configuration:** Default values in `config.py` (DTT, GSSG, Dilution Factor, pH, Urea) match `README.md`.
- **Workflow:** File output patterns (Excel files, model paths) match the documentation.
- **Constraints:** The urea constraint logic (`final_urea * dilution_factor > solubilization_urea`) is correctly implemented in `constraints/urea_dilution.py`.

### Code Implementation Fixes
- [P2] `generate_initial_design.py`: Does not pass `ConstraintConfig.SOLUBILIZATION_UREA` to `generate_initial_design()`.
    - **Issue:** The function uses its default argument (`8.0`) instead of the value from `config.py`. If a user changes the config, the initial design generation will still use 8.0M.
    - **Suggestion:** Update `generate_initial_design.py` to pass `solubilization_urea=ConstraintConfig.SOLUBILIZATION_UREA`.

### Bloat Candidates
- `config.py`: `PathConfig` defines `MODEL1_NAME` ("model1_...") and `MODEL2_NAME` ("model2_...") which are unused.
    - **Observation:** `train_models.py` dynamically generates filenames like "model_1_..." (with underscore).
    - **Action:** Remove unused constants from `PathConfig` or update `train_models.py` to use them to ensure single source of truth.
