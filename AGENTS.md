# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
VisNet is a Python research project implementing biologically-inspired hierarchical vision models. It consists of standalone `.py` scripts (no package structure beyond `setup.py`). See `README.md` for model variant descriptions.

### Dependencies
All dependencies are declared in `setup.py` (`install_requires` + `extras_require["dev"]`). Install with:
```
pip install -e ".[dev]"
```
The README references `pip install -r requirements.txt` but **no `requirements.txt` file exists**; use `setup.py` instead.

### Running
Each model variant is a standalone script with its own `main()`. Example:
```
python VisNet_Simplified_MNIST.py
```
Scripts default to 100 training epochs and auto-download datasets (MNIST/CIFAR-10) on first run. No GPU is required; scripts auto-detect and fall back to CPU.

### Linting
```
flake8 . --max-line-length=120
black --check .
```
Note: existing code has many flake8/black violations — this is normal for research code.

### Testing
```
pytest
```
No test files currently exist in the repo. The dev extras install `pytest`, `black`, and `flake8`.

### Gotchas
- `~/.local/bin` must be on `PATH` for `flake8`, `black`, `pytest` CLI commands when installed with `--user`.
- The `data/` directory is created at runtime by torchvision for dataset downloads. It is not committed to the repo.
- Training scripts are CPU-bound and slow without a GPU. For quick verification, reduce `LAYER_SIZE` and `NUM_EPOCHS` in the script's `main()` function.
