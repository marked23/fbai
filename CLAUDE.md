# CLAUDE.md - AI Assistant Guide for FizzBuzz Neural Network Trainer

## Project Overview

This is a **FizzBuzz neural network trainer** - a toy project that trains neural networks to solve the FizzBuzz problem (classifying numbers 1-100 by their divisibility by 3 and 5). The project supports parallel training runs with hyperparameter perturbation for experimentation.

**Goal**: Train a model that achieves 100% accuracy on FizzBuzz predictions for numbers 1-100.

## Codebase Structure

```
fbai/
├── main_parallel.py      # Primary entry point - uses torch.spawn() for parallel training
├── main.py               # Legacy entry point - uses Python multiprocessing (obsolete)
├── main_optuna.py        # Alternative entry point - uses Optuna for hyperparameter optimization
├── fizz_buzz_nn.py       # Neural network model definitions
├── hyperparameters.py    # Hyperparameters configuration class
├── perturbations.py      # PerturbRule system for varying hyperparameters across runs
├── loader.py             # Data loading utilities (training/validation/testing splits)
├── data_sample.py        # DataSample class - converts numbers to binary features + labels
├── animate.py            # Post-training animation generator for weight visualizations
├── plot.py               # Static weight visualization utility
├── lloging.py            # Logging utilities for single/multi-process execution
├── requirements.txt      # Python dependencies
└── results/              # Output directory (gitignored)
```

## Key Components

### Entry Points

| File | Usage | Description |
|------|-------|-------------|
| `main_parallel.py` | **Recommended** | Uses `torch.multiprocessing.spawn()` for parallel training |
| `main.py` | Legacy | Uses Python's `concurrent.futures` (obsolete) |
| `main_optuna.py` | Alternative | Optuna-based hyperparameter search |

### Core Modules

#### `fizz_buzz_nn.py` - Model Architectures
Contains multiple neural network architectures:
- **`Model`** - Default 2-layer network with configurable hidden dimension
- **`ClaudesModel`** - Sequential 3-layer network (currently used)
- **`WideModel`** - Shallow but wide architecture (64 units)
- **`DeepModel`** - 4 hidden layers of 32 units each
- **`PyramidModel`** - Tapering architecture (48→24→12)
- **`ImprovedModel`** - Includes BatchNorm and Kaiming initialization

All models accept a `Hyperparameters` instance and use `hp.input_dim`, `hp.output_dim`, `hp.hidden_dim`.

#### `hyperparameters.py` - Configuration
The `Hyperparameters` class is the central configuration object passed to all components.

Key parameters:
- `input_dim`: 10 (binary digits for numbers up to 1024)
- `output_dim`: 4 (classes: none, fizz, buzz, fizzbuzz)
- `hidden_dim`: Hidden layer size (default: 32, optimal: ~430)
- `initial_learning_rate`: Starting LR (default: 0.0063)
- `input_duplicates`: Training data duplication factor (default: 1, optimal: ~63)
- `epochs`: Maximum training epochs (default: 20000)
- `model_class_name`: Fully qualified model class (e.g., `"fizz_buzz_nn.ClaudesModel"`)
- `save_checkpoints`: Whether to save epoch checkpoints for animation

#### `perturbations.py` - Hyperparameter Grid Search
The `PerturbRule` system allows systematic variation of hyperparameters across parallel runs:

```python
rules = [
    PerturbRule("hidden_dim", start=14, step=1),              # Linear: 14,15,16...
    PerturbRule("learning_rate", start=0.001, step=2, multiply=True),  # Geometric
    PerturbRule("model_class_name", array=["fizz_buzz_nn.Model", ...], each=2),  # Categorical
]
apply_perturbations(hp_sets, rules)
```

#### `data_sample.py` - Feature Engineering
- Converts integers to 10-bit binary tensor representation
- Labels are computed via `gcd(n, 15)` mapping: {1: none, 3: fizz, 5: buzz, 15: fizzbuzz}

#### `loader.py` - Data Pipeline
- Training data: numbers 101-1023 (80% train, 20% validation)
- Testing data: numbers 1-100 (the actual FizzBuzz range)
- Training data is duplicated by `input_duplicates` factor

## Running the Project

### Basic Training
```bash
python main_parallel.py
```

### Animation Generation (after training with checkpoints)
```bash
python animate.py --all ./results/YYYY-MM-DD_HH_MM_SS/0
python animate.py --layer 1 ./results/YYYY-MM-DD_HH_MM_SS/0
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Output Structure

Training creates timestamped output in `results/`:
```
results/
└── YYYY-MM-DD_HH_MM_SS/           # Run timestamp
    ├── parallel_execution.log     # Combined log from all ranks
    ├── winners.txt                # Summary of successful runs
    └── {rank}/                    # Per-process output
        ├── hyperparameters.json   # Configuration snapshot
        ├── model.pth              # Final model (only if 100% accuracy)
        └── checkpoints/           # Epoch checkpoints (if enabled)
            ├── model_000000.pth
            ├── model_000001.pth
            └── ...
```

## Development Conventions

### Code Style
- Use type hints where present in existing code
- Follow existing import organization: standard library, third-party, local
- Hyperparameters should be accessed via the `hp` object, not hardcoded

### Model Development
- New models should follow the pattern in `fizz_buzz_nn.py`
- Accept `hp: Hyperparameters` in `__init__`
- Use `hp.input_dim` and `hp.output_dim` for layer sizing
- Register new models by adding to imports in entry point files

### Logging
- Use `hp.spit()` for output (routes to appropriate logging handler)
- Prefix multi-process output with rank: `[{rank:>4}]`

### Experimentation Workflow
1. Modify `PerturbRule` definitions in `main_parallel.py`
2. Adjust `world_size` for number of parallel runs
3. Run training
4. Check `results/{timestamp}/winners.txt` for successful configurations
5. Optionally generate animations with `animate.py`

## Known Good Configurations

From README, these parameters achieve fast convergence (6 epochs to perfection):
- `hidden_dim`: 410 or 430
- `input_duplicates`: 57 or 63
- `initial_learning_rate`: 0.0097

## Dependencies

Key libraries:
- **PyTorch** (2.5.1) - Neural network framework
- **NumPy** (2.1.3) - Numerical operations
- **Matplotlib/Seaborn** - Visualization and animation
- **Optuna** - Hyperparameter optimization (optional)
- **PyQt5** - GUI backend for plots

GPU support via CUDA 12.4 (optional, falls back to CPU).

## Notes for AI Assistants

- The primary entry point is `main_parallel.py`, not `main.py`
- Perfect accuracy means 100/100 on the test set (numbers 1-100)
- The `spit` attribute on `Hyperparameters` is a callable for logging
- Watch for the typo in `lloging.py` filename (intentional, not a bug)
- Model selection is dynamic via `model_class_name` string
- The `results/` directory is gitignored - don't expect training outputs in git
