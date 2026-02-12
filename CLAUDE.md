# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A toy FizzBuzz neural network trainer. Trains PyTorch models to predict FizzBuzz labels (1, 3, 5, 15 = gcd(n, 15)) from 10-bit binary input. Training data is numbers 101-1023; test set is 1-100. "Perfect" = 100/100 on test set.

## Commands

```bash
# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training (preferred entry point - uses torch.mp.spawn for parallel ranks)
python main_parallel.py

# Run training with Optuna hyperparameter search
python main_optuna.py

# Run training (obsolete - uses Python multiprocessing)
python main.py

# Generate weight layer animations from saved checkpoints
python animate.py --all <process_path>           # all layers
python animate.py --layer <N> <process_path>     # specific layer
python animate.py --step 5 --all <process_path>  # skip frames
# Example: python animate.py --all ./results/2024-03-14_16_16_28/0
```

## Architecture

**Entry points** share the same train/test loop pattern but differ in parallelism strategy:
- `main_parallel.py` — preferred. Uses `torch.mp.spawn()` with ranks. Configures perturbation rules in `__main__` block, then spawns workers. Model class is resolved dynamically from `hp.model_class_name` string (e.g. `"fizz_buzz_nn.ClaudesModel"`).
- `main_optuna.py` — Optuna study with `objective()` function. Uses SQLite storage (`optuna_study.db`). Same dynamic model creation.
- `main.py` — obsolete. Uses `concurrent.futures.ProcessPoolExecutor`. Hardcodes `Model` class.

**Core modules:**
- `hyperparameters.py` — `Hyperparameters` class is the central config object, injected everywhere. Accepts kwargs; `__str__` serializes to JSON. `HyperparametersLoader` reconstructs from saved JSON (used by `animate.py`). Note: `Hyperparameters.__init__` takes `**kwargs` (not positional args), but `main.py` and `main_parallel.py` still call it with positional `(rank, datetime)` — this is a known inconsistency.
- `perturbations.py` — `PerturbRule` dataclass + `apply_perturbations()`. Supports linear sweep (`start + step * i`), geometric (`start * step^i`), and array cycling (with `each` parameter for repetition).
- `fizz_buzz_nn.py` — Multiple model architectures: `Model`, `WideModel`, `DeepModel`, `PyramidModel`, `ImprovedModel`, `ClaudesModel`. All take `Hyperparameters` and use `hp.input_dim`/`hp.output_dim`/`hp.hidden_dim`.
- `loader.py` — `Loader` with static methods. Training data (101-1023) split 80/20 train/val, training portion duplicated `hp.input_duplicates` times. Test data is 1-100.
- `data_sample.py` — `DataSample` NamedTuple. Encodes number as 10-bit binary features, label = index into `[1, 3, 5, 15]` of `gcd(n, 15)`.
- `lloging.py` — Multi-process logging via `QueueHandler`/`QueueListener`. Note the intentional filename typo.
- `animate.py` — Post-training visualization. Loads checkpoints, creates seaborn heatmap animations of weight matrices as MP4 (uses h264_nvenc codec).
- `plot.py` — Standalone script for static weight/bias visualization of a single saved model.

## Output Structure

Results go to `./results/<timestamp>/<rank>/`:
- `hyperparameters.json` — config snapshot
- `model.pth` — saved on 100/100 test accuracy
- `checkpoints/model_NNNNNN.pth` — per-epoch (requires `save_checkpoints=True` in Hyperparameters)
- `winners.txt` — appended in `main_parallel.py` when a rank achieves perfection

## Known Good Hyperparameters

`hidden_dim=430`, `input_duplicates=63`, `initial_learning_rate=0.0097` — can achieve 100/100 in ~6 epochs.
