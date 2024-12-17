# 2. Reorder imports to handle dependencies
from __future__ import annotations

# Standard library imports first
import gc
import json
import logging
import logging.handlers
import os
import random
import signal
import sys
import time
from datetime import datetime
from typing import Callable, NamedTuple, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch import Size, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import optuna
from optuna import create_study, create_trial, Trial

# Local imports
import fizz_buzz_nn
from fizz_buzz_nn import Model, WideModel, DeepModel, PyramidModel, ImprovedModel
from data_sample import DataSample
from hyperparameters import Hyperparameters
from loader import Loader as loader
from lloging import setup_logging, setup_logger, listener_process
from perturbations import apply_perturbations, PerturbRule


class FixedEpochsPruner(optuna.pruners.BasePruner):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        step = trial.last_step
        if step is None:
            return False
        return step >= self.max_epochs

def setup(rank):
    torch.manual_seed(42)

def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
def save_hyperparameters(hp: Hyperparameters):
    os.makedirs(hp.process_path, exist_ok=True)
    with open(f"{hp.process_path}/hyperparameters.json", "w") as f:
        f.write(str(hp))
    # with open(f"{hp.process_path}/hyperparameters_dump.json", "w") as d:
    #     json.dump(hp.__dict__, d, indent=4)
          

def train(model: nn.Module, training_loader: DataLoader, optimizer: optim.Optimizer, hp: Hyperparameters) -> Tuple[torch.nn.Module, float]:
    model.train()
    total_loss = 0
    criterion = hp.criterion

    for features, labels in training_loader:
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(training_loader.dataset)
    return model, avg_loss

def test(model: nn.Module, data_loader: DataLoader, hp: Hyperparameters) -> Tuple[int, float]:
    model.eval()
    total_loss = 0
    num_correct = 0
    criterion = hp.criterion

    with torch.no_grad():
        for features, labels in data_loader:
            predictions = model(features)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            num_correct += (predictions.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(data_loader.dataset) 
    return num_correct, avg_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_model(hp):
    module_name, class_name = hp.model_class_name.split('.')
    module = globals()[module_name]
    model_class = getattr(module, class_name)
    return model_class(hp).to(hp.device)

def objective(trial: Trial):
    # Suggest hyperparameters
    suggested_params = {
        'initial_learning_rate'           : trial.suggest_float      ('initial_learning_rate', 0.0063, 0.0063, log=True),
        'hidden_dim'                      : trial.suggest_int        ('hidden_dim', 430, 430),
        'input_duplicates'                : trial.suggest_int        ('input_duplicates', 63, 63),
        'model_class_name'                : trial.suggest_categorical('model_class_name', [
            "fizz_buzz_nn.ClaudesModel"
        ])
    }
    
    hp = Hyperparameters(**suggested_params)
    hp.parameter_set_id = trial.number
    
    print(f"\nTrial {trial.number} suggested parameters:")
    for name, value in suggested_params.items():
        print(f"  {name:>25}: {value}")
   
    # Set up data loaders and model
    set_seed(hp.seed)
    training_loader, validation_loader = loader.create_training_loader(hp)
    testing_loader = loader.create_testing_loader(hp)
    model = create_model(hp).to(hp.device)
    optimizer = optim.Adam(model.parameters(), lr=hp.initial_learning_rate, weight_decay=hp.weight_decay)

    best_score = 0.0
    patience = hp.max_patience
    train_losses = []
    val_losses = []

    for epoch in range(hp.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for features, labels in training_loader:
            optimizer.zero_grad()
            output = model(features)
            loss = hp.criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = (train_loss / len(training_loader)) / hp.input_duplicates
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for features, labels in validation_loader:
                output = model(features)
                loss = hp.criterion(output, labels)
                val_loss += loss.item()
                val_correct += (output.argmax(dim=1) == labels).sum().item()
        val_losses.append(val_loss / len(validation_loader))
        val_total = len(validation_loader.dataset)
        val_accuracy = val_correct / val_total

        
        if val_correct >= 175:
            star = "*"
            pretest_correct, _ = test(model, testing_loader, hp)
            pretest_report = f"p: {pretest_correct:>3} / 100"
            hp.spit(f"[{trial.number:>4}]Epoch {epoch:>5} t: {train_loss:>2.7f} v: {val_loss:>.7f} c:{val_correct:>4}/{val_total:<4}{star} lr:{optimizer.param_groups[0]['lr']:.5f} {pretest_report}")
            if pretest_correct == 100:
                trial.report(1.0, epoch)
                return 1.0
            else:
                trial.report(val_accuracy, epoch)
        else:
            star = " "
            pretest_report = " "
            # Report the validation accuracy to Optuna
            trial.report(val_accuracy, epoch)
            hp.spit(f"[{trial.number:>4}]Epoch {epoch:>5} t: {train_loss:>.7f} v: {val_loss:>.7f} c:{val_correct:>4}/{val_total:<4}{star} lr:{optimizer.param_groups[0]['lr']:.5f} {pretest_report}")

        # Early stopping
        if trial.should_prune():
            hp.spit(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        if val_accuracy > best_score:
            best_score = val_accuracy
            patience = hp.max_patience
        else:
            patience -= 1
            if patience == 0:
                break

    # Final testing
    test_correct, test_loss = test(model, testing_loader, hp)
    test_accuracy = test_correct / len(testing_loader.dataset)
    return test_accuracy

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    storage_name = "sqlite:///optuna_study.db"
    
    pruner = FixedEpochsPruner(max_epochs=50)
           
    study = optuna.create_study(
        study_name=timestamp,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner
    )
    study.optimize(objective, n_jobs=5, n_trials=5)
    print("Best hyperparameters:", study.best_params)
    print("Best test accuracy:", study.best_value)

