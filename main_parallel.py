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

# Local imports
import fizz_buzz_nn
from fizz_buzz_nn import Model, WideModel, DeepModel, PyramidModel, ImprovedModel
from data_sample import DataSample
from hyperparameters import Hyperparameters
from loader import Loader as loader
from lloging import setup_logging, setup_logger, listener_process
from perturbations import apply_perturbations, PerturbRule

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

def main(rank, hp_sets: list[Hyperparameters]):
    hp = hp_sets[rank]
    setup(rank)
    set_seed(hp.seed)
    training_loader, validation_loader = loader.create_training_loader(hp)
    testing_loader  = loader.create_testing_loader(hp)

    train_losses = []
    val_losses = []
    test_losses = []

    best_score = 0.0
    patience = hp.max_patience

    if hp.save_checkpoints:
        os.makedirs(hp.checkpoint_path, exist_ok=True)

    model = create_model(hp)
    # model = DeepModel(hp).to(hp.device)
    optimizer = optim.Adam(model.parameters(), lr=hp.initial_learning_rate, weight_decay=hp.weight_decay)

    for epoch in range(hp.epochs+1):
        model, train_loss = train(model, training_loader, optimizer, hp)
        train_losses.append(train_loss)

        val_correct, val_loss = test(model, validation_loader, hp)
        val_losses.append(val_loss)
        val_total = len(validation_loader.dataset)

        learning_rate = optimizer.param_groups[0]['lr']

        if val_correct >= 175:
            star = "*"
            pretest_correct, _ = test(model, testing_loader, hp)
            pretest_report = f"p: {pretest_correct:>3} / 100"
            if pretest_correct == 100:
                hp.spit(f"[{rank:>4}]Epoch {epoch:>5} t: {train_loss:>.7f} v: {val_loss:>.7f} c:{val_correct:>4}/{val_total:<4}{star} lr:{learning_rate:.5f} {pretest_report}")
                break
        else:
            star = " "
            pretest_report = ""

        hp.spit(f"[{rank:>4}]Epoch {epoch:>5} t: {train_loss:>.7f} v: {val_loss:>.7f} c:{val_correct:>4}/{val_total:<4}{star} lr:{learning_rate:.5f} {pretest_report}")

        if hp.save_checkpoints:
            torch.save(obj=model.state_dict(), f=f'{hp.checkpoint_path}/model_{epoch:06}.pth')

        if epoch >= hp.epochs_before_patience:
            if val_correct >= best_score:
                best_score = val_correct
                hint = "*"
                patience = hp.max_patience
            else:
                patience -= 1
                hint = str(patience)
        else:
            hint = ""

        if patience == 0:
            hp.spit("Patience has run out")
            break

    test_correct, test_loss = test(model, testing_loader, hp)
    test_losses.append(test_loss)
    hp.spit("\nFinal test evaluation")
    hp.spit(f"Epoch {epoch:>5} val: {test_correct:>3} / 100")

    if test_correct == 100:
        os.makedirs(hp.run_path, exist_ok=True)
        torch.save(obj=model.state_dict(), f=f'{hp.process_path}/model.pth')
        hp.spit("WIN!... Model saved")
    else:
        hp.spit("meh")

    cleanup()

if __name__ == '__main__':
    world_size = 20  # Number of parallel instances
    now = datetime.now()
    hp_sets = [Hyperparameters(i, now) for i in range(world_size+1)]

    models = ["fizz_buzz_nn.ImprovedModel", 
              "fizz_buzz_nn.DeepModel", 
              "fizz_buzz_nn.WideModel", 
              "fizz_buzz_nn.PyramidModel"]    
    
    # Define perturbation rules
    rules = [
        # PerturbRule("hidden_dim", 16, 1),                        # Linear: 14,15,16...
        PerturbRule("initial_learning_rate", 0.001, 0.0003),  # Geometric: 0.001,0.002,0.004...
        # PerturbRule("drop", 0.01, 0.02)                         # Linear: 0.1,0.15,0.2...
        PerturbRule("model_class_name", array=models, each=1),
        PerturbRule("input_duplicates", start=1, step=1)
    ]
    
    apply_perturbations(hp_sets, rules)
    for hp in hp_sets:
        save_hyperparameters(hp)
        
    time.sleep(5)
    mp.spawn(main, args=(hp_sets,), nprocs=world_size, join=True)

