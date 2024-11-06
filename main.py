from __future__ import annotations

import logging
import concurrent.futures
from datetime import datetime
from typing import Callable, NamedTuple, Optional, Tuple, Union
import torch
from torch import Size, nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import gc
import matplotlib.pyplot as plt
from fizz_buzz_nn import Model
from data_sample import DataSample
from hyperparameters import Hyperparameters
from loader import Loader as loader
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(hp: Hyperparameters):
    log_format = f'[{hp.parameter_set_id}] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("parallel_execution.log"),
            logging.StreamHandler()
        ]
    )


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
    crierion = hp.criterion

    with torch.no_grad():
        for features, labels in data_loader:
            predictions = model(features)
            loss = crierion(predictions, labels)
            total_loss += loss.item()
            num_correct += (predictions.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / len(data_loader.dataset) 
    return num_correct, avg_loss


def cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(hp: Hyperparameters):
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

    # create an empty model and send it to the device
    model = Model(hp).to(hp.device)
    optimizer = optim.Adam(model.parameters(), lr=hp.initial_learning_rate, weight_decay=hp.weight_decay)

    for epoch in range(hp.epochs+1):

        model, train_loss = train(model, training_loader, optimizer, hp)
        train_losses.append(train_loss)

        val_correct, val_loss = test(model, validation_loader, hp)
        val_losses.append(val_loss)
        val_total = len(validation_loader.dataset)

        learning_rate = optimizer.param_groups[0]['lr']

        # scheduler.step()

        if val_correct >= 175:
            star = "*"
            pretest_correct, _ = test(model, testing_loader, hp)
            pretest_report = f"p: {pretest_correct:>3} / 100"
            if pretest_correct == 100:
                hp.spit(f"Epoch {epoch:>5} t: {train_loss:>.7f} v: {val_loss:>.7f} c:{val_correct:>4}/{val_total:<4}{star} lr:{learning_rate} {pretest_report}")
                break
        else:
            star = " "
            pretest_report = ""
        star = "*" if val_correct >= 180 else " "
        hp.spit(f"Epoch {epoch:>5} t: {train_loss:>.7f} v: {val_loss:>.7f} c:{val_correct:>4}/{val_total:<4}{star} lr:{learning_rate} {pretest_report}")

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
        
    # Final test evaluation
    test_correct, test_loss = test(model, testing_loader, hp)
    test_losses.append(test_loss)
    hp.spit("\nFinal test evaluation")
    hp.spit(f"Epoch {epoch:>5} val: {test_correct:>3} / 100")

    if test_correct == 100:
        os.makedirs(hp.run_path, exist_ok=True)
        torch.save(obj=model.state_dict(), f=f'{hp.run_path}/model.pth')
        hp.spit("WIN!... Model saved")
    else:
        hp.spit("meh")

def run_job(hp):
    setup_logging(hp)
    main(hp)

if __name__ == '__main__':
    cleanup()

    now = datetime.now()
    run_in_parallel = True

    if run_in_parallel:
        hp_sets = [Hyperparameters(i, now) for i in range(2)]
        hp_sets[0].hidden_dim = 23
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(run_job, hp_sets)
            executor.shutdown(wait=True)
    else:
        hp = Hyperparameters(0, now)
        setup_logging(hp)
        hp.spit = logging.info
        main(hp)

    cleanup()






