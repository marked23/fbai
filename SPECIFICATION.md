# Project Specification: FizzBuzz Neural Network Trainer

## 1. Overview

### Purpose and Goals
A toy PyTorch-based neural network trainer that learns to predict FizzBuzz labels from binary-encoded integers. The system trains multiple model architectures in parallel with configurable hyperparameters to achieve perfect classification on a test set.

### Success Criteria
- **Perfect**: 100/100 correct predictions on test set (numbers 1-100)
- **Pre-test threshold**: Validation accuracy >= 175/219 (80% of validation set) triggers test evaluation
- Test set evaluates: `gcd(n, 15)` for n in [1, 100]

### Key Constraints
- Input: 10-bit binary representation of integers
- Output: 4-class classification corresponding to `[1, 3, 5, 15]` (possible values of `gcd(n, 15)`)
- Training data: numbers 101-1023
- Test data: numbers 1-100
- GPU acceleration required for optimal performance (CUDA support)

## 2. Dependencies & Environment

### Python Version
Python 3.13 (inferred from `.venv/lib/python3.13/`)

### Package Dependencies
```
certifi==2024.8.30
charset-normalizer==3.4.0
contourpy==1.3.0
cycler==0.12.1
filelock==3.16.1
fonttools==4.54.1
fsspec==2024.10.0
huggingface-hub==0.26.2
idna==3.10
Jinja2==3.1.4
kiwisolver==1.4.7
MarkupSafe==3.0.2
matplotlib==3.9.2
mpmath==1.3.0
networkx==3.4.2
numpy==2.1.3
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.4.127
packaging==24.1
pandas==2.2.3
pillow==11.0.0
pyparsing==3.2.0
PyQt5==5.15.11
PyQt5-Qt5==5.15.15
PyQt5_sip==12.15.0
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0.2
regex==2024.9.11
requests==2.32.3
safetensors==0.4.5
seaborn==0.13.2
setuptools==75.3.0
six==1.16.0
sympy==1.13.1
tokenizers==0.20.2
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
tqdm==4.66.6
transformers==4.46.1
triton==3.1.0
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
optuna (implied by main_optuna.py, version not specified)
```

### System Requirements
- GPU with CUDA support (NVIDIA)
- FFmpeg with h264_nvenc codec support (for animation generation)

## 3. Project Structure

```
/home/mark/prog/fbai/
├── .venv/                      # Python virtual environment
├── .vscode/                    # VSCode configuration
├── .claude/                    # Claude Code agent memory
├── .git/                       # Git repository
├── CLAUDE.md                   # Project guidance for Claude Code
├── LICENSE                     # Project license
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
│
├── data_sample.py              # DataSample NamedTuple: encoding & labels
├── loader.py                   # Loader: creates PyTorch DataLoaders
├── fizz_buzz_nn.py             # Model architectures (6 variants)
├── hyperparameters.py          # Hyperparameters class & loader
├── perturbations.py            # PerturbRule & sweep system
├── lloging.py                  # Multi-process logging (note: typo is intentional)
│
├── main.py                     # Entry point (obsolete): ProcessPoolExecutor
├── main_parallel.py            # Entry point (preferred): torch.mp.spawn
├── main_optuna.py              # Entry point: Optuna hyperparameter search
│
├── animate.py                  # Post-training: generate weight animations
└── plot.py                     # Post-training: static weight/bias plots
```

### Output Directory Structure
```
./results/<YYYY-MM-DD_HH_MM_SS>/          # Run timestamp
├── winners.txt                            # Appended when rank achieves 100/100
└── <rank>/                                # Per-rank/process subdirectory
    ├── hyperparameters.json               # Saved hyperparameters (JSON)
    ├── model.pth                          # Saved model state_dict (on 100/100)
    └── checkpoints/                       # Optional checkpoint directory
        ├── model_000000.pth               # Epoch 0 checkpoint
        ├── model_000001.pth               # Epoch 1 checkpoint
        └── ...                            # One per epoch if save_checkpoints=True
```

Generated animations (from `animate.py`):
```
./results/<timestamp>/<rank>/
├── linear1_weight.mp4
├── linear2_weight.mp4
└── ...
```

## 4. Data Model

### DataSample NamedTuple
**File**: `data_sample.py`

```python
class DataSample(NamedTuple):
    n: int                   # Original integer
    features: torch.Tensor   # 10-bit binary encoding
    label: int               # Label index (0-3)
```

#### Static Methods

**`binary_digits(n: int, num_digits: int = 10) -> torch.Tensor`**
- Converts integer `n` to 10-bit binary representation
- Returns `torch.Tensor` of floats (0.0 or 1.0)
- Encoding: least-significant bit first
- Example: `5` → `[1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`

**`calculate_label(n: int) -> int`**
- Computes `gcd(n, 15)`
- Returns index into `[1, 3, 5, 15]`
- Label mapping:
  - `gcd(n, 15) = 1` → label `0`
  - `gcd(n, 15) = 3` → label `1`
  - `gcd(n, 15) = 5` → label `2`
  - `gcd(n, 15) = 15` → label `3`

**`create(n: int) -> DataSample`**
- Factory method combining encoding and labeling

### Loader Class
**File**: `loader.py`

Static methods for creating PyTorch DataLoaders.

**`create_training_loader(hp: Hyperparameters) -> Tuple[DataLoader, DataLoader]`**
1. Generates training data: `range(101, 1024)` → 923 samples
2. Splits 80/20 into training (738) and validation (185)
3. **Duplicates training portion** `hp.input_duplicates` times
4. Returns `(training_loader, validation_loader)`

Training data duplication logic:
```python
train_features = training_dataset[:][0].repeat(hp.input_duplicates, 1)
train_labels = training_dataset[:][1].repeat(hp.input_duplicates)
```

**`create_testing_loader(hp: Hyperparameters) -> DataLoader`**
1. Generates test data: `range(1, 101)` → 100 samples
2. Returns `testing_loader`

### Dataset Splits and Ranges

| Split      | Range        | Count Before Dup | Count After Dup         | Batch Size Param     |
|------------|--------------|------------------|-------------------------|----------------------|
| Training   | 101-1023     | 738              | 738 × input_duplicates  | train_batch_size     |
| Validation | 101-1023     | 185              | 185 (not duplicated)    | val_batch_size       |
| Test       | 1-100        | 100              | 100 (not duplicated)    | test_batch_size      |

### Data Flow
```
Integer n
    ↓
binary_digits(n) → 10-bit binary tensor (features)
    ↓
DataSample(n, features, label)
    ↓
TensorDataset(features, labels)
    ↓
random_split (80/20 for train/val)
    ↓
training data duplicated × input_duplicates
    ↓
DataLoader (batching, shuffling)
    ↓
Model training/evaluation
```

## 5. Model Architectures

**File**: `fizz_buzz_nn.py`

All models accept `Hyperparameters` object in `__init__` and use:
- `hp.input_dim` (always 10)
- `hp.output_dim` (always 4)
- `hp.hidden_dim` (configurable)
- `hp.drop` (dropout probability, default 0.2)

### Model Class (default)
```python
class Model(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(Model, self).__init__()
        self.drop = hp.drop
        hidden_x4 = hp.hidden_dim * 4
        hidden_x2 = hp.hidden_dim * 2
        self.linear1 = torch.nn.Linear(hp.input_dim, hidden_x2)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.drop)
        self.linear4 = torch.nn.Linear(hidden_x2, hp.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.dropout(x)
        x = self.linear4(x)
        return x
```
**Layers**: Input(10) → Linear(hidden_dim × 2) → ReLU → Dropout(0.2) → Linear(4)

### WideModel
```python
class WideModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(WideModel, self).__init__()
        self.layer1 = torch.nn.Linear(hp.input_dim, 64)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layer2 = torch.nn.Linear(64, hp.output_dim)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)
```
**Layers**: Input(10) → Linear(64) → ReLU → Dropout(0.2) → Linear(4)

### DeepModel
```python
class DeepModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(DeepModel, self).__init__()
        self.layer1 = torch.nn.Linear(hp.input_dim, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, 32)
        self.layer4 = torch.nn.Linear(32, hp.output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        return self.layer4(x)
```
**Layers**: Input(10) → Linear(32) → ReLU → Dropout → Linear(32) → ReLU → Dropout → Linear(32) → ReLU → Dropout → Linear(4)

### PyramidModel
```python
class PyramidModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(PyramidModel, self).__init__()
        self.layer1 = torch.nn.Linear(hp.input_dim, 48)
        self.layer2 = torch.nn.Linear(48, 24)
        self.layer3 = torch.nn.Linear(24, 12)
        self.layer4 = torch.nn.Linear(12, hp.output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        return self.layer4(x)
```
**Layers**: Input(10) → Linear(48) → ReLU → Dropout → Linear(24) → ReLU → Dropout → Linear(12) → ReLU → Dropout → Linear(4)
**Pattern**: Tapering hidden dimensions (48 → 24 → 12)

### ImprovedModel
```python
class ImprovedModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(ImprovedModel, self).__init__()
        self.layer1 = torch.nn.Linear(hp.input_dim, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.layer2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.layer3 = torch.nn.Linear(32, hp.output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)
        torch.nn.init.kaiming_normal_(self.layer1.weight)
        torch.nn.init.kaiming_normal_(self.layer2.weight)
        torch.nn.init.xavier_normal_(self.layer3.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x
```
**Layers**: Input(10) → Linear(64) → BatchNorm1d → ReLU → Dropout(0.1) → Linear(32) → BatchNorm1d → ReLU → Dropout(0.1) → Linear(4)
**Features**:
- BatchNorm1d after each hidden layer
- Lower dropout (0.1 vs 0.2)
- Kaiming initialization for ReLU layers
- Xavier initialization for output layer

### ClaudesModel (current preferred)
```python
class ClaudesModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(ClaudesModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(hp.input_dim, hp.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hp.hidden_dim, hp.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hp.hidden_dim, hp.output_dim)
        )

    def forward(self, x):
        return self.network(x)
```
**Layers**: Input(10) → Linear(hidden_dim) → ReLU → Linear(hidden_dim) → ReLU → Linear(4)
**Features**:
- No dropout
- Uses `hp.hidden_dim` directly (configurable)
- Sequential module composition

### Dynamic Model Resolution
**Files**: `main_parallel.py`, `main_optuna.py`

Models are resolved dynamically from string names:
```python
def create_model(hp):
    module_name, class_name = hp.model_class_name.split('.')
    module = globals()[module_name]
    model_class = getattr(module, class_name)
    return model_class(hp).to(hp.device)
```

**Example**: `hp.model_class_name = "fizz_buzz_nn.ClaudesModel"`
1. Split on `.` → `module_name = "fizz_buzz_nn"`, `class_name = "ClaudesModel"`
2. `module = globals()["fizz_buzz_nn"]` (imported as `import fizz_buzz_nn`)
3. `model_class = getattr(fizz_buzz_nn, "ClaudesModel")`
4. Instantiate and move to device

## 6. Hyperparameter System

**File**: `hyperparameters.py`

### Hyperparameters Class

```python
class Hyperparameters:
    def __init__(self, **kwargs):
        self.parameter_set_id = kwargs.get('parameter_set_id', 0)
        self.input_dim = 10
        self.output_dim = 4
        self.hidden_dim = kwargs.get('hidden_dim', 32)
        self.drop = kwargs.get('drop', 0.2)
        self.initial_learning_rate = kwargs.get('initial_learning_rate', 0.0063)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_class_name = kwargs.get('model_class_name', "fizz_buzz_nn.ClaudesModel")
        self.epochs = kwargs.get('epochs', 20000)
        self.seed = 42
        self.max_patience = self.epochs // 5
        self.train_batch_size = 256
        self.val_batch_size = 256
        self.test_batch_size = 256
        self.input_duplicates = kwargs.get('input_duplicates', 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.str_run_date = (kwargs.get('run_date') or datetime.now()).strftime("%Y-%m-%d_%H_%M_%S")
        self.run_path = f"./results/{self.str_run_date}"
        self.process_path = f"{self.run_path}/{self.parameter_set_id}"
        self.checkpoint_path = f"{self.process_path}/checkpoints"
        self.epochs_before_patience = int(self.epochs * kwargs.get('patience_delay', 0.75))
        self.spit = kwargs.get('spit', print)
```

#### Fields Reference

| Field                     | Type                    | Default                      | Configurable | Description                                       |
|---------------------------|-------------------------|------------------------------|--------------|---------------------------------------------------|
| `parameter_set_id`        | int                     | 0                            | Yes          | Rank/process identifier                           |
| `input_dim`               | int                     | 10                           | No           | Input feature dimensions (10-bit binary)          |
| `output_dim`              | int                     | 4                            | No           | Output classes (labels 0-3)                       |
| `hidden_dim`              | int                     | 32                           | Yes          | Hidden layer dimension multiplier                 |
| `drop`                    | float                   | 0.2                          | Yes          | Dropout probability                               |
| `initial_learning_rate`   | float                   | 0.0063                       | Yes          | Adam optimizer learning rate                      |
| `weight_decay`            | float                   | 1e-5                         | Yes          | Adam optimizer weight decay (L2 reg)              |
| `criterion`               | torch.nn.Module         | CrossEntropyLoss()           | No           | Loss function                                     |
| `model_class_name`        | str                     | "fizz_buzz_nn.ClaudesModel"  | Yes          | Fully qualified model class name                  |
| `epochs`                  | int                     | 20000                        | Yes          | Maximum training epochs                           |
| `seed`                    | int                     | 42                           | No           | Random seed for reproducibility                   |
| `max_patience`            | int                     | epochs // 5                  | Derived      | Early stopping patience (reset on improvement)    |
| `train_batch_size`        | int                     | 256                          | No           | Training DataLoader batch size                    |
| `val_batch_size`          | int                     | 256                          | No           | Validation DataLoader batch size                  |
| `test_batch_size`         | int                     | 256                          | No           | Test DataLoader batch size                        |
| `input_duplicates`        | int                     | 1                            | Yes          | Training data duplication factor                  |
| `device`                  | torch.device            | 'cuda' if available else 'cpu' | Derived    | Computation device                                |
| `run_date`                | datetime (kwarg)        | datetime.now()               | Yes          | Timestamp for run (shared across ranks)           |
| `str_run_date`            | str                     | YYYY-MM-DD_HH_MM_SS          | Derived      | String-formatted run date                         |
| `run_path`                | str                     | ./results/{str_run_date}     | Derived      | Run output directory                              |
| `process_path`            | str                     | {run_path}/{parameter_set_id}| Derived      | Process-specific output directory                 |
| `checkpoint_path`         | str                     | {process_path}/checkpoints   | Derived      | Checkpoint directory                              |
| `epochs_before_patience`  | int                     | epochs × 0.75                | Derived      | Epochs before early stopping activates            |
| `patience_delay`          | float (kwarg)           | 0.75                         | Yes          | Fraction of epochs before patience starts         |
| `spit`                    | Callable                | print                        | Yes          | Logging function (replaced by logging.info)       |
| `save_checkpoints`        | bool (not in __init__)  | False (not set)              | External     | Whether to save per-epoch checkpoints             |
| `perturb_info`            | str (added dynamically) | None                         | Set by system| Perturbation description string                   |

#### Serialization: `__str__()` Method
Returns JSON string of all instance and class variables:
- Handles torch.nn.Module → class name
- Handles torch.device → string representation
- Handles callables → `<function name>`
- Skips non-serializable with placeholder

#### Known Inconsistency
- `Hyperparameters.__init__` accepts `**kwargs`
- But `main.py` and `main_parallel.py` call it positionally in some places: `Hyperparameters(i, now)`
  - This creates kwargs: `{'parameter_set_id': i, 'run_date': now}`
  - Works because first two positional args are not defined in signature

### HyperparametersLoader Class

```python
class HyperparametersLoader:
    criterion_map = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        'MSELoss': torch.nn.MSELoss,
        'BCELoss': torch.nn.BCELoss
    }

    def from_json(cls, json_path: str) -> 'Hyperparameters':
        # Loads JSON, reconstructs Hyperparameters
        # Handles criterion mapping, timestamp parsing
        # Skips 'spit' function
```

**Used by**: `animate.py` to reconstruct hyperparameters from saved JSON

## 7. Perturbation System

**File**: `perturbations.py`

### PerturbRule Dataclass

```python
@dataclass
class PerturbRule:
    param_name: str                          # Hyperparameter to perturb
    start: Union[int, float, None] = None    # Starting value
    step: Union[int, float] = 1              # Step size or multiplier
    multiply: bool = False                   # False=linear, True=geometric
    array: Optional[List[str]] = None        # Array of values to cycle
    each: int = 1                            # Repetitions per array element
```

### Sweep Types

1. **Linear Sweep** (default: `multiply=False`)
   - Formula: `value = start + (step × i)`
   - Example: `PerturbRule("hidden_dim", start=14, step=1)` → 14, 15, 16, 17, ...

2. **Geometric Sweep** (`multiply=True`)
   - Formula: `value = start × (step ^ i)`
   - Example: `PerturbRule("initial_learning_rate", start=0.001, step=2, multiply=True)` → 0.001, 0.002, 0.004, 0.008, ...

3. **Array Cycling** (`array=...`)
   - Formula: `index = (i // each) % len(array); value = array[index]`
   - Example: `PerturbRule("model_class_name", array=["fizz_buzz_nn.Model", "fizz_buzz_nn.WideModel"], each=2)`
     - Ranks 0-1: "fizz_buzz_nn.Model"
     - Ranks 2-3: "fizz_buzz_nn.WideModel"
     - Ranks 4-5: "fizz_buzz_nn.Model" (cycle repeats)

### apply_perturbations Function

```python
def apply_perturbations(hp_sets: List[Hyperparameters], rules: List[PerturbRule]):
    # For each rank i and each rule:
    # 1. Compute value based on sweep type
    # 2. setattr(hp, rule.param_name, value)
    # 3. Store description in hp.perturb_info
    # 4. Print perturbation summary
```

**Usage Pattern** (from `main_parallel.py`):
```python
rules = [
    PerturbRule("initial_learning_rate", start=0.0097, step=0.0),
    PerturbRule("input_duplicates", start=63, step=0),
    PerturbRule("hidden_dim", start=430, step=0),
]
apply_perturbations(hp_sets, rules)
```

## 8. Training System

### Training Loop Structure

All three entry points share a similar training loop pattern:

#### Common Functions

**`train(model, training_loader, optimizer, hp) -> Tuple[nn.Module, float]`**
1. Set `model.train()`
2. For each batch in `training_loader`:
   - Zero gradients
   - Forward pass
   - Compute loss with `hp.criterion`
   - Backward pass
   - Optimizer step
3. Return `(model, avg_loss)`

**`test(model, data_loader, hp) -> Tuple[int, float]`**
1. Set `model.eval()`
2. Disable gradients (`torch.no_grad()`)
3. For each batch in `data_loader`:
   - Forward pass
   - Compute loss
   - Count correct predictions (`predictions.argmax(dim=1) == labels`)
4. Return `(num_correct, avg_loss)`

**Note**: `main.py` has a typo in test function: `crierion` instead of `criterion` (line 50)

**`set_seed(seed)`**
```python
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Training Epoch Loop (main_parallel.py pattern)

```python
optimizer = optim.Adam(model.parameters(), lr=hp.initial_learning_rate, weight_decay=hp.weight_decay)

for epoch in range(hp.epochs+1):
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

    # Early test if validation threshold met
    if val_correct >= 175:  # 175/219 validation samples
        pretest_correct, _ = test(model, testing_loader, hp)
        if pretest_correct == 100:
            # Save model and exit
            break

    # Checkpoint saving
    if hp.save_checkpoints:
        torch.save(obj=model.state_dict(), f='{hp.checkpoint_path}/model_{epoch:06}.pth')

    # Early stopping patience
    if epoch >= hp.epochs_before_patience:
        if val_correct >= best_score:
            best_score = val_correct
            patience = hp.max_patience
        else:
            patience -= 1
        if patience == 0:
            break
```

### Loss Function
- **Type**: `torch.nn.CrossEntropyLoss()` (stored in `hp.criterion`)
- Applied to raw logits (no softmax in model output)

### Optimizer
- **Type**: `torch.optim.Adam`
- **Parameters**:
  - `lr`: `hp.initial_learning_rate` (default 0.0063)
  - `weight_decay`: `hp.weight_decay` (default 1e-5)
- **No scheduler** (commented out in code)

### Validation and Test Evaluation

**Validation**: Every epoch
- Evaluates on 185 samples (20% of 101-1023 range)
- Threshold: `val_correct >= 175` triggers pre-test

**Pre-test**: Conditional (when val_correct >= 175)
- Evaluates on 100 test samples (1-100)
- If `pretest_correct == 100`: save model and break

**Final test**: After training loop completes
- Always runs at end
- Reports final accuracy

### Model Saving Logic

**Final Model** (`model.pth`):
- Condition: `test_correct == 100` (perfect accuracy)
- Path: `{hp.process_path}/model.pth`
- Saved with: `torch.save(obj=model.state_dict(), f=...)`
- Triggers: "WIN!... Model saved" log message

**Checkpoints** (`model_NNNNNN.pth`):
- Condition: `hp.save_checkpoints == True` (not set by default in Hyperparameters class)
- Path: `{hp.checkpoint_path}/model_{epoch:06}.pth`
- Saved: Every epoch
- Note: `hp.save_checkpoints` is not defined in `Hyperparameters.__init__()`, must be set externally or added to class

### Parallelism Strategies

#### main_parallel.py (Preferred)
**Method**: `torch.multiprocessing.spawn()`

```python
if __name__ == '__main__':
    world_size = 1  # Number of parallel instances
    now = datetime.now()
    hp_sets = [Hyperparameters(i, now) for i in range(world_size+1)]

    # Define perturbation rules
    rules = [...]
    apply_perturbations(hp_sets, rules)

    # Save hyperparameters before spawning
    for hp in hp_sets:
        save_hyperparameters(hp)

    # Spawn worker processes
    mp.spawn(main, args=(hp_sets,), nprocs=world_size, join=True)
```

**Characteristics**:
- Uses `torch.multiprocessing` (CUDA-aware)
- Passes `hp_sets` list to all workers
- Each worker selects `hp_sets[rank]`
- Shared timestamp across all ranks

#### main_optuna.py
**Method**: Optuna study with `n_jobs` parameter

```python
study = optuna.create_study(
    study_name=timestamp,
    direction="maximize",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    pruner=FixedEpochsPruner(max_epochs=50)
)
study.optimize(objective, n_jobs=5, n_trials=5)
```

**Characteristics**:
- Parallel trials via `n_jobs=5`
- SQLite storage backend for trial sharing
- Custom pruner: `FixedEpochsPruner` (stops trials after max_epochs)
- Trial reports validation accuracy each epoch

#### main.py (Obsolete)
**Method**: `concurrent.futures.ProcessPoolExecutor`

```python
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = [executor.submit(run_job, hp, queue) for hp in hp_sets]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")
```

**Characteristics**:
- Uses Python standard library multiprocessing
- Hardcoded to `Model` class (no dynamic resolution)
- Multiprocessing queue for logging

## 9. Entry Points

### main_parallel.py (Preferred)
**Command**: `python main_parallel.py`

**Key Features**:
- Uses `torch.multiprocessing.spawn()` for parallel ranks
- Dynamic model class resolution from `hp.model_class_name`
- Perturbation rules configured in `__main__` block
- Winners appended to `{run_path}/winners.txt`

**Workflow**:
1. Create `hp_sets` list with shared timestamp
2. Define `PerturbRule` list in code
3. Apply perturbations to `hp_sets`
4. Save hyperparameters to JSON
5. Spawn worker processes with `mp.spawn(main, args=(hp_sets,), nprocs=world_size)`
6. Each worker trains independently
7. On 100/100 test accuracy, append to `winners.txt`

**Configuration**: Edit source code to change:
- `world_size`: number of parallel processes
- `rules`: list of perturbation rules
- `models`: array of model class names

### main_optuna.py
**Command**: `python main_optuna.py`

**Key Features**:
- Optuna hyperparameter search
- SQLite storage: `optuna_study.db`
- Custom `FixedEpochsPruner` (stops trials after 50 epochs)
- Parallel trials via `n_jobs=5`

**objective() Function**:
```python
def objective(trial: Trial):
    suggested_params = {
        'initial_learning_rate': trial.suggest_float('initial_learning_rate', 0.0063, 0.0063, log=True),
        'hidden_dim': trial.suggest_int('hidden_dim', 430, 430),
        'input_duplicates': trial.suggest_int('input_duplicates', 63, 63),
        'model_class_name': trial.suggest_categorical('model_class_name', ["fizz_buzz_nn.ClaudesModel"])
    }
    hp = Hyperparameters(**suggested_params)
    hp.parameter_set_id = trial.number
    # ... train and return test_accuracy
```

**Study Configuration**:
```python
study = optuna.create_study(
    study_name=timestamp,
    direction="maximize",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
    pruner=FixedEpochsPruner(max_epochs=50)
)
study.optimize(objective, n_jobs=5, n_trials=5)
```

**Output**:
- Best hyperparameters: `study.best_params`
- Best test accuracy: `study.best_value`

### main.py (Obsolete)
**Command**: `python main.py`

**Key Features**:
- Uses `concurrent.futures.ProcessPoolExecutor`
- Hardcoded `Model` class (no dynamic resolution)
- Manual hyperparameter loop: `hp.hidden_dim = i + 30`

**Known Issues**:
- Typo in test function: `crierion` instead of `criterion` (line 50)
- Less flexible than `main_parallel.py`

**Workflow**:
1. Create `hp_sets` with `[Hyperparameters(i, now) for i in range(5)]`
2. Manually set `hp.hidden_dim = i + 30`
3. Submit jobs to `ProcessPoolExecutor`
4. Multiprocessing queue logging via `listener_process`

## 10. Visualization & Post-Processing

### animate.py

**Purpose**: Generate MP4 animations of weight matrix evolution across training epochs

**Command**:
```bash
python animate.py --all <process_path>           # all layers
python animate.py --layer <N> <process_path>     # specific layer N
python animate.py --step 5 --all <process_path>  # skip 5 frames between
```

**Arguments**:
- `--all`: Generate animations for all weight layers
- `--layer N`: Generate animation for `linearN.weight` only
- `--step N`: Skip N-1 checkpoints between frames (default: 1)
- `process_path`: Path to process directory (e.g., `./results/2024-03-14_16_16_28/0`)

**Requirements**:
- Checkpoints directory must exist: `{process_path}/checkpoints/`
- Hyperparameters JSON must exist: `{process_path}/hyperparameters.json`
- FFmpeg with h264_nvenc codec

**Workflow**:
1. Load hyperparameters from JSON using `HyperparametersLoader`
2. Instantiate model (currently hardcoded to `Model` class, not dynamic)
3. Load all checkpoint files from `{process_path}/checkpoints/`
4. For each weight layer:
   - Create `FuncAnimation` with seaborn heatmap
   - Annotate significant changes (|delta| > 0.05) with black dots
   - Highlight zero rows with yellow lines
   - Save as MP4 with h264_nvenc codec

**Output**:
- Files: `{process_path}/{layer_name}.mp4` (e.g., `linear1_weight.mp4`)
- Codec: `h264_nvenc` (NVIDIA GPU encoding)
- Frame rate: 1 fps
- Colormap: `RdBu_r` (Red-Blue diverging, centered at 0)
- Range: `vmin=-1, vmax=1`

**Known Limitation**:
- Hardcoded to `Model` class (line 158): `model = Model(hp).to(device)`
- Should use dynamic resolution like `main_parallel.py`

### plot.py

**Purpose**: Static visualization of weights and biases from a saved model

**Command**: `python plot.py` (requires editing source for model path)

**Configuration** (hardcoded in file):
```python
model = Model(input_dim=10, output_dim=4)
model.load_state_dict(torch.load('./2024-11-05_13_46_40/model.pth', ...))
```

**Features**:
- Uses Qt5Agg backend (interactive display)
- Weight layers: seaborn heatmap with 'viridis' colormap
- Bias layers: heatmap with 'coolwarm' colormap, annotated values

**Output**: Interactive matplotlib windows (not saved to files)

## 11. Logging System

**File**: `lloging.py` (note: intentional typo in filename)

### Architecture
Multi-process logging using `QueueHandler` and `QueueListener` pattern.

### Components

**`setup_logging(hp)`** - Single process logging
```python
def setup_logging(hp: Hyperparameters):
    log_format = f'[{hp.parameter_set_id:>2}] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("parallel_execution.log"),
            logging.StreamHandler()
        ]
    )
```
- Used by: `main.py` in single-process mode

**`setup_logger(hp, queue)`** - Multi-process worker logger
```python
def setup_logger(hp: Hyperparameters, queue):
    log_format = f'[{hp.parameter_set_id:>2}] %(message)s'
    queue_handler = logging.handlers.QueueHandler(queue)
    # ... configure and add to root logger
    hp.spit = logging.info  # Replace print with logging.info
```
- Used by: `main.py` worker processes
- Replaces `hp.spit` function with `logging.info`

**`listener_process(queue, run_path)`** - Multi-process log listener
```python
def listener_process(queue, run_path: str):
    listener_configurer(run_path)
    listener = logging.handlers.QueueListener(queue, *logging.getLogger().handlers)
    listener.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
```
- Runs in separate process
- Receives log records from queue
- Writes to file and stdout

### Log Format
- Per-rank prefix: `[rank_id] message`
- Example: `[  0] Epoch   123 t: 0.0234567 v: 0.0345678 c: 180/ 185  lr:0.00630`

### Log Destinations
- File: `{run_path}/parallel_execution.log` (multi-process) or `parallel_execution.log` (single-process)
- Stdout: Console output via `StreamHandler`

### Usage Pattern (main.py)
```python
manager = multiprocessing.Manager()
queue = manager.Queue()

listener = multiprocessing.Process(target=listener_process, args=(queue, run_path))
listener.start()

# Workers call setup_logger(hp, queue)
# Workers log via hp.spit(message) which is logging.info

listener.terminate()
```

## 12. Hyperparameter Search (Optuna)

**File**: `main_optuna.py`

### Study Configuration

```python
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
```

**Parameters**:
- `study_name`: Timestamp string (unique per run)
- `direction`: "maximize" (maximize test accuracy)
- `storage`: SQLite database for persistence
- `load_if_exists`: True (resume interrupted studies)
- `pruner`: Custom `FixedEpochsPruner`

### Storage Backend
- **Type**: SQLite
- **File**: `optuna_study.db` (in project root)
- **Purpose**: Persist trials across runs, enable parallel workers

### Objective Function

```python
def objective(trial: Trial):
    suggested_params = {
        'initial_learning_rate': trial.suggest_float('initial_learning_rate', 0.0063, 0.0063, log=True),
        'hidden_dim': trial.suggest_int('hidden_dim', 430, 430),
        'input_duplicates': trial.suggest_int('input_duplicates', 63, 63),
        'model_class_name': trial.suggest_categorical('model_class_name', ["fizz_buzz_nn.ClaudesModel"])
    }
    hp = Hyperparameters(**suggested_params)
    hp.parameter_set_id = trial.number

    # ... train model
    # ... report val_accuracy each epoch: trial.report(val_accuracy, epoch)
    # ... return test_accuracy (0.0-1.0)
```

**Current Configuration**: Fixed parameters (no search space)
- `initial_learning_rate`: 0.0063 to 0.0063
- `hidden_dim`: 430 to 430
- `input_duplicates`: 63 to 63
- `model_class_name`: only "fizz_buzz_nn.ClaudesModel"

**To Enable Search**: Expand ranges, e.g.:
```python
'hidden_dim': trial.suggest_int('hidden_dim', 100, 1000)
```

### Custom Pruner: FixedEpochsPruner

```python
class FixedEpochsPruner(optuna.pruners.BasePruner):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def prune(self, study, trial) -> bool:
        step = trial.last_step
        if step is None:
            return False
        return step >= self.max_epochs
```

**Purpose**: Stop trials that exceed `max_epochs` (currently 50)

### Trial Reporting
```python
trial.report(val_accuracy, epoch)
if trial.should_prune():
    raise optuna.exceptions.TrialPruned()
```

### Optimization Call
```python
study.optimize(objective, n_jobs=5, n_trials=5)
```
- `n_jobs=5`: Run 5 parallel trials
- `n_trials=5`: Total 5 trials

### Results
```python
print("Best hyperparameters:", study.best_params)
print("Best test accuracy:", study.best_value)
```

## 13. Known Configurations & Behaviors

### Known Good Hyperparameters
From `CLAUDE.md`:
```python
hidden_dim = 430
input_duplicates = 63
initial_learning_rate = 0.0097
```
**Expected**: Can achieve 100/100 in ~6 epochs

### Current Configuration (main_parallel.py)
```python
models = ["fizz_buzz_nn.ClaudesModel"]

rules = [
    PerturbRule("initial_learning_rate", start=0.0097, step=0.0),
    PerturbRule("input_duplicates", start=63, step=0),
    PerturbRule("hidden_dim", start=430, step=0),
]
```
**Behavior**: All ranks use identical hyperparameters (step=0 means no perturbation)

### Known Inconsistencies

1. **lloging.py filename**: Intentional typo (should be "logging.py" but kept as "lloging.py")

2. **Hyperparameters initialization**:
   - `__init__` signature: `def __init__(self, **kwargs)`
   - Called as: `Hyperparameters(i, now)` (positional args)
   - Works because Python treats excess positional args as kwargs with inferred names

3. **main.py typo**:
   - Line 50: `crierion = hp.criterion` (should be `criterion`)
   - Still works due to local variable name

4. **animate.py hardcoded model**:
   - Line 158: `model = Model(hp).to(device)`
   - Should use dynamic resolution like `create_model(hp)` in main_parallel.py
   - Breaks if saved model used a different architecture

5. **save_checkpoints attribute**:
   - Checked in training loop: `if hp.save_checkpoints:`
   - Not defined in `Hyperparameters.__init__()`
   - Must be set externally or added to class

6. **Validation threshold hardcoded**:
   - `if val_correct >= 175:` appears in all entry points
   - Derived from 80% of 219 validation samples (80/20 split of 923 training samples)
   - Not configurable via hyperparameters

### Performance Expectations

**Training Speed**:
- With known good hyperparameters: ~6 epochs to 100/100
- Without: can take thousands of epochs or fail to converge

**Data Duplication Effect**:
- `input_duplicates=63` means training set is 63× larger
- Each epoch sees 738 × 63 = 46,494 samples
- Increases regularization effect (model sees same data multiple times per epoch)

**Early Stopping**:
- Patience activates after 75% of max epochs (default `patience_delay=0.75`)
- Patience counter: `max_patience = epochs // 5` (default 4000 for 20000 epochs)
- Resets on validation improvement

## 14. Regeneration Notes

### Critical Implementation Details

1. **Binary Encoding Order**: Least-significant bit first
   ```python
   digits.append(float(n % 2))
   n = n // 2
   ```
   Not reverse order. Example: 5 → [1.0, 0.0, 1.0, 0.0, ...]

2. **Label Index Mapping**: `[1, 3, 5, 15].index(gcd(n, 15))`
   - Order matters: index 0 → gcd=1, index 1 → gcd=3, etc.
   - Not sorted: [1, 3, 5, 15] (not [1, 5, 3, 15])

3. **Data Split Before Duplication**:
   ```python
   training_size = int(0.8 * len(training_dataset))
   validation_size = len(training_dataset) - training_size
   training_dataset, validation_dataset = random_split(training_dataset, [training_size, validation_size])

   # Duplicate ONLY training portion
   train_features = training_dataset[:][0].repeat(hp.input_duplicates, 1)
   ```
   Duplication happens AFTER split, not before.

4. **Train Loss Normalization**:
   ```python
   train_loss = (train_loss / len(training_loader)) / hp.input_duplicates
   ```
   Divided by both loader length AND `input_duplicates`.

5. **Validation Threshold**: `val_correct >= 175`
   - Hardcoded value based on 80% of ~219 validation samples
   - Triggers pre-test evaluation

6. **Checkpoint Naming**: Zero-padded 6 digits
   ```python
   f'{hp.checkpoint_path}/model_{epoch:06}.pth'
   ```
   Example: `model_000000.pth`, `model_000001.pth`

7. **Dynamic Model Resolution**: Split on first `.` only
   ```python
   module_name, class_name = hp.model_class_name.split('.')
   ```
   Assumes exactly one dot: "module.ClassName"

8. **Perturbation Info String**: Set by `apply_perturbations()`
   ```python
   hp.perturb_info = ", ".join(rule_info)
   ```
   Not part of Hyperparameters init, added dynamically.

9. **Random Seed Consistency**:
   ```python
   torch.manual_seed(42)
   seed = 42
   ```
   Hardcoded to 42 everywhere. Critical for reproducibility.

10. **Device Selection**: Automatic
    ```python
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ```
    No override mechanism provided.

### Implicit Assumptions

1. **File System Structure**: All entry points assume writable `./results/` directory exists or can be created.

2. **Checkpoint Loading Order**: `sorted()` ensures chronological order
   ```python
   checkpoint_files = sorted([os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder) if f.endswith(".pth")])
   ```

3. **No Model Architecture Validation**: animate.py doesn't verify that loaded hyperparameters match saved checkpoint architecture.

4. **Single Criterion**: All code assumes `CrossEntropyLoss`. Changing criterion requires code changes, not just hyperparameter changes.

5. **Batch Sizes**: Hardcoded to 256 for train/val/test. Not exposed as configurable in typical usage.

6. **No Gradient Clipping**: Not implemented in training loop.

7. **No Learning Rate Scheduling**: Scheduler code is commented out.

### Order-Dependent Operations

1. **Perturbation Application**: Must happen BEFORE model initialization
   ```python
   apply_perturbations(hp_sets, rules)
   for hp in hp_sets:
       save_hyperparameters(hp)
   mp.spawn(main, args=(hp_sets,), nprocs=world_size, join=True)
   ```

2. **Directory Creation**: Must create `run_path` before `process_path`
   ```python
   os.makedirs(hp.run_path, exist_ok=True)
   os.makedirs(hp.process_path, exist_ok=True)
   ```

3. **Logger Setup**: Must call `setup_logger(hp, queue)` before `hp.spit()` is used.

4. **Listener Process**: Must start before workers start logging
   ```python
   listener.start()
   # Then submit jobs
   ```

### Regeneration Checklist

To regenerate this project from scratch:

- [ ] Install exact package versions from requirements.txt
- [ ] Preserve lloging.py filename typo
- [ ] Use 10-bit binary encoding with LSB-first order
- [ ] Label mapping: [1, 3, 5, 15].index(gcd(n, 15))
- [ ] Training data: 101-1023, split 80/20, duplicate training only
- [ ] Test data: 1-100, no duplication
- [ ] Validation threshold: val_correct >= 175
- [ ] Model saving: only on test_correct == 100
- [ ] Checkpoint naming: model_{epoch:06}.pth
- [ ] Random seed: 42 (hardcoded)
- [ ] Dynamic model resolution: module_name.class_name pattern
- [ ] Perturbation system: linear, geometric, array cycling
- [ ] Train loss normalization: divide by input_duplicates
- [ ] Early stopping: patience = epochs // 5, activates after 75% epochs
- [ ] FFmpeg h264_nvenc codec for animations
- [ ] QueueHandler/QueueListener logging pattern
- [ ] Optuna SQLite storage backend
- [ ] Hyperparameters kwargs-based initialization
- [ ] All six model architectures with exact layer dimensions
- [ ] No learning rate scheduler (commented out)
- [ ] No gradient clipping
- [ ] CrossEntropyLoss criterion only

---

## Appendices

### A. Complete File Listing

**Core Data & Model Files**:
- `data_sample.py` (28 lines) - DataSample NamedTuple
- `loader.py` (41 lines) - Loader static methods
- `fizz_buzz_nn.py` (113 lines) - 6 model architectures
- `hyperparameters.py` (107 lines) - Hyperparameters & loader

**Training Infrastructure**:
- `perturbations.py` (65 lines) - PerturbRule & apply_perturbations
- `lloging.py` (55 lines) - Multi-process logging

**Entry Points**:
- `main_parallel.py` (227 lines) - Preferred entry point
- `main_optuna.py` (222 lines) - Optuna search
- `main.py` (251 lines) - Obsolete entry point

**Visualization**:
- `animate.py` (189 lines) - Weight animation generator
- `plot.py` (35 lines) - Static plotting

**Documentation**:
- `CLAUDE.md` - Project guidance
- `README.md` - Project readme
- `LICENSE` - License file
- `requirements.txt` - Dependencies

### B. Import Dependency Graph

```
main_parallel.py
├── fizz_buzz_nn (module)
│   ├── Model
│   ├── WideModel
│   ├── DeepModel
│   ├── PyramidModel
│   ├── ImprovedModel
│   └── ClaudesModel
├── data_sample.DataSample
├── hyperparameters.Hyperparameters
├── loader.Loader
├── lloging.setup_logging, setup_logger, listener_process
└── perturbations.apply_perturbations, PerturbRule

main_optuna.py
├── (same as main_parallel.py)
└── optuna (external)

main.py
├── fizz_buzz_nn.Model (only)
├── data_sample.DataSample
├── hyperparameters.Hyperparameters
├── loader.Loader
└── lloging.setup_logging, setup_logger, listener_process

animate.py
├── fizz_buzz_nn.Model (hardcoded)
├── hyperparameters.Hyperparameters, HyperparametersLoader
└── matplotlib, seaborn, FFMpegWriter

plot.py
├── fizz_buzz_nn.Model
└── matplotlib, seaborn

data_sample.py
├── torch
└── math (gcd)

loader.py
├── torch
├── torch.utils.data.DataLoader, TensorDataset, random_split
└── data_sample.DataSample

hyperparameters.py
├── torch
├── datetime
└── json

perturbations.py
├── dataclasses.dataclass
└── hyperparameters.Hyperparameters

lloging.py
├── logging
└── hyperparameters.Hyperparameters
```

### C. Hyperparameter Search Space (Optuna Example)

To enable actual hyperparameter search in `main_optuna.py`, modify the `objective()` function:

```python
suggested_params = {
    'initial_learning_rate': trial.suggest_float('initial_learning_rate', 1e-4, 1e-2, log=True),
    'hidden_dim': trial.suggest_int('hidden_dim', 100, 1000, step=10),
    'input_duplicates': trial.suggest_int('input_duplicates', 1, 100),
    'drop': trial.suggest_float('drop', 0.0, 0.5),
    'model_class_name': trial.suggest_categorical('model_class_name', [
        "fizz_buzz_nn.Model",
        "fizz_buzz_nn.WideModel",
        "fizz_buzz_nn.DeepModel",
        "fizz_buzz_nn.PyramidModel",
        "fizz_buzz_nn.ImprovedModel",
        "fizz_buzz_nn.ClaudesModel"
    ])
}
```

This would search over:
- Learning rate: 0.0001 to 0.01 (log scale)
- Hidden dim: 100 to 1000 (step 10)
- Input duplicates: 1 to 100
- Dropout: 0.0 to 0.5
- Model architecture: all 6 variants

---

**Document Version**: 1.0
**Generated**: 2026-02-11
**Project Path**: `/home/mark/prog/fbai/`
**Git Branch**: `for-claude`
**Last Commit**: `b101f81 Add CLAUDE.md with project guidance for Claude Code`
