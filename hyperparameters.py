import torch
import datetime


class Hyperparameters:
    parameter_set_id:              int                      = 0

# for Model
    input_dim:                     int                      = 10
    output_dim:                    int                      = 4
    hidden_dim:                    int                      = 32
    drop:                          float                    = 0.2

# for Optimizer 
    initial_learning_rate:         float                    = 0.004
    weight_decay:                  float                    = 1e-5

# criterion
    criterion:                     torch.nn.Module          = torch.nn.CrossEntropyLoss()

# for StepLR
    step_size:                     int                      = 1000
    gamma:                         float                    = 0.1   

# for ReduceLROnPlateau
    patience:                      int                      = 2000
    factor:                        float                    = 0.8
    min_lr:                        float                    = 1e-4

# operational
    epochs:                        int                      = 20000
    seed:                          int                      = 42
    patience_delay:                float                    = 0.75
    save_checkpoints:              bool                     = False
    spit:                          callable                 = print
    max_patience:                  int                      # calculated
    epochs_before_patience:        int                      # calculated
    device:                        torch.device             # calculated
    str_run_date:                  str                      # calculated
    run_path:                      str                      # calculated
    process_path:                  str                      # calculated
    checkpoint_path:               str                      # calculated
    
    def __init__(self, parameter_set_id: int, run_date: datetime, spit: callable = print):
        self.parameter_set_id = parameter_set_id
        self.str_run_date = run_date.strftime("%Y-%m-%d_%H_%M_%S")
        self.run_path = f"./results/{self.str_run_date}"
        self.process_path = f"{self.run_path}/{self.parameter_set_id}"
        self.checkpoint_path = f"{self.process_path}/checkpoints"
        self.epochs_before_patience = int(self.epochs * self.patience_delay)
        self.max_patience = self.epochs // 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spit = spit