
import torch


class Hyperparameters:

# for Model
    input_dim:                     int                      = 10
    output_dim:                    int                      = 4
    hidden_dim:                    int                      = 32

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
    max_patience:                  int                      # calculated
    epochs_before_patience:        int                      # calculated
    device:                        torch.device             # calculated
    save_checkpoints:              bool                     = False  

    def __init__(self):
        self.epochs_before_patience = int(self.epochs * self.patience_delay)
        self.max_patience = self.epochs // 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        