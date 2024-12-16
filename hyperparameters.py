import torch
from torch import nn
import datetime
import json
from datetime import datetime
from typing import Dict, Any


class HyperparameterError(Exception):
    """Custom exception for hyperparameter loading errors"""
    pass

class Hyperparameters:
    parameter_set_id:              int                      = 0

    criterion_map = {
        'CrossEntropyLoss': nn.CrossEntropyLoss,
        'MSELoss': nn.MSELoss,
        'BCELoss': nn.BCELoss
    }
    
# for Model
    input_dim:                     int                      = 10
    output_dim:                    int                      = 4
    hidden_dim:                    int                      = 32
    drop:                          float                    = 0.2

# for Optimizer 
    initial_learning_rate:         float                    = 0.004
    weight_decay:                  float                    = 1e-5

# criterion
    criterion:                     nn.Module                = nn.CrossEntropyLoss()

# # for StepLR
#     step_size:                     int                      = 1000
#     gamma:                         float                    = 0.1   

# # for ReduceLROnPlateau
#     patience:                      int                      = 2000
#     factor:                        float                    = 0.8
#     min_lr:                        float                    = 1e-4

# operational
    epochs:                        int                      = 20000
    seed:                          int                      = 42
    patience_delay:                float                    = 0.75
    save_checkpoints:              bool                     = True
    spit:                          callable                 = print
    max_patience:                  int                      # calculated
    epochs_before_patience:        int                      # calculated
    device:                        torch.device             # calculated
    str_run_date:                  str                      # calculated
    run_path:                      str                      # calculated
    process_path:                  str                      # calculated
    checkpoint_path:               str                      # calculated
    
    def __init__(self, parameter_set_id: int = -1, run_date: datetime = None, spit: callable = print):
        self.parameter_set_id       = parameter_set_id
        self.str_run_date           = (run_date or datetime.now()).strftime("%Y-%m-%d_%H_%M_%S")
        self.run_path               = f"./results/{self.str_run_date}"
        self.process_path           = f"{self.run_path}/{self.parameter_set_id}"
        self.checkpoint_path        = f"{self.process_path}/checkpoints"
        self.epochs_before_patience = int(self.epochs * self.patience_delay)
        self.max_patience           = self.epochs // 5
        self.device                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spit                   = spit

    def __str__(self):
        def serialize_value(v):
            try:
                if isinstance(v, nn.Module):
                    return v.__class__.__name__
                elif isinstance(v, torch.device):
                    return str(v)
                elif callable(v):
                    return f"<function {v.__name__}>" if hasattr(v, '__name__') else str(v)
                return v
            except Exception as e:
                print(f"Error serializing {v}: {e}")
                return f"<unserializable {type(v).__name__}>"

        try:
            # Modified filter to include nn.Module
            class_vars = {
                k: v for k, v in vars(self.__class__).items() 
                if not k.startswith('__') and (not callable(v) or isinstance(v, nn.Module))
            }
            
            print(f"Criterion type: {type(self.criterion)}")  # Debug
            all_vars = {**class_vars, **self.__dict__}
            
            serializable_dict = {
                k: serialize_value(v)
                for k, v in all_vars.items()
            }
            return json.dumps(serializable_dict, indent=4)
        except Exception as e:
            print(f"Error in __str__: {e}")
            return str(e)


    @classmethod
    def from_json(cls, json_path: str) -> 'Hyperparameters':
        try:
            with open(json_path, 'r') as f:
                data: Dict[str, Any] = json.load(f)
                
            hp = cls()
            
            for key, value in data.items():
                if key == 'timestamp':
                    value = datetime.fromisoformat(value)
                elif key == 'criterion':
                    if value not in cls.criterion_map:
                        raise HyperparameterError(f"Unknown criterion: {value}")
                    value = cls.criterion_map[value]
                elif key == 'spit':
                    pass
                setattr(hp, key, value)
                    
            return hp
            
        except FileNotFoundError:
            raise HyperparameterError(f"Hyperparameter file not found: {json_path}")
        except json.JSONDecodeError:
            raise HyperparameterError(f"Invalid JSON in file: {json_path}")