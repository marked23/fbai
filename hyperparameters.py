import torch
import datetime
import json
from datetime import datetime
# from typing import Dict, Any


class HyperparameterError(Exception):
    """Custom exception for hyperparameter loading errors"""
    pass

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

    def __str__(self):
        def serialize_value(v):
            try:
                if isinstance(v, torch.nn.Module):
                    return v.__class__.__name__
                elif isinstance(v, torch.device):
                    return str(v)
                elif callable(v):
                    return f"<function {v.__name__}>"
                return v
            except Exception as e:
                print(f"Error serializing {v}: {e}")
                return f"<unserializable {type(v).__name__}>"

        try:
            class_vars = {
                k: v for k, v in vars(self.__class__).items() 
                if not k.startswith('__') and (not callable(v) or isinstance(v, torch.nn.Module) or isinstance(v, torch.device))
            }
            
            all_vars = {**class_vars, **self.__dict__}
            
            serializable_dict = {}
            for k, v in all_vars.items():
                try:
                    serialized_value = serialize_value(v)
                    serializable_dict[k] = serialized_value
                except Exception as e:
                    print(f"Error serializing key {k}: {e}")
                    serializable_dict[k] = f"<unserializable {type(v).__name__}>"

            return json.dumps(serializable_dict, indent=4)
        
        except Exception as e:
            print(f"Error in __str__: {e} line: {e.__traceback__.tb_lineno}")
            return str(e)

class HyperparametersLoader:

    criterion_map = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        'MSELoss': torch.nn.MSELoss,
        'BCELoss': torch.nn.BCELoss
    }
    

    def from_json(cls, json_path: str) -> 'Hyperparameters':
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            hp = Hyperparameters()
            
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