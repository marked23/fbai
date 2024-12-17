from dataclasses import dataclass, field
from typing import Union, List, Optional
from hyperparameters import Hyperparameters

@dataclass
class PerturbRule:
    param_name: str
    start: Union[int, float, None] = None
    step: Union[int, float] = 1
    multiply: bool = False
    array: Optional[List[str]] = field(default=None)
    each: int = 1

def describe_perturbation_scenario(hp_sets: List[Hyperparameters], rules: List[PerturbRule]) -> List[str]:
    output_lines = []
    for rule in rules:
        values = []
        for i in range(len(hp_sets)):
            if rule.array:
                index = (i // rule.each) % len(rule.array)
                value = rule.array[index]
            elif rule.multiply:
                value = rule.start * (rule.step ** i)
            else:
                value = rule.start + (rule.step * i)
            values.append(value)
        
        if rule.array:
            output_lines.append(f"[{0:>4}] {rule.param_name}: array={rule.array}, each={rule.each}")
        else:
            output_lines.append(f"[{0:>4}] {rule.param_name}: {values[0]:.3f}, {values[1]:.3f}, ..., {values[-1]:.3f}")
    
    return output_lines



def apply_perturbations(hp_sets: List[Hyperparameters], rules: List[PerturbRule]):
    info = describe_perturbation_scenario(hp_sets, rules)
    info.sort()
    info_string = "\n".join(info)
    print(info_string)
    for i, hp in enumerate(hp_sets):
        rule_info = []
        for rule in rules:
            if rule.array:
                index = (i // rule.each) % len(rule.array)
                value = rule.array[index]            
            elif rule.multiply:
                value = rule.start * (rule.step ** i)
            else:
                value = rule.start + (rule.step * i)
            setattr(hp, rule.param_name, value)
            rule_info.append(f"{rule.param_name} = {value}")
        hp.perturb_info = ", ".join(rule_info)
        print(f"[{i:>4}] {hp.perturb_info}")
        
# Usage example:
# rules = [
#     PerturbRule("hidden_dim", 14, 1),              # Linear: 14,15,16...
#     PerturbRule("initial_learning_rate", 0.001, 2, True),  # Geometric: 0.001,0.002,0.004...
#     PerturbRule("drop", 0.1, 0.05)                 # Linear: 0.1,0.15,0.2...
# ]
# hp_sets = [Hyperparameters(i, datetime.now()) for i in range(4)]
# apply_perturbations(hp_sets, rules)
