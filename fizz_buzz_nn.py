import torch
from hyperparameters import Hyperparameters

class Model(torch.nn.Module):
    drop: float = 0.2

    def __init__(self, hp: Hyperparameters):
        super(Model, self).__init__()
        self.drop = hp.drop
        hidden_x4 = hp.hidden_dim * 4
        hidden_x2 = hp.hidden_dim * 2
        self.linear1 = torch.nn.Linear(hp.input_dim, hidden_x4)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_x4, hidden_x2)
        self.relu2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.drop)
        self.linear3 = torch.nn.Linear(hidden_x2, hp.hidden_dim)
        self.linear4 = torch.nn.Linear(hp.hidden_dim, hp.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x