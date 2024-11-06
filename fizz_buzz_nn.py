import torch

class Model(torch.nn.Module):
    drop: float = 0.3

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super(Model, self).__init__()
        hidden_x4 = hidden_dim * 4
        hidden_x2 = hidden_dim * 2
        self.linear1 = torch.nn.Linear(input_dim, hidden_x4)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_x4, hidden_x2)
        self.relu2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.drop)
        self.linear3 = torch.nn.Linear(hidden_x2, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x