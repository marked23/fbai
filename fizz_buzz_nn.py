import torch
from hyperparameters import Hyperparameters

class Model(torch.nn.Module):
    drop: float = 0.2

    def __init__(self, hp: Hyperparameters):
        super(Model, self).__init__()
        self.drop = hp.drop
        hidden_x4 = hp.hidden_dim * 4
        hidden_x2 = hp.hidden_dim * 2
        self.linear1 = torch.nn.Linear(hp.input_dim, hidden_x2)
        self.relu1 = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(hidden_x4, hidden_x2)
        # self.relu2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=self.drop)
        # self.linear3 = torch.nn.Linear(hidden_x2, hp.hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_x2, hp.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        # x = self.relu2(self.linear2(x))
        x = self.dropout(x)
        # x = self.linear3(x)
        x = self.linear4(x)
        return x


class WideModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(WideModel, self).__init__()
        self.layer1 = torch.nn.Linear(10, 64)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layer2 = torch.nn.Linear(64, 4)
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)

class DeepModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(DeepModel, self).__init__()
        self.layer1 = torch.nn.Linear(10, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, 32)
        self.layer4 = torch.nn.Linear(32, 4)
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

class PyramidModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(PyramidModel, self).__init__()
        self.layer1 = torch.nn.Linear(10, 48)
        self.layer2 = torch.nn.Linear(48, 24)
        self.layer3 = torch.nn.Linear(24, 12)
        self.layer4 = torch.nn.Linear(12, 4)
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
    
class ImprovedModel(torch.nn.Module):
    def __init__(self, hp: Hyperparameters):
        super(ImprovedModel, self).__init__()
        # Input layer
        self.layer1 = torch.nn.Linear(hp.input_dim, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        
        # Hidden layer
        self.layer2 = torch.nn.Linear(64, 32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        
        # Output layer
        self.layer3 = torch.nn.Linear(32, hp.output_dim)
        
        # Activation and regularization
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)  # Reduced dropout
        
        # Better initialization
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