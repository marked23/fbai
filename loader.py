import torch
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_sample import DataSample
from hyperparameters import Hyperparameters

hp = Hyperparameters()

class Loader:

    @staticmethod        
    def create_training_loader() -> Tuple[DataLoader, DataLoader]:
        training_data = [DataSample.create(n) for n in range(101, 1024)]

        training_features = torch.stack([features for _, features, _     in training_data]).to(hp.device)
        training_labels = torch.tensor( [label    for _,        _, label in training_data]).to(hp.device)
        training_dataset = TensorDataset(training_features, training_labels)

        training_size = int(0.8 * len(training_dataset))
        validation_size = len(training_dataset) - training_size
        training_dataset, validation_dataset = random_split(training_dataset, [training_size, validation_size])
        
        training_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False)           

        return training_loader, validation_loader

    # @staticmethod        
    # def create_training_loader() -> DataLoader:
    #     training_data = [DataSample.create(n) for n in range(101, 1024)]

    #     training_features = torch.stack([features for _, features, _     in training_data]).to(hp.device)
    #     training_labels = torch.tensor( [label    for _,        _, label in training_data]).to(hp.device)
    #     training_dataset = TensorDataset(training_features, training_labels)
    #     training_loader = DataLoader(training_dataset, batch_size=256, shuffle=True)
        
    #     return training_loader

    @staticmethod
    def create_testing_loader() -> DataLoader:
        testing_data = [DataSample.create(n) for n in range(1, 101)]

        testing_features = torch.stack([features for _, features, _     in testing_data]).to(hp.device)
        testing_labels = torch.tensor( [label    for _,        _, label in testing_data]).to(hp.device)
        testing_dataset = TensorDataset(testing_features, testing_labels)
        testing_loader = DataLoader(testing_dataset, batch_size=256, shuffle=False)
        
        return testing_loader