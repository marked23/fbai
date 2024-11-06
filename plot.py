import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from fizz_buzz_nn import Model

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model and move to GPU
model = Model(input_dim=10, output_dim=4)

model.load_state_dict(torch.load('./2024-11-05_13_46_40/model.pth', map_location=device, weights_only=True))

# Visualize weights and biases
for name, param in model.named_parameters():
    # Move parameter to CPU before converting to NumPy
    param_cpu = param.detach().cpu().numpy()

    if 'weight' in name:
        plt.figure(figsize=(10, 6))
        sns.heatmap(param_cpu, annot=False, cmap='viridis')
        plt.title(f'Heatmap of {name}')
        plt.xlabel('Output Nodes')
        plt.ylabel('Input Nodes')
        plt.show()
    elif 'bias' in name:
        plt.figure(figsize=(4, 6))
        sns.heatmap(param_cpu.reshape(-1, 1), annot=True, cmap='coolwarm')
        plt.title(f'Heatmap of {name}')
        plt.xlabel('Bias Value')
        plt.ylabel('Bias Index')
        plt.show()
