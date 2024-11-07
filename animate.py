import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from fizz_buzz_nn import Model
import numpy as np
from hyperparameters import Hyperparameters 
import datetime

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reload hp from json file
hp = Hyperparameters(1, datetime.datetime.now())
hp.input_dim = 10
hp.output_dim = 4
hp.hidden_dim = 31
hp.parameter_set_id = 1

# Instantiate the model and move to device
model = Model(hp).to(device)


run_path = "./results/2024-11-06_21_38_40"
process_path = f"{run_path}/{hp.parameter_set_id}"
checkpoint_folder = f"{process_path}/checkpoints"
checkpoint_files = sorted([os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder) if f.endswith(".pth")])

# Set up the figure for plotting
fig, ax = plt.subplots(figsize=(10, 6))

vmin, vmax = -1, 1
make_it = True

def get_weight_layers(model):
    """Get names of all weight layers"""
    return [name for name, param in model.named_parameters() if 'weight' in name]


def create_weight_animation(layer_name, checkpoint_files, model, vmin=-1, vmax=1):
    """Create animation for a specific layer"""
    fig, ax = plt.subplots(figsize=(10, 6))
    make_it = True
    
    def animate(index):
        nonlocal make_it
        ax.clear()
        
        checkpoint = torch.load(checkpoint_files[index], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        
        for name, param in model.named_parameters():
            if name == layer_name:
                data = param.detach().cpu().numpy()
                sns.heatmap(data, annot=False, cmap='viridis', ax=ax, 
                          vmin=vmin, vmax=vmax, cbar=make_it)
                
                if make_it:
                    make_it = False
                    
                ax.set_title(f'Heatmap of {name} - Epoch {index + 1}')
                ax.set_xlabel('Output Nodes')
                ax.set_ylabel('Input Nodes')
                break
                
    ani = FuncAnimation(fig, animate, frames=len(checkpoint_files), repeat=False)
    writer = FFMpegWriter(
        fps=60,
        bitrate=3000,
        codec='h264_nvenc',
        extra_args=[
            '-pix_fmt', 'yuv420p',
            '-preset', 'p7'
        ]
    )
    
    return ani, writer, fig

# Get all weight layer names
weight_layers = get_weight_layers(model)

# Create and save animation for each layer
for layer_name in weight_layers:
    ani, writer, fig = create_weight_animation(layer_name, checkpoint_files, model)
    output_file = f'{process_path}/weights_{layer_name.replace(".", "_")}.mp4'
    ani.save(output_file, writer=writer)
    print(f'Saved {layer_name} to {output_file}')
    plt.close(fig)  # Clean up the figure

