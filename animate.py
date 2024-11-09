import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
from argparse import ArgumentParser, Namespace
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


def create_weight_animation(layer_name, checkpoint_files, model, step: int, vmin=-1, vmax=1):
    """Create animation for a specific layer
    
    Args:
        layer_name: Name of layer to animate
        checkpoint_files: List of checkpoint files
        model: PyTorch model 
        step: Number of checkpoints to skip between frames (default: 1)
        vmin: Minimum value for heatmap (default: -1)
        vmax: Maximum value for heatmap (default: 1)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    make_it = True
    
    def animate(frame_idx):
        nonlocal make_it
        ax.clear()
        
        # Calculate actual checkpoint index based on step
        checkpoint_idx = frame_idx * step
        checkpoint = torch.load(checkpoint_files[checkpoint_idx], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        
        for name, param in model.named_parameters():
            if name == layer_name:
                data = param.detach().cpu().numpy().T
                sns.heatmap(data, annot=False, cmap='viridis', ax=ax, 
                          vmin=vmin, vmax=vmax, cbar=make_it)
                
                if make_it:
                    make_it = False
                    
                ax.set_title(f'Heatmap of {name} - Epoch {checkpoint_idx + 1}')
                ax.set_xlabel('Output Nodes')
                ax.set_ylabel('Input Nodes')
                break
                
    num_frames = len(checkpoint_files) // step
    ani = FuncAnimation(fig, animate, frames=num_frames, repeat=False)
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

def animate_layer(layer_name: str, weight_layers: list, checkpoint_files: list, model, process_path: str, step: int) -> bool:
    if layer_name not in weight_layers:
        print(f"Error: Layer '{layer_name}' not found.")
        return False
        
    ani, writer, fig = create_weight_animation(layer_name, checkpoint_files, model, step=step)
    output_file = f'{process_path}/{layer_name.replace(".", "_")}.mp4'
    print(f'Generating animation for {layer_name}...')
    ani.save(output_file, writer=writer)
    print(f'Saved {layer_name} to {output_file}')
    plt.close(fig)
    return True

def find_layer_by_number(number: int, weight_layers: list) -> str:
    """Find layer name matching pattern linear{number}_weight"""
    pattern = f"linear{number}.weight"
    matches = [layer for layer in weight_layers if layer == pattern]
    return matches[0] if matches else None

def parse_args() -> Namespace:
    """Parse command line arguments"""
    parser: ArgumentParser = ArgumentParser(description='Generate weight layer animations')
    parser.add_argument('--all', action='store_true', help='Generate animations for all layers')
    parser.add_argument('--layer', type=int, help='Number of specific layer to animate')
    parser.add_argument('--step', type=int, default=1, help='Number of checkpoints to skip between frames')
    return parser.parse_args()


def main():
    args = parse_args()
    weight_layers = get_weight_layers(model)


    if args.layer is not None:
        layer_name = find_layer_by_number(args.layer, weight_layers)
        if not layer_name:
            print(f"Error: No layer found matching linear{args.layer}_weight")
            print("Available layers:")
            for layer_name in weight_layers:
                print(f"  - {layer_name}")
            return
        animate_layer(layer_name, weight_layers, checkpoint_files, model, process_path, args.step)

    elif args.all:
        # Create and save animation for each layer
        for layer_name in weight_layers:
            animate_layer(layer_name, weight_layers, checkpoint_files, model, process_path, args.step)

    else:
        print("\nAvailable layers:")
        for layer_name in weight_layers:
            print(f"  - {layer_name}")
        print("\nUse --all to generate animations for all layers.")
        print("Use --layer <number> to generate animation for a specific layer.\n")
        
        # output_file = f'{process_path}/{layer_name.replace(".", "_")}.mp4'
        # print(f'what-if: Would save {layer_name} to {output_file}')

if __name__ == '__main__':
    main()