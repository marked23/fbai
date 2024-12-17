import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter
import argparse
from argparse import ArgumentParser, Namespace
import os
from fizz_buzz_nn import Model
import numpy as np
from hyperparameters import Hyperparameters, HyperparametersLoader
import datetime

# leave as 'cpu' 
# because matplotlib does not use the GPU
device = 'cpu'  #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    previous_data = None
    
    # Calculate frame indices including final checkpoint
    regular_frames = list(range(0, len(checkpoint_files)-1, step))
    if regular_frames[-1] != len(checkpoint_files)-1:
        frame_indices = regular_frames + [len(checkpoint_files)-1]
    else:
        frame_indices = regular_frames

    def animate(frame_idx):
        nonlocal make_it, previous_data
        ax.clear()
        
        checkpoint_idx = frame_indices[frame_idx]
        checkpoint = torch.load(checkpoint_files[checkpoint_idx], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        
        for name, param in model.named_parameters():
            if name == layer_name:
                current_data = param.detach().cpu().numpy().T
                
                # Calculate changes from previous frame
                if previous_data is not None:
                    changes = current_data - previous_data
                    
                    significant_changes = np.abs(changes) > 0.05

                    sns.heatmap(current_data, annot=False, 
                              cmap='RdBu_r',  # Red-Blue diverging colormap
                              center=0,       # Center the colormap at 0
                              ax=ax, 
                              vmin=vmin, vmax=vmax, 
                              cbar=make_it)

                    # Find and highlight zero rows
                    zero_rows = np.all(np.abs(current_data) < 1e-5, axis=1)
                    for i in range(len(zero_rows)):
                        if zero_rows[i]:
                            ax.axhline(y=i+0.5, color='yellow', linewidth=6) 

                    # Highlight cells with significant changes
                    for i in range(changes.shape[0]):
                        for j in range(changes.shape[1]):
                            if significant_changes[i,j]:
                                # Place dot at center of cell
                                ax.plot(j + 0.5, i + 0.5, 'k.', markersize=1, alpha=0.3)  # 'k' means black, '.' means dot
                
                else:
                    sns.heatmap(current_data, annot=False, 
                              cmap='RdBu_r', center=0,
                              ax=ax, vmin=vmin, vmax=vmax, 
                              cbar=make_it)
                
                previous_data = current_data.copy()
                
                if make_it:
                    make_it = False
                    
                ax.set_title(f'Heatmap of {name} - Epoch {checkpoint_idx + 1}/{len(checkpoint_files)}')
                ax.set_xlabel('Output Nodes')
                ax.set_ylabel('Input Nodes')
                break

    num_frames = len(checkpoint_files) // step
    ani = FuncAnimation(fig, animate, frames=len(frame_indices), repeat=False)
    writer = FFMpegWriter(
        fps=1,
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
    parser.add_argument('process_path', type=str, help='Path to process directory (e.g. ./results/2024-03-14_16_16_28/1)')
    return parser.parse_args()


def main():
    args = parse_args()
    process_path = args.process_path
    
    checkpoint_folder = f"{process_path}/checkpoints"
    checkpoint_files = sorted([os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder) if f.endswith(".pth")])
    hpl = HyperparametersLoader()
    hp = hpl.from_json(f"{process_path}/hyperparameters.json")
    model = Model(hp).to(device)
        
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