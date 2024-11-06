import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
from fizz_buzz_nn import Model
import numpy as np

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model and move to GPU
model = Model(input_dim=10, output_dim=4)

checkpoint_folder = "./2024-11-05_14_34_02/checkpoints/"
checkpoint_files = sorted([os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder) if f.endswith(".pth")])

# Set up the figure for plotting
fig, ax = plt.subplots(figsize=(10, 6))

vmin, vmax = -1, 1
make_it = True

def animate(index):
    global make_it
    # Clear the axis to prepare for new plot
    ax.clear()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_files[index], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    
    # Get weights from the first linear layer
    for name, param in model.named_parameters():
        if 'linear4.weight' in name:  
            data = param.detach().cpu().numpy()
            sns.heatmap(data, annot=False, cmap='viridis', ax=ax, vmin=vmin, vmax=vmax, cbar=make_it)

            if make_it:
                make_it = False

            ax.set_title(f'Heatmap of {name} - Epoch {index + 1}')
            ax.set_xlabel('Output Nodes')
            ax.set_ylabel('Input Nodes')
            break

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(checkpoint_files), interval=500, repeat=False)

# Save as an animated GIF or video (e.g., MP4)
ani.save('weights_evolution4.gif', writer='pillow')
# For MP4, you could use:
# ani.save('weights_evolution.mp4', writer='ffmpeg')

# Show the animation (optional if you just want to save it)
plt.show()