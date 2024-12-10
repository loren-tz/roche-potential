import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter
import os

# Function to collect input from the user with default values
def get_input(prompt, default):
    user_input = input(f"{prompt} (default={default}): ").strip()
    return float(user_input) if user_input else default

# Initial prompt to choose the mode
debug_mode = input("Do you want to proceed in debug mode? (y/n, default=n): ").strip().lower() in ['s', 'si', 'y', 'yes']

# Parameter initialization
if debug_mode:
    print("Debug mode active. Using default parameters.")
    parameters = [1.0, 1.0, 3.5, 0.1, 50]  # Default: M1=1.0, M2=1.0, L=3.5, R=0.1, N=50
else:
    print("Normal mode. Please input the parameters.")
    parameters = []
    parameters.append(get_input("Enter the mass M1", 1.0)) # M1
    parameters.append(get_input("Enter the mass M2", 1.0)) # M2
    parameters.append(get_input("Enter the value of L", 3.5))  # L
    parameters.append(get_input("Enter the value of R", 0.1))  # R
    parameters.append(get_input("Enter the value of N", 50))  # N

cutoff = -parameters[2] * 30  # Cutoff potential

# Building the parameters to pass to the C program
args = [str(parameters[0]), str(parameters[1]), str(parameters[2]), str(parameters[3]), str(cutoff)]

if not os.path.exists('./roche_sim'):
    print("Error: Executable roche_sim not found.")
    exit()

# Executing the C script
try:
    result = subprocess.run(
        ["./roche_sim"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    print("Simulation successfully completed.\nProcessing the data...")
except subprocess.CalledProcessError as e:
    print("Error during the execution of roche_sim.exe:")
    print(e.stderr)
    exit()

# Reading the generated data
file_path = "roche_data.dat"
try:
    data = np.loadtxt(file_path)  # Load the data file
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
except Exception as e:
    print(f"Error while reading the file {file_path}:")
    print(e)
    exit()
if data.size == 0:
    print("Error: The file roche_data.dat is empty or invalid.")
    exit()

# Mapping onto a NxN grid
x_unique = np.linspace(np.min(x), np.max(x), int(parameters[4]))
y_unique = np.linspace(np.min(y), np.max(y), int(parameters[4]))
X, Y = np.meshgrid(x_unique, y_unique)
Z = griddata((x, y), z, (X, Y), method='cubic')

# Masking NaN values
nan_mask = np.isnan(Z)

# Temporary replacement of NaN with 0 for filtering
Z_temp = np.nan_to_num(Z, nan=0)

# Applying Gaussian filter
sigma = 3  # Increase to cover larger areas
Z_filtered = gaussian_filter(Z_temp, sigma=sigma)

# Calculating weights using the inverted mask
weights = gaussian_filter(np.logical_not(nan_mask).astype(float), sigma=sigma)

# Filling NaN values with the weighted result
Z[nan_mask] = Z_filtered[nan_mask] / np.maximum(weights[nan_mask], 1e-10)

# Mask of valid values
valid_mask = ~np.isnan(Z)
x_valid = X[valid_mask]
y_valid = Y[valid_mask]
z_valid = Z[valid_mask]

# Interpolation with RBF
rbf = Rbf(x_valid, y_valid, z_valid, function='multiquadric', smooth=1)
Z_rbf = rbf(X, Y)

# Replacing NaN with RBF
nan_mask = np.isnan(Z)
Z[nan_mask] = Z_rbf[nan_mask]

# Applying Gaussian filter to smooth transitions
sigma_smooth = 3
Z = gaussian_filter(Z, sigma=sigma_smooth)

# Creating the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Surface plot
surf = ax.plot_surface(X, Y, Z - np.min(Z) / 4, cmap="YlOrBr_r", alpha=0.8, edgecolor="black", linewidth=0.1)

# Projecting the surface onto the XY plane
ax.contourf(X, Y, Z, zdir='z', offset=np.min(Z), cmap='Greys_r', alpha=1)

# Configuring the plot
ax.set_xticks([])
ax.set_xlabel('')
ax.set_yticks([])
ax.set_ylabel('')
ax.set_zticks([])
ax.set_zlabel('')
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
cbar = fig.colorbar(surf, shrink=0.4, aspect=15, pad=0.05)
cbar.set_ticks([])
cbar.set_label(r"$V_{\mathrm{R}}(x, y)$", fontsize=12, labelpad=10, rotation=90)
ax.set_title(r"Roche Potential for $M_1$, $M_2$", fontsize=12, pad=0)

# Function to update the view for each frame
def update(frame):
    ax.view_init(elev=30, azim=frame)  # Change azimuth angle

# Creating the animation
frames = 360  # Number of frames (one full rotation = 360 degrees)
fps = 30
ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False)

save_gif = input("Do you want to save the animation? (y/n): ").strip().lower() in ['s', 'si', 'y', 'yes']

# Saving the animation
if save_gif:
    print("Exporting...")
    output_filename = "roche_potential.gif"
    try:
        writer = PillowWriter(fps=fps, metadata={"loop": 0})
        ani.save(output_filename, writer=writer, dpi=100)  # Risoluzione ridotta
        print(f"Animation successfully saved as {output_filename}")
    except Exception as e:
        print(f"Error while saving the animation: {e}")

print("Close the plot to exit.")

# Show the plot
plt.show()
