import matplotlib.pyplot as plt
from source.transformation_horizontal.rigid_transformation import Rigid_Transformation
from source.transformation_vertical.extract_elevation_labels import extract_elevation_labels
import numpy as np
from copy import deepcopy

dxf_path = "./test_data/dxf/01-05-0501_EG.dxf"
layer_name = "A_03_HOEHENKOTE"
elevation_labels = extract_elevation_labels(dxf_path, layer_name)

# Make a deep copy of the original labels before transformation
original_elevation_labels = deepcopy(elevation_labels)

rigid_params = {
    "t": [
        690952.9369419415,
        5335977.822957996
    ],
    "theta": 1.1958801418366802
}

rigid_transformation = Rigid_Transformation(t=rigid_params["t"], theta=rigid_params["theta"])

# Transform the elevation labels
for label in elevation_labels:
    label[0] = rigid_transformation.transform(label[0])

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Original elevation labels
if original_elevation_labels.size > 0:
    # Extract x and y coordinates from Shapely Points
    x_coords = [label[0].x for label in original_elevation_labels]
    y_coords = [label[0].y for label in original_elevation_labels]
    
    # Plot the points
    ax1.scatter(x_coords, y_coords, color='blue', label='Original Labels')
    
    # Annotate each point with its text label
    for label in original_elevation_labels:
        ax1.annotate(label[1], (label[0].x, label[0].y), fontsize=8, ha='right')
else:
    ax1.text(0.5, 0.5, "No elevation labels found", transform=ax1.transAxes, 
            ha="center", va="center")

ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_title("Original Elevation Labels")
ax1.legend()
ax1.grid(True)
ax1.set_aspect("equal", adjustable="box")

# Plot 2: Transformed elevation labels
if elevation_labels.size > 0:
    # Extract x and y coordinates from Shapely Points
    x_coords = [label[0].x for label in elevation_labels]
    y_coords = [label[0].y for label in elevation_labels]
    
    # Plot the points
    ax2.scatter(x_coords, y_coords, color='red', label='Transformed Labels')
    
    # Annotate each point with its text label
    for label in elevation_labels:
        ax2.annotate(label[1], (label[0].x, label[0].y), fontsize=8, ha='right')
else:
    ax2.text(0.5, 0.5, "No elevation labels found", transform=ax2.transAxes, 
            ha="center", va="center")

ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_title(f"Transformed Elevation Labels (θ={rigid_params['theta']:.2f})")
ax2.legend()
ax2.grid(True)
ax2.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()