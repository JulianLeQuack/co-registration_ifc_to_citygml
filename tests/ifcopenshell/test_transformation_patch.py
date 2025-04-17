import numpy as np
import ifcopenshell
import ifcpatch

# Load your model
model = ifcopenshell.open("./test_data/ifc/3.002 01-05-0501_EG.ifc")

# Prepare arguments:
#   x, y, z   → translation in project units
#   should_rotate_first → True to rotate then translate
#   ax, ay, az → rotation angles in degrees around X, Y, Z (we only need Z)
x = 690952.9369419415
y = 5335977.822957996
z = 0
theta_rad = 1.1958801418366802
az = np.degrees(theta_rad)  # ~68.55°

# Execute the patch
patched = ifcpatch.execute({
    "input":      "input.ifc",
    "file":       model,
    "recipe":     "OffsetObjectPlacements",
    "arguments":  [x, y, z, True, 0, 0, az]
})

# Write out
ifcpatch.write(patched, "output_transformed.ifc")
print("Done – patched via OffsetObjectPlacements.")
