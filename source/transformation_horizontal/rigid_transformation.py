import numpy as np
import json
from shapely.geometry import Point
from shapely.affinity import translate, rotate
import ifcopenshell
import ifcpatch

class Rigid_Transformation:

    def __init__(self, t:np.array=0, theta=0):
        self.t = t
        self.theta = theta

    def __str__(self):
        return f"Translation (x,y): {self.t}, Theta (radians): {self.theta}"
        

    def translation_vector(self):
        return self.t
    

    def rotation_matrix(self):
        rotation_matrix = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta),  np.cos(self.theta)]
        ])
        return rotation_matrix
    
    def transform(self, input):
        """
        Applies a 2D rigid transformation to a given input.
        First rotates around origin, then translates.
        """
        # First rotate around the origin
        output = rotate(input, angle=np.degrees(self.theta), origin=(0, 0))
        # Then translate
        output = translate(output, xoff=self.t[0], yoff=self.t[1])
        return output
    
    def transform_features(self, features: np.array):
        """
        Applies a 2D rigid transformation to a set of features.
        Each feature is expected to be in format:
        [polygon_index, vertex_index, x_coordinate, y_coordinate, turning_angle]
        
        Only the x,y coordinates (indices 2,3) are transformed.
        """
        transformed_features = features.copy()
        
        # Extract just the x,y coordinates
        coords = features[:, 2:4]
        
        # Apply rotation
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        # Rotate coords around origin (0,0)
        x_rotated = coords[:, 0] * cos_theta - coords[:, 1] * sin_theta
        y_rotated = coords[:, 0] * sin_theta + coords[:, 1] * cos_theta
        
        # Apply translation
        transformed_coords = np.column_stack([
            x_rotated + self.t[0],
            y_rotated + self.t[1]
        ])
        
        # Update just the coordinate columns
        transformed_features[:, 2:4] = transformed_coords
        
        return transformed_features        

    def transform_ifc(self, input_ifc_path, output_ifc_path):
        """
        Apply the horizontal transformation to an IFC file and save the transformed model.
        """
        # Load the IFC model
        model = ifcopenshell.open(input_ifc_path)
        
        # Prepare arguments for ifcpatch
        x, y = self.t
        z = 0
        theta_rad = self.theta
        az = np.degrees(theta_rad)

        # Execute the patch
        patched = ifcpatch.execute({
            "input": input_ifc_path,
            "file": model,
            "recipe": "OffsetObjectPlacements",
            "arguments": [x, y, z, True, 0, 0, az]
        })
        # Write out the transformed IFC model
        ifcpatch.write(patched, output_ifc_path)

    def offset_ifc(self, input_ifc_path, output_ifc_path, z=0):
        """
        Apply the vertical transformation to an IFC file and save the transformed model.
        """
        # Load the IFC model
        model = ifcopenshell.open(input_ifc_path)
        
        # Prepare arguments for ifcpatch
        x, y = 0 # No horizontal offset

        # Execute the patch
        patched = ifcpatch.execute({
            "input": input_ifc_path,
            "file": model,
            "recipe": "OffsetObjectPlacements",
            "arguments": [x, y, z, True, 0, 0, 0]
        })
        # Write out the transformed IFC model
        ifcpatch.write(patched, output_ifc_path)

    
    def transform_elevation_labels(self, elevation_labels:np.array):
        for label in elevation_labels:
            label[0] = self.transform(label[0])
        return elevation_labels


    
    @classmethod
    def import_from_json(cls, file_path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                x = data.get("x", 0) #using .get() prevents key errors.
                y = data.get("y", 0)
                theta = data.get("theta", 0)
                #Create and return new instance
                return cls(x, y, theta)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON format in: {file_path}")
            return None


    def export_to_json(self, file_path):
        data = {
            "t": self.t.tolist(), #Convert numpy array to list for JSON serialization
            "theta": self.theta,
        }
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4) #indent=4 makes the json file more readable.


if __name__ == "__main__":
    transformation_dict = {
    "t": [
        690952.9369419415,
        5335977.822957996
    ],
    "theta": 1.1958801418366802
    }

    transformation = Rigid_Transformation(t=transformation_dict["t"], theta=transformation_dict["theta"])

    ifc_path = "./test_data/ifc/3.002 01-05-0501_EG.ifc"

    transformation.transform_ifc(input_ifc_path=ifc_path, output_ifc_path=ifc_path + "transformed.ifc")