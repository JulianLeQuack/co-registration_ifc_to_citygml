import numpy as np
import json
from shapely.geometry import Polygon, MultiPolygon
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

    def transform_points(self, points):
        """
        Applies a 2D rigid transformation to a set of points.
        """
        points = np.asarray(points)
        R = self.rotation_matrix()
        t = self.translation_vector()
        transformed_points = (R @ points.T).T + t
        return transformed_points
    
    def transform_shapely_polygon(self, polygon):
        """
        Apply a given transformation to a Shapely Polygon or MultiPolygon and return the transformed geometry.
        """
        if hasattr(polygon, "geom_type"):
            if polygon.geom_type == "MultiPolygon":
                transformed_polys = []
                for poly in polygon.geoms:
                    coords = np.array(poly.exterior.coords)
                    transformed_coords = self.transform_points(coords)
                    # Ensure closure.
                    if not np.allclose(transformed_coords[0], transformed_coords[-1]):
                        transformed_coords = np.vstack([transformed_coords, transformed_coords[0]])
                    transformed_polys.append(Polygon(transformed_coords))
                return MultiPolygon(transformed_polys)
            elif polygon.geom_type == "Polygon":
                coords = np.array(polygon.exterior.coords)
                transformed_coords = self.transform_points(coords)
                if not np.allclose(transformed_coords[0], transformed_coords[-1]):
                    transformed_coords = np.vstack([transformed_coords, transformed_coords[0]])
                return Polygon(transformed_coords)
            else:
                print("Unexpected geometry type.")
                return None
        else:
            print("Input is not a Shapely geometry.")
            return None
        

    def transform_ifc(self, input_ifc_path, output_ifc_path):
        """
        Apply the transformation to an IFC file and save the transformed model.
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
        print(f"Transformed IFC model saved to: {output_ifc_path}")


    
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