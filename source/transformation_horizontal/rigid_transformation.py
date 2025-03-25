import numpy as np
import json

class Rigid_Transformation:

    x = 0
    y = 0
    theta = 0

    def __init__(self, x=0, y=0, theta=0):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        return f"Translation (x,y): {self.x, self.y}\nTheta (radiants): {self.theta}"
    
    def apply_transformation(self, points):
        """
        Applies a 2D rigid transformation to a set of points.
        """
        points = np.asarray(points)

        rotation_matrix = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                                    [np.sin(self.theta),  np.cos(self.theta)]])
        transformed_points = (rotation_matrix @ points.T).T + (self.x, self.y)
        return transformed_points
    
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
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
        }
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4) #indent=4 makes the json file more readable.