import numpy as np

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