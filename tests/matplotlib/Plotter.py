import matplotlib.pyplot as plt

class Plotter:

    def plot_points(points):
        x_coords, y_coords = zip(*points)
        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords)
        plt.grid(True)
        plt.show()

    def plot_polygon(polygon):
        plt.figure(figsize=(8, 8))
        plt.grid(True)
        plt.plot(*polygon.exterior.xy)
        plt.show()

    def plot_polygons(polygon1, polygon2):
        plt.figure(figsize=(8, 8))
        plt.grid(True)
        plt.plot(*polygon1.exterior.xy)
        plt.plot(*polygon2.exterior.xy)
        plt.show()

if __name__ == "__main__":
    print("Plotter Module")