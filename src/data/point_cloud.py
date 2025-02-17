import numpy as np


# todo: make it a class

def sample_annulus(n, r_min, r_max):
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = np.random.uniform(r_min, r_max, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))


def sample_rectangle(width, height, num_points_per_edge=0):
    """
    Generate a point cloud of a rectangle's vertices.

    :param width: float, Width of the rectangle.
    :param height: float, Height of the rectangle.
    :param num_points_per_edge: int, Number of points sampled along each edge.
    :return: ndarray, (num_points, 2) array of point coordinates.
    """
    # Define the four corners of the rectangle
    corners = np.array([
        [0, 0], [width, 0], [width, height], [0, height], [0, 0]
    ])  # Closed loop

    # Interpolate points along each edge
    points = []
    for i in range(len(corners) - 1):
        start, end = corners[i], corners[i + 1]
        edge_points = np.linspace(start, end, num_points_per_edge)
        points.append(edge_points)

    return np.vstack(points)
