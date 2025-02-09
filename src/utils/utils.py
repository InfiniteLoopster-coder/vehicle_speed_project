import numpy as np

def compute_euclidean_distance(pt1, pt2):
    """
    Compute the Euclidean distance between two points (x, y).
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
