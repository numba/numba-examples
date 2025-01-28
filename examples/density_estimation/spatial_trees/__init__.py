"""
Spatial tree data structures for efficient nearest neighbor searches.

This module provides Numba-accelerated implementations of:
- KD-Tree: A space-partitioning data structure for organizing points in k-dimensional space
- Ball Tree: A metric tree that partitions data in a series of nesting hyper-spheres

Example usage:
-------------
import numpy as np
from spatial_trees import KDTree, BallTree

# Generate some random points
np.random.seed(42)
points = np.random.randn(1000, 3)  # 1000 points in 3D space
query_point = np.array([0.5, 0.5, 0.5])

# Using KDTree
kdtree = KDTree(points)
nearest_kd = kdtree.find_nearest(query_point)

# Using BallTree
balltree = BallTree(points)
nearest_ball = balltree.find_nearest(query_point)
"""

from .kdtree import KDTree
from .balltree import BallTree

__all__ = ['KDTree', 'BallTree']