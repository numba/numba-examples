# Spatial Trees Implementation

This directory contains implementations of two spatial tree data structures for efficient nearest neighbor searches:
- KD-Tree: A space-partitioning data structure for organizing points in k-dimensional space
- Ball Tree: A metric tree that partitions data in a series of nesting hyper-spheres

## Features

- Pure Python implementation (no dependencies)
- Support for k-dimensional points
- Efficient nearest neighbor queries
- Comprehensive test suite with accuracy and performance benchmarks

## Usage

```python
# Example with KD-Tree
points = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
tree = KDTree(points)
query_point = [6, 5]
nearest = tree.find_nearest(query_point)

# Example with Ball Tree
tree = BallTree(points, leaf_size=40)  # leaf_size is optional
nearest = tree.find_nearest(query_point)
```

## Performance

Both tree structures offer significant speedup over brute force search, especially for larger datasets:

```
Points  Dims    Brute     KD-Tree   Ball-Tree
---------------------------------------------
100     2       0.000069  0.000033  0.000047
100     3       0.000085  0.000042  0.000029
1000    2       0.000716  0.000054  0.000174
1000    3       0.000848  0.000036  0.000380
```

## Implementation Notes

- KD-Tree recursively partitions space using axis-aligned hyperplanes
- Ball Tree recursively partitions space using hyperspheres
- Both implementations handle edge cases and numerical stability
- When multiple points are equidistant from the query point, any of them may be returned

Fixes #35