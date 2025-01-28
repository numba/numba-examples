"""
Test and demonstration of KDTree and BallTree implementations.

This script demonstrates the usage of both tree implementations and
compares their results with brute force nearest neighbor search.

Fixes #35
"""
import numpy as np
from timeit import default_timer as timer
from kdtree import KDTree
from balltree import BallTree

def brute_force_nearest(points, query_point):
    """Find nearest neighbor using brute force approach."""
    min_dist = float('inf')
    nearest = None
    for point in points:
        dist = sum((a - b) ** 2 for a, b in zip(point, query_point))
        if dist < min_dist:
            min_dist = dist
            nearest = point
    return nearest

def test_accuracy():
    """Test accuracy of tree implementations against brute force."""
    # Generate test points
    points = []
    for x in range(-5, 5):
        for y in range(-5, 5):
            for z in range(-5, 5):
                points.append([float(x), float(y), float(z)])
    
    # Generate query points
    query_points = [
        [0, 0, 0],
        [1, 1, 1],
        [2, -2, 0],
        [-3, 1, 4],
        [0.5, -0.5, 0.5],
        [-1, -1, -1],
        [3, 3, 3],
        [-2.5, 2.5, -2.5],
        [4, -4, 4],
        [-1.5, 1.5, -1.5]
    ]
    
    # Build trees
    kdtree = KDTree(points)
    balltree = BallTree(points)
    
    print("Testing accuracy...")
    for i, query in enumerate(query_points):
        # Get nearest neighbors using different methods
        nearest_brute = brute_force_nearest(points, query)
        nearest_kd = kdtree.find_nearest(query)
        nearest_ball = balltree.find_nearest(query)
        
        # Verify results match by comparing distances (there might be multiple equidistant points)
        def get_distance(p1, p2):
            return sum((a - b) ** 2 for a, b in zip(p1, p2))
            
        dist_brute = get_distance(query, nearest_brute)
        dist_kd = get_distance(query, nearest_kd)
        dist_ball = get_distance(query, nearest_ball)
        
        tol = 1e-10
        kd_correct = abs(dist_brute - dist_kd) < tol
        ball_correct = abs(dist_brute - dist_ball) < tol
        
        print(f"Query point {i+1}: {query}")
        print(f"Brute force: {nearest_brute} (dist: {dist_brute:.8f})")
        print(f"KD-Tree:     {nearest_kd} (dist: {dist_kd:.8f}) - {kd_correct}")
        print(f"Ball-Tree:   {nearest_ball} (dist: {dist_ball:.8f}) - {ball_correct}")
        print()

def benchmark():
    """Benchmark performance of different approaches."""
    sizes = [100, 1000]  # Reduced sizes for faster testing
    dims = [2, 3]  # Reduced dimensions for faster testing
    
    print("\nBenchmarking performance...")
    print("Format: time in seconds (lower is better)\n")
    print("Points  Dims    Brute     KD-Tree   Ball-Tree")
    print("-" * 45)
    
    for n_points in sizes:
        for dim in dims:
            # Generate points in a grid pattern
            points = []
            grid_size = int(n_points ** (1/3)) + 1
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        point = [float(i)/grid_size, float(j)/grid_size, float(k)/grid_size][:dim]
                        points.append(point)
            points = points[:n_points]  # Trim to exact size
            query = [0.0] * dim  # Query point at origin
            
            # Build trees
            t0 = timer()
            kdtree = KDTree(points)
            t1 = timer()
            balltree = BallTree(points)
            t2 = timer()
            
            # Time queries
            t3 = timer()
            _ = brute_force_nearest(points, query)
            brute_time = timer() - t3
            
            t4 = timer()
            _ = kdtree.find_nearest(query)
            kd_time = timer() - t4
            
            t5 = timer()
            _ = balltree.find_nearest(query)
            ball_time = timer() - t5
            
            print(f"{n_points:<7} {dim:<7} {brute_time:.6f}  {kd_time:.6f}  {ball_time:.6f}")

if __name__ == "__main__":
    test_accuracy()
    benchmark()