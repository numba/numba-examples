def build_kdtree(points, depth=0):
    n_points = len(points)
    if n_points == 0:
        return None
    
    n_dims = len(points[0])
    axis = depth % n_dims
    
    # Sort points based on the current axis
    points = sorted(points, key=lambda x: x[axis])
    median_idx = n_points // 2
    
    node = {
        'point': points[median_idx],
        'axis': axis,
        'left': build_kdtree(points[:median_idx], depth + 1),
        'right': build_kdtree(points[median_idx + 1:], depth + 1)
    }
    return node

def get_distance(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2))

def closest_point(root, point, best=None):
    if root is None:
        return best
    
    if best is None:
        best = root['point']
    
    # Update best if current point is closer
    if get_distance(root['point'], point) < get_distance(best, point):
        best = root['point']
    
    # Recursively search left or right subtree based on the splitting axis
    axis = root['axis']
    if point[axis] < root['point'][axis]:
        first, second = root['left'], root['right']
    else:
        first, second = root['right'], root['left']
    
    best = closest_point(first, point, best)
    
    # Check if we need to search the other subtree
    if abs(point[axis] - root['point'][axis]) ** 2 < get_distance(best, point):
        best = closest_point(second, point, best)
    
    return best

class KDTree:
    """
    K-dimensional tree implementation for efficient nearest neighbor searches.
    
    This implementation provides a simple yet efficient KD-tree data structure
    that can be used for spatial queries like finding nearest neighbors.
    
    Example:
    --------
    >>> points = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    >>> tree = KDTree(points)
    >>> query_point = [6, 5]
    >>> nearest = tree.find_nearest(query_point)
    """
    
    def __init__(self, points):
        """
        Initialize KD-tree with a set of points.
        
        Parameters:
        -----------
        points : list of lists
            List of points where each point is a list of coordinates
        """
        # Convert points to list of lists with float values
        self.points = [[float(x) for x in point] for point in points]
        self.root = build_kdtree(self.points)
    
    def find_nearest(self, query_point):
        """
        Find the nearest point in the tree to the query point.
        
        Parameters:
        -----------
        query_point : list
            Point to find nearest neighbor for
            
        Returns:
        --------
        list
            Nearest point found in the tree
        """
        # Convert query point to list of floats
        query = [float(x) for x in query_point]
        return closest_point(self.root, query)