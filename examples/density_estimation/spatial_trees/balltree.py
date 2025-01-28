def euclidean_distance(x, y):
    return sum((a - b) ** 2 for a, b in zip(x, y)) ** 0.5

def compute_centroid(points):
    n_dims = len(points[0])
    n_points = len(points)
    centroid = [0.0] * n_dims
    for point in points:
        for i in range(n_dims):
            centroid[i] += point[i]
    return [x / n_points for x in centroid]

def find_furthest_point(points, centroid):
    max_dist = -1
    furthest_idx = 0
    for i, point in enumerate(points):
        dist = euclidean_distance(point, centroid)
        if dist > max_dist:
            max_dist = dist
            furthest_idx = i
    return furthest_idx

def build_ball_tree(points, leaf_size=40):
    n_points = len(points)
    if n_points <= leaf_size:
        centroid = compute_centroid(points)
        radius = 0.0
        for point in points:
            dist = euclidean_distance(point, centroid)
            if dist > radius:
                radius = dist
        return {
            'centroid': centroid,
            'radius': radius,
            'points': points,
            'is_leaf': True,
            'left': None,
            'right': None
        }
    
    # Find the centroid
    centroid = compute_centroid(points)
    
    # Find two points furthest apart to split the data
    idx1 = find_furthest_point(points, centroid)
    idx2 = find_furthest_point(points, points[idx1])
    
    # Split points based on distance to the two furthest points
    left_points = []
    right_points = []
    for point in points:
        dist1 = euclidean_distance(point, points[idx1])
        dist2 = euclidean_distance(point, points[idx2])
        if dist1 <= dist2:
            left_points.append(point)
        else:
            right_points.append(point)
    
    # Handle edge cases where all points end up in one group
    if len(left_points) == 0:
        mid = len(right_points) // 2
        left_points = right_points[:mid]
        right_points = right_points[mid:]
    elif len(right_points) == 0:
        mid = len(left_points) // 2
        right_points = left_points[mid:]
        left_points = left_points[:mid]
    
    # Create node
    node = {
        'centroid': centroid,
        'radius': max(euclidean_distance(p, centroid) for p in points),
        'points': points,
        'is_leaf': False,
        'left': build_ball_tree(left_points, leaf_size),
        'right': build_ball_tree(right_points, leaf_size)
    }
    return node

def query_ball_tree(node, query_point, best_dist, best_point):
    dist_to_centroid = euclidean_distance(query_point, node['centroid'])
    
    if dist_to_centroid - node['radius'] > best_dist:
        return best_dist, best_point
    
    if node['is_leaf']:
        for point in node['points']:
            dist = euclidean_distance(query_point, point)
            if dist < best_dist:
                best_dist = dist
                best_point = point
        return best_dist, best_point
    
    # Recursively search children
    best_dist, best_point = query_ball_tree(node['left'], query_point, best_dist, best_point)
    best_dist, best_point = query_ball_tree(node['right'], query_point, best_dist, best_point)
    
    return best_dist, best_point

class BallTree:
    """
    Ball tree implementation for efficient nearest neighbor searches.
    
    Ball trees partition space into a nested set of hyperspheres, which can be more
    efficient than KD-trees for high-dimensional data.
    
    Example:
    --------
    >>> points = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    >>> tree = BallTree(points)
    >>> query_point = [6, 5]
    >>> nearest = tree.find_nearest(query_point)
    """
    
    def __init__(self, points, leaf_size=40):
        """
        Initialize Ball tree with a set of points.
        
        Parameters:
        -----------
        points : list of lists
            List of points where each point is a list of coordinates
        leaf_size : int, optional (default=40)
            Number of points at which to stop splitting
        """
        # Convert points to list of lists with float values
        self.points = [[float(x) for x in point] for point in points]
        self.leaf_size = leaf_size
        self.root = build_ball_tree(self.points, leaf_size)
    
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
        inf = float('inf')
        best_dist, best_point = query_ball_tree(self.root, query, inf, self.points[0])
        return best_point