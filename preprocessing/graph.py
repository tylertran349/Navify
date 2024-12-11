import math

# This module contains the node class and the edge class

class Node:
    def __init__(self, x, y, use_global_coords, building_id):
        self.x = x
        self.y = y
        self.use_global_coords = use_global_coords
        self.building_id = building_id  # ID = "Outdoor" if outdoor

    def __eq__(self, other):
        return isinstance(other, Node) and (self.x, self.y, self.use_global_coords, self.building_id) == (other.x, other.y, other.use_global_coords, other.building_id)

    def __hash__(self):
        return hash((self.x, self.y, self.use_global_coords, self.building_id))

    def __lt__(self, other):
        # Comparison is no longer needed for heapq, as scores are managed separately
        return False

class Edge: 
    def __init__(self, node1, node2, baseline_weight: float ):
        self.node1 = node1
        self.node2 = node2
        self.weight = baseline_weight
        self.baseline_weight = baseline_weight
        
    def __str__(self):
        return f"Edge between {self.node1} and {self.node2} with weight {self.weight} and baseline weight {self.baseline_weight}"  

    def get_nodes(self):
        return (self.node1, self.node2)

    def get_weight(self):
        return self.weight

    def get_baseline_weight(self):
        return self.baseline_weight

    def set_weight(self, new_weight: float):
        self.weight = new_weight

    def set_baseline_weight(self, new_baseline_weight: float):
        self.baseline_weight = new_baseline_weight

    
