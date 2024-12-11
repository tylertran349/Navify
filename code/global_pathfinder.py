from local_pathfinder import AStar
from database import Database

# TotalPathfinder is a class that finds the shortest path through different buildings linked
class TotalPathfinder:
    def __init__(self, database):
        self.database = database
        self.algorithm = AStar()
    
    # Find the shortest path
    def find_shortest_path(self, start_node, goal_node):
        node_sections = self.create_node_sections(start_node, goal_node)
        root = TreeNode(init_node=start_node)
        self.create_tree(root, node_sections)
        leaf_nodes = self.get_leaf_nodes(root)
        shortest_path_node = min(leaf_nodes, key=lambda node: node.time_to_reach_node)
        return shortest_path_node.get_time_to_reach_node(), shortest_path_node.get_path_to_reach_node()

    # Create the nodes that each part of the path will need to go through
    def create_node_sections(self, start_node, goal_node):
        node_sections = [[start_node], [goal_node]]
        if (start_node.building_id != goal_node.building_id):
            if (start_node.use_global_coords == False) and (self.database.is_entrance(start_node) == False):
                indoor_entrances, _ = self.database.get_building_entrances(start_node.building_id)
                node_sections.insert(1, indoor_entrances)
            if  (goal_node.use_global_coords == False) and (self.database.is_entrance(goal_node) == False):
                indoor_entrances, _ = self.database.get_building_entrances(goal_node.building_id)
                node_sections.insert(-1, indoor_entrances)
        return node_sections
        
    # Recursively build the tree
    def create_tree(self, tree_node, node_sections):
        current_depth = tree_node.depth
        if current_depth == len(node_sections) - 1:
            return
        for goal_node in node_sections[current_depth + 1]:
            start_node = tree_node.get_last_path_node()
            # Adjust start_node and goal_node if they have different coordinate systems
            if start_node.use_global_coords != goal_node.use_global_coords:
                if self.database.is_entrance(start_node) and self.database.is_entrance(goal_node):
                    if start_node == self.database.get_corresponding_entrance(goal_node):       
                        child = TreeNode(0, [start_node, goal_node], depth=current_depth + 1)
                        tree_node.add_child(child)
                        self.create_tree(child, node_sections)
                        return
                    if not start_node.use_global_coords:
                        start_node = self.database.get_corresponding_entrance(start_node) 
                    elif not goal_node.use_global_coords:
                        goal_node = self.database.get_corresponding_entrance(goal_node)
                else:       
                    start_node = self.database.get_corresponding_entrance(start_node)
                    goal_node = self.database.get_corresponding_entrance(goal_node)
            elif self.database.is_entrance(start_node) and self.database.is_entrance(goal_node) and not start_node.use_global_coords:
                start_node = self.database.get_corresponding_entrance(start_node)
                goal_node = self.database.get_corresponding_entrance(goal_node)

            search_building = "Outdoor" if (start_node.use_global_coords or goal_node.use_global_coords) else start_node.building_id
            adjacency_list = self.database.get_building_adjacency_list(search_building)
            time, path = self.algorithm.find_path(start_node, goal_node, adjacency_list)
            if path is not None:
                # Handle entrance nodes at the start and end of the path
                if (len(path) > 1) and self.database.is_entrance(path[0]) and (path[0].use_global_coords != node_sections[0][0].use_global_coords) and path[0] == self.database.get_corresponding_entrance(node_sections[0][0]):
                    path.insert(0, self.database.get_corresponding_entrance(path[0]))     
                if len(path) > 1 and self.database.is_entrance(path[-1]) and (path[-1].use_global_coords != node_sections[-1][0].use_global_coords) and path[-1] == self.database.get_corresponding_entrance(node_sections[-1][0]):
                    path.append(self.database.get_corresponding_entrance(path[-1]))
            child = TreeNode(tree_node.get_time_to_reach_node() + time, tree_node.get_path_to_reach_node() + path, depth=current_depth + 1)
            tree_node.add_child(child)
            self.create_tree(child, node_sections)

    # Get a list of all the leaf nodes (paths)
    def get_leaf_nodes(self, node):
        if not node.get_children():
            return [node]
        return [leaf for child in node.get_children() for leaf in self.get_leaf_nodes(child)]
    
# Helper class to find the total path
class TreeNode:
    def __init__(self, time_to_reach_node=0.0, path_to_reach_node=None, depth=0, init_node=None):
        self.children = None
        self.time_to_reach_node = time_to_reach_node
        self.path_to_reach_node = path_to_reach_node if path_to_reach_node is not None else []
        self.depth = depth
        self.last_path_node = init_node if init_node is not None else (self.path_to_reach_node[-1] if self.path_to_reach_node else None)
        
    def add_child(self, child):
        if self.children is None:
            self.children = []
        self.children.append(child)
        return None

    def __str__(self):
        return f"TreeNode with time to reach node: {self.time_to_reach_node}, Path length: {len(self.path_to_reach_node)}"

    def get_children(self):
        return self.children if self.children is not None else []

    def get_time_to_reach_node(self):
        return self.time_to_reach_node

    def get_path_to_reach_node(self):
        return self.path_to_reach_node
    
    def get_last_path_node(self):
        if self.path_to_reach_node:
            return self.path_to_reach_node[-1]
        return self.last_path_node
