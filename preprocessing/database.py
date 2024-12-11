# filename: database.py

import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points


class Database:
    def __init__(
        self,
        outdoor_adjacency_list=None,
        indoor_entrance_dict=None,
        outdoor_entrance_dict=None,
    ):
        """
        Initializes the Database with outdoor adjacency list and entrance dictionaries.
        
        Parameters:
        - outdoor_adjacency_list (dict): Maps Node instances to lists of (neighbor_node, Edge) tuples.
        - indoor_entrance_dict (dict): Maps building names to their indoor entrances.
        - outdoor_entrance_dict (dict): Maps building names to their outdoor entrances.
        """
        self.outdoor_adjacency_list = outdoor_adjacency_list if outdoor_adjacency_list is not None else {}
        self.indoor_entrance_dict = indoor_entrance_dict if indoor_entrance_dict is not None else {}
        self.outdoor_entrance_dict = outdoor_entrance_dict if outdoor_entrance_dict is not None else {}

    def add_outdoor_adjacency_list(self, adj_list):
        """
        Adds entries to the outdoor adjacency list.
        
        Parameters:
        - adj_list (dict): Maps Node instances to lists of (neighbor_node, Edge) tuples.
        """
        for node, neighbors in adj_list.items():
            if node in self.outdoor_adjacency_list:
                self.outdoor_adjacency_list[node].extend(neighbors)
            else:
                self.outdoor_adjacency_list[node] = neighbors

    def get_outdoor_adjacency_list(self):
        """
        Returns the outdoor adjacency list.
        
        Returns:
        - dict: The outdoor adjacency list.
        """
        return self.outdoor_adjacency_list

    def add_indoor_entrances(self, building_name, entrances):
        """
        Adds indoor entrances for a given building.
        
        Parameters:
        - building_name (str): The name of the building.
        - entrances (list of tuples): List of (x, y) coordinates for indoor entrances.
        """
        self.indoor_entrance_dict[building_name] = entrances

    def add_outdoor_entrances(self, building_name, entrances):
        """
        Adds outdoor entrances for a given building.
        
        Parameters:
        - building_name (str): The name of the building.
        - entrances (list of tuples): List of (longitude, latitude) coordinates for outdoor entrances.
        """
        self.outdoor_entrance_dict[building_name] = entrances

    def get_building_entrances(self, building_name):
        """
        Returns the indoor and outdoor entrances for a specific building.
        
        Parameters:
        - building_name (str): The name of the building.
        
        Returns:
        - tuple: (indoor_entrances, outdoor_entrances)
        """
        indoor = self.indoor_entrance_dict.get(building_name, [])
        outdoor = self.outdoor_entrance_dict.get(building_name, [])
        return indoor, outdoor

    def is_entrance(self, node, building_name):
        """
        Determines if a node is an entrance for a specific building.
        
        Parameters:
        - node (Node): The node to check.
        - building_name (str): The name of the building.
        
        Returns:
        - bool: True if the node is an entrance, False otherwise.
        """
        indoor_entrances = self.indoor_entrance_dict.get(building_name, [])
        outdoor_entrances = self.outdoor_entrance_dict.get(building_name, [])
        return node in indoor_entrances or node in outdoor_entrances
    
    def scale_indoor_entrances_for_building(self, building_name, scale_factor):
        """
        Scales the indoor entrance coordinates for a specific building by the given scale factor.
        The new coordinates are rounded to the nearest integer pixel values.
        
        Parameters:
            building_name (str): The name of the building whose entrances are to be scaled.
            scale_factor (float): The factor by which to scale the entrance coordinates.
        
        Raises:
            ValueError: If the specified building does not exist in the indoor_entrance_dict.
        """
        if building_name not in self.indoor_entrance_dict:
            raise ValueError(f"Building '{building_name}' does not have any indoor entrances in the database.")
        
        original_entrances = self.indoor_entrance_dict[building_name]
        scaled_entrances = []
        
        for (x, y) in original_entrances:
            scaled_x = round(x * scale_factor)
            scaled_y = round(y * scale_factor)
            scaled_entrances.append((scaled_x, scaled_y))
            #print(f"Scaled entrance from ({x}, {y}) to ({scaled_x}, {scaled_y}) for building '{building_name}'.")
        
        # Update the indoor_entrance_dict with the scaled entrances
        self.indoor_entrance_dict[building_name] = scaled_entrances

    def get_edges(self):
        """
        Retrieves all unique edges from the outdoor adjacency list.
        
        Returns:
        - list: A list of unique Edge instances.
        """
        edges_set = set()
        edges_list = []

        for node, neighbors in self.outdoor_adjacency_list.items():
            for neighbor, edge in neighbors:
                # Create a tuple representation of the edge to check uniqueness
                edge_tuple = (min(edge.node1, edge.node2, key=lambda n: (n.x, n.y)),
                              max(edge.node1, edge.node2, key=lambda n: (n.x, n.y)))

                if edge_tuple not in edges_set:
                    edges_set.add(edge_tuple)
                    edges_list.append(edge)

        return edges_list

    def get_all_edges(self):
        """
        Retrieves all unique edges from the outdoor adjacency list.
        
        Returns:
        - list: A list of unique Edge instances.
        """
        return self.get_edges()

    def get_edges_within_polygons(self, polygons):
        """
        Given a list of polygons (each polygon is a list of [latitude, longitude] coordinates),
        return a list of lists, where each inner list contains the edges that are within
        the corresponding polygon.
        An edge is considered within a polygon if both its nodes are inside the polygon.
        
        Parameters:
            polygons (List[List[List[float]]]): A list of polygons, each defined by a list of [latitude, longitude] coordinates.
        
        Returns:
            List[List[Edge]]: A list where each element is a list of Edge instances within the corresponding polygon.
        """
        outdoor_edges = self.get_edges()
        result = []

        for polygon_coords in polygons:
            if not polygon_coords:
                result.append([])
                continue

            # Create a Shapely Polygon object
            polygon = Polygon(polygon_coords)

            inside_edges = []
            for edge in outdoor_edges:
                point1 = Point(edge.node1.x, edge.node1.y)
                point2 = Point(edge.node2.x, edge.node2.y)
                if polygon.contains(point1) and polygon.contains(point2):
                    inside_edges.append(edge)

            result.append(inside_edges)

        return result

    def find_nearest_node(self, building_name, x, y, early_exit_radius=0.01):
        """
        Finds the nearest node to the given coordinates within the outdoor adjacency list.
        If building_name is not "Outdoor", it considers indoor entrances as well.
        
        Parameters:
        - building_name (str): The name of the building. Use "Outdoor" for outdoor nodes.
        - x (float): Relative x-coordinate (0.0 to 1.0) for indoor buildings or global longitude for 'Outdoor'.
        - y (float): Relative y-coordinate (0.0 to 1.0) for indoor buildings or global latitude for 'Outdoor'.
        - early_exit_radius (float): The radius within which to stop searching if a node is found (applies only to non-Outdoor).
        
        Returns:
        - nearest_node (Node): The Node instance closest to the provided coordinates.
        
        Raises:
        - ValueError: If input format is incorrect or building does not exist.
        """
            # Input Validation
        if not isinstance(building_name, str):
            raise ValueError("Building name must be a string.")

        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("Coordinates x and y must be numbers (int or float).")

        # Determine the search space
        if building_name == "Outdoor":
            adjacency_list = self.outdoor_adjacency_list
            apply_early_exit = False
        else:
            # For indoor buildings, assume there is a separate adjacency list if needed
            # Since only outdoor_adjacency_list is maintained, this part may need to be adjusted based on actual data
            adjacency_list = self.outdoor_adjacency_list  # Placeholder
            apply_early_exit = True

        nodes = list(adjacency_list.keys())

        nearest_node = None
        min_distance = float('inf')

        for node in nodes:
            distance = math.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
                if apply_early_exit and distance <= early_exit_radius:
                    #print(f"Nearest Node Found within {early_exit_radius} radius: ({nearest_node.x}, {nearest_node.y}), Distance: {min_distance:.4f}")
                    return nearest_node

        if nearest_node:
            #print(f"Nearest Node Found: ({nearest_node.x}, {nearest_node.y}), Distance: {min_distance:.4f}, Node: {nearest_node}")
            return nearest_node
        else:
            raise ValueError("No nodes found in the adjacency list.")

    def plot_outdoor(self, step_delay=0.00001, show_incrementally=True, path_nodes=None):
        """
        Plot the outdoor adjacency list and highlight a specified path.
        
        Parameters:
        - step_delay (float): Delay in seconds between plotting steps (for iterative visualization).
        - show_incrementally (bool): Whether to show the plot incrementally.
        - path_nodes (list): A list of nodes to be highlighted in green (representing the shortest path).
        """
        outdoor_adjacency = self.outdoor_adjacency_list
        path_nodes = set(path_nodes or [])

        # Initialize the plot
        plt.figure(figsize=(12, 8))
        plt.title("Outdoor Network with Shortest Path Highlighted")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        # Extract node coordinates
        nodes = list(outdoor_adjacency.keys())
        node_x = [node.x for node in nodes]
        node_y = [node.y for node in nodes]

        # Plot all nodes
        plt.scatter(node_x, node_y, s=10, color='blue', label='Nodes')

        # Highlight the nodes in the shortest path (green)
        if path_nodes:
            path_x = [node.x for node in nodes if node in path_nodes]
            path_y = [node.y for node in nodes if node in path_nodes]
            plt.scatter(path_x, path_y, s=50, color='green', label='Shortest Path Nodes')

        # Keep track of plotted edges to avoid duplicates
        plotted_edges = set()

        # Iteratively plot the edges
        for node, neighbors in outdoor_adjacency.items():
            for neighbor, edge in neighbors:
                # Create a unique identifier for the edge to prevent duplicates
                edge_id = frozenset([node, neighbor])
                if edge_id in plotted_edges:
                    continue
                plotted_edges.add(edge_id)

                plt.plot([node.x, neighbor.x], [node.y, neighbor.y], color='gray', linewidth=0.5)

                if show_incrementally:
                    plt.pause(step_delay)

        # Display the plot
        plt.legend()
        plt.grid()
        plt.show()
