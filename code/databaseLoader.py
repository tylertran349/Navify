# filename: databaseInitiator.py

import xml.etree.ElementTree as ET
import math
from database import Database
from graph import Node, Edge

class DatabaseInitiator:
    def __init__(self):
        self.nodes_path = 'Code/data/nodes.osm'
        self.edges_path = 'Code/data/ways.osm'

        # Load nodes
        self.nodes = self.parse_nodes(self.nodes_path)

        # Load edges and create outdoor adjacency list
        self.outdoor_adjacency_list = self.parse_edges(self.edges_path, self.nodes)

        # Keep only the largest connected component
        self.outdoor_adjacency_list = self.filter_largest_component(self.outdoor_adjacency_list)

        # Create the database
        self.database = self.create_database()

        # Optional: Output some information
        # self.print_summary()

    @staticmethod
    def compute_time(node1, node2):
        """
        Computes the walking time between two nodes based on their geographic coordinates.
        """
        R = 6371000  # Radius of the Earth in meters
        lat1 = math.radians(node1.y)
        lon1 = math.radians(node1.x)
        lat2 = math.radians(node2.y)
        lon2 = math.radians(node2.x)
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        a = math.sin(delta_lat / 2) ** 2 + \
            math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        walking_speed = 1.4  # Average walking speed in m/s
        return distance / walking_speed

    def parse_nodes(self, filename):
        """
        Parses the nodes.osm file to extract node information.

        Parameters:
            filename (str): Path to the nodes.osm file.

        Returns:
            dict: Maps node IDs to Node instances.
        """
        tree = ET.parse(filename)
        root = tree.getroot()
        nodes = {}

        for node_elem in root.findall('node'):
            node_id = int(node_elem.get('id'))
            lat = float(node_elem.get('lat'))
            lon = float(node_elem.get('lon'))

            tags = {tag_elem.get('k'): tag_elem.get('v') for tag_elem in node_elem.findall('tag')}
            building_id = tags.get('building', 'Outdoor')

            # Determine if the node uses global coordinates
            use_global_coords = (building_id == 'Outdoor')

            # Create the node
            node = Node(x=lon, y=lat, use_global_coords=use_global_coords, building_id=building_id)

            # Add node to the nodes dictionary
            nodes[node_id] = node

            # **Removed:** Previously, nodes belonging to buildings were added to outdoor_entrance_dict.
            # Now, entrance dictionaries are managed separately.
            # if building_id != 'Outdoor':
            #     outdoor_entrance_dict.setdefault(building_id, []).append(node)

        return nodes

    def parse_edges(self, filename, nodes):
        """
        Parses the ways.osm file to extract edge information and build the outdoor adjacency list.

        Parameters:
            filename (str): Path to the ways.osm file.
            nodes (dict): Maps node IDs to Node instances.

        Returns:
            dict: Outdoor adjacency list mapping Node instances to lists of (neighbor_node, Edge) tuples.
        """
        tree = ET.parse(filename)
        root = tree.getroot()
        adjacency_list = {}  # Single outdoor adjacency list

        for way_elem in root.findall('way'):
            node_refs = [int(nd_elem.get('ref')) for nd_elem in way_elem.findall('nd')]

            nodes_in_way = [nodes.get(node_id) for node_id in node_refs]
            for i in range(len(nodes_in_way) - 1):
                node1 = nodes_in_way[i]
                node2 = nodes_in_way[i + 1]
                if node1 is None or node2 is None:
                    continue

                baseline_weight = self.compute_time(node1, node2)
                edge = Edge(node1=node1, node2=node2, baseline_weight=baseline_weight)

                # For undirected graph, add edges in both directions
                adjacency_list.setdefault(node1, []).append((node2, edge))
                adjacency_list.setdefault(node2, []).append((node1, edge))

        return adjacency_list

    def dfs(self, node, adjacency_list, visited, component_nodes):
        """
        Performs Depth-First Search to find all nodes in the connected component.

        Parameters:
            node (Node): The starting node for DFS.
            adjacency_list (dict): The adjacency list.
            visited (set): Set of visited nodes.
            component_nodes (list): List to store nodes in the current component.

        Returns:
            int: Size of the connected component.
        """
        stack = [node]
        visited.add(node)
        size = 0

        while stack:
            current_node = stack.pop()
            component_nodes.append(current_node)
            size += 1

            for neighbor, _ in adjacency_list.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        return size

    def filter_largest_component(self, adjacency_list):
        """
        Filters the adjacency list to include only the largest connected component.
        """
        visited = set()
        largest_component_nodes = []
        largest_component_size = 0

        nodes = list(adjacency_list.keys())
        for node in nodes:
            if node not in visited:
                component_nodes = []
                component_size = self.dfs(node, adjacency_list, visited, component_nodes)

                if component_size > largest_component_size:
                    largest_component_size = component_size
                    largest_component_nodes = component_nodes

        # Build the filtered adjacency list for the largest component
        largest_component_set = set(largest_component_nodes)
        filtered_adjacency_list = {}
        for node in largest_component_set:
            filtered_adjacency_list[node] = [
                (neighbor, edge) for neighbor, edge in adjacency_list[node] if neighbor in largest_component_set
            ]

        return filtered_adjacency_list

    def create_database(self):
        """
        Initializes the Database with the outdoor adjacency list and empty entrance dictionaries.

        Returns:
            Database: An instance of the Database class.
        """
        # Initialize the Database with the outdoor adjacency list and empty entrance dictionaries
        database = Database(
            outdoor_adjacency_list=self.outdoor_adjacency_list,
            indoor_entrance_dict={},   # To be populated later
            outdoor_entrance_dict={}   # To be populated later
        )

        return database

    def print_summary(self):
        """
        Prints a summary of the loaded data.
        """
        print("Database has been successfully created.")
        total_nodes = len(self.database.outdoor_adjacency_list)
        total_edges = sum(len(neighbors) for neighbors in self.database.outdoor_adjacency_list.values()) // 2  # Divide by 2 for undirected graph
        print(f"Total nodes in the largest component: {total_nodes}")
        print(f"Total edges in the largest component: {total_edges}")
        print(f"Number of buildings with entrances: {len(self.database.outdoor_entrance_dict)}")

    def get_database(self):
        """
        Returns the initialized Database instance.

        Returns:
            Database: The initialized Database.
        """
        return self.database


# Example usage
if __name__ == "__main__":
    initiator = DatabaseInitiator()
    db = initiator.get_database()
    initiator.print_summary()
