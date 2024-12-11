## Creates an adjacency list from input image, which should already be processed (cropping and door removal)
# adjacency list should be used to run BFS or A*
# input binary array should only have 1's and 0's; if there is an error then we need to add error checking

from graph import Node, Edge
import numpy as np

def create_adjacency_list(image_array, buildingName="Test Building"):
    """
    Creates an adjacency list from a binary image array.

    Parameters:
    - image_array (np.ndarray): 2D numpy array with 0s and 255s representing walkable and non-walkable pixels.
    - buildingName (str): Name of the building. Defaults to "Outdoor".

    Returns:
    - adjacency_list (dict): Dictionary mapping Node instances to lists of (neighbor_node, Edge) tuples.
    """

    # Validate input array
    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array must be a numpy ndarray.")
    if image_array.ndim != 2:
        raise ValueError("image_array must be a 2D numpy array.")
    unique_values = np.unique(image_array)
    if not set(unique_values).issubset({0, 255}):
        raise ValueError("image_array must contain only 0 and 255 values.")

    adjacency_list = {}
    node_mapping = {}  # Maps positions (x, y) to Node instances
    rows, cols = image_array.shape

    for row in range(rows):  # row index corresponds to y-coordinate
        for col in range(cols):  # col index corresponds to x-coordinate
            pixel = image_array[row][col]
            if pixel == 0:
                # In image coordinates, x corresponds to column index, y corresponds to row index
                x = col  # x-coordinate (increases to the right)
                y = row  # y-coordinate (increases downward)

                # Convert to relative coordinates
                rel_x = x / cols  # Relative X-coordinate (0.0 to 1.0)
                rel_y = y / rows  # Relative Y-coordinate (0.0 to 1.0)

                # Create a new Node with relative coordinates (rel_x, rel_y)
                current_node = Node(rel_x, rel_y, use_global_coords=False, building_id=buildingName)

                # Store the Node in node_mapping using (x, y) as the key
                node_mapping[(x, y)] = current_node

                # Initialize adjacency list for the current Node
                adjacency_list[current_node] = []

                # Check the neighboring positions: left, up
                neighbor_coords = [
                    (x - 1, y),     # Left
                    (x,     y - 1), # Up
                ]

                for nx, ny in neighbor_coords:
                    if 0 <= nx < cols and 0 <= ny < rows:
                        neighbor_node = node_mapping.get((nx, ny))
                        if neighbor_node:
                            # Create an Edge between current_node and neighbor_node
                            edge = Edge(current_node, neighbor_node, baseline_weight=1.0)

                            # Add the neighbor and edge to the current node's adjacency list
                            adjacency_list[current_node].append((neighbor_node, edge))

                            # Also add the current node and edge to the neighbor's adjacency list
                            adjacency_list[neighbor_node].append((current_node, edge))
            elif pixel == 255:
                # Skip non-walkable pixels
                continue
            else:
                # Raise an exception for invalid pixel values
                raise ValueError(f"Invalid pixel value at position ({row}, {col}): {pixel}")

    return adjacency_list

# # Example binary image array (2D numpy array)
# image_array = np.array([
#     [0, 255, 0],
#     [0, 0, 255],
#     [255, 0, 0]
# ])

# # Call the function
# adjacency_list = create_adjacency_list(image_array)

# # Print the adjacency list
# for node, edges in adjacency_list.items():
#     print(f"Node at ({node.x:.2f}, {node.y:.2f}):")
#     for neighbor, edge in edges:
#         print(f"  Connected to ({neighbor.x:.2f}, {neighbor.y:.2f}) via Edge with baseline weight {edge.baseline_weight}")
