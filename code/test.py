from global_pathfinder import TotalPathfinder
from graph import Node, Edge
from database import Database
from local_pathfinder import AStar
from databaseLoader import DatabaseInitiator
import random
import time
from weightCalculator import WeightCalculator
#from imageCropper import crop_first_column_trim_sides_image, crop_to_content_image, process_image_array
from nodeInator import create_adjacency_list
import weatherDataRetriever
import cv2
import numpy as np
import pickle
import joblib
from entranceProjector import project_entrances

# node1 = Node(0, 0, False, "MU")
# node3 = Node(1, 1, False, "MU") 
# node2 = Node(1, 0, False, "MU") # Local coordinates for MU
# node4 = Node(0, 0, True, "MU") # Global coordinates for MU
# node5 = Node(1, 1, True, "Death Star") # Global coordinates for Death Star
# node6 = Node(1, 0, True, "Outdoor")
# edge1 = Edge(node1, node2, 3)
# edge1a = Edge(node2, node1, 3)
# edge2 = Edge(node1, node3, 9)
# edge2a = Edge(node3, node1, 9)
# edge3 = Edge(node2, node3, 5)
# edge3a = Edge(node3, node2, 5)
# edge4 = Edge(node4, node6, 10)
# edge4a = Edge(node6, node4, 10)
# edge5 = Edge(node5, node6, 1)
# edge5a = Edge(node6, node5, 1)

# node7 = Node(1, 0, False, "Death Star") # Local coordinates for Death Star
# node8 = Node(0, 0, False, "Death Star")
# node9 = Node(0, 1, False, "Death Star")
# edge6 = Edge(node7, node8, 7)
# edge6a = Edge(node8, node7, 7)  
# edge7 = Edge(node8, node9, 2)
# edge7a = Edge(node9, node8, 2)
# edge8 = Edge(node7, node9, 10)
# edge8a = Edge(node9, node7, 10)

# node_dict = {"MU": [node1, node2, node3], 
#               "Outdoor": [node4, node5, node6],
#               "Death Star": [node7, node8, node9]}

# edge_dict = {"MU": [edge1, edge2, edge3, edge1a, edge2a, edge3a], 
#               "Outdoor": [edge4, edge5, edge4a, edge5a],
#               "Death Star": [edge6, edge7, edge8, edge6a, edge7a, edge8a]}

# indoor_entrance_dict = {"MU": [node2],
#                         "Death Star": [node7]}

# outdoor_entrance_dict = {"MU": [node4],
#                          "Death Star": [node5]}

# adjlist = {node1: [(node3, edge2), (node2, edge1)],
#            node2: [(node1, edge1a), (node3, edge3)],
#            node3: [(node1, edge2a), (node2, edge3a)],
#            }

#algorithm = AStar()

#print(algorithm.find_path(node1, node3, adjlist))
#data = Database(node_dict, edge_dict, indoor_entrance_dict, outdoor_entrance_dict)
#algorithm = TotalPathfinder(data)

# print("Node 1:", node1)
# print("Node 2:", node2)
# print("Node 3:", node3)
# print("Node 4:", node4)
# print("Node 5:", node5)
# print("Node 6:", node6)
# print("Node 7:", node7)
# print("Node 8:", node8)
# print("Node 9:", node9)

#nodes = [node1, node2, node3, node4, node4, node6, node7, node8, node9]
#for i in range(len(nodes)):
    #for j in range(len(nodes)):
        #print(algorithm.find_shortest_path(nodes[i], nodes[j]))

def preprocess_image_to_binary(image):
    """
    Converts an image to a binary numpy array and inverts the colors.
    Assumes the image is grayscale or will be converted to grayscale.

    Parameters:
    - image (numpy.ndarray): The input image read by cv2.imread.

    Returns:
    - binary_array (numpy.ndarray): Binary array with 255 for walkable pixels and 0 for obstacles.
    """
    # Convert to grayscale if the image is not already in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a binary array
    _, binary_array = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Invert the binary array
    inverted_binary_array = cv2.bitwise_not(binary_array)

    return inverted_binary_array


# # Initialize database
# data = DatabaseInitiator()
# database = data.get_database()

# # Measure and print time to add adjacency lists for each building
# def time_add_adjacency_list(database, adjacency_list, building_name):
#     start_time = time.time()
#     database.add_adjacency_list(adjacency_list, building_name)
#     end_time = time.time()
#     print(f"Adding adjacency list for '{building_name}' took {end_time - start_time:.2f} seconds.")

# # Add adjacency lists and measure time
# time_add_adjacency_list(database, MU_adj_list, "Memorial Union")
# time_add_adjacency_list(database, Walker_adj_list, "Walker Hall")


# # Save the database to a pickle file
# pickle_filepath = "database.pkl"
# with open(pickle_filepath, "wb") as f:
#     pickle.dump(database, f)
#     print(f"Database saved to '{pickle_filepath}'.")

# # Benchmark nearest node search
# start = time.time()
# n = database.find_nearest_node("Memorial Union", 0.1, 0.2)
# end = time.time()
# print(f"Nearest node search took {end - start:.4f} seconds.")
# print(f"Nearest node: {n}, Coordinates: ({n.x}, {n.y})")

# filename: main.py

def main():
    # Initialize the database
    data = DatabaseInitiator()
    database = data.get_database()

    buildings = [
        {
            "name": "Memorial Union",
            "floor_plan_image_path": "Code/CroppedImages/MUFirstFloor.png"
        },
        {
            "name": "Walker Hall",
            "floor_plan_image_path": "Code/CroppedImages/WalkerHallFirstFloor.png"
        }
    ]

    for building in buildings:
        building_name = building["name"]
        floor_plan_image_path = building["floor_plan_image_path"]
        try:
            project_entrances(database, building_name, floor_plan_image_path)
            print(f"Successfully processed entrances for '{building_name}'.")
        except Exception as e:
            print(f"Error processing entrances for '{building_name}': {e}")

    try:
        database.scale_indoor_entrances_for_building("Walker Hall", 0.6)
        print(f"Scaled indoor entrances for 'Walker Hall' by a factor of 0.6.")
    except ValueError as ve:
        print(f"Error scaling entrances: {ve}")

    # Save the database object using pickle
    output_file = "database.pkl"
    try:
        with open(output_file, "wb") as f:
            pickle.dump(database, f)
        print(f"Database saved successfully to '{output_file}'.")
    except Exception as e:
        print(f"Error saving database: {e}")

if __name__ == "__main__":
    main()







# a = time.time()
# updator = WeightCalculator(database)
# b = time.time()
# updator.calculate_weights(True, True)
# c = time.time()

# print("Time to make weight calculator: ", b-a)
# print("Time to run weight calculator: ", c-b)
# #total_alg = TotalPathfinder(database)
# start_node = list(database.adjacency_dict["Outdoor"].keys())[0]
# goal_node = list(database.adjacency_dict["Outdoor"].keys())[1000]

# algorithm = AStar()
# start = time.time()
# algorithm.find_path(start_node, goal_node, database.adjacency_dict["Outdoor"])
# print("Pathfinding time: ", time.time()-start)
# #t, path = total_alg.find_shortest_path(start_node, goal_node)
# #######################################################################
# list_of_image_paths = []

# # Example usage
# # Read an image from a file (for demonstration purposes)
# input_image = cv2.imread("code/MUSecondFloor.png")
# list_of_image_paths.append(input_image)
# input_image = cv2.imread("code/ARCFirstFloor.png")
# list_of_image_paths.append(input_image)
# input_image = cv2.imread("code/DeathStarFirstFloor.png")
# list_of_image_paths.append(input_image)
# input_image = cv2.imread("code/GeidtHall.png")
# list_of_image_paths.append(input_image)
# input_image = cv2.imread("code/LibraryBasement.png")
# list_of_image_paths.append(input_image)

# for input_image in list_of_image_paths:


# # Check if the image was successfully loaded
#     if input_image is None:
#         raise FileNotFoundError("Input image not found.")

#     # Process the image array to be cropped
#     output_image = process_image_array(input_image) 
#     # TODO:
#     output_image_to_display = output_image
    
#     # convert to binary matrix (substitute for door removal function, which returns the matrix with removed doors, scaled down, and also the scale factor which we will use to scale up the displayed path)
#     def convert_to_binary_matrix(image):
#         """
#         Converts a BGR image to a binary image using grayscale conversion and thresholding.
        
#         Args:
#             image (numpy.ndarray): Input BGR image.
            
#         Returns:
#             numpy.ndarray: Binary image with pixels as 0 or 255.
#         """
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # Apply binary thresholding (invert the binary image)
#         _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#         return binary

#     output_image = convert_to_binary_matrix(output_image)

#     # shrink down output image to lowest possible resolution before making into adj list


#     # Call the function
#     start = time.time()
#     # TODO: Insert door removal function above so we can update create_adjacency_list arguments to not be hardcoded for testing
#     adjacency_list = create_adjacency_list(output_image, 1, "Test Building")
#     print(time.time()-start)

#     # run astar on adj list
#     start_node = list(adjacency_list.keys())[0]
#     goal_node = list(adjacency_list.keys())[1000]

#     # save and draw path from astar

#     # scale up path visualization using scale factor from door removal function second return value


#     print("image testing done")