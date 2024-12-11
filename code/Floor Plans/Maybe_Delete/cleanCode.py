# INIT VERSION (RANDOM 2 NODES TO CONNECT WITH THE SHORTEST PATH)
#
# import cv2
# import numpy as np
# import random
# from collections import deque

# # Load the image
# image = cv2.imread('Simple Lib.png')

# # Check if the image is loaded properly
# if image is None:
#     print("Failed to load the image. Check the file path.")
# else:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)

#     # Initialize Line Segment Detector
#     lsd = cv2.createLineSegmentDetector(0)

#     # Detect lines in the image
#     lines = lsd.detect(edges)[0]

#     # Create a binary matrix based on the detected lines
#     matrix = np.zeros_like(gray)
#     for line in lines:
#         x1, y1, x2, y2 = map(int, line[0])
#         cv2.line(matrix, (x1, y1), (x2, y2), 1, 2)

#     # Draw the detected lines on the original image and color them green
#     for line in lines:
#         x1, y1, x2, y2 = map(int, line[0])
#         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Function to place two random 1s far apart
#     def place_random_ones(matrix):
#         # x1, y1 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
#         # while matrix[x1, y1] == 1:
#         #     x1, y1 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
#         # matrix[x1, y1] = 1
#         x1, y1 = 320, 9

#         # while True:
#         #     x2, y2 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
#         #     if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 15 and matrix[x2, y2] == 0:
#         #         matrix[x2, y2] = 1
#         #         break
#         x2, y2 = 92, 756

#         return (x1, y1), (x2, y2)

#     # Place the 1s
#     start, end = place_random_ones(matrix)

#     # Function to run BFS to connect the 1s and draw the path
#     def bfs_connect(matrix, start, end):
#         directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
#         queue = deque([(start, [start])])
#         visited = set()

#         while queue:
#             (current, path) = queue.popleft()
#             if current == end:
#                 for k in range(len(path) - 1):
#                     cv2.line(image, (path[k][1], path[k][0]), (path[k + 1][1], path[k + 1][0]), (255, 0, 0), 2)
#                 cv2.imshow('Detected Lines with BFS Path', image)
#                 cv2.waitKey(50)  # Visualize each step
#                 break
#             if current not in visited:
#                 visited.add(current)
#                 for d in directions:
#                     next_node = (current[0] + d[0], current[1] + d[1])
#                     if 0 <= next_node[0] < matrix.shape[0] and 0 <= next_node[1] < matrix.shape[1]:
#                         if matrix[next_node[0], next_node[1]] == 0 or next_node == end:
#                             if next_node not in visited:
#                                 queue.append((next_node, path + [next_node]))

#     # Run BFS to connect the 1s
#     bfs_connect(matrix, start, end)

#     cv2.imshow('Final Path', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



# NOTES TO IMPROVE THE ALGO
#
#     # Function to place two random 1s far apart
#     # NO RANDOM NOW: Node

#     # 1 - detect walls
#     # 2 - MC to put node where there are no walls
#     # 3 - cut every 20 nodes
#     # 4 - output the x,y of all nodes

#     # 5 - write 1-3 x,y exits foreach building
#     # 6 - write 1-2 x,y stairs foreach building

#     # 7 -


#     #   Outside the building: Node 1 - initial/final, Node 2 or more - all exits
#     #   Inside: Node 1 - init, Node 2 - final
#         # x1, y1 = 0, 0
#         #x2, y2 =





# # GOOD MONTE CARLO FOR NODE PLACEMENT
#
# import cv2
# import numpy as np
# import random
# from collections import deque

# def load_image(file_path):
#     # Load the image
#     image = cv2.imread(file_path)
#     if image is None:
#         raise ValueError("Failed to load the image. Check the file path.")
#     return image

# def convert_to_binary_matrix(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply a binary threshold to get a binary image
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#     # Convert binary image to binary matrix (1s and 0s)
#     binary_matrix = binary // 255
#     return binary_matrix

# def detect_walls_with_lsd(image):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Apply Gaussian Blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Detect edges using Canny
#     edges = cv2.Canny(blurred, 50, 150)
#     # Initialize Line Segment Detector
#     lsd = cv2.createLineSegmentDetector(0)
#     # Detect lines in the image
#     lines = lsd.detect(edges)[0]
#     return lines

# def color_walls(image, lines):
#     # Color the detected walls green with thinner lines
#     colored_image = image.copy()
#     for line in lines:
#         x1, y1, x2, y2 = map(int, line[0])
#         cv2.line(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     return colored_image

# def place_red_nodes(colored_image, binary_matrix, num_nodes=1000):
#     # Place red nodes using Monte Carlo method in the rooms (areas with 0s)
#     nodes = []
#     for _ in range(num_nodes):
#         while True:
#             x, y = random.randint(10, binary_matrix.shape[0] - 11), random.randint(10, binary_matrix.shape[1] - 11)
#             if binary_matrix[x, y] == 0:
#                 wall_nearby = False
#                 for i in range(x-7, x+8):
#                     for j in range(y-7, y+8):
#                         if binary_matrix[i, j] == 1:
#                             wall_nearby = True
#                             break
#                     if wall_nearby:
#                         break
#                 if not wall_nearby:
#                     nodes.append((x, y))
#                     cv2.circle(colored_image, (y, x), 2, (0, 0, 255), -1)  # BGR for red
#                     break
#     return nodes, colored_image

# def mark_exits(image, exits):
#     # Mark exits with purple nodes
#     for (x, y) in exits:
#         cv2.circle(image, (y, x), 5, (255, 0, 255), -1)  # BGR for purple
#     return image

# # Function to run BFS to connect the 1s and draw the path
# def bfs_connect(matrix, start, end):
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
#     queue = deque([(start, [start])])
#     visited = set()

#     while queue:
#         (current, path) = queue.popleft()
#         if current == end:
#             for k in range(len(path) - 1):
#                 cv2.line(image, (path[k][1], path[k][0]), (path[k + 1][1], path[k + 1][0]), (255, 0, 0), 2)
#             cv2.imshow('Detected Lines with BFS Path', image)
#             cv2.waitKey(50)  # Visualize each step
#             break
#         if current not in visited:
#             visited.add(current)
#             for d in directions:
#                 next_node = (current[0] + d[0], current[1] + d[1])
#                 if 0 <= next_node[0] < matrix.shape[0] and 0 <= next_node[1] < matrix.shape[1]:
#                     if matrix[next_node[0], next_node[1]] == 0 or next_node == end:
#                         if next_node not in visited:
#                             queue.append((next_node, path + [next_node]))

# def main():
#     file_path = 'Simple Lib.png'
    
#     # Step 1: Load the image
#     image = load_image(file_path)
    
#     # Step 2: Detect walls using LSD
#     lines = detect_walls_with_lsd(image)
    
#     # Step 3: Color the walls green
#     colored_image = color_walls(image, lines)

#     # Step 4: Mark the exits with purple nodes
#     exits = [(324, 9), (321, 263), (62, 714)]
#     exitt = [(324, 9)]
#     colored_image = mark_exits(colored_image, exits)
    
#     # Step 5: Convert to binary matrix for node placement
#     binary_matrix = convert_to_binary_matrix(colored_image)
    
#     # Step 6: Place red nodes in the rooms using Monte Carlo method
#     nodes, output_image = place_red_nodes(colored_image, binary_matrix)

#     # Step 7: Mark the starting node with a brown circle
#     start_node = (83, 75)
#     cv2.circle(output_image, (start_node[1], start_node[0]), 5, (42, 42, 165), -1)  # BGR for brown

#     # # Step 6: Output the list of nodes location of x, y in pixels
#     # print("Nodes locations (x, y):")
#     # for node in nodes:
#     #     print(node)

#     bfs_connect(binary_matrix, start_node, exitt)
    
#     # Display the final image
#     cv2.imshow('Output Image with Walls, Nodes, and Exits', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()





# SHIT CODE (COULD BE BETTER)
#
# import cv2
# import numpy as np
# import random
# from collections import deque

# def load_image(file_path):
#     image = cv2.imread(file_path)
#     if image is None:
#         raise ValueError("Failed to load the image. Check the file path.")
#     return image

# def convert_to_binary_matrix(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#     binary_matrix = binary // 255
#     return binary_matrix

# def detect_walls_with_lsd(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     lsd = cv2.createLineSegmentDetector(0)
#     lines = lsd.detect(edges)[0]
#     return lines

# def color_walls(image, lines):
#     colored_image = image.copy()
#     for line in lines:
#         x1, y1, x2, y2 = map(int, line[0])
#         cv2.line(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     return colored_image

# def place_red_nodes(colored_image, binary_matrix, num_nodes=1000):
#     nodes = []
#     for _ in range(num_nodes):
#         while True:
#             x, y = random.randint(10, binary_matrix.shape[0] - 11), random.randint(10, binary_matrix.shape[1] - 11)
#             if binary_matrix[x, y] == 0:
#                 wall_nearby = False
#                 for i in range(x-7, x+8):
#                     for j in range(y-7, y+8):
#                         if binary_matrix[i, j] == 1:
#                             wall_nearby = True
#                             break
#                     if wall_nearby:
#                         break
#                 if not wall_nearby:
#                     nodes.append((x, y))
#                     cv2.circle(colored_image, (y, x), 2, (0, 0, 255), -1)
#                     break
#     return nodes, colored_image

# def mark_exits(image, exits):
#     for (x, y) in exits:
#         cv2.circle(image, (y, x), 5, (255, 0, 255), -1)
#     return image

# def bfs_connect(matrix, image, start, end):
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     queue = deque([(start, [start])])
#     visited = set()

#     while queue:
#         (current, path) = queue.popleft()
#         if current == end:
#             for k in range(len(path) - 1):
#                 cv2.line(image, (path[k][1], path[k][0]), (path[k + 1][1], path[k + 1][0]), (255, 0, 0), 2)
#             cv2.imshow('Detected Lines with BFS Path', image)
#             cv2.waitKey(50)
#             break
#         if current not in visited:
#             visited.add(current)
#             for d in directions:
#                 next_node = (current[0] + d[0], current[1] + d[1])
#                 if 0 <= next_node[0] < matrix.shape[0] and 0 <= next_node[1] < matrix.shape[1]:
#                     if matrix[next_node[0], next_node[1]] == 0 or next_node == end:
#                         if next_node not in visited:
#                             queue.append((next_node, path + [next_node]))

# def main():
#     file_path = 'Simple Lib.png'
    
#     # Step 1: Load the image
#     image = load_image(file_path)
    
#     # Step 2: Detect walls using LSD
#     lines = detect_walls_with_lsd(image)
    
#     # Step 3: Color the walls green
#     colored_image = color_walls(image, lines)

#     # Step 4: Mark the exits with purple nodes
#     exits = [(324, 9), (321, 263), (62, 714)]
#     colored_image = mark_exits(colored_image, exits)
    
#     # Step 5: Convert to binary matrix for node placement
#     binary_matrix = convert_to_binary_matrix(colored_image)
    
#     # Step 6: Place red nodes in the rooms using Monte Carlo method
#     nodes, output_image = place_red_nodes(colored_image, binary_matrix)

#     # Step 7: Print the list of node locations (optional for debugging or validation)
#     print("Nodes locations (x, y):")
#     for node in nodes:
#         print(node)

#     # Step 8: Mark the starting node with a brown circle
#     start_node = (83, 75)
#     cv2.circle(output_image, (start_node[1], start_node[0]), 5, (42, 42, 165), -1)  # BGR for brown

#     # Step 9: Use BFS to connect start node to an exit point
#     bfs_connect(binary_matrix, output_image, start_node, exits[2])
    
#     # Display the final image
#     cv2.imshow('Output Image with Walls, Nodes, and Exits', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()






# GOOD BFS TRY (RUNS THROUGH WALLS  :(   )
#
# import cv2
# import numpy as np
# import random
# from collections import deque

# def load_image(file_path):
#     image = cv2.imread(file_path)
#     if image is None:
#         raise ValueError("Failed to load the image. Check the file path.")
#     return image

# def convert_to_binary_matrix(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#     binary_matrix = binary // 255
#     return binary_matrix

# def detect_walls_with_lsd(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     lsd = cv2.createLineSegmentDetector(0)
#     lines = lsd.detect(edges)[0]
#     return lines

# def color_walls(image, lines):
#     colored_image = image.copy()
#     for line in lines:
#         x1, y1, x2, y2 = map(int, line[0])
#         cv2.line(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     return colored_image

# def place_red_nodes(colored_image, binary_matrix, num_nodes=1000):
#     nodes = []
#     for _ in range(num_nodes):
#         while True:
#             x, y = random.randint(10, binary_matrix.shape[0] - 11), random.randint(10, binary_matrix.shape[1] - 11)
#             if binary_matrix[x, y] == 0:
#                 wall_nearby = False
#                 for i in range(x-7, x+8):
#                     for j in range(y-7, y+8):
#                         if binary_matrix[i, j] == 1:
#                             wall_nearby = True
#                             break
#                     if wall_nearby:
#                         break
#                 if not wall_nearby:
#                     nodes.append((x, y))
#                     cv2.circle(colored_image, (y, x), 2, (0, 0, 255), -1)
#                     break
#     return nodes, colored_image

# def mark_exits(image, exits):
#     for (x, y) in exits:
#         cv2.circle(image, (y, x), 5, (255, 0, 255), -1)
#     return image

# def bfs_search(matrix, image, nodes, start, end, radius=40):
#     queue = deque([(start, [start])])
#     visited = set()
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

#     while queue:
#         current, path = queue.popleft()

#         if current == end:
#             return path
        
#         if current not in visited:
#             visited.add(current)
#             for node in nodes:
#                 if node not in visited and np.linalg.norm(np.array(node) - np.array(current)) <= radius:
#                     new_path = path + [node]
#                     queue.append((node, new_path))
#                     cv2.line(image, (current[1], current[0]), (node[1], node[0]), (255, 255, 0), 1)  # Yellow line for progress
#                     cv2.imshow('Search Progress', image)
#                     cv2.waitKey(1)

#             for d in directions:
#                 next_node = (current[0] + d[0], current[1] + d[1])
#                 if (0 <= next_node[0] < matrix.shape[0] and 
#                     0 <= next_node[1] < matrix.shape[1] and 
#                     matrix[next_node[0], next_node[1]] == 0 and 
#                     next_node not in visited):
#                     queue.append((next_node, path + [next_node]))

#     return []

# def draw_path(image, path):
#     for i in range(len(path) - 1):
#         cv2.line(image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), (255, 0, 0), 2)  # Blue line for final path

# def main():
#     file_path = 'Simple Lib.png'
    
#     # Step 1: Load the image
#     image = load_image(file_path)
    
#     # Step 2: Detect walls using LSD
#     lines = detect_walls_with_lsd(image)
    
#     # Step 3: Color the walls green
#     colored_image = color_walls(image, lines)

#     # Step 4: Mark the exits with purple nodes
#     exits = [(324, 9), (321, 263), (62, 714)]
#     colored_image = mark_exits(colored_image, exits)
    
#     # Step 5: Convert to binary matrix for node placement
#     binary_matrix = convert_to_binary_matrix(colored_image)
    
#     # Step 6: Place red nodes in the rooms using Monte Carlo method
#     nodes, output_image = place_red_nodes(colored_image, binary_matrix)

#     # Step 7: Print the list of node locations (optional for debugging or validation)
#     print("Nodes locations (x, y):")
#     for node in nodes:
#         print(node)

#     # Step 8: Mark the starting node with a brown circle
#     start_node = (83, 75)
#     cv2.circle(output_image, (start_node[1], start_node[0]), 5, (42, 42, 165), -1)  # BGR for brown

#     # Step 9: Connect start node to exit[2] via BFS search
#     path = bfs_search(binary_matrix, output_image, nodes, start_node, exits[2])
#     if path:
#         draw_path(output_image, path)
#         print(f"Path found to exit at {exits[2]}")
#     else:
#         print(f"No path found to exit at {exits[2]}")
    
#     # Display the final image
#     cv2.imshow('Output Image with Walls, Nodes, and Exits', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# THE BEST PATH FINDING SO FAR (BUT RUNS THROUGH THE WALLS SOMETIMES)
#
import cv2
import numpy as np
import random
from collections import deque

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Failed to load the image. Check the file path.")
    return image

def convert_to_binary_matrix(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    binary_matrix = binary // 255
    return binary_matrix

def detect_walls_with_lsd(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]
    return lines

def color_walls(image, lines):
    colored_image = image.copy()
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        cv2.line(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return colored_image

def place_red_nodes(colored_image, binary_matrix, num_nodes=1000):
    nodes = []
    for _ in range(num_nodes):
        while True:
            x, y = random.randint(10, binary_matrix.shape[0] - 11), random.randint(10, binary_matrix.shape[1] - 11)
            if binary_matrix[x, y] == 0:
                wall_nearby = False
                for i in range(x-7, x+8):
                    for j in range(y-7, y+8):
                        if binary_matrix[i, j] == 1:
                            wall_nearby = True
                            break
                    if wall_nearby:
                        break
                if not wall_nearby:
                    nodes.append((x, y))
                    cv2.circle(colored_image, (y, x), 2, (0, 0, 255), -1)
                    break
    return nodes, colored_image

def mark_exits(image, exits):
    for (x, y) in exits:
        cv2.circle(image, (y, x), 5, (255, 0, 255), -1)
    return image

def crosses_wall(start, end, binary_matrix):
    x1, y1 = start
    x2, y2 = end
    num_points = int(np.linalg.norm(np.array(end) - np.array(start)) * 2)  # Increase the number of points for better checking
    for i in range(num_points + 1):
        x = int(x1 + i * (x2 - x1) / num_points)
        y = int(y1 + i * (y2 - y1) / num_points)
        if binary_matrix[x, y] == 1:
            return True
    return False

def bfs_search(matrix, image, nodes, start, end, radius=40):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        current, path = queue.popleft()

        if current == end:
            return path
        
        if current not in visited:
            visited.add(current)
            for node in nodes:
                if node not in visited and np.linalg.norm(np.array(node) - np.array(current)) <= radius:
                    if not crosses_wall(current, node, matrix):
                        new_path = path + [node]
                        queue.append((node, new_path))
                        cv2.line(image, (current[1], current[0]), (node[1], node[0]), (255, 255, 0), 1)  # Yellow line for progress
                        cv2.imshow('Search Progress', image)
                        cv2.waitKey(1)

    return []

def draw_path(image, path):
    for i in range(len(path) - 1):
        cv2.line(image, (path[i][1], path[i][0]), (path[i + 1][1], path[i + 1][0]), (255, 0, 0), 2)  # Blue line for final path

def main():
    file_path = 'Simple Lib.png'
    
    # Step 1: Load the image
    image = load_image(file_path)
    
    # Step 2: Detect walls using LSD
    lines = detect_walls_with_lsd(image)
    
    # Step 3: Color the walls green
    colored_image = color_walls(image, lines)

    # Step 4: Mark the exits with purple nodes
    exits = [(324, 9), (321, 263), (62, 714)]
    colored_image = mark_exits(colored_image, exits)
    
    # Step 5: Convert to binary matrix for node placement
    binary_matrix = convert_to_binary_matrix(colored_image)
    
    # Step 6: Place red nodes in the rooms using Monte Carlo method
    nodes, output_image = place_red_nodes(colored_image, binary_matrix)

    # Step 7: Print the list of node locations (optional for debugging or validation)
    print("Nodes locations (x, y):")
    for node in nodes:
        print(node)

    # Step 8: Mark the starting node with a brown circle
    start_node = (83, 75)
    cv2.circle(output_image, (start_node[1], start_node[0]), 5, (42, 42, 165), -1)  # BGR for brown

    # Step 9: Connect start node to exit[2] via BFS search
    path = bfs_search(binary_matrix, output_image, nodes, start_node, exits[2])
    if path:
        draw_path(output_image, path)
        print(f"Path found to exit at {exits[2]}")
    else:
        print(f"No path found to exit at {exits[2]}")
    
    # Display the final image
    cv2.imshow('Output Image with Walls, Nodes, and Exits', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
