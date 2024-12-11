# CONNECT RANDOM NODES (RADIUS) WITH BFS
# import cv2
# import numpy as np
# import random
# from collections import deque

# # Load the floor plan image
# image_path = 'Lib.png'  # Update with your image path
# image = cv2.imread(image_path)

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

#     # Create a binary matrix (1 - wall, 0 - no wall)
#     matrix = np.zeros((35, 35), dtype=int)

#     def draw_walls(lines, image, matrix):
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = map(int, line[0])
#                 cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green color for walls
#                 cv2.line(matrix, (x1 // 20, y1 // 20), (x2 // 20, y2 // 20), 1, 2)  # mark walls on the matrix

#     # Mark gaps between detected line segments and ensure nodes are within the walls
#     def mark_gaps_and_keep_within_walls(lines, contours, image, matrix):
#         node_locations = []
#         if lines is not None:
#             counter = 0  # Initialize a counter to track the number of points
#             for i in range(len(lines)):
#                 for j in range(i + 1, len(lines)):
#                     if counter % 90 == 0:  # Only process every 90th point
#                         line1 = lines[i][0]
#                         line2 = lines[j][0]
#                         x1, y1, x2, y2 = map(int, line1)
#                         x3, y3, x4, y4 = map(int, line2)

#                         # Calculate the midpoint of the gap if on the same axis
#                         if abs(y1 - y3) < 10 or abs(y2 - y4) < 10:  # horizontal lines
#                             if abs(x2 - x3) > 20:  # gap detected
#                                 midpoint_x = (x2 + x3) // 2
#                                 midpoint_y = (y2 + y3) // 2
#                                 midpoint = (int(midpoint_x), int(midpoint_y))
#                                 # Check if midpoint is within any contour (room)
#                                 for contour in contours:
#                                     if point_in_contour(midpoint, contour):
#                                         cv2.circle(image, midpoint, 5, (0, 0, 255), -1)
#                                         node_locations.append(midpoint)
#                         elif abs(x1 - x3) < 10 or abs(x2 - x4) < 10:  # vertical lines
#                             if abs(y2 - y3) > 20:  # gap detected
#                                 midpoint_x = (x1 + x3) // 2
#                                 midpoint_y = (y1 + y3) // 2
#                                 midpoint = (int(midpoint_x), int(midpoint_y))
#                                 # Check if midpoint is within any contour (room)
#                                 for contour in contours:
#                                     if point_in_contour(midpoint, contour):
#                                         cv2.circle(image, midpoint, 5, (0, 0, 255), -1)
#                                         node_locations.append(midpoint)
#                     counter += 1  # Increment the counter
#         return node_locations

#     # Function to run BFS to connect nodes and draw the path without hitting walls
#     def bfs_connect_nodes(matrix, nodes, image):
#         directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
#         for i in range(len(nodes)):
#             for j in range(i + 1, len(nodes)):
#                 start = nodes[i]
#                 end = nodes[j]
#                 queue = deque([(start, [start])])
#                 visited = set()
#                 while queue:
#                     (current, path) = queue.popleft()
#                     if current == end:
#                         for k in range(len(path) - 1):
#                             cv2.line(image, (path[k][1] * 20 + 10, path[k][0] * 20 + 10), (path[k + 1][1] * 20 + 10, path[k + 1][0] * 20 + 10), (255, 0, 0), 2)
#                             cv2.imshow('Matrix', image)
#                             cv2.waitKey(50)  # Visualize each step
#                         break
#                     if current not in visited:
#                         visited.add(current)
#                         for d in directions:
#                             next_node = (current[0] + d[0], current[1] + d[1])
#                             if 0 <= next_node[0] < 35 and 0 <= next_node[1] < 35:
#                                 if matrix[next_node[0], next_node[1]] == 0 and next_node not in visited:
#                                     queue.append((next_node, path + [next_node]))

#     def point_in_contour(point, contour):
#         return cv2.pointPolygonTest(contour, point, False) >= 0

#     # Find contours to identify rooms or enclosed areas
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Execute the functions
#     draw_walls(lines, image, matrix)
#     node_locations = mark_gaps_and_keep_within_walls(lines, contours, image, matrix)

#     # For demonstration, we will randomly generate some nodes in non-wall areas of the matrix
#     def generate_random_nodes(matrix, num_nodes=5):
#         nodes = []
#         while len(nodes) < num_nodes:
#             x, y = random.randint(0, 34), random.randint(0, 34)
#             if matrix[x, y] == 0:  # Ensure nodes are placed in empty spaces
#                 nodes.append((x, y))
#                 cv2.circle(image, (y * 20 + 10, x * 20 + 10), 5, (0, 0, 255), -1)  # Draw nodes in red
#         return nodes

#     # Generate random nodes
#     random_nodes = generate_random_nodes(matrix)

#     # Connect nodes with BFS avoiding walls
#     bfs_connect_nodes(matrix, random_nodes, image)

#     # Display the result
#     cv2.imshow('Marked Walls, Nodes, and Edges', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# NODES NEAR THE WALLS
import cv2
import numpy as np
import random
from collections import deque

# Load the floor plan image
image_path = 'Simple Lib.png'  # Update with your image path
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    raise ValueError("Failed to load the image. Check the file path.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Initialize Line Segment Detector
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]  # Detect lines in the image

    # Create a binary matrix (1 - wall, 0 - open space)
    matrix = np.zeros((35, 35), dtype=int)

    def draw_walls(lines, image, matrix):
        """Draw detected walls on the image and update matrix with wall positions."""
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw walls in green
                cv2.line(matrix, (x1 // 20, y1 // 20), (x2 // 20, y2 // 20), 1, 2)  # Update walls in matrix

    def is_point_within_contour(point, contour):
        """Check if a given point is inside a specified contour."""
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def mark_nodes(lines, contours, image):
        """Mark nodes based on wall gaps, ensuring nodes fall within defined contours."""
        node_locations = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1, line2 = lines[i][0], lines[j][0]
                x1, y1, x2, y2 = map(int, line1)
                x3, y3, x4, y4 = map(int, line2)

                # Detect gaps and add nodes within contours only
                if abs(y1 - y3) < 10 and abs(x2 - x3) > 20:  # Horizontal gap
                    midpoint = ((x2 + x3) // 2, (y2 + y3) // 2)
                    if any(is_point_within_contour(midpoint, contour) for contour in contours):
                        cv2.circle(image, midpoint, 5, (0, 0, 255), -1)
                        node_locations.append(midpoint)
                elif abs(x1 - x3) < 10 and abs(y2 - y3) > 20:  # Vertical gap
                    midpoint = ((x1 + x3) // 2, (y1 + y3) // 2)
                    if any(is_point_within_contour(midpoint, contour) for contour in contours):
                        cv2.circle(image, midpoint, 5, (0, 0, 255), -1)
                        node_locations.append(midpoint)
        return node_locations

    def bfs_connect_nodes(matrix, nodes, image):
        """Connect each pair of nodes in the matrix with BFS paths, avoiding walls."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        for start in nodes:
            for end in nodes:
                if start == end:
                    continue
                queue = deque([(start, [start])])
                visited = set()
                while queue:
                    current, path = queue.popleft()
                    if current == end:
                        for k in range(len(path) - 1):
                            cv2.line(image, (path[k][1] * 20 + 10, path[k][0] * 20 + 10),
                                     (path[k + 1][1] * 20 + 10, path[k + 1][0] * 20 + 10), (255, 0, 0), 2)
                        break
                    visited.add(current)
                    for d in directions:
                        next_node = (current[0] + d[0], current[1] + d[1])
                        if 0 <= next_node[0] < matrix.shape[0] and 0 <= next_node[1] < matrix.shape[1]:
                            if matrix[next_node[0], next_node[1]] == 0 and next_node not in visited:
                                queue.append((next_node, path + [next_node]))

    # Find contours to define rooms or enclosed areas
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Execute functions
    draw_walls(lines, image, matrix)
    node_locations = mark_nodes(lines, contours, image)

    def generate_random_nodes(matrix, num_nodes=5):
        """Generate random nodes in open spaces within the matrix."""
        nodes = []
        while len(nodes) < num_nodes:
            x, y = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
            if matrix[x, y] == 0:
                nodes.append((x, y))
                cv2.circle(image, (y * 20 + 10, x * 20 + 10), 5, (0, 0, 255), -1)
        return nodes

    # Generate nodes and connect them
    random_nodes = generate_random_nodes(matrix)
    bfs_connect_nodes(matrix, random_nodes, image)

    # Display the result
    cv2.imshow('Connected Nodes and Walls', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
