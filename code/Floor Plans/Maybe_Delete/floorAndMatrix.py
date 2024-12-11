# NO WALLS, JUST BFS
# import cv2
# import numpy as np
# import random
# from collections import deque

# # Create a 20x20 matrix of 0s
# matrix = np.zeros((20, 20), dtype=int)

# # Function to place two random 1s far apart
# def place_random_ones(matrix):
#     # Place the first 1 randomly
#     x1, y1 = random.randint(0, 19), random.randint(0, 19)
#     matrix[x1, y1] = 1
#     # Place the second 1 far from the first one
#     while True:
#         x2, y2 = random.randint(0, 19), random.randint(0, 19)
#         if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 10:
#             matrix[x2, y2] = 1
#             break
#     return (x1, y1), (x2, y2)

# # Place the 1s
# start, end = place_random_ones(matrix)

# # Function to run BFS to connect the 1s and draw the path
# def bfs_connect(matrix, start, end):
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
#     queue = deque([(start, [start])])
#     visited = set()

#     # Create a blank image to draw the process
#     image = np.zeros((400, 400, 3), dtype=np.uint8)
#     for i in range(20):
#         for j in range(20):
#             color = (255, 255, 255) if matrix[i, j] == 0 else (0, 0, 255)
#             cv2.rectangle(image, (j * 20, i * 20), ((j + 1) * 20, (i + 1) * 20), color, -1)

#     cv2.imshow('Matrix', image)
#     cv2.waitKey(500)  # Display the initial state

#     while queue:
#         (current, path) = queue.popleft()
#         if current == end:
#             for k in range(len(path) - 1):
#                 cv2.line(image, (path[k][1] * 20 + 10, path[k][0] * 20 + 10), (path[k + 1][1] * 20 + 10, path[k + 1][0] * 20 + 10), (255, 0, 0), 2)
#                 cv2.imshow('Matrix', image)
#                 cv2.waitKey(100)  # Visualize each step
#             break
#         if current not in visited:
#             visited.add(current)
#             for d in directions:
#                 next_node = (current[0] + d[0], current[1] + d[1])
#                 if 0 <= next_node[0] < 20 and 0 <= next_node[1] < 20:
#                     if matrix[next_node[0], next_node[1]] == 0 or next_node == end:
#                         if next_node not in visited:
#                             queue.append((next_node, path + [next_node]))

# # Run BFS to connect the 1s
# bfs_connect(matrix, start, end)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# SIMPLE AND SMALL WALLS
# import cv2
# import numpy as np
# import random
# from collections import deque

# # Create a 20x20 matrix of 0s
# matrix = np.zeros((20, 20), dtype=int)

# # Function to place two random 1s far apart
# def place_random_ones(matrix):
#     # Place the first 1 randomly
#     x1, y1 = random.randint(0, 19), random.randint(0, 19)
#     matrix[x1, y1] = 1
#     # Place the second 1 far from the first one
#     while True:
#         x2, y2 = random.randint(0, 19), random.randint(0, 19)
#         if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 10:
#             matrix[x2, y2] = 1
#             break
#     return (x1, y1), (x2, y2)

# # Function to place random green walls (values of 2)
# def place_random_walls(matrix, num_walls=5):
#     for _ in range(num_walls):
#         x, y = random.randint(0, 19), random.randint(0, 19)
#         if matrix[x, y] == 0:  # Ensure walls are placed in empty spaces
#             matrix[x, y] = 2

# # Place the 1s and the walls
# start, end = place_random_ones(matrix)
# place_random_walls(matrix)

# # Function to run BFS to connect the 1s and draw the path
# def bfs_connect(matrix, start, end):
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
#     queue = deque([(start, [start])])
#     visited = set()

#     # Create a blank image to draw the process
#     image = np.zeros((400, 400, 3), dtype=np.uint8)
#     for i in range(20):
#         for j in range(20):
#             if matrix[i, j] == 0:
#                 color = (255, 255, 255)  # White for empty space
#             elif matrix[i, j] == 1:
#                 color = (0, 0, 255)  # Red for 1s
#             elif matrix[i, j] == 2:
#                 color = (0, 255, 0)  # Green for walls
#             cv2.rectangle(image, (j * 20, i * 20), ((j + 1) * 20, (i + 1) * 20), color, -1)

#     cv2.imshow('Matrix', image)
#     cv2.waitKey(500)  # Display the initial state

#     while queue:
#         (current, path) = queue.popleft()
#         if current == end:
#             for k in range(len(path) - 1):
#                 cv2.line(image, (path[k][1] * 20 + 10, path[k][0] * 20 + 10), (path[k + 1][1] * 20 + 10, path[k + 1][0] * 20 + 10), (255, 0, 0), 2)
#                 cv2.imshow('Matrix', image)
#                 cv2.waitKey(100)  # Visualize each step
#             break
#         if current not in visited:
#             visited.add(current)
#             for d in directions:
#                 next_node = (current[0] + d[0], current[1] + d[1])
#                 if 0 <= next_node[0] < 20 and 0 <= next_node[1] < 20:
#                     if matrix[next_node[0], next_node[1]] == 0 or next_node == end:
#                         if next_node not in visited:
#                             queue.append((next_node, path + [next_node]))

# # Run BFS to connect the 1s
# bfs_connect(matrix, start, end)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# THE BEST VERSION (WITH COMPLEX WALLS)
import cv2
import numpy as np
import random
from collections import deque

# Create a 35x35 matrix of 0s
matrix = np.zeros((35, 35), dtype=int)

# Function to place two random 1s far apart
def place_random_ones(matrix):
    # Place the first 1 randomly
    x1, y1 = random.randint(0, 34), random.randint(0, 34)
    matrix[x1, y1] = 1
    # Place the second 1 far from the first one
    while True:
        x2, y2 = random.randint(0, 34), random.randint(0, 34)
        if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 15:
            matrix[x2, y2] = 1
            break
    return (x1, y1), (x2, y2)

# Function to place long random green walls (values of 2)
def place_long_walls(matrix, num_walls=10, min_length=10, max_length=15):
    for _ in range(num_walls):
        while True:
            x, y = random.randint(0, 34), random.randint(0, 34)
            direction = random.choice(['horizontal', 'vertical'])
            length = random.randint(min_length, max_length)

            if direction == 'horizontal' and y + length < 35 and all(matrix[x, y:y+length] == 0):
                matrix[x, y:y+length] = 2
                break
            elif direction == 'vertical' and x + length < 35 and all(matrix[x:x+length, y] == 0):
                matrix[x:x+length, y] = 2
                break

# Place the 1s and the walls
start, end = place_random_ones(matrix)
place_long_walls(matrix)

# Function to run BFS to connect the 1s and draw the path
def bfs_connect(matrix, start, end):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    queue = deque([(start, [start])])
    visited = set()

    # Create a blank image to draw the process
    image = np.zeros((700, 700, 3), dtype=np.uint8)
    for i in range(35):
        for j in range(35):
            if matrix[i, j] == 0:
                color = (255, 255, 255)  # White for empty space
            elif matrix[i, j] == 1:
                color = (0, 0, 255)  # Red for 1s
            elif matrix[i, j] == 2:
                color = (0, 255, 0)  # Green for walls
            cv2.rectangle(image, (j * 20, i * 20), ((j + 1) * 20, (i + 1) * 20), color, -1)

    cv2.imshow('Matrix', image)
    cv2.waitKey(500)  # Display the initial state

    while queue:
        (current, path) = queue.popleft()
        if current == end:
            for k in range(len(path) - 1):
                cv2.line(image, (path[k][1] * 20 + 10, path[k][0] * 20 + 10), (path[k + 1][1] * 20 + 10, path[k + 1][0] * 20 + 10), (255, 0, 0), 2)
                cv2.imshow('Matrix', image)
                cv2.waitKey(50)  # Visualize each step
            break
        if current not in visited:
            visited.add(current)
            for d in directions:
                next_node = (current[0] + d[0], current[1] + d[1])
                if 0 <= next_node[0] < 35 and 0 <= next_node[1] < 35:
                    if matrix[next_node[0], next_node[1]] == 0 or next_node == end:
                        if next_node not in visited:
                            queue.append((next_node, path + [next_node]))

# Run BFS to connect the 1s
bfs_connect(matrix, start, end)

cv2.waitKey(0)
cv2.destroyAllWindows()