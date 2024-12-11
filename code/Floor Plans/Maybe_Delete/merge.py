import cv2
import numpy as np
import random
from collections import deque

# Load the image
image = cv2.imread('Simple Lib.png')

# Check if the image is loaded properly
if image is None:
    print("Failed to load the image. Check the file path.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Initialize Line Segment Detector
    lsd = cv2.createLineSegmentDetector(0)

    # Detect lines in the image
    lines = lsd.detect(edges)[0]

    # Create a binary matrix based on the detected lines
    matrix = np.zeros_like(gray)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        cv2.line(matrix, (x1, y1), (x2, y2), 1, 2)

    # Function to place two random 1s far apart
    def place_random_ones(matrix):
        x1, y1 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
        while matrix[x1, y1] == 1:
            x1, y1 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
        matrix[x1, y1] = 1

        while True:
            x2, y2 = random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1)
            if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 15 and matrix[x2, y2] == 0:
                matrix[x2, y2] = 1
                break

        return (x1, y1), (x2, y2)

    # Place the 1s
    start, end = place_random_ones(matrix)

    # Function to run BFS to connect the 1s and draw the path
    def bfs_connect(matrix, start, end):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            (current, path) = queue.popleft()
            if current == end:
                for k in range(len(path) - 1):
                    cv2.line(image, (path[k][1], path[k][0]), (path[k + 1][1], path[k + 1][0]), (255, 0, 0), 2)
                cv2.imshow('Detected Lines with BFS Path', image)
                cv2.waitKey(50)  # Visualize each step
                break
            if current not in visited:
                visited.add(current)
                for d in directions:
                    next_node = (current[0] + d[0], current[1] + d[1])
                    if 0 <= next_node[0] < matrix.shape[0] and 0 <= next_node[1] < matrix.shape[1]:
                        if matrix[next_node[0], next_node[1]] == 0 or next_node == end:
                            if next_node not in visited:
                                queue.append((next_node, path + [next_node]))

    # Run BFS to connect the 1s
    bfs_connect(matrix, start, end)

    cv2.imshow('Final Path', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
