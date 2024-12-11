# NODES (CROSS OF THE ROOMS) AND WALLS WITH
import cv2
import numpy as np
from scipy.spatial import distance

# Load the floor plan image
image_path = 'Simple Lib.png'
image = cv2.imread(image_path)

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

    # Draw walls (lines)
    def draw_walls(lines, image):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line[0])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green color for walls

    # Function to detect corners of walls
    def detect_corners(lines):
        corners = []
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            corners.append((x1, y1))
            corners.append((x2, y2))
        return list(set(corners))  # Remove duplicates

    # Function to find room centers using nearest corners
    def find_room_centers(corners, image):
        centers = []
        for i in range(len(corners)):
            nearest_corners = []
            for j in range(len(corners)):
                if i != j:
                    nearest_corners.append((corners[j], distance.euclidean(corners[i], corners[j])))

            # Sort corners by distance and pick the nearest four
            nearest_corners.sort(key=lambda x: x[1])
            if len(nearest_corners) >= 4:
                corners_to_use = [corners[i]] + [corner[0] for corner in nearest_corners[:3]]
                diagonal_midpoints = []

                # Draw cross-diagonals between each pair
                for k in range(1, len(corners_to_use)):
                    x1, y1 = corners_to_use[k-1]
                    x2, y2 = corners_to_use[k]
                    diagonal_midpoints.append(((x1 + x2) // 2, (y1 + y2) // 2))
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # Calculate the average midpoint (room center)
                if len(diagonal_midpoints) >= 2:
                    avg_midpoint_x = sum([point[0] for point in diagonal_midpoints]) // len(diagonal_midpoints)
                    avg_midpoint_y = sum([point[1] for point in diagonal_midpoints]) // len(diagonal_midpoints)
                    centers.append((avg_midpoint_x, avg_midpoint_y))
                    cv2.circle(image, (avg_midpoint_x, avg_midpoint_y), 5, (0, 0, 255), -1)  # Draw the center of the room in red
        return centers

    # Execute functions
    draw_walls(lines, image)
    corners = detect_corners(lines)
    room_centers = find_room_centers(corners, image)

    # Display the result
    cv2.imshow('Detected Walls, Corners, and Room Centers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print room center coordinates
    print("Room center coordinates (x, y):")
    for center in room_centers:
        print(center)
