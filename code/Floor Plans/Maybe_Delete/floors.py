# WALLS, NODES DETECTION
# import cv2
# import numpy as np

# # Load the floor plan image
# # image_path = 'Lib (1st).png'
# image_path = 'TLS (Ground Floor).png'
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

#     # Find contours to identify rooms
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Function to determine if a point is within a contour
#     def point_in_contour(point, contour):
#         return cv2.pointPolygonTest(contour, point, False) >= 0

#     # Draw walls (lines)
#     def draw_walls(lines, image):
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = map(int, line[0])
#                 cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green color for walls

#     # Mark gaps between detected line segments and ensure nodes are within the walls
#     def mark_gaps_and_keep_within_walls(lines, contours, image):
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

#     # Execute the functions
#     draw_walls(lines, image)
#     node_locations = mark_gaps_and_keep_within_walls(lines, contours, image)

#     # Display the result
#     cv2.imshow('Marked Walls and Gaps', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Print node coordinates
#     print("Node locations (x, y) coordinates:")
#     for node in node_locations:
#         print(node)




# WALLS, WITH NODES, AND THERE CONNECTIONS WITHIN THE RADIUS
# import cv2
# import numpy as np

# # Load the floor plan image
# # image_path = 'TLS (Ground Floor).png'
# image_path = 'Simple Lib.png'
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

#     # Find contours to identify rooms
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Function to determine if a point is within a contour
#     def point_in_contour(point, contour):
#         return cv2.pointPolygonTest(contour, point, False) >= 0

#     # Draw walls (lines)
#     def draw_walls(lines, image):
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = map(int, line[0])
#                 cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green color for walls

#     # Mark gaps between detected line segments and ensure nodes are within the walls
#     def mark_gaps_and_keep_within_walls(lines, contours, image):
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

#     # Connect nodes within a small radius with blue edges
#     def connect_nodes_with_edges(nodes, image, radius=50): # 50
#         for i in range(len(nodes)):
#             for j in range(i + 1, len(nodes)):
#                 node1 = nodes[i]
#                 node2 = nodes[j]
#                 distance = np.linalg.norm(np.array(node1) - np.array(node2))
#                 if distance < radius:
#                     cv2.line(image, node1, node2, (255, 0, 0), 2)  # blue color for edges

#     # Execute the functions
#     draw_walls(lines, image)
#     node_locations = mark_gaps_and_keep_within_walls(lines, contours, image)
#     connect_nodes_with_edges(node_locations, image)

#     # Display the result
#     cv2.imshow('Marked Walls, Nodes, and Edges', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Print node coordinates
#     print("Node locations (x, y) coordinates:")
#     for node in node_locations:
#         print(node)



# Structure
# (x, y, z) - z is the floor number used for the * const to increase the time
# use id for a building name


# Convert the pic into matrix
# Travers it




# DETECT WALLS
import cv2
import numpy as np

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

    # Draw the detected lines on the original image
    drawn_image = lsd.drawSegments(image, lines)

    # Display the result
    cv2.imshow('Detected Lines with LSD', drawn_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
