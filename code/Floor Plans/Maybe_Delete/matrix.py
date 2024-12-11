# OUTPUTS MATRIX (0 - NO WALL, 1 - WALL)
# import cv2
# import numpy as np

# # Load the floor plan image
# image_path = 'Lib (1st).png'  # Update with your image path
# image = cv2.imread(image_path)

# # Check if the image is loaded properly
# if image is None:
#     print("Failed to load the image. Check the file path.")
# else:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Threshold the image to binary
#     _, binary_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

#     # Convert binary image to 0's and 1's matrix
#     matrix = (binary_image == 0).astype(int)

#     # Display the binary matrix
#     for row in matrix:
#         print(' '.join(map(str, row)))

# # Print matrix dimensions
# print(f"Matrix dimensions: {matrix.shape}")



# import cv2
# import numpy as np

# # Load the floor plan image
# image_path = 'Lib (1st).png'  # Update with your image path
# image = cv2.imread(image_path)

# # Check if the image is loaded properly
# if image is None:
#     print("Failed to load the image. Check the file path.")
# else:
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Threshold the image to binary
#     _, binary_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

#     # Convert binary image to 0's and 1's matrix
#     matrix = (binary_image == 0).astype(int)

#     # Function to find the center of the room entrance
#     def find_room_entrances(matrix):
#         entrances = []
#         rows, cols = matrix.shape
#         for row in range(rows):
#             for col in range(cols):
#                 if matrix[row, col] == 1:  # Found a wall
#                     start_row, start_col = row, col
#                     while matrix[row, col] == 1:  # Follow the wall
#                         if col + 1 < cols:
#                             col += 1
#                         else:
#                             break
#                     end_row, end_col = row, col

#                     # Measure the distance to find the next wall
#                     distance = 0
#                     while matrix[row, col] == 0:  # No wall
#                         if col + 1 < cols:
#                             col += 1
#                             distance += 1
#                         else:
#                             break
#                     next_row, next_col = row, col

#                     if matrix[row, col] == 1:  # Found the next wall
#                         midpoint_x = (start_col + next_col) // 2
#                         midpoint_y = (start_row + next_row) // 2
#                         entrances.append((midpoint_y, midpoint_x))
#         return entrances

#     # Find the room entrances
#     room_entrances = find_room_entrances(matrix)

#     # Mark the room entrances on the image
#     for entrance in room_entrances:
#         cv2.circle(image, (entrance[1], entrance[0]), 5, (0, 0, 255), -1)

#     # Display the result
#     cv2.imshow('Room Entrances Marked', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Print matrix dimensions
#     print(f"Matrix dimensions: {matrix.shape}")

#     # Print entrance coordinates
#     print("Entrance coordinates (y, x):")
#     for entrance in room_entrances:
#         print(entrance)




#LAST VERSION
import cv2
import numpy as np

# Load the floor plan image
image_path = 'Club floor.jpg'  # Update with your image path
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print("Failed to load the image. Check the file path.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to binary
    _, binary_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # Convert binary image to 0's and 1's matrix
    matrix = (binary_image == 0).astype(int)

    # Function to find the center of the room entrance
    def find_room_entrances(matrix):
        entrances = []
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                if matrix[row, col] == 1:  # Found a wall
                    start_col = col
                    # Find the end of the wall
                    while col < cols and matrix[row, col] == 1:
                        col += 1
                    end_col = col

                    # If a significant gap is detected
                    if end_col - start_col > 10:
                        midpoint_x = (start_col + end_col) // 2
                        midpoint_y = row
                        entrances.append((midpoint_y, midpoint_x))
        return entrances

    # Find the room entrances
    room_entrances = find_room_entrances(matrix)

    # Mark the room entrances on the image
    for entrance in room_entrances:
        cv2.circle(image, (entrance[1], entrance[0]), 5, (0, 0, 255), -1)

    # Display the result
    cv2.imshow('Room Entrances Marked', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print matrix dimensions
    print(f"Matrix dimensions: {matrix.shape}")

    # Print entrance coordinates
    print("Entrance coordinates (y, x):")
    for entrance in room_entrances:
        print(entrance)
