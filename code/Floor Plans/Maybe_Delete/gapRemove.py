import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Failed to load the image. Check the file path.")
    return image

def detect_walls_with_lsd(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Bilateral Filter to reduce noise while keeping edges sharp
    blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # Edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, 25, 75)
    # Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # Use Line Segment Detector
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]  # Get the detected lines
    return lines

def color_walls(image, lines):
    colored_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(colored_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green walls
    return colored_image

def walls_to_binary_matrix(image, lines):
    # Create a binary matrix with the same dimensions as the image
    binary_matrix = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(binary_matrix, (x1, y1), (x2, y2), 1, 2)  # Draw the line with value 1 and thickness 2
    return binary_matrix

def fill_gaps_in_binary_matrix(binary_matrix, radius):
    # Create a circular structuring element (kernel)
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Perform morphological closing to fill gaps
    closed_matrix = cv2.morphologyEx(binary_matrix, cv2.MORPH_CLOSE, kernel)
    return closed_matrix

def binary_matrix_to_image(binary_matrix):
    # Convert the binary matrix back to an image
    image = (binary_matrix * 255).astype(np.uint8)
    return image

def overlay_filled_gaps(original_image, filled_binary_matrix):
    # Create a mask from the filled binary matrix
    filled_mask = filled_binary_matrix.astype(np.uint8) * 255
    filled_mask_colored = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)
    
    # Define the color for filled gaps (e.g., blue)
    filled_color = np.array([255, 0, 0], dtype=np.uint8)  # Blue in BGR
    
    # Create a colored version of the filled mask
    filled_colored = np.zeros_like(original_image)
    filled_colored[filled_mask == 255] = filled_color
    
    # Overlay the filled gaps on the original image
    overlay_image = original_image.copy()
    # Blend the original image with the filled gaps
    alpha = 0.7  # Transparency factor for original image
    beta = 0.3   # Transparency factor for filled gaps
    cv2.addWeighted(overlay_image, alpha, filled_colored, beta, 0, overlay_image)
    
    return overlay_image

def get_random_free_points(binary_matrix):
    free_positions = np.argwhere(binary_matrix == 0)
    if len(free_positions) < 2:
        raise ValueError("Not enough free space to select two points.")
    start_idx, end_idx = np.random.choice(len(free_positions), size=2, replace=False)
    start_point = tuple(free_positions[start_idx][::-1])  # Reverse (y,x) to (x,y)
    end_point = tuple(free_positions[end_idx][::-1])
    return start_point, end_point

def bfs_pathfinding(binary_matrix, start, end):
    rows, cols = binary_matrix.shape
    visited = np.full((rows, cols), False, dtype=bool)
    prev = np.full((rows, cols, 2), -1, dtype=int)  # To reconstruct path
    queue = deque()
    queue.append(start)
    visited[start[1], start[0]] = True

    # 4-connected grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.popleft()
        if current == end:
            # Reconstruct path
            path = []
            while current != start:
                path.append(current)
                current = tuple(prev[current[1], current[0]])
            path.append(start)
            path.reverse()
            return path
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if not visited[ny, nx] and binary_matrix[ny, nx] == 0:
                    visited[ny, nx] = True
                    prev[ny, nx] = current
                    queue.append((nx, ny))
    # No path found
    return None

def overlay_path(image, path):
    if path is None:
        print("No path found.")
        return image
    path_image = image.copy()
    for point in path:
        cv2.circle(path_image, point, radius=1, color=(0, 0, 255), thickness=-1)  # Red path
    return path_image

def main():
    file_path = 'code/Floor Plans/Simple Lib.png'
    
    # Step 1: Load the image
    image = load_image(file_path)
    
    # Step 2: Detect walls using improved edge detection
    lines = detect_walls_with_lsd(image)
    
    # Step 3: Create binary matrix from walls
    binary_matrix = walls_to_binary_matrix(image, lines)
    
    # Step 4: Fill gaps in the binary matrix using radius search
    radius = 2  # Adjust radius as needed
    filled_binary_matrix = fill_gaps_in_binary_matrix(binary_matrix.copy(), radius)
    
    # Step 5: Convert the filled binary matrix back to an image
    filled_image = binary_matrix_to_image(filled_binary_matrix)
    
    # Step 6: Overlay the filled gaps on the original image
    overlay_image = overlay_filled_gaps(image, filled_binary_matrix)
    
    # Optional: Color the walls on the overlay image
    overlay_image_with_walls = color_walls(overlay_image, lines)
    
    # Step 7: Pathfinding between two random points
    start_point, end_point = get_random_free_points(filled_binary_matrix)
    print(f"Start point: {start_point}")
    print(f"End point: {end_point}")
    
    path = bfs_pathfinding(filled_binary_matrix, start_point, end_point)
    if path is None:
        print("No path found between the two points.")
    else:
        # Overlay the path on the image
        overlay_image_with_path = overlay_path(overlay_image_with_walls, path)
        # Mark the start and end points
        cv2.circle(overlay_image_with_path, start_point, radius=3, color=(255, 0, 255), thickness=-1)  # Magenta start point
        cv2.circle(overlay_image_with_path, end_point, radius=3, color=(255, 255, 0), thickness=-1)  # Cyan end point

        # Step 8: Display the final image with walls, filled gaps, and path using matplotlib for zooming
        # Convert the image from BGR to RGB for displaying with matplotlib
        overlay_image_rgb = cv2.cvtColor(overlay_image_with_path, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(overlay_image_rgb)
        plt.title('Image with Walls, Filled Gaps, and Path (Zoomable)')
        plt.axis('on')  # Show axis to assist with zooming
        plt.show()
    
        # Optional: Save the final image
        # cv2.imwrite('path_overlay.png', overlay_image_with_path)

if __name__ == "__main__":
    main()
