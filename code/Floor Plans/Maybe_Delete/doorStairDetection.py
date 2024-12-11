import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import time

# File paths for the uploaded images
floorplan_image_path = "Code/code/Floor Plans/Simple Lib.png"

def load_image(file_path):
    """
    Loads an image from the specified file path.
    
    Args:
        file_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Loaded image.
        
    Raises:
        ValueError: If the image fails to load.
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Failed to load the image. Check the file path.")
    return image

def convert_to_binary_matrix(image):
    """
    Converts a BGR image to a binary image using grayscale conversion and thresholding.
    
    Args:
        image (numpy.ndarray): Input BGR image.
        
    Returns:
        numpy.ndarray: Binary image with pixels as 0 or 255.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding (invert the binary image)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary

import cv2
import numpy as np

# Pre-define composite kernels globally
composite_kernels = [
    np.array([[1, 1, 1]], dtype=np.uint8),          # Horizontal
    np.array([[1], [1], [1]], dtype=np.uint8),      # Vertical
    np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]], dtype=np.uint8)             # Diagonal
]

def remove_one_pixel_wide_lines(binary_image):
    """
    Removes 1-pixel wide lines from a binary image using a combined morphological approach.
    
    Args:
        binary_image (numpy.ndarray): Input binary image.
        
    Returns:
        numpy.ndarray: Cleaned binary image with 1-pixel wide lines removed.
    """
    # Initialize an empty mask for all detected lines
    one_pixel_wide_lines = np.zeros_like(binary_image, dtype=np.uint8)
    
    # Iterate through each composite kernel and accumulate the detected lines
    for kernel in composite_kernels:
        detected_lines = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel)
        one_pixel_wide_lines = cv2.bitwise_or(one_pixel_wide_lines, detected_lines)
    
    # Invert the detected lines to create a mask for removal
    mask = cv2.bitwise_not(one_pixel_wide_lines)
    
    # Remove the 1-pixel wide lines from the binary image using bitwise AND
    cleaned_image = cv2.bitwise_and(binary_image, mask)
    
    return cleaned_image


def remove_small_connected_components(binary_image, max_group_size=3):
    """
    Removes small connected components from a binary image using an optimized approach.
    
    Args:
        binary_image (numpy.ndarray): Input binary image with pixels as 0 or 255.
        max_group_size (int): Maximum size of connected groups to remove.
        
    Returns:
        numpy.ndarray: Binary image with small connected components removed.
    """
    # Convert binary image to 0 and 1
    binary = (binary_image > 0).astype(np.uint8)

    # Perform connected components analysis with 8-connectivity
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Create a mask of small components (size <= max_group_size)
    sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude the background (label 0)
    small_components = np.where(sizes <= max_group_size)[0] + 1  # Labels start from 1

    # Create a mask to remove small components
    mask = np.isin(labels, small_components)

    # Set small components to 0
    binary[mask] = 0

    # Convert back to 0 and 255 format
    cleaned_image = binary * 255

    return cleaned_image

def apply_mask_to_binary_image(binary_image, mask):
    """
    Applies a mask to the binary image. Sets all pixels in the binary image to 0
    where the mask has a value of 1 (or 255).

    Args:
        binary_image (numpy.ndarray): The original binary image (0 or 255).
        mask (numpy.ndarray): The mask (0 or 255) indicating pixels to remove.

    Returns:
        numpy.ndarray: The binary image after applying the mask.
    """
    # Ensure the mask is in binary format (0 and 1)
    binary_mask = (mask > 0).astype(np.uint8)

    # Invert the mask: where the mask is 1, set to 0; otherwise, set to 1
    inverted_mask = cv2.bitwise_not(binary_mask * 255)

    # Apply the inverted mask to the binary image using bitwise AND
    result_image = cv2.bitwise_and(binary_image, inverted_mask)

    return result_image

def plot_image(image, title):
    """
    Plots a grayscale image using matplotlib with inverted colors.
    
    Args:
        image (numpy.ndarray): Image to plot.
        title (str): Title of the plot.
    """
    # Plot the image with switched colors (black background, white foreground)
    plt.figure(figsize=(10, 10))
    plt.imshow(255 - image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_random_point(binary_image):
    """
    Selects a random point in the binary image that is not an obstacle.
    
    Args:
        binary_image (numpy.ndarray): Binary image with obstacles as 1 (255) and free space as 0.
        
    Returns:
        tuple: (x, y) coordinates of the selected point.
        
    Raises:
        ValueError: If no free space is available in the image.
    """
    free_space = np.argwhere(binary_image == 0)
    if free_space.size == 0:
        raise ValueError("No free space available to select a random point.")
    selected = random.choice(free_space)
    # Note: OpenCV uses (y, x) coordinates
    return (selected[1], selected[0])

def bfs(final_image, start, goal):
    """
    Performs BFS to find the shortest path from start to goal in the binary image.

    Args:
        final_image (numpy.ndarray): Binary image with obstacles as 1 (255) and free space as 0.
        start (tuple): (x, y) coordinates of the start point.
        goal (tuple): (x, y) coordinates of the goal point.

    Returns:
        list: List of (x, y) tuples representing the path from start to goal.
              Returns an empty list if no path is found.
    """
    height, width = final_image.shape
    queue = deque()
    queue.append(start)
    visited = set()
    visited.add(start)
    parent = {}

    # Define possible movements: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            x, y = neighbor
            if 0 <= x < width and 0 <= y < height:
                if final_image[y, x] == 0 and neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current

    # Reconstruct the path
    path = []
    current = goal
    if current not in parent and current != start:
        # No path found
        return path

    while current != start:
        path.append(current)
        current = parent.get(current, start)
        if current == start:
            path.append(start)
            break
    path.reverse()
    return path

def plot_path_on_image(final_image, path, start, goal):
    """
    Plots the path on the image using matplotlib.

    Args:
        final_image (numpy.ndarray): Binary image.
        path (list): List of (x, y) tuples representing the path.
        start (tuple): (x, y) coordinates of the start point.
        goal (tuple): (x, y) coordinates of the goal point.
    """
    # Convert image to RGB for colored plotting
    image_rgb = cv2.cvtColor(255 - final_image, cv2.COLOR_GRAY2RGB)

    # Draw the path
    for i in range(len(path) - 1):
        cv2.line(image_rgb, path[i], path[i+1], (255, 0, 0), 1)  # Blue color in RGB

    # Draw start and goal points
    cv2.circle(image_rgb, start, radius=5, color=(0, 255, 0), thickness=-1)  # Green
    cv2.circle(image_rgb, goal, radius=5, color=(255, 0, 0), thickness=-1)   # Blue

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Path from Start to Goal")
    plt.axis('off')
    plt.show()

def main():
    start = time.time()

    # Load the image
    image = load_image(floorplan_image_path)
    
    
    # Convert to binary image
    binary_image = convert_to_binary_matrix(image)
    plot_image(binary_image,"removed thin")
    # Remove 1-pixel wide lines from the binary image
    cleaned_image = remove_one_pixel_wide_lines(binary_image)
    plot_image(cleaned_image,"removed thin")
    
    # Remove small connected components (groups of 3 or fewer pixels)
    start = time.time()
    cleaned_image = remove_small_connected_components(cleaned_image, max_group_size=2)
    end = time.time()
    print("time is: ", end-start)
    plot_image(cleaned_image, "removed connected")
    # Apply the cleaned mask to the original binary image
    final_image = apply_mask_to_binary_image(binary_image, cleaned_image)
    
    # Plot the final processed image
    plot_image(final_image, "Final Processed Image")
    
    # Select two random points
    try:
        start = get_random_point(final_image)
        goal = get_random_point(final_image)
        while goal == start:
            goal = get_random_point(final_image)
        print(f"Start Point: {start}")
        print(f"Goal Point: {goal}")
    except ValueError as e:
        print(e)
        return
    
    # Find path using BFS
    path = bfs(final_image, start, goal)
    
    if path:
        print(f"Path found with {len(path)} steps.")
        # Plot the path on the image
        plot_path_on_image(final_image, path, start, goal)
    else:
        print("No path found between the selected points.")

if __name__ == "__main__":
    main()
