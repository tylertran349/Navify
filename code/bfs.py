import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
from PIL import Image

def read_binary_image(image_path):
    """
    Reads a binary image and converts it to a numpy array.
    """
    img = Image.open(image_path).convert('L')
    binary_img = np.array(img) > 128
    return binary_img.astype(np.uint8)

def compress_image(binary_img, factor):
    """
    Compresses the binary image by the specified factor.

    Parameters:
    - binary_img (numpy.ndarray): Original binary image.
    - factor (int): Compression factor (e.g., 2 will reduce the resolution by 1/2).

    Returns:
    - numpy.ndarray: Compressed binary image.
    """
    if factor <= 1:
        return binary_img  # No compression if factor is 1 or less

    compressed_rows = binary_img.shape[0] // factor
    compressed_cols = binary_img.shape[1] // factor

    # Downsample by taking the maximum in each block
    compressed_img = binary_img[:compressed_rows * factor, :compressed_cols * factor].reshape(
        compressed_rows, factor, compressed_cols, factor
    ).max(axis=(1, 3))

    return compressed_img.astype(np.uint8)

def bfs_shortest_path(binary_img, start, end):
    """
    Performs BFS to find the shortest path in a binary image.

    Parameters:
    - binary_img (numpy.ndarray): Binary image (1 for walkable, 0 for obstacles).
    - start (tuple): Start position (row, col).
    - end (tuple): End position (row, col).

    Returns:
    - list: The shortest path as a list of (row, col) tuples, or an empty list if no path exists.
    """
    rows, cols = binary_img.shape
    queue = deque([start])
    visited = set()
    visited.add(start)
    parent = {}

    while queue:
        current = queue.popleft()

        if current == end:
            # Path found
            return reconstruct_path(parent, start, end)

        row, col = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)

            if (
                0 <= nr < rows and 0 <= nc < cols and
                neighbor not in visited and binary_img[nr, nc] == 1
            ):
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    # No path found
    return []

def reconstruct_path(parent, start, end):
    """
    Reconstructs the shortest path from the start to the end using the parent dictionary.
    """
    path = []
    node = end
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path

def main():
    # Load binary image
    image_path = "Code/CutImages/Memorial Union.png"  # Replace with your image path
    binary_img = read_binary_image(image_path)
    # Compression factor
    compression_factor = 2  # Change this value to adjust compression
    print(f"Compressing image by a factor of {compression_factor}...")

    # Compress the image
    compressed_img = compress_image(binary_img, compression_factor)

    # Show original and compressed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(binary_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f"Compressed Image (Factor {compression_factor})")
    plt.imshow(compressed_img, cmap='gray')
    plt.show()

    # Define start and end points in the compressed image
    start = (10 // compression_factor, 15 // compression_factor)
    end = (1990 // compression_factor, 1985 // compression_factor)

    print("Performing BFS on the compressed image...")
    start_time = time.time()
    path = bfs_shortest_path(compressed_img, start, end)
    end_time = time.time()

    if path:
        print(f"Path found with length {len(path)} in compressed image.")
    else:
        print("No path found in the compressed image.")

    print(f"Pathfinding completed in {end_time - start_time:.4f} seconds.")

    # Map the path back to the original image coordinates
    if path:
        path_original = [(row * compression_factor, col * compression_factor) for row, col in path]

        # Plot the path on the original image
        plt.imshow(binary_img, cmap='gray')
        path_rows, path_cols = zip(*path_original)
        plt.plot(path_cols, path_rows, color='red', linewidth=2, label='Path')
        plt.scatter([15, 1985], [10, 1990], color='blue', s=100, label='Start/End')
        plt.title("Path Mapped Back to Original Image")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
