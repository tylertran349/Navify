import numpy as np
import heapq
import matplotlib.pyplot as plt
import time

def read_binary_image(image_path):
    """
    Reads a binary image and converts it to a numpy array.
    """
    from PIL import Image
    img = Image.open(image_path).convert('L')
    binary_img = np.array(img) > 128
    return binary_img.astype(np.uint8)

def heuristic(node, goal):
    """
    Heuristic function to estimate the distance between two points (Manhattan distance).
    """
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def explore_neighbors_optimized(binary_img, current, goal, queue, visited, parent, g_score, heuristic_cache, update_interval, processed_nodes, direction=None):
    """
    Explores neighbors of the current node during optimized A* search.
    Adds directional exploration and periodic heuristic updates.
    """
    rows, cols = binary_img.shape
    row, col = current
    neighbors = []

    # Define neighbor priority based on the direction of movement
    if direction == "horizontal":
        neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    elif direction == "vertical":
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    else:  # Default to all directions
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

    for dr, dc in neighbors:
        nr, nc = row + dr, col + dc
        neighbor = (nr, nc)

        if (
            0 <= nr < rows and 0 <= nc < cols and
            binary_img[nr, nc] == 1 and
            neighbor not in visited
        ):
            tentative_g_score = g_score[current] + np.hypot(dr, dc)

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g_score
                parent[neighbor] = current

                # Update heuristic cache periodically
                if neighbor not in heuristic_cache or processed_nodes >= update_interval:
                    heuristic_cache[neighbor] = heuristic(neighbor, goal)

                f_score = tentative_g_score + heuristic_cache[neighbor]
                heapq.heappush(queue, (f_score, neighbor))

def bidirectional_a_star_optimized(binary_img, start, end, initial_update_interval=100):
    """
    Performs a comprehensive bidirectional A* search with all optimizations.
    """
    rows, cols = binary_img.shape

    # Initialization
    forward_queue = []
    backward_queue = []
    forward_visited = set()
    backward_visited = set()
    forward_parent = {start: None}
    backward_parent = {end: None}
    forward_g_score = {start: 0}
    backward_g_score = {end: 0}

    heapq.heappush(forward_queue, (heuristic(start, end), start))
    heapq.heappush(backward_queue, (heuristic(end, start), end))

    forward_heuristic_cache = {start: heuristic(start, end)}
    backward_heuristic_cache = {end: heuristic(end, start)}

    # Adaptive update interval
    update_interval = initial_update_interval
    processed_nodes = 0

    # Bidirectional search loop
    while forward_queue and backward_queue:
        # Forward search
        if forward_queue:
            _, current_forward = heapq.heappop(forward_queue)

            if current_forward in backward_visited:
                return reconstruct_bidirectional_path(forward_parent, backward_parent, current_forward)

            if current_forward not in forward_visited:
                forward_visited.add(current_forward)
                explore_neighbors_optimized(binary_img, current_forward, end, forward_queue, forward_visited,
                                            forward_parent, forward_g_score, forward_heuristic_cache,
                                            update_interval, processed_nodes, direction="horizontal")

        # Backward search
        if backward_queue:
            _, current_backward = heapq.heappop(backward_queue)

            if current_backward in forward_visited:
                return reconstruct_bidirectional_path(forward_parent, backward_parent, current_backward)

            if current_backward not in backward_visited:
                backward_visited.add(current_backward)
                explore_neighbors_optimized(binary_img, current_backward, start, backward_queue, backward_visited,
                                            backward_parent, backward_g_score, backward_heuristic_cache,
                                            update_interval, processed_nodes, direction="vertical")

        # Dynamically adjust update interval
        processed_nodes += 1
        if processed_nodes >= update_interval:
            update_interval = min(update_interval * 2, len(forward_queue) + len(backward_queue))
            processed_nodes = 0

    # No path found
    return []

def reconstruct_bidirectional_path(forward_parent, backward_parent, meeting_point):
    """
    Reconstructs the shortest path from the start to the end using forward and backward parents.
    """
    path = []
    node = meeting_point

    # Reconstruct forward path
    while node:
        path.append(node)
        node = forward_parent[node]
    path.reverse()

    # Reconstruct backward path
    node = backward_parent[meeting_point]
    while node:
        path.append(node)
        node = backward_parent[node]

    return path

def main():
    # Load binary image
    image_path = "Code/CutImages/Memorial Union.png"  # Replace with your image path
    binary_img = read_binary_image(image_path)

    # Define start and end points
    start = (1900,200)  # Replace with your start point
    end = (400,2000)  # Replace with your end point

    print("Performing Fully Optimized Bidirectional A*...")
    start_time = time.time()
    path = bidirectional_a_star_optimized(binary_img, start, end, initial_update_interval=100)
    end_time = time.time()

    if path:
        print(f"Path found with length {len(path)}.")
    else:
        print("No path found.")

    print(f"Pathfinding completed in {end_time - start_time:.4f} seconds.")

    # Plot result
    plt.imshow(binary_img, cmap='gray')
    if path:
        path_rows, path_cols = zip(*path)
        plt.plot(path_cols, path_rows, color='red', linewidth=2, label='Path')
    plt.scatter([start[1], end[1]], [start[0], end[0]], color='blue', s=100, label='Start/End')
    plt.title("Fully Optimized Bidirectional A*")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
