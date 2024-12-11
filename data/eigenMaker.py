import numpy as np
from scipy.sparse.csgraph import shortest_path

# Define the adjacency list with connections to the sink (node 37)
adjacency_list = [
    [4, 37], [2, 37], [1, 5], [6, 37], [0, 5, 21], [2, 4, 6, 14],
    [3, 5, 7, 10], [6, 8, 11], [7, 8, 12], [8, 37], [6, 11, 15],
    [7, 10, 12, 16], [8, 11, 13, 19], [12, 37], [5, 15, 24],
    [10, 14, 16, 17], [11, 15, 18], [15, 18, 26], [16, 17, 19, 27],
    [12, 18, 20, 28], [19, 37], [22, 37], [4, 21, 23], [22, 24, 30],
    [14, 23, 25], [24, 26, 31], [17, 25, 27, 32], [18, 26, 28],
    [19, 27, 29], [28, 37], [23, 31, 35], [24, 30, 32],
    [25, 31, 33, 36], [34, 37], [30, 33, 35],
    [32, 34, 36], [36, 37], [37]
]

teleportation_factor = 0.1       # 10% stays in the same node for stability
sink_damping_factor = 0.5        # 50% for nodes connected to sink
sink_node = 37
priority_nodes = [5, 6, 7, 14, 15, 19, 24, 26, 27]  # Nodes to prioritize
priority_factor = 1.5  # Boost factor for priority nodes

# Determine the maximum node index to size the matrix
max_index = max(max(neighbors) for neighbors in adjacency_list) + 1

# Initialize the adjacency matrix for shortest path calculations
adjacency_matrix_for_distance = np.zeros((max_index, max_index))

# Convert the adjacency list to a matrix format for distance calculation
for i, neighbors in enumerate(adjacency_list):
    for neighbor in neighbors:
        adjacency_matrix_for_distance[i][neighbor] = 1  # Set distance as 1 for direct connections

# Calculate Manhattan distances (shortest paths) from each node to the sink node
distances, _ = shortest_path(csgraph=adjacency_matrix_for_distance, directed=False, indices=sink_node, return_predecessors=True)

# Reinitialize the adjacency matrix for the distance-weighted flow distribution
adjacency_matrix = np.zeros((max_index, max_index))

# Build the adjacency matrix with distance-based weighting and priority for specific nodes
for i, neighbors in enumerate(adjacency_list):
    if neighbors == [i]:  # Self-referencing nodes keep their flow
        adjacency_matrix[i][i] = 1
    elif neighbors:
        if sink_node in neighbors:
            # Nodes directly connected to the sink lose 50% to the sink
            non_sink_neighbors = [n for n in neighbors if n != sink_node]
            n_neighbors = len(non_sink_neighbors)
            if n_neighbors > 0:
                recirculation_weight = (1 - sink_damping_factor) / n_neighbors
                for neighbor in non_sink_neighbors:
                    adjacency_matrix[neighbor][i] += recirculation_weight
            adjacency_matrix[sink_node][i] += sink_damping_factor
        else:
            # Distribute flow among neighbors, weighted by distance and priority
            non_sink_neighbors = [n for n in neighbors if distances[n] > 0]
            weight_sum = sum(
                distances[neighbor] * (priority_factor if neighbor in priority_nodes else 1)
                for neighbor in non_sink_neighbors
            )
            for neighbor in non_sink_neighbors:
                weight = (distances[neighbor] * (priority_factor if neighbor in priority_nodes else 1)) / weight_sum
                adjacency_matrix[neighbor][i] += weight

# Apply teleportation factor for stability
adjacency_matrix = (1 - teleportation_factor) * adjacency_matrix + (teleportation_factor / max_index) * np.ones((max_index, max_index))

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(adjacency_matrix)

# Find the eigenvector corresponding to the largest eigenvalue (steady-state vector)
dominant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

np.savetxt('data/eigenvector.txt', dominant_eigenvector)  # Save as a .txt file

print(dominant_eigenvector)
