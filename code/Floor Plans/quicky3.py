import numpy as np
import matplotlib.pyplot as plt

# Define polygon vertices (in order)
# For example, let's define a pentagon:
polygon_vertices = np.array([
    [0.0, 0.0],
    [4.0, 0.0],
    [4.0, 3.0],
    [2.0, 5.0],
    [0.0, 3.0]
])

# Close the polygon by repeating the first vertex at the end
if not np.array_equal(polygon_vertices[0], polygon_vertices[-1]):
    polygon_vertices = np.vstack([polygon_vertices, polygon_vertices[0]])

# Function to interpolate points along the edges of the polygon
def interpolate_polygon_edges(vertices, num_points_per_edge=20):
    interpolated_points = []
    for i in range(len(vertices)-1):
        start = vertices[i]
        end = vertices[i+1]
        # Interpolate points between start and end (including start, excluding end to avoid duplication)
        xs = np.linspace(start[0], end[0], num_points_per_edge, endpoint=False)
        ys = np.linspace(start[1], end[1], num_points_per_edge, endpoint=False)
        edge_points = np.column_stack((xs, ys))
        interpolated_points.append(edge_points)
    # Combine all edges and re-add the last vertex explicitly to close the polygon visually
    all_points = np.vstack(interpolated_points)
    all_points = np.vstack([all_points, vertices[-1]])  # Add the final closing point
    return all_points

# Compute centroid (center of mass) of polygon using the standard polygon centroid formula
def polygon_centroid(vertices):
    # Assuming vertices is closed (first == last)
    x = vertices[:, 0]
    y = vertices[:, 1]
    # shoelace formula for area
    A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    # centroid
    Cx = (1/(6*A)) * np.sum((x[:-1] + x[1:])*(x[:-1]*y[1:] - x[1:]*y[:-1]))
    Cy = (1/(6*A)) * np.sum((y[:-1] + y[1:])*(x[:-1]*y[1:] - x[1:]*y[:-1]))
    return Cx, Cy

# Interpolate the polygon edges
points = interpolate_polygon_edges(polygon_vertices, num_points_per_edge=20)

# Calculate centroid
centroid_x, centroid_y = polygon_centroid(polygon_vertices)

# Plotting
fig, ax = plt.subplots(figsize=(6,6))

# Plot the interpolated polygon points
ax.plot(points[:, 0], points[:, 1], 'o', markersize=3, color='red')

# Also draw the polygon boundary for clarity
ax.plot(polygon_vertices[:, 0], polygon_vertices[:, 1], '-k', linewidth=1)

# Mark the centroid with a cross
ax.plot(centroid_x, centroid_y, 'rx', markersize=10, mew=2)

# Remove axes, ticks, and grid
ax.axis('off')  # This removes the axis lines and labels
# Alternatively, you can individually turn off ticks and spines:
# ax.set_xticks([])
# ax.set_yticks([])
# for spine in ax.spines.values():
#     spine.set_visible(False)

# Make the aspect ratio equal so the polygon isn't skewed
ax.set_aspect('equal', adjustable='box')

# Save the figure as a PNG with transparent background
plt.savefig('polygon.png', dpi=300, bbox_inches='tight', transparent=True)

# Optionally, display the plot
# plt.show()
