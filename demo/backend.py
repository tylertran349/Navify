# filename: backend.py

import os
import pickle
from PIL import Image, ImageDraw
from collections import deque
from database import Database
from graph import Node, Edge
from astar import AStar
import math
import heapq

# Import the WeightCalculator
from weight_calculator import WeightCalculator

def initialize_backend():
    """
    Initializes the database and WeightCalculator.
    """
    global database, weight_calculator
    database = load_database()
    weight_calculator = WeightCalculator(database)

def find_shortest_path(building1, x1, y1, building2, x2, y2, transport_mode, safety_mode):
    """
    Finds the shortest path between two points.
    Updates weights based on transport_mode and safety_mode before pathfinding.
    """
    # Update weights using the WeightCalculator
    weight_calculator.calculate_weights(transport_mode, safety_mode)

    if building1 == building2 and building1 != "Outdoor":
        # Indoor pathfinding within the same building
        image, travel_time = find_indoor_path(building1, x1, y1, x2, y2)
        return {"type": "indoor", "building": building1, "image": image, "travel_time": travel_time}
    elif building1 == building2 == "Outdoor":
        # Outdoor pathfinding
        path_coords, travel_time = find_outdoor_path(x1, y1, x2, y2, transport_mode)
        return {"type": "outdoor", "path": path_coords, "travel_time": travel_time}
    elif building1 != building2:
        if building1 != "Outdoor" and building2 == "Outdoor":
            # Indoor to Outdoor
            return find_indoor_to_outdoor_path(building1, x1, y1, x2, y2, transport_mode)
        elif building1 == "Outdoor" and building2 != "Outdoor":
            # Outdoor to Indoor
            return find_outdoor_to_indoor_path(x1, y1, building2, x2, y2, transport_mode)
        elif building1 != "Outdoor" and building2 != "Outdoor":
            # Indoor to Indoor between different buildings
            return find_indoor_to_indoor_path(building1, x1, y1, building2, x2, y2, transport_mode)
        else:
            raise ValueError("Unsupported pathfinding scenario.")
    else:
        raise ValueError("Unsupported pathfinding scenario.")

def find_indoor_path(building, x1, y1, x2, y2):
    """
    Finds the shortest path within an indoor building using Dijkstra's algorithm.
    Overlays the path on the high-resolution image and returns it.
    Also calculates the travel time.
    """
    # Paths to images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_image_path = os.path.join(script_dir, "processedFloorPlans", f"{building}.png")
    hd_image_path = os.path.join(script_dir, "hdFloorPlans", f"{building}.png")

    if not os.path.exists(processed_image_path):
        raise FileNotFoundError(f"Processed image for '{building}' not found.")
    if not os.path.exists(hd_image_path):
        raise FileNotFoundError(f"High-resolution image for '{building}' not found.")

    # Load the low-resolution image for pathfinding
    processed_img = Image.open(processed_image_path).convert('L')  # Convert to grayscale
    processed_width, processed_height = processed_img.size
    processed_pixels = processed_img.load()

    # Load the high-resolution image for display
    hd_img = Image.open(hd_image_path)
    hd_width, hd_height = hd_img.size

    # Resize the high-resolution image to make it smaller
    max_size = (800, 600)
    hd_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    hd_width, hd_height = hd_img.size

    # Define start and end positions
    start = (x1, y1)
    end = (x2, y2)

    # Check if start and end points are on walkable paths (white pixels)
    if processed_pixels[start] == 0 or processed_pixels[end] == 0:
        raise ValueError("Start or end point is on a wall (non-walkable area).")

    # Dijkstra's algorithm
    heap = []
    heapq.heappush(heap, (0, start))
    visited = set()
    came_from = {}

    distances = {start: 0}

    while heap:
        current_distance, current = heapq.heappop(heap)

        if current in visited:
            continue

        visited.add(current)

        if current == end:
            break

        for neighbor in get_neighbors(current, processed_width, processed_height):
            if neighbor in visited:
                continue
            if processed_pixels[neighbor] == 255:  # White pixel (walkable)
                # Calculate distance to neighbor
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                if dx != 0 and dy != 0:
                    # Diagonal move
                    distance = math.sqrt(2)
                else:
                    # Orthogonal move
                    distance = 1
                new_distance = current_distance + distance
                if neighbor not in distances or new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
                    came_from[neighbor] = current

    if end not in distances:
        raise ValueError("No path found between the start and end points.")

    # Reconstruct path
    path = []
    current = end
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    # Calculate total distance in pixels
    total_distance_pixels = 0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        if dx != 0 and dy != 0:
            # Diagonal move
            distance = math.sqrt(2)
        else:
            # Orthogonal move
            distance = 1
        total_distance_pixels += distance

    # Convert pixels to meters (15 pixels = 1 meter)
    total_distance_meters = total_distance_pixels / 15

    # Walking speed is 1.4 m/s
    travel_time_seconds = total_distance_meters / 1.4

    # Map path to high-resolution image coordinates
    # Assuming both images are of the same aspect ratio and aligned
    path_in_hd = []
    for x, y in path:
        hd_x = int(x / processed_width * hd_width)
        hd_y = int(y / processed_height * hd_height)
        path_in_hd.append((hd_x, hd_y))

    # Overlay path on high-resolution image
    hd_img_with_path = hd_img.copy()
    draw = ImageDraw.Draw(hd_img_with_path)
    # Draw the path with thicker line
    draw.line(path_in_hd, fill="red", width=5)

    # Add start and end markers
    marker_radius = 8  # Radius of the marker circles
    # Start marker (green circle)
    start_hd_x = int(start[0] / processed_width * hd_width)
    start_hd_y = int(start[1] / processed_height * hd_height)
    draw.ellipse(
        (start_hd_x - marker_radius, start_hd_y - marker_radius,
         start_hd_x + marker_radius, start_hd_y + marker_radius),
        fill="green", outline="black", width=2
    )

    # End marker (blue circle)
    end_hd_x = int(end[0] / processed_width * hd_width)
    end_hd_y = int(end[1] / processed_height * hd_height)
    draw.ellipse(
        (end_hd_x - marker_radius, end_hd_y - marker_radius,
         end_hd_x + marker_radius, end_hd_y + marker_radius),
        fill="blue", outline="black", width=2
    )

    return hd_img_with_path, travel_time_seconds

def find_outdoor_path(lon1, lat1, lon2, lat2, transport_mode):
    """
    Finds the shortest path between two outdoor points using the A* algorithm.
    Returns a list of (latitude, longitude) tuples representing the path.
    Also calculates the travel time.
    """
    # Get the outdoor adjacency list
    adjacency_list = database.get_outdoor_adjacency_list()

    # Create Node instances for start and end
    start_node = database.find_nearest_node("Outdoor", lon1, lat1)
    end_node = database.find_nearest_node("Outdoor", lon2, lat2)

    # Run A* algorithm
    astar = AStar()
    total_cost, path_nodes = astar.find_path(start_node, end_node, adjacency_list)

    if path_nodes is None:
        raise ValueError("No path found between the selected outdoor positions.")

    # Prepare data for visualization
    # Extract the coordinates from the nodes
    path_coords = [(node.y, node.x) for node in path_nodes]  # (latitude, longitude)

    # Calculate total distance
    total_distance_meters = 0
    for i in range(1, len(path_coords)):
        lat1, lon1 = path_coords[i - 1]
        lat2, lon2 = path_coords[i]
        distance = haversine_distance((lat1, lon1), (lat2, lon2)) * 1000  # Convert km to meters
        total_distance_meters += distance

    # Determine speed based on transport mode
    if transport_mode:
        # Biking speed (e.g., 5 m/s)
        speed = 5.0
    else:
        # Walking speed
        speed = 1.4

    travel_time_seconds = total_distance_meters / speed

    return path_coords, travel_time_seconds

def find_indoor_to_outdoor_path(building, x1, y1, lon2, lat2, transport_mode):
    """
    Finds a path from an indoor location to an outdoor location through entrances.
    """
    # Get entrances
    indoor_entrances = database.indoor_entrance_dict[building]
    outdoor_entrances = database.outdoor_entrance_dict[building]

    if len(indoor_entrances) != len(outdoor_entrances):
        raise ValueError("Mismatch in number of indoor and outdoor entrances.")

    # Sort entrances by distance from the indoor starting point
    indoor_entrances_sorted = sorted(
        indoor_entrances,
        key=lambda pos: euclidean_distance((x1, y1), pos)
    )

    # Try each entrance until a path is found
    for idx, indoor_entrance in enumerate(indoor_entrances_sorted):
        try:
            # Find indoor path to entrance
            indoor_path_image, indoor_travel_time = find_indoor_path(building, x1, y1, indoor_entrance[0], indoor_entrance[1])

            # Corresponding outdoor entrance
            entrance_index = indoor_entrances.index(indoor_entrance)
            outdoor_entrance = outdoor_entrances[entrance_index]

            # Find outdoor path from entrance to destination
            outdoor_path_coords, outdoor_travel_time = find_outdoor_path(
                outdoor_entrance[0], outdoor_entrance[1], lon2, lat2, transport_mode
            )

            total_travel_time = indoor_travel_time + outdoor_travel_time

            return {
                "type": "indoor_to_outdoor",
                "building": building,
                "indoor_image": indoor_path_image,
                "outdoor_path": outdoor_path_coords,
                "indoor_travel_time": indoor_travel_time,
                "outdoor_travel_time": outdoor_travel_time,
                "total_travel_time": total_travel_time
            }
        except Exception as e:
            # If no path found to this entrance, try the next one
            continue

    # If no path found through any entrance
    raise ValueError("No path found from indoor location to outdoor destination through any entrance.")

def find_outdoor_to_indoor_path(lon1, lat1, building, x2, y2, transport_mode):
    """
    Finds a path from an outdoor location to an indoor location through entrances.
    """
    # Get entrances
    outdoor_entrances = database.outdoor_entrance_dict[building]
    indoor_entrances = database.indoor_entrance_dict[building]

    if len(outdoor_entrances) != len(indoor_entrances):
        raise ValueError("Mismatch in number of outdoor and indoor entrances.")

    # Sort entrances by distance from the outdoor starting point
    outdoor_entrances_sorted = sorted(
        outdoor_entrances,
        key=lambda pos: haversine_distance((lat1, lon1), (pos[1], pos[0]))
    )

    # Try each entrance until a path is found
    for idx, outdoor_entrance in enumerate(outdoor_entrances_sorted):
        try:
            # Find outdoor path to entrance
            outdoor_path_coords, outdoor_travel_time = find_outdoor_path(
                lon1, lat1, outdoor_entrance[0], outdoor_entrance[1], transport_mode
            )

            # Corresponding indoor entrance
            entrance_index = outdoor_entrances.index(outdoor_entrance)
            indoor_entrance = indoor_entrances[entrance_index]

            # Find indoor path from entrance to destination
            indoor_path_image, indoor_travel_time = find_indoor_path(building, indoor_entrance[0], indoor_entrance[1], x2, y2)

            total_travel_time = outdoor_travel_time + indoor_travel_time

            return {
                "type": "outdoor_to_indoor",
                "building": building,
                "outdoor_path": outdoor_path_coords,
                "indoor_image": indoor_path_image,
                "outdoor_travel_time": outdoor_travel_time,
                "indoor_travel_time": indoor_travel_time,
                "total_travel_time": total_travel_time
            }
        except Exception as e:
            # If no path found through this entrance, try the next one
            continue

    # If no path found through any entrance
    raise ValueError("No path found from outdoor location to indoor destination through any entrance.")

def find_indoor_to_indoor_path(building1, x1, y1, building2, x2, y2, transport_mode):
    """
    Finds a path from an indoor location in one building to an indoor location in another building.
    """
    # Get entrances for both buildings
    indoor_entrances_1 = database.indoor_entrance_dict[building1]
    outdoor_entrances_1 = database.outdoor_entrance_dict[building1]

    indoor_entrances_2 = database.indoor_entrance_dict[building2]
    outdoor_entrances_2 = database.outdoor_entrance_dict[building2]

    if len(indoor_entrances_1) != len(outdoor_entrances_1) or len(indoor_entrances_2) != len(outdoor_entrances_2):
        raise ValueError("Mismatch in number of entrances for one or both buildings.")

    # Find the best pair of entrances connecting the two buildings
    entrance_pairs = []
    for idx1, (indoor_ent1, outdoor_ent1) in enumerate(zip(indoor_entrances_1, outdoor_entrances_1)):
        for idx2, (indoor_ent2, outdoor_ent2) in enumerate(zip(indoor_entrances_2, outdoor_entrances_2)):
            # Calculate distance between outdoor entrances
            distance = haversine_distance((outdoor_ent1[1], outdoor_ent1[0]), (outdoor_ent2[1], outdoor_ent2[0]))
            entrance_pairs.append(((indoor_ent1, outdoor_ent1, idx1), (indoor_ent2, outdoor_ent2, idx2), distance))

    # Sort entrance pairs by total estimated distance (indoor to entrance + outdoor between entrances + entrance to indoor end)
    entrance_pairs_sorted = sorted(
        entrance_pairs,
        key=lambda pair: (
            euclidean_distance((x1, y1), pair[0][0]) +  # Indoor start to entrance
            pair[2] +  # Outdoor distance between entrances
            euclidean_distance(pair[1][0], (x2, y2))  # Entrance to indoor end
        )
    )

    # Try each pair until a path is found
    for (indoor_ent1, outdoor_ent1, idx1), (indoor_ent2, outdoor_ent2, idx2), _ in entrance_pairs_sorted:
        try:
            # Indoor path from start to entrance in building1
            indoor_path_image_1, indoor_travel_time_1 = find_indoor_path(building1, x1, y1, indoor_ent1[0], indoor_ent1[1])

            # Outdoor path from entrance of building1 to entrance of building2
            outdoor_path_coords, outdoor_travel_time = find_outdoor_path(
                outdoor_ent1[0], outdoor_ent1[1], outdoor_ent2[0], outdoor_ent2[1], transport_mode
            )

            # Indoor path from entrance to end in building2
            indoor_path_image_2, indoor_travel_time_2 = find_indoor_path(building2, indoor_ent2[0], indoor_ent2[1], x2, y2)

            total_travel_time = indoor_travel_time_1 + outdoor_travel_time + indoor_travel_time_2

            return {
                "type": "indoor_to_indoor",
                "building1": building1,
                "building2": building2,
                "indoor_image_1": indoor_path_image_1,
                "outdoor_path": outdoor_path_coords,
                "indoor_image_2": indoor_path_image_2,
                "indoor_travel_time_1": indoor_travel_time_1,
                "outdoor_travel_time": outdoor_travel_time,
                "indoor_travel_time_2": indoor_travel_time_2,
                "total_travel_time": total_travel_time
            }
        except Exception as e:
            # If no path found through this pair, try the next one
            continue

    # If no path found through any pair of entrances
    raise ValueError("No path found between indoor locations through any entrances.")

def load_database():
    """
    Helper function to load the database.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_path = os.path.join(script_dir, 'database.pkl')
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database file '{database_path}' not found.")

    with open(database_path, 'rb') as f:
        database = pickle.load(f)  # Assuming database is a Database object

    return database

def euclidean_distance(p1, p2):
    """
    Calculates Euclidean distance between two points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def haversine_distance(coord1, coord2):
    """
    Calculates the Haversine distance between two (lat, lon) coordinates.
    Returns distance in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(min(1, math.sqrt(a)))  # Ensure the value is within the domain of asin
    r = 6371  # Radius of earth in kilometers
    return c * r

def get_neighbors(position, width, height):
    """
    Retrieves 8-connected neighbors for Dijkstra's algorithm.
    """
    x, y = position
    neighbors = []
    # 8-connected neighbors (including diagonals)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    return neighbors

# Initialize the backend when the module is loaded
initialize_backend()
