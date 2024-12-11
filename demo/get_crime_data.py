import math
import pandas as pd
import os

# Existing Haversine function
def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in feet
    R = 20925524.9
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Function to calculate crime multiplier based on distance to crimes
def calculate_crime_multiplier(lat_lower, lat_upper, lon_lower, lon_upper, crime_df):
    lat_center = (lat_lower + lat_upper) / 2
    lon_center = (lon_lower + lon_upper) / 2

    multiplier = 0
    for _, crime in crime_df.iterrows():
        distance = haversine(lat_center, lon_center, crime["lat"], crime["long"])
        if distance <= 100:
            multiplier += 3
        elif distance <= 500:
            multiplier += 2
        elif distance <= 1000:
            multiplier += 1
    return multiplier

# Function to calculate phone multiplier based on proximity to phones
def calculate_phone_multiplier(lat_lower, lat_upper, lon_lower, lon_upper, phone_df):
    lat_center = (lat_lower + lat_upper) / 2
    lon_center = (lon_lower + lon_upper) / 2

    # Default to no multiplier
    phone_multiplier = None
    for _, phone in phone_df.iterrows():
        distance = haversine(lat_center, lon_center, phone["lat"], phone["long"])
        # Apply lowest possible multiplier within range
        if distance <= 50:
            phone_multiplier = 0.25
            break  # Stop if closest range multiplier is found
        elif distance <= 100 and (phone_multiplier is None or phone_multiplier > 0.5):
            phone_multiplier = 0.5
        elif distance <= 200 and (phone_multiplier is None or phone_multiplier > 0.67):
            phone_multiplier = 0.67
    return phone_multiplier

# Set grid parameters
num_squares_per_side = int(math.sqrt(1000))
grid_data = {}

# Load crime and phone locations
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_crime = os.path.join(current_dir, "crime_locs.txt")
file_path_phone = os.path.join(current_dir, "phone_locs.txt")
crime_df = pd.read_csv(file_path_crime, header=None, names=["lat", "long"])
phone_df = pd.read_csv(file_path_phone, header=None, names=["lat", "long"])

# Define latitude and longitude increment for grid squares
lat_min, lat_max = 38.53410627387911, 38.54626869715259
lon_min, lon_max = -121.76112584132399, -121.7455565855204
lat_step = (lat_max - lat_min) / num_squares_per_side
lon_step = (lon_max - lon_min) / num_squares_per_side

# Iterate through grid squares and store multiplier in a dictionary
for i in range(num_squares_per_side):
    for j in range(num_squares_per_side):
        lat_lower = lat_min + i * lat_step
        lat_upper = lat_lower + lat_step
        lon_lower = lon_min + j * lon_step
        lon_upper = lon_lower + lon_step

        # Calculate the crime and phone multipliers
        crime_multiplier = calculate_crime_multiplier(lat_lower, lat_upper, lon_lower, lon_upper, crime_df)
        phone_multiplier = calculate_phone_multiplier(lat_lower, lat_upper, lon_lower, lon_upper, phone_df)

        # Apply phone multiplier if crime multiplier is 0, otherwise apply both with priority to crime
        if crime_multiplier == 0 and phone_multiplier is not None:
            final_multiplier = phone_multiplier
        elif phone_multiplier is not None:
            final_multiplier = crime_multiplier * phone_multiplier
        else:
            final_multiplier = crime_multiplier

        # Set final multiplier to 1 if both crime and phone multipliers are 0
        if final_multiplier == 0:
            final_multiplier = 1

        # Store in dictionary
        grid_data[((lat_lower, lat_upper), (lon_lower, lon_upper))] = final_multiplier

# Function to get the multiplier for a specific latitude and longitude
def get_multiplier(lat, lon):
    for ((lat_min, lat_max), (lon_min, lon_max)), multiplier in grid_data.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return multiplier
    return None

# --- Updated Function Implementation Below ---

def calculate_edge_multipliers(edges):
    """
    Calculates and returns a dictionary mapping each Edge to its corresponding multiplier.

    Parameters:
    - edges (list of Edge): List of Edge objects.

    Returns:
    - dict: Dictionary where keys are Edge objects and values are their multipliers.
    """
    edge_multipliers = {}
    for edge in edges:
        # Get coordinates from node1 and node2
        node1 = edge.node1
        node2 = edge.node2

        # Use y for latitude and x for longitude to calculate midpoint
        midpoint_lat = (node1.y + node2.y) / 2
        midpoint_lon = (node1.x + node2.x) / 2

        # Get the multiplier for the midpoint
        multiplier = get_multiplier(midpoint_lat, midpoint_lon)

        # If no multiplier is found, default to 1
        if multiplier is None:
            multiplier = 1

        # Map the Edge to its multiplier
        edge_multipliers[edge] = multiplier

    return edge_multipliers
