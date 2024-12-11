# filename: plot_with_heatmap.py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import pandas as pd
import numpy as np
import os
import pickle
from database import Database  # Ensure Node and Edge classes are defined appropriately
from graph import Node, Edge  # Ensure these classes are properly defined in graph.py

# Existing Haversine function
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth.
    
    Parameters:
    - lat1, lon1: Latitude and longitude of the first point in degrees.
    - lat2, lon2: Latitude and longitude of the second point in degrees.
    
    Returns:
    - Distance in feet.
    """
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
    """
    Calculates the crime multiplier for a grid cell based on proximity to crimes.
    
    Parameters:
    - lat_lower, lat_upper: Latitude bounds of the grid cell.
    - lon_lower, lon_upper: Longitude bounds of the grid cell.
    - crime_df (DataFrame): DataFrame containing crime locations with 'lat' and 'long' columns.
    
    Returns:
    - Multiplier (int).
    """
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
    """
    Calculates the phone multiplier for a grid cell based on proximity to phones.
    
    Parameters:
    - lat_lower, lat_upper: Latitude bounds of the grid cell.
    - lon_lower, lon_upper: Longitude bounds of the grid cell.
    - phone_df (DataFrame): DataFrame containing phone locations with 'lat' and 'long' columns.
    
    Returns:
    - Multiplier (float) or None.
    """
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

# Function to load crime and phone locations
def load_locations():
    """
    Loads crime and phone locations from CSV files.
    
    Returns:
    - crime_df (DataFrame): DataFrame containing crime locations.
    - phone_df (DataFrame): DataFrame containing phone locations.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_crime = os.path.join(current_dir, "crime_locs.txt")
    file_path_phone = os.path.join(current_dir, "phone_locs.txt")
    
    # Check if files exist
    if not os.path.exists(file_path_crime):
        raise FileNotFoundError(f"Crime locations file '{file_path_crime}' not found.")
    if not os.path.exists(file_path_phone):
        raise FileNotFoundError(f"Phone locations file '{file_path_phone}' not found.")
    
    # Load data
    crime_df = pd.read_csv(file_path_crime, header=None, names=["lat", "long"])
    phone_df = pd.read_csv(file_path_phone, header=None, names=["lat", "long"])
    
    return crime_df, phone_df

# Function to calculate grid multipliers
def calculate_grid_multipliers(lat_min, lat_max, lon_min, lon_max, crime_df, phone_df, num_squares_per_side):
    """
    Divides the geographic area into a grid and calculates multipliers for each grid square.
    
    Parameters:
    - lat_min (float): Minimum latitude.
    - lat_max (float): Maximum latitude.
    - lon_min (float): Minimum longitude.
    - lon_max (float): Maximum longitude.
    - crime_df (DataFrame): DataFrame with crime locations.
    - phone_df (DataFrame): DataFrame with phone locations.
    - num_squares_per_side (int): Number of squares per side of the grid.
    
    Returns:
    - grid_data (dict): Dictionary mapping grid squares to multipliers.
    """
    lat_step = (lat_max - lat_min) / num_squares_per_side
    lon_step = (lon_max - lon_min) / num_squares_per_side
    
    grid_data = {}
    
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

            # Set final multiplier to 0 if both crime and phone multipliers are 0
            if final_multiplier == 0:
                final_multiplier = 0

            # Store in dictionary
            grid_data[((lat_lower, lat_upper), (lon_lower, lon_upper))] = final_multiplier
    
    return grid_data

# Function to get the multiplier for a specific latitude and longitude
def get_multiplier(lat, lon, grid_data):
    """
    Retrieves the multiplier for a specific latitude and longitude from the grid data.
    
    Parameters:
    - lat (float): Latitude of the point.
    - lon (float): Longitude of the point.
    - grid_data (dict): Dictionary mapping grid squares to multipliers.
    
    Returns:
    - Multiplier (float) or 0.
    """
    for ((lat_min, lat_max), (lon_min, lon_max)), multiplier in grid_data.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return multiplier
    return 0  # Default to 0 if not found

# --- Plotting and Integration ---

class EdgeDisplayWithHeatmap:
    def __init__(self, database, grid_data, lat_min, lat_max, lon_min, lon_max):
        """
        Initializes the plot with heatmap and edges.
        
        Parameters:
        - database (Database): The database object containing edges.
        - grid_data (dict): Dictionary mapping grid squares to multipliers.
        - lat_min, lat_max (float): Latitude bounds.
        - lon_min, lon_max (float): Longitude bounds.
        """
        self.database = database
        self.grid_data = grid_data
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        # Prepare the plot
        self.setup_plot()

        # Plot the heatmap
        self.plot_heatmap()

        # Plot the edges
        self.plot_edges()

        # Add colorbar
        self.add_colorbar()

        # Show the plot
        plt.show()

    def setup_plot(self):
        """
        Sets up the matplotlib figure and axes with a white background and equal aspect ratio.
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_facecolor('white')  # Set background to white

        # Set labels
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')

        # Set title
        # self.ax.set_title('Crime Multiplier Heatmap with Edges')

        # Ensure equal aspect ratio to prevent distortion
        self.ax.set_aspect('equal', adjustable='datalim')

    def plot_heatmap(self):
        """
        Plots the crime multiplier heatmap as a grid, covering the entire plotting area.
        """
        # Extract grid boundaries
        lat_bounds = sorted(list(set([bound for ((lat_min, lat_max), _) in self.grid_data.keys() for bound in (lat_min, lat_max)])))
        lon_bounds = sorted(list(set([bound for ((_, _), (lon_min, lon_max)) in self.grid_data.keys() for bound in (lon_min, lon_max)])))

        lat_bins = lat_bounds
        lon_bins = lon_bounds

        # Create a 2D array for multipliers
        grid_multipliers = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
        for ((lat_min, lat_max), (lon_min, lon_max)), multiplier in self.grid_data.items():
            # Find the index
            i = lat_bins.index(lat_min)
            j = lon_bins.index(lon_min)
            grid_multipliers[i, j] = multiplier

        # Define the custom colormap: 0 -> light gray, >0 follows viridis
        viridis = plt.cm.get_cmap('viridis', 256)
        colors = viridis(np.linspace(0, 1, 256))
        colors[0] = [0.9, 0.9, 0.9, 1]  # Set the first color (multiplier 0) to light gray
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('viridis_lightgray', colors)

        # Normalize from 0 to max multiplier
        norm = mcolors.Normalize(vmin=0, vmax=np.max(grid_multipliers))

        # Plot the heatmap using pcolormesh
        mesh = self.ax.pcolormesh(
            lon_bins,
            lat_bins,
            grid_multipliers,
            cmap=custom_cmap,
            norm=norm,
            shading='auto',
            alpha=0.8  # Adjust transparency as needed
        )

        self.heatmap = mesh  # Store reference for colorbar

        # Determine plot limits based on grid and edges
        edge_lats = [node.y for edge in self.database.get_all_edges() for node in [edge.node1, edge.node2]]
        edge_lons = [node.x for edge in self.database.get_all_edges() for node in [edge.node1, edge.node2]]

        # Calculate overall min and max with a small buffer
        buffer = 0.001  # Approximately 111 feet per 0.001 degree
        overall_lat_min = min(self.lat_min, min(edge_lats, default=self.lat_min)) - buffer
        overall_lat_max = max(self.lat_max, max(edge_lats, default=self.lat_max)) + buffer
        overall_lon_min = min(self.lon_min, min(edge_lons, default=self.lon_min)) - buffer
        overall_lon_max = max(self.lon_max, max(edge_lons, default=self.lon_max)) + buffer

        self.ax.set_xlim(overall_lon_min, overall_lon_max)
        self.ax.set_ylim(overall_lat_min, overall_lat_max)

    def plot_edges(self):
        """
        Plots all edges from the database in a uniform, more visible black color.
        """
        edges = self.database.get_all_edges()

        for edge in edges:
            node1 = edge.node1
            node2 = edge.node2

            # Extract coordinates
            x_coords = [node1.x, node2.x]
            y_coords = [node1.y, node2.y]

            # Plot the edge in uniform black color with increased visibility
            self.ax.plot(
                x_coords,
                y_coords,
                color='black',      # Uniform black color for all edges
                linewidth=1.5,      # Increased linewidth for better visibility
                alpha=0.7           # Increased opacity for better visibility
            )

    def add_colorbar(self):
        """
        Adds a colorbar to the plot to indicate the mapping from crime multipliers to colors.
        """
        # Heatmap colorbar
        cbar = self.fig.colorbar(self.heatmap, ax=self.ax, orientation='vertical', label='Crime Multiplier')
        
        # Set specific ticks based on multipliers
        max_multiplier = int(np.max([v for v in self.grid_data.values()]))
        ticks = range(0, max_multiplier + 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(tick) for tick in ticks])

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

def main():
    # Load crime and phone locations
    try:
        crime_df, phone_df = load_locations()
    except FileNotFoundError as e:
        print(e)
        print("Please ensure 'crime_locs.txt' and 'phone_locs.txt' are present in the script directory.")
        return

    # Define latitude and longitude bounds for the grid
    # These should match the area covered by your data
    lat_min, lat_max = 38.53410627387911, 38.54626869715259
    lon_min, lon_max = -121.76112584132399, -121.7455565855204

    # Set grid parameters
    num_squares_per_side = 31  # Since sqrt(1000) ≈ 31.62

    # Calculate grid multipliers
    grid_data = calculate_grid_multipliers(lat_min, lat_max, lon_min, lon_max, crime_df, phone_df, num_squares_per_side)

    # Load the Database
    try:
        db = load_database()
    except FileNotFoundError as e:
        print(e)
        print("Creating mock data for testing purposes.")

        # Create a new Database instance
        db = Database()

        # Example: Populate the Database with mock data
        node_a = Node(x=-121.7617, y=38.5382)  # Example coordinates
        node_b = Node(x=-121.7610, y=38.5390)
        node_c = Node(x=-121.7620, y=38.5385)
        node_d = Node(x=-121.7605, y=38.5375)
        node_e = Node(x=-121.7630, y=38.5400)

        edge_ab = Edge(node1=node_a, node2=node_b)
        edge_ac = Edge(node1=node_a, node2=node_c)
        edge_ad = Edge(node1=node_a, node2=node_d)
        edge_be = Edge(node1=node_b, node2=node_e)
        edge_ce = Edge(node1=node_c, node2=node_e)
        edge_de = Edge(node1=node_d, node2=node_e)

        # Add edges to the outdoor adjacency list
        db.outdoor_adjacency_list[node_a] = [(node_b, edge_ab), (node_c, edge_ac), (node_d, edge_ad)]
        db.outdoor_adjacency_list[node_b] = [(node_a, edge_ab), (node_e, edge_be)]
        db.outdoor_adjacency_list[node_c] = [(node_a, edge_ac), (node_e, edge_ce)]
        db.outdoor_adjacency_list[node_d] = [(node_a, edge_ad), (node_e, edge_de)]
        db.outdoor_adjacency_list[node_e] = [(node_b, edge_be), (node_c, edge_ce), (node_d, edge_de)]

    # Initialize and run the edge display with heatmap
    EdgeDisplayWithHeatmap(database=db, grid_data=grid_data, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

if __name__ == "__main__":
    main()
