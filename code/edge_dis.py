# filename: display_edges.py

import matplotlib.pyplot as plt
import math
import os
from database import Database  # Ensure Node and Edge classes are defined appropriately
import pickle
from graph import Node, Edge  # Ensure these classes are properly defined in graph.py

class EdgeDisplay:
    def __init__(self, database):
        self.database = database

        # Prepare the plot
        self.setup_plot()

        # Plot the edges
        self.plot_edges()

        # Add a colorbar to the plot
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
        #self.ax.set_title('Edge Display with Orientation-Based Colors')

        # Ensure equal aspect ratio to prevent distortion
        self.ax.set_aspect('equal', adjustable='datalim')

    def calculate_orientation(self, node1, node2):
        """
        Calculates the orientation angle between two nodes in degrees.
        Maps the angle to [0, 180) to ensure that edges facing 180 degrees apart have the same color.
        """
        delta_x = node2.x - node1.x
        delta_y = node2.y - node1.y
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = (math.degrees(angle_rad) + 360) % 360  # Normalize to [0, 360)
        angle_deg = angle_deg if angle_deg < 180 else angle_deg - 180  # Map to [0, 180)
        return angle_deg

    def angle_to_color(self, angle_deg):
        """
        Converts an angle in degrees [0, 180) to an RGB color.
        Creates a gradient from strong red to strong blue.
        - 0°: Strong Red (#FF0000)
        - 90°: Strong Blue (#0000FF)
        - 180°: Strong Red (#FF0000)
        """
        # Normalize angle to [0, 180)
        angle_norm = angle_deg % 180

        # Calculate interpolation factor
        if angle_norm <= 90:
            factor = angle_norm / 90
        else:
            factor = (180 - angle_norm) / 90

        # Interpolate between red and blue
        r = int(255 * (1 - factor))
        g = 0
        b = int(255 * factor)

        # Return color as RGB tuple normalized to [0,1]
        color = (r / 255, g / 255, b / 255)
        return color

    def plot_edges(self):
        """
        Plots all edges from the database with colors based on their orientation.
        """
        edges = self.database.get_all_edges()

        # Collect all longitude and latitude values to determine plot limits
        all_longitudes = []
        all_latitudes = []

        for edge in edges:
            node1 = edge.node1
            node2 = edge.node2

            # Collect coordinates
            all_longitudes.extend([node1.x, node2.x])
            all_latitudes.extend([node1.y, node2.y])

            # Calculate orientation
            angle = self.calculate_orientation(node1, node2)
            color = self.angle_to_color(angle)

            # Plot the edge
            self.ax.plot(
                [node1.x, node2.x],
                [node1.y, node2.y],
                color=color,
                linewidth=1.5,
                alpha=0.7
            )

        # Set plot limits with some padding
        if all_longitudes and all_latitudes:
            lon_min, lon_max = min(all_longitudes), max(all_longitudes)
            lat_min, lat_max = min(all_latitudes), max(all_latitudes)
            lon_padding = (lon_max - lon_min) * 0.05
            lat_padding = (lat_max - lat_min) * 0.05
            self.ax.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
            self.ax.set_ylim(lat_min - lat_padding, lat_max + lat_padding)

    def add_colorbar(self):
        """
        Adds a colorbar to the plot to indicate the mapping from orientation angles to colors.
        """
        # Create a ScalarMappable with the same colormap used for edges
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        # Define a custom colormap from red to blue
        cmap = cm.get_cmap('coolwarm')  # Alternatively, create a custom colormap

        # Normalize from 0 to 180 degrees
        norm = Normalize(vmin=0, vmax=180)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for older versions of matplotlib

        # Add colorbar to the figure
        cbar = self.fig.colorbar(sm, ax=self.ax, orientation='vertical', pad=0.02)
        cbar.set_label('Orientation Angle (degrees)')

        # Customize the colorbar ticks to match our color mapping
        cbar.set_ticks([0, 45, 90, 135, 180])
        cbar.set_ticklabels(['0°', '45°', '90°', '135°', '180°'])

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
    try:
        # Initialize the Database
        db = load_database()
    except FileNotFoundError as e:
        print(e)
        print("Creating mock data for testing purposes.")

        # Create a new Database instance
        db = Database()

    # Initialize and run the edge display
    EdgeDisplay(database=db)

if __name__ == "__main__":
    main()
