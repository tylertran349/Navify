import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
from pyproj import Transformer
import xml.etree.ElementTree as ET
from scipy.spatial.distance import pdist

def find_floor_plan_outline(image_path, 
                            kernel_size=5, 
                            epsilon_ratio=0.001, 
                            display=False):
    """
    Finds the outer outline of a floor plan image and returns it as a closed polygon,
    along with all detected contours and the largest contour.

    Parameters:
    - image_path (str): Path to the input floor plan PNG image.
    - kernel_size (int): Size of the kernel for morphological operations. Default is 5.
    - epsilon_ratio (float): Approximation accuracy as a ratio of the contour perimeter.
    - display (bool): If True, displays the image with the outline. Default is False.

    Returns:
    - polygon (list of tuples): Closed outer outline polygon (x, y).
    - contours (list of np.ndarray): All detected contours.
    - max_contour (np.ndarray): The largest contour found.
    """

    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at '{image_path}'.")
            return None, None, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image to binary (invert if necessary)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological closing to close small gaps
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found in the image.")
            return None, None, None
        
        # Select the largest contour assuming it's the outer outline
        max_contour = max(contours, key=cv2.contourArea)
        
        # Approximate the contour to reduce the number of points
        epsilon = epsilon_ratio * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # Extract (x, y) coordinates
        polygon = [(point[0][0], point[0][1]) for point in approx]
        
        # Ensure the polygon is closed
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        
        if display:
            # Draw the outline on the original image
            outlined_image = img.copy()
            cv2.drawContours(outlined_image, [approx], -1, (0, 255, 0), 2)  # Green outline
            
            # Display the image
            cv2.imshow('Floor Plan with Outer Outline', outlined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return polygon, contours, max_contour
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def parse_osm_buildings(buildings_osm_file_path, nodes_osm_file_path):
    """
    Parses building polygons and entrances from OSM files.

    :param buildings_osm_file_path: Path to the OSM file containing buildings (ways and relations).
    :param nodes_osm_file_path: Path to the OSM file containing nodes.
    :return: Dictionary of buildings with their polygons and entrances.
    """
    # Dictionaries to hold the parsed data
    nodes = {}      # node_id -> {'coords': (lon, lat), 'tags': {tag_key: tag_value}}
    ways = {}       # way_id -> {'nodes': [node_ids], 'tags': {tag_key: tag_value}}

    # Parse nodes file
    tree_nodes = ET.parse(nodes_osm_file_path)
    root_nodes = tree_nodes.getroot()

    # Collect all nodes from the nodes file
    for element in root_nodes.findall('node'):
        node_id = element.attrib['id']
        lat = float(element.attrib['lat'])
        lon = float(element.attrib['lon'])
        tags = {}
        for child in element:
            if child.tag == 'tag':
                tags[child.attrib['k']] = child.attrib['v']
        nodes[node_id] = {'coords': (lon, lat), 'tags': tags}

    # Parse buildings file
    tree_buildings = ET.parse(buildings_osm_file_path)
    root_buildings = tree_buildings.getroot()

    # Collect any additional nodes found in the buildings file
    for element in root_buildings.findall('node'):
        node_id = element.attrib['id']
        lat = float(element.attrib['lat'])
        lon = float(element.attrib['lon'])
        tags = {}
        for child in element:
            if child.tag == 'tag':
                tags[child.attrib['k']] = child.attrib['v']
        nodes[node_id] = {'coords': (lon, lat), 'tags': tags}

    # Collect all ways
    for element in root_buildings.findall('way'):
        way_id = element.attrib['id']
        nds = []
        tags = {}
        for child in element:
            if child.tag == 'nd':
                nds.append(child.attrib['ref'])
            elif child.tag == 'tag':
                tags[child.attrib['k']] = child.attrib['v']
        ways[way_id] = {'nodes': nds, 'tags': tags}

    buildings = {}

    # Process ways that are buildings
    for way_id, way_data in ways.items():
        tags = way_data['tags']
        if 'building' in tags:
            building_id = tags.get('name', way_id)
            node_ids = way_data['nodes']
            polygon = []
            entrances = []
            missing_nodes = []
            for node_id in node_ids:
                if node_id in nodes:
                    node = nodes[node_id]
                    lon_lat = node['coords']
                    polygon.append(lon_lat)
                    if 'entrance' in node['tags']:
                        entrances.append(lon_lat)
                else:
                    missing_nodes.append(node_id)
            if missing_nodes:
                print(f"Warning: Nodes {missing_nodes} are missing for building '{building_id}'.")
            building_info = {'polygons': [polygon], 'entrances': entrances}
            if building_id in buildings:
                buildings[building_id]['polygons'].append(polygon)
                buildings[building_id]['entrances'].extend(entrances)
            else:
                buildings[building_id] = building_info

    return buildings


def plot_all_contours_with_image(image, contours, title='All Contours'):
    """
    Plots all contours on a single plot with the given image as the background.
    Removes axes, grid, and labels for a clean visualization.

    Parameters:
        image (numpy.ndarray): The background image.
        contours (list): A list of contours to plot.
        title (str): Title of the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    #ax.set_title(title, fontsize=14)
    
    # Show the image as background (convert BGR to RGB if needed)
    # OpenCV loads images in BGR, matplotlib expects RGB
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot all contours
    for contour in contours:
        contour = contour.reshape(-1, 2)
        x, y = contour[:, 0], contour[:, 1]
        ax.plot(x, y, color='red', linewidth=1)
    
    ax.axis('off')  # Remove axes, grid, and labels
    plt.show()


def plot_best_contour_with_image(image, best_contour, title='Best Contour'):
    """
    Plots the best (largest) contour on a separate plot with the image as background.
    Removes axes, grid, and labels for a clean visualization.

    Parameters:
        image (numpy.ndarray): The background image.
        best_contour (numpy.ndarray): The largest contour to plot.
        title (str): Title of the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    #ax.set_title(title, fontsize=14)
    
    # Show the image as background
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot the best contour
    best_contour = best_contour.reshape(-1, 2)
    x, y = best_contour[:, 0], best_contour[:, 1]
    ax.plot(x, y, color='blue', linewidth=2)
    
    ax.axis('off')  # Remove axes, grid, and labels
    plt.show()


def main():
    # Path to the image
    image_path = "demo/hdFloorPlans/Activities and Recreation Center.png"
    
    # Run the outline extraction function
    polygon, contours, max_contour = find_floor_plan_outline(image_path, kernel_size=5, epsilon_ratio=0.001, display=False)
    
    if polygon is None or contours is None or max_contour is None:
        print("Could not process contours due to an earlier error.")
        return
    
    # Load the image again for plotting (already loaded inside function, but let's load again for clarity)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image for plotting from '{image_path}'.")
        return
    
    # Plot all contours in one plot
    print(contours)
    plot_all_contours_with_image(img, contours, title='All Contours')
    
    # Plot just the best (largest) contour in another plot
    plot_best_contour_with_image(img, max_contour, title='Best Contour')


if __name__ == "__main__":
    main()
