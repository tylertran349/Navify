import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
from pyproj import Transformer
import xml.etree.ElementTree as ET  # Ensure this is imported
from scipy.spatial.distance import pdist

def find_floor_plan_outline(image_path, 
                            kernel_size=5, 
                            epsilon_ratio=0.001, 
                            display=False):
    """
    Finds the outer outline of a floor plan image and returns it as a closed polygon.

    Parameters:
    - image_path (str): Path to the input floor plan PNG image.
    - kernel_size (int): Size of the kernel for morphological operations. Default is 5.
    - epsilon_ratio (float): Approximation accuracy as a ratio of the contour perimeter. Default is 0.01.
    - display (bool): If True, displays the image with the outline. Default is False.

    Returns:
    - polygon (list of tuples): List of (x, y) coordinates representing the closed outer outline.
                                 Returns None if no contour is found or an error occurs.
    """

    try:
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Unable to load image at '{image_path}'.")
            return None
        
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
            return None
        
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
        
        return polygon
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

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
    # relations parsing can be added here if needed

    # First, parse the nodes OSM file to collect nodes
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

    # Now parse the buildings OSM file
    tree_buildings = ET.parse(buildings_osm_file_path)
    root_buildings = tree_buildings.getroot()

    # Collect any nodes in the buildings file (in case there are additional nodes)
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

    # Initialize the buildings dictionary
    buildings = {}  # building_id_or_name -> {'polygons': [list of polygons], 'entrances': [list of (lon, lat)]}

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
                # Append to existing building data
                buildings[building_id]['polygons'].append(polygon)
                buildings[building_id]['entrances'].extend(entrances)
            else:
                buildings[building_id] = building_info

    return buildings

