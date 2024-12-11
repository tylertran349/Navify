# filename: entrance_mapper.py

from entranceProjector import project_entrances
from graph import Node
import sys

def map_entrances_to_nodes(database, building_name, floor_plan_image_path):
    """
    Projects the entrances of a building from global to local coordinates,
    finds the nearest nodes in the database to these coordinates, and updates
    the database's entrance dictionaries and corresponding nodes.

    Parameters:
        database (Database): The database instance.
        building_name (str): The name of the building to process.
        floor_plan_image_path (str): The path to the floor plan image.

    Returns:
        None
    """
    
    # === Project Entrances ===
    try:
        # Project entrances into local coordinates and get global entrances
        adjusted_entrances, entrances_global = project_entrances(database, building_name, floor_plan_image_path)
    except Exception as e:
        print(f"Error in projecting entrances for building '{building_name}': {e}")
        return

    # === Find Nearest Nodes ===

    # Indoor entrances: adjusted_entrances (local coordinates between 0 and 1)
    # Outdoor entrances: entrances_global (global coordinates)

    # Find nearest indoor nodes
    indoor_nodes = []
    for x, y in adjusted_entrances:
        try:
            nearest_node = database.find_nearest_node(building_name, x, y)
            indoor_nodes.append(nearest_node)
        except ValueError as ve:
            print(f"Error finding nearest indoor node: {ve}")
            continue

    # Find nearest outdoor nodes
    outdoor_nodes = []
    for lon, lat in entrances_global:
        try:
            nearest_node = database.find_nearest_node('Outdoor', lon, lat)
            outdoor_nodes.append(nearest_node)
        except ValueError as ve:
            print(f"Error finding nearest outdoor node: {ve}")
            continue

    # === Update Database Dictionaries ===

    # Update indoor_entrance_dict
    if building_name not in database.indoor_entrance_dict:
        database.indoor_entrance_dict[building_name] = []
    database.indoor_entrance_dict[building_name].extend(indoor_nodes)

    # Update outdoor_entrance_dict
    if building_name not in database.outdoor_entrance_dict:
        database.outdoor_entrance_dict[building_name] = []
    database.outdoor_entrance_dict[building_name].extend(outdoor_nodes)

    # Update corresponding_node dictionary
    for indoor_node, outdoor_node in zip(indoor_nodes, outdoor_nodes):
        database.corresponding_node[outdoor_node] = indoor_node
        database.corresponding_node[indoor_node] = outdoor_node

    print(f"Entrances mapped for building '{building_name}'.")
