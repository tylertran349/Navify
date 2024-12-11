import csv
import requests
from graph import Node, Edge  # Import Node and Edge classes
from database import Database  # Import Database class

#Function to get the construction zones from the ArcGIS API from UCDAVIS ROAD CLOSURE MAP 
def fetch_construction_zones():
    url = "https://gis.ucdavis.edu/server/rest/services/TAPS_Road_Circulation_Updates/MapServer/2/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson"
    }
    response = requests.get(url, params=params)
    construction_zones = []

    if response.status_code == 200:
        data = response.json()
        for feature in data["features"]:
            geometry = feature["geometry"]
            if geometry["type"] == "Polygon":
                coordinates = geometry["coordinates"][0]
                construction_zones.append(coordinates)
    else:
        print("Failed to retrieve data:", response.status_code)
    
    return construction_zones

# Set up nodes and edges for testing
node1 = Node(x=-121.752, y=38.538, use_global_coords=True, building_id="Outdoor")
node2 = Node(x=-121.753, y=38.539, use_global_coords=True, building_id="Outdoor")
edge1 = Edge(node1=node1, node2=node2, baseline_weight=1.0)

node_dict = {"Outdoor": [node1, node2]}
edge_dict = {"Outdoor": [edge1]}
indoor_entrance_dict = {}
outdoor_entrance_dict = {}

# REPRESENT the Database with the node and edge dictionaries
db = Database(node_dict=node_dict, edge_dict=edge_dict, 
              indoor_entrance_dict=indoor_entrance_dict, 
              outdoor_entrance_dict=outdoor_entrance_dict)

##############################################################################################################################

#MAIN ACTION: Get construction zones and find affected edges
construction_zones = fetch_construction_zones()
if construction_zones:
    affected_edges = db.get_edges_within_polygons(construction_zones)
    
    # Mark affected edges as closed
    for edges_in_zone in affected_edges:
        for edge in edges_in_zone:
            db.update_edge_status(edge, closed=True)  

##############################################################################################################################            

# Class to load local construction data from CSV
class constructionDataRetriever:
    def __init__(self, data_source):
        self.data_source = data_source
        self.construction_locations = []  #Stores the polygons of construction areas

    def load_data(self):
        try:
            with open(self.data_source, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    polygon = [(float(row[i]), float(row[i+1])) for i in range(0, len(row), 2)]
                    self.construction_locations.append(polygon)
        except Exception as e:
            print(f"Error loading data from {self.data_source}: {e}")

    def get_construction_data(self):
        if not self.construction_locations:
            self.load_data()
        return self.construction_locations