import osmium as osm
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Custom handler to extract nodes, ways (edges), and buildings
class OSMHandler(osm.SimpleHandler):
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.nodes = []
        self.node_dict = {}
        self.node_to_building = {}
        self.building_id_to_name = {}  # Map building IDs to names
        self.ways = []
        self.buildings = []

    # Handle node elements
    def node(self, n):
        tags = {tag.k: tag.v for tag in n.tags}

        # Check for entrance tags and standardize the value to 'yes'
        if 'entrance' in tags and tags['entrance'] in ['main', 'yes']:
            tags['entrance'] = 'yes'

        node_data = {
            'id': n.id,
            'lat': n.location.lat,
            'lon': n.location.lon,
            'tags': tags
        }

        self.nodes.append(node_data)
        self.node_dict[n.id] = node_data  # Store node data for later reference

    # Handle way elements
    def way(self, w):
        tags = {tag.k: tag.v for tag in w.tags}

        # Extract highways (edges) based on specific values
        if 'highway' in tags and tags['highway'] in [
            'unclassified', 'residential', 'service', 'living_street',
            'pedestrian', 'footway', 'cycleway', 'path', 'track', 'bridleway', 'steps'
        ]:
            self.ways.append({
                'id': w.id,
                'nodes': [n.ref for n in w.nodes],
                'tags': tags
            })

        # Extract buildings
        if 'building' in tags:
            building = {
                'id': w.id,
                'nodes': [n.ref for n in w.nodes],
                'tags': tags
            }
            self.buildings.append(building)

            # Map building ID to name (if any)
            building_name = tags.get('name')
            self.building_id_to_name[w.id] = building_name

            # Map nodes to this building
            for node_ref in building['nodes']:
                if node_ref in self.node_to_building:
                    self.node_to_building[node_ref].add(w.id)
                else:
                    self.node_to_building[node_ref] = {w.id}

# Initialize the handler and apply it to read the OSM file
handler = OSMHandler()
handler.apply_file('data/map.osm')

# Create three separate XML trees for nodes, ways, and buildings
nodes_root = ET.Element("osm", version="0.6", generator="python")
ways_root = ET.Element("osm", version="0.6", generator="python")
buildings_root = ET.Element("osm", version="0.6", generator="python")

# Write nodes to nodes.osm
for node in handler.nodes:
    node_element = ET.SubElement(nodes_root, "node", id=str(node['id']), lat=str(node['lat']), lon=str(node['lon']))

    # Add entrance tag if present
    if 'entrance' in node['tags']:
        ET.SubElement(node_element, "tag", k="entrance", v=node['tags']['entrance'])

        # Add building tag(s) with name if available
        building_ids = handler.node_to_building.get(node['id'], set())
        for building_id in building_ids:
            building_name = handler.building_id_to_name.get(building_id)
            if building_name:
                building_value = building_name
            else:
                building_value = str(building_id)
            ET.SubElement(node_element, "tag", k="building", v=building_value)

# Write ways (edges) to ways.osm
for way in handler.ways:
    way_element = ET.SubElement(ways_root, "way", id=str(way['id']))
    for node_ref in way['nodes']:
        ET.SubElement(way_element, "nd", ref=str(node_ref))
    if 'highway' in way['tags']:
        ET.SubElement(way_element, "tag", k="highway", v=way['tags']['highway'])

# Collect entrance nodes to include in buildings.osm
entrance_node_ids = set()
for building in handler.buildings:
    for node_ref in building['nodes']:
        node = handler.node_dict.get(node_ref)
        if node and 'entrance' in node['tags']:
            entrance_node_ids.add(node_ref)

# Write entrance nodes to buildings.osm
for node_id in entrance_node_ids:
    node = handler.node_dict[node_id]
    node_element = ET.SubElement(buildings_root, "node", id=str(node['id']), lat=str(node['lat']), lon=str(node['lon']))
    ET.SubElement(node_element, "tag", k="entrance", v=node['tags']['entrance'])

# Write buildings to buildings.osm
for building in handler.buildings:
    building_element = ET.SubElement(buildings_root, "way", id=str(building['id']))
    for node_ref in building['nodes']:
        ET.SubElement(building_element, "nd", ref=str(node_ref))
    if 'building' in building['tags']:
        ET.SubElement(building_element, "tag", k="building", v=building['tags']['building'])
    if 'name' in building['tags']:
        ET.SubElement(building_element, "tag", k="name", v=building['tags']['name'])

# Function to write a pretty-printed XML tree to a file
def write_pretty_xml(tree, filename):
    xml_str = ET.tostring(tree.getroot(), encoding='utf-8', xml_declaration=True)
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

# Write the trees to separate XML files with pretty-printing
write_pretty_xml(ET.ElementTree(nodes_root), "data/nodes.osm")
write_pretty_xml(ET.ElementTree(ways_root), "data/ways.osm")
write_pretty_xml(ET.ElementTree(buildings_root), "data/buildings.osm")

print("Files created with pretty formatting: nodes.osm, ways.osm, buildings.osm")
