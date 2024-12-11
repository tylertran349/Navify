# filename: entranceProjector.py

from polygonMaker import find_floor_plan_outline, parse_osm_buildings
from polygonComparer import (
    interpolate_polygon,
    calculate_polygon_centroid,
    calculate_polygon_area,
    scale_polygon_by_area,
    get_utm_crs,
    rotate_polygon,
    reflect_polygon,
    compute_iou,
)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyproj import Transformer, CRS
import numpy as np
import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import math
import sys
from database import Database  # Ensure correct import path


def project_entrances(
    database,
    building_name,
    floor_plan_image_path,
    buildings_osm_file_path='Code/data/buildings.osm',
    nodes_osm_file_path='Code/data/nodes.osm',
    interpolation_points=1000,
):
    """
    Projects the entrances of a building from global coordinates into local coordinates,
    moves them 50 pixels inwards relative to the projected global polygon,
    ensures they are placed on a white pixel in the floor plan image,
    scales the entrances, rounds them to the nearest pixel, and updates the Database.

    Parameters:
        database (Database): The Database object to update entrances in.
        building_name (str): The name of the building to process.
        floor_plan_image_path (str): The path to the floor plan image.
        buildings_osm_file_path (str): Path to the buildings OSM file.
        nodes_osm_file_path (str): Path to the nodes OSM file.
        interpolation_points (int): Number of points for polygon interpolation.

    Returns:
        None: The function updates the Database's entrance dictionaries in place.
    """
    # === Parse OSM Data ===
    buildings = parse_osm_buildings(buildings_osm_file_path, nodes_osm_file_path)
    # Get the building polygon and entrances for the given building name
    if building_name in buildings:
        building_data = buildings[building_name]
        polygons_global = building_data['polygons']  # List of polygons
        entrances_global = building_data['entrances']  # List of (lon, lat)
    else:
        raise ValueError(f"Building '{building_name}' not found in the OSM data.")

    #print(f"Processing building: {building_name}")
    #print(f"Number of global polygons: {len(polygons_global)}")
    #print(f"Number of global entrances: {len(entrances_global)}")

    # === Interpolate Polygon ===
    interpolated_polygons_global = []
    for polygon in polygons_global:
        interpolated = interpolate_polygon(
            polygon.copy(), num_points=interpolation_points
        )
        interpolated_polygons_global.append(interpolated)

    # === Coordinate Transformation ===
    # Assume the first point's latitude and longitude for CRS determination
    sample_lon, sample_lat = polygons_global[0][0]
    crs_utm = get_utm_crs(sample_lat, sample_lon)
    crs_wgs84 = CRS.from_epsg(4326)  # WGS84 Latitude/Longitude
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

    # Transform building polygons
    polygons_projected = []
    for polygon in interpolated_polygons_global:
        # Transformer expects (lon, lat), so ensure correct order
        projected = [transformer.transform(lon, lat) for lon, lat in polygon]
        polygons_projected.append(projected)

    # Transform entrances
    entrances_projected = []
    for lon, lat in entrances_global:
        x, y = transformer.transform(lon, lat)
        entrances_projected.append((x, y))

    # === Get Floor Plan Outline (Local Coordinates) ===
    floor_plan_polygon = find_floor_plan_outline(floor_plan_image_path)
    if floor_plan_polygon is None:
        raise ValueError("Failed to get floor plan outline.")

    # Interpolate floor plan polygon
    floor_plan_interpolated = interpolate_polygon(
        floor_plan_polygon.copy(), num_points=interpolation_points
    )

    # Load the floor plan image
    img = mpimg.imread(floor_plan_image_path)
    height, width = img.shape[:2]

    # Calculate centroids
    centroid_global = calculate_polygon_centroid(
        polygons_projected[0]
    )  # Assuming one global polygon
    centroid_local = calculate_polygon_centroid(floor_plan_interpolated)

    # === Scale Global Polygon and Entrances to Match Local Polygon's Area ===
    scaled_global_polygon = scale_polygon_by_area(
        polygons_projected[0], floor_plan_interpolated
    )

    # Scale entrances
    scale_factor = math.sqrt(
        calculate_polygon_area(floor_plan_interpolated)
        / calculate_polygon_area(polygons_projected[0])
    )
    scaled_entrances = []
    for x, y in entrances_projected:
        x_scaled = (x - centroid_global[0]) * scale_factor + centroid_global[0]
        y_scaled = (y - centroid_global[1]) * scale_factor + centroid_global[1]
        scaled_entrances.append((x_scaled, y_scaled))

    # Recalculate centroid of the scaled global polygon
    centroid_scaled_global = calculate_polygon_centroid(scaled_global_polygon)

    # === Translate Scaled Global Polygon and Entrances to Align Centroids ===
    translation_x = centroid_local[0] - centroid_scaled_global[0]
    translation_y = centroid_local[1] - centroid_scaled_global[1]
    translated_global_polygon = [
        (x + translation_x, y + translation_y) for x, y in scaled_global_polygon
    ]
    translated_entrances = [
        (x + translation_x, y + translation_y) for x, y in scaled_entrances
    ]

    # === Rotate and Reflect Global Polygon and Entrances to Maximize Similarity ===
    angles = np.arange(-180, 180, 1)  # Angles in degrees
    best_similarity = 0
    best_transformation = None
    best_aligned_polygon = None
    best_aligned_entrances = None
    for angle in angles:
        for reflection in [False, True]:
            # Rotate the polygon
            transformed_polygon = rotate_polygon(
                translated_global_polygon, angle, centroid_local
            )
            # Rotate the entrances
            transformed_entrances = rotate_polygon(
                translated_entrances, angle, centroid_local
            )
            # Reflect the polygon and entrances if needed
            if reflection:
                transformed_polygon = reflect_polygon(
                    transformed_polygon, axis='x', origin=centroid_local
                )
                transformed_entrances = reflect_polygon(
                    transformed_entrances, axis='x', origin=centroid_local
                )
            # Compute similarity
            iou = compute_iou(transformed_polygon, floor_plan_interpolated)
            # Update best similarity
            if iou > best_similarity:
                best_similarity = iou
                best_transformation = (angle, reflection)
                best_aligned_polygon = transformed_polygon
                best_aligned_entrances = transformed_entrances
    if best_aligned_entrances is None:
        raise ValueError("Failed to align entrances.")
    # Update the transformed entrances and polygon
    transformed_entrances = best_aligned_entrances
    transformed_polygon = best_aligned_polygon

    # === Move Entrances 50 Pixels Inwards and Ensure on White Pixel ===
    def is_white_pixel(img, point, height, width):
        """
        Check if the point is on a white pixel in the image.
        Parameters:
            img (numpy.ndarray): The image array.
            point (tuple): The point to check (x, y).
            height (int): The height of the image in pixels.
            width (int): The width of the image in pixels.
        Returns:
            bool: True if the pixel at the point is white, False otherwise.
        """
        x, y = point
        # Map coordinates to pixel indices
        pixel_x = int(round(x))
        pixel_y = int(round(y))
        if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
            return False  # Point is outside the image
        # Since y axis is flipped in display, we need to flip pixel_y
        pixel_y_flipped = height - pixel_y - 1
        pixel_value = img[pixel_y_flipped, pixel_x]
        # Check if pixel is white
        if len(pixel_value.shape) == 0 or len(pixel_value) == 1:
            # Grayscale image
            return pixel_value >= 0.9
        else:
            # Color image
            return np.all(pixel_value >= 0.9)

    def move_point_inwards_to_white_pixel(polygon, point, distance, img, height, width):
        """
        Move a point into the polygon along the inward normal by a specified distance,
        and adjust it to be on a white pixel in the image.
        Parameters:
            polygon (shapely.geometry.Polygon): The polygon to consider.
            point (tuple): The point to move (x, y).
            distance (float): The distance to move the point inwards.
            img (numpy.ndarray): The image array.
            height (int): The height of the image.
            width (int): The width of the image.
        Returns:
            tuple: The new point moved inwards and adjusted to a white pixel.
        """
        # Convert point to shapely Point
        entrance_point = Point(point)
        # Get the nearest point on the polygon boundary
        nearest_point = nearest_points(entrance_point, polygon.exterior)[1]
        # Find the segment on the polygon boundary closest to the nearest_point
        coords = list(polygon.exterior.coords)
        min_dist = float('inf')
        closest_segment = None
        for i in range(len(coords) - 1):
            segment = (coords[i], coords[i + 1])
            line = shapely.geometry.LineString(segment)
            projected_point = line.interpolate(line.project(nearest_point))
            dist = projected_point.distance(nearest_point)
            if dist < min_dist:
                min_dist = dist
                closest_segment = segment
        if closest_segment is None:
            # Should not happen
            raise ValueError("Failed to find the closest segment on the polygon boundary.")
        # Compute the direction vector of the segment
        (x1, y1), (x2, y2) = closest_segment
        segment_vector = np.array([x2 - x1, y2 - y1])
        if np.linalg.norm(segment_vector) == 0:
            # Degenerate segment
            return point
        # Compute the normal vector (perpendicular to the segment)
        normal_vector = np.array([-segment_vector[1], segment_vector[0]])
        # Normalize the normal vector
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        # Ensure the normal vector points into the polygon
        test_point = np.array(nearest_point.coords[0]) + 0.1 * normal_vector
        if not polygon.contains(Point(test_point)):
            # Reverse the normal vector
            normal_vector = -normal_vector
        # Move the nearest_point along the normal vector into the polygon by 'distance'
        new_point_coords = np.array(nearest_point.coords[0]) + normal_vector * distance
        new_point = (new_point_coords[0], new_point_coords[1])
        # Check if the new point is on a white pixel
        if is_white_pixel(img, new_point, height, width):
            return new_point
        # If not, search in a small neighborhood along the normal vector
        step_size = 1.0  # pixels
        max_steps = 50  # Max steps to search
        for step in range(1, int(max_steps)):
            adjusted_point_coords = new_point_coords + normal_vector * step_size * step
            adjusted_point = (adjusted_point_coords[0], adjusted_point_coords[1])
            if is_white_pixel(img, adjusted_point, height, width):
                return adjusted_point
        # If still not found, try moving in the opposite direction
        for step in range(1, int(max_steps)):
            adjusted_point_coords = new_point_coords - normal_vector * step_size * step
            adjusted_point = (adjusted_point_coords[0], adjusted_point_coords[1])
            if is_white_pixel(img, adjusted_point, height, width):
                return adjusted_point
        # If still not found, return the original new_point
        return new_point

    # === Update Indoor and Outdoor Entrances in the Database ===
    # Moving the entrances inwards
    adjusted_entrances = []
    for entrance in transformed_entrances:
        # Move the entrance 50 pixels inwards relative to the aligned global polygon
        adjusted_entrance = move_point_inwards_to_white_pixel(
            Polygon(transformed_polygon), entrance, 50, img, height, width  # Corrected distance to 50
        )
        # Round to nearest pixel
        adjusted_entrance_rounded = (round(adjusted_entrance[0]), round(adjusted_entrance[1]))
        adjusted_entrances.append(adjusted_entrance_rounded)
        #print(f"Adjusted indoor entrance from {entrance} to {adjusted_entrance_rounded}")

    # Add entrances to the Database
    database.add_outdoor_entrances(building_name, entrances_global)
    database.add_indoor_entrances(building_name, adjusted_entrances)

    # === Optional: Transform Outdoor Entrances to Local Coordinates ===
    # This step assumes that outdoor entrances need to be transformed similarly.
    # If not required, you can skip this.
    # For demonstration, let's assume they are already in local coordinates.
    # If needed, implement transformation similar to indoor entrances.


def main():
    """
    Main function to demonstrate projecting entrances, scaling, and updating the Database.
    It updates the entrances within the Database object with the processed data.
    """
    from databaseInitiator import DatabaseInitiator  # Ensure correct import path

    buildings = [
        {
            "name": "Memorial Union",
            "floor_plan_image_path": "Code/CroppedImages/MUFirstFloor.png"
        },
        {
            "name": "Walker Hall",
            "floor_plan_image_path": "Code/CroppedImages/WalkerHallFirstFloor.png"
        }
    ]

    # Initialize the database
    data = DatabaseInitiator()
    database = data.get_database()

    #print("Initial Indoor Entrances Dictionary:", database.indoor_entrance_dict)
    #print("Initial Outdoor Entrances Dictionary:", database.outdoor_entrance_dict)
    #print("\nStarting entrance projection...\n")

    for building in buildings:
        building_name = building["name"]
        floor_plan_image_path = building["floor_plan_image_path"]
        try:
            project_entrances(
                database=database,
                building_name=building_name,
                floor_plan_image_path=floor_plan_image_path
            )
            #print(f"Successfully projected entrances for '{building_name}'.\n")
        except Exception as e:
            #print(f"Error projecting entrances for '{building_name}': {e}\n")
            pass

    # Now, scale the indoor entrances for "Walker Hall"
    try:
        scale_factor = 0.6  # Example scale factor
        database.scale_indoor_entrances_for_building("Walker Hall", scale_factor)
        #print(f"Successfully scaled indoor entrances for 'Walker Hall' by a factor of {scale_factor}.\n")
    except ValueError as ve:
        #print(f"Scaling Error: {ve}\n")
        pass

    # Verify the updated indoor entrances
    indoor, outdoor = database.get_building_entrances("Walker Hall")
    #print(f"Indoor Entrances for 'Walker Hall': {indoor}")
    #print(f"Outdoor Entrances for 'Walker Hall': {outdoor}\n")

    # Plot the floor plan with entrances
    floor_plan_image_path = "Code/CroppedImages/WalkerHallFirstFloor.png"
    img = mpimg.imread(floor_plan_image_path)
    height, width = img.shape[:2]
    plt.figure(figsize=(10, 8))
    plt.imshow(img, extent=[0, width, height, 0], aspect='auto')

    # Plot the adjusted and scaled indoor entrances
    if indoor:
        x, y = zip(*indoor)
        plt.scatter(
            x,
            y,
            color="red",
            edgecolor="black",
            s=100,
            label="Indoor Entrances",
            zorder=5
        )

    # Plot the outdoor entrances transformed to local coordinates
    if outdoor:
        # Assuming outdoor entrances have been transformed appropriately in project_entrances
        x_out, y_out = zip(*outdoor)
        plt.scatter(
            x_out,
            y_out,
            color="blue",
            edgecolor="black",
            s=100,
            label="Outdoor Entrances",
            marker='x',
            zorder=5
        )

    plt.title("Entrances on Floor Plan", fontsize=16)
    plt.xlabel("X (pixels)", fontsize=12)
    plt.ylabel("Y (pixels)", fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    plt.axis("equal")
    plt.grid(False)  # Remove grid lines
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
