from polygonMaker import find_floor_plan_outline, parse_osm_buildings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyproj import Transformer, CRS
import numpy as np
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
import sys
import math

def interpolate_polygon(polygon, num_points=1000):
    """
    Interpolate a polygon to have a specified number of evenly spaced points along its perimeter.
    """
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least three vertices for interpolation.")

    # Close the polygon loop if not already closed
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    # Calculate the cumulative distance along the polygon perimeter
    distances = [0]
    for i in range(1, len(polygon)):
        prev_point = np.array(polygon[i - 1])
        curr_point = np.array(polygon[i])
        distance = np.linalg.norm(curr_point - prev_point)
        distances.append(distances[-1] + distance)

    total_length = distances[-1]
    interpolated_distances = np.linspace(0, total_length, num_points)

    # Extract x and y coordinates
    x, y = zip(*polygon)

    # Create interpolation functions
    interp_func_x = interp1d(distances, x, kind='linear')
    interp_func_y = interp1d(distances, y, kind='linear')

    # Generate interpolated points
    interpolated_x = interp_func_x(interpolated_distances)
    interpolated_y = interp_func_y(interpolated_distances)

    interpolated_polygon = list(zip(interpolated_x, interpolated_y))
    return interpolated_polygon

def calculate_polygon_centroid(polygon):
    """
    Calculate the centroid (center of mass) of a polygon.
    """
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least three vertices.")

    # Ensure the polygon is closed
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    x_list = [vertex[0] for vertex in polygon]
    y_list = [vertex[1] for vertex in polygon]
    n = len(polygon) - 1  # Number of edges (closed polygon)
    area = 0.0
    x_centroid = 0.0
    y_centroid = 0.0

    for i in range(n):
        xi = x_list[i]
        yi = y_list[i]
        xi1 = x_list[i + 1]
        yi1 = y_list[i + 1]
        cross = xi * yi1 - xi1 * yi
        area += cross
        x_centroid += (xi + xi1) * cross
        y_centroid += (yi + yi1) * cross

    area *= 0.5
    if area == 0:
        raise ValueError("Polygon area is zero; cannot compute centroid.")

    x_centroid /= (6.0 * area)
    y_centroid /= (6.0 * area)

    return x_centroid, y_centroid

def calculate_polygon_area(polygon):
    """
    Calculate the area of a polygon.
    """
    shapely_polygon = Polygon(polygon)
    if not shapely_polygon.is_valid:
        shapely_polygon = shapely_polygon.buffer(0)  # Attempt to fix invalid polygons
    return shapely_polygon.area

def scale_polygon_by_area(global_polygon, local_polygon):
    """
    Scale the global polygon to match the area of the local polygon.
    """
    global_area = calculate_polygon_area(global_polygon)
    local_area = calculate_polygon_area(local_polygon)

    if global_area == 0 or local_area == 0:
        raise ValueError("One of the polygons has zero area; scaling is not possible.")

    # Calculate scale factor (square root of area ratio)
    scale_factor = math.sqrt(local_area / global_area)

    # Calculate centroid of the global polygon
    global_centroid = calculate_polygon_centroid(global_polygon)

    # Scale the global polygon around its centroid
    scaled_polygon = [
        ((x - global_centroid[0]) * scale_factor + global_centroid[0],
         (y - global_centroid[1]) * scale_factor + global_centroid[1])
        for x, y in global_polygon
    ]

    return scaled_polygon

def get_utm_crs(lat, lon):
    """
    Determine the UTM CRS for a given latitude and longitude.
    """
    utm_zone = int((lon + 180) / 6) + 1  # UTM zone calculation
    hemisphere = 'north' if lat >= 0 else 'south'  # Determine hemisphere
    epsg_code = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone  # EPSG code for UTM
    return CRS.from_epsg(epsg_code)

def rotate_polygon(polygon, angle_deg, origin):
    """
    Rotate a polygon around a given origin by a specified angle in degrees.
    """
    angle_rad = math.radians(angle_deg)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    ox, oy = origin

    rotated_polygon = []
    for x, y in polygon:
        qx = ox + cos_theta * (x - ox) - sin_theta * (y - oy)
        qy = oy + sin_theta * (x - ox) + cos_theta * (y - oy)
        rotated_polygon.append((qx, qy))
    return rotated_polygon

def reflect_polygon(polygon, axis='x', origin=(0, 0)):
    """
    Reflect a polygon over the x-axis or y-axis.
    """
    reflected_polygon = []
    ox, oy = origin
    for x, y in polygon:
        if axis == 'x':
            reflected_polygon.append((x, 2 * oy - y))
        elif axis == 'y':
            reflected_polygon.append((2 * ox - x, y))
        else:
            raise ValueError("Axis must be 'x' or 'y'")
    return reflected_polygon

def compute_iou(polygon1, polygon2):
    """
    Compute the Intersection over Union (IoU) between two polygons.
    """
    shapely_poly1 = Polygon(polygon1)
    shapely_poly2 = Polygon(polygon2)

    if not shapely_poly1.is_valid:
        shapely_poly1 = shapely_poly1.buffer(0)
    if not shapely_poly2.is_valid:
        shapely_poly2 = shapely_poly2.buffer(0)

    intersection_area = shapely_poly1.intersection(shapely_poly2).area
    union_area = shapely_poly1.union(shapely_poly2).area

    if union_area == 0:
        return 0
    else:
        iou = intersection_area / union_area
        return iou

def main():
    # === Configuration ===
    building_name = "Walker Hall"
    floor_plan_image_path = "demo/hdFloorPlans/Walker Hall.png"

    # Paths to the OSM files
    buildings_osm_file_path = 'data/buildings.osm'
    nodes_osm_file_path = 'data/nodes.osm'

    # Number of points for interpolation
    interpolation_points = 1000

    # === Parse OSM Data ===
    print("Parsing OSM data...")
    buildings = parse_osm_buildings(buildings_osm_file_path, nodes_osm_file_path)

    # Get the building polygon and entrances for the given building name
    if building_name in buildings:
        building_data = buildings[building_name]
        polygons_global = building_data['polygons']  # List of polygons
        entrances_global = building_data['entrances']  # List of (lon, lat)
    else:
        print(f"Error: Building '{building_name}' not found in the OSM data.")
        sys.exit(1)

    # === Interpolate Polygon ===
    print("Interpolating polygon...")
    interpolated_polygons_global = []
    for polygon in polygons_global:
        interpolated = interpolate_polygon(polygon.copy(), num_points=interpolation_points)
        interpolated_polygons_global.append(interpolated)

    # === Coordinate Transformation ===
    print("Transforming global coordinates to projected coordinates...")
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
    print("Processing floor plan outline...")
    floor_plan_polygon = find_floor_plan_outline(floor_plan_image_path)

    if floor_plan_polygon is None:
        print("Error: Failed to get floor plan outline.")
        sys.exit(1)

    # Interpolate floor plan polygon
    floor_plan_interpolated = interpolate_polygon(floor_plan_polygon.copy(), num_points=interpolation_points)

    # Calculate centroids
    print("Calculating centroids...")
    try:
        # For global coordinates, use projected polygons
        centroid_global = calculate_polygon_centroid(polygons_projected[0])  # Assuming one global polygon

        # For local coordinates, use the interpolated floor plan polygon
        centroid_local = calculate_polygon_centroid(floor_plan_interpolated)
    except ValueError as ve:
        print(f"Error in centroid calculation: {ve}")
        sys.exit(1)

    # === Scale Global Polygon and Entrances to Match Local Polygon's Area ===
    print("Scaling global polygon and entrances to match local polygon's area...")
    scaled_global_polygon = scale_polygon_by_area(polygons_projected[0], floor_plan_interpolated)

    # Scale entrances
    scale_factor = math.sqrt(calculate_polygon_area(floor_plan_interpolated) / calculate_polygon_area(polygons_projected[0]))
    scaled_entrances = []
    for x, y in entrances_projected:
        x_scaled = (x - centroid_global[0]) * scale_factor + centroid_global[0]
        y_scaled = (y - centroid_global[1]) * scale_factor + centroid_global[1]
        scaled_entrances.append((x_scaled, y_scaled))

    # Recalculate centroid of the scaled global polygon
    centroid_scaled_global = calculate_polygon_centroid(scaled_global_polygon)

    # === Translate Scaled Global Polygon and Entrances to Align Centroids ===
    print("Translating scaled global polygon and entrances to align centroids...")
    translation_x = centroid_local[0] - centroid_scaled_global[0]
    translation_y = centroid_local[1] - centroid_scaled_global[1]

    translated_global_polygon = [
        (x + translation_x, y + translation_y) for x, y in scaled_global_polygon
    ]

    translated_entrances = [
        (x + translation_x, y + translation_y) for x, y in scaled_entrances
    ]

    # === Rotate and Reflect Global Polygon and Entrances to Maximize Similarity ===
    print("Rotating and reflecting global polygon and entrances to maximize similarity...")
    angles = np.arange(-180, 180, 1)  # Angles in degrees
    similarities = []
    best_similarity = 0
    best_transformation = None
    best_aligned_polygon = None
    best_aligned_entrances = None

    for angle in angles:
        for reflection in [False, True]:
            # Rotate the polygon
            transformed_polygon = rotate_polygon(translated_global_polygon, angle, centroid_local)
            # Rotate the entrances
            transformed_entrances = rotate_polygon(translated_entrances, angle, centroid_local)
            # Reflect the polygon and entrances if needed
            if reflection:
                transformed_polygon = reflect_polygon(transformed_polygon, axis='x', origin=centroid_local)
                transformed_entrances = reflect_polygon(transformed_entrances, axis='x', origin=centroid_local)
            # Compute similarity
            iou = compute_iou(transformed_polygon, floor_plan_interpolated)
            similarities.append((angle, reflection, iou))
            # Update best similarity
            if iou > best_similarity:
                best_similarity = iou
                best_transformation = (angle, reflection)
                best_aligned_polygon = transformed_polygon
                best_aligned_entrances = transformed_entrances

    best_angle, best_reflection = best_transformation
    print(f"Best rotation angle: {best_angle} degrees")
    print(f"Reflection applied: {'Yes' if best_reflection else 'No'}")
    print(f"Maximum IoU similarity: {best_similarity:.4f}")

    # Extract similarities for plotting
    angles_list = [item[0] for item in similarities if not item[1]]  # Without reflection
    similarities_list = [item[2] for item in similarities if not item[1]]

    angles_reflected = [item[0] for item in similarities if item[1]]  # With reflection
    similarities_reflected = [item[2] for item in similarities if item[1]]

    # === Plotting ===
    print("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Define a color palette
    colors = {
        'building': '#1f77b4',               # Soft Blue
        'entrances': '#17becf',              # Cyan
        'centroid': '#2ca02c',               # Green
        'floor_plan_outline': '#d62728',      # Deep Red
        'aligned_polygon': '#9467bd',         # Purple
        'similarity_no_reflect': '#17becf',   # Cyan
        'similarity_reflect': '#bcbd22'        # Olive
    }
    
    # --- First Subplot: Global Coordinates ---
    ax_global = axes[0, 0]
    ax_global.set_title('Global Coordinates', fontsize=14)

    # Plot the building polygon
    x, y = zip(*polygons_projected[0])
    ax_global.plot(x, y, label='Building Polygon', color=colors['building'], linewidth=1.5)

    # Plot the entrances
    if entrances_projected:
        x_entrances, y_entrances = zip(*entrances_projected)
        ax_global.scatter(x_entrances, y_entrances, color=colors['entrances'], marker='o', edgecolor='black', s=50, label='Entrances')

    # Plot the global centroid
    ax_global.scatter(centroid_global[0], centroid_global[1], color=colors['centroid'], marker='X', s=100, label='Centroid')

    ax_global.set_xlabel('Easting (meters)', fontsize=12)
    ax_global.set_ylabel('Northing (meters)', fontsize=12)
    ax_global.axis('equal')
    ax_global.legend(loc='upper right', fontsize=10)
    ax_global.grid(False)  # Remove grid lines

    # --- Second Subplot: Local Coordinates ---
    ax_local = axes[0, 1]
    ax_local.set_title('Local Coordinates', fontsize=14)

    # Load the floor plan image
    try:
        img = mpimg.imread(floor_plan_image_path)
        height, width = img.shape[:2]
        ax_local.imshow(img, extent=[0, width, height, 0], aspect='auto')  # Flip y-axis to match image coordinates
    except FileNotFoundError:
        print(f"Error: Floor plan image '{floor_plan_image_path}' not found.")
        sys.exit(1)

    # Plot the floor plan outline
    x_fp, y_fp = zip(*floor_plan_interpolated)
    ax_local.plot(x_fp, y_fp, color=colors['floor_plan_outline'], linewidth=1.5, label='Floor Plan Outline')

    # Plot the local centroid
    ax_local.scatter(centroid_local[0], centroid_local[1], color=colors['centroid'], marker='X', s=100, label='Centroid')

    ax_local.set_xlabel('X (pixels)', fontsize=12)
    ax_local.set_ylabel('Y (pixels)', fontsize=12)
    ax_local.axis('equal')
    ax_local.legend(loc='upper right', fontsize=10)
    ax_local.grid(False)  # Remove grid lines

    # --- Third Subplot: Similarity Metric vs Angle ---
    ax_similarity = axes[1, 0]
    ax_similarity.set_title('Similarity Metric vs Rotation Angle', fontsize=14)

    # Plot similarity without reflection
    ax_similarity.plot(angles_list, similarities_list, color=colors['similarity_no_reflect'], label='No Reflection')

    # Plot similarity with reflection
    ax_similarity.plot(angles_reflected, similarities_reflected, color=colors['similarity_reflect'], linestyle='--', label='With Reflection')

    # Highlight the best angle
    ax_similarity.axvline(best_angle, color='black', linestyle=':', linewidth=1, label=f'Best Angle: {best_angle}°')

    ax_similarity.set_xlabel('Rotation Angle (degrees)', fontsize=12)
    ax_similarity.set_ylabel('IoU Similarity', fontsize=12)
    ax_similarity.legend(loc='upper right', fontsize=10)
    ax_similarity.grid(False)  # Remove grid lines

    # --- Fourth Subplot: Best Aligned Polygon Overlaid on Local ---
    ax_overlay = axes[1, 1]
    ax_overlay.set_title('Best Alignment of Global Polygon with Local Polygon', fontsize=14)

    # Display the background image
    ax_overlay.imshow(img, extent=[0, width, height, 0], aspect='auto')

    # Plot the floor plan outline
    ax_overlay.plot(x_fp, y_fp, color=colors['floor_plan_outline'], linewidth=1.5, label='Floor Plan Outline')

    # Plot the best aligned global polygon
    x_aligned, y_aligned = zip(*best_aligned_polygon)
    ax_overlay.plot(x_aligned, y_aligned, color=colors['aligned_polygon'], linestyle='--', linewidth=1.5, label='Aligned Global Polygon')

    # Plot the transformed entrances
    if best_aligned_entrances:
        x_entrances_aligned, y_entrances_aligned = zip(*best_aligned_entrances)
        ax_overlay.scatter(x_entrances_aligned, y_entrances_aligned, color=colors['entrances'], marker='o', edgecolor='black', s=50, label='Entrances')

    # Plot the local centroid
    ax_overlay.scatter(centroid_local[0], centroid_local[1], color=colors['centroid'], marker='X', s=100, label='Centroid')

    ax_overlay.set_xlabel('X (pixels)', fontsize=12)
    ax_overlay.set_ylabel('Y (pixels)', fontsize=12)
    ax_overlay.axis('equal')
    ax_overlay.legend(loc='upper right', fontsize=10)
    ax_overlay.grid(False)  # Remove grid lines

    # Adjust layout and aesthetics
    plt.tight_layout()
    # Save the figure to a local file with a descriptive name and high resolution
    plt.savefig('best_alignment_enhanced.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots generated and saved successfully as 'best_alignment_enhanced.png'.")

if __name__ == "__main__":
    main()
