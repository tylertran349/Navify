import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle

# File paths for the uploaded images and detections
floorplan_image_path = "Code/data/floor_plans/DeathStarFirstFloor.png"
detections_file_path = "detections.pkl"  # File to save/load detections

def load_image(file_path):
    """
    Loads an image from the specified file path.
    
    Args:
        file_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Loaded image.
        
    Raises:
        ValueError: If the image fails to load.
    """
    print(f"Loading image from: {file_path}")
    image = cv2.imread(file_path)
    
    # Check if the image was loaded correctly
    if image is None:
        raise ValueError(f"Failed to load the image. Check the file path: {file_path}")
    
    print("Image loaded successfully!")
    return image

def convert_to_binary_matrix(image):
    """
    Converts a BGR image to a binary image using grayscale conversion and thresholding.
    
    Args:
        image (numpy.ndarray): Input BGR image.
        
    Returns:
        numpy.ndarray: Binary image with pixels as 0 or 255.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding (invert the binary image)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary

def expand_pixels_with_3x3_neighborhood(binary_image):
    """
    Expands all pixels with value 1 by filling a 3x3 neighborhood around each pixel with 1s.
    
    Args:
        binary_image (numpy.ndarray): Input binary image (0 or 255).
        
    Returns:
        numpy.ndarray: Expanded binary image with a 3x3 neighborhood filled around each 1 pixel.
    """
    # Create a 3x3 structuring element of 1s
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    # Apply morphological dilation using the 3x3 kernel
    expanded_image = cv2.dilate(binary_image, kernel, iterations=1)
    
    return expanded_image

def remove_thin_walls(binary_image, thickness_threshold=5):
    """
    Removes thin wall regions thinner than the specified thickness threshold while keeping thick walls.
    
    Args:
        binary_image (numpy.ndarray): Input binary image (0 or 255). Walls are 255, background is 0.
        thickness_threshold (int): Thickness threshold.
        
    Returns:
        numpy.ndarray: Binary image with thin walls removed.
    """
    # Compute the distance transform
    distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    
    # Calculate the maximum distance corresponding to thin walls
    max_distance = (thickness_threshold - 1) / 2.0
    
    # Threshold the distance transform to find thin regions
    _, thin_regions = cv2.threshold(distance_transform, max_distance, 255, cv2.THRESH_BINARY_INV)
    thin_regions = thin_regions.astype(np.uint8)
    
    # Invert thin_regions to get a mask of thick regions
    thick_regions = cv2.bitwise_not(thin_regions)
    
    # Mask the original binary image to keep only thick walls
    thick_walls = cv2.bitwise_and(binary_image, binary_image, mask=thick_regions)
    
    return thick_walls

def define_template():
    """
    Defines the 23x20 binary template as provided.
    
    Returns:
        numpy.ndarray: Binary template image with pixels as 0 or 255.
    """
    template_matrix = np.array([
        list("00000000000000000000"),
        list("00000000000000111111"),
        list("00000000000001111111"),
        list("00000000000111111111"),
        list("00000000001111111111"),
        list("00000001111111101111"),
        list("00000001111111101111"),
        list("00000111111100001111"),
        list("00001111111000001111"),
        list("00011111110000001111"),
        list("00011111000000001111"),
        list("00011110000000001111"),
        list("00111110000000001111"),
        list("00111100000000001111"),
        list("01111100000000001111"),
        list("01111000000000001111"),
        list("01110000000000001111"),
        list("01110000000000001111"),
        list("11110000000000001111"),
        list("11110000000000001111"),
    ], dtype=str)
    
    # Convert '0' and '1' to binary values (0 and 255)
    template_binary = np.where(template_matrix == '1', 255, 0).astype(np.uint8)
    
    # Check the shape
    print(f"Template shape: {template_binary.shape}")
    
    return template_binary

def detect_template(image, template, scales=[1.2, 1.1, 1.0, 0.9, 0.8], angles=range(0, 360, 15), threshold=0.7):
    """
    Detects the given template in the image, handling rotations, mirrored versions, and multiple scales.
    Applies Non-Maximum Suppression (NMS) to remove closely overlapping detections.

    Args:
        image (numpy.ndarray): Input binary image (0 or 255).
        template (numpy.ndarray): Binary template image (0 or 255).
        scales (list): List of scales to resize the template.
        angles (iterable): Angles in degrees to rotate the template for matching.
        threshold (float): Matching threshold between 0 and 1.

    Returns:
        list of tuples: Final filtered detections with their top-left coordinates, rotation angles, and scale factors.
        numpy.ndarray: Image with filtered detected templates outlined.
    """
    detected = []
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Create a mirrored (flipped) version of the template
    mirrored_template = cv2.flip(template, 1)

    # Define a list of templates to check (original and mirrored)
    templates = [template, mirrored_template]

    # Iterate over each template (original and mirrored)
    for current_template in templates:
        for scale in scales:
            # Resize the template based on the current scale
            scaled_template = cv2.resize(current_template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            template_height, template_width = scaled_template.shape

            for angle in angles:
                # Rotate the scaled template
                center = (template_width // 2, template_height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_template = cv2.warpAffine(scaled_template, M, (template_width, template_height), flags=cv2.INTER_NEAREST)

                # Perform template matching
                result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)

                # Find locations where the match exceeds the threshold
                loc = np.where(result >= threshold)

                for pt in zip(*loc[::-1]):
                    score = result[pt[1], pt[0]]
                    detected.append((pt, angle, scale, score, template_width, template_height))

    # Apply Non-Maximum Suppression (NMS)
    final_detections = apply_local_nms(detected, distance_threshold=10)

    # Draw final detections
    for det in final_detections:
        pt, angle, scale, score, template_width, template_height = det
        bottom_right = (pt[0] + template_width, pt[1] + template_height)
        cv2.rectangle(result_image, pt, bottom_right, (0, 255, 0), 2)  # Green rectangles

    print(f"Total final detections after NMS: {len(final_detections)}")
    return final_detections, result_image

def apply_local_nms(detections, distance_threshold=10):
    """
    Applies Non-Maximum Suppression (NMS) locally to groups of close detections.

    Args:
        detections (list): List of detections, each as (pt, angle, scale, score, width, height).
        distance_threshold (int): Maximum distance between detections to be considered close.

    Returns:
        list: Final list of filtered detections after local NMS.
    """
    if not detections:
        return []

    # Helper function to compute the Euclidean distance
    def euclidean_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    # Convert detections to a list of centers and scores
    centers = [(det[0][0] + det[4] // 2, det[0][1] + det[5] // 2) for det in detections]
    scores = [det[3] for det in detections]

    # Group detections using a list of visited flags
    visited = [False] * len(detections)
    final_detections = []

    # Iterate through detections to find groups
    for i, (center1, score1) in enumerate(zip(centers, scores)):
        if visited[i]:
            continue

        group = [i]
        visited[i] = True

        # Find all detections close to the current one
        for j, (center2, score2) in enumerate(zip(centers, scores)):
            if i != j and not visited[j] and euclidean_distance(center1, center2) < distance_threshold:
                group.append(j)
                visited[j] = True

        # Apply NMS to the group by selecting the detection with the highest score
        best_index = max(group, key=lambda idx: scores[idx])
        final_detections.append(detections[best_index])

    return final_detections

def has_axis_aligned_corner(window):
    """
    Checks if the given window contains an axis-aligned 90-degree corner.

    Args:
        window (numpy.ndarray): Binary image window (0 or 255) of size 10x10.

    Returns:
        bool: True if the window contains an axis-aligned right-angle corner, False otherwise.
    """
    # Ensure the window is 10x10
    if window.shape != (10, 10):
        return False

    # Define corner patterns
    patterns = [
        ('top-left', window[0, :] == 255, window[:, 0] == 255),
        ('top-right', window[0, :] == 255, window[:, -1] == 255),
        ('bottom-left', window[-1, :] == 255, window[:, 0] == 255),
        ('bottom-right', window[-1, :] == 255, window[:, -1] == 255)
    ]

    for corner_type, row_condition, col_condition in patterns:
        if np.all(row_condition) and np.all(col_condition):
            # Check that the rest of the window is 0
            mask = np.zeros((10, 10), dtype=bool)
            if corner_type in ['top-left', 'top-right']:
                mask[0, :] = True
            if corner_type in ['top-left', 'bottom-left']:
                mask[:, 0] = True
            if corner_type in ['top-right', 'bottom-right']:
                mask[:, -1] = True
            if corner_type in ['bottom-left', 'bottom-right']:
                mask[-1, :] = True

            # Invert mask to check the rest
            rest = window.copy()
            rest[mask] = 0
            if np.all(rest == 0):
                return True

    return False

def filter_detections_with_axis_aligned_corners(detections, image, window_size=10):
    """
    Filters template detections that include an axis-aligned 90-degree corner within a 10x10 window.

    Args:
        detections (list): List of detections (pt, angle, scale, score, width, height).
        image (numpy.ndarray): Input binary image (0 or 255).
        window_size (int): Size of the window to check for corners around each detection.

    Returns:
        list: Filtered list of detections.
    """
    filtered_detections = []
    height, width = image.shape

    for det in detections:
        pt, angle, scale, score, w, h = det
        x1, y1 = pt

        # Define the window boundaries
        # We'll slide the window within the detection's bounding box
        detection_x_start = max(x1, 0)
        detection_y_start = max(y1, 0)
        detection_x_end = min(x1 + w, width)
        detection_y_end = min(y1 + h, height)

        contains_corner = False

        # Slide a 10x10 window across the detection's region
        for y in range(detection_y_start, detection_y_end - window_size + 1):
            for x in range(detection_x_start, detection_x_end - window_size + 1):
                window = image[y:y + window_size, x:x + window_size]
                if has_axis_aligned_corner(window):
                    contains_corner = True
                    break  # No need to check further windows in this detection
            if contains_corner:
                break

        if not contains_corner:
            filtered_detections.append(det)

    return filtered_detections

def save_detections(detections, filename):
    """
    Saves detections to a file using pickle.

    Args:
        detections (list): List of detections to save.
        filename (str): Path to the file where detections will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(detections, f)
    print(f"Detections saved to {filename}.")

def load_detections(filename):
    """
    Loads detections from a pickle file.

    Args:
        filename (str): Path to the file from which detections will be loaded.

    Returns:
        list: Loaded detections.
    """
    with open(filename, 'rb') as f:
        detections = pickle.load(f)
    print(f"Detections loaded from {filename}.")
    return detections

def plot_color_image(image, title):
    """
    Plots a color image using matplotlib.
    
    Args:
        image (numpy.ndarray): Image to plot.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 10))
    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(255-image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    start_time = time.time()

    if not os.path.exists(floorplan_image_path):
        raise ValueError(f"File not found: {floorplan_image_path}")

    # Load and preprocess the image
    image = load_image(floorplan_image_path)
    binary_image = convert_to_binary_matrix(image)
    expanded_image = expand_pixels_with_3x3_neighborhood(binary_image)
    template = define_template()

    # Check if detections file exists
    if os.path.exists(detections_file_path):
        # Load detections from the file
        detections = load_detections(detections_file_path)
    else:
        # Detect the template in the expanded image
        detections, result_image = detect_template(
            expanded_image,
            template,
            scales=[1.1, 1.0, 0.9, 0.8],
            angles=range(0, 360, 30),
            threshold=0.69
        )

        # Save the detections for future use
        save_detections(detections, detections_file_path)

    print(f"Total detections before filtering: {len(detections)}")

    # Filter detections that include axis-aligned 90-degree corners
    final_detections = filter_detections_with_axis_aligned_corners(
        detections,
        expanded_image,
        window_size=20
    )

    print(f"Total detections after filtering: {len(final_detections)}")

    # Draw final detections after corner filtering
    filtered_detections_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    for det in final_detections:
        pt, angle, scale, score, template_width, template_height = det
        bottom_right = (pt[0] + template_width, pt[1] + template_height)
        cv2.rectangle(filtered_detections_image, pt, bottom_right, (0, 255, 0), 2)  # Green rectangles

    # Plot the filtered detections
    plot_color_image(filtered_detections_image, "Door detection")

    end_time = time.time()
    print("Processing time:", end_time - start_time, "seconds")

if __name__ == "__main__":
    main()
