import cv2
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# Template scales (fixed list)
template_scales = [1.0]

def define_template():
    """
    Defines the 20x20 binary template as provided.
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
    return np.where(template_matrix == '1', 255, 0).astype(np.uint8)

def preprocess_image(image):
    """
    Converts the image to grayscale, applies binary thresholding, and expands pixels.
    Args:
        image (numpy.ndarray): Input BGR image.
    Returns:
        numpy.ndarray: Preprocessed binary image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), dtype=np.uint8)
    expanded = cv2.dilate(binary, kernel, iterations=1)
    return expanded

def detect_template(image, template, tmpl_scale, angles=range(0, 360, 15), threshold=0.7):
    """
    Detects the template in the image without applying Non-Maximum Suppression (NMS).
    Args:
        image (numpy.ndarray): Preprocessed binary image.
        template (numpy.ndarray): Binary template image.
        tmpl_scale (float): Scale factor for the template.
        angles (iterable): Angles in degrees to rotate the template.
        threshold (float): Matching threshold.
    Returns:
        list: List of detections as (pt, angle, scale, score, width, height).
    """
    detected = []

    # Create a mirrored version of the template
    mirrored_template = cv2.flip(template, 1)
    templates = [template, mirrored_template]

    for current_template in templates:
        scaled_template = cv2.resize(current_template, (0, 0), fx=tmpl_scale, fy=tmpl_scale, interpolation=cv2.INTER_NEAREST)
        template_height, template_width = scaled_template.shape

        for angle in angles:
            center = (template_width // 2, template_height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_template = cv2.warpAffine(scaled_template, M, (template_width, template_height), flags=cv2.INTER_NEAREST)

            result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)

            for pt in zip(*loc[::-1]):
                score = result[pt[1], pt[0]]
                detected.append((pt, angle, tmpl_scale, score, template_width, template_height))

    return detected

def remove_straight_line_detections(image, detections, line_threshold=5, angle_variance=10):
    """
    Removes detections that contain only straight lines.
    Args:
        image (numpy.ndarray): Preprocessed binary image.
        detections (list): List of detections.
        line_threshold (int): Minimum number of lines to consider as non-straight.
        angle_variance (int): Maximum deviation from vertical/horizontal to consider as straight.
    Returns:
        list: Filtered detections.
    """
    filtered_detections = []

    for det in detections:
        pt, angle, scale, score, w, h = det
        x, y = pt
        w_scaled, h_scaled = int(w * scale), int(h * scale)

        # Ensure ROI is within image bounds
        if y + h_scaled > image.shape[0] or x + w_scaled > image.shape[1]:
            continue  # Skip detections that go out of bounds

        roi = image[y:y+h_scaled, x:x+w_scaled]

        # Edge detection
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)

        # Hough Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        if lines is not None:
            line_angles = []
            for line in lines:
                rho, theta = line[0]
                degree = theta * 180 / np.pi
                line_angles.append(degree)

            # Check if lines are mostly horizontal or vertical
            horizontal = 0
            vertical = 0
            total_lines = len(line_angles)

            for deg in line_angles:
                if (deg < angle_variance) or (deg > 180 - angle_variance):
                    horizontal += 1
                elif (90 - angle_variance <= deg <= 90 + angle_variance):
                    vertical += 1

            # Define what constitutes "only straight lines"
            if (horizontal / total_lines > 0.8 or vertical / total_lines > 0.8) and total_lines >= line_threshold:
                continue
            else:
                filtered_detections.append(det)
        else:
            filtered_detections.append(det)

    return filtered_detections

def apply_global_nms(detections, distance_threshold=10):
    """
    Applies Non-Maximum Suppression (NMS) globally on all detections.
    Args:
        detections (list): List of detections, each as (pt, angle, scale, score, width, height).
        distance_threshold (int): Maximum distance between detections to be considered close.
    Returns:
        list: Final list of filtered detections after NMS.
    """
    if not detections:
        return []

    # Extract centers and scores
    centers = np.array([(det[0][0] + det[4] // 2, det[0][1] + det[5] // 2) for det in detections])
    scores = np.array([det[3] for det in detections])

    # Sort indices by score descending
    sorted_indices = np.argsort(-scores)

    # Build KD-tree
    tree = cKDTree(centers)

    final_detections = []
    suppressed = np.zeros(len(detections), dtype=bool)

    for idx in sorted_indices:
        if suppressed[idx]:
            continue
        # Add detection to final list
        final_detections.append(detections[idx])

        # Find detections within distance_threshold
        indices = tree.query_ball_point(centers[idx], distance_threshold)

        # Suppress detections
        for i in indices:
            if i != idx:
                suppressed[i] = True

    return final_detections

def analyze_average_confidence(scale_confidence_dict):
    """
    Analyzes detections to find which image scale has the highest average confidence across all template scales.
    Args:
        scale_confidence_dict (dict): Dictionary mapping image scales to their average confidences.
    Returns:
        tuple: (best_image_scale, highest_average_confidence)
    """
    best_average = -1
    best_image_scale = None

    for image_scale, average_confidence in scale_confidence_dict.items():
        if average_confidence > best_average:
            best_average = average_confidence
            best_image_scale = image_scale

    return best_image_scale, best_average

def door_remover(image):
    """
    Removes doors from an image and returns the optimal scale factor and the modified image without doors.
    Args:
        image (numpy.ndarray): Input image as a numpy array.
    Returns:
        tuple: (scale_factor, modified_image_without_doors)
    """
    # Define the template
    template = define_template()

    # Define image scales with 0.1 increments between 0.1 and 1.0
    image_scales = np.arange(0.1, 1.1, 0.1).round(1).tolist()

    # Initialize a dictionary to store average confidences
    scale_confidence_dict = {}

    # Iterate over each image scale
    for img_scale in image_scales:
        # Resize the original image according to the current image scale
        if img_scale != 1.0:
            scaled_image = cv2.resize(image, (0, 0), fx=img_scale, fy=img_scale, interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = image.copy()

        # Preprocess the scaled image
        preprocessed_image = preprocess_image(scaled_image)

        # Initialize a list to store confidences for current image scale
        confidences = []

        # Iterate over each template scale
        for tmpl_scale in template_scales:
            # Detect template without NMS during scale search
            detections = detect_template(preprocessed_image, template, tmpl_scale, angles=range(0, 360, 30), threshold=0.65)

            if detections:
                # Apply straight line removal
                # filtered_detections = remove_straight_line_detections(preprocessed_image, detections, line_threshold=5, angle_variance=10)
                # Since line removal is commented out, use detections directly
                filtered_detections = detections

                if filtered_detections:
                    # Compute average confidence from remaining detections
                    scores = [det[3] for det in filtered_detections]
                    average_confidence = np.mean(scores)
                    confidences.append(average_confidence)

        # Compute average confidence for current image scale
        if len(confidences) > 0:
            overall_average_confidence = sum(confidences) / len(confidences)
        else:
            overall_average_confidence = 0

        # Store the average confidence
        scale_confidence_dict[img_scale] = overall_average_confidence

    # Analyze and find the best image scale
    best_image_scale, best_average_confidence = analyze_average_confidence(scale_confidence_dict)

    if best_image_scale is not None and best_average_confidence > 0:
        # Resize the original image to the best scale
        if best_image_scale != 1.0:
            best_scaled_image = cv2.resize(image, (0, 0), fx=best_image_scale, fy=best_image_scale, interpolation=cv2.INTER_LINEAR)
        else:
            best_scaled_image = image.copy()

        # Preprocess the best scaled image
        best_preprocessed_image = preprocess_image(best_scaled_image)

        # Perform template matching and collect all detections from all template scales
        all_detections = []
        for tmpl_scale in template_scales:
            # Detect template without NMS
            detections = detect_template(best_preprocessed_image, template, tmpl_scale, angles=range(0, 360, 30), threshold=0.69)

            if detections:
                # Apply straight line removal
                # filtered_detections = remove_straight_line_detections(best_preprocessed_image, detections, line_threshold=5, angle_variance=10)
                # Since line removal is commented out, use detections directly
                filtered_detections = detections

                if filtered_detections:
                    all_detections.extend(filtered_detections)

        # Apply Global NMS on all detections
        final_detections = apply_global_nms(all_detections, distance_threshold=10)

        # Modify the preprocessed image by setting the pixels in the detected areas to 0 (black)
        for det in final_detections:
            pt, angle, scale, score, w, h = det
            x, y = pt
            w_scaled, h_scaled = int(w * scale), int(h * scale)  # Adjust size based on scale

            # Ensure coordinates are within image bounds
            x_start = max(x, 0)
            y_start = max(y, 0)
            x_end = min(x + w_scaled, best_preprocessed_image.shape[1])
            y_end = min(y + h_scaled, best_preprocessed_image.shape[0])

            # Set the pixel values within the bounding box to 0 (black) to remove doors
            best_preprocessed_image[y_start:y_end, x_start:x_end] = 0  # For binary image

        # Return the best image scale and the modified preprocessed image
        return best_image_scale, 255 - best_preprocessed_image
    else:
        # No optimal image scale found with positive average confidence.
        return None, None

# # Load the original image
# floorplan_image_path = "Code/CroppedImages/WalkerHallFirstFloor.png"
# image = cv2.imread(floorplan_image_path)

# # Run the door_remover function (assuming it is defined)
# scale_factor, image_without_doors = door_remover(image)

# print("Scale Factor:", scale_factor)

# # Save the resulting image
# output_path = "MUWithoutDoors.png"
# cv2.imwrite(output_path, image_without_doors)
# print(f"Image saved to {output_path}")

# # Plot the original and resulting image
# fig, axes = plt.subplots(1, 2, figsize=(14, 7))
# axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# axes[0].set_title("Original Image")
# axes[0].axis('off')

# axes[1].imshow(image_without_doors, cmap='gray')
# axes[1].set_title("Image Without Doors")
# axes[1].axis('off')

# plt.show()
