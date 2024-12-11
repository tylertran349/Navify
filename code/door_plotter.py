import cv2
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os

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
        # Resize the template
        try:
            scaled_template = cv2.resize(current_template, (0, 0), fx=tmpl_scale, fy=tmpl_scale, interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Error resizing template: {e}")
            continue
        template_height, template_width = scaled_template.shape

        for angle in angles:
            center = (template_width // 2, template_height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_template = cv2.warpAffine(scaled_template, M, (template_width, template_height), flags=cv2.INTER_NEAREST)

            # Perform template matching
            result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)

            for pt in zip(*loc[::-1]):
                score = result[pt[1], pt[0]]
                detected.append((pt, angle, tmpl_scale, score, template_width, template_height))

    return detected

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

def draw_detections_on_image(original_image, detections, scale_factor, color=(255, 0, 0), thickness=2):
    """
    Draws bounding boxes on the original image based on detections.
    Args:
        original_image (numpy.ndarray): The original HD image in RGB format.
        detections (list): List of detections as (pt, angle, scale, score, width, height).
        scale_factor (float): The scale factor used during detection.
        color (tuple): Color of the bounding boxes in BGR format.
        thickness (int): Thickness of the bounding box lines.
    Returns:
        numpy.ndarray: Original image with bounding boxes drawn.
    """
    image_with_boxes = original_image.copy()

    for det in detections:
        pt, angle, scale, score, w, h = det
        x, y = pt
        # Scale detection coordinates back to original image size
        x_original = int(x / scale_factor)
        y_original = int(y / scale_factor)
        w_scaled = int(w * scale / scale_factor)
        h_scaled = int(h * scale / scale_factor)

        # Define top-left and bottom-right points
        top_left = (x_original, y_original)
        bottom_right = (x_original + w_scaled, y_original + h_scaled)

        # Draw rectangle (OpenCV uses BGR)
        cv2.rectangle(image_with_boxes, top_left, bottom_right, color, thickness)

    return image_with_boxes

def door_remover(image):
    """
    Removes doors from an image and returns the optimal scale factor and the modified image without doors.
    Args:
        image (numpy.ndarray): Input image as a numpy array.
    Returns:
        tuple: (scale_factor, modified_image_without_doors, initial_detections, final_detections)
    """
    # Define the template
    template = define_template()

    # Define image scales with 0.1 increments between 0.1 and 1.0
    image_scales = np.arange(0.1, 1.1, 0.1).round(1).tolist()

    # Initialize a dictionary to store average confidences
    scale_confidence_dict = {}

    # Initialize variables to store the best detections
    best_image_scale = None
    best_average_confidence = -1
    all_detections = []
    final_detections = []

    # Initialize variables to capture initial detections for best scale
    initial_detections_best_scale = []

    # Iterate over each image scale to find the best scale based on average confidence
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

        print(f"\nProcessing Image Scale: {img_scale}")

        # Iterate over each template scale
        for tmpl_scale in template_scales:
            # Detect template without NMS during scale search
            detections = detect_template(preprocessed_image, template, tmpl_scale, angles=range(0, 360, 30), threshold=0.65)

            # Print number of detections before any filtering
            num_detections_before = len(detections)
            print(f"  Template Scale: {tmpl_scale} - Detections Before Filtering: {num_detections_before}")

            # Commented out straight line removal
            # Apply straight line removal
            # filtered = remove_straight_line_detections(preprocessed_image, detections, line_threshold=5, angle_variance=10)

            # Print number of detections after filtering (line removal is skipped)
            # num_detections_after = len(filtered)
            # print(f"  Template Scale: {tmpl_scale} - Detections After Line Removal: {num_detections_after}")

            # Since line removal is commented out, consider all detections as filtered
            filtered = detections
            num_detections_after = len(filtered)
            print(f"  Template Scale: {tmpl_scale} - Detections After Filtering: {num_detections_after}")

            if filtered:
                # Compute average confidence from remaining detections
                scores = [det[3] for det in filtered]
                average_confidence = np.mean(scores)
                confidences.append(average_confidence)

        # Compute average confidence for current image scale
        if len(confidences) > 0:
            overall_average_confidence = sum(confidences) / len(confidences)
        else:
            overall_average_confidence = 0

        # Store the average confidence
        scale_confidence_dict[img_scale] = overall_average_confidence

        print(f"  Image Scale: {img_scale} - Overall Average Confidence: {overall_average_confidence:.4f}")

    # Analyze and find the best image scale
    best_image_scale, best_average_confidence = analyze_average_confidence(scale_confidence_dict)

    print(f"\nBest Image Scale: {best_image_scale} with Average Confidence: {best_average_confidence:.4f}")

    if best_image_scale is not None and best_average_confidence > 0:
        # Resize the original image to the best scale
        if best_image_scale != 1.0:
            best_scaled_image = cv2.resize(image, (0, 0), fx=best_image_scale, fy=best_image_scale, interpolation=cv2.INTER_LINEAR)
        else:
            best_scaled_image = image.copy()

        # Preprocess the best scaled image
        best_preprocessed_image = preprocess_image(best_scaled_image)

        # Perform template matching and collect all detections from all template scales
        for tmpl_scale in template_scales:
            # Detect template without NMS
            detections = detect_template(best_preprocessed_image, template, tmpl_scale, angles=range(0, 360, 30), threshold=0.69)

            # Print number of detections before any filtering
            num_detections_before = len(detections)
            print(f"\nBest Scale - Template Scale: {tmpl_scale} - Detections Before Filtering: {num_detections_before}")

            # Commented out straight line removal
            # Apply straight line removal
            # filtered = remove_straight_line_detections(best_preprocessed_image, detections, line_threshold=5, angle_variance=10)

            # Print number of detections after filtering (line removal is skipped)
            # num_detections_after = len(filtered)
            # print(f"Best Scale - Template Scale: {tmpl_scale} - Detections After Line Removal: {num_detections_after}")

            # Since line removal is commented out, consider all detections as filtered
            filtered = detections
            num_detections_after = len(filtered)
            print(f"Best Scale - Template Scale: {tmpl_scale} - Detections After Filtering: {num_detections_after}")

            if filtered:
                # Store initial detections
                initial_detections_best_scale.extend(filtered)

                # Add to global list for NMS
                all_detections.extend(filtered)

        # Apply Global NMS on all detections
        final_detections = apply_global_nms(all_detections, distance_threshold=10)

        print(f"\nFinal Detections After NMS: {len(final_detections)}")

        # Modify the original image by setting the pixels in the detected areas to white (255)
        image_without_doors = image.copy()
        for det in final_detections:
            pt, angle, scale, score, w, h = det
            x, y = pt
            # Scale detection coordinates back to original image size
            x_original = int(x / best_image_scale)
            y_original = int(y / best_image_scale)
            w_scaled = int(w * scale / best_image_scale)
            h_scaled = int(h * scale / best_image_scale)

            # Ensure coordinates are within image bounds
            x_start = max(x_original, 0)
            y_start = max(y_original, 0)
            x_end = min(x_original + w_scaled, image_without_doors.shape[1])
            y_end = min(y_original + h_scaled, image_without_doors.shape[0])

            # Set the pixel values within the bounding box to white (255) to remove doors
            image_without_doors[y_start:y_end, x_start:x_end] = (255, 255, 255)  # White

        # Return the best image scale, the modified image, and all detection stages
        return best_image_scale, image_without_doors, initial_detections_best_scale, final_detections
    else:
        # No optimal image scale found with positive average confidence.
        print("No optimal image scale found with positive average confidence.")
        return None, None, [], []

def main():
    # Path to the floorplan image
    floorplan_image_path = "preprocessing/CroppedFloorPlans/Robbins Hall.png"

    # Check if the image exists
    if not os.path.exists(floorplan_image_path):
        print(f"Image file '{floorplan_image_path}' not found. Please check the path.")
        return

    # Load the original image
    image = cv2.imread(floorplan_image_path)
    if image is None:
        print(f"Failed to load image '{floorplan_image_path}'. Please check the file format and integrity.")
        return

    # Convert original image to RGB for plotting
    original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the door_remover function
    scale_factor, image_without_doors, initial_detections, final_detections = door_remover(image)

    # Check if door removal was successful
    if scale_factor is None:
        print("Door removal was unsuccessful or no doors were detected.")
        return

    print(f"\nScale Factor: {scale_factor}")
    print(f"Initial Detections: {len(initial_detections)}")
    print(f"Final Detections After NMS: {len(final_detections)}")

    # Draw initial detections on the original image
    image_with_initial_detections = draw_detections_on_image(
        original_image_rgb, initial_detections, scale_factor, color=(255, 0, 0), thickness=2
    )  # Red

    # Draw final detections after NMS on the original image
    image_with_final_detections = draw_detections_on_image(
        original_image_rgb, final_detections, scale_factor, color=(255, 0, 0), thickness=2
    )  # Green

    # Final Image Without Doors (already in BGR, convert to RGB for plotting)
    image_without_doors_rgb = cv2.cvtColor(image_without_doors, cv2.COLOR_BGR2RGB)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Top Left: Template
    template = define_template()
    template_inverted = cv2.bitwise_not(template)
    ax1.imshow(template_inverted, cmap='gray')
    ax1.set_title("Door template")
    ax1.axis('off')

    # Top Right: Initial Detections on Original Image
    ax2.imshow(image_with_initial_detections)
    ax2.set_title("Initial Detections")
    ax2.axis('off')

    # Bottom Left: After NMS on Original Image
    ax3.imshow(image_with_final_detections)
    ax3.set_title("After Non-Maximum Suppression (NMS)")
    ax3.axis('off')


    # Bottom Right: Final Image Without Doors
    ax4.imshow(image_without_doors_rgb)
    ax4.set_title("Final Image Without Doors")
    ax4.axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to leave space for titles

    # Display the plot
    plt.show()

    # Save the resulting image without doors
    output_path = "MUWithoutDoors.png"
    cv2.imwrite(output_path, image_without_doors)
    print(f"Image saved to {output_path}")

    # Create a figure
    fig, ax3 = plt.subplots(figsize=(6, 6))  # Adjust size as needed

    # Plot the image in the "Bottom Left" style
    ax3.imshow(image_with_final_detections)
    ax3.set_title("After Non-Maximum Suppression (NMS)", fontsize=12)  # Set title
    ax3.axis('off')  # Turn off axis for a cleaner look

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
