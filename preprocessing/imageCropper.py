import cv2
import numpy as np

def crop_first_column_trim_sides_image(image, trim_percentage, trim_pixels):
    """
    Crop the input image to exclude the box outlines, keep only the content inside
    the first column, trim a specific percentage off the right side, and remove a
    fixed number of pixels from the top, left, and bottom.

    Args:
        image (numpy.ndarray): Input image array.
        trim_percentage (float): Percentage of the width to trim from the right side.
        trim_pixels (int): Number of pixels to trim from the top, left, and bottom.

    Returns:
        numpy.ndarray: The cropped image array.
    """
    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Threshold the image to binary (black and white)
    _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Find contours to detect the main box outline
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Identify the main contour (largest box)
    height, width = binary_image.shape
    main_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > width * 0.8 and h > height * 0.8:  # Main box criteria
            main_contour = contour
            break

    if main_contour is None:
        raise ValueError("Could not detect the main box outline.")

    # Step 5: Get bounding box of the main contour and crop to it
    x, y, w, h = cv2.boundingRect(main_contour)
    cropped_main_box = image[y:y+h, x:x+w]

    # Step 6: Detect columns in the main box
    gray_cropped_main_box = cv2.cvtColor(cropped_main_box, cv2.COLOR_BGR2GRAY)
    _, binary_cropped_main_box = cv2.threshold(gray_cropped_main_box, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_cropped_main_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Sort contours from left to right and find the first column
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    first_column_contour = contours[0]
    x_col, y_col, w_col, h_col = cv2.boundingRect(first_column_contour)
    first_column_image = cropped_main_box[y_col:y_col+h_col, x_col:x_col+w_col]

    # Step 8: Trim the right side of the first column
    col_height, col_width = first_column_image.shape[:2]
    trim_width = int(col_width * (1 - trim_percentage / 100))
    first_column_trimmed = first_column_image[:, :trim_width]

    # Step 9: Trim fixed pixels from the top, left, and bottom
    final_cropped_image = first_column_trimmed[trim_pixels:-trim_pixels, trim_pixels:]

    return final_cropped_image

def crop_to_content_image(image):
    """
    Further crop the image to remove any whitespace margins around the central
    black and white figure without cutting off any dark-colored pixels.
    Leaves a margin of 10 pixels around the figure.

    Args:
        image (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: The final cropped image array.
    """
    # Step 1: Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Threshold the image to binary (black and white)
    _, binary_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Find all non-zero (dark-colored) pixels
    coords = cv2.findNonZero(binary_image)

    # Step 4: Get bounding rectangle of non-zero pixels
    x, y, w, h = cv2.boundingRect(coords)

    # Step 4b: Adjust x, y, w, h to include a margin of 10 pixels
    margin = 10
    x_new = max(x - margin, 0)
    y_new = max(y - margin, 0)
    x_max = min(x + w + margin, image.shape[1])
    y_max = min(y + h + margin, image.shape[0])
    w_new = x_max - x_new
    h_new = y_max - y_new

    # Step 5: Crop the image to the bounding rectangle with margin
    cropped_image = image[y_new:y_new+h_new, x_new:x_new+w_new]

    return cropped_image

def process_image_array(input_image):
    """
    Process the image array by first applying the initial crop and then removing
    whitespace margins around the central figure, leaving a 10-pixel margin.

    Args:
        input_image (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: The final cropped image array.
    """
    # Hard-coded parameters
    trim_percentage = 7.14  # Percentage to trim from the right side in the first step
    trim_pixels = 10        # Number of pixels to trim from the top, left, and bottom in the first step

    # Apply the first cropping function
    intermediate_image = crop_first_column_trim_sides_image(input_image, trim_percentage, trim_pixels)
    
    # Apply the second cropping function on the result of the first
    final_cropped_image = crop_to_content_image(intermediate_image)

    return final_cropped_image

# Example usage
# Read an image from a file (for demonstration purposes)
input_image = cv2.imread("code/ARCFirstFloor.png")

# Check if the image was successfully loaded
if input_image is None:
    raise FileNotFoundError("Input image not found.")

# Process the image array
output_image = process_image_array(input_image)

# Display the output image (optional)
# cv2.imshow("Cropped Image", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# If needed, save the output image to a file
# cv2.imwrite("code/Floor Plans/cropped.png", output_image)
