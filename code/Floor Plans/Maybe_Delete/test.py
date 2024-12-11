import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the binary matrix template (23x20)
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
], dtype=np.uint8)

# Convert the '0' and '1' to binary values (0 and 255)
template = np.where(template_matrix == '1', 255, 0).astype(np.uint8)

# Function to perform template matching with rotation handling
def detect_shape(image, template, angles=(0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165)):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_match = None
    best_angle = 0

    for angle in angles:
        # Rotate the template
        center = (template.shape[1] // 2, template.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_template = cv2.warpAffine(template, M, (template.shape[1], template.shape[0]))

        # Perform template matching
        result = cv2.matchTemplate(image_gray, rotated_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Check if we have a new best match
        if max_val > best_score:
            best_score = max_val
            best_match = max_loc
            best_angle = angle

    if best_match is not None:
        top_left = best_match
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        result_image = image.copy()
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        print(f"Best match found at {top_left} with rotation {best_angle} degrees and score {best_score:.2f}")
        return result_image
    else:
        print("No match found.")
        return image

# Example usage
if __name__ == "__main__":
    # Load an input image (replace with your own image path)
    input_image = cv2.imread("code/Floor Plans/thick_death.png")

    # Detect the shape
    result = detect_shape(input_image, template)

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Detected Shape")
    plt.axis('off')
    plt.show()
