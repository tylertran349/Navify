import os
import cv2
import numpy as np
from imageCropper import process_image_array

def cropper_processor(folder_path):
    """
    Processes all PNG images in the specified folder by cropping them and saving the results
    in a new folder called 'CroppedImages' in the current directory.

    Args:
        folder_path (str): The path to the folder containing PNG images.
    """
    # Verify that the provided folder path exists
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"The provided path '{folder_path}' is not a valid directory.")

    # Define the output directory path
    output_dir = os.path.join(os.getcwd(), 'CroppedImages')
    
    # Create the 'CroppedImages' directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory is set to: {output_dir}")

    # List all files in the input folder
    files = os.listdir(folder_path)
    
    # Filter out only PNG files (case-insensitive)
    png_files = [file for file in files if file.lower().endswith('.png')]

    if not png_files:
        print(f"No PNG files found in the directory: {folder_path}")
        return

    # Process each PNG file
    for filename in png_files:
        input_file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {input_file_path}")

        # Read the image
        input_image = cv2.imread(input_file_path)
        
        # Check if the image was successfully loaded
        if input_image is None:
            print(f"Warning: Unable to read image '{input_file_path}'. Skipping this file.")
            continue

        try:
            # Process the image using the provided function
            output_image = process_image_array(input_image)
        except Exception as e:
            print(f"Error processing image '{filename}': {e}")
            continue

        # Define the output file path
        output_file_path = os.path.join(output_dir, filename)

        # Save the cropped image
        success = cv2.imwrite(output_file_path, output_image)
        
        if success:
            print(f"Successfully saved cropped image to: {output_file_path}")
        else:
            print(f"Failed to save cropped image to: {output_file_path}")

    print("Processing complete.")

# Example usage:
if __name__ == "__main__":
    # Replace this path with the path to your folder containing PNG images
    input_folder = "data/floor_plans"
    
    # Call the cropper_processor function
    cropper_processor(input_folder)
