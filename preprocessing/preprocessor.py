# main_processor.py
import os
import cv2
import time
import json
from imageCropper import process_image_array
from doorRemover import door_remover
from databaseLoader import DatabaseInitiator
from entranceProjector import project_entrances
import pickle
from database import Database
from entrance_mapper import map_entrances_to_nodes

def cropper_processor(input_folder_path, output_folder_path):
    """
    Processes all PNG images in the specified input folder by cropping them and saving the results
    in the specified output folder, with a progress bar indicating progress and time remaining.

    Args:
        input_folder_path (str): The path to the folder containing PNG images.
        output_folder_path (str): The path to the folder where processed images will be saved.
    """
    print("Cropping started")
    # Verify that the provided input folder path exists
    if not os.path.isdir(input_folder_path):
        raise NotADirectoryError(f"The provided input path '{input_folder_path}' is not a valid directory.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder_path)

    # Filter out only PNG files (case-insensitive)
    png_files = [file for file in files if file.lower().endswith('.png')]

    if not png_files:
        print(f"No PNG files found in the directory: {input_folder_path}")
        return

    total_files = len(png_files)
    processed_count = 0
    start_time = time.time()

    # Process each PNG file
    for filename in png_files:
        input_file_path = os.path.join(input_folder_path, filename)

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
        output_file_path = os.path.join(output_folder_path, filename)

        # Save the cropped image
        success = cv2.imwrite(output_file_path, output_image)

        if not success:
            print(f"Failed to save cropped image to: {output_file_path}")

        # Update the progress
        processed_count += 1
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / processed_count
        remaining_time = avg_time_per_image * (total_files - processed_count)

        # Display progress bar
        progress = int((processed_count / total_files) * 50)  # Progress bar length of 50
        bar = f"[{'#' * progress}{'.' * (50 - progress)}]"
        print(
            f"\r{bar} {processed_count}/{total_files} | "
            f"Elapsed: {elapsed_time:.2f}s | "
            f"Remaining: {remaining_time:.2f}s", end=""
        )

    print("\nCropping complete.")

def door_processor(input_folder_path, output_folder_path):
    """
    Processes all PNG images in the specified input folder by removing doors and saving the results
    in the specified output folder. It also returns a dictionary mapping each building's name to its scale factor.

    Args:
        input_folder_path (str): The path to the folder containing cropped PNG images.
        output_folder_path (str): The path to the folder where processed images will be saved.

    Returns:
        dict: A dictionary with keys as building names and values as scale factors.
    """
    print("Door removal started")

    # Verify that the provided input folder path exists
    if not os.path.isdir(input_folder_path):
        raise NotADirectoryError(f"The provided input path '{input_folder_path}' is not a valid directory.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder_path)

    # Filter out only PNG files (case-insensitive)
    png_files = [file for file in files if file.lower().endswith('.png')]

    if not png_files:
        print(f"No PNG files found in the directory: {input_folder_path}")
        return {}

    total_files = len(png_files)
    processed_count = 0
    start_time = time.time()
    scales = {}

    # Process each PNG file
    for filename in png_files:
        input_file_path = os.path.join(input_folder_path, filename)

        # Read the image
        image = cv2.imread(input_file_path)

        # Check if the image was successfully loaded
        if image is None:
            print(f"Warning: Unable to read image '{input_file_path}'. Skipping this file.")
            continue

        
        # Remove doors from the image
        scale_factor, image_without_doors = door_remover(image)

        if scale_factor is not None and image_without_doors is not None:
            # Define the output file path
            output_file_path = os.path.join(output_folder_path, filename)

            # Save the modified image
            success = cv2.imwrite(output_file_path, image_without_doors)

            if success:
                # Extract building name from filename (assuming filename without extension)
                building_name = os.path.splitext(filename)[0]
                scales[building_name] = scale_factor
            else:
                print(f"Failed to save processed image to: {output_file_path}")
        else:
            print(f"Door removal failed for image '{filename}'. Skipping saving.")

        # Update the progress
        processed_count += 1
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / processed_count
        remaining_time = avg_time_per_image * (total_files - processed_count)

        # Display progress bar
        progress = int((processed_count / total_files) * 50)  # Progress bar length of 50
        bar = f"[{'#' * progress}{'.' * (50 - progress)}]"
        print(
            f"\r{bar} {processed_count}/{total_files} | "
            f"Elapsed: {elapsed_time:.2f}s | "
            f"Remaining: {remaining_time:.2f}s", end=""
        )

    print("\nDoor removal complete.")
    return scales

def entrance_processor(scales, output_folder_path, database):
    """
    Processes entrances for each building based on the scales dictionary, updates the database,
    and saves the database object to a file.

    Args:
        scales (dict): Dictionary mapping building names to scale factors.
        output_folder_path (str): Path to the folder containing processed floor plan images.
        database (Database): Database object to update with entrance data.

    Returns:
        None
    """
    print("Entrance processing started")
    if not scales:
        print("No scales available. Exiting the processing pipeline.")
        return

    total_buildings = len(scales)
    processed_count = 0
    start_time = time.time()

    # Process each building
    for building_name, scale_factor in scales.items():
        # Construct the floor plan image path
        floor_plan_image_path = os.path.join(output_folder_path, f"{building_name}.png")

        # Check if the processed image exists
        if not os.path.exists(floor_plan_image_path):
            print(f"Processed image for '{building_name}' not found. Skipping.")
            continue

        try:
            # Process entrances for the building    
            project_entrances(database, building_name, floor_plan_image_path)
            # Scale indoor entrances based on the scale factor
            database.scale_indoor_entrances_for_building(building_name, scale_factor)
            
        except Exception as e:
            print(f"Error processing building '{building_name}': {e}")
            continue

        # Update the progress
        processed_count += 1
        elapsed_time = time.time() - start_time
        avg_time_per_building = elapsed_time / processed_count
        remaining_time = avg_time_per_building * (total_buildings - processed_count)

        # Display progress bar
        progress = int((processed_count / total_buildings) * 50)  # Progress bar length of 50
        bar = f"[{'#' * progress}{'.' * (50 - progress)}]"
        print(
            f"\r{bar} {processed_count}/{total_buildings} | "
            f"Elapsed: {elapsed_time:.2f}s | "
            f"Remaining: {remaining_time:.2f}s", end=""
        )

    print("\nProcessing entrances complete.")

    # Save the database object using pickle
    output_file = "database.pkl"
    try:
        with open(output_file, "wb") as f:
            pickle.dump(database, f)
        print(f"Database saved successfully to '{output_file}'.")
    except Exception as e:
        print(f"Error saving database: {e}")

def main():
    """
    Main function to execute the cropping, door removal, and entrance processing pipeline.
    """
    # Initialize the database
    data = DatabaseInitiator()
    database = data.get_database()

    # Define your folder paths
    input_folder_path = "preprocessing/InputFloorPlans"
    cropped_folder_path = "preprocessing/CroppedFloorPlans"
    output_folder_path = "preprocessing/OutputFloorPlans"

    # Step 1: Crop the input images
    cropper_processor(input_folder_path, cropped_folder_path)

    # Step 2: Remove the doors from the cropped images and store the scales in a dictionary
    scales = door_processor(cropped_folder_path, output_folder_path)
    print(scales)

    # Step 3: Process entrances based on the scales dictionary
    entrance_processor(scales, cropped_folder_path, database)

if __name__ == "__main__":
    main()
