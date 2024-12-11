# filename: create_subplots.py

import os
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    """
    Loads an image from the specified path.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        Image.Image: The loaded PIL Image object.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    
    image = Image.open(image_path)
    return image

def create_subplots(image_paths, titles=None, figsize=(18, 6), layout=(1, 3)):
    """
    Creates a subplot grid from the specified images.

    Parameters:
        image_paths (list of str): List of image file paths.
        titles (list of str, optional): List of titles for each subplot.
        figsize (tuple, optional): Figure size in inches (width, height).
        layout (tuple, optional): Layout of subplots as (rows, cols).

    Returns:
        matplotlib.figure.Figure: The created matplotlib figure.
    """
    num_images = len(image_paths)
    rows, cols = layout
    
    if num_images > rows * cols:
        raise ValueError("Number of images exceeds the subplot grid size.")
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten in case of multiple rows and columns

    for idx, image_path in enumerate(image_paths):
        ax = axes[idx]
        try:
            img = load_image(image_path)
        except FileNotFoundError as e:
            print(e)
            ax.axis('off')
            continue

        ax.imshow(img)
        # if titles and idx < len(titles):
        #     ax.set_title(titles[idx], fontsize=14)
        ax.axis('off')  # Hide axis ticks

    # Hide any unused subplots
    for idx in range(num_images, rows * cols):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Define image filenames
    image_filenames = ['figures/rain.png', 'figures/traffic.png', 'figures/crime.png']
    
    # Optionally, define titles for each subplot
    subplot_titles = ['Rain Data', 'Traffic Data', 'Crime Data']
    
    # Create subplots
    try:
        fig = create_subplots(
            image_paths=image_filenames,
            titles=subplot_titles,
            figsize=(18, 6),       # Adjust figure size as needed
            layout=(1, 3)           # 1 row x 3 columns
        )
        print("Subplots created successfully.")
    except ValueError as ve:
        print(f"Error creating subplots: {ve}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return
    
    # Save the combined figure as an image
    output_path = 'combined_subplots.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined subplot image saved as '{output_path}'.")
    
    # Optionally, display the plot
    plt.show()

if __name__ == "__main__":
    main()
