# filename: frontend.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from tkintermapview import TkinterMapView  # Import TkinterMapView

# Import the backend's find_shortest_path function
from backend import find_shortest_path

# Define constants for colors
LIGHT_GREY = "#ECECEC"  # Light grey color

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pathfinding Application")
        self.geometry("800x600")  # Initial window size
        self.resizable(True, True)  # Allow window resizing

        # Base path for hdFloorPlans and processedFloorPlans folders
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.hd_base_path = os.path.join(script_dir, "hdFloorPlans")
        self.processed_base_path = os.path.join(script_dir, "processedFloorPlans")

        # Fetch building names from hdFloorPlans folder
        self.building_names = self.get_building_names()

        # Variables to store user selections
        self.start_position = tk.StringVar()
        self.end_position = tk.StringVar()
        self.transportation_mode = tk.StringVar(value="Walking")
        self.safe_mode = tk.StringVar(value="On")

        # Initialize frames
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)
        self.selection_frame = None  # Will be created when needed
        self.path_frame = None       # Will be created when needed

        # Initialize selection variables
        self.click_positions = []
        self.current_step = 1  # Steps: 1-Select Start, 2-Select End, 3-Display Paths
        self.path_result = None  # To store the result from backend
        self.current_substep = None  # For multi-step paths

        # Store the selected buildings
        self.start_building = None
        self.end_building = None

        # Keep references to Image objects to prevent them from being garbage collected
        self.loaded_images = []      # To keep Image objects open
        self.photo_images = []       # To keep PhotoImage objects referenced

        # Setup the main_frame UI
        self.init_main_ui()

    def init_main_ui(self):
        # Use grid layout
        self.main_frame.columnconfigure(0, weight=1)

        # Start position dropdown
        ttk.Label(self.main_frame, text="Start Position:").grid(row=0, column=0, pady=5)
        self.start_combo = ttk.Combobox(self.main_frame, textvariable=self.start_position, values=self.building_names, state='readonly')
        self.start_combo.grid(row=1, column=0, pady=5)

        # End position dropdown
        ttk.Label(self.main_frame, text="End Position:").grid(row=2, column=0, pady=5)
        self.end_combo = ttk.Combobox(self.main_frame, textvariable=self.end_position, values=self.building_names, state='readonly')
        self.end_combo.grid(row=3, column=0, pady=5)

        # Transportation Mode selector
        modes = ["Walking", "Biking"]
        ttk.Label(self.main_frame, text="Transportation Mode:").grid(row=4, column=0, pady=5)
        frame_mode = ttk.Frame(self.main_frame)
        frame_mode.grid(row=5, column=0)
        for mode in modes:
            ttk.Radiobutton(frame_mode, text=mode, variable=self.transportation_mode, value=mode).pack(side=tk.LEFT)

        # Safe Mode selector
        safe_options = ["On", "Off"]
        ttk.Label(self.main_frame, text="Safe Mode:").grid(row=6, column=0, pady=5)
        frame_safe = ttk.Frame(self.main_frame)
        frame_safe.grid(row=7, column=0)
        for option in safe_options:
            ttk.Radiobutton(frame_safe, text=option, variable=self.safe_mode, value=option).pack(side=tk.LEFT)

        # Next Button
        ttk.Button(self.main_frame, text="Next", command=self.on_next).grid(row=8, column=0, pady=15)

        # Status label for displaying messages
        self.status_label = ttk.Label(self.main_frame, text="", foreground="red")
        self.status_label.grid(row=9, column=0, pady=5)

    def get_building_names(self):
        """
        Scans the hdFloorPlans folder for .png files and extracts building names.
        Ensures that 'Outdoor' is always included.
        """
        if not os.path.exists(self.hd_base_path):
            self.show_error(f"Folder 'hdFloorPlans' not found in {self.hd_base_path}")
            self.destroy()
            return []

        files = os.listdir(self.hd_base_path)
        building_names = []
        for file in files:
            if file.lower().endswith('.png'):
                name = os.path.splitext(file)[0]
                building_names.append(name)

        # Ensure 'Outdoor' exists
        if "Outdoor" not in building_names:
            self.show_error("'Outdoor.png' is missing in 'hdFloorPlans' folder.")
            self.destroy()
            return []

        return building_names

    def show_error(self, message):
        """
        Displays an error message in the status label.
        """
        self.status_label.config(text=message, foreground="red")

    def show_info(self, message):
        """
        Displays an informational message in the status label.
        """
        self.status_label.config(text=message, foreground="blue")

    def on_next(self):
        # Get the selected options
        start = self.start_position.get()
        end = self.end_position.get()

        # Validate selections
        if not start or not end:
            self.show_error("Please select both start and end positions.")
            return

        # Proceed to display images
        print(f"Selected start: {start}, end: {end}")
        # Hide main_frame
        self.main_frame.pack_forget()

        # Initialize selection variables
        self.click_positions = []
        self.current_step = 1  # Steps: 1-Select Start, 2-Select End, 3-Display Paths
        self.path_result = None  # To store the result from backend
        self.current_substep = None  # For multi-step paths

        # Store the selected buildings
        self.start_building = start
        self.end_building = end

        # Initialize selection UI
        self.init_selection_ui()

    def init_selection_ui(self):
        # Create selection frame
        self.selection_frame = ttk.Frame(self)
        self.selection_frame.pack(fill="both", expand=True)

        # Create top frame for label and confirm button
        top_frame = ttk.Frame(self.selection_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Use grid layout
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        # Text Label
        self.text_label = ttk.Label(top_frame, text="Select start position", font=("Helvetica", 14))
        self.text_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)

        # Confirm Button
        self.confirm_button = ttk.Button(top_frame, text="Confirm", command=self.on_confirm, state=tk.DISABLED)
        self.confirm_button.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

        # Status label for selection messages
        self.selection_status_label = ttk.Label(self.selection_frame, text="", foreground="red")
        self.selection_status_label.pack(pady=5)

        # Initialize building display
        self.init_building_display()

    def init_building_display(self):
        self.building_name = self.current_building()

        if self.building_name == "Outdoor":
            self.init_map()
        else:
            self.init_canvas()
            self.load_images()
            self.display_image()

    def current_building(self):
        return self.start_building if self.current_step == 1 else self.end_building

    def init_map(self):
        # Create a map widget using TkinterMapView
        self.map_widget = TkinterMapView(self.selection_frame, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)

        # Set initial map location
        self.map_widget.set_position(38.5382, -121.7617)
        self.map_widget.set_zoom(15)

        # Bind click event
        self.map_widget.add_left_click_map_command(self.on_map_click)

    def init_canvas(self):
        # Create a frame to hold the canvas
        self.canvas_frame = ttk.Frame(self.selection_frame)
        self.canvas_frame.pack(fill="both", expand=True)

        # Create a canvas with light grey background
        self.canvas = tk.Canvas(self.canvas_frame, bg=LIGHT_GREY)
        self.canvas.pack(fill="both", expand=True)

        # Bind the canvas to configure events
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Initialize marker
        self.marker = None
        self.selected_position = None

        # Bind click event
        self.canvas.bind("<Button-1>", self.on_click)

    def on_canvas_configure(self, event):
        # Update the canvas size
        canvas_width = event.width
        canvas_height = event.height
        # Resize the image to fit the canvas
        self.update_image_size(canvas_width, canvas_height)

    def load_images(self):
        """
        Loads the images based on the selected start and end positions.
        """
        building_name = self.current_building()
        image_path = os.path.join(self.hd_base_path, f"{building_name}.png")
        # Load high-resolution image
        if not os.path.exists(image_path):
            self.show_error(f"Image '{building_name}.png' not found in 'hdFloorPlans' folder.")
            self.on_close()
            return
        try:
            self.original_image = Image.open(image_path)
        except Exception as e:
            self.show_error(f"Failed to load '{building_name}.png': {e}")
            self.on_close()
            return

    def display_image(self):
        """
        Displays the image on the canvas.
        """
        self.canvas.delete("all")  # Clear previous image and markers
        # Display the image on the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.update_image_size(canvas_width, canvas_height)
        self.marker = None
        self.selected_position = None
        self.confirm_button.config(state=tk.DISABLED)
        # Update text label based on step
        if self.current_step == 1:
            self.text_label.config(text="Select start position")
        elif self.current_step == 2:
            self.text_label.config(text="Select end position")

    def update_image_size(self, canvas_width, canvas_height):
        # Ensure canvas dimensions are valid
        if canvas_width <= 0 or canvas_height <= 0:
            # Skip resizing if canvas dimensions are invalid
            return

        # Resize the image to fit the canvas, maintaining aspect ratio
        img_ratio = self.original_image.width / self.original_image.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            # Image is wider relative to canvas
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            # Image is taller relative to canvas
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)

        # Ensure new dimensions are valid
        if new_width <= 0 or new_height <= 0:
            return  # Skip resizing if calculated dimensions are invalid

        # Resize and display the image
        self.displayed_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(self.displayed_image)

        # Center the image on the canvas
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        # Keep a reference to the image to prevent garbage collection
        self.canvas.image = self.photo_image

    def on_click(self, event):
        """
        Handles click events on the image canvas.
        Places or moves a marker and stores the relative position.
        """
        x = event.x
        y = event.y
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Get the position of the image on the canvas
        img_x0 = (canvas_width - self.displayed_image.width) // 2
        img_y0 = (canvas_height - self.displayed_image.height) // 2

        # Check if the click is within the image
        if img_x0 <= x <= img_x0 + self.displayed_image.width and img_y0 <= y <= img_y0 + self.displayed_image.height:
            # Adjust x and y to be relative to the image
            img_x = x - img_x0
            img_y = y - img_y0

            # Calculate relative coordinates
            rel_x = img_x / self.displayed_image.width
            rel_y = img_y / self.displayed_image.height

            # Store the relative coordinates
            self.selected_position = (rel_x, rel_y)
            self.confirm_button.config(state=tk.NORMAL)

            # Remove existing marker
            if self.marker:
                self.canvas.delete(self.marker)

            # Draw new marker (a red circle)
            r = 5  # radius
            self.marker = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="")
        else:
            # Click is outside the image
            self.show_error("Please click inside the image.")

    def on_map_click(self, coords):
        """
        Callback for map clicks.
        Stores the latitude and longitude of the clicked point.
        """
        lat, lon = coords
        self.selected_position = (lat, lon)
        self.confirm_button.config(state=tk.NORMAL)

        # Remove existing markers
        self.map_widget.delete_all_marker()
        self.map_widget.set_marker(lat, lon, text="Selected Position")

    def on_confirm(self):
        """
        Handles the confirmation of the selected position.
        Stores the position and moves to the next step or completes the process.
        """
        if not self.selected_position:
            self.show_error("Please select a position before confirming.")
            return

        building_name = self.current_building()

        if building_name == "Outdoor":
            # For Outdoor, selected_position contains (latitude, longitude)
            lat, lon = self.selected_position
            if self.current_step == 1:
                print(f"Start position on '{building_name}': Latitude={lat}, Longitude={lon}")
            elif self.current_step == 2:
                print(f"End position on '{building_name}': Latitude={lat}, Longitude={lon}")

            # Store the position as (lon, lat) for consistency with backend
            self.click_positions.append([building_name, (lon, lat)])
        else:
            # For indoor buildings, calculate pixel coordinates in the lower-resolution image
            processed_image_path = os.path.join(self.processed_base_path, f"{building_name}.png")
            if not os.path.exists(processed_image_path):
                self.show_error(f"Processed image '{building_name}.png' not found in 'processedFloorPlans' folder.")
                self.on_close()
                return
            try:
                processed_img = Image.open(processed_image_path)
                processed_width, processed_height = processed_img.size
                processed_img.close()  # Close the image after getting size
            except Exception as e:
                self.show_error(f"Failed to load '{building_name}.png' from 'processedFloorPlans': {e}")
                self.on_close()
                return

            # Calculate pixel coordinates in the lower-resolution image
            pixel_x = int(self.selected_position[0] * processed_width)
            pixel_y = int(self.selected_position[1] * processed_height)

            # Ensure pixel coordinates are within image bounds
            pixel_x = min(max(pixel_x, 0), processed_width - 1)
            pixel_y = min(max(pixel_y, 0), processed_height - 1)

            # Print the pixel coordinates
            if self.current_step == 1:
                print(f"Start position on '{building_name}': Pixel coordinates in processed image: ({pixel_x}, {pixel_y})")
            elif self.current_step == 2:
                print(f"End position on '{building_name}': Pixel coordinates in processed image: ({pixel_x}, {pixel_y})")

            # Store the position
            self.click_positions.append([building_name, (pixel_x, pixel_y)])

        # Move to next step or finish
        if self.current_step == 1:
            self.current_step = 2
            self.confirm_button.config(state=tk.DISABLED)
            self.text_label.config(text="Select end position")
            self.selection_status_label.config(text="")

            # Clear previous markers or widgets
            if hasattr(self, 'map_widget'):
                self.map_widget.pack_forget()
                self.map_widget.destroy()
                del self.map_widget
            if hasattr(self, 'canvas_frame'):
                self.canvas_frame.pack_forget()
                self.canvas_frame.destroy()
                del self.canvas_frame
                self.loaded_images.clear()
                self.photo_images.clear()

            # Initialize UI for next building
            self.init_building_display()
        elif self.current_step == 2:
            # All selections done, attempt to find the shortest path
            try:
                start_building, start_pos = self.click_positions[0]
                end_building, end_pos = self.click_positions[1]

                # Retrieve user selections for transport mode and safety mode
                transport_mode = self.transportation_mode.get()
                safety_mode = self.safe_mode.get()

                # Convert selections to boolean flags expected by the WeightCalculator
                transport_mode_flag = True if transport_mode == "Biking" else False
                safety_mode_flag = True if safety_mode == "On" else False

                # Call the find_shortest_path function with additional parameters
                path_result = find_shortest_path(
                    start_building, *start_pos, end_building, *end_pos,
                    transport_mode=transport_mode_flag,
                    safety_mode=safety_mode_flag
                )

                self.path_result = path_result
                self.current_step = 3
                self.display_paths()
            except Exception as e:
                self.show_error(f"An error occurred while finding the shortest path:\n{e}")
                self.on_close()

    def display_paths(self):
        """
        Displays the paths based on the path result from the backend.
        Replaces the current viewing area.
        """
        path_type = self.path_result["type"]

        # Clear the selection frame
        for widget in self.selection_frame.winfo_children():
            widget.destroy()

        # Create a new frame for displaying paths
        self.path_frame = ttk.Frame(self.selection_frame)
        self.path_frame.pack(fill="both", expand=True)

        # Create top frame for label and button
        top_frame = ttk.Frame(self.selection_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Use grid layout
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        # Status label for displaying messages
        self.display_status_label = ttk.Label(top_frame, text="", foreground="blue")
        self.display_status_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.E)

        # Confirm button (Next or Close)
        self.confirm_button = ttk.Button(top_frame, text="Next", command=self.on_next_step)
        self.confirm_button.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        self.confirm_button.config(state=tk.NORMAL)

        if path_type == "indoor":
            # Display indoor path image
            path_image = self.path_result["image"]
            travel_time = self.path_result["travel_time"]
            self.display_indoor_path(path_image)
            time_str = self.format_travel_time(travel_time)
            self.show_display_info(f"Indoor path displayed.\nEstimated travel time: {time_str}")
            self.confirm_button.config(text="Close")
        elif path_type == "outdoor":
            # Display outdoor path on map
            path_coords = self.path_result["path"]
            travel_time = self.path_result["travel_time"]
            self.display_outdoor_path(path_coords)
            time_str = self.format_travel_time(travel_time)
            self.show_display_info(f"Outdoor path displayed.\nEstimated travel time: {time_str}")
            self.confirm_button.config(text="Close")
        elif path_type == "indoor_to_outdoor":
            # Handle indoor to outdoor path
            self.current_substep = 1  # 1: Indoor, 2: Outdoor, 3: Total Time
            self.indoor_image = self.path_result["indoor_image"]
            self.outdoor_path_coords = self.path_result["outdoor_path"]
            self.travel_time = self.path_result["total_travel_time"]

            # Display indoor path first
            self.display_indoor_path(self.indoor_image)
            time_str = self.format_travel_time(self.path_result["indoor_travel_time"])
            self.show_display_info(f"Indoor path displayed.\nEstimated travel time: {time_str}\nClick 'Next' to proceed to outdoor path.")
            self.confirm_button.config(text="Next")
        elif path_type == "outdoor_to_indoor":
            # Handle outdoor to indoor path
            self.current_substep = 1  # 1: Outdoor, 2: Indoor, 3: Total Time
            self.outdoor_path_coords = self.path_result["outdoor_path"]
            self.indoor_image = self.path_result["indoor_image"]
            self.travel_time = self.path_result["total_travel_time"]

            # Display outdoor path first
            self.display_outdoor_path(self.outdoor_path_coords)
            time_str = self.format_travel_time(self.path_result["outdoor_travel_time"])
            self.show_display_info(f"Outdoor path displayed.\nEstimated travel time: {time_str}\nClick 'Next' to proceed to indoor path.")
            self.confirm_button.config(text="Next")
        elif path_type == "indoor_to_indoor":
            # Handle indoor to indoor path
            self.current_substep = 1  # 1: Indoor start, 2: Outdoor, 3: Indoor end, 4: Total Time
            self.indoor_image_1 = self.path_result["indoor_image_1"]
            self.outdoor_path_coords = self.path_result["outdoor_path"]
            self.indoor_image_2 = self.path_result["indoor_image_2"]
            self.travel_time = self.path_result["total_travel_time"]

            # Display first indoor path
            self.display_indoor_path(self.indoor_image_1)
            time_str = self.format_travel_time(self.path_result["indoor_travel_time_1"])
            self.show_display_info(f"First indoor path displayed.\nEstimated travel time: {time_str}\nClick 'Next' to proceed to outdoor path.")
            self.confirm_button.config(text="Next")
        else:
            self.show_display_info("Unknown path type.")
            self.on_close()

    def display_indoor_path(self, image):
        # Create a canvas with light grey background
        self.canvas = tk.Canvas(self.path_frame, bg=LIGHT_GREY)
        self.canvas.pack(fill="both", expand=True)

        # Store the original image
        self.original_image = image

        # Bind the canvas to configure events
        self.canvas.bind("<Configure>", self.on_canvas_configure_path)

        # Keep a reference to the image to prevent garbage collection
        self.canvas.image = None

    def on_canvas_configure_path(self, event):
        # Update the canvas size
        canvas_width = event.width
        canvas_height = event.height
        # Resize the image to fit the canvas
        self.update_path_image_size(canvas_width, canvas_height)

    def update_path_image_size(self, canvas_width, canvas_height):
        # Ensure canvas dimensions are valid
        if canvas_width <= 0 or canvas_height <= 0:
            return  # Skip resizing if dimensions are invalid

        # Resize the image to fit the canvas, maintaining aspect ratio
        img_ratio = self.original_image.width / self.original_image.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            # Image is wider relative to canvas
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            # Image is taller relative to canvas
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)

        # Ensure new dimensions are valid
        if new_width <= 0 or new_height <= 0:
            return  # Skip resizing if calculated dimensions are invalid

        self.displayed_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(self.displayed_image)

        # Center the image on the canvas
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=self.photo_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        # Keep a reference to the image to prevent garbage collection
        self.canvas.image = self.photo_image

    def display_outdoor_path(self, path_coords):
        # Create a map widget using TkinterMapView
        self.map_widget = TkinterMapView(self.path_frame, corner_radius=0)
        self.map_widget.pack(fill="both", expand=True)

        # Center the map on the path
        if path_coords:
            mid_index = len(path_coords) // 2
            mid_lat, mid_lon = path_coords[mid_index]
            self.map_widget.set_position(mid_lat, mid_lon)
            self.map_widget.set_zoom(15)

            # Calculate bounding box positions
            min_lat = min(coord[0] for coord in path_coords)  # Minimum latitude
            max_lat = max(coord[0] for coord in path_coords)  # Maximum latitude
            min_lon = min(coord[1] for coord in path_coords)  # Minimum longitude
            max_lon = max(coord[1] for coord in path_coords)  # Maximum longitude

            # Adjust the map to fit the path
            self.map_widget.fit_bounding_box((max_lat, min_lon), (min_lat, max_lon))
        else:
            # Default map position
            self.map_widget.set_position(38.5382, -121.7617)
            self.map_widget.set_zoom(15)

        # Create markers for start and end positions
        if path_coords:
            start_lat, start_lon = path_coords[0]
            end_lat, end_lon = path_coords[-1]

            self.map_widget.set_marker(start_lat, start_lon, text="Start Position")
            self.map_widget.set_marker(end_lat, end_lon, text="End Position")

        # Create path
        if path_coords:
            self.map_widget.set_path([(lat, lon) for lat, lon in path_coords])

    def show_display_info(self, message):
        """
        Displays a message in the display status label.
        """
        self.display_status_label.config(text=message, foreground="blue")

    def format_travel_time(self, travel_time_seconds):
        """
        Formats the travel time from seconds to minutes and seconds.
        """
        minutes = int(travel_time_seconds // 60)
        seconds = int(travel_time_seconds % 60)
        if minutes > 0:
            return f"{minutes} min {seconds} sec"
        else:
            return f"{seconds} sec"

    def on_next_step(self):
        """
        Handles the transition to the next substep in the path.
        """
        if hasattr(self, 'current_substep'):
            # Clear the path frame
            for widget in self.path_frame.winfo_children():
                widget.destroy()

            if self.current_substep == 1:
                if self.path_result["type"] == "indoor_to_outdoor":
                    # Now display outdoor path
                    self.display_outdoor_path(self.outdoor_path_coords)
                    time_str = self.format_travel_time(self.path_result["outdoor_travel_time"])
                    self.show_display_info(f"Outdoor path displayed.\nEstimated travel time: {time_str}\nClick 'Next' to see total travel time.")
                    self.confirm_button.config(text="Next")
                    self.current_substep += 1
                elif self.path_result["type"] == "outdoor_to_indoor":
                    # Now display indoor path
                    self.display_indoor_path(self.indoor_image)
                    time_str = self.format_travel_time(self.path_result["indoor_travel_time"])
                    self.show_display_info(f"Indoor path displayed.\nEstimated travel time: {time_str}\nClick 'Next' to see total travel time.")
                    self.confirm_button.config(text="Next")
                    self.current_substep += 1
                elif self.path_result["type"] == "indoor_to_indoor":
                    # Display outdoor path
                    self.display_outdoor_path(self.outdoor_path_coords)
                    time_str = self.format_travel_time(self.path_result["outdoor_travel_time"])
                    self.show_display_info(f"Outdoor path displayed.\nEstimated travel time: {time_str}\nClick 'Next' to proceed to second indoor path.")
                    self.confirm_button.config(text="Next")
                    self.current_substep += 1
                else:
                    self.on_close()
            elif self.current_substep == 2:
                if self.path_result["type"] == "indoor_to_outdoor":
                    # Display total travel time
                    time_str = self.format_travel_time(self.path_result["total_travel_time"])
                    self.show_display_info(f"Total estimated travel time: {time_str}")
                    self.confirm_button.config(text="Close")
                    self.current_substep += 1
                elif self.path_result["type"] == "outdoor_to_indoor":
                    # Display total travel time
                    time_str = self.format_travel_time(self.path_result["total_travel_time"])
                    self.show_display_info(f"Total estimated travel time: {time_str}")
                    self.confirm_button.config(text="Close")
                    self.current_substep += 1
                elif self.path_result["type"] == "indoor_to_indoor":
                    # Display second indoor path
                    self.display_indoor_path(self.indoor_image_2)
                    time_str = self.format_travel_time(self.path_result["indoor_travel_time_2"])
                    self.show_display_info(f"Second indoor path displayed.\nEstimated travel time: {time_str}\nClick 'Next' to see total travel time.")
                    self.confirm_button.config(text="Next")
                    self.current_substep += 1
                else:
                    self.on_close()
            
            else:
                self.on_close()
        else:
            self.on_close()

    def on_close(self):
        """
        Handles the closing of the current frames and returns to the main menu.
        """
        # Destroy frames
        if self.selection_frame:
            self.selection_frame.pack_forget()
            self.selection_frame.destroy()
            self.selection_frame = None
        if self.path_frame:
            self.path_frame.pack_forget()
            self.path_frame.destroy()
            self.path_frame = None
        # Clear variables
        self.click_positions = []
        self.current_step = 1
        self.path_result = None
        self.current_substep = None
        self.loaded_images.clear()
        self.photo_images.clear()
        # Show main_frame
        self.main_frame.pack(fill="both", expand=True)
        self.show_info("")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
