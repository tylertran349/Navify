# app.py

from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
import os
from PIL import Image
from io import BytesIO
import base64
from backend import find_shortest_path  # Ensure this function is properly defined in backend.py
import datetime

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure secret key

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
script_dir = os.path.dirname(os.path.abspath(__file__))
app.config['SESSION_FILE_DIR'] = os.path.join(script_dir, 'flask_sessions')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True  # Adds an extra layer of security
Session(app)

# Base paths for hdFloorPlans and processedFloorPlans folders
hd_base_path = os.path.join(script_dir, "hdFloorPlans")
processed_base_path = os.path.join(script_dir, "processedFloorPlans")

def encode_image(image_path):
    """Encodes an image to Base64."""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""

def get_image_dimensions(image_path):
    """Returns image width and height in pixels."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return 1000, 800  # Default dimensions if error occurs

@app.route('/select_position_image', methods=['GET', 'POST'])
def select_position_image():
    displayed_image_path = 'static/images/floorplan_displayed.png'  # Path to the displayed image
    processed_image_path = 'static/images/floorplan_processed.png'  # Path to the processed image used for pathfinding

    if request.method == 'POST':
        selected_position = request.form.get('selected_position')
        if selected_position:
            try:
                x, y = map(int, selected_position.split(','))
                
                # Get dimensions of both images
                displayed_width, displayed_height = get_image_dimensions(displayed_image_path)
                processed_width, processed_height = get_image_dimensions(processed_image_path)

                # Calculate relative positions
                rel_x = x / displayed_width
                rel_y = y / displayed_height

                # Map to processed image coordinates
                proc_x = int(rel_x * processed_width)
                proc_y = int(rel_y * processed_height)

                # Store the selected position in the session
                session['selected_position'] = {'x': proc_x, 'y': proc_y}

                # Redirect to the next step (e.g., pathfinding)
                return redirect(url_for('next_step'))
            except ValueError:
                error_message = "Invalid position coordinates."
        else:
            error_message = "Please select a position on the image."

        # If there's an error, re-render the template with the error message
        encoded_image = encode_image(displayed_image_path)
        return render_template('select_position_image.html', 
                               building_name="Building A", 
                               prompt_text="Click on the floor plan to select a position.", 
                               error_message=error_message,
                               image_data=encoded_image,
                               image_width=get_image_dimensions(displayed_image_path)[0],
                               image_height=get_image_dimensions(displayed_image_path)[1],
                               current_year=datetime.now().year)
    else:
        # GET request: Render the template without error message
        encoded_image = encode_image(displayed_image_path)
        return render_template('select_position_image.html', 
                               building_name="Building A", 
                               prompt_text="Click on the floor plan to select a position.", 
                               image_data=encoded_image,
                               image_width=get_image_dimensions(displayed_image_path)[0],
                               image_height=get_image_dimensions(displayed_image_path)[1],
                               current_year=datetime.now().year)

@app.route('/next_step')
def next_step():
    selected_position = session.get('selected_position')
    if not selected_position:
        return redirect(url_for('select_position_image'))
    # Here you would implement your pathfinding logic using selected_position['x'] and selected_position['y']
    return f"Selected Position on Processed Image: X={selected_position['x']}, Y={selected_position['y']}"


def get_building_names():
    """
    Scans the hdFloorPlans folder for .png files and extracts building names.
    Ensures that 'Outdoor' is always included.
    """
    if not os.path.exists(hd_base_path):
        raise FileNotFoundError(f"Folder 'hdFloorPlans' not found in {hd_base_path}")

    files = os.listdir(hd_base_path)
    building_names = []
    for file in files:
        if file.lower().endswith('.png'):
            name = os.path.splitext(file)[0]
            building_names.append(name)

    # Ensure 'Outdoor' exists
    if "Outdoor" not in building_names:
        raise FileNotFoundError("'Outdoor.png' is missing in 'hdFloorPlans' folder.")

    return building_names

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        start_building = request.form.get('start_position')
        end_building = request.form.get('end_position')
        transportation_mode = request.form.get('transportation_mode')
        safe_mode = request.form.get('safe_mode')

        # Validate selections
        if not start_building or not end_building:
            error_message = "Please select both start and end positions."
            return render_template('index.html', building_names=get_building_names(), error_message=error_message)

        # Store selections in session
        session['start_building'] = start_building
        session['end_building'] = end_building
        session['transportation_mode'] = transportation_mode
        session['safe_mode'] = safe_mode
        session['click_positions'] = []
        session['current_step'] = 1  # Steps: 1-Select Start, 2-Select End

        return redirect(url_for('select_position'))

    # GET request
    try:
        building_names = get_building_names()
    except FileNotFoundError as e:
        error_message = str(e)
        return render_template('error.html', error_message=error_message)
    return render_template('index.html', building_names=building_names)

@app.route('/select_position', methods=['GET', 'POST'])
def select_position():
    if 'current_step' not in session:
        return redirect(url_for('index'))

    current_step = session['current_step']
    click_positions = session.get('click_positions', [])
    if current_step == 1:
        building_name = session['start_building']
        prompt_text = "Select start position"
    elif current_step == 2:
        building_name = session['end_building']
        prompt_text = "Select end position"
    else:
        return redirect(url_for('index'))

    if request.method == 'POST':
        # Process the selected position
        selected_position = request.form.get('selected_position')
        if not selected_position:
            error_message = "Please select a position before confirming."
            if building_name == "Outdoor":
                return render_template('select_position_map.html', building_name=building_name, prompt_text=prompt_text, error_message=error_message)
            else:
                return render_template('select_position_image.html', building_name=building_name, prompt_text=prompt_text, image_data=session.get('current_image_data'), error_message=error_message)

        # Store the selected position
        if building_name == "Outdoor":
            # For Outdoor, selected_position contains "lat,lon"
            lat, lon = map(float, selected_position.split(','))
            click_positions.append([building_name, (lon, lat)])  # Store as (lon, lat)
        else:
            # For indoor buildings, selected_position contains "rel_x,rel_y"
            rel_x, rel_y = map(float, selected_position.split(','))
            # Calculate pixel coordinates in the processed image
            processed_image_path = os.path.join(processed_base_path, f"{building_name}.png")
            if not os.path.exists(processed_image_path):
                error_message = f"Processed image '{building_name}.png' not found in 'processedFloorPlans' folder."
                return render_template('error.html', error_message=error_message)
            try:
                with Image.open(processed_image_path) as processed_img:
                    processed_width, processed_height = processed_img.size
            except Exception as e:
                error_message = f"Failed to load '{building_name}.png' from 'processedFloorPlans': {e}"
                return render_template('error.html', error_message=error_message)

            # Calculate pixel coordinates in the lower-resolution image
            pixel_x = int(rel_x * processed_width)
            pixel_y = int(rel_y * processed_height)

            # Ensure pixel coordinates are within image bounds
            pixel_x = min(max(pixel_x, 0), processed_width - 1)
            pixel_y = min(max(pixel_y, 0), processed_height - 1)

            click_positions.append([building_name, (pixel_x, pixel_y)])

        # Update session
        session['click_positions'] = click_positions

        if current_step == 1:
            session['current_step'] = 2
            return redirect(url_for('select_position'))
        elif current_step == 2:
            # All selections done, attempt to find the shortest path
            try:
                start_building, start_pos = click_positions[0]
                end_building, end_pos = click_positions[1]

                # Retrieve user selections for transport mode and safety mode
                transport_mode = session.get('transportation_mode', 'Walking')
                safety_mode = session.get('safe_mode', 'On')

                # Convert selections to boolean flags expected by the WeightCalculator
                transport_mode_flag = True if transport_mode == "Biking" else False
                safety_mode_flag = True if safety_mode == "On" else False

                # Call the find_shortest_path function with additional parameters
                path_result = find_shortest_path(
                    start_building, *start_pos, end_building, *end_pos,
                    transport_mode=transport_mode_flag,
                    safety_mode=safety_mode_flag
                )

                # Process path_result to make it serializable
                # Convert Image objects to base64 strings
                processed_path_result = {}
                for key, value in path_result.items():
                    if isinstance(value, Image.Image):
                        buffered = BytesIO()
                        value.save(buffered, format="PNG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        processed_path_result[f"{key}_base64"] = image_base64
                    elif isinstance(value, list):
                        # If the value is a list of coordinates, ensure it's serializable
                        processed_path_result[key] = value
                    else:
                        processed_path_result[key] = value

                # Store only serializable data in session
                session['path_result'] = processed_path_result
                session['current_substep'] = 1  # For multi-step paths
                return redirect(url_for('display_path'))

            except Exception as e:
                error_message = f"An error occurred while finding the shortest path:\n{e}"
                return render_template('error.html', error_message=error_message)
    else:
        # GET request
        # Load the image or map
        if building_name == "Outdoor":
            # Render the map selection template
            return render_template('select_position_map.html', building_name=building_name, prompt_text=prompt_text)
        else:
            # Load the image
            image_path = os.path.join(hd_base_path, f"{building_name}.png")
            if not os.path.exists(image_path):
                error_message = f"Image '{building_name}.png' not found in 'hdFloorPlans' folder."
                return render_template('error.html', error_message=error_message)
            # Encode image to base64 for embedding in HTML
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            # Store current image data in session to handle errors
            session['current_image_data'] = image_data
            return render_template('select_position_image.html', building_name=building_name, prompt_text=prompt_text, image_data=image_data)

@app.route('/display_path', methods=['GET', 'POST'])
def display_path():
    if 'path_result' not in session:
        return redirect(url_for('index'))

    path_result = session.get('path_result', {})
    path_type = path_result.get("type")
    current_substep = session.get('current_substep', 1)

    if request.method == 'POST':
        # Handle next steps in multi-step paths
        session['current_substep'] = current_substep + 1
        return redirect(url_for('display_path'))
    else:
        # GET request
        if path_type == "indoor":
            # Display indoor path image
            image_data = path_result.get("image_base64")
            travel_time = path_result.get("travel_time")
            time_str = format_travel_time(travel_time)
            info_message = f"Indoor path displayed. Estimated travel time: {time_str}"
            return render_template('display_path_image.html', image_data=image_data, info_message=info_message)
        elif path_type == "outdoor":
            # Display outdoor path on map
            path_coords = path_result.get("path")
            travel_time = path_result.get("travel_time")
            time_str = format_travel_time(travel_time)
            info_message = f"Outdoor path displayed. Estimated travel time: {time_str}"
            return render_template('display_path_map.html', path_coords=path_coords, info_message=info_message)
        elif path_type == "indoor_to_outdoor":
            # Handle indoor to outdoor path
            if current_substep == 1:
                # Display indoor path image
                indoor_image_data = path_result.get("indoor_image_base64")
                travel_time = path_result.get("indoor_travel_time")
                time_str = format_travel_time(travel_time)
                info_message = f"Indoor path displayed. Estimated travel time: {time_str}"
                return render_template('display_path_image.html', image_data=indoor_image_data, info_message=info_message, next_step=True)
            elif current_substep == 2:
                # Display outdoor path
                path_coords = path_result.get("outdoor_path")
                travel_time = path_result.get("outdoor_travel_time")
                time_str = format_travel_time(travel_time)
                info_message = f"Outdoor path displayed. Estimated travel time: {time_str}"
                return render_template('display_path_map.html', path_coords=path_coords, info_message=info_message, next_step=True)
            else:
                # Display total travel time
                total_travel_time = path_result.get("total_travel_time")
                time_str = format_travel_time(total_travel_time)
                info_message = f"Total estimated travel time: {time_str}"
                return render_template('display_total_time.html', info_message=info_message)
        elif path_type == "outdoor_to_indoor":
            # Handle outdoor to indoor path
            if current_substep == 1:
                # Display outdoor path
                path_coords = path_result.get("outdoor_path")
                travel_time = path_result.get("outdoor_travel_time")
                time_str = format_travel_time(travel_time)
                info_message = f"Outdoor path displayed. Estimated travel time: {time_str}"
                return render_template('display_path_map.html', path_coords=path_coords, info_message=info_message, next_step=True)
            elif current_substep == 2:
                # Display indoor path
                indoor_image_data = path_result.get("indoor_image_base64")
                travel_time = path_result.get("indoor_travel_time")
                time_str = format_travel_time(travel_time)
                info_message = f"Indoor path displayed. Estimated travel time: {time_str}"
                return render_template('display_path_image.html', image_data=indoor_image_data, info_message=info_message, next_step=True)
            else:
                # Display total travel time
                total_travel_time = path_result.get("total_travel_time")
                time_str = format_travel_time(total_travel_time)
                info_message = f"Total estimated travel time: {time_str}"
                return render_template('display_total_time.html', info_message=info_message)
        elif path_type == "indoor_to_indoor":
            # Handle indoor to indoor path
            if current_substep == 1:
                # Display first indoor path
                indoor_image_1_data = path_result.get("indoor_image_1_base64")
                travel_time = path_result.get("indoor_travel_time_1")
                time_str = format_travel_time(travel_time)
                info_message = f"First indoor path displayed. Estimated travel time: {time_str}"
                return render_template('display_path_image.html', image_data=indoor_image_1_data, info_message=info_message, next_step=True)
            elif current_substep == 2:
                # Display outdoor path
                path_coords = path_result.get("outdoor_path")
                travel_time = path_result.get("outdoor_travel_time")
                time_str = format_travel_time(travel_time)
                info_message = f"Outdoor path displayed. Estimated travel time: {time_str}"
                return render_template('display_path_map.html', path_coords=path_coords, info_message=info_message, next_step=True)
            elif current_substep == 3:
                # Display second indoor path
                indoor_image_2_data = path_result.get("indoor_image_2_base64")
                travel_time = path_result.get("indoor_travel_time_2")
                time_str = format_travel_time(travel_time)
                info_message = f"Second indoor path displayed. Estimated travel time: {time_str}"
                return render_template('display_path_image.html', image_data=indoor_image_2_data, info_message=info_message, next_step=True)
            else:
                # Display total travel time
                total_travel_time = path_result.get("total_travel_time")
                time_str = format_travel_time(total_travel_time)
                info_message = f"Total estimated travel time: {time_str}"
                return render_template('display_total_time.html', info_message=info_message)
        else:
            # Unknown path type
            error_message = "Unknown path type."
            return render_template('error.html', error_message=error_message)

def format_travel_time(travel_time_seconds):
    """
    Formats the travel time from seconds to minutes and seconds.
    """
    minutes = int(travel_time_seconds // 60)
    seconds = int(travel_time_seconds % 60)
    if minutes > 0:
        return f"{minutes} min {seconds} sec"
    else:
        return f"{seconds} sec"

@app.route('/reset')
def reset():
    # Clear session and redirect to index
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
