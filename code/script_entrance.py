# filename: entrance_mapping.py

import joblib
from entrance_mapper import map_entrances_to_nodes

def entrance_mapping():
    # Load the pre-database from "preDatabase.pkl"
    pre_pickle_filepath = "preDatabase.pkl"
    try:
        with open(pre_pickle_filepath, "rb") as f:
            database = joblib.load("preDatabase.joblib")
        print(f"Pre-database loaded from '{pre_pickle_filepath}'.")
    except Exception as e:
        print(f"Error loading pre-database from pickle: {e}")
        return

    # Define buildings and their corresponding floor plan image paths
    buildings = [
        {
            "name": "Memorial Union",
            "floor_plan_image_path": "Code/CutImages/Memorial Union.png"
        },
        {
            "name": "Walker Hall",
            "floor_plan_image_path": "Code/CutImages/Walker Hall.png"
        }
    ]

    # Map entrances for each building
    from entrance_mapper import map_entrances_to_nodes  # Ensure it's imported here

    for building in buildings:
        building_name = building["name"]
        floor_plan_image_path = building["floor_plan_image_path"]
        map_entrances_to_nodes(database, building_name, floor_plan_image_path)

    # Save the updated database to "database.pkl"
    final_pickle_filepath = "database.pkl"
    try:
        with open(final_pickle_filepath, "wb") as f:
            joblib.dump(database, f)
        print(f"Final database has been saved to '{final_pickle_filepath}'.")
    except Exception as e:
        print(f"Error saving final database to pickle: {e}")

if __name__ == "__main__":
    entrance_mapping()
