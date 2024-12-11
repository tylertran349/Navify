from database import Database
from flowDataRetriever import flowDataRetriever
from get_crime_data import haversine, get_multiplier, calculate_edge_multipliers
import math
from weatherDataRetriever import get_weather_data
import time


class WeightCalculator:
    def __init__(self, database):
        self.flow_factor = 0.01
        self.rain_factor = 1
        self.biking_factor = 1  
        self.crime_factor = 1
        self.database = database
        self.flow_data_collector = flowDataRetriever(database) # Precompute the what edges are affected by flows
        self.crime_multipliers = calculate_edge_multipliers(database.get_edges())
    def reset_weights(self):
        for edge in self.database.get_all_edges():
            edge.weight = edge.baseline_weight
        return
    
    # Transport mode = "bike" or "walk", safety_mode = True
    def calculate_weights(self, transport_mode, safety_mode):
        self.reset_weights()
        self.update_transportation_weights(transport_mode)
        self.update_traffic_weights()
        self.update_safety_weights(safety_mode)
        self.update_weather_weights()
        return

    def update_transportation_weights(self, transport_mode):
        if transport_mode == False:
            return 
        elif transport_mode == True: # We are biking
            for edge in self.database.get_edges(): 
                edge.weight *= self.biking_factor
            return

    def update_traffic_weights(self):
        flow_edges = self.flow_data_collector.get_flow_edges()
        flows = self.flow_data_collector.get_traffic_data()
        for i in range(len(flow_edges)-4):
            for edge in flow_edges[i]:
                edge.weight += self.flow_factor * edge.baseline_weight * flows[i]

    def update_weather_weights(self):
        current_wind_speed_10m, current_wind_direction_10m, current_precipitation = get_weather_data(121.75412639535978, 38.54100201651368)
        for edge in self.database.get_edges():
            node1_x, node1_y = edge.node1.x, edge.node1.y
            node2_x, node2_y = edge.node2.x, edge.node2.y
            travel_direction = math.atan2(node2_y - node1_y, node2_x - node1_x)
            rain_slowdown = 0
            wind_adjustment = 0
            if(current_wind_speed_10m >= 5): # Wind speed is only noticeable when it's >=5 mph
                wind_angle = math.radians(current_wind_direction_10m - travel_direction)
                wind_adjustment = 0.5 * current_wind_speed_10m * math.cos(wind_angle)

            if(current_precipitation < 0.01): # If precipitation < 0.01, travel speed is unaffected
                if(current_precipitation >= 0.01 and current_precipitation <= 0.1): # Light rain (slow down by 10% of baseline speed)
                    rain_slowdown = 0.1
                elif(current_precipitation > 0.1 and current_precipitation <= 0.3): # Moderate rain (slow down by 20% of baseline speed)
                    rain_slowdown = 0.2
                elif(current_precipitation > 0.3 and current_precipitation <= 2): # Heavy rain (slow down by 30%)
                    rain_slowdown = 0.3
                else: # Violent rain (slow down by 50%)
                    rain_slowdown = 0.5 

            edge.weight += self.rain_factor * edge.baseline_weight * wind_adjustment * rain_slowdown
    
    # New method to update edge weights based on crime data
    # Updated method to update edge weights based on crime data
    def update_safety_weights(self, safetyMode):
        if not safetyMode:
            return
        # Iterate through all outdoor edges in the database
        for edge in self.database.get_edges():
            crime_multiplier = self.crime_multipliers.get(edge, None)
            if crime_multiplier is not None:
                edge.weight += self.crime_factor * edge.baseline_weight * crime_multiplier
