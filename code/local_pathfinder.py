import math
import heapq
from itertools import count

class AStar:
    def __init__(self):
        pass

    def heuristic(self, node, goal_node):
        """
        Efficient heuristic function tailored for small areas like UC Davis.
        Uses Euclidean distance with scaling factors for latitude and longitude.
        """
        if not node.use_global_coords:
            # Local coordinates (assumed to be in meters)
            return math.hypot(node.x - goal_node.x, node.y - goal_node.y)
        else:
            # Global coordinates (longitude and latitude)
            dx = (goal_node.x - node.x) * 87700   # Meters per degree longitude at UC Davis latitude
            dy = (goal_node.y - node.y) * 111320  # Meters per degree latitude
            return math.hypot(dx, dy)

    def find_path(self, start_node, goal_node, adjacency_list):
        """
        Finds the shortest path from start_node to goal_node using the A* algorithm.

        Parameters:
            start_node: The starting node.
            goal_node: The goal node.
            adjacency_list: A dictionary mapping nodes to a list of (neighbor, edge) tuples.

        Returns:
            total_cost (float): The cost of the shortest path.
            path (List): The list of nodes representing the shortest path.
        """
        open_heap = []
        heapq.heappush(open_heap, (0, 0, start_node))
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self.heuristic(start_node, goal_node)}
        open_set = {start_node}
        closed_set = set()
        counter = count()  # Unique sequence count to prevent comparison of nodes

        while open_heap:
            current_f, _, current_node = heapq.heappop(open_heap)

            if current_node == goal_node:
                # Reconstruct the shortest path
                path = [current_node]
                while current_node in came_from:
                    current_node = came_from[current_node]
                    path.append(current_node)
                path.reverse()
                total_cost = g_score[goal_node]
                return total_cost, path

            if current_node in closed_set:
                continue

            closed_set.add(current_node)
            open_set.discard(current_node)

            for neighbor, edge in adjacency_list.get(current_node, []):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current_node] + edge.get_weight()

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # Found a better path to neighbor
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self.heuristic(neighbor, goal_node)
                    f_score[neighbor] = f

                    if neighbor not in open_set:
                        heapq.heappush(open_heap, (f, next(counter), neighbor))
                        open_set.add(neighbor)
                    else:
                        # Since heapq doesn't support decrease-key, push a new entry
                        heapq.heappush(open_heap, (f, next(counter), neighbor))
                        # Old entries for neighbor will be skipped when popped if already in closed_set

        # If we reach here, no path was found
        return None, None
