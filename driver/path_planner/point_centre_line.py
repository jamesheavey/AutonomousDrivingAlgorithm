from .path_planner_abc import PathPlannerABC
import numpy as np


class PointCentreLine(PathPlannerABC):
    def __init__(self):
        super().__init__()

        self.include_radius = 25

    def path_generator(self, blue_coords, yellow_coords, vehicle_vectors, parameters={}):
        """
        Generate a centre line by calculating centre points between pairs of selected mapped cones
        """
        vehicle_position, vehicle_forward, vehicle_right = vehicle_vectors

        if blue_coords is None or yellow_coords is None:
            return

        blue_coords, _ = zip(*blue_coords)
        yellow_coords, _ = zip(*yellow_coords)

        included_b = list(self.select_included_coords(np.array(blue_coords),
                                                      vehicle_position,
                                                      vehicle_forward,
                                                      vehicle_right))

        included_y = list(self.select_included_coords(np.array(yellow_coords),
                                                      vehicle_position,
                                                      vehicle_forward,
                                                      vehicle_right))

        # For each selected point find a corresponding pair
        pairs = list(self.find_nearest(included_b, included_y)) + \
            list(self.find_nearest(included_y, included_b))

        centre_points = []
        # For each pair of points calculate the average
        for a, b in pairs:
            avg = (a + b) / 2
            # If the average is too close to a known cone, ignore it
            if any(np.linalg.norm(p - avg) < 1 for p in centre_points):
                continue
            centre_points.append(avg)

        return centre_points, []

    def find_nearest(self, this, other, limit=25/2):
        """
        for each cone find a cone from the other list that is closest to it and within the limited range
        """
        for cone in this:
            # Find nearest cone
            min_o = None
            for other_cone in other:
                # If no nearest cone, next cone is the nearest
                # Otherwise check next if cone is closer than the current nearest
                if min_o is None or\
                        np.linalg.norm(cone - min_o) > np.linalg.norm(cone - other_cone):
                    min_o = other_cone
            # Return pairs that are within the limit
            if min_o is not None and np.linalg.norm(cone - min_o) < limit:
                yield (cone, min_o)

    def select_included_coords(self, coords, vehicle_position, vehicle_forward, vehicle_right):
        """
        Select the relevant coords based on distance and angle
        """
        def distance(A, B):
            """
            Calculate distance between point A(x,y,z) and point B(x,y,z)
            """
            return np.linalg.norm(A - B)

        def angle(A, B, f):
            """
            Calculate angle between a direction vector (f) and the vector pointing from (p) to the cone
            """
            dist = distance(A, B)
            rel = A - B
            return np.dot(f, rel) / dist

        for coord in coords:
            dist = distance(coord, vehicle_position)
            ang = angle(coord, vehicle_position, vehicle_forward)
            # Select cones either close to the vehicle, or in front of it, but further away
            if dist < self.include_radius / 3 or (ang > 0.7 and ang * dist < self.include_radius):
                yield coord
