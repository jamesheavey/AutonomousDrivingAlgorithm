from abc import ABC


class PathPlannerABC(ABC):
    def path_generator(self, blue_coords, yellow_coords, vehicle_vectors, parameters={}):
        return [], []
