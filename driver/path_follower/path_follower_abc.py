from abc import ABC


class PathFollowerABC(ABC):
    def get_steer(self, path, vehicle_vectors, parameters={}):
        return None, None
