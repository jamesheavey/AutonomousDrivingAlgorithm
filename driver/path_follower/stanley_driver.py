from .path_follower_abc import PathFollowerABC
import numpy as np
from driver.utils.meta_helper import points_to_point_collection

VEHICLE_LENGTH = 1.753
K_VALUE = 2
KS_VALUE = 0.1
THETA_MAX = 0.6458


class StanleyDriver(PathFollowerABC):
    def __init__(self):
        super().__init__()

    def get_steer(self, path, vehicle_vectors, parameters={}):
        """
        Steer command generated using a stanley controller. Heading error (he) and cross track error (cte)
        are minimised independently.
        Ref: http://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
        """

        vehicle_position, vehicle_forward, vehicle_right, speed = vehicle_vectors

        vehicle_length = parameters.get("vehicle_length", VEHICLE_LENGTH)

        # vehicle position measured from centre
        front_axle = (vehicle_position + (vehicle_forward * (vehicle_length/2)))

        def closest_node(node):
            nodes = np.asarray(path[:-1])
            dist_2 = np.sum((nodes - node)**2, axis=1)
            return path[np.argmin(dist_2)], path[np.argmin(dist_2)+1]

        path_point1, path_point2 = closest_node(front_axle)

        track_forward = (path_point1 - path_point2) / np.linalg.norm(path_point1 - path_point2)

        track_yaw = np.arctan(track_forward[1] / track_forward[0])
        vehicle_yaw = np.arctan(vehicle_forward[1] / vehicle_forward[0])

        heading_error = track_yaw + vehicle_yaw \
            if np.sign(track_yaw) != np.sign(vehicle_yaw) else track_yaw - vehicle_yaw

        cross_track_error = np.dot(path_point1-vehicle_position, vehicle_right)

        steer = heading_error + np.arctan(
            (parameters.get("K", K_VALUE) * cross_track_error) /
            (speed + parameters.get("Ks", KS_VALUE))
        )

        steer = np.clip(
            steer,
            a_min=-parameters.get("max_steer_angle", THETA_MAX),
            a_max=parameters.get("max_steer_angle", THETA_MAX)
        )

        meta_collection = points_to_point_collection([path_point1], 'closest', 'black', 'cone')

        return steer, meta_collection
