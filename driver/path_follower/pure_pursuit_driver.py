import numpy as np
from .path_follower_abc import PathFollowerABC
from driver.utils.meta_helper import points_to_point_collection

LOOK_AHEAD_DIST = 3
VEHICLE_LENGTH = 1.753


class PurePursuitDriver(PathFollowerABC):
    def __init__(self):
        super().__init__()

    def get_steer(self, path, vehicle_vectors, parameters={}):
        """
        Steer command generated using pure pursuit principles. Vehicle corrects
        heading to align with a 'Look-Ahead' point on the track.
        Ref: http://www.enseignement.polytechnique.fr/profs/informatique/Eric.Goubault/MRIS/coulter_r_craig_1992_1.pdf
        """

        look_ahead_dist = parameters.get("look_ahead_dist", LOOK_AHEAD_DIST)
        vehicle_length = parameters.get("vehicle_length", VEHICLE_LENGTH)

        vehicle_position, vehicle_forward, vehicle_right, _ = vehicle_vectors

        # vehicle position measured from centre
        rear_axle = (vehicle_position - (vehicle_forward * vehicle_length/2))

        # Ref: https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
        def closest_node(node):
            nodes = np.asarray(path)
            dist_2 = np.sum((nodes - node)**2, axis=1)
            return path[np.argmin(dist_2)]

        path_point = closest_node(vehicle_position + vehicle_forward * look_ahead_dist)

        ld = np.linalg.norm(rear_axle - path_point)

        sin_alpha = np.dot(
            (path_point - rear_axle) / ld,
            vehicle_right)

        steer = np.arctan((2*vehicle_length*sin_alpha) / ld)

        meta_collection = points_to_point_collection([path_point], 'lookahead', 'black', 'cone')

        return steer, meta_collection
