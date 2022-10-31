from .path_follower_abc import PathFollowerABC
from scipy.optimize import curve_fit
import numpy as np
from driver.utils.meta_helper import points_to_point_collection


class SemiQuadraticApproximationDriver(PathFollowerABC):
    def __init__(self):
        super().__init__()
        self.vehicle_position = None
        self.vehicle_forward = None
        self.vehicle_right = None

    def get_steer(self, path, vehicle_vectors, parameters={}):
        """
        Steer command generated by semi-quadratic approximation of the path-points.
        """

        self.vehicle_position, self.vehicle_forward, self.vehicle_right, _ = vehicle_vectors

        steering_P0 = parameters.get("steering_p0", 80 * 0.4)
        steering_P1 = parameters.get("steering_p1", 1 * 0.3)

        # Transform points to be relative to the car
        xs = np.array([0] + [np.dot(p - self.vehicle_position, self.vehicle_forward)
                             for p in path])
        ys = np.array([0] + [np.dot(p - self.vehicle_position, self.vehicle_right)
                             for p in path])

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        popt = curve_fit(SemiQuadraticApproximationDriver.path_func, xs, ys)[0]

        # Steer is proportional to the distance from the curve and the square of the curvature
        steer = popt[1] * steering_P1 + popt[0] ** 2 * np.sign(popt[0]) * steering_P0

        meta_collection = points_to_point_collection(
            self.path_to_centreline(popt),
            'vehicle_path', 'red', 'line')

        return steer, meta_collection

    @staticmethod
    def path_func(x, c1, c2): return c1 * x ** 2 + c2

    def path_to_centreline(self, path, resolution=0.1, max_distance=10):
        """
        Calculate a set of points on the path, at with a given resolution and maximum distance
        """
        def point_at_d(d):
            y = SemiQuadraticApproximationDriver.path_func(d, *path)
            return self.vehicle_forward * d + self.vehicle_right * y + self.vehicle_position
        d = 0
        start_point = point_at_d(0)
        point = start_point

        while np.linalg.norm(point - start_point) < max_distance:
            yield point
            d += resolution
            point = point_at_d(d)
