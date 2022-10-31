from rclpy.node import Node
from fs_msgs.msg import MetaData, MultiplePoints
from geometry_msgs.msg import Vector3       # noqa: F401
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import numpy as np
from driver.utils import PIDController
from driver.path_follower import PathFollowerABC, StanleyDriver, PurePursuitDriver, SemiQuadraticApproximationDriver
from fs_utils.conversions import quat_to_euler
from fs_utils.parameters import get_parameters
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from driver.run_module import run_module
from driver.utils.meta_helper import points_to_point_collection, dict_to_kvps


class PathFollower(Node):
    """
    Map building based on identified cones
    """

    def __init__(self, path_follower):
        assert isinstance(path_follower, PathFollowerABC)
        super().__init__('path_follower')
        self._path_follower = path_follower

        self.parameters = get_parameters(
            self,
            [
                "max_speed", "steering_p0", "steering_p1",
                "look_ahead_dist", "vehicle_length", "max_steer_angle",
                "P", "I", "D", "K", "Ks"
            ])
        self.get_logger().info(f"params: {self.parameters}")

        # Topic to publish vehicle commands
        self.drive_publisher = self.create_publisher(
            Twist,
            '/car/cmd',
            10
        )
        # Topic to publish meta data
        self.meta_publisher = self.create_publisher(
            MetaData,
            "/driver/meta",
            1
        )
        # Topic where cone positions are sent
        self.create_subscription(
            MultiplePoints,
            "/driver/path",
            self.path_callback,
            1
        )
        # Topic where simulation time is sent,
        # callback also responsible for vehicle command updates
        self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            1
        )
        # Topic where the vehicle odometry (position, rotation) are sent
        self.create_subscription(
            Odometry,
            "/car/odom/fuzzed",
            self.odom_callback,
            1
        )
        # Topic where total distance traveled by the vehicle is sent
        self.create_subscription(
            Float32,
            "/car/distance",
            self.distance_callback,
            1
        )

        self.time = None
        self.last_measured_time = None
        self.time_received = False

        self.vehicle_position = None
        self.vehicle_rotation_matrix = None
        self.vehicle_forward = None
        self.vehicle_right = None

        self.speed = None
        self.distance = None
        self.acceleration = None
        self.max_speed = self.parameters.get("max_speed", 5)
        self.min_speed = 1
        self.speed_pid = PIDController(
            P_value=self.parameters.get("P", 0.5),
            I_value=self.parameters.get("I", 0.05),
            D_value=self.parameters.get("D", 0.01))

        self.path = None

        self.meta_collections = None

    def clock_callback(self, msg):
        """
        Handler for gazebo time update, calculates and publishes vehicle commands every tick
        """
        self.time = msg.clock.sec + msg.clock.nanosec / 1e9
        self.time_received = True

        # Don't run if didn't receive odom
        if self.vehicle_position is None or self.speed is None:
            return

        # Try to stop
        acceleration = 0
        steer = 0

        # Else calculate vehicle commands
        if self.path:
            # Translate path to speed and steer commands
            acceleration, steer = self.path_to_command(self.path)

            self.publish_meta_path(acceleration, steer)

        self.publish_vehicle_command(acceleration, steer)

    def odom_callback(self, msg):
        """
        Handler for the car odometry update
        """
        pose = msg.pose.pose

        rot = quat_to_euler(pose.orientation)

        self.vehicle_position = np.array(
            [pose.position.x, pose.position.y, pose.position.z]
        )

        yaw = -rot["yaw"]

        # Rotation matrix from car own frame (left, up, forward) to world frame (right, forward, up)
        # The change in ordering requires the double translation around the rotation
        self.vehicle_rotation_matrix = np.around(
            np.matmul(
                np.array([
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1]
                ]),
                np.matmul(
                    np.array([
                        [np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]
                    ]),
                    np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]
                    ]).astype(np.float),
                )), decimals=3)

        # Calculate vehicle forward and up
        self.vehicle_forward = np.matmul(
            self.vehicle_rotation_matrix,
            np.array([0, 0, 1]))

        self.vehicle_right = np.matmul(
            self.vehicle_rotation_matrix,
            np.array([1, 0, 0]))

    def path_callback(self, msg):
        """
        Handler for the path update
        """
        # Don't run if didn't receive odom
        if self.vehicle_position is None or self.speed is None:
            return

        # If clock is stopped, ignore new cones
        if not self.time_received:
            return

        self.time_received = False

        def vec3_msg_to_array(a):
            return np.array([a.x, a.y, a.z])

        self.path = list(map(vec3_msg_to_array, msg.points))

    @ staticmethod
    def gaussian_weighting(x, b, c, a=10): return a * np.exp((-1*(x-b)**2) / (2*(c**2)))

    def future_curvature(self, line, weight_centre=8):
        """
        Function to determine the future curvature of the Path, returning a weighted value
        used to scale the target velocity for PID update.
        """

        # Convert centreline to vehicle reference frame
        vf_line = list(zip(np.array([np.dot(p - self.vehicle_position, self.vehicle_forward) for p in line]),
                           np.array([np.dot(p - self.vehicle_position, self.vehicle_right) for p in line])))

        # create a vector postion of position infront of the vehicle that
        # is most important (i.e. X metres infront of the vehicle)
        centre = np.array([0, weight_centre])

        curve_cum_sum = 0

        # store straight and curved portions of the path line to colour code
        # for meta map
        straights, arcs = [], []

        for n in range(len(vf_line)-2):

            dx = np.diff([vf_line[n][0], vf_line[n+1][0], vf_line[n+2][0]])
            dy = np.diff([vf_line[n][1], vf_line[n+1][1], vf_line[n+2][1]])

            # calculate the 2nd derivative (rate of change of gradient) of the path at every point
            g_roc = np.diff(dy/dx)

            # calculate the distance of the point from the 'centre' position
            dist_from_weight_centre = np.linalg.norm(centre - vf_line[n])

            # apply gaussian weighting to the curvature result based on the distance from the 'centre' location
            # Ref: https://en.wikipedia.org/wiki/Gaussian_function
            curve_cum_sum += abs(g_roc) * self.gaussian_weighting(x=dist_from_weight_centre,
                                                                  b=weight_centre, c=weight_centre)

            if abs(g_roc) <= 0.01:
                straights.append(line[n])
            else:
                arcs.append(line[n])

        self.meta_collections = [points_to_point_collection(straights, 'straight_sections', 'blue', 'point')] + \
            [points_to_point_collection(arcs, 'arc_sections', 'red', 'point')]

        return np.clip(curve_cum_sum/50, a_min=0, a_max=1)

    def path_to_command(self, path):
        """
        Convert path to speed and steer commands
        """
        # Generate Speed Command

        future_path_curve = self.future_curvature(path)

        target_speed = self.max_speed - (future_path_curve*(self.max_speed-self.min_speed))

        acceleration = self.speed_pid.update(
            target_speed - self.speed,
            self.time
        )

        # Generate Steer Command

        steer, meta = self._path_follower.get_steer(path,
                                                    (self.vehicle_position, self.vehicle_forward,
                                                     self.vehicle_right, self.speed),
                                                    parameters=self.parameters
                                                    )

        self.meta_collections.append(meta)

        return acceleration, steer

    def publish_vehicle_command(self, speed, twist):
        """
        Publish vehicle command with certain speed and twist
        """
        msg = Twist()
        msg.linear.x = float(speed)
        msg.angular.z = float(twist)
        self.drive_publisher.publish(msg)
        return msg

    def distance_callback(self, msg):
        """
        Handler for vehicle odometry distance topic
        """
        # If time not set, won't be able to calculate anything
        if self.time is None:
            return
        # If time didn't change, won't be able to calculate anything
        if self.time == self.last_measured_time:
            return
        # If this is the first update, nothing can be calculated, just store last measured time
        if self.last_measured_time is None:
            self.last_measured_time = self.time
            return

        # Calculate time since last update
        dt = self.time - self.last_measured_time
        self.last_measured_time = self.time

        # Store previous distance value, update distance value
        last_distance = self.distance
        self.distance = msg.data

        # If this is the first distance measurement, won't be able to calculate speed
        if last_distance is None:
            return

        # Store previous speed, calculate speed from distance change
        last_speed = self.speed
        self.speed = (self.distance - last_distance) / dt

        # If this is the first speed calculated, won't be able to calculate acceleration
        if last_speed is None:
            return

        # Calculate acceleration from change in spee
        self.acceleration = (self.speed - last_speed) / dt

    def publish_meta_path(self, acceleration, steer):
        """
        Publish relevant data collections
        """
        meta_data = MetaData(
            collections=self.meta_collections,
            timestamp=self.time,
            source='PathFollower',
            properties=dict_to_kvps({
                'steer': steer,
                'acceleration': acceleration})
        )

        self.meta_publisher.publish(meta_data)

    def stop(self):
        """
        Send out stop command
        """
        self.stopped = True
        self.publish_vehicle_command(0, 0)
        self.get_logger().info('Stopping')


def run_stanley(args=None):
    path_follower = StanleyDriver()
    run_path_follower(args, path_follower)


def run_pure_pursuit(args=None):
    path_follower = PurePursuitDriver()
    run_path_follower(args, path_follower)


def run_quadratic_approx(args=None):
    path_follower = SemiQuadraticApproximationDriver()
    run_path_follower(args, path_follower)


def run_path_follower(args, path_follower):
    def create_path_follower():
        return PathFollower(path_follower)
    run_module(args, create_path_follower)


if __name__ == '__main__':
    run_pure_pursuit()
