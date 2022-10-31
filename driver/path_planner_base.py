from rclpy.node import Node
from fs_msgs.msg import AllCones, MetaData, MultiplePoints
from geometry_msgs.msg import Vector3       # noqa: F401
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import numpy as np
from fs_utils.conversions import quat_to_euler
from fs_utils.parameters import get_parameters
from std_msgs.msg import Float32
from driver.path_planner import PathPlannerABC, PointCentreLine, WallCentreLine
from driver.run_module import run_module


class PathPlanner(Node):
    """
    Map building based on identified cones
    """

    def __init__(self, path_planner):
        assert isinstance(path_planner, PathPlannerABC)
        super().__init__('path_planner')
        self._path_planner = path_planner

        self.parameters = get_parameters(
            self,
            [
                "inclusion_x_0", "inclusion_x_1", "inclusion_y_0", "inclusion_y_1"
            ])
        self.get_logger().info(f"params: {self.parameters}")

        # Topic to publish meta data
        self.path_publisher = self.create_publisher(
            MultiplePoints,
            "/driver/path",
            1
        )
        # Topic to publish meta data
        self.meta_publisher = self.create_publisher(
            MetaData,
            "/driver/meta",
            1
        )
        # Topic where cone positions are sent
        self.create_subscription(
            AllCones,
            "/driver/map",
            self.map_callback,
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

        self.blue_coords = None
        self.yellow_coords = None

        self.speed = None
        self.distance = None

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

        # Don't run if no map recieved
        if self.blue_coords is None or self.yellow_coords is None:
            return

        try:
            path = self.path_plan(self.blue_coords, self.yellow_coords)
        except Exception:
            return

        def to_vec3(coord):
            return Vector3(x=coord[0], y=coord[1], z=coord[2])

        self.path_publisher.publish(
            MultiplePoints(points=list(map(to_vec3, path)))
        )

        self.publish_meta()

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

    def map_callback(self, msg):
        """
        Handler for the map update
        """
        # Don't run if didn't receive odom
        if self.vehicle_position is None or self.speed is None:
            return

        # If clock is stopped, ignore new cones
        if not self.time_received:
            return

        self.time_received = False

        def cone_msg_to_array(a):
            return np.array([[a.position.x, a.position.y, a.position.z], a.probability])

        for cone_list in msg.cones:
            if cone_list.type == "blue_cones":
                self.blue_coords = list(map(cone_msg_to_array, cone_list.cones))

            if cone_list.type == "yellow_cones":
                self.yellow_coords = list(map(cone_msg_to_array, cone_list.cones))

    def path_plan(self, blue_coords, yellow_coords):
        """
        Generate a centreline using designated abstract module, fit projected vehicle path
        """
        path, self.meta_collections = self._path_planner.path_generator(
            blue_coords,
            yellow_coords,
            (self.vehicle_position, self.vehicle_forward, self.vehicle_right),
            parameters=self.parameters
        )

        return path

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

    def publish_meta(self):
        """
        Publish relevant data collections
        """
        meta_data = MetaData(
            collections=self.meta_collections,
            timestamp=self.time,
            source='PathPlanner',
        )

        self.meta_publisher.publish(meta_data)


def run_point_centre_line(args=None):
    path_planner = PointCentreLine()
    run_path_planner(args, path_planner)


def run_wall_centre_line(args=None):
    path_planner = WallCentreLine()
    run_path_planner(args, path_planner)


def run_path_planner(args, path_planner):
    def create_path_planner():
        return PathPlanner(path_planner)
    run_module(args, create_path_planner)


if __name__ == '__main__':
    run_wall_centre_line()
