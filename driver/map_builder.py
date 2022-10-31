from rclpy.node import Node
from fs_msgs.msg import SingleCone, MultipleCones, AllCones, MetaData
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import numpy as np
from driver.utils import ConeCollection
from driver.utils.meta_helper import cones_to_point_collection
from fs_utils.conversions import quat_to_euler
from driver.run_module import run_module


class MapBuilder(Node):
    """
    Map building based on identified cones
    """

    def __init__(self, local, publish_variance):
        super().__init__('map_builder')

        # Topic to publish wall positions
        self.map_publisher = self.create_publisher(
            AllCones,
            "/driver/map",
            1
        )
        # Topic to publish meta data
        self.meta_publisher = self.create_publisher(
            MetaData,
            "/driver/meta",
            1
        )
        # Topic where simulation time is sent
        self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            1
        )
        # Topic where camera cone positions are sent
        self.create_subscription(
            AllCones,
            "/camera/cone_positions",
            self.camera_mapping_callback,
            1
        )
        # Topic where lidar cone positions are sent
        self.create_subscription(
            MultipleCones,
            "/driver/lidar_cones",
            self.lidar_mapping_callback,
            1
        )
        # Topic where the vehicle odometry (position, rotation) are sent
        self.create_subscription(
            Odometry,
            "/car/odom/fuzzed",
            self.odom_callback,
            1
        )

        self.time = None
        self.last_measured_time = None
        self.time_received = False

        self.vehicle_position = None
        self.vehicle_rotation_matrix = None
        self.vehicle_forward = None
        self.vehicle_right = None

        self.cone_radius = 2
        self.blue_cones = ConeCollection(self.cone_radius)
        self.yellow_cones = ConeCollection(self.cone_radius)
        self.unknown_cones = ConeCollection(self.cone_radius)

        self.local = local
        self.publish_variance = publish_variance

    def clock_callback(self, msg):
        """
        Handler for gazebo time update
        """
        self.time = msg.clock.sec + msg.clock.nanosec / 1e9
        self.time_received = True

        if self.unknown_cones.cones != []:
            self.sensor_fusion()

        if self.local:
            self.forget_cones()

        self.publish_meta_map()

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

    def camera_mapping_callback(self, msg):
        """
        Handler for the camera perception update
        """
        # Don't run if didn't receive odom
        if self.vehicle_position is None:
            return

        # If clock is stopped, ignore new cones
        if not self.time_received:
            return

        self.time_received = False

        def cones_to_list(a):
            return [np.array([a.position.x, a.position.y, a.position.z]), a.probability]

        # Add cones percieved in most recent frame to current Map
        for cone_list in msg.cones:
            if cone_list.type == "blue_cones":
                self.blue_cones.add_cones(
                    map(cones_to_list, cone_list.cones),
                    self.vehicle_position,
                    np.array([msg.heading.x, msg.heading.y, msg.heading.z]))

            if cone_list.type == "yellow_cones":
                self.yellow_cones.add_cones(
                    map(cones_to_list, cone_list.cones),
                    self.vehicle_position,
                    np.array([msg.heading.x, msg.heading.y, msg.heading.z]))

        def to_single_cones(a):
            return SingleCone(
                position=Vector3(x=a.position[0], y=a.position[1], z=a.position[2]),
                probability=float(a.var)
            )

        self.cross_pruning()

        self.map_publisher.publish(
            AllCones(cones=[
                MultipleCones(
                    cones=list(map(to_single_cones, self.yellow_cones.cones)),
                    type="yellow_cones"),
                MultipleCones(
                    cones=list(map(to_single_cones, self.blue_cones.cones)),
                    type="blue_cones")
            ]))

    def lidar_mapping_callback(self, msg):
        """
        Handler for the lidar perception update
        """
        # Don't run if didn't receive odom
        if self.vehicle_position is None:
            return

        # Dont run if no map built from camera
        if self.blue_cones is None or self.yellow_cones is None:
            return

        # Dont run if no map built from camera
        if self.blue_cones.cones == [] or self.yellow_cones.cones == []:
            return

        # If clock is stopped, ignore new cones
        if not self.time_received:
            return

        self.time_received = False

        if msg.type != "lidar":
            return

        def cones_to_list(a):
            return [np.array([a.position.x, a.position.y, a.position.z]), a.probability]

        # Add cones to map
        self.unknown_cones.add_cones(
            list(map(cones_to_list, msg.cones)),
            self.vehicle_position,
            self.vehicle_forward)

        self.lidar_pruning()

        self.publish_meta_map()

    def sensor_fusion(self):
        """
        For each lidar cone, find the nearest mapped blue and yellow cones.
        If within a cone radius, merge cones
        """
        remove = []

        for cone in self.unknown_cones.cones:

            Ib, Iy = None, None
            min_b, min_y = None, None

            # find the closest blue cone, save the index
            Ib, min_b = min(enumerate(self.blue_cones.cones),
                            key=lambda blue_cone: np.linalg.norm(cone.position - blue_cone[1].position))

            # find the closest yellow cone, save the index
            Iy, min_y = min(enumerate(self.yellow_cones.cones),
                            key=lambda yellow_cone: np.linalg.norm(cone.position - yellow_cone[1].position))

            if Ib is None and Iy is None:
                continue

            # If lidar cone is within existing cone radius, merge the cones
            if np.linalg.norm(cone.position - min_b.position) < self.cone_radius:
                self.blue_cones.external_merge(self.blue_cones.cones[Ib], cone)
                remove.append(cone)

            elif np.linalg.norm(cone.position - min_y.position) < self.cone_radius:
                self.yellow_cones.external_merge(self.yellow_cones.cones[Iy], cone)
                remove.append(cone)

        for i in reversed(remove):
            self.unknown_cones.remove_cone(i)

    def lidar_pruning(self, threshold=2):
        """
        Function to prune map cones if not seen by Lidar
        """
        for cone in self.blue_cones.cones + self.yellow_cones.cones:

            closest = min(self.unknown_cones.cones,
                          key=lambda lidar_cone: np.linalg.norm(cone.position - lidar_cone.position))

            if np.linalg.norm(cone.position - closest.position) > self.cone_radius + threshold:
                cone.var += 10

    def cross_pruning(self):
        """
        Function to merge a yellow and a blue cone if they get too close to one another.
        Resulting variance should be greater than the highest variance of the 2 cones.
        """
        if self.unknown_cones.cones != []:
            for cone in self.unknown_cones.cones:
                cone.var += 10

        for cone in self.blue_cones.cones:

            closest = min(self.yellow_cones.cones,
                          key=lambda yellow_cone: np.linalg.norm(cone.position - yellow_cone.position))

            if np.linalg.norm(cone.position - closest.position) < self.cone_radius:
                self.unknown_cones.add_cones(
                    [[cone.position, cone.var], [closest.position, closest.var]],
                    self.vehicle_position,
                    self.vehicle_forward)
                self.blue_cones.remove_cone(cone)
                self.yellow_cones.remove_cone(closest)

    def publish_meta_map(self):
        """
        Publish relevant data collections
        """

        if self.unknown_cones.cones != []:
            meta_data = MetaData(
                collections=[
                    *cones_to_point_collection(self.blue_cones.cones, 'map_blue_cones',
                                               'blue', 'cone', self.publish_variance),
                    *cones_to_point_collection(self.yellow_cones.cones, 'map_yellow_cones',
                                               'yellow', 'cone', self.publish_variance),
                    *cones_to_point_collection(self.unknown_cones.cones, 'map_unknown_cones',
                                               'grey', 'cone', self.publish_variance)
                ],
                timestamp=self.time,
                source='MapBuilder'
            )

        else:
            meta_data = MetaData(
                collections=[
                    *cones_to_point_collection(self.blue_cones.cones, 'map_blue_cones',
                                               'blue', 'cone', self.publish_variance),
                    *cones_to_point_collection(self.yellow_cones.cones, 'map_yellow_cones',
                                               'yellow', 'cone', self.publish_variance)
                ],
                timestamp=self.time,
                source='MapBuilder'
            )

        self.meta_publisher.publish(meta_data)

    def forget_cones(self, forget_radius=30):
        """
        Remove cones from the map that are outside a set radius from the vehicle
        """
        for cone in self.blue_cones.cones:
            if np.linalg.norm(cone.position - self.vehicle_position) > forget_radius:
                self.blue_cones.remove_cone(cone)

        for cone in self.yellow_cones.cones:
            if np.linalg.norm(cone.position - self.vehicle_position) > forget_radius:
                self.yellow_cones.remove_cone(cone)


def run_map_remember(args=None):
    run_mapper(args, False, False)


def run_map_remember_var(args=None):
    run_mapper(args, False, True)


def run_map_forget(args=None):
    run_mapper(args, True, False)


def run_map_forget_var(args=None):
    run_mapper(args, True, True)


def run_mapper(args, *node_args):
    def create_path_planner():
        return MapBuilder(*node_args)
    run_module(args, create_path_planner)


if __name__ == '__main__':
    run_map_remember()
