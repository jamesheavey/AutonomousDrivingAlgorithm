from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from fs_msgs.msg import SingleCone, MultipleCones, AllCones, MetaData
from geometry_msgs.msg import Vector3
import numpy as np
import cv2      # noqa: F401
from cv_bridge import CvBridge, CvBridgeError
from rosgraph_msgs.msg import Clock
from fs_utils.conversions import quat_to_euler
from driver.camera_perception import CameraPerceptionABC, CVCameraPerception  # , TFCameraPerception
from driver.run_module import run_module
from driver.utils.meta_helper import percieved_cones_to_point_collection
from driver.utils import PercievedCone
from fs_utils.parameters import get_parameters


class CameraPerception(Node):
    """
    Cone detection and location
    """

    def __init__(self, camera_perception, raw_cam_topic, cam_info_topic,
                 labelled_image_topic, camera_position, camera_pitch, camera_yaw):
        assert isinstance(camera_perception, CameraPerceptionABC)
        super().__init__('camera_perception')
        self._camera_perception = camera_perception

        self.parameters = get_parameters(
            self,
            [
                "smoothing_factor", "cone_probability_scale"
            ])
        self.get_logger().info(f"params: {self.parameters}")

        # Topic to re-publish image with labels applied
        self.labelled_image_publisher = self.create_publisher(
            Image,
            labelled_image_topic,
            1
        )
        # Topic to publish seen cone positions (blue, yellow)
        self.cone_position_publisher = self.create_publisher(
            AllCones,
            "/camera/cone_positions",
            1
        )
        # Topic to publish meta data
        self.meta_publisher = self.create_publisher(
            MetaData,
            "/driver/meta",
            1
        )
        qos_profile_sensor_data.history = QoSHistoryPolicy.KEEP_LAST
        # Topic where the raw camera image is sent
        self.create_subscription(
            Image,
            raw_cam_topic,
            self.camera_callback,
            qos_profile_sensor_data
        )
        # Topic where the camera characteristics are sent
        self.create_subscription(
            CameraInfo,
            cam_info_topic,
            self.info_callback,
            qos_profile_sensor_data
        )
        # Topic where the vehicle odometry (position, rotation) are sent
        self.create_subscription(
            Odometry,
            "/car/odom/fuzzed",
            self.odom_callback,
            1
        )
        # Topic where simulation time is sent
        self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            1
        )

        self.camera_position = camera_position
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        # https://en.wikipedia.org/wiki/Rotation_matrix
        self.camera_rotation_matrix = np.around(
            np.matmul(
                np.matmul(
                    # Z: Flip 180 degrees
                    np.array([
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]
                    ]),
                    # Y: in accordance with camera configuration
                    np.array([
                        [np.cos(self.camera_yaw), 0, np.sin(self.camera_yaw)],
                        [0, 1, 0],
                        [-np.sin(self.camera_yaw), 0, np.cos(self.camera_yaw)]
                    ])
                ),
                # Z: in accordance with camera configuration
                np.array([
                    [1, 0, 0],
                    [0, np.cos(self.camera_pitch), -np.sin(self.camera_pitch)],
                    [0, np.sin(self.camera_pitch), np.cos(self.camera_pitch)]
                ])
            ), decimals=3)

        self.bridge = CvBridge()

        # Properties to be populated by updates
        self.camera_matrix = None
        self.inv_camera_matrix = None
        self.vehicle_position = None
        self.vehicle_rotation_matrix = None
        self.vehicle_forward = None
        self.vehicle_right = None
        self.time = None
        self.last_measured_time = None
        self.time_received = False
        self.processing = False

    def clock_callback(self, msg):
        """
        Handler for gazebo time update
        """
        self.time = msg.clock.sec + msg.clock.nanosec / 1e9
        self.time_received = True

    def info_callback(self, msg):
        """
        Handler for the camera info topic
        msg ref: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        """
        k = msg.k
        # Update camera matrix; Convert from flat array to matrix
        self.camera_matrix = np.array([
            k[0:3],
            k[3:6],
            k[6:9]
        ])
        # Calculate inverse as well
        self.inv_camera_matrix = np.linalg.inv(
            self.camera_matrix
        )

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

        self.camera_forward = np.matmul(
            np.array([
                [np.cos(self.camera_yaw), -np.sin(self.camera_yaw), 0],
                [np.sin(self.camera_yaw), np.cos(self.camera_yaw), 0],
                [0, 0, 1]
            ]),
            self.vehicle_forward)

    def camera_callback(self, msg):
        """
        Handler for camera callback
        """
        # Don't run if didn't receive odom or camera info
        if self.camera_matrix is None or self.vehicle_position is None:
            return

        # If clock is stopped, ignore new images (it happens sometimes)
        if not self.time_received:
            return

        if self.processing:
            return

        self.processing = True

        self.time_received = False

        self.get_logger().info("Image received")

        try:
            # Convert image to a usable format
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        else:
            # Locate cones in image
            img, blue_cones, yellow_cones = self.find_cones(cv2_img, self.vehicle_position)

            self.labelled_image_publisher.publish(
                self.bridge.cv2_to_imgmsg(img, "bgr8")
            )

            def to_single_cones(a):
                return SingleCone(
                    position=Vector3(x=a.position[0], y=a.position[1], z=a.position[2]),
                    probability=a.probability)

            self.cone_position_publisher.publish(
                AllCones(
                    cones=[
                        MultipleCones(
                            cones=list(map(to_single_cones, yellow_cones)),
                            type="yellow_cones"),
                        MultipleCones(
                            cones=list(map(to_single_cones, blue_cones)),
                            type="blue_cones")],
                    heading=Vector3(x=self.camera_forward[0],
                                    y=self.camera_forward[1],
                                    z=self.camera_forward[2])
                ))

            self.publish_meta_cones(blue_cones, yellow_cones)

            self.processing = False

    def find_cones(self, frame, position):
        """
        Find cones within certain colour range and return them in the world frame
        """
        frame, blue, yellow = self._camera_perception.get_cones(frame)

        blue_world = []
        yellow_world = []

        for x, y, area in blue:
            p = self.camera_point_to_vehicle_point(x, y)
            blue_world.append((
                self.vehicle_point_to_world_point(p, position),
                area / np.linalg.norm(p)
            ))

        for x, y, area in yellow:
            p = self.camera_point_to_vehicle_point(x, y)
            yellow_world.append((
                self.vehicle_point_to_world_point(p, position),
                area / np.linalg.norm(p)
            ))

        def to_percieved_cone(a):
            return PercievedCone(position=a[0], probability=a[1])

        b_cones = list(map(to_percieved_cone, blue_world))
        y_cones = list(map(to_percieved_cone, yellow_world))

        return frame, b_cones, y_cones

    def camera_point_to_vehicle_point(self, u, v):
        """
        Use the camera matrix to transform pixel information to world point
        Converts pixels (u, v) into a ray w and finds the intersection with the ground plane
        Assumes that the car is flat on the ground
        Assumes that the point is also on the ground. This is not actually true for cones, but it's close enough
        """

        p = np.array([u, v, 1])
        r = np.matmul(self.inv_camera_matrix, p)    # Calculate projection ray

        # Rotate projection ray with camera rotation matrix
        w = np.matmul(self.camera_rotation_matrix, r)

        w = w / w[1] * -self.camera_position[1] + \
            self.camera_position  # Find intersection with ground plane

        return w

    def vehicle_point_to_world_point(self, p, position):
        """
        Move point from vehicle ref frame to world ref frame
        """
        return np.matmul(self.vehicle_rotation_matrix, p) + position

    def publish_meta_cones(self, blue, yellow):
        """
        Publish relevant data collections
        """
        meta_data = MetaData(
            collections=[
                percieved_cones_to_point_collection(blue, 'percieved_blue_cones', 'green', 'square'),
                percieved_cones_to_point_collection(yellow, 'percieved_yellow_cones', 'purple', 'square')
            ],
            timestamp=self.time,
            source='CameraPerception'
        )

        self.meta_publisher.publish(meta_data)


# def run_tf(args=None):
#     camera_perception = TFCameraPerception()
#     run_percep(args, camera_perception)


def run_cv_cam1(args=None):
    camera_perception = CVCameraPerception()
    run_percep(args, camera_perception,
               "/camera1/image_fuzzed", "/camera1/camera_info", "/camera1/image_labelled",
               np.array([0, 0.592, 0]), -0.4, 0)


def run_cv_cam0(args=None):
    camera_perception = CVCameraPerception()
    run_percep(args, camera_perception,
               "/camera0/image_fuzzed", "/camera0/camera_info", "/camera0/image_labelled",
               np.array([0.1, 0.592, 0]), -0.4, -0.4)


def run_cv_cam2(args=None):
    camera_perception = CVCameraPerception()
    run_percep(args, camera_perception,
               "/camera2/image_fuzzed", "/camera2/camera_info", "/camera2/image_labelled",
               np.array([-0.1, 0.592, 0]), -0.4, 0.4)


def run_percep(args, *node_args):
    def create_percep():
        return CameraPerception(*node_args)
    run_module(args, create_percep)


if __name__ == '__main__':
    run_cv_cam1()
