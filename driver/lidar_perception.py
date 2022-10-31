from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
import rclpy
from driver.utils.meta_helper import points_to_point_collection
from fs_msgs.msg import MetaData, MultipleCones, SingleCone
from fs_utils.conversions import quat_to_euler
import numpy as np


class LidarPerception(Node):
    def __init__(self):
        super().__init__("lidar_perception")

        # Topic to recieve Faux Lidar scan data
        self.create_subscription(
            LaserScan, "/lidar/fuzzed", self._lidar_handler, 1
        )
        # Topic where the vehicle odometry (position, rotation) are sent
        self.create_subscription(
            Odometry, "/car/odom/fuzzed", self._odom_handler, 1
        )
        # Topic to publish identified cone coordinates
        self.lidar_cone_publisher = self.create_publisher(
            MultipleCones,
            "/driver/lidar_cones",
            1
        )
        # Topic where Meta Data is sent
        self.meta_publisher = self.create_publisher(
            MetaData,
            "/driver/meta",
            1
        )

        self._position = None

    def _odom_handler(self, msg):
        self._position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            quat_to_euler(msg.pose.pose.orientation)["yaw"]
        )

    def _msg_to_points(self, msg):
        x, y, t = self._position

        for i, d in enumerate(msg.ranges):
            angle = i * msg.angle_increment + msg.angle_min + t
            yield np.array([np.cos(angle) * d + x, np.sin(angle) * d + y, 0])

    def _lidar_handler(self, msg):
        if self._position is None:
            return

        points = self._msg_to_points(msg)

        cone_clusters = self.clustering(points, msg.range_min, msg.range_max)

        cones = [
            [np.array(cluster).mean(axis=0), np.tanh(len(cluster)/5)*2]
            for cluster in cone_clusters]

        if len(cones) <= 0:
            return

        def to_single_cones(a):
            return SingleCone(position=Vector3(x=a[0][0], y=a[0][1], z=a[0][2]), probability=a[1])

        self.lidar_cone_publisher.publish(
            MultipleCones(
                cones=list(map(to_single_cones, cones)),
                type="lidar"
            ))

        meta_data = MetaData(
            collections=[
                points_to_point_collection(np.array(cones)[:, 0], 'lidar_points', 'red', 'square'),
            ],
            # timestamp=self.time,
            source='LidarPerception'
        )

        self.meta_publisher.publish(meta_data)

    def clustering(self, points, range_min, range_max, proximity=2, fuzzy_buffer=1):
        """
        Convert Lidar ray scan coordinates into clusters of points in close proximity
        """
        included_points = [point for point in points if range_min + fuzzy_buffer <
                           np.linalg.norm(self._position - point) < range_max - fuzzy_buffer]

        clusters = []

        while len(included_points):

            locus = included_points.pop()
            cluster = [point for point in included_points if (np.linalg.norm(locus - point) <= proximity)]
            clusters.append(cluster + [locus])

            for point1 in cluster:
                remove = []
                for i, point2 in enumerate(included_points):
                    if all(np.array(point1) == np.array(point2)):
                        remove.append(i)
                for i in remove:
                    included_points.pop(i)

        return clusters


def main(args=None):
    rclpy.init(args=args)

    lidar = LidarPerception()

    rclpy.spin(lidar)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    lidar.destroy_node()
    rclpy.shutdown()
