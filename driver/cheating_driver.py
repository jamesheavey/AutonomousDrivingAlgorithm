import rclpy
import numpy as np
from rclpy.node import Node
from fs_msgs.msg import GazeboStatus, DriverMetadata, Map
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from fs_utils.conversions import quat_to_euler, dict_to_pose

import atexit


class CheatingDriverPublisher(Node):
    """
    This driver cheats, and grabs the map published by the simulator manager
    Only used to test the simulator.
    Must be running before track is launched!
    """

    def __init__(self):
        super().__init__('cheating_driver_publisher')
        self.drive_publisher = self.create_publisher(
            Twist, '/car/cmd', 10)
        # Who needs sensors, when you can just listen to the map
        self.create_subscription(
            GazeboStatus, "/simulator_manager/gazebo_status", self.gazebo_status_callback, 1)
        self.metadata_publisher = self.create_publisher(
            DriverMetadata,
            "/meta/driver",
            1
        )
        self.create_subscription(
            Odometry,
            "/car/odom",
            self.odom_callback,
            1
        )

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.stopped = False

        self.map = None
        self.position = None
        self.forward = None
        self.right = None

    def timer_callback(self):
        """
        handle driving at every update
        """
        if self.stopped:
            return
        # If simulation is not running, don't do anything
        if self.map is None or self.position is None:
            return
        # If the map has no centerline, don't do anything
        if len(self.map.centerline) <= 0:
            self.publish_vehicle_command(0, 0)
            return
        # If we've reached the end of the track, see if the track loops
        if self.i >= len(self.map.centerline):
            # Check if the first point on the track is in range and forward
            start = self.map.centerline[0]
            start = np.array([start.position.x, start.position.y])
            if np.linalg.norm(start - self.position) < 5 and np.dot(start - self.position, self.forward) > 0:
                self.i = 0  # Start from the beginning
            else:
                self.publish_vehicle_command(0, 0)  # Don't do anything
                return

        # Select the next centerline
        pose = self.map.centerline[self.i]
        target = np.array([pose.position.x, pose.position.y])

        # If too close, target next one
        if np.linalg.norm(target - self.position) < 2:
            self.i += 1

        # Calculate angle between heading and next point
        angle = np.dot(target - self.position, self.right) / \
            np.linalg.norm(target - self.position)

        steer = -angle
        acceleration = 4

        # Drive in the direction of the next point
        self.publish_metadata(acceleration, steer)
        self.publish_vehicle_command(acceleration, steer)

    def odom_callback(self, msg):
        """
        handle odometry messages
        """
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])
        yaw = quat_to_euler(msg.pose.pose.orientation)["yaw"]
        self.forward = np.array([
            np.cos(yaw), np.sin(yaw)
        ])
        self.right = np.array([
            np.sin(yaw), -np.cos(yaw)
        ])

    def gazebo_status_callback(self, msg):
        """
        Handle map updates
        """
        # Map and car is loaded: save map
        if msg.status == msg.STATUS_HEALTHY and msg.car.name != "":
            self.map = msg.map
        # Else: Reset variables
        else:
            self.map = None
            self.position = None
            self.i = 0

    def publish_metadata(self, acceleration, steer):
        """
        Publis some info from the map we already have
        """
        def np_to_pose(v):
            return dict_to_pose({
                "position": {
                    "x": v[0],
                    "y": v[1]
                }})

        m = Map(
            name="meta",
            start=np_to_pose(self.position),
            centerline=self.map.centerline[self.i-2:self.i+2],
            blue_cones=self.map.blue_cones,
            yellow_cones=self.map.yellow_cones,
            orange_cones=self.map.orange_cones
        )

        self.metadata_publisher.publish(DriverMetadata(
            map=m,
            acceleration=float(acceleration),
            steer=float(steer),
            timestamp=0.0
        ))

    def publish_vehicle_command(self, acceleration, steer):
        """
        Publish vehicle command
        """
        msg = Twist()
        msg.linear.x = float(acceleration)
        msg.angular.z = float(steer)
        self.drive_publisher.publish(msg)
        return msg

    def stop(self):
        self.stopped = True
        self.publish_vehicle_command(0, 0)
        self.get_logger().info('Stopping')


def main(args=None):
    rclpy.init(args=args)

    driver_publisher = CheatingDriverPublisher()

    atexit.register(lambda: driver_publisher.stop())

    rclpy.spin(driver_publisher)

    driver_publisher.stop()
    driver_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
