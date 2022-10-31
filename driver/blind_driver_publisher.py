import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

import atexit


class BlindDriverPublisher(Node):

    def __init__(self):
        super().__init__('blind_driver_publisher')
        self.publisher_ = self.create_publisher(Twist, '/car/cmd', 10)
        timer_period = 15  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.stopped = False
        self.timer_callback()

    def publish(self, speed, twist):
        msg = Twist()
        msg.linear.x = float(speed)
        msg.angular.z = float(twist)
        self.publisher_.publish(msg)
        return msg

    def timer_callback(self):
        if self.stopped:
            return

        msg = self.publish(3, 0 if self.i % 2 == 0 else 0.5)
        self.get_logger().info(f'Publishing: {msg}')
        self.i += 1

    def stop(self):
        self.stopped = True
        self.publish(0, 0)
        self.get_logger().info('Stopping')


def main(args=None):
    rclpy.init(args=args)

    blind_driver_publisher = BlindDriverPublisher()

    atexit.register(lambda: blind_driver_publisher.stop())

    rclpy.spin(blind_driver_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    blind_driver_publisher.stop()
    blind_driver_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
