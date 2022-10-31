import atexit
import rclpy


def run_module(args, node_factory):
    rclpy.init(args=args)

    module = node_factory()

    atexit.register(lambda: module.stop())

    rclpy.spin(module)

    module.stop()
    module.destroy_module()
    rclpy.shutdown()
