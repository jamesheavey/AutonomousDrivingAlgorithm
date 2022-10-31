from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="driver",
            executable="camera1_cv"
        ),
        Node(
            package="driver",
            executable="map_build"
        ),
        Node(
            package="driver",
            executable="path_plan_wcl"
        ),
        Node(
            package="driver",
            executable="path_follow_pursuit"
        )
    ])
