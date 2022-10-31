from setuptools import setup
from glob import glob
import os

package_name = 'driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools', "opencv-python", "tensorflow"],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='james@heavey.net',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "blind_drive = driver.blind_driver_publisher:main",
            "cheating_drive = driver.cheating_driver:main",
            "camera1_cv = driver.camera_perception_base:run_cv_cam1",
            "camera0_cv = driver.camera_perception_base:run_cv_cam0",
            "camera2_cv = driver.camera_perception_base:run_cv_cam2",
            "camera_tf = driver.camera_perception_base:run_tf",
            "map_build = driver.map_builder:run_map_remember",
            "map_build_forget = driver.map_builder:run_map_forget",
            "map_build_var = driver.map_builder:run_map_remember_var",
            "map_build_forget_var = driver.map_builder:run_map_forget_var",
            "path_plan_wcl = driver.path_planner_base:run_wall_centre_line",
            "path_plan_pcl = driver.path_planner_base:run_point_centre_line",
            "lidar = driver.lidar_perception:main",
            "path_follow_stanley = driver.path_follower_base:run_stanley",
            "path_follow_pursuit = driver.path_follower_base:run_pure_pursuit",
            "path_follow_quadratic = driver.path_follower_base:run_quadratic_approx"
        ],
    },
)
