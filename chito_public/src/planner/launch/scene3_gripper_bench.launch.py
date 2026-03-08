from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('iris_planner_demo')
    param_file = os.path.join(pkg_share, 'config', 'scene3_gripper.yaml')

    return LaunchDescription([
        Node(
            package='iris_planner_demo',
            executable='plan_benchmark_scene3_gripper_waypoints',
            name='plan_benchmark_scene3_gripper_waypoints',
            output='screen',
            parameters=[param_file],
        )
    ])
