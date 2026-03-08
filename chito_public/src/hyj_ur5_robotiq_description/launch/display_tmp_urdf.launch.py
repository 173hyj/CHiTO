from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    urdf_path = "/tmp/ur5_robotiq.urdf"
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
        ),
    ])
